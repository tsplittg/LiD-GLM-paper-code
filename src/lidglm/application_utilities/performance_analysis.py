import torch
import numpy as np
import pandas as pd
from datetime import date
from ray import train, tune
from ray.tune import Tuner

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from lidglm import ExtendedGLM_Utils
from lidglm.application_utilities.training_functions import train_lidglm, train_nn, train_custom_sddr, HeteroscedasticGaussian, BernoulliSDDR
import os
import json
import torch.nn as nn

def train_set_split(config, data, other_specifications=None):

    """
    Function without sddr to enable function usage without having sddr installed.
    Trains each model with the appropriate training function.
    The specfic keys that each training function expects can be found in the respective docstrings.

    data should contain a key "index" that contains a tuple of the form (train_indices, test_indices).
    """
    used_config = config["model_specifications"]
    
    if other_specifications is None:
        other_specifications = {}

    match used_config["model_type"]:
        case "ldglm": return train_lidglm(used_config, data, other_specifications)
        case "basic_nn": return train_nn(used_config, data, other_specifications)
        case "custom_sddr": return train_custom_sddr(used_config, data, other_specifications)
        case _: raise ValueError("This model does not have an implemented training function.")

def ray_hyperparam_opt(covariates, target, index_train, index_test, parameter_space, metrics, temp_dir, experiment_name = "ray_hyperparam_opt", number_of_trials_per_split=8):

    data = {"covariates":np.array(covariates), "target": np.array(target), "index": (index_train, index_test)}

    other_specifications = {"metrics": metrics}


    tuner = Tuner(
        tune.with_parameters(tune.with_resources(train_set_split,resources = {"cpu":1.} ), data =data, other_specifications=other_specifications),
        run_config=train.RunConfig(
                name=experiment_name,
                log_to_file="output.txt"
                ,storage_path=temp_dir, 
                checkpoint_config= train.CheckpointConfig(num_to_keep = 30, checkpoint_score_attribute= "validation_loss", checkpoint_score_order = "min")
            ),
        param_space = {
            "model_specifications": tune.grid_search(parameter_space)
        },
        tune_config = tune.TuneConfig(metric = "validation_loss", mode="min", 
                                    trial_dirname_creator= lambda trial: "Test"+trial.trial_id
                                    ,max_concurrent_trials=number_of_trials_per_split
                                    )
        )
    result_grid = tuner.fit()

    return result_grid

def k_fold_performance_analysis_lidglm(covariates, target, parameter_space, metrics, temp_dir, mode, k=5, test_name = "k_fold_performance_analysis" + date.today().strftime("%Y-%m-%d"),
                                        number_of_trials_per_split=8, normalize = True):
    """Performs a k-fold cross-validation where for each split a hyperparameter optimization is performed on the "training set" and the best model evaluated on the "test set".
    
    "mode" is one of: "binary", "continuous"
    """
    covariates = np.array(covariates)
    target = np.array(target)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    #result_grids = []
    if mode == "binary":
        metrics_dataframe = pd.DataFrame(columns=["split", "test loglike", "test brier score", "test roc_auc"])
    elif mode == "continuous":
        metrics_dataframe = pd.DataFrame(columns=["split", "test loglike", "test mse"])
    metrics_dataframe.set_index("split", inplace=True)
    fold_number = 0
    for train_index, test_index in kf.split(target):
        print(f"Training on split number {fold_number+1} of {k}")
        fold_number += 1
        actual_train_index, validation_index = train_test_split(train_index, test_size=0.2, random_state=42)
        
        if normalize:
            covariates_in_split = normalize_covariates(covariates, actual_train_index)
            if mode == "continuous":
                targets_in_split = normalize_targets(target, actual_train_index)
            else:
                targets_in_split = target
        else:
            covariates_in_split = covariates
            targets_in_split = target
        result_grid = ray_hyperparam_opt(covariates_in_split, targets_in_split, actual_train_index, validation_index, parameter_space, metrics,temp_dir, experiment_name=f"{test_name}_split_{fold_number}", number_of_trials_per_split=number_of_trials_per_split)

        # get best configuration
        best_result = result_grid.get_best_result(metric="validation_loss", mode="min", scope="all")
        # get best checkpoint (retroactive early stopping):
        best_model_checkpoint = best_result.get_best_checkpoint(metric="validation_loss", mode="min")

        # load the corresponding model:
        best_model = ExtendedGLM_Utils().load_from_ray_checkpoint(best_model_checkpoint).eval()
        # set dispersion:
        best_model.set_dispersion(exog= torch.tensor(covariates_in_split[actual_train_index,], dtype=torch.float32),
                                  endog= torch.tensor(targets_in_split[actual_train_index], dtype=torch.float32).reshape(-1))
        # evaluate the model on the test set:
        
        test_covariates_tensor = torch.tensor(covariates_in_split[test_index,], dtype=torch.float32)
        test_target_tensor = torch.tensor(targets_in_split[test_index], dtype=torch.float32).reshape(-1)

        #compute loglike:
        loglike = best_model.log_like_obs(endog = test_target_tensor, exog = test_covariates_tensor).mean().detach().cpu().numpy()
        if mode == "binary":
            #compute brier score:
            predictions = best_model.predict_endog(test_covariates_tensor).detach().cpu().numpy()
            brier_score = np.mean((predictions - test_target_tensor.numpy())**2)

            # compute ROC AUC:
            roc_auc = roc_auc_score(test_target_tensor.numpy(), predictions)

            # append the results to the dataframe:
            metrics_dataframe.loc[fold_number] = [loglike, brier_score, roc_auc]
            #result_grids.append(result_grid)
        elif mode == "continuous":
            #compute test mse:
            predictions = best_model.predict_endog(test_covariates_tensor).detach().cpu().numpy()
            mse = np.mean((predictions - test_target_tensor.numpy())**2)
            # append the results to the dataframe:
            metrics_dataframe.loc[fold_number] = [loglike, mse]
            #result_grids.append(result_grid)

    return metrics_dataframe

def traditional_glm_hyperparam_opt(covariates, target, parameter_space, metrics, mode, regularization= None, normalize = True):
    X_train, X_val, y_train, y_val = train_test_split(covariates, target, test_size=0.2, random_state=42)
    current_best_val_log_like = -np.inf
    best_model_params = None
    for param_config in parameter_space:
        match param_config["link"]:
            case "logit":
                link = sm.families.links.Logit()
            case "power":
                link = sm.families.links.Power()
            case "inversepower":
                link = sm.families.links.InversePower()
            case "sqrt":
                link = sm.families.links.Sqrt()
            case "inversesquared":
                link = sm.families.links.InverseSquared()
            case "identity":
                link = sm.families.links.Identity()
            case "log":
                link = sm.families.links.Log()
            case "logc":
                link = sm.families.links.LogC()
            case "probit":
                link = sm.families.links.Probit()
            case "cdflink":
                link = sm.families.links.CDFLink()
            case "cauchy":
                link = sm.families.links.Cauchy()
            case "cloglog":
                link = sm.families.links.CLogLog()
            case "loglog":
                link = sm.families.links.LogLog()
            case "negativebinomial":
                link = sm.families.links.NegativeBinomial()
            case _: raise ValueError("Unsupported link_str argument")

        # Translate the given family_str argument to a sm.families.links.Family() instance of the appropriate subclass
        # with the above defined link-function:
        match param_config["family"]:
            case "poisson":
                family = sm.families.family.Poisson(link = link, check_link= True)
            case "gaussian":
                family = sm.families.family.Gaussian(link=link, check_link=True)
            case "gamma":
                family = sm.families.family.Gamma(link=link, check_link=True)
            case "binomial":
                family = sm.families.family.Binomial(link=link, check_link=True)
            case "inversegaussian":
                family = sm.families.family.InverseGaussian(link=link, check_link=True)
            case "negativebinomial":
                family = sm.families.family.NegativeBinomial(link=link, check_link=True)
            case "tweedie":
                family = sm.families.family.Tweedie(link=link, check_link=True)
            case _: raise ValueError("Unsupported family_str argument")

        sm_glm = sm.GLM(endog=y_train, exog=sm.add_constant(X_train, prepend=True, has_constant="add"), family=family,
                        offset=None, exposure=None, freq_weights=None, var_weights=None,
                        missing='none')
        if regularization is None:
            sm_results = sm_glm.fit(maxiter=10**3)
        elif regularization == "elastic_net":
            sm_results = sm_glm.fit_regularized(method="elastic_net", alpha=param_config["alpha"], maxiter=10**3)
        if mode == "binary":
            # compute validation loglike:
            val_loglike = sm_results.model.get_distribution(exog=sm.add_constant(X_val, prepend=True, has_constant="add"), params = sm_results.params, scale = sm_results.model.scale).logpmf(y_val).mean()
        elif mode == "continuous":
            # compute validation loglike
            val_loglike = sm_results.model.get_distribution(exog=sm.add_constant(X_val, prepend=True, has_constant="add"), params = sm_results.params, scale = sm_results.model.scale).logpdf(y_val).mean()

        if val_loglike > current_best_val_log_like:
            current_best_val_log_like = val_loglike
            best_model_params = (family, link, param_config["alpha"] if "alpha" in param_config else None)

    return best_model_params

        

def k_fold_performance_analysis_smglm(covariates, target, parameter_space, metrics, temp_dir, mode, k=5, test_name = "k_fold_performance_analysis" + date.today().strftime("%Y-%m-%d"), number_of_trials_per_split=8,
                                      regularization=None, normalize = True):
    """Performs a k-fold cross-validation where for each split a traditional glm is trained on the "training set" and the best model evaluated on the "test set".
    
    "mode" is one of "binary", "continuous"
    """
    covariates = np.array(covariates)
    target = np.array(target).reshape(-1)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    if mode == "binary":
        metrics_dataframe = pd.DataFrame(columns=["split", "test loglike", "test brier score", "test roc_auc"])
    elif mode == "continuous":
        metrics_dataframe = pd.DataFrame(columns=["split", "test loglike", "test mse"])
    metrics_dataframe.set_index("split", inplace=True)
    fold_number = 0
    for train_index, test_index in kf.split(target):
        actual_train_index, validation_index = train_test_split(train_index, test_size=0.2, random_state=42)
        if normalize:
            covariates_in_split = normalize_covariates(covariates, actual_train_index)
            if mode == "continuous":
                targets_in_split = normalize_targets(target, actual_train_index)
            else:
                targets_in_split = target
        else:
            covariates_in_split = covariates
            targets_in_split = target
        print(f"Training on split number {fold_number+1} of {k}")
        fold_number += 1
        best_model_params = traditional_glm_hyperparam_opt(covariates_in_split[actual_train_index,], targets_in_split[actual_train_index], parameter_space, metrics, mode=mode, regularization= regularization)
        best_sm_glm = sm.GLM(endog=targets_in_split[train_index], exog=sm.add_constant(covariates_in_split[train_index,], prepend=True, has_constant="add"), family=best_model_params[0])
        if regularization is None:
            sm_results = best_sm_glm.fit(maxiter=10**3)
        elif regularization == "elastic_net":
            sm_results = best_sm_glm.fit_regularized(method="elastic_net", alpha=best_model_params[2], maxiter=10**3)
        test_covariates = sm.add_constant(covariates_in_split[test_index,], prepend = True, has_constant="add")
        test_target = targets_in_split[test_index]
        model_estimated_scale = sm_results.model.estimate_scale(mu=sm_results.predict(sm.add_constant(covariates_in_split[train_index,], prepend=True)))
        if mode == "binary":
                # compute loglike:
                loglike = sm_results.model.get_distribution(exog =test_covariates, params = sm_results.params, scale = model_estimated_scale).logpmf(test_target).mean()
                #compute brier score:
                predictions = sm_results.predict(test_covariates)
                brier_score = np.mean((predictions - test_target)**2)

                # compute ROC AUC:
                roc_auc = roc_auc_score(test_target, predictions)

                # append the results to the dataframe:
                metrics_dataframe.loc[fold_number] = [loglike, brier_score, roc_auc]

        elif mode == "continuous":
            # compute loglike:
            loglike = sm_results.model.get_distribution(exog =test_covariates, params = sm_results.params, scale = model_estimated_scale).logpdf(test_target).mean()
            #compute test mse:
            predictions = sm_results.predict(test_covariates)
            mse = np.mean((predictions - test_target)**2)
            # append the results to the dataframe:
            metrics_dataframe.loc[fold_number] = [loglike, mse]

    
    return metrics_dataframe


def k_fold_performance_analysis_custom_sddr(covariates, target, parameter_space, metrics, temp_dir, mode, k=5, test_name = "k_fold_performance_analysis" + date.today().strftime("%Y-%m-%d"), number_of_trials_per_split=8,
                                      regularization=None, normalize = True):
    """Performs a k-fold cross-validation where for each split an sddr model is trained on the "training set" and the best model evaluated on the "test set".
    
    "mode" is one of "binary", "continuous"
    """
    covariates = np.array(covariates)
    target = np.array(target).reshape(-1)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    #result_grids = []
    if mode == "binary":
        metrics_dataframe = pd.DataFrame(columns=["split", "test loglike", "test brier score", "test roc_auc"])
    elif mode == "continuous":
        metrics_dataframe = pd.DataFrame(columns=["split", "test loglike", "test mse"])
    metrics_dataframe.set_index("split", inplace=True)

    for fold_number, (train_index, test_index) in enumerate(kf.split(target)):
        print(f"Training on split number {fold_number+1} of {k}")
        fold_number += 1
        actual_train_index, validation_index = train_test_split(train_index, test_size=0.2, random_state=42)
        if normalize:
            covariates_in_split = normalize_covariates(covariates, actual_train_index)
            if mode == "continuous":
                targets_in_split = normalize_targets(target, actual_train_index)
            else:
                targets_in_split = target
        else:
            covariates_in_split = covariates
            targets_in_split = target

        result_grid = ray_hyperparam_opt(covariates_in_split, targets_in_split, actual_train_index, validation_index, parameter_space, metrics,temp_dir, experiment_name=f"{test_name}_split_{fold_number}", number_of_trials_per_split=number_of_trials_per_split)
        print(result_grid)
        # get best configuration
        best_result = result_grid.get_best_result(metric="validation_loss", mode="min", scope="all")
        # get best checkpoint (retroactive early stopping):
        best_model_checkpoint = best_result.get_best_checkpoint(metric="validation_loss", mode="min")
        config = json.load(open(os.path.join(best_result.path, "params.json")))["model_specifications"]
        num_params = len(covariates_in_split[0,])
        distribution = config["family"] if "family" in config else "gaussian"
        if distribution == "gaussian":
            sddr_model = HeteroscedasticGaussian(in_dim= num_params, hidden_dims_mean= [config["width"]]*config["depth"],
                                            hidden_dims_scale= [config["width_2"]]*config["depth_2"],
                                            dropout= config["dropout_rate"])
                                            
        elif distribution == "bernoulli":
            sddr_model = BernoulliSDDR(in_dim= num_params, hidden_dims= [config["width"]]*config["depth"],
                                            dropout= config["dropout_rate"])
        else:
            raise ValueError("Unsupported distribution for custom SDDR model.")
        
        


        
        with best_model_checkpoint.as_directory() as checkpoint_dir:
            checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))  # This will load the checkpoint and set the working directory
            sddr_model.load_state_dict(checkpoint["model_state"])
            sddr_model.eval()  # Set the model to evaluation mode
        best_model = sddr_model

        # evaluate the model on the test set:
        
        test_covariates_tensor = torch.tensor(covariates_in_split[test_index,], dtype=torch.float32)
        test_target_tensor = torch.tensor(targets_in_split[test_index], dtype=torch.float32).reshape(-1)

        
        if mode == "binary":
            
            #compute brier score:
            predictions = best_model(test_covariates_tensor).detach().cpu().numpy()
            eps = np.finfo(np.float64).eps
            predictions = predictions.astype(np.float64)  # otherwise, np.clip doesn't work as expected
            predictions = np.clip(predictions, a_min = eps,a_max=  1-eps)
            brier_score = np.mean((predictions - targets_in_split[test_index])**2)
            # compute loglike:
            test_log_like = best_model.log_like_obs(test_target_tensor.numpy(), predictions).mean()
            test_log_like = test_log_like.mean().item()


            # compute ROC AUC:
            roc_auc = roc_auc_score(test_target_tensor.numpy(), predictions)

            # append the results to the dataframe:
            metrics_dataframe.loc[fold_number] = [test_log_like, brier_score, roc_auc]
        elif mode == "continuous":
            #compute test mse:
            pred_mu, pred_sigma = best_model(test_covariates_tensor)
            predictions = pred_mu.detach().cpu().numpy().reshape(-1)
            mse = np.mean((predictions - test_target_tensor.numpy())**2)
            test_log_like = best_model.log_like_obs(test_target_tensor.numpy(), pred_mu.detach().cpu().numpy(), pred_sigma.detach().cpu().numpy()).mean()
            # append the results to the dataframe:
            metrics_dataframe.loc[fold_number] = [test_log_like, mse]
    return metrics_dataframe
        
def k_fold_performance_analysis_nn(covariates, target, parameter_space, metrics, temp_dir, mode, k=5, 
                                   test_name = "k_fold_performance_analysis" + date.today().strftime("%Y-%m-%d"), number_of_trials_per_split=8,
                                   normalize = True):
    """Performs a k-fold cross-validation where for each split a hyperparameter optimization is performed on the "training set" and the best model evaluated on the "test set".
    
    "mode" is one of: "binary", "continuous"
    """
    covariates = np.array(covariates)
    target = np.array(target)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    #result_grids = []
    if mode == "binary":
        metrics_dataframe = pd.DataFrame(columns=["split", "test loglike", "test brier score", "test roc_auc"])
    elif mode == "continuous":
        metrics_dataframe = pd.DataFrame(columns=["split", "test loglike", "test mse"])
    metrics_dataframe.set_index("split", inplace=True)
    fold_number = 0
    for train_index, test_index in kf.split(target):
        print(f"Training on split number {fold_number+1} of {k}")
        fold_number += 1
        actual_train_index, validation_index = train_test_split(train_index, test_size=0.2, random_state=42)
        if normalize:
            covariates_in_split = normalize_covariates(covariates, actual_train_index)
            if mode == "continuous":
                targets_in_split = normalize_targets(target, actual_train_index)
            else:
                targets_in_split = target
        else:
            covariates_in_split = covariates
            targets_in_split = target
        result_grid = ray_hyperparam_opt(covariates_in_split, targets_in_split, actual_train_index, validation_index, parameter_space, metrics,temp_dir, experiment_name=f"{test_name}_split_{fold_number}", number_of_trials_per_split=number_of_trials_per_split)
        print(result_grid)
        # get best configuration
        best_result = result_grid.get_best_result(metric="validation_loss", mode="min", scope="all")
        # get best checkpoint (retroactive early stopping):
        best_model_checkpoint = best_result.get_best_checkpoint(metric="validation_loss", mode="min")
        config = json.load(open(os.path.join(best_result.path, "params.json")))["model_specifications"]
        num_params = len(covariates_in_split[0,])
        layers = [nn.Linear(num_params, config["width"]), nn.ReLU()]
        for i in range(config["depth"]):
            layers.append(nn.Linear(config["width"], config["width"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config["width"], 1))

        layers.append(nn.Sigmoid())
        basic_nn = nn.Sequential(*layers)


        
        with best_model_checkpoint.as_directory() as checkpoint_dir:
            checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))  # This will load the checkpoint and set the working directory
            basic_nn.load_state_dict(checkpoint["model_state"])
            basic_nn.eval()  # Set the model to evaluation mode
        best_model = basic_nn

        # evaluate the model on the test set:
        
        test_covariates_tensor = torch.tensor(covariates_in_split[test_index,], dtype=torch.float32)
        test_target_tensor = torch.tensor(targets_in_split[test_index], dtype=torch.float32).reshape(-1)

        
        if mode == "binary":
            
            #compute brier score:
            predictions = best_model(test_covariates_tensor).detach().cpu().numpy()
            print("predictions max and min:", predictions.max(), predictions.min())
            eps = np.finfo(np.float64).eps
            predictions = predictions.astype(np.float64)  # otherwise, np.clip doesn't work as expected
            predictions = np.clip(predictions, a_min = eps,a_max=  1-eps)
            brier_score = np.mean((predictions - targets_in_split[test_index])**2)
            # compute loglike:
            test_log_like = np.log(predictions).reshape(-1)*targets_in_split.reshape(-1)[test_index] +np.log(1- predictions).reshape(-1)*(1-targets_in_split.reshape(-1)[test_index])
            test_log_like = test_log_like.mean().item()


            # compute ROC AUC:
            roc_auc = roc_auc_score(test_target_tensor.numpy(), predictions)

            # append the results to the dataframe:
            metrics_dataframe.loc[fold_number] = [test_log_like, brier_score, roc_auc]
        elif mode == "continuous":
            raise NotImplementedError("The neural network should only be used in the binary case.")
            #compute test mse:
            #predictions = best_model(test_covariates_tensor).detach().cpu().numpy()
            #mse = np.mean((predictions - test_target_tensor.numpy())**2)
            # append the results to the dataframe:
            #metrics_dataframe.loc[fold_number] = [mse]
            #result_grids.append(result_grid)

    return metrics_dataframe

def normalize_covariates(covariates, train_index, mode = "non_binary"):
    
    match mode:
        case "non_binary":
            # we only normalize non-binary covariates
            non_binary_indices =  np.array([i for i in range(covariates.shape[1]) if len(np.unique(covariates[:, i])) > 2])
            mean_on_train = np.zeros(covariates.shape[1])
            std_on_train = np.ones(covariates.shape[1])

            mean_on_train[non_binary_indices] = covariates[train_index, :][:, non_binary_indices].mean(axis=0)
            std_on_train[non_binary_indices] = covariates[train_index, :][:, non_binary_indices].std(axis=0)
        case "all":
            mean_on_train = covariates[train_index, :].mean(axis=0)
            std_on_train = covariates[train_index, :].std(axis=0)

    covariates_in_split = (covariates - mean_on_train)/ std_on_train

    return covariates_in_split

def normalize_targets(targets, train_index):

    mean_on_train = targets[train_index].mean(axis=0)
    std_on_train = targets[train_index].std(axis=0)
    targets = (targets - mean_on_train)/std_on_train
    return targets
