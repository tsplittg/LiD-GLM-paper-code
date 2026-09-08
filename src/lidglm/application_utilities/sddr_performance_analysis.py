# This script includes the functions necessary for performing the nested cross validation with the official SDDR python implementation

import torch
from torch import optim
import numpy as np
import pandas as pd
from datetime import date
from ray import train, tune
from ray.tune import Tuner
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from lidglm.application_utilities.training_functions import train_lidglm, train_nn
import os
import torch.nn as nn
from lidglm.application_utilities.custom_sddr import CustomSddr
from lidglm.application_utilities.performance_analysis import normalize_targets, normalize_covariates
import copy


def train_set_split_with_sddr(config, data, other_specifications=None):

    """
    Function with sddr to enable function usage with sddr installed.
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
        case "sddr": return train_sddr(used_config, data, other_specifications)
        case _: raise ValueError("This model does not have an implemented training function.")

def ray_hyperparam_opt_with_sddr(covariates, target, index_train, index_test, parameter_space, metrics, temp_dir, experiment_name = "ray_hyperparam_opt", number_of_trials_per_split=8):


    data = {"covariates":np.array(covariates), "target": np.array(target), "index": (index_train, index_test)}

    other_specifications = {"metrics": metrics}


    tuner = Tuner(
        tune.with_parameters(tune.with_resources(train_set_split_with_sddr,resources = {"cpu":1.} ), data =data, other_specifications=other_specifications),
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

def train_sddr(config, data, other_specifications=None):
    """
    Trains a SDDR model on the given data.
    
    """
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    covariates = data["covariates"]
    target = data["target"]
    train_index, test_index = data["index"]

    num_params = covariates[train_index].shape[1]
    

    # define distribution and the formula for the distibutional parameter
    distribution  = config["family"]
    X_df = pd.DataFrame(covariates[train_index,], columns=[f"x{i}" for i in range(num_params)])
    covariate_names = X_df.columns.tolist()
    orthog = False
    if not "orthog" in config.keys() or not config["orthog"]:
        orthog = True
        for i in range(num_params):
            X_df[f"z{i}"] = X_df[f"x{i}"]
    
    y = pd.DataFrame(target[train_index], columns=["y"])
    target_name = y.columns[0]
    
    nn_covariate_names = covariate_names if not orthog else [f"z{i}" for i in range(num_params)]

    linear_formula = " + ".join(covariate_names)
    distribution = distribution.capitalize()
    match config["optimizer"].lower():
            case "adam":
                optimizer = optim.Adam
            case "rmsprop":
                optimizer = optim.RMSprop
            case "sgd":
                optimizer = optim.SGD
            case _: 
                raise NotImplementedError("Currently only 'adam', 'rmsprop' and 'sgd' are implemented as optimizers.")
        
    if "spline_terms" in config.keys() and config["spline_terms"] is not None:
        spline_term = []
        for var in config["spline_terms"]:
            spline_term.append("+spline(" + var + ", bs='bs', df=10)")
        linear_formula = linear_formula + "".join(spline_term)

    if distribution == "Normal":
        
        NN_formula_loc = "+d1("+",".join(nn_covariate_names)+")"
        NN_formula_scale = "+d2("+",".join(nn_covariate_names)+")"
        
        formulas = {"loc": "~1+" + linear_formula + NN_formula_loc,
                    "scale": "~1+" + linear_formula + NN_formula_scale}
        # define the deep neural networks' architectures and output shapes used in the above formula
        layers_1 = [nn.Linear(num_params, config["width"]), nn.ReLU()]
        for i in range(config["depth"]):
            layers_1.append(nn.Linear(config["width"], config["width"]))
            layers_1.append(nn.ReLU())
        layers_1.append(nn.Linear(config["width"], config["output_shape"]))
        basic_nn_1 = nn.Sequential(*layers_1)

        layers_2 = [nn.Linear(num_params, config["width_2"]), nn.ReLU()]
        for i in range(config["depth_2"]):
            layers_2.append(nn.Linear(config["width_2"], config["width_2"]))
            layers_2.append(nn.ReLU())
        layers_2.append(nn.Linear(config["width_2"], config["output_shape_2"]))
        basic_nn_2 = nn.Sequential(*layers_2)

        deep_models_dict = {
        'd1': {
            'model': basic_nn_1,
            'output_shape': config["output_shape"]},
        'd2': {
            'model': basic_nn_2,
            'output_shape': config["output_shape_2"]}
        }

        # define your training hyperparameters
        
        optimizer_params = {
            'lr': config["lr"],
            'weight_decay': config["weight_decay"]
        }

        train_parameters = {
            'batch_size': config["batch_size"],
            'epochs': config["epochs"],
            'degrees_of_freedom': {'loc': config["df"], 'scale': config["df_2"]},
            'optimizer' : optimizer,
            'dropout_rate': config["dropout_rate"],
            'optimizer_params': optimizer_params
            }
    elif distribution == "Bernoulli":

        NN_formula_logits = "+d1("+",".join(nn_covariate_names)+")"
        formulas = {"logits": "~1+" + linear_formula + NN_formula_logits}
        layers = [nn.Linear(num_params, config["width"]), nn.ReLU()]
        for i in range(config["depth"]):
            layers.append(nn.Linear(config["width"], config["width"]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(config["width"], config["output_shape"]))
        basic_nn = nn.Sequential(*layers)

        deep_models_dict = {
            'd1': {
                'model': basic_nn,
                'output_shape': config["output_shape"]
            }
        }
        optimizer_params = {
            'lr': config["lr"],
            'weight_decay': config["weight_decay"]
        }

        train_parameters = {
            'batch_size': config["batch_size"],
            'epochs': config["epochs"],
            'degrees_of_freedom': {'logits': config["df"]},
            'optimizer' : optimizer,
            'dropout_rate': config["dropout_rate"],
            'optimizer_params': optimizer_params
            }

    else:
        raise ValueError("Currently only normal and Bernoulli distributions are supported for SDDR models.")

    if config["early_stop_epochs"] > 0:
        train_parameters = train_parameters | {"early_stop_epochs": config["early_stop_epochs"]}

    # define output directory
    output_dir = './outputs'

    sddr = CustomSddr(distribution=distribution,
                formulas=formulas,
                deep_models_dict=deep_models_dict,
                train_parameters=train_parameters,
                output_dir=output_dir)
    X_val_df = pd.DataFrame(covariates[test_index,], columns=[f"x{i}" for i in range(num_params)])
    if orthog:
        for i in range(num_params):
            X_val_df[f"z{i}"] = X_val_df[f"x{i}"]
    y_val_df = pd.DataFrame(target[test_index], columns=["y"])

    hist = sddr.custom_train(train_structured_data=X_df,
        train_target=y, val_structured_data=X_val_df,
        val_target=y_val_df, plot=False, hyperparam_opt=True)

def k_fold_performance_analysis_sddr(covariates, target, parameter_space, metrics, temp_dir, mode, k=5,
                                      test_name = "k_fold_performance_analysis" + date.today().strftime("%Y-%m-%d"), number_of_trials_per_split=8, normalize = True):
    """Performs a k-fold cross-validation where for each split a hyperparameter optimization is performed on the "training set" and the best model evaluated on the "test set".
    
    "mode" is one of "binary", "continuous"
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

        result_grid = ray_hyperparam_opt_with_sddr(covariates_in_split, targets_in_split, actual_train_index, validation_index, parameter_space, metrics,temp_dir, experiment_name=f"{test_name}_split_{fold_number}", number_of_trials_per_split=number_of_trials_per_split)

        # get best configuration
        best_result = result_grid.get_best_result(metric="validation_loss", mode="min", scope="all")
        best_result_dataframe = result_grid.get_dataframe(filter_metric="validation_loss", filter_mode="min")
        best_checkpoint = best_result.checkpoint
        # load the best model
        # first, get working directory
        with best_checkpoint.as_directory() as checkpoint_dir:
            working_dir = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))["working_dir"].replace("\\", "/")
        print("trying to load model from", working_dir)
        config_dict = torch.load(os.path.join(working_dir, "outputs\\custom_config.pth").replace("\\", "/"))
        print(list(pd.DataFrame(covariates_in_split[train_index, ], columns=[f"x{i}" for i in range(covariates_in_split.shape[1])]).columns))
        best_model = CustomSddr(distribution=config_dict["distribution"],
                formulas=config_dict["formulas"],
                deep_models_dict=config_dict["deep_models_dict"],
                train_parameters=config_dict["train_parameters"],
                output_dir=config_dict["output_dir"],)
        print(pd.DataFrame(covariates_in_split[train_index, ]).columns)
        best_model.load(os.path.join(working_dir, "outputs/model.pth"), training_data=pd.DataFrame(covariates_in_split[train_index, ], columns=[f"x{i}" for i in range(covariates_in_split.shape[1])]))
        print("loading successful")
        print(best_result_dataframe.columns)
        test_covariates = pd.DataFrame(covariates_in_split[test_index, ], columns=[f"x{i}" for i in range(covariates_in_split.shape[1])])
        if not "orthog" in config_dict.keys() or not config_dict["orthog"]:
            for i in range(covariates_in_split.shape[1]):
                test_covariates[f"z{i}"] = test_covariates[f"x{i}"]
        test_target_tensor = torch.tensor(targets_in_split[test_index], dtype=torch.float32).reshape(-1)
        test_loglike = best_model.predict(test_covariates, clipping = True)[0].log_prob(test_target_tensor.to(best_model.device).reshape(-1, 1)).mean().item()
        if mode == "binary":
            predictions = best_model.predict(test_covariates, clipping=True)[0].logits.detach().cpu().numpy()
            brier_score = np.mean((predictions - test_target_tensor.numpy())**2)

            # compute ROC AUC:
            roc_auc = roc_auc_score(test_target_tensor.numpy(), predictions)

            # append the results to the dataframe:
            metrics_dataframe.loc[fold_number] = [test_loglike, brier_score, roc_auc]
        elif mode == "continuous":
            #compute test mse:
            predictions = best_model.predict(test_covariates, clipping=True)[0].loc.detach().cpu().numpy()
            mse = np.mean((predictions - test_target_tensor.numpy())**2)
            # append the results to the dataframe:
            metrics_dataframe.loc[fold_number] = [test_loglike, mse]

    return metrics_dataframe