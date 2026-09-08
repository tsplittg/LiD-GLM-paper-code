from matplotlib import pyplot as plt
import lidglm
from lidglm.extended_glm.extended_glm import ExtendedGLM
from lidglm.extended_glm.transformer.transformer import TransformerUtils
from lidglm.torch_glm.torch_glm import TorchGLM
import torch
import numpy as np
import itertools
from tqdm import tqdm
from ray import train
from ray.train import Checkpoint
import normflows as nf
import os 
import tempfile
import statsmodels
import statsmodels.api as sm
from lidglm.torch_glm.torch_glm_utils import TorchGLM_Utils
import copy

from pathlib import Path, PurePath
import ntpath

class ExtendedGLM_Utils():
    """
    Helper class containing various functions surrounding the extended Glm. Usable for easy initialization, training (without lightning) as well as
    for various interpretation tasks.
    """

    def start_with_statsmodels(cls, endog, exog,
                               statsmodels_family: sm.families.Family = None, family_str: str = None, link_str: str = None, 
                               offset=None, exposure=None, freq_weights=None, var_weights=None, bias = True,
                               missing='none', fit_max_iter: int = 10 ** 3,
                               net_transform = None, net_transform_endog = None,
                               orthog =False):
        
        torch_glm, endog, exog = TorchGLM_Utils.start_with_statsmodels(endog=endog, exog=exog, statsmodels_family=statsmodels_family,
                                                          family_str=family_str, link_str=link_str, offset=offset, exposure=exposure,
                                                          freq_weights=freq_weights, var_weights=var_weights, missing=missing, fit_max_iter=fit_max_iter, bias=bias)
        
        return ExtendedGLM(regular_glm=torch_glm, net_transform=net_transform, net_transform_endog=net_transform_endog, orthog=orthog), endog, exog

    def convert_sm_glm(cls, glm: statsmodels.genmod.generalized_linear_model.GLMResults, net_transform = None, net_transform_endog = None, bias: str = None, orthog = False):
        """
        Parameters:
        -----------
        bias: str
            string indicating whether the glm contains no bias ("none"), the bias is the first variable ("first") or the last ("last")
        """

        torch_glm, endog, exog = TorchGLM_Utils.convert_sm_glm(glm=glm, bias =bias)
        return ExtendedGLM(regular_glm=torch_glm, net_transform=net_transform, net_transform_endog=net_transform_endog, orthog=orthog, dispersion = glm.scale), endog, exog

    def train_model(self, extended_glm: ExtendedGLM, exog: torch.Tensor, endog: torch.Tensor,
                    epochs: int = 10**2, num_batches: int = 10, batch_size: int = None, lr: float= 5e-5,
                    part_to_train: str = "full", loss_crit: str = "MSE",
                    X_test: torch.Tensor = None, y_test:torch.Tensor = None,
                    metrics:dict = None,
                    hyperparam_opt:bool = False, traditional_beta:np.array =None,
                    report_every_n_epochs= None,
                    patience = 10, early_stopping = False,
                    get_best_losses: bool = False, use_tqdm: bool = True, **kwargs):
        """
        Implementation of a basic training loop for the deepGLM. The model can of course also be trained by external functions.

        Parameters
        -----------

        part_to_train : string, optional (default: "full")
            Determines which part of the model should be trained. Enables to, for example, pre-train a regular GLM and then
            train the network afterward to optimize further.
            One of : "full", "linear_part", "network", "only_y_transform", "only_glm_x_transform"
        loss_crit : string, optional (default: "MSE")
            Determines the loss criterion that should be used for training.
            One of: "MSE"
        X_test : torch.Tensor (optional)
            If either X_test or y_test is specified, both have to be specified. If they are, at the end of epoch
            during draining, model performance is evaluated on this test set.
        y_test : torch.Tensor (optional)
            See X_test.
        hyperparam_opt : bool
            Boolean signaling whether hyperparameter tuning via raytune is currently being performed. If true, some
            callbacks, reports etc. are being handled
        traditional_beta : np.array
            Array containing the beta values of a traditional GLM that the beta values
            of the LDGLM should be comapred to. It is assumed that a bias is given as the
            first element
            
        """
        
        

         
        if report_every_n_epochs is None:
            if epochs < 100:
                report_every_n_epochs = 1
            else:
                report_every_n_epochs = epochs //100
            
            
        # make sure that Module and all relevant parameters are on right device (probably GPU)
        extended_glm = extended_glm.to(extended_glm.used_device)
        extended_glm.train()
        exog = exog.to(extended_glm.used_device)
        endog = endog.to(extended_glm.used_device)
        #initialize output arrays
        average_batch_loss_hist = np.array([])
        test_loss_hist = np.array([])


        if X_test is not None or y_test is not None:
            if X_test is None or y_test is None:
                raise ValueError("Either both or neither X_test and y_test must be specified")
            X_test = X_test.to(extended_glm.used_device)
            y_test = y_test.to(extended_glm.used_device)
            test_set_provided = True
        else:
            test_set_provided = False

        # translate loss_ criterion:
        
        loss_crit = loss_crit.lower().replace(" ", "")
        match loss_crit:
            case "mse":
                loss_criterion = lambda batch_X, ground_truth: torch.nn.MSELoss()(extended_glm.predict_endog(batch_X), ground_truth)
            case "log_like":
                loss_criterion = lambda batch_X, ground_truth : -extended_glm.log_like_obs(exog=batch_X, endog= ground_truth, dispersion=extended_glm.compute_dispersion(exog, endog)).mean()
            case "orthog_log_like":
                loss_criterion = lambda batch_X, ground_truth : self.orthog_loss(extended_glm = extended_glm,
                                                                            exog = batch_X, endog = ground_truth,
                                                                            weight = 1,
                                                                            loss_without_orthog =-torch.mean(extended_glm.log_like_obs(exog=batch_X, endog= ground_truth, dispersion=extended_glm.compute_dispersion(exog, endog))))
            case "bce":
                loss_criterion = lambda batch_X, ground_truth: torch.nn.BCELoss()(extended_glm.predict_endog(batch_X), ground_truth)
            case _:
                raise ValueError("Specified Loss not implemented or not recognized")

        # translate which part of the model should be trained:
        part_to_train = part_to_train.lower().replace(" ", "")
        match part_to_train:
            case "full":
                optimizer = torch.optim.Adam(itertools.chain(extended_glm.nu_1.parameters(),extended_glm.glm.parameters(), extended_glm.net_transform_endog.parameters(), 
                                                             extended_glm.T_2.parameters() if hasattr(extended_glm, "T_2") else [] ), lr=lr)
            case "only_glm_x_transform":
                optimizer = torch.optim.Adam(itertools.chain(extended_glm.nu_1.parameters(),extended_glm.glm.parameters()), lr=lr)
            case "only_y_transform":
                optimizer = torch.optim.Adam(itertools.chain(extended_glm.net_transform_endog.parameters(), 
                                                             extended_glm.T_2.parameters()if hasattr(extended_glm, "T_2") else [] ), lr=lr)
            case "network":
                optimizer = torch.optim.Adam(extended_glm.nu_1.parameters(), lr=lr)
            case "linear_part":
                optimizer = torch.optim.Adam(extended_glm.glm.parameters(), lr=lr)
            case _:
                raise ValueError(f"Value {part_to_train} for parameter part_to_train not recognized")


        used_batch_size = batch_size if batch_size is not None else exog.shape[0]//num_batches
        data_loader = torch.utils.data.DataLoader(list(zip(endog, exog)), batch_size=used_batch_size,
                                                  shuffle=True)
        computed_metrics ={}
        #apparently, raytune and tqdm combined can cause issues
        checkpoint = None
        if hyperparam_opt:
            checkpoint = train.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
                starting_epoch = checkpoint_dict["epoch"] + 1
                epochs = epochs-starting_epoch
                extended_glm.load_state_dict(checkpoint_dict["model_state"])
        epoch_range = range(epochs) if "current_epoch" not in kwargs else range(kwargs["current_epoch"], kwargs["current_epoch"]+epochs)
        epoch_range = epoch_range if (hyperparam_opt or not use_tqdm) else tqdm(epoch_range)
        encountered_error = False

        #initialize parameters for early stopping
        best_loss = np.inf
        current_patience = patience
        best_train_loss = None
        best_test_loss = None
        best_model_state = extended_glm.state_dict()
        saved_epochs = []
        best_epoch = 0
        for epoch in epoch_range:
            batchwise_loss = np.array([])
            for batch_id, (batch_y, batch_X) in enumerate(data_loader):
                optimizer.zero_grad()
                loss = loss_criterion(batch_X, batch_y)
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                else:
                    print(" encountered loss zero or infinity")
                    print("batch_X: ", batch_X)
                    print("batch_y: ", batch_y)
                    print("predicted mean:", extended_glm.predict(batch_X))
                    print("final prediction: ", extended_glm.predict_endog(batch_X))
                    #encountered_error = True
                    raise ValueError("Encountered loss zero or infinity")
                    continue
                batchwise_loss = np.append(batchwise_loss, loss.to('cpu').data.numpy())
                nf.utils.update_lipschitz(extended_glm, 25) #ToDo: still needed?
            average_batch_loss = batchwise_loss.mean()
            loss_dict = {"average_loss": average_batch_loss}
            if test_set_provided:
                extended_glm.eval()
                with torch.no_grad():
                    test_loss = loss_criterion(X_test, y_test).detach().to("cpu").detach().numpy()
                    test_loss_hist = np.append(test_loss_hist, test_loss)
                    loss_dict["validation_loss"] = test_loss
                    #early stopping:
                    if hyperparam_opt and metrics is not None:
                        for key, metric in metrics.items():
                            computed_metrics[key] = metric(extended_glm=extended_glm, X_test=X_test, y_test=y_test, traditional_beta= traditional_beta)

                    if test_loss <= best_loss:
                        best_loss = test_loss
                        best_model_state = copy.deepcopy(extended_glm.state_dict())
                        best_epoch = epoch
                        best_loss_dict = loss_dict|computed_metrics
                        current_patience = patience
                    elif early_stopping:
                        current_patience -= 1
                        if current_patience == 0:
                            extended_glm.load_state_dict(best_model_state)
                            break
                    

                extended_glm.train()
            
            if get_best_losses:
                if best_train_loss is None or average_batch_loss < best_train_loss:
                    best_train_loss = average_batch_loss
                if test_set_provided and (best_test_loss is None or test_loss < best_test_loss):
                    best_test_loss = test_loss
            average_batch_loss_hist = np.append(average_batch_loss_hist, average_batch_loss)
            if hyperparam_opt and (epoch%report_every_n_epochs==0 or epoch == epochs-1):
                #save checkpoint in temporary file; will be permanent after being reported
                saved_epochs.append(epoch)
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save({"epoch":epoch, "model_state": extended_glm.state_dict(), "working_dir": os.getcwd()},
                               os.path.join(tempdir, "checkpoint.pt"))
                    train.report(metrics=loss_dict|computed_metrics,
                                 checkpoint = Checkpoint.from_directory(tempdir))
            if encountered_error:
                break
            
        # load best model state before returning:
        if hyperparam_opt and best_epoch not in saved_epochs:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save({"epoch":best_epoch, "model_state": best_model_state, "working_dir": os.getcwd()},
                           os.path.join(tempdir, "checkpoint.pt"))
                train.report(metrics=best_loss_dict,
                             checkpoint = Checkpoint.from_directory(tempdir))

        if early_stopping:
            extended_glm.load_state_dict(best_model_state)
        extended_glm.set_dispersion(exog=exog, endog=endog, gradient=False)
        if get_best_losses:
            return average_batch_loss_hist, test_loss_hist, best_train_loss, best_test_loss
        return average_batch_loss_hist, test_loss_hist
    
    def orthog_loss(self, extended_glm: ExtendedGLM, exog: torch.Tensor, endog: torch.Tensor, weight: float, loss_without_orthog: torch.Tensor):
        """
        Adds an orthogonalization penalty on top of another loss term. This sum is weighted by the parameter "weight".
        """
        if extended_glm.orthog is not False:
            raise ValueError("Orthogonalization should not simultaneously be used as loss-based and rigorous orthogonolization.")
        

        neural_net = extended_glm.NN_transform(exog) - exog
        normalized_exog = (exog - torch.mean(exog, dim=0)) / torch.std(exog, dim=0)
        normalized_neural_net = (neural_net - torch.mean(neural_net, dim=0)) / torch.std(neural_net, dim=0)

        orthog_loss_per_obs = torch.pow(torch.sum(normalized_exog * normalized_neural_net, dim=1),2)
        orthog_loss = torch.mean(orthog_loss_per_obs)
        return loss_without_orthog + weight * orthog_loss
    
    def _compute_input_bounds(self, input_bounds, exog: torch.Tensor, verbose: bool):
        if input_bounds is None and exog is None:
            raise ValueError("Either input_bounds or an exog to compute the bounds from needs to be specified.")
        if input_bounds is not None and exog is not None:
            raise ValueError("input_bounds and exog cannot both be specified.")
        if input_bounds is None and exog is not None:
            #get min and max of each column
            input_bounds_min = list(exog.min(axis = 0)[0].detach().cpu().numpy())
            input_bounds_min = [float(value) for value in input_bounds_min] #np.float causes errors down the line
            input_bounds_max = list(exog.max(axis = 0)[0].detach().cpu().numpy())
            input_bounds_max = [float(value) for value in input_bounds_max]
            input_bounds =[list(bounds) for bounds in zip(input_bounds_min, input_bounds_max)]
            if verbose:
                print("computed the input bounds: ", input_bounds)

        return input_bounds
    
    def approximate_lipschitz_selected_dimensions(self, extended_glm: ExtendedGLM, dimensions: tuple, pnorm: int = 2):
        """approximates the Lipschitz constant of nu_1 seen as a function between a specific input covariate dimension to a specific output dimension with a layerwise computation."""
        first_index, second_index = dimensions
        total_constant = 1
        for block_index in range(len(extended_glm.nu_1.blocks)):
            weights = []
            biases = []
            block_constant = 1
            for layer in extended_glm.nu_1.blocks[block_index].nnet.net:
                if isinstance(layer, lidglm.extended_glm.transformer.Custom_Lipschitz.Custom_Lipschitz.Custom_InducedNormLinear):
                    weights.append(layer.normalized_weight.detach().cpu().numpy())
                    biases.append(layer.bias.detach().cpu().numpy())
                # now, we adjsut the first and last linear layers to only consider the desired input/output dimensions:
            if block_index == 0:
                weights[0] = weights[0][:,first_index].reshape(-1,1)
            if block_index == len(extended_glm.nu_1.blocks)-1:
                weights[-1] = weights[-1][second_index, :].reshape(1,-1)
            for weight in weights:
                block_constant *= np.linalg.norm(weight, ord=pnorm)
            total_constant *= block_constant
        return total_constant


    def scale_nonlinear_part(self, extended_glm: ExtendedGLM, scale_1:float, scale_2:float=1):
        """
        Scale the nonlinear parts of the network (\nu_1 and \nu_2) by two different
        constants.

        Parameters
        ----------
        extended_glm : ExtendedGLM
            DESCRIPTION.
        scale_1 : float
            A float to scale \nu_1 by.
        scale_2 : float
            A float to scale \nu_2 by. Defaults to one. Must be one if \nu_2=Identity

        Returns
        -------
        ExtendedGLM:
            A deepcopy of the input extended GLM with the weights adjusted.

        """
        #The deepcopy function will result in an error if the dispersion parameter is not a leaf node; after trainnig, the function set_dispersion should therefore be called once with gradient = False
        lidglm = copy.deepcopy(extended_glm)
        
        if isinstance(lidglm.net_transform_endog, torch.nn.Identity) and scale_2 != 1:
            raise ValueError("Cannot scale strictly linear transformation.")
        
        #we scale the last weight matrix in the last block of each transformation:
        print(lidglm.nu_1.blocks[-1])
        with torch.no_grad():
            lidglm.nu_1.blocks[-1] = lidglm.nu_1.blocks[-1].scale_last_layer(scale_1)
            if not isinstance(lidglm.net_transform_endog, torch.nn.Identity):
                lidglm.net_transform_endog.blocks[-1] = lidglm.net_transform_endog.blocks[-1].scale_last_layer(scale_2)
        print(lidglm.nu_1.blocks[-1])
        return lidglm

    def load_from_param_dict_and_state(self, param_dict, state_dict) -> ExtendedGLM:
        torch_glm = TorchGLM(**(param_dict["glm_params"])) 
        T_1 = TransformerUtils().initialize_from_saved_params(param_dict["nu_1_params"])
        T_2 = TransformerUtils().initialize_from_saved_params(param_dict["nu_2_params"]) if param_dict["nu_2_params"] is not None else None
        loaded_lidglm = ExtendedGLM(regular_glm=torch_glm, net_transform=T_1, net_transform_endog=T_2, orthog=param_dict["orthog"],
                                    dispersion=param_dict["dispersion"], enable_cuda= param_dict["enable_cuda"])
        loaded_lidglm.load_state_dict(state_dict)
        loaded_lidglm.original_weights = None
        loaded_lidglm.original_bias = None
        return loaded_lidglm

    def load_from_saved_files(self, path_to_param_dict: str, path_to_state_dict: str) -> ExtendedGLM:
        state_dict = torch.load(path_to_state_dict)
        param_dict = torch.load(path_to_param_dict)
        return self.load_from_param_dict_and_state(param_dict, state_dict)
        
        
    def load_from_ray_checkpoint(self, checkpoint, artifact_dir = None) -> ExtendedGLM:
        """
        Checkpoint has to include the keys "model_state" and "working_dir", where a file "initialization_params.pt" is expected to be found in that working directory.
        """
        artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        with checkpoint.as_directory() as checkpoint_dir:
            if artifact_dir is not None:
                original_working_dir = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))["working_dir"]
                _, subdir = str(original_working_dir).split("artifacts")
                subdir = subdir[1:].replace("\\", "/")
                subdir = ntpath.normpath(subdir)
                new_working_dir = Path.joinpath(artifact_dir, subdir)
            else:
                new_working_dir = Path(torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))["working_dir"])
            model_state_dict = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))["model_state"]
            param_dict = torch.load(str(PurePath(ntpath.normpath(new_working_dir / "initialization_params.pt"))).replace("\\", "/"))

        return self.load_from_param_dict_and_state(param_dict, model_state_dict)
