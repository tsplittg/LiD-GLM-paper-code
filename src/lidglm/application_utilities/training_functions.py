"""Script containing training functions used during a raytune optimization/comparison of different models. Mostly used for data trials and model comparison on data."""

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import os
from ray import train
from ray.train import Checkpoint
from lidglm.application_utilities.metric_functions import LidglmMetrics
from lidglm import ExtendedGLM_Utils, TransformerUtils
import tempfile
import copy


def train_models(config, data, other_specifications=None):
    """
    Trains each model with the appropriate training function.
    The specific keys that each training function expects can be found in the respective docstrings.
    """
    used_config = config["model_specifications"]
    kfold = data["kf"]
    
    if other_specifications is None:
        other_specifications = {}
    data["index"] = list(kfold.split(data["target"]))[config["index"]]

    match used_config["model_type"]:
        case "ldglm": return train_lidglm(used_config, data, other_specifications)
        case "basic_nn": return train_nn(used_config, data, other_specifications)
        case _: raise ValueError("This model does not have an implemented training function.")
    
    
def train_lidglm(config, data, other_specifications=None):
    """
    Trains a Lipschitz-DeepGLM model on the given data.
    
    config contains the following keys:
    "index"  -> tuple of the form (train_index, test_index)
    "family" -> string specifying the glm family
    "link" -> string specifying the glm link
    "lipschitz_const" -> float
    "norm" -> float, which norm to use
    "lr" -> float:  learning rate
    "epochs" -> integer: number of epochs
    "loss_crit" -> string, specifies th loss function to be used
    
    
    optional keys:
    "width" -> int, width of each layer of the NN but the first and last
    "depth" -> number of hidden layers not counting the first and last layer of each block
    "lipschitz_const_2"
    "width_2"
    "depth_2"
    "num_blocks_2"
    "activation_2"

    data contains the following keys:
    "target" -> np.array
    "covariates" -> np.array
    """

    covariates = data["covariates"]
    target = data["target"]
    train_index, test_index = data["index"]

    num_params = covariates[train_index].shape[1]
    util =ExtendedGLM_Utils()
    total_lipschitz_const = config["lipschitz_const"]
    norm_used = config["norm"]
    
    width = config["width"] if "width" in config else 5
    depth = config["depth"] if "depth" in config else np.ceil(np.log(total_lipschitz_const)/np.log(2)).astype(int)
    
    if "num_blocks" in config:
        num_blocks = config["num_blocks"]
    else:
        num_blocks = None

    if "group_size" not in config:
        group_size = 3
    else:
        group_size = config["group_size"]
    
    transf_util = TransformerUtils()

    #chck if some parameter keys for the second network are present:
    if all([key in config for key in ["lipschitz_const_2", "width_2", "depth_2", "num_blocks_2", "activation_2"]]):
        present_keys = [key for key in ["lipschitz_const_2", "width_2", "depth_2", "num_blocks_2", "activation_2"] if key in config]
        nu_2_params ={}
        for key in present_keys:
            if key == "lipschitz_const_2":
                nu_2_params["lipschitz_const"] = config["lipschitz_const_2"]
            elif key == "width_2":
                nu_2_params["width"] = config["width_2"]
            elif key == "depth_2":
                nu_2_params["depth"] = config["depth_2"]
            elif key == "num_blocks_2":
                nu_2_params["num_blocks"] = config["num_blocks_2"]
            elif key == "activation_2":
                nu_2_params["activation_string"] = config["activation_2"]
        nu_2_params["domain_codomain"] = norm_used
        nu_2_params["group_size"] = group_size
        nu_2_params["num_params"] = 1
        nu_2 = transf_util.initialize_transformer(**nu_2_params)
    else:
        nu_2 = None

    nu_1 = transf_util.initialize_transformer(num_params=num_params, lipschitz_const=total_lipschitz_const,
                                              domain_codomain=norm_used, width=width, depth=depth, num_blocks=num_blocks, activation_string=config["activation"], group_size=group_size)

    extended_glm, target_tensor, covariate_tensor = util.start_with_statsmodels(endog = target[train_index], exog = covariates[train_index,],
                               family_str= config["family"], link_str = config["link"], bias = True,
                               missing='none', fit_max_iter = config["epochs"],
                               net_transform = nu_1, net_transform_endog = nu_2)
    extended_glm.used_device = "cpu"
    extended_glm.to(torch.device("cpu"))
    
    # save parameters necessary to restore the model later:
    torch.save(extended_glm.get_initialization_params(), "initialization_params.pt")

    if "dist_trad_beta" in other_specifications["metrics"]:
        traditional_beta = np.append(extended_glm.read_linear_weights(), extended_glm.read_linear_bias()).reshape(-1)
        torch.save(traditional_beta, "traditional_beta.pt")
    else:
        traditional_beta = None
    if config["part_to_train"]=="nu_p_then_nu_d":
        loss_hist = []
        test_error_hist = []
        _loss_hist, _test_error_hist = util.train_model(extended_glm, exog=covariate_tensor, endog=target_tensor,
                                                      X_test=torch.Tensor(covariates[test_index,]),
                                                      y_test=torch.tensor(target[test_index], dtype=torch.float32),
                                                      num_batches=config["num_batches"],
                                                      loss_crit="log_like", metrics=other_specifications["metrics"],
                                                      epochs=config["epochs"]//2, part_to_train="only_glm_x_transform",
                                                      lr=config["lr"], hyperparam_opt=True,
                                                      traditional_beta=traditional_beta,
                                                      report_every_n_epochs=30, early_stopping=True, patience=200)
        loss_hist = np.append(loss_hist, list(_loss_hist))
        test_error_hist = np.append(test_error_hist, list(_test_error_hist))
        _loss_hist, _test_error_hist = util.train_model(extended_glm, exog=covariate_tensor, endog=target_tensor,
                                                        X_test=torch.Tensor(covariates[test_index,]),
                                                        y_test=torch.tensor(target[test_index], dtype=torch.float32),
                                                        num_batches=config["num_batches"],
                                                        loss_crit="log_like", metrics=other_specifications["metrics"],
                                                        epochs=config["epochs"] // 2,
                                                        part_to_train="only_y_transform",
                                                        lr=config["lr"], hyperparam_opt=True,
                                                        traditional_beta=traditional_beta,
                                                        report_every_n_epochs=30, early_stopping=True, patience=100, current_epoch = len(loss_hist)+1)
        loss_hist = np.append(loss_hist, list(_loss_hist))
        test_error_hist = np.append(test_error_hist, list(_test_error_hist))
    else:
        loss_hist, test_error_hist = util.train_model(extended_glm,exog=covariate_tensor, endog = target_tensor,
                                                      X_test=torch.Tensor(covariates[test_index,]), 
                                                      y_test=torch.tensor(target[test_index], dtype=torch.float32),num_batches=config["num_batches"],
                                                      loss_crit="log_like", metrics=other_specifications["metrics"],
                                                      epochs=config["epochs"], part_to_train=config["part_to_train"], lr=config["lr"], hyperparam_opt=True,
                                                      traditional_beta=traditional_beta,
                                                      report_every_n_epochs=30, early_stopping=True, patience=100)
    
    #for debugging:
    extended_glm.eval()
    computed_metrics = {}
    for key, metric in other_specifications["metrics"].items():
        computed_metrics[key] = metric(extended_glm=extended_glm, X_test=torch.Tensor(covariates[test_index,]),
                                       y_test=torch.Tensor(target[test_index]), traditional_beta= traditional_beta)
    train.report(metrics = {"average_loss": loss_hist[-1], "validation_loss":test_error_hist[-1]}|computed_metrics|dict(alternative_loglike=-extended_glm.log_like_obs(torch.tensor(target[test_index], dtype=covariate_tensor.dtype),
                                                                                                                                                                       torch.tensor(covariates[test_index,], dtype = covariate_tensor.dtype)).detach().numpy().sum()))
    
    if "evaluation_plots" in other_specifications:
        util.evaluation_plots(extended_glm, exog=covariate_tensor, endog=target_tensor,
                              custom_plot_functions=other_specifications["evaluation_plots"] if other_specifications is not None else None)

def train_nn(config, data, other_specifications=None):
    """
    Trains a basic Neural Network on the given data. Only really makes sense for categorical data where the 
    traditional NN also predicts a probability distribution.
    
    config contains the following keys:
    "index"  -> tuple of the form (train_index, test_index)
    "lr" -> float:  learning rate
    "epochs" -> integer: number of epochs
    "width" -> int, width of each layer of the NN but the first and last
    "depth" -> number of hidden layers not counting the first and last layer of each block

    data contains the following keys:
    "target" -> np.array
    "covariates" -> np.array
    """
    early_stopping = True if "early_stopping" not in config else config["early_stopping"]
    patience = 150 if "patience" not in config else config["patience"]
    report_every_n_epochs = 30 if "report_every_n_epochs" not in config else config["report_every_n_epochs"]
    covariates = torch.tensor(data["covariates"], dtype=torch.float32)
    target = torch.tensor(data["target"], dtype=torch.float32)
    train_index, test_index = data["index"]
    num_params = data["covariates"][train_index,].shape[1]

    layers = [nn.Linear(num_params, config["width"]), nn.ReLU()]
    for i in range(config["depth"]):
        layers.append(nn.Linear(config["width"], config["width"]))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(config["width"], 1))

    layers.append(nn.Sigmoid()) # we only compare to a NN in the binary setting
    basic_nn = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(basic_nn.parameters(), lr=config["lr"], weight_decay=10**-4)
    loss_criterion = nn.BCELoss()
    num_epochs = config["epochs"]
    train_covariates_tensor = covariates[train_index,]
    train_target_tensor = target[train_index].reshape(-1)
    test_covariates_tensor = covariates[test_index,]
    test_target_tensor = target[test_index].reshape(-1)
    data_loader = torch.utils.data.DataLoader(list(zip(train_target_tensor, train_covariates_tensor)), batch_size = 512 if "num_batches" not in config else len(train_index)//config["num_batches"],
                                                  shuffle=True)
    average_loss_hist = np.array([])
    best_val_loss = np.inf
    current_patience = 0

    for epoch in range(num_epochs):
        batchwise_loss = np.array([])
        for batch_id, (batch_y, batch_X) in enumerate(data_loader):
            optimizer.zero_grad()
            loss = loss_criterion(basic_nn(batch_X).reshape(batch_y.shape), batch_y)
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
            else:
                print(" encountered loss zero or infinity")
                basic_nn.load_state_dict(best_model_state)
                break
            batchwise_loss = np.append(batchwise_loss, loss.detach().cpu().numpy())
        average_loss = batchwise_loss.flatten().mean()
        average_loss_hist = np.append(average_loss_hist, average_loss)
        with torch.no_grad():
            val_prediction = basic_nn(test_covariates_tensor).reshape(-1).clamp(torch.finfo(torch.float32).eps, 1-torch.finfo(torch.float32).eps)
            if isinstance(basic_nn[-1], nn.Sigmoid):
                val_log_like = torch.log(val_prediction).reshape(-1)*test_target_tensor +torch.log(1- val_prediction).reshape(-1)*(1-test_target_tensor)
                val_log_like = float(val_log_like.sum().detach().cpu().numpy().item())
                neg_val_log_like = -val_log_like
            if neg_val_log_like < best_val_loss:
                best_val_loss = neg_val_log_like
                current_patience = 0
                best_model_state = copy.deepcopy(basic_nn.state_dict())
            else:
                current_patience += 1
        if early_stopping and current_patience >= patience:
            basic_nn.load_state_dict(best_model_state)
            break
        if epoch%report_every_n_epochs==0 or epoch == num_epochs-1:
                #save checkpoint in temporary file; will be permanent after being reported
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save({"epoch":epoch, "model_state": basic_nn.state_dict(), "working_dir": os.getcwd()},
                               os.path.join(tempdir, "checkpoint.pt"))
                    checkpoint = Checkpoint.from_directory(tempdir)
                    LidglmMetrics.numpy_based_metrics(np.round(val_prediction), test_target_tensor.detach().cpu().numpy(), model_string="NN", metrics_list = list(other_specifications["metrics"].keys()),
                                      log_like=val_log_like, checkpoint = checkpoint, other_computed_metrics={"validation_loss": neg_val_log_like, "average_loss": average_loss})
        #else:
        #    LidglmMetrics.numpy_based_metrics(val_prediction, test_target_tensor.detach().cpu().numpy(), model_string="NN", metrics_list = list(other_specifications["metrics"].keys()),
        #                              log_like=val_log_like, checkpoint = None, other_computed_metrics={"validation_loss": neg_val_log_like, "average_loss": average_loss})

    with tempfile.TemporaryDirectory() as tempdir:
                    torch.save({"epoch":epoch, "model_state": basic_nn.state_dict(), "working_dir": os.getcwd()},
                               os.path.join(tempdir, "checkpoint.pt"))
                    checkpoint = Checkpoint.from_directory(tempdir)
                    LidglmMetrics.numpy_based_metrics(np.round(val_prediction), test_target_tensor.detach().cpu().numpy(), model_string="NN", metrics_list = list(other_specifications["metrics"].keys()),
                                      log_like=val_log_like, checkpoint = checkpoint, other_computed_metrics={"validation_loss": neg_val_log_like, "average_loss": average_loss})




# We thank David Rügamer for his help in implementing the following custom implementation of simple SDDR models
# in cases where the official implementation gave implausible results:


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(16, 16, 16), dropout: float = 0.0):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class HeteroscedasticGaussian(nn.Module):
    # Normal( MLP1(x) + x^T beta1, softplus(MLP2(x) + x^T beta2) )
    def __init__(
        self,
        in_dim: int,
        hidden_dims_mean=(16, 16, 16),
        hidden_dims_scale=(16, 16, 16),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp_mean = MLP(in_dim, hidden_dims=hidden_dims_mean, dropout=dropout)
        self.lin_mean = nn.Linear(in_dim, 1, bias=False)
        self.mlp_scale = MLP(in_dim, hidden_dims=hidden_dims_scale, dropout=dropout)
        self.lin_scale = nn.Linear(in_dim, 1, bias=False)

    def forward(self, x: torch.Tensor):
        mu = self.mlp_mean(x) + self.lin_mean(x)
        raw = self.mlp_scale(x) + self.lin_scale(x)
        sigma = F.softplus(raw) + 1e-6
        return mu, sigma
    
    def gaussian_nll(self, y: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        y, mu, sigma = y.reshape(-1), mu.reshape(-1), sigma.reshape(-1)
        eps = 1e-8
        sigma = torch.clamp(sigma, min=eps)
        return (0.5 * math.log(2.0 * math.pi) + torch.log(sigma) + 0.5 * ((y - mu) / sigma) ** 2).mean()
    
    def log_like_obs(self, y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        y, mu, sigma = y.reshape(-1), mu.reshape(-1), sigma.reshape(-1)
        sigma = np.maximum(sigma, 1e-8)
        ll = -0.5 * np.log(2.0 * np.pi) - np.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2
        return ll
    
    @torch.no_grad()
    def mean_loglik(self, y_np: np.ndarray, mu_np: np.ndarray, sigma_np: np.ndarray) -> float:
        y_np, mu_np, sigma_np = y_np.reshape(-1), mu_np.reshape(-1), sigma_np.reshape(-1)
        sigma_np = np.maximum(sigma_np, 1e-8)
        ll = -0.5 * np.log(2.0 * np.pi) - np.log(sigma_np) - 0.5 * ((y_np - mu_np) / sigma_np) ** 2
        return float(np.mean(ll))
    
class BernoulliSDDR(nn.Module):
    # Bernoulli( sigmoid( MLP1(x) + x^T beta))
    def __init__(
        self,
        in_dim: int,
        hidden_dims=(16, 16, 16),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dims=hidden_dims, dropout=dropout)
        self.lin = nn.Linear(in_dim, 1, bias=False)
    
    def forward(self, x: torch.Tensor):
        logits = self.mlp(x) + self.lin(x)
        probs = torch.sigmoid(logits)
        return probs
    
    def bernoulli_nll(self, y: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        y, probs = y.reshape(-1), probs.reshape(-1)
        eps = 1e-8
        probs = torch.clamp(probs, min=eps, max=1 - eps)
        return -(y * torch.log(probs) + (1 - y) * torch.log(1 - probs)).mean()
    
    def log_like_obs(self, y: np.ndarray, probs: np.ndarray) -> np.ndarray:
        y, probs = y.reshape(-1), probs.reshape(-1)
        eps = 1e-8
        probs = np.clip(probs, eps, 1 - eps)
        ll = y * np.log(probs) + (1 - y) * np.log(1 - probs)
        return ll
    

def train_custom_sddr(config, data, other_specifications=None):
    """
    Trains a custom SDDR model on the given data.
    
    config contains the following keys:
    "index"  -> tuple of the form (train_index, test_index)
    "lr" -> float:  learning rate
    "epochs" -> integer: number of epochs
    "width" -> int, width of each layer of the NN but the first and last
    "depth" -> number of hidden layers not counting the first and last layer of each block
    "num_batches" -> int, number of batches to use during training
    data contains the following keys:
    "target" -> np.array
    "covariates" -> np.array
    """
    early_stopping = True if "early_stopping" not in config else config["early_stopping"]
    patience = 50 if "patience" not in config else config["patience"]
    report_every_n_epochs = 30 if "report_every_n_epochs" not in config else config["report_every_n_epochs"]
    covariates = torch.tensor(data["covariates"], dtype=torch.float32)
    target = torch.tensor(data["target"], dtype=torch.float32)
    train_index, test_index = data["index"]
    num_params = data["covariates"][train_index,].shape[1]
    distribution = config["family"] if "family" in config else "gaussian"


    if distribution == "gaussian":
        sddr_model = HeteroscedasticGaussian(
            in_dim=num_params,
            hidden_dims_mean=tuple([config["width"]]*config["depth"]),
            hidden_dims_scale=tuple([config["width_2"]]*config["depth_2"]),
            dropout=config["dropout_rate"] if "dropout_rate" in config else 0.2,
        )
        loss_criterion = sddr_model.gaussian_nll
    elif distribution == "bernoulli":
        sddr_model = BernoulliSDDR(
            in_dim=num_params,
            hidden_dims=tuple([config["width"]]*config["depth"]),
            dropout=config["dropout_rate"] if "dropout_rate" in config else 0.2,
        )
        loss_criterion = sddr_model.bernoulli_nll
    else:
        raise ValueError("Unsupported distribution for custom SDDR model.")

    optimizer = torch.optim.Adam(sddr_model.parameters(), lr=config["lr"])

    num_epochs = config["epochs"]
    train_covariates_tensor = covariates[train_index,]
    train_target_tensor = target[train_index].reshape(-1)
    test_covariates_tensor = covariates[test_index,]
    test_target_tensor = target[test_index].reshape(-1)
    data_loader = torch.utils.data.DataLoader(list(zip(train_target_tensor, train_covariates_tensor)), batch_size = 512 if "num_batches" not in config else len(train_index)//config["num_batches"],
                                                  shuffle=True)
    average_loss_hist = np.array([])
    best_val_loss = np.inf
    current_patience = 0

    for epoch in range(num_epochs):
        sddr_model.train()
        batchwise_loss = np.array([])
        for batch_id, (batch_y, batch_X) in enumerate(data_loader):
            optimizer.zero_grad()
            if distribution == "gaussian":
                mu, sigma = sddr_model(batch_X)
                sigma = sigma.clamp(min=1e-8)
                loss = loss_criterion(batch_y, mu, sigma)
            elif distribution == "bernoulli":
                probs = sddr_model(batch_X)
                loss = loss_criterion(batch_y, probs)
            
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
            else:
                print(" encountered loss zero or infinity")
                sddr_model.load_state_dict(best_model_state)
                break
            batchwise_loss = np.append(batchwise_loss, loss.detach().cpu().numpy())
        average_loss = batchwise_loss.flatten().mean()
        average_loss_hist = np.append(average_loss_hist, average_loss)
        with torch.no_grad():
            sddr_model.eval()
            if distribution == "gaussian":
                val_mu, val_sigma = sddr_model(test_covariates_tensor)
                val_prediction = val_mu.reshape(-1).detach().cpu().numpy()

                neg_val_log_like = sddr_model.gaussian_nll(test_target_tensor, val_mu, val_sigma).detach().cpu().numpy().item()
            elif distribution == "bernoulli":
                val_probs = sddr_model(test_covariates_tensor)
                val_prediction = np.round(val_probs.reshape(-1).detach().cpu().numpy())
                neg_val_log_like = sddr_model.bernoulli_nll(test_target_tensor, val_probs).detach().cpu().numpy().item()
            val_log_like = -neg_val_log_like
            if neg_val_log_like < best_val_loss:
                best_val_loss = neg_val_log_like
                current_patience = 0
                best_model_state = copy.deepcopy(sddr_model.state_dict())
            else:
                current_patience += 1
        if early_stopping and current_patience >= patience:
            sddr_model.load_state_dict(best_model_state)
            break
        if epoch%report_every_n_epochs==0 or epoch == num_epochs-1:
                #save checkpoint in temporary file; will be permanent after being reported
                with tempfile.TemporaryDirectory() as tempdir:
                    torch.save({"epoch":epoch, "model_state": sddr_model.state_dict(), "working_dir": os.getcwd()},
                               os.path.join(tempdir, "checkpoint.pt"))
                    checkpoint = Checkpoint.from_directory(tempdir)
                    #print("Current validation predictions:", val_prediction)
                    LidglmMetrics.numpy_based_metrics(val_prediction, test_target_tensor.detach().cpu().numpy(), model_string="custom_sddr", metrics_list = list(other_specifications["metrics"].keys()),
                                      log_like=val_log_like, checkpoint = checkpoint, other_computed_metrics={"validation_loss": neg_val_log_like, "average_loss": average_loss})
        #else:
        #    LidglmMetrics.numpy_based_metrics(val_prediction, test_target_tensor.detach().cpu().numpy(), model_string="custom_sddr", metrics_list = list(other_specifications["metrics"].keys()),
        #                              log_like=val_log_like, checkpoint = None, other_computed_metrics={"validation_loss": neg_val_log_like, "average_loss": average_loss})

    with tempfile.TemporaryDirectory() as tempdir:
                    torch.save({"epoch":epoch, "model_state": sddr_model.state_dict(), "working_dir": os.getcwd()},
                               os.path.join(tempdir, "checkpoint.pt"))
                    checkpoint = Checkpoint.from_directory(tempdir)
                    LidglmMetrics.numpy_based_metrics(val_prediction, test_target_tensor.detach().cpu().numpy(), model_string="custom_sddr", metrics_list = list(other_specifications["metrics"].keys()),
                                      log_like=val_log_like, checkpoint = checkpoint, other_computed_metrics={"validation_loss": neg_val_log_like, "average_loss": average_loss})
    
