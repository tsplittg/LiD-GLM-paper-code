from matplotlib import pyplot as plt
from lidglm.extended_glm.extended_glm import ExtendedGLM
from lidglm.torch_glm.families import TorchGaussian
import torch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator


class sklearn_pdp_adapter(BaseEstimator):
    """Adapter class to use the extended GLM with sklearn's PDP functions.
    This is needed because sklearn's PDP functions expect a sklearn-like estimator.
    """
    def __init__(self, extended_glm, pdp_type="prediction", _estimator_type = "regressor"):
        """
        Parameters
        ----------
        extended_glm : ExtendedGLM
            The extended GLM to use for predictions.
        pdp_type : str, optional
            The type of PDP to compute. The default is "prediction".
        """
        self.extended_glm = extended_glm
        self.pdp_type = pdp_type
        self.linear_weights_ = extended_glm.read_linear_weights()
        self._estimator_type = _estimator_type
        self.fit()

    def predict(self, X):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32, device=self.extended_glm.used_device)
        if self.pdp_type == "prediction":
            return self.extended_glm.predict_endog(X_tensor).reshape(-1).detach().cpu().numpy()

    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
        
class sklearn_classifier_pdp_adapter(BaseEstimator):

    def __init__(self, extended_glm, pdp_type="prediction"):
        """
        Parameters
        ----------
        extended_glm : ExtendedGLM
            The extended GLM to use for predictions.
        pdp_type : str, optional
            The type of PDP to compute. The default is "prediction".
        """
        self.extended_glm = extended_glm
        self.pdp_type = pdp_type
        self.linear_weights_ = extended_glm.read_linear_weights()
        self._estimator_type = "classifier"
        self.classes_ = np.array([0,1])
        self.fit()
    
    def predict(self, X):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32, device=self.extended_glm.used_device)
        if self.pdp_type == "prediction":
            return self.extended_glm.predict(X_tensor)[:,1].reshape(-1).detach().cpu().numpy()
    
    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32, device=self.extended_glm.used_device)
        probs = self.extended_glm.predict(X_tensor).detach().cpu().numpy()
        return np.vstack((1 - probs, probs)).T
    


class sklearn_statsmodels_pdp_adapter(BaseEstimator):
    """Adapter class to use the traditional GLM with sklearn's PDP functions.
    This is needed because sklearn's PDP functions expect a sklearn-like estimator.
    """
    def __init__(self, sm_glm):
        """
        Parameters
        ----------
        sm_glm : sm.GLM
            The statsmodels GLM to use for predictions.
        pdp_type : str, optional
            The type of PDP to compute. The default is "prediction".
        """
        self.sm_glm = sm_glm
        self._estimator_type = "regressor"
        self.fit()

    def predict(self, X):
        return self.sm_glm.predict(X)
    
    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self

class Extended_glm_eval_utils():

    def custom_PHO(self, extended_glm: ExtendedGLM, exog: torch.Tensor):
        """
        Applies post-hoc orthogonalization to the first transformation in the extended GLM.
        Returns a k+1-dimensional vector containing the orthogonalized bias and coefficients and an (n x k) -dimensional tensor
        containing the orthogonalized nu_1 output computed on the training data exog.
        """
        exog = exog.to(extended_glm.used_device)
        # add constant first column to the exog tensor:
        exog_B = torch.cat((torch.ones((exog.shape[0], 1), device=extended_glm.used_device), exog), dim=1)

        # get output of nu_1:
        nu_1_output = extended_glm.NN_transform(exog)-exog

        #read the beta vector from the extended GLM:
        coefficients = extended_glm.read_linear_weights()
        bias = extended_glm.read_linear_bias()
        beta_vector = np.append(bias.reshape(-1), coefficients)

        # compute Gamma-matrix; optimal parameters of multivariate regression:
        Gamma = ((exog_B.T @ exog_B).inverse() @ (exog_B.T @ nu_1_output)).detach().cpu().numpy()

        # split Gamma into first row and Gamma_minus_B
        Gamma_zero = Gamma[0, :]
        Gamma_minus_B = Gamma[1:, :]

        #compute the adjusted bias beta_zero_tilde:
        beta_zero_tilde = bias + Gamma_zero @ coefficients

        # compute the adjusted coefficient vector; beta_tilde 
        beta_tilde = (np.eye(len(coefficients)) + Gamma_minus_B) @ coefficients

        # we return the betas as one vector with the bias as the first element:
        beta_tilde_return_vector = np.concatenate([beta_zero_tilde, beta_tilde])

        # the orthogonalized nu_1 is returned in the form of a lambda function
        # compute the normalization diagonal matrix:
        D_matrix = np.diag(beta_vector[1:] / beta_tilde_return_vector[1:])
        orthog_nu_1 = lambda X_tensor: ((extended_glm.NN_transform(X_tensor.to(extended_glm.used_device))-X_tensor.to(extended_glm.used_device)).detach().cpu().numpy()   
                                         -(torch.cat((torch.ones((X_tensor.shape[0], 1), device="cpu"), X_tensor.to("cpu")), dim=1).detach().cpu().numpy() @ Gamma)) @ D_matrix

        return beta_tilde_return_vector, orthog_nu_1, Gamma




    def one_dimensional_R_squared(self, extended_glm, exog: torch.Tensor, pho: bool = True, features: list[int]|None = None):
        """
        Computes a version of the R^2 to examine how much the original covariate x_i determines the state of T_p(x_i).
        Parameters
        ----------
        extended_glm : ExtendedGLM
            The extended GLM to evaluate.
        exog : torch.Tensor
            The covariates to use for the R^2 computation.
        pho : bool, optional
            Whether to apply post-hoc orthogonalization before computing the R^2. The default is True.
        features : list[int]|None, optional
            The indices (starting from 1) of features to compute the R^2 for. If None, all features are used. The default is None.
        """
        if features is None:
            features = list(range(exog.shape[1]))
        if pho:
            _, nu1_orthog, _ = self.custom_PHO(extended_glm, exog)
            exog_transformed = torch.tensor(nu1_orthog(exog), dtype=exog.dtype, device=exog.device) + exog
        else:
            exog_transformed = extended_glm.NN_transform(exog)
        
        exog_transf_mean = exog_transformed.mean(dim=0, keepdim=True)
        
        R_squared = (torch.pow(exog- exog_transf_mean, 2).sum(dim=0) /(torch.pow(exog_transformed - exog_transf_mean, 2).sum(dim=0))).detach().cpu().numpy()
        r2_dict = {i+1: R_squared[i] for i in features}
        return r2_dict

    def coefficient_table(self, extended_glm, exog, coefficient_names: list[str]|None = None, number_decimals: int = 2, pnorm = 2,
                          original_glm = None) -> pd.DataFrame:
        """
        Helper function that produces a coefficient table in the style of traditional (G)LM. It includes the following columns:
        - the coefficients (including the bias) of the LM that the model is based on
        - the linear coefficients (including the bias) of the LiD-GLM after training
        - the linear coefficients (including the bias) of the LiD-GLM after training with PHO applied
        - the one-dimensional R-squared described in the paper
        """
        if extended_glm.original_weights is None or extended_glm.original_bias is None:
            if original_glm is None:
                raise ValueError("LiD-GLM object does not contain original weights/bias; perhaps loaded from state dict?")
            else:
                original_linear_coeffs = original_glm.params
        else:
            original_linear_coeffs = np.append(extended_glm.original_bias.reshape(-1), extended_glm.original_weights)
        trained_linear_coeffs = np.append(extended_glm.read_linear_bias().reshape(-1), extended_glm.read_linear_weights())
        pho_linear_coeffs, _, _ = self.custom_PHO(extended_glm, exog)
        one_dim_r2 = self.one_dimensional_R_squared(extended_glm, exog, pho=True)
        # the last 3 only have len(num_weights) as the values don't make sense for the intercept, so we need to pas them with NaN.
        # We also convert them to arrays in the right order:
        one_dim_r2 = np.array([np.nan] + [one_dim_r2[i] for i in range(1, exog.shape[1]+1)])

        coeff_names = coefficient_names if coefficient_names is not None else ["Intercept"] + [f"Covariate beta_{i}" for i in range(1, exog.shape[1]+1)]
        if len(coeff_names) != len(original_linear_coeffs):
            raise ValueError("Length of coefficient_names does not match number of coefficients. Did you forget the intercept?")
        coeff_table = pd.DataFrame(np.round(np.array([original_linear_coeffs, trained_linear_coeffs, pho_linear_coeffs, one_dim_r2]).T, number_decimals),
                                   columns = ["LM", "LiD-GLM", "LiD-GLM with PHO", "1-dim R-sq."],
                                   index = coeff_names)
        return coeff_table


    def plot_nu_2(self, extended_glm: ExtendedGLM, exog: torch.Tensor = None,endog:torch.Tensor = None, direction = "forwards", range: tuple = None, num_plot_points: int = 1000, fig = None, ax = None, 
                  label = "T_d", plot_identity: bool = False, legend: bool = True):
        """Plots nu_2 as a function \mathbb{R}\to\mathbb{R}. Either in "forwards" or "inverse" direction.
        Either exog and endog data or a range tuple have to specified to specify a range of plausible z-values."""

        if exog is None and endog is None and range is None:
            raise ValueError("Either exog/endog or range have to be specified.")
        elif (exog is not None or endog is not None) and range is not None:
            raise ValueError("Either exog/endog or range have to be specified, not both.")
        elif exog is not None and endog is not None:
            if direction == "forwards":
                predicted_mu = extended_glm.predict(exog).reshape(endog.shape)
            elif direction == "inverse":
                predicted_mu = extended_glm.predict_endog(exog).reshape(endog.shape)
            
            if isinstance(extended_glm.glm.family, TorchGaussian):
                residuals = endog - predicted_mu
                mins = torch.min(residuals, dim=0)[0]
                maxs = torch.max(residuals, dim=0)[0]
                z_range = (mins, maxs)
            else:
                raise ValueError("Automatic z value range generation not implemented for non-Gaussian families. Please specify range manually.")
        else:
            z_range = range

        plot_points = torch.arange(start=float(z_range[0]), end=float(z_range[1]), step=float((z_range[1]-z_range[0])/num_plot_points)).reshape(-1,1)
        if direction == "forwards":
            nu_2_values = extended_glm.net_transform_endog(plot_points)[0].detach().numpy()
            title = r"$T_d$ as a function $\mathbb{R}\to\mathbb{R}$"
        elif direction == "inverse":
            nu_2_values = extended_glm.nu_2_inverse_approx(plot_points).detach().numpy()
            title = r"Inverse of $T_d$ as a function of $z$"
        plot_points = plot_points.detach().numpy()
        if fig is None and ax is None:
            fig, ax = plt.subplots()
        ax.plot(plot_points, nu_2_values, label=label, color="red")
        ax.set_title(title, fontsize = 15)
        if plot_identity:
            ax.plot(plot_points, plot_points, linestyle=":", color="blue", label="Identity")
        if legend:
            ax.legend(fontsize = 13, loc = "lower right")
        ax.set_xlabel("$v$", fontsize = 15)
        ax.set_ylabel("$T_d(v)$" if direction == "forwards" else "$T_d^{-1}(z)$", fontsize = 15)
        return fig, ax

            

        

    
    
