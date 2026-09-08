import numpy as np
import torch
import copy
from lidglm.torch_glm.torch_glm import TorchGLM
from lidglm.torch_glm import families as Fam, links as Lin
from lidglm.extended_glm.transformer.transformer import Transformer, TransformerUtils
from normflows.nets.lipschitz import *


class ExtendedGLM(torch.nn.Module):
    """
    Implementation of LiD-GLM in pytorch. Internally uses TorchGLM for the GLM components.
    """

    def __init__(self, regular_glm: TorchGLM, net_transform: Transformer = None, 
                 net_transform_endog: Transformer = None, orthog: bool =False, enable_cuda: bool = True, dispersion = 1):
        """
        Initialization function for the DeepGLM. The basic components that make up such a DeepGLM have to be provided:
        
        1) A regular GLM in the form of a TorchGLM object
        2) A first neural network (in the form of a TorchGLM.Transformer object) that transforms the input data
        3) A second neural network (in the form of a TorchGLM.Transform or a TorchGLM.ParameterizedTransformer object) that performs distributional correction (OPTIONAL)
        
        More conventient means of initialization have been implemented in the ExtendedGLM_Utils class.

        Parameters
        -----------
        """

        # some pytorch-specific initialization:
        super().__init__()
        self.used_device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
        self.orthog = orthog
        self.to(self.used_device)

        # for later evaluation, we will save the original parameters of the GLM:
        self.original_bias = copy.deepcopy(regular_glm.read_linear_bias())
        self.original_weights = copy.deepcopy(regular_glm.read_linear_weights())

        # transfer relevant parameters from GLM for later use:
        self.glm = regular_glm
        self.num_params = self.glm.num_params
        self.bias = self.glm.bias

        
        if net_transform is None:
            total_lipschitz_const = 1.99
            self.nu_1 = TransformerUtils().initialize_transformer(num_params=self.num_params, lipschitz_const=total_lipschitz_const, parameterized=False, width=5, 
                                                                  depth = np.ceil(np.log(total_lipschitz_const)/np.log(2)).astype(int), domain_codomain=2)
        else:
            self.nu_1 = net_transform

        if net_transform_endog is None:
            self.net_transform_endog = torch.nn.Identity()
        else:
            if not isinstance(self.glm.family, Fam.TorchGaussian):
                raise ValueError("Endog transformation only possible for continuous target distribution.")
            self.net_transform_endog = net_transform_endog
        self.dispersion = dispersion

        #for debugging:
        self.starting_dispersion = dispersion
        

    @classmethod
    def initialize_extended_GLM(cls, num_params:int,
                 family: Fam.TorchFamily = None, family_str: str = None,
                 link: Lin.TorchLink = None, link_str: str = None,
                 bias: bool =True,
                 net_transform: torch.nn.Module = None):
        """

        Alternative constructor for a deep GLM without first having to construct a TorchGLM and a Neural Net by hand.
        """
        glm = TorchGLM(num_params=num_params, family= family,
                             family_str= family_str, link= link, link_str= link_str, bias = bias)

        return cls(glm, net_transform=net_transform)
    
    def compute_dispersion(self, exog, endog):
        """
        Function for estimating the dispersion for a given set of data.
        """
        #compute the estimated (untransformed) target mean:
        temporary_mu = self.predict(exog).reshape(-1)

        #compute the residuals of the on the given data:
        residuals = endog.reshape(-1) - temporary_mu

        #backtransform the residuals in the inverse direction of the second neural network to get normalized residuals:
        temporary_backtransf_residuals = residuals if isinstance(self.net_transform_endog, torch.nn.Identity) else self.net_transform_endog.inverse(residuals.reshape(-1,1))[0].reshape(-1)

        #estimate the dispersion of the GLM contained in the deep GLM:
        estimated_dispersion = self.glm.family.dispersion(endog=temporary_backtransf_residuals + temporary_mu, mu=temporary_mu)
        return estimated_dispersion


    def set_dispersion(self, exog, endog, gradient = True):
        """
        Function that should be called once after training on the full training data (if computationally feasible)
        to estimate the dispersion/variance of the estimated distribution.
        
        For training, it seems helpful to also compute a gradient over the dispersion parameter. After training this might cause issues since dispersion then is not a leaf node in the gradient computation, meaning deepcopy won't work.
        I suppose therefore that after training set_dispersion should be called once with gradient = False; the computation should be done after training on the full training data anyway.
        """

        if isinstance(self.glm.family, Fam.TorchPoisson) or isinstance(self.glm.family, Fam.TorchBinomial):
            dispersion = torch.tensor([1.0], device=self.used_device)
        else:
            dispersion = self.compute_dispersion(exog=exog, endog=endog).reshape(-1,1).to(self.used_device)
        if not gradient:
            dispersion = dispersion.detach()
        self.dispersion = dispersion

    def NN_transform(self, exog):
        """
        Applies the first neural network transformation to the input data. Also applies an othogonolization, following Rügamer et al. (2022), if self.othog is set to True.

        """
        exog = exog.reshape(-1, self.num_params)
        transformed_exog, _ = self.nu_1(exog)


        if(self.orthog):
            #if orthogonalization is to be applied, a corresponding projection matrix is computed following Rügamer et al. (2022)
            Q, _ = torch.linalg.qr(exog)
            projection = torch.eye(exog.shape[0], device= self.used_device) - Q @ Q.T
            transformed_exog = projection @ transformed_exog

        
        return transformed_exog




    def predict(self, exog: torch.Tensor=None):
        """
        Compute predicted values for a given design matrix.

        Parameters
        ----------
        exog : torch.Tensor
            Tensor containing a set of values for the dependet variables.
            Either a one-dimensional tensor of length (num_params) or a
            2 dimensional tensor with dimensions (n)x(num_params)

        Returns
        -------
        pred : torch.Tensor
            A one-dimensional tensor containing the computed predicted means.

        Notes
        ------

        """

        x_transformed = self.NN_transform(exog=exog.reshape(-1, self.num_params))
        pred = self.glm.predict(exog=x_transformed).reshape(-1)
        return pred
    
    def predict_endog(self, exog: torch.Tensor=None):
        """
        Compute predicted values of the target variable for a given design matrix.
        Essentially just outputs the mean, but also applies another neural network if applicable


        """
        exog = exog.reshape(-1, self.num_params)
        mean = self.predict(exog=exog).reshape(-1, 1)


        if isinstance(self.net_transform_endog,torch.nn.Identity):
            return mean.reshape(-1)

        return (self.net_transform_endog(torch.zeros_like(mean))[0]+mean).reshape(-1)
    def linear_predictor(self, exog: torch.Tensor=None):
        """
        Computes what would be the linear predictor for a regular GLM; for a deep-GLM this will usually not be a simple linear
        function. For the Formula see paragraph "Notes".

        Parameters
        ----------
        exog : torch.Tensor
            Tensor containing a set of values for the dependet variables.
            Either a one-dimensional tensor of length (num_params) or a
            2 dimensional tensor with dimensions (n)x(num_params)

        Returns
        -------
        lin_pred : torch.Tensor
            A one-dimensional tensor containing the computed "linear predictors".

        Notes
        ------
        lin_pred = Lin(phi(x)+x)
        with phi being the neural network and Lin being a (learned) linear transformation (possibly including adding a bias)
        """
        exog = exog.reshape(-1, self.num_params)
        x_transformed = self.NN_transform(exog=exog)
        lin_pred = self.glm.linear_predictor(exog=x_transformed).reshape(-1)
        return lin_pred

    #inverse approximation taken very directly from https://github.com/rtqichen/residual-flows/blob/master/resflows/layers/iresblock.py ;
    def inverse_approx(self, z, atol=1e-7, rtol=1e-7):
        return self.nu_1.custom_inverse(z, atol=atol, rtol=rtol)

    def nu_2_inverse_approx(self, z, atol=1e-7, rtol=1e-7):
        
        if isinstance(self.net_transform_endog,torch.nn.Identity):
            return z
        else:
            return self.net_transform_endog.custom_inverse(z, atol=atol, rtol=rtol)
    
    def log_like_untransformed(self, endog: torch.Tensor, exog: torch.Tensor, dispersion: float = None, **kwargs):
        """
        Compute the log-likelihood of the given endogenous variables conditioned on the given exogenous variables.
        In this function the endogenous variables is treated as being distributed according to self.family with a mean as computed by self.predict, i.e. no tranformation is applied after
        the inverse link function.
        """
        if dispersion is None and "compute_dispersion" in kwargs and kwargs["compute_dispersion"]:
            dispersion = self.dispersion if hasattr(self, 'dispersion') else self.compute_dispersion(exog=exog, endog=endog)  
        return self.glm.log_like_obs(exog=self.NN_transform(exog), endog= endog.reshape(-1), in_training=self.training, dispersion=dispersion)
    
    def log_like_obs(self, endog: torch.Tensor, exog: torch.Tensor, dispersion: float = None, evaluation=False):
        """
        Compute the log-likelihood of the given endogenous variables conditioned on the given exogenous variables. 
        The endogenous variables are treated as being a transformation of the distribution learned in log_like_untransformed.

        Parameters
        ----------
        endog : torch.Tensor
            One-dimensional tensor of length (n) containing a set of values for the endogenous variables.

        exog : torch.Tensor
            n x k Tensor containing a set of values for the exogenous variables.

        Returns
        -------
        log_like : torch.Tensor
            A one-dimensional tensor containing the computed log-likelihoods.

        Notes
        ------
        """
        exog= exog.reshape(-1, self.num_params)
        endog = endog.reshape(-1,1)
        if isinstance(self.net_transform_endog,torch.nn.Identity):
            return self.log_like_untransformed(endog, exog, dispersion=dispersion)
        

        mu = self.predict(exog)

        #We no longer transform y itself, but instead the residuals y-\mu
        
        if evaluation:
            backtransf_residual = self.net_transform_endog.custom_inverse((endog.reshape(-1)-self.predict(exog)).reshape(-1,1), atol=1e-8, rtol=1e-8).reshape(-1)
        else:
            backtransf_residual = self.net_transform_endog.inverse((endog.reshape(-1)-self.predict(exog)).reshape(-1,1))[0].reshape(-1)

        if dispersion is None:
            dispersion = self.dispersion if hasattr(self, 'dispersion') else self.compute_dispersion(exog=exog, endog=endog)

        log_p_y = self.log_like_untransformed(backtransf_residual+mu, exog, dispersion=dispersion)
        _, log_p_y_tilde = self.net_transform_endog(backtransf_residual.reshape(-1,1), logpx = log_p_y.reshape(-1,1))
        return log_p_y_tilde.squeeze(1)

    def cdf(self, endog: torch.Tensor, exog: torch.Tensor, training: bool = False):
        """
        Evaluate the approximate cumulative distribution function of the specified target observations conditioned on the given exogenous variables.
        """
        endog = endog.reshape(-1,1).to(self.used_device)
        exog = exog.reshape(-1, self.num_params).to(self.used_device)
        transformed_exog = self.NN_transform(exog)
        if(isinstance(self.net_transform_endog,torch.nn.Identity)):
            inverse = endog
        else:
            if training:
                inverse = self.net_transform_endog.inverse(endog)
            else:
                inverse = self.net_transform_endog.custom_inverse(endog)
        endog = endog.reshape(-1)
        return self.glm.cdf(endog = inverse, exog = transformed_exog)
    
    def get_distribution(self, exog: torch.Tensor, dispersion = None):
        """
        Get a frozen distribution for a given set of exogenous variables.
        """
        if dispersion is None:
            dispersion = self.dispersion
        transformed_exog = self.NN_transform(exog=exog)
        return self.glm.get_distribution(exog = transformed_exog, dispersion= dispersion)
        

    
    def read_lipschitz(self, domain=None, codomain=None, part_str="nu1"):
        """
        Reads out the Lipschitz constant of each layer in the neural networks 
        comprising the neural network.
        Also returns the total Lipschitz constant of each block individually
        (formula: 1+Lip(net)) 
        and the total constant which is the product of the Lipschitz constants 
        of all individual blocks.
        
        Parameters
        ----------
        domain and codomain are optional parameters fur debugging and should be used with care. If none are given, the values ingrained in the network are used.

        part_str : str
            The part of the model for which the Lipschitz constant is to be computed.
            
            
        Returns
        -------
        A list of dictionaries containing the Lipschitz constants for each component requested.
        The dictionaries contain the constants for each linear layer, as well as approximations for each block and the approximate total Lipschitz constant.
        """
        return_list = []
        if part_str == "nu1":
            return_list.append(self.nu_1.read_lipschitz_const(domain=domain, codomain=codomain))
        if part_str == "nu2":
            return_list.append(self.nu_2.read_lipschitz_const(domain=domain, codomain=codomain))
        return return_list
    
    def _compute_norm(self, matrix: np.array, domain: int, codomain: int):
        """
        Compute the operator/matrix-norm of a given matrix as induced by the 
        specified vector Norms.

        Parameters
        ----------
        matrix : np.array
        domain : int
            The domain w.r.t. which the Operator norm is to be computed
        codomain : int
            The codomain w.r.t. which the Operator norm is to be computed.

        Returns
        -------
        norm : int
            The operator norm.
        """
        
        match domain, codomain:
            case 1,1: 
                norm = np.max(np.sum(np.abs(matrix),axis=0))
            case 2,2:
                norm = np.linalg.svd(matrix,compute_uv=False).max()
            case _: raise ValueError("Combination of Domain, Codomain either not recognized or not implemented.")
            
        return norm

    def read_linear_weights(self):
        """
        Return the weights of the linear transformation inside the GLM as a tensor.
        """
        return self.glm.read_linear_weights()
    def read_linear_bias(self):
        """
        Return the weights of the linear transformation inside the GLM as a tensor.
        """
        return self.glm.read_linear_bias()
    
    def get_initialization_params(self):
        """
        Returns the parameters used for the initialization of the extended GLM. Used for saving the extended GLM and dependent models.
        """
        return {
            "glm_params": self.glm.get_initialization_params(),
            "nu_1_params": self.nu_1.get_initialization_params(),
            "nu_2_params": self.net_transform_endog.get_initialization_params() if not isinstance(self.net_transform_endog, torch.nn.Identity) else None,
            "orthog": self.orthog,
            "enable_cuda": torch.cuda.is_available() and self.used_device == 'cuda',
            "dispersion": self.dispersion.item() if isinstance(self.dispersion, torch.Tensor) else self.dispersion
        }
    
    def save_model(self, path: str = "extended_glm"):
        """
        Save the model to two separate files. One contains the parameters of the model and all sub-models, the other contains the state_dict of the model.

        returns a tuple containing the paths to the saved files.
        """
        torch.save(self.get_initialization_params(), f"{path}_params.pth")
        torch.save(self.state_dict(), f"{path}_state_dict.pth")

        return (f"{path}_params.pth", f"{path}_state_dict.pth")



        
        




        

        
        

    
        
