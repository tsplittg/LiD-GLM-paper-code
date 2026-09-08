"""
Core file of the TorchGLM package. Combines a family from TorchGLM.families, a 
link from TorchGLM.links and (implicitly) a variance function from TorchGLM.varfctns
into a single generalized linear model compatible with backpropagation via pytorch.

Created from scratch but roughly follows the code structure of the statsmodels
GLM implementation (https://github.com/statsmodels/statsmodels).
"""

import torch
import numpy as np
import lidglm.torch_glm.families as Fam
import lidglm.torch_glm.links as Lin


class TorchGLM(torch.nn.Module):
    """
    Implementation of a pytorch-compatible GLM.
    
    Parameters
    ----------

    """
    
    def __init__(self, num_params: int,
                 family: Fam.TorchFamily = None, family_str: str = None, 
                 link: Lin.TorchLink = None, link_str: str = None, 
                 bias: bool =True):
        """
        

        Parameters
        ----------
        family : Fam.TorchFamily, optional
            The default is None.
        family_str : str, optional
            The default is None.
        link : Lin.TorchLink, optional
            The default is None.
        link_str : str, optional
            The default is None.
        bias : bool, optional (default: True)
            Whether or not to add a bias to the linear transformation internally.

        Returns
        -------
        None.

        """
        
        super(TorchGLM, self).__init__()
        self._check_input_validity(family=family, family_str=family_str, link=link, link_str=link_str)
        self.num_params = num_params
        self.family, self.link = (
                                    self._preprocess_inputs(family= family, family_str= family_str, link= link, link_str= link_str)
                                    )

        self.bias = bias
        self.linear_transform = torch.nn.Linear(self.num_params, 1, bias = self.bias)

    def _check_input_validity(self,
                 family: Fam.TorchFamily = None, family_str: str = None, 
                 link: Lin.TorchLink = None, link_str: str = None):
        """
        Helper function to run some validity and plausibility checks on the
        input for initialization of the GLM. Some more checks are performed in 
        TorchGLM._preprocess_inputs.
        """
        
        if family is not None:
            if family_str is not None:
                raise ValueError("Arguments 'family' and 'family_str' cannot both be specified")
            if link is not None or link_str is not None:
                raise ValueError("""Input TorchFamily instance already contains a TorchLink,
                                 additional links cannot be specified.
                                 """)
        else:
            if(family_str is None) or (link is None and link_str is None):
                raise ValueError(f"""Either a TorchFamily instance or a family_str and
                                 and a link function (directly or as string) have to be specified.
                                 the following arguments were given:
                                    family: {family}
                                    family_str: {family_str}
                                    link: {link}
                                    link_str: {link_str}
                                 """)
        if link is not None and link_str is not None:
            raise ValueError("Arguments 'link' and 'link_str' cannot both be specified")
        
        
            
    def _preprocess_inputs(self,
                 family: Fam.TorchFamily = None, family_str: str = None, 
                 link: Lin.TorchLink = None, link_str: str = None):
        """
        Helper function to perform various preprocessing tasks on the input during 
        initialization. Also performs some validity checks.
        """
        
        if link_str is not None:
            link_str = link_str.lower().replace(" ", "")
            #Translate the given link_str argument to a links.TorchLink() instance of the appropriate subclass:
            match link_str:
                case "logit":
                    link = Lin.TorchLogit()
                case "power":
                    link = Lin.TorchPower()
                case "inversepower":
                    link = Lin.TorchInversePower()
                case "sqrt":
                    link = Lin.TorchSqrt()
                case "identity":
                    link = Lin.TorchIdentity()
                case "log":
                    link = Lin.TorchLog()
                case _: raise ValueError("Unsupported link_str argument")  
        
        if family_str is not None:
            family_str = family_str.lower().replace(" ", "")
            # Translate the given family_str argument to a families.TorchFamily() instance of the appropriate subclass
            # with the specified link-function:
            match family_str:
                case "gaussian":
                    family = Fam.TorchGaussian(link)
                case "binomial":
                    family = Fam.TorchBinomial(link)
                case _: raise ValueError("Specified family not (yet) supported.")

        return family, link
        
    def linear_predictor(self, exog: torch.Tensor):
        """
        Computes the linear predictor as the vector product between each row in
        exog and the parameter vector.

        Parameters
        ----------
        exog : torch.Tensor
            Tensor containing a set of values for the dependent variables.
            Either a one-dimensional tensor of length (num_params) or a
            2 dimensional tensor with dimensions (num_obs)x(num_params)

        Returns
        -------
        lin_pred : torch.Tensor
            A one-dimensional tensor containing the computed linear predictors.
            
        Notes
        ------

        """
        
        return torch.squeeze(self.linear_transform(exog),dim=1)
    
    def predict(self, exog: torch.Tensor):
        """
        Compute predicted values for a given design matrix.

        Parameters
        ----------
        exog : torch.Tensor
            Tensor containing a set of values for the dependent variables.
            Either a one-dimensional tensor of length (num_params) or a
            2-dimensional tensor with dimensions (num_obs)x(num_params)

        Returns
        -------
        pred : torch.Tensor
            A one-dimensional tensor containing the computed predicted means.

        """
        
        lin_pred = self.linear_predictor(exog=exog)
        return self.family.compute_mean(lin_pred)
        
    def log_like_obs(self, exog: torch.Tensor = None, endog: torch.Tensor = None, 
                 in_training: bool = True, dispersion = None):
        """
        

        Parameters
        ----------
        exog : torch.Tensor, optional
            The default is None.
        endog : torch.Tensor, optional
            The default is None.
        in_training : bool, optional (default: True)
            If the loglikelihood is used for training this parameter should be set 
            as true. This will only return the loglikelihood core (if defined) which speeds
            up computation.

        Returns
        -------
        log_like : torch.Tensor
            A one-dimensional tensor containing the computed log-likelihoods.

        """
        mu = self.predict(exog=exog)

        if in_training:
            return self.family.core_log_like_obs(endog=endog, mu=mu, dispersion=dispersion)
            
        else:
            return self.family.log_like_obs(endog=endog.reshape(-1), mu=mu.reshape(-1), dispersion=dispersion)
            
        
        
    def log_like(self, exog: torch.Tensor = None, endog: torch.Tensor = None, 
                 in_training: bool = True, dispersion = None):
        
        return self.log_like_obs(exog=exog, endog=endog, in_training=in_training, dispersion=dispersion).sum()
    
    def cdf(self, endog: torch.Tensor, exog: torch.Tensor):
        """
        Evaluate the approximate cumulative distribution function of the specified target observations conditioned on the given exogenous variables.
        """

        mu = self.predict(exog=exog)
        return self.family.cdf(endog=endog, mu=mu)
    
    def cdf_given_mu(self, endog: torch.Tensor, mu: torch.Tensor):
        """
        EXPERIMENTAL:

        Evaluate the approximate cumulative distribution function of the specified target observations conditioned on the given exogenous variables.
        """

        return self.family.cdf(endog=endog, mu=mu)

    def read_linear_weights(self):
        """
        Return the weights of the linear transformation as a tensor.
        """
        return self.linear_transform.weight.detach().cpu().numpy().reshape(-1)
    def read_linear_bias(self):
        """
        Return the intercept of the linear transformation as a tensor.
        """
        return self.linear_transform.bias.reshape(-1).detach().cpu().numpy() if self.bias else np.zeros(1)
    
    def get_distribution(self, exog: torch.Tensor, dispersion = None):
        """
        Get a frozen distribution of the GLM for a given set of exogenous variables.
        """
        
        mu = self.predict(exog=exog)
        return self.family.get_distribution(mu=mu, dispersion= dispersion)
    
    def get_initialization_params(self):
        """
        Returns the parameters used for the initialization of the GLM. Used for saving the GLM and dependent models.
        """
        match self.family:
            case Fam.TorchGaussian():
                family_str = "gaussian"
            case Fam.TorchBinomial():
                family_str = "binomial"
            case _:
                raise ValueError("Unsupported family type in TorchGLM.get_initialization_params()")
        match self.family.link:
            case Lin.TorchIdentity():
                link_str = "identity"
            case Lin.TorchLogit():
                link_str = "logit"
            case Lin.TorchInversePower():
                link_str = "inversepower"
            case Lin.TorchSqrt():
                link_str = "sqrt"
            case Lin.TorchLog():
                link_str = "log"
            case Lin.TorchPower():
                link_str = "power"
            case _:
                raise ValueError(f"Unsupported link type in TorchGLM.get_initialization_params(), got {self.link}")

        return {
            "num_params": self.num_params,
            "family_str": family_str,
            "link_str": link_str,
            "bias": self.bias
        }