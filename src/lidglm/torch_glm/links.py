"""
Defines the link functions to be used in combination with the families defined
in torch_glm.families.

Created from scratch but roughly follows the code structure of the statsmodels 
GLM implementation (https://github.com/statsmodels/statsmodels).
"""

import torch

_float_eps = torch.finfo(float).eps

class TorchLink:
    """
    Abstract parent class for all (1 parameter) link functions. 
    For more details on link functions see McCullagh pg. 27 ff. and esp. pg. 30
    """
    
    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the link function at mu componentwise; abstract placeholder function!

        Parameters
        ----------
        mu : torch.Tensor
            A pytorch tensor of means.

        Returns
        -------
        g(mu) : torch.Tensor
            The values of the link function evaluated componentwise on mu.
        """
        
        return NotImplementedError
    
    def inverse(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the inverse of the link function at eta componentwise;
        placeholder function !

        Parameters
        ----------
        eta : torch.Tensor
            tensor of inputs; often just a single linear predictor

        Returns
        -------
        g^(-1)(eta) : torch.Tensor
            The values of the inverse of the link function evaluated componentwise on eta.
        """
        
        return NotImplementedError
    

class TorchLogit(TorchLink):
    """
    The logit link. See McCullagh pg. 14.
    
    Notes
    -----
    Following the statsmodels conventions, the functions __call__ and deriv (if implemented)
    project their input to the interval [eps, 1-eps] (i.e. (0,1) excluding borders).
    This prevents division by zero, but should rarely come up.
    The corresponding pytorch function, torch.clamp(), is compatible with backpropagation
    and just sets the gradient at the borders to 1.

    """
    
    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the logit at mu componentwise.

        Parameters
        ----------
        mu : torch.Tensor
            A pytorch tensor of means (or, usually equivalently, probabilities).

        Returns
        -------
        g(mu) : torch.Tensor
            The values of the logit evaluated componentwise on mu.
            
        Notes
        -----
        g(mu)=log( mu / (1-mu) )
        """
        
        return torch.logit(mu, eps=_float_eps)
    
    def inverse(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the inverse of the logit at eta componentwise.

        Parameters
        ----------
        eta : torch.Tensor
            tensor of inputs; often just a single linear predictor

        Returns
        -------
        mu : torch.Tensor
            The values of the inverse of the logit evaluated componentwise on eta.
            
        Notes
        ------
        McCullagh,pg.30:
        g^(-1)(eta) = exp(eta) / (1 + exp(eta)) = 1 / (exp(-eta) + 1)
        """
        
        return 1. / (1. + torch.exp(-eta))
    

class TorchPower(TorchLink):
    """
    The power transform, see McCullagh pg. 31
    
    Parameters
    ----------
    power : float
        The specific expoenent of the power transform, corresponds to lambda in McCullagh pg. 31.
        
    Notes
    ------
    Special cases/Aliases of Power:
        
    Identity = Power(power= 1.)
    Inverse  = Power(power= -1.)
    Sqrt     = Power(power= 0.5)
    
    Unlike in statsmodels, the Identity case is reimplemented with wholly seperate functions; this
    alleviates the need for a corresponding if-statement in every single function.
    It is thus better to use Identity() instead of Power(power=1.)
    """
    
    def __init__(self, power: float = 1.):
        self.power = power
    
    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the power transform at mu componentwise.

        Parameters
        ----------
        mu : torch.Tensor
            A pytorch tensor of means.

        Returns
        -------
        g(mu) : torch.Tensor
            The values of the power transform evaluated componentwise on mu.
            
        Notes
        ------
        g(mu) = mu ^ self.power 
        It isn't checked that the exponential of mu exists or that it is unique!
        Problems could occur for ex. for power = 0.5, mu < 0
        """
        return torch.pow(mu, self.power)
    
    def inverse(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the inverse of the power transform at eta componentwise.

        Parameters
        ----------
        eta : torch.Tensor
            tensor of inputs; often just a single linear predictor

        Returns
        -------
        g^(-1)(eta) : torch.Tensor
            The values of the inverse of the power transform evaluated componentwise on eta.
            
        Notes
        -----
        g^(-1)(eta) = eta ^ (1 / self.power)
        
        It isn't checked that the inverse on eta exists or that it is unique!
        """
        
        return torch.pow(eta, 1./self.power)
    
class TorchIdentity(TorchPower):
    """
    The identity link function; equivalent to the power transform with power 1
    
    Notes
    -----
    g(mu)=mu
    
    Unlike in statsmodels, the Identity case is reimplemented with wholly separate functions; this
    alleviates the need for a corresponding if-statement in every single function.
    It is thus better to use Identity() instead of Power(power=1.).
    The identitiy still inherits from TorchPower however to compensate for not-implemented 
    functionaliyty if it should come up.
    
    """
    
    def __init__(self):
        super().__init__(power =1.)
    
    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the identity at mu componentwise.

        Parameters
        ----------
        mu : torch.Tensor
            A pytorch tensor of means.

        Returns
        -------
        g(mu) : torch.Tensor
            The values of the identity evaluated componentwise on mu.
            
        Notes
        ------
        g(mu) = mu 
        """
        return mu
    
    def inverse(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the inverse of the identity (i.e. the identity) at eta componentwise.

        Parameters
        ----------
        eta : torch.Tensor
            tensor of inputs; often just a single linear predictor

        Returns
        -------
        g^(-1)(eta) : torch.Tensor
            The values of the inverse of the identity evaluated componentwise on eta.
            
        Notes
        -----
        g^(-1)(eta) = eta 
        """
        
        return eta
   

class TorchSqrt(TorchPower):
    """
    The square-root transform; equivalent to the power transform with power 0.5.
    
    Notes
    ------
    g(mu) = sqrt(mu) = mu ^ 0.5
    
    Alias of torchGLM.TorchPower(power=0.5)
    """     
    
    def __init__(self):
        super().__init__(power=0.5)
        
        
class TorchInversePower(TorchPower):
    """
    The inverse transform; equivalent to the power transform with power -1.
    
    Notes
    ------
    g(mu) = 1/mu = mu ^ (-1.)
    
    Alias of torchGLM.TorchPower(power=-1)
    """     
    
    def __init__(self):
        super().__init__(power=-1)
        
        
class TorchLog(TorchLink):
    """
    The logarithm link function.
    
    Notes
    -----
    Following the statsmodels conventions, the functions __call__ and deriv (if implemented)
    project their to the real positive axis (i.e. (0,infty) excluding borders).
    This prevents undefined operations, but should rarely come up.
    The corresponding pytorch function, torch.clamp(), is compatible with backpropagation
    and just sets the gradient at the borders to 1.
    """      
    
    def _clean(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, _float_eps)
    
    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the logarithm on the input tensor mu componentwise.

        Parameters
        ----------
        mu : torch.Tensor
            A pytorch tensor of means.

        Returns
        -------
        g(mu) : torch.Tensor
            The values of the logarithm evaluated componentwise on mu.
            
        Notes
        -----
        g(mu) = log(mu)

        """

        return torch.log(self._clean(mu))
    
    def inverse(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the inverse of the logarithm at eta componentwise.

        Parameters
        ----------
        eta : torch.Tensor
            tensor of inputs; often just a single linear predictor

        Returns
        -------
        mu : torch.Tensor
            The values of the inverse of the logarithm evaluated componentwise on eta.
            
        Notes
        ------
        g^(-1)(eta) = exp(eta)
        """
        
        return torch.exp(eta)
    


