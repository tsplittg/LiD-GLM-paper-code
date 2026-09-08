"""
Variance functions that are utilized by torchGLM.TorchFamily subclasses. 
Each variance function is a function of the canonical parameter theta. It
relates the canonical parameter (which in turn relates to the mean) to the
variance of the distribution. 

More precisely:
    Var(Y)= a(phi) * v(mu) ,
    
where a(phi) is a function of the dispersion parameter (independent of theta or mu)
 and v(mu) is a function of theta or mu respectively.
 
For more details and variance functions for some common distributions, see McCullagh pg. 29 ff.

Unlike in the statsmodels package, all variance functions inherit from the parent class
TorchVarianceFunction. This opens up the possibility to check vor validity in the family class.

Created from scratch but roughly follows the code structure of the statsmodels
GLM implementation (https://github.com/statsmodels/statsmodels).
"""

import torch


class TorchVarianceFunction:
    """
    Parent class for variance functions to be used in torchGLMs. 
    Defaults to constant 1-function.
    
    Notes
    ------
    After a TorchVarianceFunction object ha been initialized, its call method can be used.
    
    The constant 1-variance-function is aliased as "constant_ones".
    """
    
    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the default, constant variance function at mu componentwise.

        Parameters
        ----------
        mu : torch.Tensor
            tensor of means

        Returns
        -------
        v(mu)
            tensor of the variance function evaluated on mu componentwise.
            For the constant variance function just tensor of ones with same
            shape,dtype, device etc. as mu.
        """
        
        return torch.ones_like(mu)
    

constant_ones = TorchVarianceFunction()
constant_ones.__doc__= """
The variance function that's identical to one.

Alias of TorchVarianceFunction()
"""

class TorchPower(TorchVarianceFunction):
    """
    Power variance function of the form v(mu)= mu ^ power, defaults to power=1.
    
    Parameters
    -----------
    power : float
        exponent used
        
        
    Methods
    --------
    call
        Evaluates the variance function on a given mu-tensor componentwise
        
    Notes
    -----
    important special cases/ aliases:
        
    mu_identity = TorchPower()
    mu_squared = TorchPower(2.)
    mu_cubed = TorchPower(3.)
    """
    
    def __init__(self, power: float =1.):
        self.power = power
        
    def __call__(self, mu: torch.Tensor, use_abs: bool =True) -> torch.Tensor:
        """
        Evaluates the power variance function on a given tensor componentwise.

        Parameters
        ----------
        mu : torch.Tensor
            tensor of means
            
        use_abs: bool
            determines whether the absolute value of the mean is used; in statsmodels 
            always the case;

        Returns
        -------
        v(mu) : torch.Tensor
            v(mu)= mu ^ self.power if use_abs=False
            v(mu)= |mu| ^ self.power if use_abs=True
        """
        
        if use_abs:
            return torch.pow(torch.abs(mu), self.power)
        else:
            return torch.pow(mu, self.power)
        
        
mu_identity = TorchPower()
mu_identity.__doc__="""
The absolute-value variance function. 
Alias of TorchPower()
"""

mu_squared = TorchPower(2.)
mu_squared.__doc__="""
Returns mu^2. 
Alias of TorchPower(2.)
"""

mu_cubed = TorchPower(3.)
mu_cubed.__doc__="""
Returns mu^3. 
Alias of TorchPower(3.)
"""


class TorchBinomial(TorchVarianceFunction):
    """
    The variance function for the binomial distribution, see for example McCullagh pg. 30
    
    Parameters
    ----------
    n: int, optional
        number of trials. Defaults to 1, in which case mu should lie in (0,1)
        
    Methods
    -------
    call
        returns the variance function evaluated at a given input
        
    Notes
    ------
    Like in statsmodels, the formula used for the variance function is
    V(mu) = p * (1-p) * n, 
    where p = mu / n.
    
    Special Cases/aliases: 
        bernoulli = TorchBinomial()    
    """

    def __init__(self, n: int = 1):
        self.n = n
        
    def __call__(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Evaulates the binomial variance function for each given mu.

        Parameters
        ----------
        mu : torch.Tensor
            tensor of means.

        Returns
        -------
        variances : torch.Tensor
            tensor filled with variances computed from given means by the formula
            mentioned in the TorchBinomial docstring.
            
        Notes
        ------


        """
        
        p = mu / self.n
        return p * (1-p) * self.n

bernoulli = TorchBinomial()
bernoulli.__doc__="""
The variance function for a bernoulli distribution, i.e. for a binomial distr. with n=1.
Alias of TorchBinomial().
"""