"""
The distribution families following a one parameter exponential family distribution
that can be used in a TorchGLM.
Created from scratch but roughly follows the code structure of the statsmodels
GLM implementation (https://github.com/statsmodels/statsmodels).
"""

import torch
import numpy as np
import lidglm.torch_glm.links as L, lidglm.torch_glm.varfctns as V


class TorchFamily:
    """
    Abstract parent class for all one parameter exponential family distributions that
    can be used in a TorchGLM.
    
    Parameters
    ----------
    link : instance of TorchGLM.links.TorchLink or a subclass
        The link function transforming the mean into the linear predictor. 
        For more details see for ex. McCullagh pg. 31 ff.
        Each family has a class attribute specifying all available links for that class.
    var_fctn : instance of TorchVarianceFunction or a subclass
        Expresses the functional dependence of the variance of the distribution on
        its mean. See also McCullagh pg. 30.
        See the documentation in TorchGlm.links for each individual function and see the
        docstring of each family for their respective variance functions.
    Notes
    ---------
    During initialization, one of the parameters needs to be an instance of a link function
    that will be bound to the instance of the family. 
    It would also be possible to implement this via strings for example, but it would
    bloat the code with if-cases and in practice the user will mostly have no direct interaction 
    with TorchFamily.__init__()
    """

    links: list[L.TorchLink] = []

    
    
    def __init__(self, link: L.TorchLink, var_fctn: V.TorchVarianceFunction, check_link: bool = True):
        """

        Parameters
        ----------
        link : L.TorchLink
            For details see class docstring
        var_fctn : V.TorchVarianceFunction
            For details see class docstring
        check_link : bool, optional
            If True (default), it is checked during initialization that the specified link is 
            valid for the chosen family.
            If False, this check is skipped and a (possibly inappropriate) link can be freely chosen
            from TorchGLM.links.

        """
        
        if not isinstance(link, L.TorchLink):
            msg = f"""link needs to be an instance of TorchGLM.links.TorchLink, 
                got object of type "{type(link)}" instead."""
            raise TypeError(msg)
        if check_link:
            link_valid = max([isinstance(link, possible_link) for possible_link in self.links])
            if not link_valid:
                raise ValueError(f"""Link not appropriate for chosen family; should be one of
                                 {self.links},
                                 got type "{type(link)}" instead.
                                 """)
        if not isinstance(var_fctn, V.TorchVarianceFunction):
            msg = f"""variance function needs to be an instance of TorchGLM.varfctns.TorchVarianceFunction, 
                got object of type "{type(var_fctn)}" instead."""
            raise TypeError(msg)
            
        self.link = link
        self.var_fctn = var_fctn

    def dispersion(self, endog: torch.Tensor, mu: torch.Tensor) -> float:
        """
        Estimate/get the dispersion parameter of the exponential family
        """

        raise NotImplementedError
    
    def estimate_variance(self, mu: torch.Tensor, endog: torch.Tensor=None, dispersion: float = None) -> float:
        """
        Compute/estimate the variance of the distribution given a vector of estimated means and observed values.
        
        The Formula for a GLM is:

        Var(Y) = Var_fctn(\mu) * dispersion
        """
        if dispersion is None:
            used_dispersion = self.dispersion(endog, mu)
        else:
            used_dispersion = dispersion
        return self.var_fctn(mu) * used_dispersion
        

    def compute_mean(self, lin_pred: torch.Tensor) -> torch.Tensor:
        """
        Computes the mean values for the dependent variable based on each linear 
        predictors given in lin_pred.

        Parameters
        ----------
        lin_pred : torch.Tensor
            Tensor of linear predictors

        Returns
        -------
        mu: torch.Tensor
            The inverse link function applied to each linear predictor; i.e. 
            the predicted means
        """
        
        return self.link.inverse(lin_pred)
    
    def compute_lin_pred(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Computes the values of the linear predictors for each mean given in mu.

        Parameters
        ----------
        mu: torch.Tensor
            Means or values of the dependent variable
        
        Returns
        -------
        lin_pred : torch.Tensor
            The link function applied to the means, i.e. tensor of linear predictors
        """
        
        return self.link(mu)
    
    def log_like_obs(self, endog: torch.Tensor, mu: torch.Tensor, dispersion: float = 1) -> torch.Tensor:
        """
        Computes a tensor containing the log-likelihood for each observation for 
        specific distribution and using mu[i] as the mean for the computation of
        log_like[i].

        Parameters
        ----------
        endog : torch.Tensor
            Tensor of values of the dependent variable.
        mu : torch.Tensor
            Tensor of mean values to set each distribution.

        Returns
        -------
        log_likes : torch.Tensor
            Tensor containing the log-likelihood evaluated componentwise at
            endog[i],mu[i]
        """
        
        raise NotImplementedError
    
    def cdf(self, endog: torch.Tensor, mu: torch.Tensor):
        """
        Compute the cumulative distribution function at each value in endog with the distribution having parameter mu. 
        
        endog and mu must be tensors of the same length.
        """

        raise NotImplementedError

    
    def log_like(self, endog: torch.Tensor, mu: torch.Tensor, dispersion: float = 1) -> torch.Tensor:
        """
        Parameters
        ----------
        endog : torch.Tensor
            Tensor of values of the dependent variable.
        mu : torch.Tensor
            Tensor of mean values to set each distribution.

        Returns
        -------
        log_likes : torch.Tensor
            Tensor of length 1 containing the log-likelihood of the value-tensor
            endog as a sum of the individual log-likelihoods (assumed independence
                                                              of observaions).
        """

        return torch.sum(self.log_like_obs(endog=endog, mu=mu, dispersion=dispersion))
    
    def core_log_like_obs(self, endog: torch.Tensor, mu: torch.Tensor, dispersion: float = 1) -> torch.Tensor:
        """
        Computes a tensor containing the log-likelihood for each observation for 
        specific distribution and using mu[i] as the mean for the computation of
        log_like[i]. This function only includes the parts of the loglikelihhod 
        relevant for training and irrelevant constants have been omitted to 
        reduce unnecessary computational load during training.
        If it isn't implemented for a given family, the regular log-likelihood 
        is computed instead.

        Parameters
        ----------
        endog : torch.Tensor
            Tensor of values of the dependent variable.
        mu : torch.Tensor
            Tensor of mean values to set each distribution.

        Returns
        -------
        log_likes : torch.Tensor
            Tensor containing the core log-likelihood evaluated componentwise at
            endog[i],mu[i]
        """
        
        return self.log_like_obs(endog=endog, mu=mu, dispersion=dispersion)
    
    def core_log_like(self, endog: torch.Tensor, mu: torch.Tensor, dispersion: float = 1) -> torch.Tensor:
        """
        

        Parameters
        ----------
        endog : torch.Tensor
            Tensor of values of the dependent variable.
        mu : torch.Tensor
            Tensor of mean values to set each distribution.

        Returns
        -------
        log_likes : torch.Tensor
            Tensor of length 1 containing the log-likelihood of the value-tensor
            endog as a sum of the individual log-likelihoods (assumed independence
                                                              of observaions).
        """

        return torch.sum(self.core_log_like_obs(endog=endog, mu=mu, dispersion=dispersion))
        
    
class TorchGaussian(TorchFamily):
    """
    Implementation of a Gaussian distribution family as a special case of an
    exponential family.
    
    Attributes
    ----------
    TorchGaussian.link : instance of TorchGLM.links.TorchLink or a subclass; default: identity
        The link function transforming the mean into the linear predictor. 
        For more details see for ex. McCullagh pg. 31 ff.
        The default link function for the Gaussian family is the identity function.
    TorchGaussian.variance : instance of TorchVarianceFunction or a subclass
        Expresses the functional dependence of the variance of the distribution on
        its mean. See also McCullagh pg. 30.
        For Gaussian distribution: constant variance function
    """
     
    links = [L.TorchIdentity, L.TorchLog, L.TorchInversePower]
    variance = V.constant_ones

    def dispersion(self, endog: torch.Tensor, mu: torch.Tensor) -> float:
        """
        Estimate/get the dispersion parameter of the exponential family.

        For the Gaussian distribution this corresponds to the estimated variance
        """
        return torch.sum(torch.pow(endog-mu, 2)) / (len(endog)-1)
    
    def __init__(self, link: L.TorchLink = None, check_link:bool = True):
        """
        Parameters
        ----------
        link : L.TorchLink or a subclass; optional (default: identity)
            For details see class docstring
        check_link : bool, optional
            If True (default), it is checked during initialization that the specified link is 
            valid for the chosen family.
            If False, this check is skipped and a (possibly inappropriate) link can be freely chosen
            from TorchGLM.links.
        """
        
        link = L.Identity if link is None else link
        super(TorchGaussian, self).__init__(
            link = link, 
            var_fctn = TorchGaussian.variance, 
            check_link = check_link
            )
        
    def log_like_obs(self, endog: torch.Tensor, mu: torch.Tensor, dispersion: float) -> torch.Tensor:
        """
        Computes a tensor containing the log-likelihood for each observation for 
        the Normal distribution using mu[i] as the mean for the computation of
        log_like[i].

        Parameters
        ----------
        endog : torch.Tensor
            Tensor of values of the dependent variable.
        mu : torch.Tensor
            Tensor of mean values to set each distribution.
        scale : float
            (estimated) variance

        Returns
        -------
        log_likes : torch.Tensor
            Tensor containing the log-likelihood evaluated componentwise at
            endog[i],mu[i]
            
        Notes:
        -------
        """
        estimated_variance = self.estimate_variance(mu, endog, dispersion)
        log_like = torch.log(2*np.pi*estimated_variance) + (torch.pow(endog - mu, 2)/estimated_variance)
        log_like /= -2
        return log_like

    def core_log_like_obs(self, endog, mu, dispersion):
        """
        For fixed sigma^2 the maximization of the log-likelihood essentially 
        corresponds to the minimization of the MSE (McCullagh pg. 70)

        Parameters
        ----------
        endog : TYPE
            DESCRIPTION.
        mu : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        return -torch.nn.MSELoss(reduction ="none")(endog, mu)

    def cdf(self, endog: torch.Tensor, mu: torch.Tensor):
        """
        Compute the cumulative distribution function at each value in endog with the distribution having parameter mu. 
        
        endog and mu must be tensors of the same length.
        """


        endog = endog.reshape(-1)
        mu = mu.reshape(-1)
        estimated_variance = self.estimate_variance(mu, endog)

        return torch.distributions.Normal(loc = mu, scale = torch.sqrt(estimated_variance)).cdf(endog)
    
    def get_distribution(self, mu: torch.Tensor, dispersion: float = 1) -> torch.distributions.Distribution:
        """
        Returns a torch.distributions.Distribution object for the Gaussian distribution with mean mu and scale parameter scale.
        """
        variance = self.estimate_variance(mu, dispersion= dispersion)
        return torch.distributions.Normal(loc = mu, scale = torch.sqrt(variance))
    
    
class TorchBinomial(TorchFamily):
    """
    Implementation of a Binomial distribution family as a special case of an
    exponential family.
    
    Attributes
    ----------
    TorchBinomial.link : instance of TorchGLM.links.TorchLink or a subclass; default: TorchLogit
        The link function transforming the mean into the linear predictor. 
        For more details see for ex. McCullagh pg. 31 ff.
        The default link function for the Binomial family is the Logit function.
    TorchBinomial.variance : instance of TorchVarianceFunction or a subclass
        Expresses the functional dependence of the variance of the distribution on
        its mean. See also McCullagh pg. 30.
        For Binomial distribution: TorchBinomial variance function
    """
    
    links=[L.TorchLogit, L.TorchLog]

    def dispersion(self, endog: torch.Tensor, mu: torch.Tensor) -> float:
        """
        For the binomial distribution, the dispersion parameter is 1"""

        return 1
    
    def __init__(self, link: L.TorchLink = None, check_link: bool = True, n: int = 1):
        """
        Parameters
        ----------
        link : L.TorchLink or a subclass; optional (default: TorchLogit)
            For details see class docstring
        check_link : bool, optional
            If True (default), it is checked during initialization that the specified link is 
            valid for the chosen family.
            If False, this check is skipped and a (possibly inappropriate) link can be freely chosen
            from TorchGLM.links.
        n : int (default: 1)
            Parameter n of the binomial distribution
        """
        self.n = n
        link = L.TorchLogit() if link is None else link
        super(TorchBinomial, self).__init__(
            link = link, 
            var_fctn = V.TorchBinomial(self.n), 
            check_link = check_link,
            )
    
    def core_log_like_obs(self, endog: torch.Tensor, mu: torch.Tensor, dispersion: float = 1) -> torch.Tensor:
        """
        Computes a tensor containing the log-likelihood core for each observation for 
        the Binomial distribution using mu[i] as the mean for the computation of
        log_like[i]. This function only includes the parts of the loglikelihhod 
        relevant for training and, like in the source listed below, combinatorial
        constants have been omitted to reduce unnecessary computational load during training.

        Parameters
        ----------
        endog : torch.Tensor
            Tensor of values of the dependent variable.
        mu : torch.Tensor
            Tensor of mean values to set each distribution.

        Returns
        -------
        log_likes : torch.Tensor
            Tensor containing the log-likelihood evaluated componentwise at
            endog[i],mu[i]
            
        Notes:
        -------
        Source: McCullagh pg. 114 f.
        
        Formula:
            log_like_obs_i = y_i log(mu_i/(1-mu_i)) + n log(1-mu_i)
        
        """
        log_like = endog * torch.log( mu / (1-mu+torch.finfo(float).eps) ) + self.n * torch.log(1 - mu)
        return log_like
    
    def log_like_obs(self, endog: torch.Tensor, mu: torch.Tensor, dispersion: float = 1) -> torch.Tensor:
        """
        Computes a tensor containing the log-likelihood for each observation for 
        the Binomial distribution using mu[i] as the mean for the computation of
        log_like[i].

        Parameters
        ----------
        endog : torch.Tensor
            Tensor of values of the dependent variable.
        mu : torch.Tensor
            Tensor of mean values to set each distribution.

        Returns
        -------
        log_likes : torch.Tensor
            Tensor containing the log-likelihood evaluated componentwise at
            endog[i],mu[i]
            
        Notes:
        -------
        
        Formula: (see also https://de.wikipedia.org/wiki/Binomialkoeffizient#Anwendung_f%C3%BCr_algebraisch_darstellbare_F%C3%A4lle) 
            log_like_obs_i = log( Gamma(n+1) / (Gamma(y_i+1) * Gamma(n-y_i+1)) + [y_i log(mu_i/(1-mu_i)) + n log(1-mu_i)]
        
        """
        log_like = self.core_log_like_obs(endog=endog, mu=mu)
        log_like = log_like + torch.lgamma(torch.tensor([self.n +1], device = endog.device)) - torch.lgamma(endog + 1) - torch.lgamma(self.n - endog +1)
        return log_like
    
    def get_distribution(self, mu: torch.Tensor, dispersion: float = 1) -> torch.distributions.Distribution:
        """
        Returns a torch.distributions.Distribution object for the Binomial distribution with mean mu. The dispersion parameter is ignored.
        """
        return torch.distributions.Binomial(total_count=self.n, probs=mu)
    
    
class TorchPoisson(TorchFamily):
    """
    Implementation of a Poisson distribution family as a special case of an
    exponential family.
    
    Attributes
    ----------
    TorchPoisson.link : instance of TorchGLM.links.TorchLink or a subclass; default: TorchLog
        The link function transforming the mean into the linear predictor. 
        For more details see for ex. McCullagh pg. 31 ff.
        The default link function for the Poisson family is the Log function.
    TorchPoisson.variance : instance of TorchVarianceFunction or a subclass
        Expresses the functional dependence of the variance of the distribution on
        its mean. See also McCullagh pg. 30.
        For Poisson distribution: Identity variance function
    """
    
    links=[L.TorchLog, L.TorchIdentity, L.TorchSqrt]
    
    def dispersion(self, endog: torch.Tensor, mu: torch.Tensor) -> float:
        """
        For the Poisson distribution, the dispersion parameter is 1"""

        return 1

    def __init__(self, link: L.TorchLink = None, check_link: bool = True):
        """
        Parameters
        ----------
        link : L.TorchLink or a subclass; optional (default: TorchLog)
            For details see class docstring
        check_link : bool, optional
            If True (default), it is checked during initialization that the specified link is 
            valid for the chosen family.
            If False, this check is skipped and a (possibly inappropriate) link can be freely chosen
            from TorchGLM.links.
        """
        link = L.TorchLog() if link is None else link
        super(TorchPoisson, self).__init__(
            link = link, 
            var_fctn = V.mu_identity,  
            check_link = check_link,
            )
    
    def core_log_like_obs(self, endog: torch.Tensor, mu: torch.Tensor, dispersion: float = 1) -> torch.Tensor:
        """
        Computes a tensor containing the log-likelihood core for each observation for 
        the Binomial distribution using mu[i] as the mean for the computation of
        log_like[i]. This function only includes the parts of the loglikelihhod 
        relevant for training and, like in the source listed below, combinatorial
        constants have been omitted to reduce unnecessary computational load during training.

        Parameters
        ----------
        endog : torch.Tensor
            Tensor of values of the dependent variable.
        mu : torch.Tensor
            Tensor of mean values to set each distribution.

        Returns
        -------
        log_likes : torch.Tensor
            Tensor containing the log-likelihood evaluated componentwise at
            endog[i],mu[i]
            
        Notes:
        -------
        Source: McCullagh pg. 197.
        
        Formula:
            core_log_like_obs_i = y_i * log(mu_i) - mu_i
        
        """

        endog = endog.reshape(-1)
        mu = mu.reshape(-1)
        
        log_like = endog * torch.log( mu ) - mu
        return log_like
    
    def log_like_obs(self, endog: torch.Tensor, mu: torch.Tensor, dispersion: float = 1) -> torch.Tensor:
        """
        Computes a tensor containing the log-likelihood for each observation for 
        the Poisson distribution using mu[i] as the mean for the computation of
        log_like[i].

        Parameters
        ----------
        endog : torch.Tensor
            Tensor of values of the dependent variable.
        mu : torch.Tensor
            Tensor of mean values to set each distribution.

        Returns
        -------
        log_likes : torch.Tensor
            Tensor containing the log-likelihood evaluated componentwise at
            endog[i],mu[i]
            
        Notes:
        -------
        
        Formula: (see also https://de.wikipedia.org/wiki/Poisson-Verteilung#Definition) 
            log_like_obs_i = y_i * log(mu_i) -y_i + log(Gamma(y_i))
        
        """
        
        endog = endog.reshape(-1)
        mu = mu.reshape(-1)

        log_like = self.core_log_like_obs(endog=endog, mu=mu)
        log_like = log_like - torch.lgamma(endog+1)
        return log_like
    
    def get_distribution(self, mu: torch.Tensor, dispersion: float = 1) -> torch.distributions.Distribution:
        """
        Returns a torch.distributions.Distribution object for the Poisson distribution with mean (also called:rate) mu. The dispersion parameter is ignored.
        """
        return torch.distributions.Poisson(rate=mu)