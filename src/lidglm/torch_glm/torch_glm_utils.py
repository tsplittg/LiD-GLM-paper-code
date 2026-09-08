import statsmodels.api as sm
import statsmodels
import torch
import numpy as np
import warnings
from lidglm.torch_glm.torch_glm import TorchGLM
import lidglm.torch_glm.families as Fam
import lidglm.torch_glm.links as Lin

class TorchGLM_Utils():
    """
    Helper class containing various functions surrounding TorchGlm.
    """

    @classmethod
    def start_with_statsmodels(cls, endog,  exog,
                               statsmodels_family: sm.families.Family = None, family_str: str = None, link_str: str = None, 
                               offset=None, exposure=None, freq_weights=None, var_weights=None,
                               missing='none', fit_max_iter: int = 10 ** 3, bias =True):
        """
        initializes a TorchGLM, given very similar arguments as Statsmodels. 
        Allows the user to either input a statsmodels.family type as statsmodels_family to alternatively determine the used family or to input the family
        and link to be used as string arguments. The string arguments are neihter case- nor whitespace-sensitive.

        The Glm is then fitted with the given data and the results are returned as a TorchGLM object together with converted endog and exog tensors.
        """
        if bias:
            exog = sm.add_constant(exog, prepend = True, has_constant="add")
            bias = "first"
        else: 
            bias = None
        if(statsmodels_family == None and (family_str == None or link_str == None)):
            raise ValueError("Either a sm.families.family or both family and link specifying strings need to be provided.")
        elif(statsmodels_family != None and (family_str != None or link_str != None)):
            raise ValueError("Cannot specify both a sm.families.family to use and family-determining strings.")
        elif(family_str != None and link_str != None):
            #preprocess the input strings:
            family_str = family_str.lower().replace(" ","")
            link_str = link_str.lower().replace(" ", "")

            #Translate the given link_str argument to a sm.families.links.Links() instance of the appropriate subclass:
            match link_str:
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
            match family_str:
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

            statsmodels_family = family

        glm = sm.GLM(endog, exog, statsmodels_family, offset,
                 exposure, freq_weights, var_weights,
                 missing)
        res = glm.fit(max_iter=fit_max_iter)
        return cls.convert_sm_glm(res, bias= "first")

    @classmethod
    def convert_sm_glm(cls, glm: statsmodels.genmod.generalized_linear_model.GLMResults, bias: str = None):
        """
        A method to convert a statsmodels GLM (i.e. a GLMResults object) into a TorchGLM object. The exog and endog datasets saved in the results object are also converted to pytorch tensors 
        and returned.

        Parameters:
        -----------
        bias: str
            string indicating whether the glm contains no bias ("none"), the bias is the first variable ("first") or the last ("last")

        """

        #torch.Linear expects a weight matrix of shape (out_features, in_features) and a bias of shape (out_features)

        params = torch.reshape(torch.tensor(np.array(glm.params), dtype=torch.float),(-1,glm.model.exog.shape[1]))

        train_endog: torch.Tensor = torch.tensor(glm.model.endog, dtype=torch.float)
        train_exog: torch.Tensor = torch.tensor(glm.model.exog, dtype=torch.float)

        #some preprocessing of the input bias string:
        if isinstance(bias, str):
            bias = bias.lower().replace(" ", "")
            use_bias = True
        match bias:
            case "first":
                bias = params[:,0]
                params = params[:,1:]
                train_exog = train_exog[:,1:]
            case "last":
                bias = params[:,-1]
                params = params[:,:-1]
                train_exog = train_exog[:,:-1]
            case None:
                warnings.warn("Torch GLM was not given a bias parameter, will be initialized without bias.")
                use_bias = False
            case "none":
                use_bias = False
            case _:
                raise ValueError("Unsupported bias argument.")


        #Find the right link function to use; order is important since Power is Superclass
        match glm.model.family.link:
            case sm.families.links.Logit():
                link = Lin.TorchLogit()
            case sm.families.links.Identity():
                link = Lin.TorchIdentity()
            case sm.families.links.Power():
                link = Lin.TorchPower()
            case sm.families.links.Log():
                link = Lin.TorchLog()
            case _:
                raise TypeError("Link Function not supported.")

        #Find the right Family to use, initialize with previously determined link function
        match glm.model.family:
            case sm.families.Gaussian():
                family = Fam.TorchGaussian(link)
            case sm.families.Binomial():
                family = Fam.TorchBinomial(link)
            case sm.families.Poisson():
                family = Fam.TorchPoisson(link)
            case _: 
                raise TypeError("Family not supported.")
        torch_glm = TorchGLM(family=family, num_params=train_exog.shape[1], bias= use_bias)
        torch_glm.linear_transform.weight = torch.nn.Parameter(params)
        if bias is not None and bias != "none":
            torch_glm.linear_transform.bias = torch.nn.Parameter(bias)
        return torch_glm, train_endog, train_exog