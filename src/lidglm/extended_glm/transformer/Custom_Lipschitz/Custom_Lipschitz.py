"""
A customized version of the Lipschitz MLP as it is used in the normflows package https://github.com/VincentStimper/normalizing-flows.
This version makes use of the fact that the induced norm linear layers support different p-Norms.

"""

from normflows.nets.lipschitz import *
import copy
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from lidglm.extended_glm.transformer.Custom_Lipschitz.linalg_utils import *
from lidglm.extended_glm.transformer.Custom_Lipschitz.Group_Sort import GroupSort

class CustomLipschitzMLP(nn.Module):
    """Fully connected neural net which is Lipschitz continuous with Lipschitz constant L < 1"""

    def __init__(
        self,
        channels,
        lipschitz_const=0.97,
        max_lipschitz_iter=5,
        lipschitz_tolerance=None,
        init_zeros=True,
        domain=2,
        codomain=2,
        activation_string = "swish",
        activation_list = None, 
        last_layer_no_bias=False
    ):
        """
        Constructor
          channels: Integer list with the number of channels of
            the layers
        lipschitz_const: Maximum Lipschitz constant of each layer
        max_lipschitz_iter: Maximum number of iterations used to
            ensure that layers are Lipschitz continuous with L smaller than
            set maximum; if None, tolerance is used
        lipschitz_tolerance: Float, tolerance used to ensure
        Lipschitz continuity if max_lipschitz_iter is None, typically 1e-3
        init_zeros: Flag, whether to initialize last layer
        approximately with zeros
        activation, activation_list: the activation function to be used in the network can either be specified by a string for all layers at once or with a list for each layer individually, in which case activation_string should be set to "list".
                 Default ist "swish".
        last_layer_no_bias: if True, the last linear layer does not have a bias. This is needed if scaling methods are used later. Default is False. 
        """
        super().__init__()
        if activation_list is not None and activation_string != "list":
            raise ValueError("If activation_list is not None, activation_string has to be set to 'list'.")
        if activation_list is not None and len(activation_list) != len(channels)-1:
            raise ValueError("If activation_list is not None, it has to have the same length as the number of layers.")
        self.n_layers = len(channels) - 1
        self.channels = channels
        self.lipschitz_const = lipschitz_const
        self.max_lipschitz_iter = max_lipschitz_iter
        self.lipschitz_tolerance = lipschitz_tolerance
        self.init_zeros = init_zeros
        self.domain = domain
        self.codomain = codomain
        self.activation_string = activation_string
        self.activation_list = activation_list
        self.last_layer_no_bias = last_layer_no_bias



        layers = []
        for i in range(self.n_layers):
            if(codomain==1 and domain==1):
                layers += [
                    # layer has to inherit from "InducedNormLinear" or lipschitz-normalization has to be adjusted
                    ColumnSumNormLinear(
                        in_features=channels[i],
                        out_features=channels[i + 1],
                        coeff=lipschitz_const,
                        bias = (i!=self.n_layers-1 or not last_layer_no_bias),
                        zero_init=init_zeros if i == (self.n_layers - 1) else False,
                    )                    
                ]
            elif(codomain==float("inf") and domain==float("inf")):
                layers += [
                    # layer has to inherit from "InducedNormLinear" or lipschitz-normalization has to be adjusted
                    RowSumNormLinear(
                        in_features=channels[i],
                        out_features=channels[i + 1],
                        coeff=lipschitz_const,
                        bias = (i!=self.n_layers-1 or not last_layer_no_bias),
                        zero_init=init_zeros if i == (self.n_layers - 1) else False,
                    )
                ]
            else:
                layers += [
                    # layer has to inherit from "InducedNormLinear" or lipschitz-normalization has to be adjusted
                    Custom_InducedNormLinear(
                        in_features=channels[i],
                        out_features=channels[i + 1],
                        coeff=lipschitz_const,
                        domain=domain,
                        codomain=codomain,
                        n_iterations=max_lipschitz_iter,
                        atol=lipschitz_tolerance,
                        rtol=lipschitz_tolerance,
                        bias = (i!=self.n_layers-1 or not last_layer_no_bias),
                        zero_init=init_zeros if i == (self.n_layers - 1) else False,
                    )
                ]
            if i < self.n_layers - 1:
                match activation_string:
                    case "swish":
                        layers += [Swish()]
                    case "relu":
                        layers += [nn.ReLU()]
                    case "fullsort":
                        layers += [GroupSort(channels[i+1])]
                    case "groupsort":
                        if i < self.n_layers-1:
                            layers += [GroupSort(2)]
                    case "tanh":
                        layers += [nn.Tanh()]
                    case "list":
                        layers += [activation_list[i]]
                    case _:
                        raise ValueError("Activation function not recognized.")

            

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    def read_lipschitz_const(self, domain=None, codomain=None):
        """
        Slow and non-gradient preserving method to read the Lipschitz constant of the network. Meant mostly for post-hoc evaluation.
        domain and codomain are optional parameters fur debugging and should be used with care. If none are given, the values ingrained in the network are used.
        """

        lipschitz_constants = np.array([])
        for layer in self.net:
            if isinstance(layer, Custom_InducedNormLinear):
                lipschitz_constants = np.append(lipschitz_constants,layer.read_lipschitz_const(domain,codomain))

        return np.prod(lipschitz_constants), lipschitz_constants
    
    def get_initialization_params(self):
        """
        Returns the parameters used for initialization of the network. Used for saving the network and dependent models.
        """
        return {
            "channels": self.channels,
            "lipschitz_const": self.lipschitz_const,
            "max_lipschitz_iter": self.max_lipschitz_iter,
            "lipschitz_tolerance": self.lipschitz_tolerance,
            "init_zeros": self.init_zeros,
            "domain": self.domain,
            "codomain": self.codomain,
            "activation_string": self.activation_string,
            "activation_list": self.activation_list,
            "last_layer_no_bias": self.last_layer_no_bias
        }
    
    def projected_to_one_dimension(self, dimension: int):
        """
        Returns a 1-dimensional version of the network, which we use in evaluation of the Lipschitz constant for each feature. Essentially, we just replace the last layer with a m x 1 - matrix
        """
        new_network = copy.deepcopy(self)
        new_network.channels[-1] = 1
        new_network.net[-1] = new_network.net[-1].projected_to_one_dimension(dimension)

        return new_network
        

class Custom_InducedNormLinear(InducedNormLinear):
    """
    Class that covers the case of easily computed norms.
    """
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        coeff=0.97,
        domain=2,
        codomain=2,
        n_iterations=None,
        atol=None,
        rtol=None,
        zero_init=False,
        **unused_kwargs
    ):
        del unused_kwargs
        super(InducedNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.domain = domain
        self.codomain = codomain
        #boolean whether or not normalization is applied during forward; should only be turned off after training, when internal weights have been adjusted and for good reason
        self._normalized = True   
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters(zero_init)
        if (domain == 1 and codomain == 1) or (domain == float("inf") or codomain == float("inf")):
            return
        with torch.no_grad():
            domain, codomain = self.compute_domain_codomain()
        

        h, w = self.weight.shape
        self.register_buffer("scale", torch.tensor(0.0))
        self.register_buffer(
            "u", normalize_u(self.weight.new_empty(h).normal_(0, 1), codomain)
        )
        self.register_buffer(
            "v", normalize_v(self.weight.new_empty(w).normal_(0, 1), domain)
        )

        # Try different random seeds to find the best u and v.
        with torch.no_grad():
            self.compute_weight(True, n_iterations=200, atol=None, rtol=None)
            best_scale = self.scale.clone()
            best_u, best_v = self.u.clone(), self.v.clone()
            if not (domain == 2 and codomain == 2):
                for _ in range(10):
                    self.register_buffer(
                        "u",
                        normalize_u(self.weight.new_empty(h).normal_(0, 1), codomain),
                    )
                    self.register_buffer(
                        "v", normalize_v(self.weight.new_empty(w).normal_(0, 1), domain)
                    )
                    self.compute_weight(True, n_iterations=200)
                    if self.scale > best_scale:
                        best_u, best_v = self.u.clone(), self.v.clone()
            self.u.copy_(best_u)
            self.v.copy_(best_v)

    def reset_parameters(self, zero_init=False):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if zero_init:
            # normalize cannot handle zero weight in some cases.
            self.weight.data.div_(1000)
        if self.bias is not None:
            #ToDo: compuation useless if not using image data?
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            # unlike base normflows the bias is also initialized as zeros in last layer
            self.bias.data.div_(1000)

    def read_lipschitz_const(self, domain=None, codomain=None):
        """
        Slow and non-gradient preserving method to read the Lipschitz constant of the layer. Meant mostly for post-hoc evaluation.
        """
        if domain is None and codomain is None:
            domain, codomain = self.domain, self.codomain
        return matrix_norm(self.normalized_weight, self.domain, self.codomain)
    
    def compute_domain_codomain(self):
        if torch.is_tensor(self.domain):
            domain = asym_squash(self.domain)
            codomain = asym_squash(self.codomain)
        else:
            domain, codomain = self.domain, self.codomain
        return domain, codomain

    def compute_one_iter(self):
        domain, codomain = self.compute_domain_codomain()
        u = self.u.detach()
        v = self.v.detach()
        weight = self.weight.detach()
        u = normalize_u(torch.mv(weight, v), codomain)
        v = normalize_v(torch.mv(weight.t(), u), domain)
        return torch.dot(u, torch.mv(weight, v))

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        u = self.u
        v = self.v
        weight = self.weight

        if update:

            n_iterations = self.n_iterations if n_iterations is None else n_iterations
            atol = self.atol if atol is None else atol
            rtol = self.rtol if rtol is None else atol

            if n_iterations is None and (atol is None or rtol is None):
                raise ValueError("Need one of n_iteration or (atol, rtol).")

            max_itrs = 200
            if n_iterations is not None:
                max_itrs = n_iterations

            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                for _ in range(max_itrs):
                    # Algorithm from http://www.qetlab.com/InducedMatrixNorm.
                    if n_iterations is None and atol is not None and rtol is not None:
                        old_v = v.clone()
                        old_u = u.clone()

                    u = normalize_u(torch.mv(weight, v), codomain, out=u)
                    v = normalize_v(torch.mv(weight.t(), u), domain, out=v)

                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement() ** 0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement() ** 0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                self.v.copy_(v)
                self.u.copy_(u)
                u = u.clone()
                v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight
    
    @property
    def normalized_weight(self):
        """function name slightly misleading. The function usually returns the normalized weight, but returns the unnormalized weight if it has been explicitly told to do so.
        """
        if self.is_normalized:
            return self.compute_weight(update=False)
        else:
            return self.weight
    
    def set_internal_weight(self):
        """
        Adjusts the internal weights to be normalized. This should only be done after training if self.is_normalized is set to False.
        """
        with torch.no_grad():
            self.weight = torch.nn.Parameter(self.compute_weight(update=False))
    
    @property
    def is_normalized(self):
        """
        Returns a boolean indicating whether the weight is normalized during the forward pass or not.
        """
        return self._normalized

    @is_normalized.setter
    def is_normalized(self, value):
        """
        Ability to turn normalization off. The internal weights will be normalized first, which doesn't necessarily happen during training. 
        """
        if not isinstance(value, bool):
            raise ValueError("is_normalized can only be a boolean value.")
        if not value:
            self.set_internal_weight()
            self._normalized = value
        else:
            self._normalized = value

    def forward(self, input):
        weight = self.normalized_weight
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        domain, codomain = self.compute_domain_codomain()
        return (
            "in_features={}, out_features={}, bias={}"
            ", coeff={}, domain={:.2f}, codomain={:.2f}, n_iters={}, atol={}, rtol={}, learnable_ord={}, weight={}, bias={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.coeff,
                domain,
                codomain,
                self.n_iterations,
                self.atol,
                self.rtol,
                torch.is_tensor(self.domain),
                self.normalized_weight,
                self.bias
            )
        )
    
    def projected_to_one_dimension(self, dimension: int):
        """
        Returns an in_features x 1 - version of the linear layer, which we use in evaluation of the Lipschitz constant for each feature.
        """
        if self.domain == 1 and self.codomain == 1:
            linear_layer_constructor = ColumnSumNormLinear
        elif self.domain == float("inf") and self.codomain == float("inf"):
            linear_layer_constructor = RowSumNormLinear
        else:
            linear_layer_constructor = Custom_InducedNormLinear
        
        projected_layer = linear_layer_constructor(
            in_features=self.in_features,
            out_features=1,
            bias = self.bias is not None, 
            coeff=self.coeff,
            domain=self.domain,
            codomain=self.codomain,
            n_iterations=self.n_iterations,
            atol=self.atol,
            rtol=self.rtol,
            zero_init=False
        )
        projected_layer.weight = torch.nn.Parameter(self.weight[:, dimension].unsqueeze(1))
        if self.bias is not None:
            projected_layer.bias = torch.nn.Parameter(self.bias[dimension].unsqueeze(0))
        return projected_layer




class ColumnSumNormLinear(Custom_InducedNormLinear):
    """
    More efficient implementation a layer in which the norm of the linear layer
    is being induced by the 1-Norm on both domain and codomain. 
    In this special case, the norm can be directly computed as the maximum over all 
    sums over the columns of the corresponding matrix.
    (see also https://en.wikipedia.org/wiki/Matrix_norm#p_=_1,_%E2%88%9E)
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        coeff=0.97,
        zero_init=False,
        **unused_kwargs
    ):
        del unused_kwargs
        super().__init__(in_features = in_features, out_features=out_features, domain=1, codomain=1, bias=bias, coeff=coeff, zero_init=zero_init)

            
    def compute_weight(self, **kwargs):
        """
        Only implemented to keep compatibility with normflows; if the normalized weights
        are to be computed, use "normalized_weight" instead
        """
        sigma = torch.max(torch.sum(torch.abs(self.weight),axis=0))
        factor = torch.max(torch.ones(1).to(self.weight.device), sigma / self.coeff)
        weight=self.weight/factor
        return weight

    

class RowSumNormLinear(Custom_InducedNormLinear):
    """
    More efficient implementation a layer in which the norm of the linear layer
    is being induced by the sup-Norm on both domain and codomain. 
    In this special case, the norm can be directly computed as the maximum over all 
    rows over the columns of the corresponding matrix.
    (see also https://en.wikipedia.org/wiki/Matrix_norm#p_=_1,_%E2%88%9E)
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        coeff=0.97,
        zero_init=False,
        **unused_kwargs):
        
        del unused_kwargs
        
        super().__init__(in_features = in_features, out_features=out_features, domain=float("inf"), codomain=float("inf"), bias=bias, coeff=coeff, zero_init=zero_init)
            
    def compute_weight(self, **kwargs):
        sigma = torch.max(torch.sum(torch.abs(self.weight),axis=1))
        factor = torch.max(torch.ones(1).to(self.weight.device), sigma / self.coeff)
        weight=self.weight/factor
        return weight