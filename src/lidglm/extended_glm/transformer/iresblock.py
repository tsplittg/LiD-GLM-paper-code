"""
Code taken from Chen et al. https://github.com/rtqichen/residual-flows/blob/8170138c850a3574319491d97093bc860ce4922d/resflows/layers/iresblock.py
with some additional comments, restructuring and added functionality

The original Code adapted in this script was published under the MIT license:

MIT License

Copyright (c) 2019 Ricky Tian Qi Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import math
import numpy as np
import torch
import torch.nn as nn
from lidglm.extended_glm.transformer.Custom_Lipschitz.Custom_Lipschitz import CustomLipschitzMLP
from lidglm.extended_glm.transformer.Custom_Lipschitz.Group_Sort import GroupSort
import logging
import copy

logger = logging.getLogger()

__all__ = ['iResBlock']


class iResBlock(nn.Module):

    def __init__(
        self,
        nnet,
        geom_p=0.5,
        lamb=2.,
        n_power_series=None,
        exact_trace=False,
        brute_force=False,
        n_samples=1,
        n_exact_terms=2,
        n_dist='geometric',
        neumann_grad=True,
        grad_in_forward=False,
        activation_string="swish",
        activation_list=None,
        **kwargs
    ):
        """
        Args:
            nnet: a nn.Module
            n_power_series: number of power series. If not None, uses a biased approximation to logdet.
            exact_trace: if False, uses a Hutchinson trace estimator. Otherwise computes the exact full Jacobian.
            brute_force: Computes the exact logdet. Only available for 2D inputs.
        """
        nn.Module.__init__(self)
        self.nnet = nnet
        self.n_dist = n_dist
        self.input_geom_p = geom_p

        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = nn.Parameter(torch.tensor(lamb))
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.exact_trace = exact_trace
        self.brute_force = brute_force
        self.n_exact_terms = n_exact_terms
        self.grad_in_forward = grad_in_forward
        self.neumann_grad = neumann_grad

        # store the samples of n.
        self.register_buffer('last_n_samples', torch.zeros(self.n_samples))
        self.register_buffer('last_firmom', torch.zeros(1))
        self.register_buffer('last_secmom', torch.zeros(1))

    # more convenient initialization function:
    @classmethod
    def initialize_from_params(cls, num_params, lipschitz_const, width = 5, depth=1, domain=2, codomain=2, activation_string="swish", last_layer_no_bias=False, **kwargs):
        """
        last_layer_no_bias: if True, the last linear layer of the network doesn't have a bias. This is needed if scaling methods are used later. Default is False. 
        """
        # each of the [depth+1] weight matrices is assigned a lipschitz constant of lipschitz_const**(1/(depth+1)); therefore the total Lipschitz constant of the block is bounded by  lipschitz_const
        if activation_string == "group_sort":
            assert "group_size" in kwargs, "If group_sort is chosen as activation function, a group_size has to be specified in the kwargs"
            activation_list=[]
            for i in range(depth):
                activation_list.append(GroupSort(kwargs["group_size"]))
            activation_list.append(torch.nn.Identity())
            activation_string = "list"
        else:
            activation_list = None
        # if the Lipschitz constant is larger than one, the network isn't an i-Res-block anymore. However, we still allow this for experimentation purposes.
        # Theoretically, the same applies to Lipschitz constant exactly one, but in practice the upper bound won't be fully exhausted, so we allow this case.
        if lipschitz_const > 1:
            SyntaxWarning("Lipschitz constant larger than one: the block is not an i-Res-block anymore.")
        lipsch_per_layer = lipschitz_const**(1/(depth+1))
        net = CustomLipschitzMLP([num_params] * 1 + [width] * depth + [num_params] * 1,
                                             lipschitz_const=lipsch_per_layer, domain=domain, codomain=codomain, activation_string=activation_string, activation_list=activation_list, last_layer_no_bias= last_layer_no_bias)
        return cls(net, exact_trace = True, n_exact_terms = 7)
    
    def forward(self, x, logpx=None):
        """
        returns y=x+g(x) and log p_y(y) = log p_x(x) - log|det(d(x+g(x))/dx)|,
            unified the dimensionality of the return argument; now logpx is always returned even if it zero for consistency reasons
        """
        if logpx is None:
            y = x + self.nnet(x)
            return y, logpx
        else:
            g, logdetgrad = self._logdetgrad(x)
            return x + g, logpx - logdetgrad

    def inverse(self, y, logpy=None):
        """
        returns x and log p_x(x) = log p_y(y) + log|det(d(x+g(x))/dx)|,
            unified the dimensionality of the return argument; now logpy is aleays returned even if it zero for consistency reasons
        """
        x = self._inverse_fixed_point(y)
        if logpy is None:
            return x, logpy
        else:
            return x, logpy + self._logdetgrad(x)[1]
        
    def read_lipschitz_const(self, domain=None, codomain=None):
        """
        new function, returns the Lipschitz constant of the network. Slow and not gradient preserving, meant for post-hoc evaluation
        domain and codomain are optional parameters fur debugging and should be used with care. If none are given, the values ingrained in the network are used.
        """
        return self.nnet.read_lipschitz_const()

    def _inverse_fixed_point(self, y, atol=1e-5, rtol=1e-5):
        x, x_prev = y - self.nnet(y), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev)**2 / tol < 1):
            x, x_prev = y - self.nnet(x), x
            i += 1
            if i > 1000:
                logger.info('Iterations exceeded 1000 for inverse.')
                break
        return x

    def _logdetgrad(self, x):
        """Returns g(x) and logdet|d(x+g(x))/dx|."""

        with torch.enable_grad():
            if (self.brute_force or not self.training) and (x.ndimension() == 2 and x.shape[1] == 2):
                ###########################################
                # Brute-force compute Jacobian determinant.
                ###########################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                # Brute-force logdet only available for 2D.
                jac = batch_jacobian(g, x)
                batch_dets = (jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1) - jac[:, 0, 1] * jac[:, 1, 0]
                return g, torch.log(torch.abs(batch_dets)).view(-1, 1)


            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn: callable[[int], np.array] = lambda m: np.random.geometric(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset) #TS: P(N >= k+offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn: callable[[int], np.array] = lambda m: np.random.poisson(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)

            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples) #ToDo: in what case would it make sense for self.n_samples to ever be >1? especially if then the max over all samples is used?
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples) #ToDo: what does this seconds row do? If self.num_samples =1 (the only case that makes sense to me right now), 
                                                                                    # then n_samples is a number and this row is just the indicator whether  
                                                                                    # k <= n_samples + self.n_exact_terms
                                                                                    # But is coeff_fn ever called for k>n_samples+self.n_exact_terms anyway?
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.
            else:
                # Unbiased estimation with more exact terms.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = lambda k: 1 / rcdf_fn(k, 20) * \
                    sum(n_samples >= k - 20) / len(n_samples)

            if not self.exact_trace:
                ####################################
                # Power series with trace estimator.
                ####################################
                vareps = torch.randn_like(x)

                # Choose the type of estimator.
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

                # Do backprop-in-forward to save memory.
                if self.training and self.grad_in_forward:
                    g, logdetgrad = mem_eff_wrapper(
                        estimator_fn, self.nnet, x, n_power_series, vareps, coeff_fn, self.training
                    )
                else:
                    x = x.requires_grad_(True)
                    g = self.nnet(x)
                    logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, self.training)
            else:
                ############################################
                # Power series with exact trace computation.
                ############################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                jac = batch_jacobian(g, x)
                logdetgrad = batch_trace(jac)
                jac_k = jac
                for k in range(2, n_power_series + 1):
                    jac_k = torch.bmm(jac, jac_k)
                    logdetgrad = logdetgrad + (-1)**(k + 1) / k * coeff_fn(k) * batch_trace(jac_k)

            if self.training and self.n_power_series is None:
                self.last_n_samples.copy_(torch.tensor(n_samples).to(self.last_n_samples))
                estimator = logdetgrad.detach()
                self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))
            return g, logdetgrad.view(-1, 1)

    def extra_repr(self):
        return 'dist={}, n_samples={}, n_power_series={}, neumann_grad={}, exact_trace={}, brute_force={}'.format(
            self.n_dist, self.n_samples, self.n_power_series, self.neumann_grad, self.exact_trace, self.brute_force
        )


    def scale_last_layer(self, factor: float):
        """
        Adds a scaling layer with a given factor to a deepcopy of self. 
        Not meant to be used during training, meant for post-hoc evaluation/simulation studies.
        """
        copied_block = copy.deepcopy(self)
        copied_block.nnet.net.append(multiplication_layer(factor))
        return copied_block
    
    def get_initialization_params(self):
        """
        Returns the parameters used for the initialization of the i-Res-block. Used for saving the transformer and dependent models.
        """
        return {
            "nnet_params": self.nnet.get_initialization_params(),
            "geom_p": self.input_geom_p,
            "lamb": self.lamb.item(),
            "n_power_series": self.n_power_series,
            "exact_trace": self.exact_trace,
            "brute_force": self.brute_force,
            "n_samples": self.n_samples,
            "n_exact_terms": self.n_exact_terms,
            "n_dist": self.n_dist,
            "neumann_grad": self.neumann_grad,
            "grad_in_forward": self.grad_in_forward
        }


def batch_jacobian(g, x):
    jac = []
    for d in range(g.shape[1]):
        jac.append(torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=True)[0].view(x.shape[0], 1, x.shape[1]))
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)


#####################
# Logdet Estimators
#####################
class MemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params


def basic_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = (-1)**(k + 1) / k * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad


def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad


def mem_eff_wrapper(estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *list(gnet.parameters())
    )


# -------- Helper distribution functions --------
# These take python ints or floats, not PyTorch tensors.

def geometric_1mcdf(p, k, offset):

    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)

def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


# -------------- Helper functions --------------


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)


def _flatten(sequence):
    flat = [p.reshape(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [p.reshape(-1) if p is not None else torch.zeros_like(q).view(-1) for p, q in zip(sequence, like_sequence)]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])

class multiplication_layer(nn.Module):
    """
    Layer that multiplies the input with a scalar. This is needed for the scaling methods.
    """
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        
    def forward(self, x):
        return x * self.factor
    def extra_repr(self):
        return 'factor={}'.format(self.factor)
    