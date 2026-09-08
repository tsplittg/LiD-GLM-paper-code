import torch
import numpy as np
from lidglm.extended_glm.transformer.iresblock import iResBlock as Chen_iResBlock
from lidglm.extended_glm.transformer.Custom_Lipschitz.Custom_Lipschitz import CustomLipschitzMLP


class Transformer(torch.nn.Module):
    def __init__(self, block_list, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.blocks = torch.nn.ModuleList(block_list)
        

    def forward(self, x, logpx=None):
        for block in self.blocks:
            x, logpx = block(x, logpx)
        return x, logpx

    def inverse(self, x, logpy=None):
        for block in reversed(self.blocks):
            x, logpy = block.inverse(x, logpy)
        return x, logpy
    
    def custom_inverse(self, x, atol=1e-7, rtol=1e-7):
        """
        Internally uses the same approximation as the regular inverse method, but allows for custom error tolerances. Meant for evaluation.
        """
        for block in reversed(self.blocks):
            x = block._inverse_fixed_point(x, atol=atol, rtol=rtol)
        return x
    
    def read_lipschitz_const(self, domain=None, codomain=None):
        """
        Slow and not gradient-preserving function to compute (or approximate) the lipschitz constant of each block. This function is mostly meant for post-hoc evaluation.

        domain and codomain are optional parameters fur debugging and should be used with care. If none are given, the values ingrained in the network are used.
        """
        lipschitz_constants_per_block = np.array([])
        lipschitz_constants_per_layer_per_block = np.array([])
        bound_for_total_const = 1
        for block in self.blocks:
            total, layerwise = block.read_lipschitz_const()
            lipschitz_constants_per_block = np.append(lipschitz_constants_per_block, total)
            lipschitz_constants_per_layer_per_block = np.append(lipschitz_constants_per_layer_per_block, layerwise)
            bound_for_total_const *= total
        return {"total_const_bound":bound_for_total_const, "bound_per_block":lipschitz_constants_per_block, "const_per_layer_per_block": lipschitz_constants_per_layer_per_block}
    
    def get_initialization_params(self):
        """
        Returns the parameters used for the initialization of the transformer. Used for saving the transformer and dependent models.
        """
        return [block.get_initialization_params() for block in self.blocks]
    

class TransformerUtils():
    """
    Helper class to assist in initialization of Transformers.
    """

    def initialize_transformer(self, num_params, lipschitz_const, parameterized:bool = False, width = 5, 
                               depth=None, domain_codomain=2, activation_string="tanh", num_blocks = None, very_last_no_bias =False, **kwargs):
        """
        For now only equal domain and codomain are supported.

        Parameters:
        depth: the total number of hidden layers the transformer is allowed to use, not counting the first and last layers of each block. Since each block can only achieve a Lipschitz constant less than 2, 
                the number of layers must be at least: num_blocks = np.ceil(np.log(lipschitz_const)/np.log(2)).astype(int)
        activation_string: the activation function to be used in the transformer. If group_sort is chose, then a group_size by which width is divisible has to be specified in the kwargs.
        very_last_no_bias: if True, the last linear layer of last block not have a bias. This is needed if scaling methods are used later. Default is False.
        """

        if num_blocks is None:
            num_blocks = np.ceil(np.log(lipschitz_const)/np.log(2)).astype(int)
        if depth is None:
            depth = num_blocks
        if num_blocks > depth:
            raise ValueError("The number of blocks needed to achieve the desired Lipschitz constant is too large for the given depth. Please increase the depth or decrease the lipschitz constant")


        block_lipschitz_const = lipschitz_const**(1/num_blocks) -1
        blocks = []

        #the depth is distributed among the blocks, the number of hidden layers for each block is simply compute by division without remainder; 
            # the first (depth mod num_blocks) blocks then get one additional layer if the division depth/num_blocks has a remainder
        blockwise_depth = [(depth//num_blocks)+1]* (depth%num_blocks) + [depth//num_blocks]*(num_blocks - depth%num_blocks)

        for b in range(num_blocks-1):
            blocks.append(Chen_iResBlock.initialize_from_params(num_params=num_params, lipschitz_const=block_lipschitz_const, domain=domain_codomain, codomain=domain_codomain, width=width, depth=blockwise_depth[b], 
                                                                activation_string=activation_string, **kwargs))
        #last layer is appended seperately as there are cases where it shouldn't have a bias and last activation function
        blocks.append(Chen_iResBlock.initialize_from_params(num_params=num_params, lipschitz_const=block_lipschitz_const, domain=domain_codomain, codomain=domain_codomain, width=width, depth=blockwise_depth[num_blocks-1], 
                                                                activation_string=activation_string, last_layer_no_bias =very_last_no_bias, **kwargs))
        return Transformer(blocks)
        
    def initialize_from_saved_params(self, saved_params, parameterized:bool = False):
        """
        Initializes a Transformer from saved parameters. The saved_params should be a list of dictionaries, each containing the parameters for one block.
        """
        block_list = []
        for param_dict in saved_params:
            nnet = CustomLipschitzMLP(**(param_dict["nnet_params"]))
            block_list.append(Chen_iResBlock(nnet=nnet, **param_dict))
        return Transformer(block_list)
