import torch.nn as nn
import numpy as np

class GroupSort(nn.Module):
    """
    Implementation of the GroupSort activation proposed in Anil et al. (2019); coded from scratch, roughly following the original code at https://github.com/cemanil/LNets/blob/master/lnets/models/activations/group_sort.py
    """

    def __init__(self, num_groups, axis = -1):
        super(GroupSort, self).__init__()
        self.num_groups = num_groups
        self.axis = axis

    def forward(self, x):

        grouped_shape, sort_dim = self._compute_grouped_shape(x)

        grouped_tensor = x.view(grouped_shape)
        grouped_tensor, _ = grouped_tensor.sort(dim=sort_dim)

        sorted_tensor = grouped_tensor.view(x.shape)
        if self.check_group_sorted(sorted_tensor) == 0:
            raise ValueError("GroupSort failed to sort the tensor correctly.")
        return sorted_tensor
    def extra_repr(self):
        return 'num_groups={}'.format(self.num_groups)



    

    def _compute_grouped_shape(self, x):
        
        shape_list = list(x.shape)
        axis = len(shape_list)-1 if self.axis == -1 else self.axis

        if shape_list[axis] % self.num_groups != 0:
            raise ValueError("Number of elements on given axis not divisible by number of groups")
        
        sort_dim = axis+1
        shape_list.insert(sort_dim, shape_list[axis] // self.num_groups)
        shape_list[axis] = self.num_groups
        return shape_list, sort_dim
    
    #temporary, for debugging:
    def check_group_sorted(self, x):
        num_units = self.num_groups
        axis = self.axis
        size, _ = self._compute_grouped_shape(x)

        x_np = x.cpu().data.numpy()
        x_np = x_np.reshape(*size)
        axis = axis if axis == -1 else axis + 1
        x_np_diff = np.diff(x_np, axis=axis)

        # Return 1 iff all elements are increasing.
        if np.sum(x_np_diff < 0) > 0:
            return 0
        else:
            return 1


MaxMin = GroupSort(2)

    