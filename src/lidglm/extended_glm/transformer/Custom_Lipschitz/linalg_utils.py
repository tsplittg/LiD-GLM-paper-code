"""
Collection of utility functions to aid in linear algebra computations. 
"""

import numpy as np
import torch

def matrix_norm(A, domain=2, codomain=2):
    """
    A slow and not gradient-preserving function to compute the matrix norm of a matrix A. This function is mostly meant for post-hoc evaluation
    """
    
    if domain is float("inf"):
        domain = np.inf
    if codomain is float("inf"):
        codomain = np.inf

    if isinstance(A, torch.Tensor):
        A = A.detach().cpu().numpy()
    
    match domain, codomain:
        case 1, 1:
            return np.linalg.norm(A, ord=1)
        case 2, 2:
            return np.linalg.norm(A, ord=2)
        case np.inf, np.inf:
            return np.linalg.norm(A, ord=np.inf)
        case _, _:
            raise ValueError("Combination of domain and codomain not recognized or not supported")