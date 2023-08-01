import numpy as np

def compute_sin2(A):
    m, n = A.shape
    # compute numerator
    inner_p = A@np.transpose(A)
    num = inner_p[np.triu_indices(m, k = 1)]
    # compute denominator
    inner_p_dig = np.diagonal(inner_p)
    inner_p_dig = np.reshape(inner_p_dig, (m, 1))
    prod = inner_p_dig @ np.transpose(inner_p_dig)
    denom = prod[np.triu_indices(m, k = 1)]
    # compute sin2
    cos2 = num**2 / denom
    sin2 = 1 - cos2
    
    return sin2