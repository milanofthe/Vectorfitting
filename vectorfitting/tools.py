#########################################################################################
##
##                         TOOLS FOR DATA CONVERSION
##
##                                Milan Rother
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np

from functools import wraps
from time import perf_counter


# MISC ==================================================================================

def timer(func):
    
    """
    This function shows the execution time 
    of the function object passed
    
    """
    
    def wrap_func(*args, **kwargs):
        
        t1 = perf_counter()
        result = func(*args, **kwargs)
        t2 = perf_counter()
        
        print(f'Function {func.__name__!r} executed in {(t2-t1)*1e3:.2f}ms')
        
        return result
    
    return wrap_func


def dB(H):
    return 20*np.log10(abs(H))


def ang(H):
    return np.unwrap(np.angle(H, deg=True), 180)

    
# MAXIMA ==================================================================================

def find_local_maxima(arr):
    """
    Finds all the local maxima in a multidimensional array

    """

    # Compute the differences between adjacent elements along axis 0
    diffs = np.diff(arr, axis=0)

    # Find the indices of the peaks (sign change in differences)
    idx = np.argwhere((diffs[:-1] > 0) & (diffs[1:] < 0))
    return idx[:, 0]


# AUX HELPER FUNCTIONS =====================================================

def smartblock(AA):
    
    """
    extended version of numpy.block() where 
    the dimensions of integer entries get
    upscaled automatically to the other blocks

    """

    # Collect shapes
    shapes = np.array([[x.shape if isinstance(x, np.ndarray) else (1, 1) for x in R] for R in AA])
    r_shapes = np.max(shapes[:, :, 0], axis=1)
    c_shapes = np.max(shapes[:, :, 1], axis=0)

    # Upscale integer entries
    AA = np.where(np.vectorize(lambda x: not isinstance(x, np.ndarray))(AA),
                  np.vectorize(lambda x, r, c: np.full((r, c), x))(AA, r_shapes[:, np.newaxis], c_shapes),
                  AA)

    return np.block(AA)
