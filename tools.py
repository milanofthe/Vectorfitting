#########################################################################################
##
##                         TOOLS FOR DATA CONVERSION
##
##                                Milan Rother
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np

import re

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


def forall_dec(func):
    """
    wrapper for functions that operate 
    on frequency-data pairs
    -> execution on all pairs
    """
    @wraps(func)
    def wrapper(arr, *args, **kwargs):
        if isinstance(arr, (list, tuple, np.ndarray)):
            return [func(x, *args, **kwargs) for x in arr]
        return func(arr, *args, **kwargs)
    return wrapper

    
# FUNCTIONS FOR EVALUATION ==================================================

def evaluate_tf_laurent(f, poles, residues, const, diff):
    
    """
    evaluate rational function at frequency f
    from given poles and residues
    
    """
    
    s = 2j * np.pi * f
    
    H = const + s*diff 
    
    for r, p in zip(residues,poles):
        H += r / (s - p)
        
    return H

    
# MAXIMA ==================================================================================

def find_local_maxima(arr):
    """
    Finds all the local maxima in a multidimensional array

    """

    # Compute the differences between adjacent elements along axis 0
    diffs = np.diff(arr, axis=0)

    # Find the indices of the peaks 
    #(where the differences change from positive to negative)
    idx = np.argwhere((diffs[:-1] > 0) & (diffs[1:] < 0))
    return idx[:,0]



    
# STATESPACE REALIZATIONS =================================================================

def gilbert_realization(Poles, Residues, Const, Diff):
        
    """
    build real valued statespace model 
    from transfer function in pole residue form
    by Gilberts method and an additional 
    similarity transformation to get fully real matrices
    
    pole residue form:
        H(s) = Const + s * Diff + sum( Residues / (s - Poles) )
    
    statespace form:
        H(s) = C * (s*I - A)^-1 * B + D + s * E
        
    """
        
    #some dimensions
    N, m, n = Residues.shape
    
    #build real companion matrix from poles
    a = np.zeros(( N, N ))
    b = np.zeros(N)
        
    #residues
    C = np.ones((m, n*N))
        
    p_old = 0
    for k, (p, R) in enumerate(zip(Poles, Residues)):
        
        #check if complex conjugate
        is_cc = (p.imag != 0 and p == np.conj(p_old))
        p_old = p
        
        a[k,k] = p.real
        b[k] = 1
        if is_cc:
            a[k, k-1] = -p.imag
            a[k-1, k] = p.imag
            b[k]   = 0
            b[k-1] = 2
            
        #iterate columns or residue
        for i in range(n):
            C[:,k+N*i] = R[:,i].real if not is_cc else R[:,i].imag
                
    #build block diagional
    A = np.kron( np.eye(n, dtype=int), a )
    B = np.kron( np.eye(n, dtype=int), b ).T
    D = Const
    E = Diff
    
    return  A, B, C, D, E 


# AUX HELPER FUNCTIONS =====================================================


def wlstsq(A, B, W):

    """
    Compute the weighted least squares solution to a linear matrix equation.

    INPUTS:
        A : (numpy array) 2D array of shape (m, n) representing the input matrix.
        B : (numpy array) 1D or 2D array of shape (m,) or (m, k) dependent variables
        W : (numpy array) 1D array of shape (m,) representing the weights.

    """

    # Check input dimensions
    if A.ndim != 2 or B.ndim > 2 or W.ndim != 1:
        raise ValueError("Input dimensions are incorrect.")
    
    if A.shape[0] != B.shape[0] or A.shape[0] != W.shape[0]:
        raise ValueError(f"Input shapes are not aligned, A.shape={A.shape}, B.shape={B.shape}, W.shape={W.shape} ")
    
    # Broadcasting the weights to A and B
    A_weighted = A * W[:, np.newaxis]
    B_weighted = B * W[:, np.newaxis] if B.ndim == 1 else B * W[:, np.newaxis, np.newaxis]

    # Compute the normal equations
    lhs = A_weighted.T.dot(A_weighted)
    rhs = A_weighted.T.dot(B_weighted)

    # Cholesky decomposition
    L = np.linalg.cholesky(lhs)
    y = np.linalg.solve(L, rhs)
    x = np.linalg.solve(L.T, y)

    return x


# @timer
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
