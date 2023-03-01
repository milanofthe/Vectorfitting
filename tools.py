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



# PARSER ===============================================================================

def is_snp(filename):
    """
    check if filename provided is a .snp touchstone file
    """
    return filename.endswith( tuple([ f".s{n}p" for n in range(12) ]) )

def read_snp(path):
    
    """
    read S-parameter n-port touchstone files (*.snp)
    and returns frequency, nxn matrices and reference impedance
    
    Note:
        works for up to 12x12 S-parameter data 
        (if thats not enough, modify rps dict)
    """
    
    #check if
    if not is_snp(path):
        raise ValueError("path specified is not valid .snp touchstone file!")
    
    #init data
    Data  = []
    Freq  = []
    Lines = []
    
    #extract number of ports from path
    ns = re.search(".s(.*)p", path[-4:]).group(1)
    n = eval( ns )
    
    #rows that each datasample occupies
    rps = {1:1, 2:1, 3:3, 4:4, 5:10, 6:12, 7:14, 8:16, 9:27, 10:30, 11:33, 12:36}
    rows_per_sample = rps[n]
    
    #dictionary for frequency unit assignment
    unit_dict= {"ghz":1e9, "mhz":1e6, "khz":1e3, "hz":1}
    
    #default
    freq_unit   = 1
    Z_0         = 50
    data_format = "ma"
    
    #phase angle scale correction
    ph = np.pi / 180
    
    #read file
    with open(path, "r") as file:
        
        for line in file:
            
            #skip comments
            if "!" in line:
                continue
            
            #split line at spaces
            line_lst = line.split()
            
            #skip empty lines
            if len(line_lst) <= 1:
                continue
            
            #get header
            if "#" in line:
                
                #unpack line
                _, unit, _, data_format, _, Z0, *_ = line_lst
                
                #extract reference impedance
                Z0 = eval(Z0)
                
                #extract frequency unit
                freq_unit = unit_dict[unit.lower()]
                
                #extract format and set conversion
                if data_format.lower() == "ma":
                    conversion = lambda a, b: a * np.exp(1j * b * ph)
                elif data_format.lower() == "db":
                    conversion = lambda a, b: 10**(a/20) * np.exp(1j * b * ph)
                elif data_format.lower() == "ri":
                    conversion = lambda a, b: a + 1j * b
                    
                continue
            
            #save all data
            Lines.append(line_lst)
            
    #identifiy samples
    Samples = []
    S_tmp   = []
    for i, line in enumerate(Lines):
        if i % rows_per_sample == 0 and i>0:
            Samples.append(S_tmp)
            S_tmp = line
        else:
            S_tmp += line
    Samples.append(S_tmp)
    
    #process samples
    for f, *D in Samples:
        
        #save frequency
        Freq.append(eval(f) * freq_unit)
        
        #process data (evaluate and reshape)
        D = np.array([eval(d) for d in D]).reshape((n, 2*n))
        
        #separate and convert to complex
        D = conversion(D[:, 0::2], D[:, 1::2])
        
        #save converted data
        Data.append(D)
        
    return np.array(Freq), np.array(Data), Z_0



    
# FUNCTIONS FOR EVALUATION ==================================================


def evaluate_tf_laurent(f, poles, residues, const, diff):
    
    """
    evaluate rational function at frequency f
    from given poles and residues
    
    """
    
    s = 2j * np.pi * f
    
    H = const + s * diff
    
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
