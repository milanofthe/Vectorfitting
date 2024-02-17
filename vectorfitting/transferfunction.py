#########################################################################################
##
##                              TRANSFERFUNCTION CLASS
##
##                                   Milan Rother
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np

from vectorfitting.tools import (
    timer,
    gilbert_realization
    )


# AUX FUNCTIONS ========================================================================

def H_rng(shape=(1), n_cpx=2, n_real=2, f_min=0, f_max=1000):
    
    """
    generate ramdom frequency response 
    
    """
    
    omega_max = 2 * np.pi * f_max
    omega_min = 2 * np.pi * f_min
    
    #real poles
    poles_real = - (omega_min + (omega_max - omega_min) * (0.2 + 0.8 * np.random.rand(n_real))) / 3
    res_real   = 100 * (np.random.rand(n_real, *shape)-0.5)
    
    #complex poles
    poles_cpx = - (omega_min + (omega_max - omega_min) * (0.1 + 0.9 * np.random.rand(n_cpx))) / 15  \
                + 1j * (omega_min + (omega_max - omega_min) * (0.1 + 0.9 * np.random.rand(n_cpx)))
    res_cpx   = 1000 * (np.random.rand(n_cpx, *shape)-0.5) + 1j * 100 * (np.random.rand(n_cpx, *shape)-0.5)
    
    #combine
    Poles = np.hstack(( poles_real, poles_cpx, poles_cpx.conj() ))
    Residues = np.vstack(( res_real, res_cpx, res_cpx.conj() ))
    
    #const and diff
    const = 10 * ( 0.5 - np.random.rand(*shape))
    diff  = 20 / omega_max * (0.5 - np.random.rand(*shape))
    
    return TransferFunction(Poles, Residues, const, diff )


# TRANSFERFUNCION CLASS ===============================================================
    
class TransferFunction:
    
    """
    transfer function in pole residue form
    
    H(s) = Const + s * Diff + Zero / s + sum( Residues / (s - Poles) )
    
    """
    
    def __init__(self, Poles, Residues, Const=0, Diff=0, Zero=0):
        
        """
        initialize transfer function model
        
        INPUTS : 
            Poles    : array containing the poles 
            Residues : array of residue matrices
            Const    : matrix for constant value
            Diff     : matrix for differentiating value
            Zero     : matrix for poles at zero
        """
        
        self.Const    = Const
        self.Diff     = Diff
        self.Zero     = Zero
        self.Poles    = Poles
        self.Residues = Residues
            
        
    # methods for analysis ---------------------------------------
    
    def is_passive(self, Freq):
        
        """
        check if transfer function describes a 
        passive system in a given frequency range

        INPUTS :
            Freq : numpy array of frequency values
        """
        
        #not stable => not passive
        stable = np.all(self.Poles.real < 0)
        
        #evaluate in frequency range
        H = self.evaluate(Freq)
        
        Eigs = []
        for h in H:
            #compute eigenvalues of hermitian part
            ev = np.linalg.eigvals( h + h.T.conj() )
            Eigs.append(ev.real)
        
        #cast as array
        Eigs = np.array(Eigs)
        
        #update passivity
        passive = stable and np.all(Eigs > 0)
        
        return passive, Eigs
    
    # methods for conversions ------------------------------------
    
    def to_SS(self):
        """
        build minimal statespace realization
        """

        if self.Zero != 0 or self.Zero is not None:
            raise ValueError("'gilbert_realization' with pole at zero not implemented!")

        A, B, C, D, E, *_ = gilbert_realization(self.Poles, self.Residues, self.Const, self.Diff)

        return A, B, C, D, E
    
    # methods for evaluation  ------------------------------------
    
    # @timer
    def evaluate(self, Freq):
        """
        evaluate the transfer function in a given frequency range
        """

        jOmega = 2j * np.pi * Freq
        
        H = []
        for jw in jOmega:
            H_ = self.Const + jw * self.Diff + self.Zero / jw
            for r, p in zip(self.Residues, self.Poles):
                H_ += r / (jw - p)
            H.append(H_)

        return np.array(H)

    def get(self):
        """
        simply returns the transfer function model parameters
        """
        return self.Poles, self.Residues, self.Const, self.Diff, self.Zero