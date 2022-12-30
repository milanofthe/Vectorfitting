#########################################################################################
##
##                              TRANSFERFUNCTION CLASS
##
##                                   Milan Rother
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np

from tools import (
    timer,
    evaluate_tf_laurent,
    gilbert_realization
    )



# TRANSFERFUNCION CLASS ===============================================================
    
class TransferFunction:
    
    """
    transfer function in pole residue form
    
    H(s) = Const + s * Diff + sum( Residues / (s - Poles) )
    
    """
    
    def __init__(self, Poles, Residues, Const=0, Diff=0):
        
        """
        initialize transfer function model
        
        INPUTS : 
            Poles    : array containing the poles 
            Residues : array of residue matrices
            Const    : matrix for constant value
            Diff     : matrix for differentiating value
        """
        
        self.Const    = Const
        self.Diff     = Diff
        self.Poles    = Poles
        self.Residues = Residues
        
        
    # methods for analysis ---------------------------------------
    
    def is_passive(self, Freq):
        
        """
        check if transfer function describes a 
        passive system in a given frequency range
        """
        
        #not stable => not passive
        stable = np.all(self.Poles.real < 0)
        
        #evaluate in frequency range
        H = self.evaluate(Freq)
        
        Eigs = []
        for h in H:
            #compute eigenvalues of heamitian part
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
        return gilbert_realization(self.Poles, self.Residues, self.Const, self.Diff)
    
    
    # methods for evaluation  ------------------------------------
    
    # @timer
    def evaluate(self, Freq):
        return np.array([evaluate_tf_laurent(f, self.Poles, self.Residues, self.Const, self.Diff) for f in Freq])

    
