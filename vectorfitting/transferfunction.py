#########################################################################################
##
##                              TRANSFERFUNCTION CLASS
##
##                                   Milan Rother
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np


# TRANSFERFUNCION CLASS =================================================================
    
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
            

    # methods for analysis --------------------------------------------------------------
    
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


    # methods for evaluation  -----------------------------------------------------------

    def __call__(self, Freq):
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


    def evaluate(self, Freq):
        #alias for call method
        return self(Freq)
        

    def get(self):
        """
        simply returns the transfer function model parameters
        """
        return self.Poles, self.Residues, self.Const, self.Diff, self.Zero