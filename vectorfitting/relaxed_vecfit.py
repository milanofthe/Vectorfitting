#########################################################################################
##
##                         RELAXED VECTORFITTING ALGORITHM
##
##                                   Milan Rother
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np

from .vecfit import VecFit

    
# MATRIX VECFIT ====================================================================

class RelaxedVecFit(VecFit):
    """
    relaxed implementation of vectorfitting algorithm for mimo transfer functions
    
    see:
        RATIONAL APPROXIMATION OF FREQUENCY DOMAIN 
        RESPONSES BY VECTOR FITTING (Gustavsen, 1999)
        
        Improving the Pole Relocating Properties of 
        Vector Fitting (Gustavsen, 2006)
        
        Macromodeling of Multiport Systems Using a Fast Implementation 
        of the Vector Fitting Method (Deschrijver, 2008)
    """
    
    
    def _setup(self):
        """
        Setup fitting procedure.
        Set initial poles, compute initial residues 
        and calculate error of initial fit
        """

        #set initial poles
        self._set_poles()
        
        #compute initial residues according to selected mode
        self._compute_residues_relax()

        #compute initial error
        self.err_max , self.err_mean = self._evaluate_fit()

    
    def _compute_A_relax(self):
        
        """
        build matrix A for least squares calculation 
        of the residues with added relaxation
        """
        
        #some dimensions
        N, m, n = self.Data.shape
        nf = m * n
        
        #reshape data
        D_flat = self.Data.reshape((N, nf)).T
        
        #adding up all data for relaxation
        D_total_flat = np.sum( np.abs(D_flat), axis=0).flatten()
        
        #build blockmatrix with partial fractions of F
        X_F, X_S = self._build_X()
        
        N, nr = X_F.shape
        
        #dummy row for relaxation
        relax_F = np.zeros((nr * nf), dtype="complex128")
        
        #dummy matrix for padding
        Z = lambda n : np.zeros(( N, nr * n), dtype="complex128")
        
        #build blocks
        for k, (D, D_tot) in enumerate(zip(D_flat, D_total_flat)):
            
            #weight for relaxation
            W = D_tot / abs(D) / N
            
            #build blockmatrix for residues of Sigma
            FX_S = - (X_S.T * D).T
            
            #combine with relaxation column
            HH = np.hstack((Z(k), X_F, Z(nf-(k+1)), FX_S, -D.reshape((N,1)) ))
            
            #relaxation row
            relax_S = np.sum( (X_S.T * W).T , axis=0 )
            relax = np.hstack((relax_F, relax_S, N))
            
            #build subblock
            A_k = np.vstack(( HH, relax.real ))
            
            if k == 0:
                A = A_k
            else:
                A = np.vstack((A, A_k))
        
        return A
    

    # residue computation ----------------------------------------------------------

    def _compute_residues_relax(self):
        
        """
        compute new residues from poles 
        and data via least squares fit
        """
        
        #some dimensions
        N, m, n = self.Data.shape
        
        #number of io relations
        nf = m * n

        ncd = int(self.fit_Const) + int(self.fit_Diff) + int(self.fit_Zero)
        
        nr = ncd + self.n_real + 2*self.n_cpx
        
        #reformat matrix A
        A  = self._compute_A_relax()
        AA = np.vstack((A.real, A.imag))
        
        #reformat data and add relaxation row
        D_flat = self.Data.reshape((N, nf))
        F = np.vstack(( D_flat, np.zeros((1, nf)) )).T.flatten()
        FF = np.hstack((F.real, F.imag)).reshape((2*F.size, 1))
        
        #normalize to improve conditioning
        _AA_max = AA.max(axis=0)
        _AA_max = np.where(_AA_max == 0, 1, _AA_max)
        _FF_max = max( FF.max(), 1)
        
        AA_norm = AA / _AA_max / _FF_max
        FF_norm = FF / _FF_max
        
        #solve least squares problem
        R_norm, *_ = np.linalg.lstsq(AA_norm, FF_norm, rcond=None)
        
        #renormalize
        R = R_norm.flatten() / _AA_max
        
        #residues for F
        R_F = R[:nr*nf].reshape((nf, nr)).T.reshape((nr, m, n))
        
        #residues for Sigma
        R_S = R[nr*nf:-1]
        
        #const, diff and zero terms
        if self.fit_Const and self.fit_Diff and self.fit_Zero:
            self.Const = R_F[0]
            self.Diff  = R_F[1]
            self.Zero  = R_F[2]

        elif self.fit_Const and self.fit_Diff:
            self.Const = R_F[0]
            self.Diff  = R_F[1]
            self.Zero  = 0
        elif self.fit_Const and self.fit_Zero:
            self.Const = R_F[0]
            self.Diff  = 0
            self.Zero  = R_F[1]
        elif self.fit_Diff and self.fit_Zero:
            self.Const = 0
            self.Diff  = R_F[0]
            self.Zero  = R_F[1]

        elif self.fit_Const:
            self.Const = R_F[0]
            self.Diff  = 0
            self.Zero  = 0
        elif self.fit_Diff:
            self.Const = 0
            self.Diff  = R_F[0]
            self.Zero  = 0
        elif self.fit_Zero:
            self.Const = 0
            self.Diff  = 0
            self.Zero  = R_F[0]  

        else:
            self.Const = 0
            self.Diff  = 0
            self.Zero  = 0

        #handle real residues of F
        self.Residues_real = R_F[ncd:self.n_real+ncd]
        
        #handle complex residues of F
        res_cpx = R_F[self.n_real+ncd:]
        self.Residues_cpx = res_cpx[:self.n_cpx] + 1j * res_cpx[self.n_cpx:]
        
        #real residues of Sigma
        self.Residues_Sig_real = R_S[:self.n_real]
        
        #complex residues of Sigma
        res_cpx = R_S[self.n_real:]
        self.Residues_Sig_cpx  = res_cpx[:self.n_cpx] + 1j * res_cpx[self.n_cpx:]
        
        #relaxation constant
        self.d_relax = R[-1] + 1
        

    # fitting call ------------------------------------------------
        
    def fit(self, tol=1e-3, max_steps=5, debug=False):
        
        """
        perform fitting procedure
        
        INPUTS :
            tol       : (float) fitting tolerance for max relative error
            max_steps : (int) maximum number of iteration steps
            debug     : (bool) print error and final model order
        """

        for self.step in range(max_steps):

            #discard poles
            if self.autoreduce and self.step > 0:
                self._reduce_order(tol)

            #perform fitting iteration according to selected mode
            self._compute_poles()

            self._enforce_stability()

            self._compute_residues_relax()

            self._update_TF()

            #compute error
            self.err_max, self.err_mean = self._evaluate_fit()

            if debug: self._debug()

            if self.err_max < tol:
                return self.TF

        #directly return transfer funnction object
        return self.TF