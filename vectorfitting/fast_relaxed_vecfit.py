#########################################################################################
##
##                        FAST RELAXED VECTORFITTING ALGORITHM
##
##                                   Milan Rother
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np

from .vecfit import VecFit


# MATRIX VECFIT ====================================================================

class FastRelaxedVecFit(VecFit):
    """
    fast relaxed implementation of vectorfitting algorithm for mimo transfer functions
    
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
        
        #compute initial residues
        self._compute_residues_F_fast_relax()

        #compute initial error
        self.err_max , self.err_mean = self._evaluate_fit()


    # residue computation ----------------------------------------------------------
        
    def _compute_residues_Sigma_fast_relax(self):
        
        """
        compute residues and relaxation constant with 
        relaxation, fast implementation with block qr
        """
        
        #some dimensions
        N, m, n = self.Data.shape
        nf = m * n
        
        #number of residues, const and diff variables
        ncd = int(self.fit_Const) + int(self.fit_Diff) + int(self.fit_Zero)    
        nr = ncd + self.n_real + 2*self.n_cpx
        
        #reshape data
        D_flat = self.Data.reshape((N, nf)).T
        
        #adding up all data for relaxation
        D_total_flat = np.sum( np.abs(D_flat), axis=0).flatten()
        
        #init dummy row for relaxation
        relax_F = np.zeros((nr), dtype="complex128")
        
        #build blockmatrix with partial fractions of F
        X_F, X_S = self._build_X()
        
        #build blocks
        for k, (D, D_tot) in enumerate(zip(D_flat, D_total_flat)):
            
            #weight for relaxation
            W = D_tot / abs(D) / N
            
            #build blockmatrix for residues of Sigma
            FX_S = - (X_S.T * D).T
            
            #combine with relaxation column
            HH = np.hstack(( X_F, FX_S, -D.reshape((N,1)) ))
            
            #relaxation row
            relax_S = np.sum( (X_S.T * W).T , axis=0 )
            relax = np.hstack((relax_F, relax_S, N))
            
            #make real
            DD_real = np.hstack(( D.real, D.imag, 0 ))
            HH_real = np.vstack(( HH.real, HH.imag, relax.real))
            
            #QR decomposition
            Q, R = np.linalg.qr(HH_real, mode="reduced")
            
            #submatrix for residues of sigma
            Q2  = Q[:  , nr:]
            R22 = R[nr:, nr:]
            
            #projected data vector
            y = np.dot(Q2.T, DD_real).flatten()
            
            #assemble
            if k==0:
                yy = y
                RR = R22
            else:
                yy = np.hstack((yy, y))
                RR = np.vstack((RR, R22))
        
        #normalize to improve conditioning
        _RR_max = RR.max(axis=0)
        _RR_max = np.where(_RR_max == 0, 1, _RR_max)
        _yy_max = max( yy.max(), 1)
        
        RR_norm = RR / _RR_max / _yy_max
        yy_norm = yy / _yy_max

        #compute least squares fit of residues for Sigma
        res_norm, *_ = np.linalg.lstsq(RR_norm, yy_norm, rcond=None)
        
        #renormalize results
        res = res_norm / _RR_max
        
        #decompose result into residues and relaxation constant
        self.Residues_Sig_real, res_cpx, dd = np.split( res, [self.n_real, nr-ncd] )
        self.Residues_Sig_cpx = res_cpx[:self.n_cpx] + 1j * res_cpx[self.n_cpx:]
        
        self.d_relax = dd + 1
    
    
    def _compute_residues_F_fast_relax(self):
        
        """
        compute residues of function F via individual blocks
        """
        
        #some dimensions
        N, m, n = self.Data.shape
        nf = m * n
        
        #number of residues, const and diff variables
        ncd = int(self.fit_Const) + int(self.fit_Diff) + int(self.fit_Zero)
        
        nr = ncd + self.n_real + 2*self.n_cpx
        
        #reshape data
        D_flat = self.Data.reshape((N, nf)).T
        
        #build blockmatrix with partial fractions of F
        X_F, _ = self._build_X()
        
        #make real
        XX = np.vstack((X_F.real, X_F.imag))
        
        #build blocks
        for k, D in enumerate(D_flat):
            
            #make real
            DD = np.hstack((D.real, D.imag))
            
            #solve lstsq for resiues of func k
            r, *_ = np.linalg.lstsq(XX, DD, rcond=None)
            
            if k == 0:
                res = r.flatten()
            else:
                res = np.hstack(( res, r.flatten() ))
            
        #reshape results
        R_F = res.reshape((nf, nr)).T.reshape((nr, m, n))
            
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

            #perform fitting iteration according to selected mode
            self._compute_residues_Sigma_fast_relax()

            #discard poles
            if self.autoreduce and self.step > 0:
                self._reduce_order(tol)
        
            self._compute_poles()

            self._enforce_stability()

            self._compute_residues_F_fast_relax()

            self._update_TF()

            #compute error
            self.err_max, self.err_mean = self._evaluate_fit()

            if debug: self._debug()

            if self.err_max < tol:
                return self.TF

        #directly return transfer funnction object
        return self.TF