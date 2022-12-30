#########################################################################################
##
##                              VECTORFITTING ALGORITHM
##
##                                   Milan Rother
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np

from tools import (
    timer,
    find_local_maxima
    )

from transferfunction import (
    TransferFunction
    )


    
# MATRIX VECFIT ====================================================================

class VecFit:
    
    """
    relaxed and non relaxed implementation 
    of vectorfitting algorithm for mimo transfer functions
    (with fast implementation of multiport residue calculation via QR)
    
    see:
        RATIONAL APPROXIMATION OF FREQUENCY DOMAIN 
        RESPONSES BY VECTOR FITTING (Gustavsen, 1999)
        
        Improving the Pole Relocating Properties of 
        Vector Fitting (Gustavsen, 2006)
        
        Macromodeling of Multiport Systems Using a Fast Implementation 
        of the Vector Fitting Method (Deschrijver, 2008)
    """
    
    def __init__(self, Data, Freq, n_cpx=2, n_real=1, mode="normal", smart=False, autoreduce=False, fit_Const=True, fit_Diff=True):
        
        """
        initialize vectorfitting engine with settings
        
        INPUTS : 
            Data       : matrix data for each frequency as numpy array
            Freq       : frequency points as numpy array
            n_cpx      : (int) number of complex starting poles
            n_real     : (int) number of real starting poles
            mode       : (str : 'normal', 'relax', 'fast_relax') fitting mode
            smart      : (bool) autodetect initial pole configuration
            autoreduce : (bool) reduce order of model when possible
            fit_Const  : (bool) fit constant of model
            fit_Diff   : (bool) fit differenciating part of model
        """
        
        #get number of samples
        N , *_ = Freq.shape
        
        #catch SISO case
        if Data.shape == Freq.shape:
            self.Data = Data.reshape((N,1,1))
        else:
            self.Data = Data
        
        #scale frequencies for better conditioning
        self.Freq_scale = np.max(Freq)
        self.Freq  = Freq / self.Freq_scale
        self.Omega = Freq / self.Freq_scale * 2 * np.pi
        
        #number of initial poles
        self.n_real = n_real
        self.n_cpx  = n_cpx
        
        #modes for fitting
        mode_funcs = {"normal"     : [self._compute_residues             , self._iteration           ], 
                      "relax"      : [self._compute_residues_relax       , self._iteration_relax     ], 
                      "fast_relax" : [self._compute_residues_F_fast_relax, self._iteration_fast_relax]
                      }
        
        #fitting functions by mode -> defaults to normal
        self._resd, self._iter = mode_funcs[mode if mode in mode_funcs else "normal"]
        
        #fitting options
        self.smart      = smart
        self.autoreduce = autoreduce
        self.fit_Const  = fit_Const
        self.fit_Diff   = fit_Diff
        
    
    def _has_TF(self):
        return hasattr(self, "TF")
    
    
    def _reduce_order(self, tol=1e-2):
        
        """
        check if some residues are small 
        compared to the others (tol)
        and then discarded them
        
        INPUTS : 
            tol : (float) tolerance for discarding poles
        """
        
        #check complex residues
        res_abs_cpx = np.mean( np.abs(self.Residues_cpx), axis=(1,2) )
        
        #find small residues
        idx_cpx = np.argwhere(res_abs_cpx / np.mean(res_abs_cpx) < tol)
        
        #discard poles and residues
        if res_abs_cpx.size > 1 and idx_cpx.size > 0:
            if hasattr(self, "Residues_Sig_cpx"):
                self.Residues_Sig_cpx = np.delete(self.Residues_Sig_cpx, idx_cpx)
            self.Residues_cpx = np.delete(self.Residues_cpx, idx_cpx)
            self.Poles_cpx    = np.delete(self.Poles_cpx, idx_cpx)
            self.n_cpx -= idx_cpx.size
            print(f"n_cpx: {self.n_cpx+idx_cpx.size} -> {self.n_cpx}")
            
        #check real residues
        res_abs_real = np.mean( np.abs(self.Residues_real), axis=(1,2) )
        
        #find small residues
        idx_real = np.argwhere(res_abs_real / np.mean(res_abs_real) < tol)
        
        #discard poles and residues
        if res_abs_real.size > 1 and idx_real.size > 0:
            if hasattr(self, "Residues_Sig_real"):
                self.Residues_Sig_real = np.delete(self.Residues_Sig_real, idx_real)
            self.Residues_real = np.delete(self.Residues_real, idx_real)
            self.Poles_real    = np.delete(self.Poles_real, idx_real)
            self.n_real -= idx_real.size
            print(f"n_real: {self.n_real+idx_real.size} -> {self.n_real}")
    
    
    def _set_poles(self):
        
        """
        generate initial estimate for poles
        (based on extrema in data if smart)
        """
        
        omega_max = max(self.Omega)
        
        if self.smart:
            
            #process data
            Magnitude = abs( np.prod(self.Data, axis=(1,2)) )
            
            #find all local maxima in magnitude of data
            maxima = find_local_maxima(Magnitude, self.Omega)
            
            #allocate complex poles near maxima
            self.n_cpx     = len(maxima)
            self.Poles_cpx = maxima * (- 1/100 +  1j)
            
            #evaluate phase, first unwrap
            Phase = np.unwrap(np.angle(self.Data, deg=True), 180)
            
            #number of 90 degree phase transitions
            transitions = int( max( (np.abs(Phase - Phase[0]) / 90).flatten() ) )
            
            #real poles
            self.n_real     = max(transitions - len(maxima), 1)
            self.Poles_real = -np.linspace(1 , omega_max, self.n_real)/50 + 0j
            
        else:
            #equally distributed poles
            self.Poles_cpx  = -np.linspace(omega_max/100, omega_max, self.n_cpx) * (1/100 + 1j)
            self.Poles_real = -np.linspace(omega_max/75 , omega_max, self.n_real) * ( 1/50 + 0j)
        
        
    def _build_X(self):
        
        """
        build matrix block X, containing 
        the partial fractions of the fitted 
        function F for the least squares problem
        """
        
        if self.fit_Const and self.fit_Diff:
            pp = lambda w : [ 1 , 1j*w ] 
        elif self.fit_Const:
            pp = lambda w : [ 1 ] 
        elif self.fit_Diff:
            pp = lambda w : [ 1j*w ] 
        else:
            pp = lambda w : [ ] 
        
        pr   = lambda w : [ 1/(1j * w - p) for p in self.Poles_real ]
        pc_r = lambda w : [ 1/(1j * w - p) + 1/(1j * w - p.conj()) for p in self.Poles_cpx ]
        pc_i = lambda w : [ 1j/(1j * w - p) - 1j/(1j * w - p.conj()) for p in self.Poles_cpx ]
        
        #build X with partial fractions of F and Sigma
        X_F = np.array([ pp(w) + pr(w) + pc_r(w) + pc_i(w) for w in self.Omega ])
        X_S = np.array([ pr(w) + pc_r(w) + pc_i(w) for w in self.Omega ])
        
        return X_F, X_S
        
        
    def _compute_A(self):
        
        """
        build matrix A for least squares calculation 
        of the residues
        """
        
        #some dimensions
        N, m, n = self.Data.shape
        nf = m * n
        
        #reshape data
        D_flat = self.Data.reshape((N, nf)).T
        
        #build blockmatrix with partial fractions of F
        X_F, X_S = self._build_X()
        
        N, nr = X_F.shape
        
        #dummy matrix for padding
        Z = lambda n : np.zeros(( N, nr * n), dtype="complex128")
        
        #build blocks
        for k, D in enumerate(D_flat):
            
            #build blockmatrix for residues of Sigma
            FX_S = - (X_S.T * D).T
            
            #build subblock
            A_k = np.hstack(( Z(k), X_F, Z(nf-(k+1)), FX_S ))
            
            if k == 0:
                A = A_k
            else:
                A = np.vstack((A, A_k))
        
        return A
    
    
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
    
    
    def _compute_residues(self):
        
        """
        compute new residues from poles and data 
        via least squares fit
        """
        
        #some dimensions
        N, m, n = self.Data.shape
        
        #number of io relations
        nf = m * n
        
        #number of residues, const and diff variables
        ncd = 0
        if self.fit_Const:
            ncd += 1
        if self.fit_Diff:
            ncd += 1
        
        nr = ncd + self.n_real + 2*self.n_cpx
        
        #reformat matrix A
        A  = self._compute_A()
        AA = np.vstack((A.real, A.imag))
        
        #reformat data and add relaxation row
        F = self.Data.reshape((N, nf)).T.flatten()
        FF = np.hstack((F.real, F.imag)).reshape((2*F.size, 1))
        
        #solve least squares problem
        R, *_ = np.linalg.lstsq(AA, FF, rcond=None)
        
        #flatten results
        R = R.flatten()
        
        #reshape results
        R_F = R[:nr*nf].reshape((nf, nr)).T.reshape((nr, m, n))
        R_S = R[nr*nf:]
        
        #const and diff terms
        if self.fit_Const and self.fit_Diff:
            self.Const = R_F[0]
            self.Diff  = R_F[1]
        elif self.fit_Const:
            self.Const = R_F[0]
            self.Diff  = 0
        elif self.fit_Diff:
            self.Const = 0
            self.Diff  = R_F[0]
        else:
            self.Const = 0
            self.Diff  = 0
        
        #handle real residues of F
        self.Residues_real = R_F[ncd:self.n_real+ncd]
        
        #handle complex residues of F
        res_cpx  = R_F[self.n_real+ncd:]
        self.Residues_cpx = res_cpx[:self.n_cpx] + 1j * res_cpx[self.n_cpx:]
        
        # real residues of Sigma
        self.Residues_Sig_real = R_S[:self.n_real]
        
        #complex residues of Sigma
        res_cpx  = R_S[self.n_real:]
        self.Residues_Sig_cpx  = res_cpx[:self.n_cpx] + 1j * res_cpx[self.n_cpx:]
        
        
    def _compute_residues_relax(self):
        
        """
        compute new residues from poles 
        and data via least squares fit
        """
        
        #some dimensions
        N, m, n = self.Data.shape
        
        #number of io relations
        nf = m * n
        
        #number of residues, const and diff variables
        ncd = 0
        if self.fit_Const:
            ncd += 1
        if self.fit_Diff:
            ncd += 1
        
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
        
        #const and diff terms
        if self.fit_Const and self.fit_Diff:
            self.Const = R_F[0]
            self.Diff  = R_F[1]
        elif self.fit_Const:
            self.Const = R_F[0]
            self.Diff  = 0
        elif self.fit_Diff:
            self.Const = 0
            self.Diff  = R_F[0]
        else:
            self.Const = 0
            self.Diff  = 0
        
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
        
        
    def _compute_residues_Sigma_fast_relax(self):
        
        """
        compute residues and relaxation constant with 
        relaxation, fast implementation with block qr
        """
        
        #some dimensions
        N, m, n = self.Data.shape
        nf = m * n
        
        #number of residues, const and diff variables
        ncd = 0
        if self.fit_Const:
            ncd += 1
        if self.fit_Diff:
            ncd += 1
        
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
        self.Residues_Sig_real , res_cpx, dd = np.split( res, [self.n_real, nr-ncd] )
        self.Residues_Sig_cpx  = res_cpx[:self.n_cpx] + 1j * res_cpx[self.n_cpx:]
        
        self.d_relax = dd + 1
    
    
    def _compute_residues_F_fast_relax(self):
        
        """
        compute residues of function F via individual blocks
        """
        
        #some dimensions
        N, m, n = self.Data.shape
        nf = m * n
        
        #number of residues, const and diff variables
        ncd = 0
        if self.fit_Const:
            ncd += 1
        if self.fit_Diff:
            ncd += 1
        
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
            
        #const and diff terms
        if self.fit_Const and self.fit_Diff:
            self.Const = R_F[0]
            self.Diff  = R_F[1]
        elif self.fit_Const:
            self.Const = R_F[0]
            self.Diff  = 0
        elif self.fit_Diff:
            self.Const = 0
            self.Diff  = R_F[0]
        else:
            self.Const = 0
            self.Diff  = 0
        
        #handle real residues of F
        self.Residues_real = R_F[ncd:self.n_real+ncd]
        
        #handle complex residues of F
        res_cpx = R_F[self.n_real+ncd:]
        self.Residues_cpx = res_cpx[:self.n_cpx] + 1j * res_cpx[self.n_cpx:]
        
        
    def _compute_poles(self):
        
        """
        compute new poles as zeros 
        of weighing function Sigma
        
        with optional relaxation 
        for pole relocation
        """
        
        #build companion matrix from poles
        A = np.zeros(( self.n_real + 2*self.n_cpx, self.n_real + 2*self.n_cpx ))
        A[:self.n_real, :self.n_real] = np.diag(self.Poles_real).real
        
        for i, p in enumerate(self.Poles_cpx):
            
            j = 2*i + self.n_real
            
            A[j, j]     =  p.real
            A[j+1, j+1] =  p.real
            A[j, j+1]   =  p.imag
            A[j+1, j]   = -p.imag
            
        #mapping of residues
        b = np.zeros(self.n_real+ 2*self.n_cpx)
        b[:self.n_real] = 1
        b[self.n_real::2] = 2
        b = b.reshape((self.n_real+ 2*self.n_cpx,1))
        
        #process residues
        r_cpx = np.vstack((self.Residues_Sig_cpx.real , self.Residues_Sig_cpx.imag)).T.flatten()
        r = np.hstack((self.Residues_Sig_real, r_cpx)).reshape((1, self.n_real+ 2*self.n_cpx))
        
        #check if relaxation available
        if hasattr(self, "d_relax"):
            #build residue mapping matrix with relaxation
            B = b.dot(r) / self.d_relax
        else:
            #build residue mapping matrix without relaxation
            B = b.dot(r) 
        
        #assemble eigenvalue problem
        H = A - B 
        
        #compute new poles
        Poles = np.linalg.eigvals(H)
        
        #discard small imaginary parts
        i_tol = np.max(self.Omega) * 1e-12
        Poles = np.where(abs(Poles.imag) < i_tol, Poles.real, Poles )
        
        #sort new poles
        self.Poles_real =  Poles[np.argwhere(Poles.imag == 0.0)].flatten()
        self.Poles_cpx  =  Poles[np.argwhere(Poles.imag >  0.0)].flatten()
        
        #update sizes
        self.n_real = self.Poles_real.size
        self.n_cpx  = self.Poles_cpx.size
        
        
    def _enforce_stability(self):
        
        """
        enforce stability by flipping poles 
        with positive real part
        """
        
        self.Poles_real = - np.abs(self.Poles_real) + 0j
        self.Poles_cpx  = - np.abs(self.Poles_cpx.real) + 1j * self.Poles_cpx.imag
    
    
    # @timer
    def _iteration(self):
        
        """
        perform one iteration of 
        the fitting procedure
        """
        
        #discard poles
        if self.autoreduce:
            self._reduce_order(tol=1e-2)
        
        self._compute_poles()
        self._enforce_stability()
        self._compute_residues()
        self._update_TF()
        
        
    # @timer
    def _iteration_relax(self):
        
        """
        perform one iteration of 
        the relaxed fitting procedure
        """
        
        #discard poles
        if self.autoreduce:
            self._reduce_order(tol=1e-2)
        
        self._compute_poles()
        self._enforce_stability()
        self._compute_residues_relax()
        self._update_TF()
        
        
    # @timer
    def _iteration_fast_relax(self):
        
        """
        perform one iteration of 
        the fast relaxed fitting procedure
        """
        
        self._compute_residues_Sigma_fast_relax()
        
        #discard poles
        if self.autoreduce:
            self._reduce_order(tol=1e-2)
            
        self._compute_poles()
        self._enforce_stability()
        self._compute_residues_F_fast_relax()
        self._update_TF()
        
        
    def _evaluate_fit(self):
        
        """
        evaluate the fit by computing 
        the relative error
        """
        
        #check if transferfunction available
        if not self._has_TF():
            self._update_TF()
        
        #evaluate fit as rational function
        D_fit = self.TF.evaluate(self.Freq * self.Freq_scale)
        
        #compute relative error
        err = abs((self.Data - D_fit) / self.Data)
        
        #processing
        err_max  = np.max(err)
        err_mean = np.mean(err)
        
        return err_max, err_mean
    
    
    def _update_TF(self):
        
        """
        update internal representation of transfer function
        -> rescale frequencies
        """
        
        #build conjugate pairs
        Poles_cpx    = sum( ([p, p.conj()] for p in self.Poles_cpx), [] )
        Residues_cpx = sum( ([R, R.conj()] for R in self.Residues_cpx), [] )
        
        #build all poles
        Poles    = np.array([ *self.Poles_real , *Poles_cpx  ])
        Residues = np.array([ *self.Residues_real, *Residues_cpx ])
        
        self.TF = TransferFunction(Poles*self.Freq_scale, 
                                   Residues*self.Freq_scale, 
                                   self.Const, 
                                   self.Diff/self.Freq_scale )
        
        
    def fit(self, tol=1e-3, max_steps=5, debug=False):
        
        """
        perform fitting procedure
        
        INPUTS :
            tol       : (float) fitting tolerance for max relative error
            max_steps : (int) maximum number of iteration steps
            debug     : (bool) print error and final model order
        """
        
        #set initial poles
        self._set_poles()
        
        #compute initial residues
        self._resd()
        
        #compute initial error
        self.err_max , self.err_mean = self._evaluate_fit()
        
        if debug:
            print("err_max  =",self.err_max)
            print("err_mean =",self.err_mean)
        
        steps = 0
        while self.err_max > tol and steps < max_steps:
            steps += 1
            
            #perform fitting iteration
            self._iter()
            
            #compute error
            self.err_max , self.err_mean = self._evaluate_fit()
            
            if debug:
                print("err_max  =",self.err_max)
                print("err_mean =",self.err_mean)
                
        if debug:
            print("n_real =", self.n_real)
            print("n_cpx  =", self.n_cpx)
