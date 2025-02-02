#########################################################################################
##
##                        BASE VERSION VECTORFITTING ALGORITHM
##
##                                   Milan Rother
##
#########################################################################################


# imports -------------------------------------------------------------------------------

import numpy as np

from .tools import (
    find_local_maxima
    )

from .parsers import read_touchstone

from .transferfunction import (
    TransferFunction
    )


# MATRIX VECFIT =========================================================================

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
    
    def __init__(self, 
                 Data, 
                 Freq, 
                 n_cpx=2, 
                 n_real=1, 
                 smart=False, 
                 autoreduce=False, 
                 fit_Const=False, 
                 fit_Diff=False,
                 fit_Zero=False
                 ):
        
        """
        initialize vectorfitting engine with settings
        
        INPUTS : 
            Data       : matrix data for each frequency as numpy array
            Freq       : frequency points as numpy array
            n_cpx      : (int) number of complex starting poles
            n_real     : (int) number of real starting poles
            smart      : (bool) autodetect initial pole configuration
            autoreduce : (bool) reduce order of model when possible
            fit_Const  : (bool) fit constant of model
            fit_Diff   : (bool) fit differenciating part of model
            fit_Zero   : (bool) fit pole at zero
        """

        #get number of samples
        N , *_ = Freq.shape
        
        #catch SISO case and expand axes
        if Data.shape == Freq.shape:
            self.Data = Data[:, np.newaxis, np.newaxis]
        else:
            self.Data = Data
        
        #scale frequencies for better conditioning
        self.Freq_scale = np.max(Freq)
        self.Freq  = Freq / self.Freq_scale
        self.Omega = Freq / self.Freq_scale * 2 * np.pi

        #get shape of data
        N, n, m = self.Data.shape
        
        #number of initial poles
        self.n_real = n_real
        self.n_cpx  = n_cpx

        #other fitting options
        self.smart      = smart
        self.autoreduce = autoreduce
        self.fit_Const  = fit_Const
        self.fit_Diff   = fit_Diff
        self.fit_Zero   = fit_Zero
        
        #setup initial model fit
        self._setup()


    @classmethod
    def from_touchstone(cls, filepath="", **kwargs):
        """
        Initiate the VF engine directy from a touchstone file.
        """
        Freq, Data, *_ = read_touchstone(filepath)
        return cls(Data, Freq, **kwargs), Data, Freq


    # checkers --------------------------------------------------------------------------
    
    def _has_TF(self):
        return hasattr(self, "TF")


    # setup -----------------------------------------------------------------------------
    
    def _set_poles(self):
        """
        generate initial estimate for poles
        (based on extrema in data if smart)
        """
        
        omega_max = max(self.Omega)
        
        if self.smart:
            
            #find all local maxima in magnitude of data
            maxima_idx = find_local_maxima(abs(self.Data))
            maxima = self.Omega[maxima_idx]

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
    

    def _setup(self):
        """
        Setup fitting procedure.
        Set initial poles, compute initial residues 
        and calculate error of initial fit
        """

        #set initial poles
        self._set_poles()
        
        #compute initial residues according to selected mode
        self._compute_residues()

        #compute initial error
        self.err_max , self.err_mean = self._evaluate_fit()


    # intermediate processing -----------------------------------------------------------

    def _reduce_order(self, tol=5e-3):
        """
        Recursively discard poles if their absence still satisfies 
        the specified relative tolerance (mean)
        
        INPUTS: 
            tol : (float) tolerance for discarding poles
        """

        #compute new transfer function
        self._update_TF()

        #current fit
        D_fit = self.TF.evaluate(self.Freq * self.Freq_scale)

        #discard real poles
        for i, (p, r) in enumerate(zip(self.Poles_real, self.Residues_real)):

            #rescale pole and residue
            _p = p * self.Freq_scale
            _r = r * self.Freq_scale

            #transfer function contribution of pole
            _TF = TransferFunction([_p], [_r])

            #evaluate contribution of pole
            R_fit = _TF.evaluate(self.Freq * self.Freq_scale)
            
            #error condition still satisfied after removal?
            err = np.mean(abs((self.Data - (D_fit - R_fit)) / self.Data))
            if err < tol:
                print(f"discarding real pole, err={err}")
                
                self.Poles_real = np.delete(self.Poles_real, i, axis=0)
                self.Residues_real = np.delete(self.Residues_real, i, axis=0)

                self.n_real -= 1

                if hasattr(self, "Residues_Sig_real"):
                    self.Residues_Sig_real = np.delete(self.Residues_Sig_real, i, axis=0)

                return self._reduce_order(tol)

        #discard complex poles
        for i, (p, r) in enumerate(zip(self.Poles_cpx, self.Residues_cpx)):

            #rescale pole and residue
            _p = p * self.Freq_scale
            _r = r * self.Freq_scale

            #transfer function contribution of pole
            _TF = TransferFunction([_p, _p.conj()], [_r, _r.conj()])

            #evaluate contribution of pole
            R_fit = _TF.evaluate(self.Freq * self.Freq_scale)
            
            #error condition still satisfied after removal?
            err = np.mean(abs((self.Data - (D_fit - R_fit)) / self.Data))
            if err < tol:
                print(f"discarding complex pole pair, err={err}")
                
                self.Poles_cpx = np.delete(self.Poles_cpx, i, axis=0)
                self.Residues_cpx = np.delete(self.Residues_cpx, i, axis=0)

                self.n_cpx -= 1
                
                if hasattr(self, "Residues_Sig_cpx"):
                    self.Residues_Sig_cpx = np.delete(self.Residues_Sig_cpx, i, axis=0)
                
                return self._reduce_order(tol)

        return None


    def _enforce_stability(self):
        
        """
        enforce stability by flipping poles 
        with positive real part
        """
        
        self.Poles_real = - np.abs(self.Poles_real) + 0j
        self.Poles_cpx  = - np.abs(self.Poles_cpx.real) + 1j * self.Poles_cpx.imag
    

    def _evaluate_fit(self):
        
        """
        evaluate the fit by computing 
        the relative error
        """
        
        #check if transferfunction available
        if not self._has_TF():
            self._update_TF()
        
        #evaluate fit as rational function
        D_fit = self.TF(self.Freq * self.Freq_scale)
        
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
        Poles_cpx    = sum(([p, p.conj()] for p in self.Poles_cpx), [])
        Residues_cpx = sum(([R, R.conj()] for R in self.Residues_cpx), [])
        
        #build all poles
        Poles    = np.array([ *self.Poles_real , *Poles_cpx  ])
        Residues = np.array([ *self.Residues_real, *Residues_cpx ])
        
        self.TF = TransferFunction(
            Poles*self.Freq_scale, 
            Residues*self.Freq_scale,
            self.Const, 
            self.Diff/self.Freq_scale,
            self.Zero*self.Freq_scale
            )


    # matrix building -------------------------------------------------------------------
        
    def _build_X(self):
        
        """
        build matrix block X, containing 
        the partial fractions of the fitted 
        function F for the least squares problem
        """
        
        if self.fit_Const and self.fit_Diff and self.fit_Zero:
            pp = lambda w : [ 1 , 1j*w, -1j/w ] 

        elif self.fit_Const and self.fit_Diff:
            pp = lambda w : [ 1 , 1j*w] 
        elif self.fit_Const and self.fit_Zero:
            pp = lambda w : [ 1 , -1j/w ] 
        elif self.fit_Diff and self.fit_Zero:
            pp = lambda w : [ 1j*w, -1j/w ]

        elif self.fit_Const:
            pp = lambda w : [ 1 ] 
        elif self.fit_Diff:
            pp = lambda w : [ 1j*w ] 
        elif self.fit_Zero:
            pp = lambda w : [ -1j/w ]     

        else:
            pp = lambda w : [ ] 
        
        #generate lists of laurent partial fractions
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


    # residue computation ---------------------------------------------------------------
    
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
        ncd = int(self.fit_Const) + int(self.fit_Diff) + int(self.fit_Zero)
        nr = ncd + self.n_real + 2*self.n_cpx
        
        #reformat matrix A
        A  = self._compute_A()
        AA = np.vstack((A.real, A.imag))
        
        #reformat data 
        F = self.Data.reshape((N, nf)).T.flatten()
        FF = np.hstack((F.real, F.imag)).reshape((2*F.size, 1))
        
        #solve least squares problem
        R, *_ = np.linalg.lstsq(AA, FF, rcond=None)

        #flatten results
        R = R.flatten()
        
        #reshape results
        R_F = R[:nr*nf].reshape((nf, nr)).T.reshape((nr, m, n))
        R_S = R[nr*nf:]
        
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
        
        # real residues of Sigma
        self.Residues_Sig_real = R_S[:self.n_real]
        
        #complex residues of Sigma
        res_cpx = R_S[self.n_real:]
        self.Residues_Sig_cpx  = res_cpx[:self.n_cpx] + 1j * res_cpx[self.n_cpx:]
        

    # pole computation ------------------------------------------------------------------
        
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
        Poles = np.where(abs(Poles.imag) < i_tol, Poles.real, Poles)
        
        #sort new poles
        self.Poles_real =  Poles[np.argwhere(Poles.imag == 0.0)].flatten()
        self.Poles_cpx  =  Poles[np.argwhere(Poles.imag >  0.0)].flatten()
        
        #update sizes
        self.n_real = self.Poles_real.size
        self.n_cpx  = self.Poles_cpx.size
        

    # debugging -------------------------------------------------------------------------

    def _debug(self):
        
        """
        print current state of VF engine for debugging
        """
        
        s  =  "debugging status : \n"
        s += f"    iteration step number  (step)          : {self.step}\n"
        s += f"    model order            (n_real, n_cpx) : {self.n_real}, {self.n_cpx}\n"
        s += f"    fitting relative error (mean, max)     : {self.err_mean}, {self.err_max}\n"

        print(s)


    # fitting call ----------------------------------------------------------------------
        
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
           
            self._compute_residues()

            self._update_TF()

            #compute error
            self.err_max, self.err_mean = self._evaluate_fit()

            if debug: self._debug()

            if self.err_max < tol:
                return self.TF

        #directly return transfer funnction object
        return self.TF