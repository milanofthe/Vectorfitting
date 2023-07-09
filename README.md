# Vectorfitting

This is a from scratch pure python implementation of the fast relaxed vectorfitting algorithm for MIMO frequency domain data. Different modes (standard VF, relaxed VF and fast relaxed VF) are implemented. Matrix shaped frequency domain data is supported, and a model with common poles is fitted.


$$
\mathbf{H}_{fit}(s) = \mathbf{D} + s \cdot \mathbf{E} + \sum_{k=1}^{n} \mathbf{R}_{k} \cdot \frac{1}{s - p_k}
$$

Here $\mathbf{D}$ is the constant term, $\mathbf{E}$ is the linear term and $\mathbf{R}_{k}$, $p_k$ are the (possibly complex) residues in matrix form and poles. 

## Example


```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

from vectorfitting import VecFit
from transferfunction import H_rng
```


```python
#create random test data
Freq = np.linspace(0, 1000, 500)
H    = H_rng(shape=(2, 2), n_cpx=8, n_real=2, f_min=0, f_max=1000).evaluate(Freq)

```


```python
#initialize vectorfitting engine
VF = VecFit(H, Freq, n_cpx=8, n_real=2, mode="fast_relax", smart=False, autoreduce=False, fit_Const=True, fit_Diff=True)

#run fitting procedure
VF.fit(tol=1e-3, max_steps=10, debug=True)
```

    debugging status : 
        iteration step number  (step)          : 0
        model order            (n_real, n_cpx) : 2, 8
        fitting relative error (mean, max)     : 0.20947220144389897, 9.873754669895343
    
    debugging status : 
        iteration step number  (step)          : 1
        model order            (n_real, n_cpx) : 4, 7
        fitting relative error (mean, max)     : 0.009325251862710026, 0.11294815333674414
    
    debugging status : 
        iteration step number  (step)          : 2
        model order            (n_real, n_cpx) : 2, 8
        fitting relative error (mean, max)     : 0.009238337677479759, 0.17829589878851784
    
    debugging status : 
        iteration step number  (step)          : 3
        model order            (n_real, n_cpx) : 2, 8
        fitting relative error (mean, max)     : 0.008941892283839993, 0.08606583008523701
    
    debugging status : 
        iteration step number  (step)          : 4
        model order            (n_real, n_cpx) : 0, 9
        fitting relative error (mean, max)     : 0.010310394416765052, 0.1334009912197929
    
    debugging status : 
        iteration step number  (step)          : 5
        model order            (n_real, n_cpx) : 0, 9
        fitting relative error (mean, max)     : 0.007934015433107247, 0.1275905688155805
    
    debugging status : 
        iteration step number  (step)          : 6
        model order            (n_real, n_cpx) : 0, 9
        fitting relative error (mean, max)     : 0.005136035831997903, 0.11904704025676377
    
    debugging status : 
        iteration step number  (step)          : 7
        model order            (n_real, n_cpx) : 0, 9
        fitting relative error (mean, max)     : 0.0030934834512960523, 0.07824607490636737
    
    debugging status : 
        iteration step number  (step)          : 8
        model order            (n_real, n_cpx) : 0, 9
        fitting relative error (mean, max)     : 0.001885113862262611, 0.05200394706314038
    
    debugging status : 
        iteration step number  (step)          : 9
        model order            (n_real, n_cpx) : 2, 8
        fitting relative error (mean, max)     : 0.0012306423246825483, 0.03610242395271164
    
    debugging status : 
        iteration step number  (step)          : 10
        model order            (n_real, n_cpx) : 2, 8
        fitting relative error (mean, max)     : 0.0008741956237063604, 0.02478883302559956
    
    


```python
#evaluate fit
H_fit = VF.TF.evaluate(Freq)

#compute relative error
err_rel = (H - H_fit) / H

#dB helper
dB  = lambda x: 20*np.log10(abs(x))

#plot results
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,4), tight_layout=True, dpi=120)

N, n, m = H.shape

for i in range(n):
    for j in range(m):
        ax.plot(Freq, dB(H[:,i,j]), "d",  color=cycle[0], markevery=15, markersize=5, label="H" if i==j==0 else None)
        ax.plot(Freq, dB(H_fit[:,i,j]), "--", color=cycle[1], lw=2, label="H_fit" if i==j==0 else None)
        ax.plot(Freq, dB(err_rel[:,i,j]), ":", color=cycle[2], lw=2, label="err_rel" if i==j==0 else None)
        

ax.set_xlabel("freq [Hz]")
ax.set_ylabel("mag in dB")
ax.grid(True)
ax.legend()

plt.savefig("test.svg")
```


![test](https://user-images.githubusercontent.com/105657697/212681435-67aa3a6e-55d8-4f22-8ee1-e6e25a6f0ed1.svg)



## References

[1] Gustavsen, B. and Adam Semlyen. “Rational approximation of frequency domain responses by vector fitting.” IEEE Transactions on Power Delivery 14 (1999): 1052-1061.

[2] B. Gustavsen, "Improving the pole relocating properties of vector fitting," in IEEE Transactions on Power Delivery, vol. 21, no. 3, pp. 1587-1592, July 2006, doi: 10.1109/TPWRD.2005.860281.

[3] D. Deschrijver, M. Mrozowski, T. Dhaene and D. De Zutter, "Macromodeling of Multiport Systems Using a Fast Implementation of the Vector Fitting Method," in IEEE Microwave and Wireless Components Letters, vol. 18, no. 6, pp. 383-385, June 2008, doi: 10.1109/LMWC.2008.922585.



