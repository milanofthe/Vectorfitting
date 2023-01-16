# Vectorfitting

This is a from scratch pure python implementation of the fast relaxed vectorfitting algorithm for MIMO frequency domain data. Different modes (standard VF, relaxed VF and fast relaxed VF) are implemented. Matrix shaped frequency domain data is supported, and a model with common poles is fitted

$$ \mathbf{H}_{fit}(s) = \mathbf{D} + s \cdot \mathbf{E} + \sum_{k=1}^{n} \mathbf{R}_{k} \cdot \frac{1}{s - p_k} $$

where $\mathbf{D}$ is the constant term, $\mathbf{E}$ is the linear term and $\mathbf{R}_{k}$, $p_k$ are the (possibly complex) residues in matrix form and poles. 

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
VF.fit(tol=1e-3, max_steps=0, debug=True)
```

    err_max  = 10.783463188560583
    err_mean = 0.4183956352192532
    err_max  = 0.030505900483405884
    err_mean = 0.0029563663721864247
    err_max  = 0.017597388099753444
    err_mean = 0.0007699885912381359
    err_max  = 0.018231845749601364
    err_mean = 0.0007665771706315633
    err_max  = 0.018202742046339117
    err_mean = 0.0007659174051087016
    err_max  = 0.01818766973278153
    err_mean = 0.0007652731230669731
    err_max  = 0.018172290370446202
    err_mean = 0.0007646294887398828
    err_max  = 0.018156919896438996
    err_mean = 0.0007639860439388242
    err_max  = 0.0181415519195631
    err_mean = 0.0007633427826562838
    err_max  = 0.018126186514793554
    err_mean = 0.0007626997020003051
    err_max  = 0.018110823618152957
    err_mean = 0.0007620567991713711
    err_max  = 0.018095463167374762
    err_mean = 0.0007614140713279675
    err_max  = 0.01808010509887936
    err_mean = 0.0007607715155779249
    err_max  = 0.018064749348039488
    err_mean = 0.0007601291289795888
    err_max  = 0.0180493958487832
    err_mean = 0.0007594869085386267
    err_max  = 0.018034044533908094
    err_mean = 0.0007588448512110868
    err_max  = 0.018018695334889808
    err_mean = 0.00075820295389819
    err_max  = 0.018003348181847244
    err_mean = 0.0007575612134424022
    err_max  = 0.017988003003457105
    err_mean = 0.0007569196266336888
    err_max  = 0.017972659727199077
    err_mean = 0.0007562781902041478
    err_max  = 0.017957318278930826
    err_mean = 0.0007556369008220201
    n_real = 2
    n_cpx  = 8
    


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


    
![png](README_files/README_5_0.png)
    


## References

[1] Gustavsen, B. and Adam Semlyen. “Rational approximation of frequency domain responses by vector fitting.” IEEE Transactions on Power Delivery 14 (1999): 1052-1061.

[2] B. Gustavsen, "Improving the pole relocating properties of vector fitting," in IEEE Transactions on Power Delivery, vol. 21, no. 3, pp. 1587-1592, July 2006, doi: 10.1109/TPWRD.2005.860281.

[3] D. Deschrijver, M. Mrozowski, T. Dhaene and D. De Zutter, "Macromodeling of Multiport Systems Using a Fast Implementation of the Vector Fitting Method," in IEEE Microwave and Wireless Components Letters, vol. 18, no. 6, pp. 383-385, June 2008, doi: 10.1109/LMWC.2008.922585.


