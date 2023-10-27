![test_poles](https://github.com/milanofthe/Vectorfitting/assets/105657697/84d7f38a-5c29-4fa4-92f3-de38c4d8140a)# Vectorfitting

This is a from scratch pure python implementation of the fast relaxed vectorfitting algorithm for MIMO frequency domain data. Different modes (standard VF, relaxed VF and fast relaxed VF) are implemented. Matrix shaped frequency domain data is supported, and a model with common poles is fitted

```math
\mathbf{H}_{fit}(s) = \mathbf{D} + s \cdot \mathbf{E} + \sum_{k=1}^{n} \mathbf{R}_{k} \cdot \frac{1}{s - p_k}
```

where $\mathbf{D}$ is the constant term, $\mathbf{E}$ is the linear term and $\mathbf{R}_{k}$, $p_k$ are the (possibly complex) residues in matrix form and poles. 

## Example


```python
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

from vectorfitting import VecFit
from parsers import read_touchstone
```


```python
#load data from snp file
Freq, H, *_ = read_touchstone(r"example_data/3port.s3p")
```


```python
#initialize vectorfitting engine
VF = VecFit(H, Freq, n_cpx=5, n_real=1, mode="fast_relax", smart=False, autoreduce=True, fit_Const=True, fit_Diff=False)

#run fitting procedure
VF.fit(tol=1e-5, max_steps=15, debug=True)
```

    debugging status : 
        iteration step number  (step)          : 0
        model order            (n_real, n_cpx) : 1, 5
        fitting relative error (mean, max)     : 0.49136901566282815, 2.657850202535424
    
    debugging status : 
        iteration step number  (step)          : 1
        model order            (n_real, n_cpx) : 3, 4
        fitting relative error (mean, max)     : 0.0005314484496944583, 0.0037071766858198726
    
    n_cpx: 4 -> 3
    n_real: 3 -> 1
    debugging status : 
        iteration step number  (step)          : 2
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006018634849112722, 0.009863669001006633
    
    debugging status : 
        iteration step number  (step)          : 3
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006076050098630738, 0.00891959739913526
    
    debugging status : 
        iteration step number  (step)          : 4
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006076550011962279, 0.008918540946273213
    
    debugging status : 
        iteration step number  (step)          : 5
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006076321701213752, 0.008918566425843944
    
    debugging status : 
        iteration step number  (step)          : 6
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006076146740866635, 0.008918478457277485
    
    debugging status : 
        iteration step number  (step)          : 7
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006075996265262533, 0.008918338497520125
    
    debugging status : 
        iteration step number  (step)          : 8
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006075856683255433, 0.008918175406856925
    
    debugging status : 
        iteration step number  (step)          : 9
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006075721968997722, 0.008918002032898739
    
    debugging status : 
        iteration step number  (step)          : 10
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.000607558944599357, 0.00891782409279443
    
    debugging status : 
        iteration step number  (step)          : 11
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006075457924386948, 0.008917644130316479
    
    debugging status : 
        iteration step number  (step)          : 12
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006075326875050087, 0.008917463277103225
    
    debugging status : 
        iteration step number  (step)          : 13
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006075196062665287, 0.008917282036525207
    
    debugging status : 
        iteration step number  (step)          : 14
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006075065382574725, 0.008917100632443484
    
    debugging status : 
        iteration step number  (step)          : 15
        model order            (n_real, n_cpx) : 1, 3
        fitting relative error (mean, max)     : 0.0006074934788237583, 0.008916919164409773
    
    


```python
#evaluate fit
H_fit = VF.TF.evaluate(Freq)

#compute relative error
err_rel = (H - H_fit) / H

#dB helper
dB  = lambda x: 20*np.log10(abs(x))

#plot results
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), tight_layout=True, dpi=150)

N, n, m = H.shape

for i in range(n):
    for j in range(m):
        ax.plot(Freq, dB(H[:,i,j]), "-d",  color=cycle[0], markevery=15, markersize=5, label="H" if i==j==0 else None)
        ax.plot(Freq, dB(H_fit[:,i,j]), "--", color=cycle[1], lw=2, label="H_fit" if i==j==0 else None)
        ax.plot(Freq, dB(err_rel[:,i,j]), ":", color=cycle[2], lw=2, label="err_rel" if i==j==0 else None)
        

ax.set_xlabel("freq [Hz]")
ax.set_ylabel("mag in dB")
ax.grid(True)
ax.legend(loc="lower right")

plt.savefig("test.svg")
```


![test](https://github.com/milanofthe/Vectorfitting/assets/105657697/71675237-b3d0-4481-be57-3c15e4b8d104)


```python
#evaluate location of poles
poles = VF.TF.Poles

print("poles :", poles)

#plot results
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6), tight_layout=True, dpi=150)

ax.scatter(poles.real, poles.imag, marker="x", s=60, color=cycle[0], label="poles")

ax.set_xlabel("Real")
ax.set_ylabel("Imag")
ax.grid(True)
ax.legend(loc="lower right")

fig.savefig("test_poles.svg")

```

    poles : [-1.82950411e+11+0.00000000e+00j -3.69223122e+10+6.37578277e+11j
     -3.69223122e+10-6.37578277e+11j -4.08834650e+11+7.83778848e+11j
     -4.08834650e+11-7.83778848e+11j -4.28507940e+11+4.25829589e+11j
     -4.28507940e+11-4.25829589e+11j]
    

![Upload<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns:xlink="http://www.w3.org/1999/xlink" width="432pt" height="432pt" viewBox="0 0 432 432" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2023-10-27T11:24:42.917161</dc:date>
    <dc:format>image/svg+xml</dc:format>
    <dc:creator>
     <cc:Agent>
      <dc:title>Matplotlib v3.6.1, https://matplotlib.org/</dc:title>
     </cc:Agent>
    </dc:creator>
   </cc:Work>
  </rdf:RDF>
 </metadata>
 <defs>
  <style type="text/css">*{stroke-linejoin: round; stroke-linecap: butt}</style>
 </defs>
 <g id="figure_1">
  <g id="patch_1">
   <path d="M 0 432 
L 432 432 
L 432 0 
L 0 0 
z
"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 46.220312 390.84375 
L 421.2 390.84375 
L 421.2 21.398438 
L 46.220312 21.398438 
z
"/>
   </g>
   <g id="PathCollection_1">
    <defs>
     <path id="mdc8df0de17" d="M -3.872983 3.872983 
L 3.872983 -3.872983 
M -3.872983 -3.872983 
L 3.872983 3.872983 
" style="stroke: #8dd3c7; stroke-width: 1.5"/>
    </defs>
    <g clip-path="url(#p20336d2121)">
     <use xlink:href="#mdc8df0de17" x="277.032289" y="206.121094" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-width: 1.5"/>
     <use xlink:href="#mdc8df0de17" x="404.155469" y="69.515824" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-width: 1.5"/>
     <use xlink:href="#mdc8df0de17" x="404.155469" y="342.726363" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-width: 1.5"/>
     <use xlink:href="#mdc8df0de17" x="80.391213" y="38.191406" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-width: 1.5"/>
     <use xlink:href="#mdc8df0de17" x="80.391213" y="374.050781" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-width: 1.5"/>
     <use xlink:href="#mdc8df0de17" x="63.264844" y="114.88435" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-width: 1.5"/>
     <use xlink:href="#mdc8df0de17" x="63.264844" y="297.357837" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-width: 1.5"/>
    </g>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <path d="M 88.082122 390.84375 
L 88.082122 21.398438 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_2">
      <defs>
       <path id="me59479bbd6" d="M 0 0 
L 0 3.5 
" style="stroke: #ffffff; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#me59479bbd6" x="88.082122" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_1">
      <!-- −4.0 -->
      <g style="fill: #ffffff" transform="translate(75.940715 405.442187) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-2212" d="M 678 2272 
L 4684 2272 
L 4684 1741 
L 678 1741 
L 678 2272 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-34" d="M 2419 4116 
L 825 1625 
L 2419 1625 
L 2419 4116 
z
M 2253 4666 
L 3047 4666 
L 3047 1625 
L 3713 1625 
L 3713 1100 
L 3047 1100 
L 3047 0 
L 2419 0 
L 2419 1100 
L 313 1100 
L 313 1709 
L 2253 4666 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-2e" d="M 684 794 
L 1344 794 
L 1344 0 
L 684 0 
L 684 794 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-30" d="M 2034 4250 
Q 1547 4250 1301 3770 
Q 1056 3291 1056 2328 
Q 1056 1369 1301 889 
Q 1547 409 2034 409 
Q 2525 409 2770 889 
Q 3016 1369 3016 2328 
Q 3016 3291 2770 3770 
Q 2525 4250 2034 4250 
z
M 2034 4750 
Q 2819 4750 3233 4129 
Q 3647 3509 3647 2328 
Q 3647 1150 3233 529 
Q 2819 -91 2034 -91 
Q 1250 -91 836 529 
Q 422 1150 422 2328 
Q 422 3509 836 4129 
Q 1250 4750 2034 4750 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-34" x="83.789062"/>
       <use xlink:href="#DejaVuSans-2e" x="147.412109"/>
       <use xlink:href="#DejaVuSans-30" x="179.199219"/>
      </g>
     </g>
    </g>
    <g id="xtick_2">
     <g id="line2d_3">
      <path d="M 131.60908 390.84375 
L 131.60908 21.398438 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_4">
      <g>
       <use xlink:href="#me59479bbd6" x="131.60908" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_2">
      <!-- −3.5 -->
      <g style="fill: #ffffff" transform="translate(119.467674 405.442187) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-33" d="M 2597 2516 
Q 3050 2419 3304 2112 
Q 3559 1806 3559 1356 
Q 3559 666 3084 287 
Q 2609 -91 1734 -91 
Q 1441 -91 1130 -33 
Q 819 25 488 141 
L 488 750 
Q 750 597 1062 519 
Q 1375 441 1716 441 
Q 2309 441 2620 675 
Q 2931 909 2931 1356 
Q 2931 1769 2642 2001 
Q 2353 2234 1838 2234 
L 1294 2234 
L 1294 2753 
L 1863 2753 
Q 2328 2753 2575 2939 
Q 2822 3125 2822 3475 
Q 2822 3834 2567 4026 
Q 2313 4219 1838 4219 
Q 1578 4219 1281 4162 
Q 984 4106 628 3988 
L 628 4550 
Q 988 4650 1302 4700 
Q 1616 4750 1894 4750 
Q 2613 4750 3031 4423 
Q 3450 4097 3450 3541 
Q 3450 3153 3228 2886 
Q 3006 2619 2597 2516 
z
" transform="scale(0.015625)"/>
        <path id="DejaVuSans-35" d="M 691 4666 
L 3169 4666 
L 3169 4134 
L 1269 4134 
L 1269 2991 
Q 1406 3038 1543 3061 
Q 1681 3084 1819 3084 
Q 2600 3084 3056 2656 
Q 3513 2228 3513 1497 
Q 3513 744 3044 326 
Q 2575 -91 1722 -91 
Q 1428 -91 1123 -41 
Q 819 9 494 109 
L 494 744 
Q 775 591 1075 516 
Q 1375 441 1709 441 
Q 2250 441 2565 725 
Q 2881 1009 2881 1497 
Q 2881 1984 2565 2268 
Q 2250 2553 1709 2553 
Q 1456 2553 1204 2497 
Q 953 2441 691 2322 
L 691 4666 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-33" x="83.789062"/>
       <use xlink:href="#DejaVuSans-2e" x="147.412109"/>
       <use xlink:href="#DejaVuSans-35" x="179.199219"/>
      </g>
     </g>
    </g>
    <g id="xtick_3">
     <g id="line2d_5">
      <path d="M 175.136038 390.84375 
L 175.136038 21.398438 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_6">
      <g>
       <use xlink:href="#me59479bbd6" x="175.136038" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_3">
      <!-- −3.0 -->
      <g style="fill: #ffffff" transform="translate(162.994632 405.442187) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-33" x="83.789062"/>
       <use xlink:href="#DejaVuSans-2e" x="147.412109"/>
       <use xlink:href="#DejaVuSans-30" x="179.199219"/>
      </g>
     </g>
    </g>
    <g id="xtick_4">
     <g id="line2d_7">
      <path d="M 218.662996 390.84375 
L 218.662996 21.398438 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_8">
      <g>
       <use xlink:href="#me59479bbd6" x="218.662996" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_4">
      <!-- −2.5 -->
      <g style="fill: #ffffff" transform="translate(206.52159 405.442187) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-32" d="M 1228 531 
L 3431 531 
L 3431 0 
L 469 0 
L 469 531 
Q 828 903 1448 1529 
Q 2069 2156 2228 2338 
Q 2531 2678 2651 2914 
Q 2772 3150 2772 3378 
Q 2772 3750 2511 3984 
Q 2250 4219 1831 4219 
Q 1534 4219 1204 4116 
Q 875 4013 500 3803 
L 500 4441 
Q 881 4594 1212 4672 
Q 1544 4750 1819 4750 
Q 2544 4750 2975 4387 
Q 3406 4025 3406 3419 
Q 3406 3131 3298 2873 
Q 3191 2616 2906 2266 
Q 2828 2175 2409 1742 
Q 1991 1309 1228 531 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-32" x="83.789062"/>
       <use xlink:href="#DejaVuSans-2e" x="147.412109"/>
       <use xlink:href="#DejaVuSans-35" x="179.199219"/>
      </g>
     </g>
    </g>
    <g id="xtick_5">
     <g id="line2d_9">
      <path d="M 262.189955 390.84375 
L 262.189955 21.398438 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_10">
      <g>
       <use xlink:href="#me59479bbd6" x="262.189955" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_5">
      <!-- −2.0 -->
      <g style="fill: #ffffff" transform="translate(250.048548 405.442187) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-32" x="83.789062"/>
       <use xlink:href="#DejaVuSans-2e" x="147.412109"/>
       <use xlink:href="#DejaVuSans-30" x="179.199219"/>
      </g>
     </g>
    </g>
    <g id="xtick_6">
     <g id="line2d_11">
      <path d="M 305.716913 390.84375 
L 305.716913 21.398438 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_12">
      <g>
       <use xlink:href="#me59479bbd6" x="305.716913" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_6">
      <!-- −1.5 -->
      <g style="fill: #ffffff" transform="translate(293.575507 405.442187) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-31" d="M 794 531 
L 1825 531 
L 1825 4091 
L 703 3866 
L 703 4441 
L 1819 4666 
L 2450 4666 
L 2450 531 
L 3481 531 
L 3481 0 
L 794 0 
L 794 531 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-31" x="83.789062"/>
       <use xlink:href="#DejaVuSans-2e" x="147.412109"/>
       <use xlink:href="#DejaVuSans-35" x="179.199219"/>
      </g>
     </g>
    </g>
    <g id="xtick_7">
     <g id="line2d_13">
      <path d="M 349.243871 390.84375 
L 349.243871 21.398438 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_14">
      <g>
       <use xlink:href="#me59479bbd6" x="349.243871" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_7">
      <!-- −1.0 -->
      <g style="fill: #ffffff" transform="translate(337.102465 405.442187) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-31" x="83.789062"/>
       <use xlink:href="#DejaVuSans-2e" x="147.412109"/>
       <use xlink:href="#DejaVuSans-30" x="179.199219"/>
      </g>
     </g>
    </g>
    <g id="xtick_8">
     <g id="line2d_15">
      <path d="M 392.770829 390.84375 
L 392.770829 21.398438 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_16">
      <g>
       <use xlink:href="#me59479bbd6" x="392.770829" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_8">
      <!-- −0.5 -->
      <g style="fill: #ffffff" transform="translate(380.629423 405.442187) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-30" x="83.789062"/>
       <use xlink:href="#DejaVuSans-2e" x="147.412109"/>
       <use xlink:href="#DejaVuSans-35" x="179.199219"/>
      </g>
     </g>
    </g>
    <g id="text_9">
     <!-- Real -->
     <g style="fill: #ffffff" transform="translate(222.93125 419.120313) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-52" d="M 2841 2188 
Q 3044 2119 3236 1894 
Q 3428 1669 3622 1275 
L 4263 0 
L 3584 0 
L 2988 1197 
Q 2756 1666 2539 1819 
Q 2322 1972 1947 1972 
L 1259 1972 
L 1259 0 
L 628 0 
L 628 4666 
L 2053 4666 
Q 2853 4666 3247 4331 
Q 3641 3997 3641 3322 
Q 3641 2881 3436 2590 
Q 3231 2300 2841 2188 
z
M 1259 4147 
L 1259 2491 
L 2053 2491 
Q 2509 2491 2742 2702 
Q 2975 2913 2975 3322 
Q 2975 3731 2742 3939 
Q 2509 4147 2053 4147 
L 1259 4147 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-65" d="M 3597 1894 
L 3597 1613 
L 953 1613 
Q 991 1019 1311 708 
Q 1631 397 2203 397 
Q 2534 397 2845 478 
Q 3156 559 3463 722 
L 3463 178 
Q 3153 47 2828 -22 
Q 2503 -91 2169 -91 
Q 1331 -91 842 396 
Q 353 884 353 1716 
Q 353 2575 817 3079 
Q 1281 3584 2069 3584 
Q 2775 3584 3186 3129 
Q 3597 2675 3597 1894 
z
M 3022 2063 
Q 3016 2534 2758 2815 
Q 2500 3097 2075 3097 
Q 1594 3097 1305 2825 
Q 1016 2553 972 2059 
L 3022 2063 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-61" d="M 2194 1759 
Q 1497 1759 1228 1600 
Q 959 1441 959 1056 
Q 959 750 1161 570 
Q 1363 391 1709 391 
Q 2188 391 2477 730 
Q 2766 1069 2766 1631 
L 2766 1759 
L 2194 1759 
z
M 3341 1997 
L 3341 0 
L 2766 0 
L 2766 531 
Q 2569 213 2275 61 
Q 1981 -91 1556 -91 
Q 1019 -91 701 211 
Q 384 513 384 1019 
Q 384 1609 779 1909 
Q 1175 2209 1959 2209 
L 2766 2209 
L 2766 2266 
Q 2766 2663 2505 2880 
Q 2244 3097 1772 3097 
Q 1472 3097 1187 3025 
Q 903 2953 641 2809 
L 641 3341 
Q 956 3463 1253 3523 
Q 1550 3584 1831 3584 
Q 2591 3584 2966 3190 
Q 3341 2797 3341 1997 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-6c" d="M 603 4863 
L 1178 4863 
L 1178 0 
L 603 0 
L 603 4863 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-52"/>
      <use xlink:href="#DejaVuSans-65" x="64.982422"/>
      <use xlink:href="#DejaVuSans-61" x="126.505859"/>
      <use xlink:href="#DejaVuSans-6c" x="187.785156"/>
     </g>
    </g>
    <g id="text_10">
     <!-- 1e11 -->
     <g style="fill: #ffffff" transform="translate(395.959375 418.120313) scale(0.1 -0.1)">
      <use xlink:href="#DejaVuSans-31"/>
      <use xlink:href="#DejaVuSans-65" x="63.623047"/>
      <use xlink:href="#DejaVuSans-31" x="125.146484"/>
      <use xlink:href="#DejaVuSans-31" x="188.769531"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_2">
    <g id="ytick_1">
     <g id="line2d_17">
      <path d="M 46.220312 377.526268 
L 421.2 377.526268 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_18">
      <defs>
       <path id="md4ac40841b" d="M 0 0 
L -3.5 0 
" style="stroke: #ffffff; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#md4ac40841b" x="46.220312" y="377.526268" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_11">
      <!-- −8 -->
      <g style="fill: #ffffff" transform="translate(24.478125 381.325487) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-38" d="M 2034 2216 
Q 1584 2216 1326 1975 
Q 1069 1734 1069 1313 
Q 1069 891 1326 650 
Q 1584 409 2034 409 
Q 2484 409 2743 651 
Q 3003 894 3003 1313 
Q 3003 1734 2745 1975 
Q 2488 2216 2034 2216 
z
M 1403 2484 
Q 997 2584 770 2862 
Q 544 3141 544 3541 
Q 544 4100 942 4425 
Q 1341 4750 2034 4750 
Q 2731 4750 3128 4425 
Q 3525 4100 3525 3541 
Q 3525 3141 3298 2862 
Q 3072 2584 2669 2484 
Q 3125 2378 3379 2068 
Q 3634 1759 3634 1313 
Q 3634 634 3220 271 
Q 2806 -91 2034 -91 
Q 1263 -91 848 271 
Q 434 634 434 1313 
Q 434 1759 690 2068 
Q 947 2378 1403 2484 
z
M 1172 3481 
Q 1172 3119 1398 2916 
Q 1625 2713 2034 2713 
Q 2441 2713 2670 2916 
Q 2900 3119 2900 3481 
Q 2900 3844 2670 4047 
Q 2441 4250 2034 4250 
Q 1625 4250 1398 4047 
Q 1172 3844 1172 3481 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-38" x="83.789062"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_19">
      <path d="M 46.220312 334.674974 
L 421.2 334.674974 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_20">
      <g>
       <use xlink:href="#md4ac40841b" x="46.220312" y="334.674974" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_12">
      <!-- −6 -->
      <g style="fill: #ffffff" transform="translate(24.478125 338.474193) scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-36" d="M 2113 2584 
Q 1688 2584 1439 2293 
Q 1191 2003 1191 1497 
Q 1191 994 1439 701 
Q 1688 409 2113 409 
Q 2538 409 2786 701 
Q 3034 994 3034 1497 
Q 3034 2003 2786 2293 
Q 2538 2584 2113 2584 
z
M 3366 4563 
L 3366 3988 
Q 3128 4100 2886 4159 
Q 2644 4219 2406 4219 
Q 1781 4219 1451 3797 
Q 1122 3375 1075 2522 
Q 1259 2794 1537 2939 
Q 1816 3084 2150 3084 
Q 2853 3084 3261 2657 
Q 3669 2231 3669 1497 
Q 3669 778 3244 343 
Q 2819 -91 2113 -91 
Q 1303 -91 875 529 
Q 447 1150 447 2328 
Q 447 3434 972 4092 
Q 1497 4750 2381 4750 
Q 2619 4750 2861 4703 
Q 3103 4656 3366 4563 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-36" x="83.789062"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_21">
      <path d="M 46.220312 291.823681 
L 421.2 291.823681 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_22">
      <g>
       <use xlink:href="#md4ac40841b" x="46.220312" y="291.823681" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_13">
      <!-- −4 -->
      <g style="fill: #ffffff" transform="translate(24.478125 295.6229) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-34" x="83.789062"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_23">
      <path d="M 46.220312 248.972387 
L 421.2 248.972387 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_24">
      <g>
       <use xlink:href="#md4ac40841b" x="46.220312" y="248.972387" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_14">
      <!-- −2 -->
      <g style="fill: #ffffff" transform="translate(24.478125 252.771606) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-32" x="83.789062"/>
      </g>
     </g>
    </g>
    <g id="ytick_5">
     <g id="line2d_25">
      <path d="M 46.220312 206.121094 
L 421.2 206.121094 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_26">
      <g>
       <use xlink:href="#md4ac40841b" x="46.220312" y="206.121094" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_15">
      <!-- 0 -->
      <g style="fill: #ffffff" transform="translate(32.857812 209.920312) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="ytick_6">
     <g id="line2d_27">
      <path d="M 46.220312 163.2698 
L 421.2 163.2698 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_28">
      <g>
       <use xlink:href="#md4ac40841b" x="46.220312" y="163.2698" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_16">
      <!-- 2 -->
      <g style="fill: #ffffff" transform="translate(32.857812 167.069019) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-32"/>
      </g>
     </g>
    </g>
    <g id="ytick_7">
     <g id="line2d_29">
      <path d="M 46.220312 120.418507 
L 421.2 120.418507 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_30">
      <g>
       <use xlink:href="#md4ac40841b" x="46.220312" y="120.418507" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_17">
      <!-- 4 -->
      <g style="fill: #ffffff" transform="translate(32.857812 124.217725) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-34"/>
      </g>
     </g>
    </g>
    <g id="ytick_8">
     <g id="line2d_31">
      <path d="M 46.220312 77.567213 
L 421.2 77.567213 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_32">
      <g>
       <use xlink:href="#md4ac40841b" x="46.220312" y="77.567213" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_18">
      <!-- 6 -->
      <g style="fill: #ffffff" transform="translate(32.857812 81.366432) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-36"/>
      </g>
     </g>
    </g>
    <g id="ytick_9">
     <g id="line2d_33">
      <path d="M 46.220312 34.71592 
L 421.2 34.71592 
" clip-path="url(#p20336d2121)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_34">
      <g>
       <use xlink:href="#md4ac40841b" x="46.220312" y="34.71592" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_19">
      <!-- 8 -->
      <g style="fill: #ffffff" transform="translate(32.857812 38.515138) scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-38"/>
      </g>
     </g>
    </g>
    <g id="text_20">
     <!-- Imag -->
     <g style="fill: #ffffff" transform="translate(18.398438 218.704688) rotate(-90) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-49" d="M 628 4666 
L 1259 4666 
L 1259 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-6d" d="M 3328 2828 
Q 3544 3216 3844 3400 
Q 4144 3584 4550 3584 
Q 5097 3584 5394 3201 
Q 5691 2819 5691 2113 
L 5691 0 
L 5113 0 
L 5113 2094 
Q 5113 2597 4934 2840 
Q 4756 3084 4391 3084 
Q 3944 3084 3684 2787 
Q 3425 2491 3425 1978 
L 3425 0 
L 2847 0 
L 2847 2094 
Q 2847 2600 2669 2842 
Q 2491 3084 2119 3084 
Q 1678 3084 1418 2786 
Q 1159 2488 1159 1978 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1356 3278 1631 3431 
Q 1906 3584 2284 3584 
Q 2666 3584 2933 3390 
Q 3200 3197 3328 2828 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-67" d="M 2906 1791 
Q 2906 2416 2648 2759 
Q 2391 3103 1925 3103 
Q 1463 3103 1205 2759 
Q 947 2416 947 1791 
Q 947 1169 1205 825 
Q 1463 481 1925 481 
Q 2391 481 2648 825 
Q 2906 1169 2906 1791 
z
M 3481 434 
Q 3481 -459 3084 -895 
Q 2688 -1331 1869 -1331 
Q 1566 -1331 1297 -1286 
Q 1028 -1241 775 -1147 
L 775 -588 
Q 1028 -725 1275 -790 
Q 1522 -856 1778 -856 
Q 2344 -856 2625 -561 
Q 2906 -266 2906 331 
L 2906 616 
Q 2728 306 2450 153 
Q 2172 0 1784 0 
Q 1141 0 747 490 
Q 353 981 353 1791 
Q 353 2603 747 3093 
Q 1141 3584 1784 3584 
Q 2172 3584 2450 3431 
Q 2728 3278 2906 2969 
L 2906 3500 
L 3481 3500 
L 3481 434 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-49"/>
      <use xlink:href="#DejaVuSans-6d" x="29.492188"/>
      <use xlink:href="#DejaVuSans-61" x="126.904297"/>
      <use xlink:href="#DejaVuSans-67" x="188.183594"/>
     </g>
    </g>
    <g id="text_21">
     <!-- 1e11 -->
     <g style="fill: #ffffff" transform="translate(46.220312 18.398438) scale(0.1 -0.1)">
      <use xlink:href="#DejaVuSans-31"/>
      <use xlink:href="#DejaVuSans-65" x="63.623047"/>
      <use xlink:href="#DejaVuSans-31" x="125.146484"/>
      <use xlink:href="#DejaVuSans-31" x="188.769531"/>
     </g>
    </g>
   </g>
   <g id="patch_3">
    <path d="M 46.220312 390.84375 
L 46.220312 21.398438 
" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_4">
    <path d="M 421.2 390.84375 
L 421.2 21.398438 
" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_5">
    <path d="M 46.220312 390.84375 
L 421.2 390.84375 
" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_6">
    <path d="M 46.220312 21.398438 
L 421.2 21.398438 
" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="legend_1">
    <g id="patch_7">
     <path d="M 355.592188 385.84375 
L 414.2 385.84375 
Q 416.2 385.84375 416.2 383.84375 
L 416.2 370.165625 
Q 416.2 368.165625 414.2 368.165625 
L 355.592188 368.165625 
Q 353.592188 368.165625 353.592188 370.165625 
L 353.592188 383.84375 
Q 353.592188 385.84375 355.592188 385.84375 
z
" style="opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter"/>
    </g>
    <g id="PathCollection_2">
     <g>
      <use xlink:href="#mdc8df0de17" x="367.592188" y="377.139063" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-width: 1.5"/>
     </g>
    </g>
    <g id="text_22">
     <!-- poles -->
     <g style="fill: #ffffff" transform="translate(385.592188 379.764063) scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-70" d="M 1159 525 
L 1159 -1331 
L 581 -1331 
L 581 3500 
L 1159 3500 
L 1159 2969 
Q 1341 3281 1617 3432 
Q 1894 3584 2278 3584 
Q 2916 3584 3314 3078 
Q 3713 2572 3713 1747 
Q 3713 922 3314 415 
Q 2916 -91 2278 -91 
Q 1894 -91 1617 61 
Q 1341 213 1159 525 
z
M 3116 1747 
Q 3116 2381 2855 2742 
Q 2594 3103 2138 3103 
Q 1681 3103 1420 2742 
Q 1159 2381 1159 1747 
Q 1159 1113 1420 752 
Q 1681 391 2138 391 
Q 2594 391 2855 752 
Q 3116 1113 3116 1747 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-6f" d="M 1959 3097 
Q 1497 3097 1228 2736 
Q 959 2375 959 1747 
Q 959 1119 1226 758 
Q 1494 397 1959 397 
Q 2419 397 2687 759 
Q 2956 1122 2956 1747 
Q 2956 2369 2687 2733 
Q 2419 3097 1959 3097 
z
M 1959 3584 
Q 2709 3584 3137 3096 
Q 3566 2609 3566 1747 
Q 3566 888 3137 398 
Q 2709 -91 1959 -91 
Q 1206 -91 779 398 
Q 353 888 353 1747 
Q 353 2609 779 3096 
Q 1206 3584 1959 3584 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-73" d="M 2834 3397 
L 2834 2853 
Q 2591 2978 2328 3040 
Q 2066 3103 1784 3103 
Q 1356 3103 1142 2972 
Q 928 2841 928 2578 
Q 928 2378 1081 2264 
Q 1234 2150 1697 2047 
L 1894 2003 
Q 2506 1872 2764 1633 
Q 3022 1394 3022 966 
Q 3022 478 2636 193 
Q 2250 -91 1575 -91 
Q 1294 -91 989 -36 
Q 684 19 347 128 
L 347 722 
Q 666 556 975 473 
Q 1284 391 1588 391 
Q 1994 391 2212 530 
Q 2431 669 2431 922 
Q 2431 1156 2273 1281 
Q 2116 1406 1581 1522 
L 1381 1569 
Q 847 1681 609 1914 
Q 372 2147 372 2553 
Q 372 3047 722 3315 
Q 1072 3584 1716 3584 
Q 2034 3584 2315 3537 
Q 2597 3491 2834 3397 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-70"/>
      <use xlink:href="#DejaVuSans-6f" x="63.476562"/>
      <use xlink:href="#DejaVuSans-6c" x="124.658203"/>
      <use xlink:href="#DejaVuSans-65" x="152.441406"/>
      <use xlink:href="#DejaVuSans-73" x="213.964844"/>
     </g>
    </g>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="p20336d2121">
   <rect x="46.220312" y="21.398438" width="374.979688" height="369.445312"/>
  </clipPath>
 </defs>
</svg>
ing test_poles.svg…]()


## References

[1] Gustavsen, B. and Adam Semlyen. “Rational approximation of frequency domain responses by vector fitting.” IEEE Transactions on Power Delivery 14 (1999): 1052-1061.

[2] B. Gustavsen, "Improving the pole relocating properties of vector fitting," in IEEE Transactions on Power Delivery, vol. 21, no. 3, pp. 1587-1592, July 2006, doi: 10.1109/TPWRD.2005.860281.

[3] D. Deschrijver, M. Mrozowski, T. Dhaene and D. De Zutter, "Macromodeling of Multiport Systems Using a Fast Implementation of the Vector Fitting Method," in IEEE Microwave and Wireless Components Letters, vol. 18, no. 6, pp. 383-385, June 2008, doi: 10.1109/LMWC.2008.922585.



