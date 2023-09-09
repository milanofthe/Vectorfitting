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
<?xml version="1.0" encoding="utf-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns:xlink="http://www.w3.org/1999/xlink" width="720pt" height="432pt" viewBox="0 0 720 432" xmlns="http://www.w3.org/2000/svg" version="1.1">
 <metadata>
  <rdf:RDF xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:cc="http://creativecommons.org/ns#" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
   <cc:Work>
    <dc:type rdf:resource="http://purl.org/dc/dcmitype/StillImage"/>
    <dc:date>2023-09-09T10:51:13.225265</dc:date>
    <dc:format>image/svg+xml</dc:format>
    <dc:creator>
     <cc:Agent>
      <dc:title>Matplotlib v3.5.1, https://matplotlib.org/</dc:title>
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
L 720 432 
L 720 0 
L 0 0 
z
"/>
  </g>
  <g id="axes_1">
   <g id="patch_2">
    <path d="M 52.582813 390.84375 
L 709.2 390.84375 
L 709.2 10.8 
L 52.582813 10.8 
z
"/>
   </g>
   <g id="matplotlib.axis_1">
    <g id="xtick_1">
     <g id="line2d_1">
      <path d="M 82.423079 390.84375 
L 82.423079 10.8 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_2">
      <defs>
       <path id="mfb2a845213" d="M 0 0 
L 0 3.5 
" style="stroke: #ffffff; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#mfb2a845213" x="82.423079" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_1">
      <!-- 0.0 -->
      <g style="fill: #ffffff" transform="translate(74.471516 405.442187)scale(0.1 -0.1)">
       <defs>
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
        <path id="DejaVuSans-2e" d="M 684 794 
L 1344 794 
L 1344 0 
L 684 0 
L 684 794 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-30" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="xtick_2">
     <g id="line2d_3">
      <path d="M 201.809216 390.84375 
L 201.809216 10.8 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_4">
      <g>
       <use xlink:href="#mfb2a845213" x="201.809216" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_2">
      <!-- 0.2 -->
      <g style="fill: #ffffff" transform="translate(193.857654 405.442187)scale(0.1 -0.1)">
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
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-32" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="xtick_3">
     <g id="line2d_5">
      <path d="M 321.195353 390.84375 
L 321.195353 10.8 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_6">
      <g>
       <use xlink:href="#mfb2a845213" x="321.195353" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_3">
      <!-- 0.4 -->
      <g style="fill: #ffffff" transform="translate(313.243791 405.442187)scale(0.1 -0.1)">
       <defs>
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
       </defs>
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-34" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="xtick_4">
     <g id="line2d_7">
      <path d="M 440.58149 390.84375 
L 440.58149 10.8 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_8">
      <g>
       <use xlink:href="#mfb2a845213" x="440.58149" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_4">
      <!-- 0.6 -->
      <g style="fill: #ffffff" transform="translate(432.629928 405.442187)scale(0.1 -0.1)">
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
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-36" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="xtick_5">
     <g id="line2d_9">
      <path d="M 559.967627 390.84375 
L 559.967627 10.8 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_10">
      <g>
       <use xlink:href="#mfb2a845213" x="559.967627" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_5">
      <!-- 0.8 -->
      <g style="fill: #ffffff" transform="translate(552.016065 405.442187)scale(0.1 -0.1)">
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
       <use xlink:href="#DejaVuSans-30"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-38" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="xtick_6">
     <g id="line2d_11">
      <path d="M 679.353764 390.84375 
L 679.353764 10.8 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_12">
      <g>
       <use xlink:href="#mfb2a845213" x="679.353764" y="390.84375" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_6">
      <!-- 1.0 -->
      <g style="fill: #ffffff" transform="translate(671.402202 405.442187)scale(0.1 -0.1)">
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
       <use xlink:href="#DejaVuSans-31"/>
       <use xlink:href="#DejaVuSans-2e" x="63.623047"/>
       <use xlink:href="#DejaVuSans-30" x="95.410156"/>
      </g>
     </g>
    </g>
    <g id="text_7">
     <!-- freq [Hz] -->
     <g style="fill: #ffffff" transform="translate(359.0625 419.120313)scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-66" d="M 2375 4863 
L 2375 4384 
L 1825 4384 
Q 1516 4384 1395 4259 
Q 1275 4134 1275 3809 
L 1275 3500 
L 2222 3500 
L 2222 3053 
L 1275 3053 
L 1275 0 
L 697 0 
L 697 3053 
L 147 3053 
L 147 3500 
L 697 3500 
L 697 3744 
Q 697 4328 969 4595 
Q 1241 4863 1831 4863 
L 2375 4863 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-72" d="M 2631 2963 
Q 2534 3019 2420 3045 
Q 2306 3072 2169 3072 
Q 1681 3072 1420 2755 
Q 1159 2438 1159 1844 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1341 3275 1631 3429 
Q 1922 3584 2338 3584 
Q 2397 3584 2469 3576 
Q 2541 3569 2628 3553 
L 2631 2963 
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
       <path id="DejaVuSans-71" d="M 947 1747 
Q 947 1113 1208 752 
Q 1469 391 1925 391 
Q 2381 391 2643 752 
Q 2906 1113 2906 1747 
Q 2906 2381 2643 2742 
Q 2381 3103 1925 3103 
Q 1469 3103 1208 2742 
Q 947 2381 947 1747 
z
M 2906 525 
Q 2725 213 2448 61 
Q 2172 -91 1784 -91 
Q 1150 -91 751 415 
Q 353 922 353 1747 
Q 353 2572 751 3078 
Q 1150 3584 1784 3584 
Q 2172 3584 2448 3432 
Q 2725 3281 2906 2969 
L 2906 3500 
L 3481 3500 
L 3481 -1331 
L 2906 -1331 
L 2906 525 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-20" transform="scale(0.015625)"/>
       <path id="DejaVuSans-5b" d="M 550 4863 
L 1875 4863 
L 1875 4416 
L 1125 4416 
L 1125 -397 
L 1875 -397 
L 1875 -844 
L 550 -844 
L 550 4863 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-48" d="M 628 4666 
L 1259 4666 
L 1259 2753 
L 3553 2753 
L 3553 4666 
L 4184 4666 
L 4184 0 
L 3553 0 
L 3553 2222 
L 1259 2222 
L 1259 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-7a" d="M 353 3500 
L 3084 3500 
L 3084 2975 
L 922 459 
L 3084 459 
L 3084 0 
L 275 0 
L 275 525 
L 2438 3041 
L 353 3041 
L 353 3500 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-5d" d="M 1947 4863 
L 1947 -844 
L 622 -844 
L 622 -397 
L 1369 -397 
L 1369 4416 
L 622 4416 
L 622 4863 
L 1947 4863 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-66"/>
      <use xlink:href="#DejaVuSans-72" x="35.205078"/>
      <use xlink:href="#DejaVuSans-65" x="74.068359"/>
      <use xlink:href="#DejaVuSans-71" x="135.591797"/>
      <use xlink:href="#DejaVuSans-20" x="199.068359"/>
      <use xlink:href="#DejaVuSans-5b" x="230.855469"/>
      <use xlink:href="#DejaVuSans-48" x="269.869141"/>
      <use xlink:href="#DejaVuSans-7a" x="345.064453"/>
      <use xlink:href="#DejaVuSans-5d" x="397.554688"/>
     </g>
    </g>
    <g id="text_8">
     <!-- 1e11 -->
     <g style="fill: #ffffff" transform="translate(683.959375 418.120313)scale(0.1 -0.1)">
      <use xlink:href="#DejaVuSans-31"/>
      <use xlink:href="#DejaVuSans-65" x="63.623047"/>
      <use xlink:href="#DejaVuSans-31" x="125.146484"/>
      <use xlink:href="#DejaVuSans-31" x="188.769531"/>
     </g>
    </g>
   </g>
   <g id="matplotlib.axis_2">
    <g id="ytick_1">
     <g id="line2d_13">
      <path d="M 52.582813 357.382947 
L 709.2 357.382947 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_14">
      <defs>
       <path id="m19a73be898" d="M 0 0 
L -3.5 0 
" style="stroke: #ffffff; stroke-width: 0.8"/>
      </defs>
      <g>
       <use xlink:href="#m19a73be898" x="52.582813" y="357.382947" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_9">
      <!-- −80 -->
      <g style="fill: #ffffff" transform="translate(24.478125 361.182166)scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-2212" d="M 678 2272 
L 4684 2272 
L 4684 1741 
L 678 1741 
L 678 2272 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-38" x="83.789062"/>
       <use xlink:href="#DejaVuSans-30" x="147.412109"/>
      </g>
     </g>
    </g>
    <g id="ytick_2">
     <g id="line2d_15">
      <path d="M 52.582813 314.727633 
L 709.2 314.727633 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_16">
      <g>
       <use xlink:href="#m19a73be898" x="52.582813" y="314.727633" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_10">
      <!-- −70 -->
      <g style="fill: #ffffff" transform="translate(24.478125 318.526852)scale(0.1 -0.1)">
       <defs>
        <path id="DejaVuSans-37" d="M 525 4666 
L 3525 4666 
L 3525 4397 
L 1831 0 
L 1172 0 
L 2766 4134 
L 525 4134 
L 525 4666 
z
" transform="scale(0.015625)"/>
       </defs>
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-37" x="83.789062"/>
       <use xlink:href="#DejaVuSans-30" x="147.412109"/>
      </g>
     </g>
    </g>
    <g id="ytick_3">
     <g id="line2d_17">
      <path d="M 52.582813 272.07232 
L 709.2 272.07232 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_18">
      <g>
       <use xlink:href="#m19a73be898" x="52.582813" y="272.07232" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_11">
      <!-- −60 -->
      <g style="fill: #ffffff" transform="translate(24.478125 275.871538)scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-36" x="83.789062"/>
       <use xlink:href="#DejaVuSans-30" x="147.412109"/>
      </g>
     </g>
    </g>
    <g id="ytick_4">
     <g id="line2d_19">
      <path d="M 52.582813 229.417006 
L 709.2 229.417006 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_20">
      <g>
       <use xlink:href="#m19a73be898" x="52.582813" y="229.417006" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_12">
      <!-- −50 -->
      <g style="fill: #ffffff" transform="translate(24.478125 233.216225)scale(0.1 -0.1)">
       <defs>
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
       <use xlink:href="#DejaVuSans-35" x="83.789062"/>
       <use xlink:href="#DejaVuSans-30" x="147.412109"/>
      </g>
     </g>
    </g>
    <g id="ytick_5">
     <g id="line2d_21">
      <path d="M 52.582813 186.761692 
L 709.2 186.761692 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_22">
      <g>
       <use xlink:href="#m19a73be898" x="52.582813" y="186.761692" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_13">
      <!-- −40 -->
      <g style="fill: #ffffff" transform="translate(24.478125 190.560911)scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-34" x="83.789062"/>
       <use xlink:href="#DejaVuSans-30" x="147.412109"/>
      </g>
     </g>
    </g>
    <g id="ytick_6">
     <g id="line2d_23">
      <path d="M 52.582813 144.106379 
L 709.2 144.106379 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_24">
      <g>
       <use xlink:href="#m19a73be898" x="52.582813" y="144.106379" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_14">
      <!-- −30 -->
      <g style="fill: #ffffff" transform="translate(24.478125 147.905598)scale(0.1 -0.1)">
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
       </defs>
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-33" x="83.789062"/>
       <use xlink:href="#DejaVuSans-30" x="147.412109"/>
      </g>
     </g>
    </g>
    <g id="ytick_7">
     <g id="line2d_25">
      <path d="M 52.582813 101.451065 
L 709.2 101.451065 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_26">
      <g>
       <use xlink:href="#m19a73be898" x="52.582813" y="101.451065" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_15">
      <!-- −20 -->
      <g style="fill: #ffffff" transform="translate(24.478125 105.250284)scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-32" x="83.789062"/>
       <use xlink:href="#DejaVuSans-30" x="147.412109"/>
      </g>
     </g>
    </g>
    <g id="ytick_8">
     <g id="line2d_27">
      <path d="M 52.582813 58.795752 
L 709.2 58.795752 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_28">
      <g>
       <use xlink:href="#m19a73be898" x="52.582813" y="58.795752" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_16">
      <!-- −10 -->
      <g style="fill: #ffffff" transform="translate(24.478125 62.59497)scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-2212"/>
       <use xlink:href="#DejaVuSans-31" x="83.789062"/>
       <use xlink:href="#DejaVuSans-30" x="147.412109"/>
      </g>
     </g>
    </g>
    <g id="ytick_9">
     <g id="line2d_29">
      <path d="M 52.582813 16.140438 
L 709.2 16.140438 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linecap: square"/>
     </g>
     <g id="line2d_30">
      <g>
       <use xlink:href="#m19a73be898" x="52.582813" y="16.140438" style="fill: #ffffff; stroke: #ffffff; stroke-width: 0.8"/>
      </g>
     </g>
     <g id="text_17">
      <!-- 0 -->
      <g style="fill: #ffffff" transform="translate(39.220313 19.939657)scale(0.1 -0.1)">
       <use xlink:href="#DejaVuSans-30"/>
      </g>
     </g>
    </g>
    <g id="text_18">
     <!-- mag in dB -->
     <g style="fill: #ffffff" transform="translate(18.398438 226.271094)rotate(-90)scale(0.1 -0.1)">
      <defs>
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
       <path id="DejaVuSans-69" d="M 603 3500 
L 1178 3500 
L 1178 0 
L 603 0 
L 603 3500 
z
M 603 4863 
L 1178 4863 
L 1178 4134 
L 603 4134 
L 603 4863 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-6e" d="M 3513 2113 
L 3513 0 
L 2938 0 
L 2938 2094 
Q 2938 2591 2744 2837 
Q 2550 3084 2163 3084 
Q 1697 3084 1428 2787 
Q 1159 2491 1159 1978 
L 1159 0 
L 581 0 
L 581 3500 
L 1159 3500 
L 1159 2956 
Q 1366 3272 1645 3428 
Q 1925 3584 2291 3584 
Q 2894 3584 3203 3211 
Q 3513 2838 3513 2113 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-64" d="M 2906 2969 
L 2906 4863 
L 3481 4863 
L 3481 0 
L 2906 0 
L 2906 525 
Q 2725 213 2448 61 
Q 2172 -91 1784 -91 
Q 1150 -91 751 415 
Q 353 922 353 1747 
Q 353 2572 751 3078 
Q 1150 3584 1784 3584 
Q 2172 3584 2448 3432 
Q 2725 3281 2906 2969 
z
M 947 1747 
Q 947 1113 1208 752 
Q 1469 391 1925 391 
Q 2381 391 2643 752 
Q 2906 1113 2906 1747 
Q 2906 2381 2643 2742 
Q 2381 3103 1925 3103 
Q 1469 3103 1208 2742 
Q 947 2381 947 1747 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-42" d="M 1259 2228 
L 1259 519 
L 2272 519 
Q 2781 519 3026 730 
Q 3272 941 3272 1375 
Q 3272 1813 3026 2020 
Q 2781 2228 2272 2228 
L 1259 2228 
z
M 1259 4147 
L 1259 2741 
L 2194 2741 
Q 2656 2741 2882 2914 
Q 3109 3088 3109 3444 
Q 3109 3797 2882 3972 
Q 2656 4147 2194 4147 
L 1259 4147 
z
M 628 4666 
L 2241 4666 
Q 2963 4666 3353 4366 
Q 3744 4066 3744 3513 
Q 3744 3084 3544 2831 
Q 3344 2578 2956 2516 
Q 3422 2416 3680 2098 
Q 3938 1781 3938 1306 
Q 3938 681 3513 340 
Q 3088 0 2303 0 
L 628 0 
L 628 4666 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-6d"/>
      <use xlink:href="#DejaVuSans-61" x="97.412109"/>
      <use xlink:href="#DejaVuSans-67" x="158.691406"/>
      <use xlink:href="#DejaVuSans-20" x="222.167969"/>
      <use xlink:href="#DejaVuSans-69" x="253.955078"/>
      <use xlink:href="#DejaVuSans-6e" x="281.738281"/>
      <use xlink:href="#DejaVuSans-20" x="345.117188"/>
      <use xlink:href="#DejaVuSans-64" x="376.904297"/>
      <use xlink:href="#DejaVuSans-42" x="440.380859"/>
     </g>
    </g>
   </g>
   <g id="line2d_31">
    <path d="M 82.429048 58.675851 
L 89.535295 58.622334 
L 96.641542 58.30816 
L 103.747788 57.707838 
L 110.854035 56.860467 
L 120.32903 55.448036 
L 132.172775 53.406433 
L 170.072757 46.657022 
L 184.343024 44.43319 
L 198.902164 42.42124 
L 213.461303 40.665261 
L 228.020442 39.146852 
L 242.579582 37.841745 
L 259.565244 36.555055 
L 276.550907 35.486721 
L 295.963093 34.49087 
L 317.801802 33.608392 
L 342.067034 32.867978 
L 368.75879 32.287442 
L 397.877069 31.875848 
L 431.848394 31.624565 
L 470.672766 31.569291 
L 516.776707 31.743287 
L 565.307172 32.153071 
L 596.447554 32.610552 
L 613.028796 33.037205 
L 624.87254 33.540036 
L 634.132195 34.162466 
L 640.91543 34.839508 
L 645.437587 35.449258 
L 649.959744 36.236678 
L 654.481901 37.263424 
L 659.004058 38.613154 
L 661.265136 39.442912 
L 663.526215 40.398137 
L 665.787293 41.498461 
L 668.048372 42.766091 
L 670.30945 44.225869 
L 672.570529 45.905216 
L 674.831607 47.833876 
L 677.092686 50.043364 
L 679.353764 52.565971 
L 679.353764 52.565971 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
    <defs>
     <path id="mf021a1f644" d="M 0 3.535534 
L 2.12132 0 
L 0 -3.535534 
L -2.12132 0 
z
" style="stroke: #8dd3c7; stroke-linejoin: miter"/>
    </defs>
    <g clip-path="url(#pf67e761123)">
     <use xlink:href="#mf021a1f644" x="82.429048" y="58.675851" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="117.960281" y="55.825096" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="153.491514" y="49.51776" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="189.196071" y="43.733441" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="225.593919" y="39.384421" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="261.991768" y="36.389892" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="298.389616" y="34.38125" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="334.787465" y="33.066637" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="371.185313" y="32.245027" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="407.583162" y="31.781444" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="443.98101" y="31.583766" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="480.378859" y="31.586763" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="516.776707" y="31.743287" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="553.174556" y="32.027245" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="589.341307" y="32.480194" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="624.87254" y="33.540036" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="659.004058" y="38.613154" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
    </g>
   </g>
   <g id="line2d_32">
    <path d="M 82.429048 59.007704 
L 89.535295 58.862398 
L 96.641542 58.436521 
L 103.747788 57.756958 
L 110.854035 56.863285 
L 120.32903 55.418896 
L 132.172775 53.361093 
L 167.704008 47.008639 
L 181.916501 44.76273 
L 196.47564 42.717287 
L 211.03478 40.929203 
L 225.593919 39.380987 
L 240.153059 38.048439 
L 257.138721 36.732349 
L 274.124384 35.636935 
L 293.53657 34.612506 
L 315.375279 33.700713 
L 339.640511 32.931599 
L 366.332267 32.325166 
L 395.450546 31.893272 
L 429.421871 31.629081 
L 468.246243 31.569824 
L 514.350184 31.741749 
L 565.307172 32.158937 
L 596.447554 32.604028 
L 613.028796 33.025873 
L 624.87254 33.527908 
L 634.132195 34.151758 
L 640.91543 34.830906 
L 645.437587 35.442398 
L 649.959744 36.231652 
L 654.481901 37.260148 
L 659.004058 38.611444 
L 661.265136 39.44196 
L 663.526215 40.398017 
L 665.787293 41.499397 
L 668.048372 42.768544 
L 670.30945 44.230667 
L 672.570529 45.913719 
L 674.831607 47.848186 
L 677.092686 50.066574 
L 679.353764 52.602431 
L 679.353764 52.602431 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
   </g>
   <g id="line2d_33">
    <path d="M 82.429048 191.008897 
L 84.797797 194.179853 
L 87.166546 195.429625 
L 89.535295 197.310388 
L 91.904044 199.682132 
L 94.272793 202.385252 
L 99.01029 208.279415 
L 106.116537 217.243689 
L 110.854035 222.925814 
L 115.591533 228.283502 
L 120.32903 233.307969 
L 125.066528 238.009014 
L 129.804026 242.40203 
L 134.541523 246.503968 
L 139.279021 250.332752 
L 144.016519 253.907743 
L 148.754017 257.250224 
L 153.491514 260.383552 
L 160.597761 264.746788 
L 167.704008 268.785654 
L 177.179003 273.820777 
L 191.622594 281.130715 
L 206.181733 288.539761 
L 215.887826 293.684065 
L 228.020442 300.405312 
L 247.432628 311.309579 
L 252.285675 313.814987 
L 257.138721 316.094162 
L 261.991768 318.06163 
L 266.844814 319.637543 
L 269.271337 320.259044 
L 271.697861 320.763084 
L 274.124384 321.147914 
L 276.550907 321.414482 
L 281.403954 321.609662 
L 286.257 321.404253 
L 291.110047 320.878985 
L 295.963093 320.124158 
L 303.242663 318.748495 
L 317.801802 315.922624 
L 325.081372 314.753031 
L 332.360942 313.848859 
L 339.640511 313.249324 
L 344.493558 313.030678 
L 349.346604 312.96222 
L 354.199651 313.047159 
L 359.052697 313.287883 
L 363.905744 313.686141 
L 368.75879 314.243037 
L 373.611837 314.958867 
L 378.464883 315.832737 
L 383.317929 316.861934 
L 390.597499 318.683836 
L 397.877069 320.803514 
L 407.583162 323.952299 
L 417.289255 327.123448 
L 422.142301 328.526469 
L 426.995348 329.671082 
L 431.848394 330.447426 
L 434.274917 330.666984 
L 436.701441 330.76194 
L 439.127964 330.726897 
L 441.554487 330.560005 
L 443.98101 330.263136 
L 446.407534 329.841747 
L 451.26058 328.662537 
L 456.113627 327.117923 
L 460.966673 325.321149 
L 473.099289 320.416247 
L 480.378859 317.587769 
L 487.658429 315.056911 
L 492.511475 313.586152 
L 497.364522 312.313367 
L 502.217568 311.254977 
L 507.070614 310.425351 
L 511.923661 309.838343 
L 516.776707 309.508536 
L 521.629754 309.452296 
L 526.4828 309.688721 
L 531.335847 310.240565 
L 536.188893 311.13514 
L 541.04194 312.405134 
L 545.894986 314.089107 
L 550.748033 316.231006 
L 553.174556 317.48822 
L 555.601079 318.877033 
L 558.027602 320.401996 
L 560.454126 322.065819 
L 565.307172 325.802241 
L 570.160219 329.993559 
L 575.013265 334.311539 
L 577.439788 336.296873 
L 579.866312 337.975467 
L 582.23506 339.147163 
L 584.603809 339.696174 
L 586.972558 339.513679 
L 589.341307 338.580962 
L 591.710056 336.977711 
L 594.078805 334.853693 
L 596.447554 332.383448 
L 605.922549 321.796655 
L 608.291298 319.393383 
L 610.660047 317.17454 
L 613.028796 315.161275 
L 615.397545 313.369607 
L 617.766294 311.813373 
L 620.135042 310.50641 
L 622.503791 309.46427 
L 624.87254 308.705746 
L 627.241289 308.254402 
L 629.610038 308.140344 
L 631.871116 308.381736 
L 634.132195 309.009694 
L 636.393273 310.081476 
L 638.654352 311.672958 
L 640.91543 313.886731 
L 643.176509 316.865197 
L 645.437587 320.812792 
L 647.698666 326.035644 
L 649.959744 333.015389 
L 652.220823 342.542036 
L 654.481901 355.810677 
L 656.74298 372.18884 
L 659.004058 373.569034 
L 661.265136 356.794694 
L 663.526215 342.382908 
L 665.787293 331.981446 
L 668.048372 324.143663 
L 670.30945 317.298132 
L 672.570529 309.492889 
L 674.831607 298.943526 
L 677.092686 285.535741 
L 679.353764 270.669858 
L 679.353764 270.669858 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
   </g>
   <g id="line2d_34">
    <path d="M 82.429048 31.354947 
L 103.747788 31.400871 
L 577.439788 32.252097 
L 598.816303 32.5318 
L 613.028796 32.914599 
L 622.503791 33.347147 
L 629.610038 33.829664 
L 636.393273 34.48544 
L 640.91543 35.074148 
L 645.437587 35.829773 
L 649.959744 36.807558 
L 654.481901 38.083082 
L 659.004058 39.759874 
L 661.265136 40.791145 
L 663.526215 41.9794 
L 665.787293 43.35032 
L 668.048372 44.933674 
L 670.30945 46.763848 
L 672.570529 48.880385 
L 674.831607 51.328518 
L 677.092686 54.159581 
L 679.353764 57.431043 
L 679.353764 57.431043 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
    <g clip-path="url(#pf67e761123)">
     <use xlink:href="#mf021a1f644" x="82.429048" y="31.354947" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="117.960281" y="31.421237" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="153.491514" y="31.462927" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="189.196071" y="31.510048" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="225.593919" y="31.566618" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="261.991768" y="31.628704" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="298.389616" y="31.69283" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="334.787465" y="31.756211" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="371.185313" y="31.816797" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="407.583162" y="31.87314" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="443.98101" y="31.924537" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="480.378859" y="31.972254" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="516.776707" y="32.024302" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="553.174556" y="32.113875" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="589.341307" y="32.378657" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="624.87254" y="33.489519" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="659.004058" y="39.759874" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
    </g>
   </g>
   <g id="line2d_35">
    <path d="M 82.429048 31.394663 
L 181.916501 31.505655 
L 582.23506 32.288539 
L 601.185051 32.572799 
L 615.397545 33.002319 
L 624.87254 33.491629 
L 631.871116 34.029964 
L 638.654352 34.773276 
L 643.176509 35.44223 
L 647.698666 36.302681 
L 652.220823 37.4186 
L 656.74298 38.877677 
L 659.004058 39.77191 
L 661.265136 40.800157 
L 663.526215 41.98442 
L 665.787293 43.35033 
L 668.048372 44.927661 
L 670.30945 46.750882 
L 672.570529 48.85973 
L 674.831607 51.299777 
L 677.092686 54.122882 
L 679.353764 57.387245 
L 679.353764 57.387245 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
   </g>
   <g id="line2d_36">
    <path d="M 82.429048 269.517759 
L 84.797797 286.108956 
L 87.166546 291.858352 
L 89.535295 294.227658 
L 94.272793 297.995064 
L 96.641542 300.036833 
L 101.379039 304.540323 
L 110.854035 313.930138 
L 113.222784 315.984843 
L 115.591533 317.779934 
L 117.960281 319.252504 
L 120.32903 320.356645 
L 122.697779 321.070592 
L 125.066528 321.399968 
L 127.435277 321.375719 
L 129.804026 321.047467 
L 132.172775 320.474703 
L 134.541523 319.718503 
L 139.279021 317.874317 
L 148.754017 313.919138 
L 153.491514 312.147744 
L 158.229012 310.610928 
L 162.96651 309.331135 
L 167.704008 308.312748 
L 172.441505 307.551505 
L 177.179003 307.039597 
L 181.916501 306.768334 
L 186.769547 306.731377 
L 191.622594 306.930663 
L 196.47564 307.360012 
L 201.328687 308.014735 
L 206.181733 308.891487 
L 211.03478 309.98798 
L 215.887826 311.302525 
L 220.740873 312.833333 
L 225.593919 314.577436 
L 230.446966 316.52902 
L 235.300012 318.676852 
L 242.579582 322.217652 
L 259.565244 330.934806 
L 264.418291 333.006435 
L 266.844814 333.858905 
L 269.271337 334.555124 
L 271.697861 335.072559 
L 274.124384 335.393531 
L 276.550907 335.50707 
L 278.97743 335.410153 
L 281.403954 335.108053 
L 283.830477 334.613708 
L 286.257 333.946265 
L 288.683523 333.129121 
L 293.53657 331.148318 
L 298.389616 328.871185 
L 315.375279 320.621795 
L 320.228325 318.512653 
L 325.081372 316.579346 
L 329.934418 314.833558 
L 334.787465 313.280086 
L 339.640511 311.919783 
L 344.493558 310.751451 
L 349.346604 309.773039 
L 354.199651 308.982399 
L 359.052697 308.377757 
L 363.905744 307.958012 
L 368.75879 307.722924 
L 373.611837 307.673239 
L 378.464883 307.810764 
L 383.317929 308.138418 
L 388.170976 308.660238 
L 393.024022 309.38135 
L 397.877069 310.307863 
L 402.730115 311.446628 
L 407.583162 312.804783 
L 412.436208 314.388909 
L 417.289255 316.203524 
L 422.142301 318.248474 
L 426.995348 320.514481 
L 434.274917 324.264348 
L 446.407534 330.749335 
L 448.834057 331.894633 
L 451.26058 332.913932 
L 453.687103 333.767826 
L 456.113627 334.417235 
L 458.54015 334.827217 
L 460.966673 334.971186 
L 463.393196 334.834653 
L 465.819719 334.417457 
L 468.246243 333.733842 
L 470.672766 332.810326 
L 473.099289 331.682037 
L 477.952336 328.969733 
L 482.805382 325.903187 
L 492.511475 319.620489 
L 497.364522 316.680813 
L 502.217568 313.972095 
L 507.070614 311.527753 
L 511.923661 309.365336 
L 516.776707 307.494205 
L 521.629754 305.920309 
L 526.4828 304.64915 
L 531.335847 303.687711 
L 536.188893 303.045793 
L 541.04194 302.737082 
L 545.894986 302.780109 
L 550.748033 303.199192 
L 555.601079 304.025346 
L 558.027602 304.602771 
L 560.454126 305.29693 
L 562.880649 306.113728 
L 565.307172 307.059373 
L 567.733695 308.140181 
L 570.160219 309.362212 
L 572.586742 310.730671 
L 575.013265 312.248913 
L 577.439788 313.91683 
L 582.23506 317.619773 
L 591.710056 325.43513 
L 594.078805 326.936505 
L 596.447554 327.927426 
L 598.816303 328.228816 
L 601.185051 327.725784 
L 603.5538 326.412596 
L 605.922549 324.396412 
L 608.291298 321.85693 
L 613.028796 315.972758 
L 617.766294 309.977487 
L 620.135042 307.161056 
L 622.503791 304.530489 
L 624.87254 302.115628 
L 627.241289 299.938699 
L 629.610038 298.018665 
L 631.871116 296.44293 
L 634.132195 295.137123 
L 636.393273 294.123858 
L 638.654352 293.431689 
L 640.91543 293.097384 
L 643.176509 293.168841 
L 645.437587 293.708993 
L 647.698666 294.801155 
L 649.959744 296.556128 
L 652.220823 299.120413 
L 654.481901 302.680018 
L 656.74298 307.432917 
L 659.004058 313.411856 
L 661.265136 319.716802 
L 663.526215 322.644496 
L 665.787293 317.510437 
L 668.048372 306.883888 
L 670.30945 295.711196 
L 672.570529 285.808915 
L 674.831607 277.452767 
L 677.092686 270.604157 
L 679.353764 265.258961 
L 679.353764 265.258961 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
   </g>
   <g id="line2d_37">
    <path d="M 82.429048 31.871748 
L 103.747788 32.300875 
L 117.960281 32.842272 
L 132.172775 33.60755 
L 148.754017 34.753211 
L 165.335259 36.130007 
L 184.343024 37.930351 
L 208.608257 40.461346 
L 293.53657 49.553625 
L 315.375279 51.57644 
L 334.787465 53.172463 
L 354.199651 54.55049 
L 373.611837 55.693566 
L 393.024022 56.595385 
L 412.436208 57.260673 
L 431.848394 57.70458 
L 453.687103 57.969476 
L 477.952336 58.024258 
L 507.070614 57.848157 
L 545.894986 57.365442 
L 575.013265 56.816213 
L 591.710056 56.280906 
L 601.185051 55.798142 
L 610.660047 55.07683 
L 617.766294 54.294519 
L 622.503791 53.613034 
L 627.241289 52.768829 
L 631.871116 51.754232 
L 636.393273 50.550386 
L 640.91543 49.106295 
L 645.437587 47.397312 
L 649.959744 45.408932 
L 654.481901 43.142909 
L 659.004058 40.623693 
L 665.787293 36.494212 
L 674.831607 30.855269 
L 679.353764 28.277219 
L 679.353764 28.277219 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
    <g clip-path="url(#pf67e761123)">
     <use xlink:href="#mf021a1f644" x="82.429048" y="31.871748" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="117.960281" y="32.842272" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="153.491514" y="35.125047" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="189.196071" y="38.41959" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="225.593919" y="42.317613" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="261.991768" y="46.30786" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="298.389616" y="50.021697" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="334.787465" y="53.172463" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="371.185313" y="55.563853" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="407.583162" y="57.115863" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="443.98101" y="57.879955" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="480.378859" y="58.018307" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="516.776707" y="57.748512" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="553.174556" y="57.250057" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="589.341307" y="56.376201" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="624.87254" y="53.213316" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="659.004058" y="40.623693" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
    </g>
   </g>
   <g id="line2d_38">
    <path d="M 82.429048 31.986168 
L 94.272793 32.07946 
L 108.485286 32.433542 
L 122.697779 33.038664 
L 136.910272 33.872896 
L 153.491514 35.097988 
L 170.072757 36.545812 
L 191.622594 38.673808 
L 220.740873 41.81007 
L 276.550907 47.878638 
L 300.816139 50.276925 
L 322.654849 52.211423 
L 342.067034 53.715592 
L 361.47922 54.994434 
L 380.891406 56.036495 
L 400.303592 56.841005 
L 419.715778 57.417712 
L 439.127964 57.785718 
L 460.966673 57.983357 
L 485.231905 57.986745 
L 516.776707 57.751524 
L 555.601079 57.224418 
L 579.866312 56.701293 
L 594.078805 56.186111 
L 603.5538 55.651876 
L 610.660047 55.082979 
L 617.766294 54.301219 
L 622.503791 53.621232 
L 627.241289 52.779623 
L 631.871116 51.768692 
L 636.393273 50.569414 
L 640.91543 49.130553 
L 645.437587 47.426839 
L 649.959744 45.442754 
L 654.481901 43.178627 
L 659.004058 40.657103 
L 665.787293 36.511885 
L 674.831607 30.820838 
L 679.353764 28.203235 
L 679.353764 28.203235 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
   </g>
   <g id="line2d_39">
    <path d="M 82.429048 230.351535 
L 84.797797 236.466592 
L 87.166546 246.244834 
L 89.535295 253.522683 
L 91.904044 258.158995 
L 94.272793 260.972157 
L 96.641542 262.633404 
L 99.01029 263.593584 
L 101.379039 264.142844 
L 106.116537 264.68194 
L 113.222784 265.308159 
L 117.960281 266.003738 
L 122.697779 267.032919 
L 127.435277 268.4261 
L 132.172775 270.195603 
L 136.910272 272.349419 
L 141.64777 274.896401 
L 146.385268 277.846922 
L 151.122766 281.209817 
L 155.860263 284.984094 
L 160.597761 289.141081 
L 167.704008 295.860559 
L 172.441505 300.26901 
L 174.810254 302.263445 
L 177.179003 303.99569 
L 179.547752 305.36297 
L 181.916501 306.270813 
L 184.343024 306.655273 
L 186.769547 306.465825 
L 189.196071 305.736112 
L 191.622594 304.545912 
L 194.049117 303.000497 
L 196.47564 301.209068 
L 201.328687 297.261095 
L 208.608257 291.338229 
L 213.461303 287.749167 
L 218.314349 284.549702 
L 223.167396 281.754035 
L 228.020442 279.351176 
L 232.873489 277.321325 
L 237.726535 275.643409 
L 242.579582 274.298387 
L 247.432628 273.270594 
L 252.285675 272.548217 
L 257.138721 272.123438 
L 261.991768 271.992478 
L 266.844814 272.155696 
L 271.697861 272.617811 
L 276.550907 273.388323 
L 281.403954 274.482191 
L 286.257 275.920899 
L 291.110047 277.734042 
L 295.963093 279.961712 
L 300.816139 282.658096 
L 303.242663 284.204152 
L 305.669186 285.897024 
L 308.095709 287.750438 
L 310.522232 289.780705 
L 312.948756 292.007349 
L 315.375279 294.453902 
L 317.801802 297.148948 
L 320.228325 300.127455 
L 322.654849 303.432461 
L 325.081372 307.117069 
L 327.507895 311.246432 
L 329.934418 315.898477 
L 332.360942 321.159353 
L 334.787465 327.101062 
L 342.067034 346.66628 
L 344.493558 349.422157 
L 346.920081 347.147382 
L 349.346604 341.256925 
L 354.199651 327.462609 
L 356.626174 321.323767 
L 359.052697 315.871835 
L 361.47922 311.040386 
L 363.905744 306.743375 
L 366.332267 302.901571 
L 368.75879 299.448376 
L 371.185313 296.329449 
L 373.611837 293.500778 
L 376.03836 290.926662 
L 378.464883 288.577982 
L 380.891406 286.430822 
L 383.317929 284.465402 
L 385.744453 282.665241 
L 388.170976 281.016531 
L 393.024022 278.128682 
L 397.877069 275.728399 
L 402.730115 273.762502 
L 407.583162 272.192452 
L 412.436208 270.990746 
L 417.289255 270.138588 
L 422.142301 269.624426 
L 426.995348 269.4431 
L 431.848394 269.595473 
L 436.701441 270.088486 
L 441.554487 270.935648 
L 446.407534 272.158028 
L 451.26058 273.785912 
L 456.113627 275.861396 
L 458.54015 277.08432 
L 460.966673 278.442406 
L 463.393196 279.946394 
L 465.819719 281.608991 
L 468.246243 283.445299 
L 470.672766 285.473383 
L 473.099289 287.714991 
L 475.525812 290.196502 
L 477.952336 292.950144 
L 480.378859 296.015566 
L 482.805382 299.441769 
L 485.231905 303.289277 
L 487.658429 307.631845 
L 490.084952 312.555327 
L 492.511475 318.146054 
L 494.937998 324.445144 
L 499.791045 337.966327 
L 502.217568 342.478609 
L 504.644091 342.315634 
L 507.070614 337.512181 
L 514.350184 317.032845 
L 516.776707 311.240783 
L 519.203231 306.118587 
L 521.629754 301.58051 
L 524.056277 297.541755 
L 526.4828 293.92976 
L 528.909324 290.68524 
L 531.335847 287.760469 
L 533.76237 285.117119 
L 536.188893 282.724353 
L 538.615417 280.557273 
L 541.04194 278.595727 
L 543.468463 276.823385 
L 545.894986 275.227047 
L 548.321509 273.796107 
L 550.748033 272.522155 
L 553.174556 271.398674 
L 555.601079 270.42082 
L 558.027602 269.58526 
L 560.454126 268.890067 
L 562.880649 268.334669 
L 565.307172 267.919822 
L 567.733695 267.647646 
L 570.160219 267.521694 
L 572.586742 267.547068 
L 575.013265 267.730599 
L 577.439788 268.081094 
L 579.866312 268.609667 
L 582.23506 269.310684 
L 584.603809 270.210605 
L 586.972558 271.329211 
L 589.341307 272.691051 
L 591.710056 274.326805 
L 594.078805 276.275188 
L 596.447554 278.585653 
L 598.816303 281.322321 
L 601.185051 284.569759 
L 603.5538 288.441561 
L 605.922549 293.092858 
L 608.291298 298.736739 
L 610.660047 305.65534 
L 613.028796 314.143032 
L 615.397545 324.032667 
L 617.766294 332.458777 
L 620.135042 331.754656 
L 622.503791 321.997544 
L 624.87254 310.858773 
L 627.241289 301.175846 
L 629.610038 293.124817 
L 631.871116 286.71044 
L 634.132195 281.30717 
L 636.393273 276.749522 
L 638.654352 272.924733 
L 640.91543 269.760625 
L 643.176509 267.216778 
L 645.437587 265.279498 
L 647.698666 263.960234 
L 649.959744 263.29741 
L 652.220823 263.362549 
L 654.481901 264.273062 
L 656.74298 266.21727 
L 659.004058 269.505226 
L 661.265136 274.682231 
L 663.526215 282.825474 
L 665.787293 296.55091 
L 670.30945 356.624439 
L 672.570529 298.266346 
L 674.831607 274.355524 
L 677.092686 258.49001 
L 679.353764 246.411187 
L 679.353764 246.411187 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
   </g>
   <g id="line2d_40">
    <path d="M 82.429048 31.354947 
L 103.747788 31.400871 
L 577.439788 32.252097 
L 598.816303 32.5318 
L 613.028796 32.914599 
L 622.503791 33.347147 
L 629.610038 33.829664 
L 636.393273 34.48544 
L 640.91543 35.074148 
L 645.437587 35.829773 
L 649.959744 36.807558 
L 654.481901 38.083082 
L 659.004058 39.759874 
L 661.265136 40.791145 
L 663.526215 41.9794 
L 665.787293 43.35032 
L 668.048372 44.933674 
L 670.30945 46.763848 
L 672.570529 48.880385 
L 674.831607 51.328518 
L 677.092686 54.159581 
L 679.353764 57.431043 
L 679.353764 57.431043 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
    <g clip-path="url(#pf67e761123)">
     <use xlink:href="#mf021a1f644" x="82.429048" y="31.354947" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="117.960281" y="31.421237" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="153.491514" y="31.462927" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="189.196071" y="31.510048" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="225.593919" y="31.566618" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="261.991768" y="31.628704" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="298.389616" y="31.69283" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="334.787465" y="31.756211" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="371.185313" y="31.816797" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="407.583162" y="31.87314" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="443.98101" y="31.924537" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="480.378859" y="31.972254" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="516.776707" y="32.024302" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="553.174556" y="32.113875" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="589.341307" y="32.378657" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="624.87254" y="33.489519" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="659.004058" y="39.759874" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
    </g>
   </g>
   <g id="line2d_41">
    <path d="M 82.429048 31.394663 
L 181.916501 31.505655 
L 582.23506 32.288539 
L 601.185051 32.572799 
L 615.397545 33.002319 
L 624.87254 33.491629 
L 631.871116 34.029964 
L 638.654352 34.773276 
L 643.176509 35.44223 
L 647.698666 36.302681 
L 652.220823 37.4186 
L 656.74298 38.877677 
L 659.004058 39.77191 
L 661.265136 40.800157 
L 663.526215 41.98442 
L 665.787293 43.35033 
L 668.048372 44.927661 
L 670.30945 46.750882 
L 672.570529 48.85973 
L 674.831607 51.299777 
L 677.092686 54.122882 
L 679.353764 57.387245 
L 679.353764 57.387245 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
   </g>
   <g id="line2d_42">
    <path d="M 82.429048 269.517759 
L 84.797797 286.108956 
L 87.166546 291.858352 
L 89.535295 294.227658 
L 94.272793 297.995064 
L 96.641542 300.036833 
L 101.379039 304.540323 
L 110.854035 313.930138 
L 113.222784 315.984843 
L 115.591533 317.779934 
L 117.960281 319.252504 
L 120.32903 320.356645 
L 122.697779 321.070592 
L 125.066528 321.399968 
L 127.435277 321.375719 
L 129.804026 321.047467 
L 132.172775 320.474703 
L 134.541523 319.718503 
L 139.279021 317.874317 
L 148.754017 313.919138 
L 153.491514 312.147744 
L 158.229012 310.610928 
L 162.96651 309.331135 
L 167.704008 308.312748 
L 172.441505 307.551505 
L 177.179003 307.039597 
L 181.916501 306.768334 
L 186.769547 306.731377 
L 191.622594 306.930663 
L 196.47564 307.360012 
L 201.328687 308.014735 
L 206.181733 308.891487 
L 211.03478 309.98798 
L 215.887826 311.302525 
L 220.740873 312.833333 
L 225.593919 314.577436 
L 230.446966 316.52902 
L 235.300012 318.676852 
L 242.579582 322.217652 
L 259.565244 330.934806 
L 264.418291 333.006435 
L 266.844814 333.858905 
L 269.271337 334.555124 
L 271.697861 335.072559 
L 274.124384 335.393531 
L 276.550907 335.50707 
L 278.97743 335.410153 
L 281.403954 335.108053 
L 283.830477 334.613708 
L 286.257 333.946265 
L 288.683523 333.129121 
L 293.53657 331.148318 
L 298.389616 328.871185 
L 315.375279 320.621795 
L 320.228325 318.512653 
L 325.081372 316.579346 
L 329.934418 314.833558 
L 334.787465 313.280086 
L 339.640511 311.919783 
L 344.493558 310.751451 
L 349.346604 309.773039 
L 354.199651 308.982399 
L 359.052697 308.377757 
L 363.905744 307.958012 
L 368.75879 307.722924 
L 373.611837 307.673239 
L 378.464883 307.810764 
L 383.317929 308.138418 
L 388.170976 308.660238 
L 393.024022 309.38135 
L 397.877069 310.307863 
L 402.730115 311.446628 
L 407.583162 312.804783 
L 412.436208 314.388909 
L 417.289255 316.203524 
L 422.142301 318.248474 
L 426.995348 320.514481 
L 434.274917 324.264348 
L 446.407534 330.749335 
L 448.834057 331.894633 
L 451.26058 332.913932 
L 453.687103 333.767826 
L 456.113627 334.417235 
L 458.54015 334.827217 
L 460.966673 334.971186 
L 463.393196 334.834653 
L 465.819719 334.417457 
L 468.246243 333.733842 
L 470.672766 332.810326 
L 473.099289 331.682037 
L 477.952336 328.969733 
L 482.805382 325.903187 
L 492.511475 319.620489 
L 497.364522 316.680813 
L 502.217568 313.972095 
L 507.070614 311.527753 
L 511.923661 309.365336 
L 516.776707 307.494205 
L 521.629754 305.920309 
L 526.4828 304.64915 
L 531.335847 303.687711 
L 536.188893 303.045793 
L 541.04194 302.737082 
L 545.894986 302.780109 
L 550.748033 303.199192 
L 555.601079 304.025346 
L 558.027602 304.602771 
L 560.454126 305.29693 
L 562.880649 306.113728 
L 565.307172 307.059373 
L 567.733695 308.140181 
L 570.160219 309.362212 
L 572.586742 310.730671 
L 575.013265 312.248913 
L 577.439788 313.91683 
L 582.23506 317.619773 
L 591.710056 325.43513 
L 594.078805 326.936505 
L 596.447554 327.927426 
L 598.816303 328.228816 
L 601.185051 327.725784 
L 603.5538 326.412596 
L 605.922549 324.396412 
L 608.291298 321.85693 
L 613.028796 315.972758 
L 617.766294 309.977487 
L 620.135042 307.161056 
L 622.503791 304.530489 
L 624.87254 302.115628 
L 627.241289 299.938699 
L 629.610038 298.018665 
L 631.871116 296.44293 
L 634.132195 295.137123 
L 636.393273 294.123858 
L 638.654352 293.431689 
L 640.91543 293.097384 
L 643.176509 293.168841 
L 645.437587 293.708993 
L 647.698666 294.801155 
L 649.959744 296.556128 
L 652.220823 299.120413 
L 654.481901 302.680018 
L 656.74298 307.432917 
L 659.004058 313.411856 
L 661.265136 319.716802 
L 663.526215 322.644496 
L 665.787293 317.510437 
L 668.048372 306.883888 
L 670.30945 295.711196 
L 672.570529 285.808915 
L 674.831607 277.452767 
L 677.092686 270.604157 
L 679.353764 265.258961 
L 679.353764 265.258961 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
   </g>
   <g id="line2d_43">
    <path d="M 82.429048 57.631169 
L 89.535295 57.722714 
L 196.47564 58.173863 
L 293.53657 58.718786 
L 378.464883 59.398051 
L 436.701441 60.070514 
L 482.805382 60.819276 
L 526.4828 61.753762 
L 565.307172 62.550045 
L 579.866312 62.611925 
L 589.341307 62.451766 
L 596.447554 62.160196 
L 603.5538 61.65751 
L 610.660047 60.868536 
L 615.397545 60.139109 
L 620.135042 59.212182 
L 624.87254 58.056792 
L 629.610038 56.64318 
L 634.132195 55.029292 
L 638.654352 53.140778 
L 643.176509 50.971185 
L 647.698666 48.52674 
L 652.220823 45.82948 
L 659.004058 41.40455 
L 672.570529 32.173337 
L 677.092686 29.402143 
L 679.353764 28.148814 
L 679.353764 28.148814 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
    <g clip-path="url(#pf67e761123)">
     <use xlink:href="#mf021a1f644" x="82.429048" y="57.631169" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="117.960281" y="57.877388" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="153.491514" y="58.003492" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="189.196071" y="58.141841" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="225.593919" y="58.314956" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="261.991768" y="58.518643" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="298.389616" y="58.751621" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="334.787465" y="59.017955" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="371.185313" y="59.328922" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="407.583162" y="59.70434" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="443.98101" y="60.173597" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="480.378859" y="60.773745" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="516.776707" y="61.530389" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="553.174556" y="62.349624" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="589.341307" y="62.451766" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="624.87254" y="58.056792" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="659.004058" y="41.40455" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
    </g>
   </g>
   <g id="line2d_44">
    <path d="M 82.429048 57.810327 
L 136.910272 57.91891 
L 237.726535 58.405483 
L 349.346604 59.104406 
L 410.009685 59.700499 
L 458.54015 60.383211 
L 502.217568 61.21052 
L 570.160219 62.596111 
L 582.23506 62.588088 
L 591.710056 62.373467 
L 598.816303 62.02238 
L 605.922549 61.438629 
L 610.660047 60.881061 
L 615.397545 60.15677 
L 620.135042 59.235588 
L 624.87254 58.086349 
L 629.610038 56.678994 
L 634.132195 55.070791 
L 638.654352 53.187202 
L 643.176509 51.021182 
L 647.698666 48.578236 
L 652.220823 45.879534 
L 656.74298 42.964837 
L 665.787293 36.754615 
L 672.570529 32.152292 
L 677.092686 29.346981 
L 679.353764 28.074716 
L 679.353764 28.074716 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
   </g>
   <g id="line2d_45">
    <path d="M 82.429048 213.77101 
L 84.797797 227.291076 
L 87.166546 231.453211 
L 89.535295 233.272898 
L 94.272793 236.518603 
L 96.641542 238.366024 
L 99.01029 240.390511 
L 103.747788 244.912804 
L 108.485286 249.959567 
L 113.222784 255.427342 
L 117.960281 261.219311 
L 127.435277 273.049513 
L 129.804026 275.791829 
L 132.172775 278.297055 
L 134.541523 280.467683 
L 136.910272 282.208552 
L 139.279021 283.444599 
L 141.64777 284.138668 
L 144.016519 284.301595 
L 146.385268 283.989336 
L 148.754017 283.288667 
L 151.122766 282.29847 
L 155.860263 279.816377 
L 165.335259 274.562025 
L 170.072757 272.284956 
L 174.810254 270.355303 
L 179.547752 268.786061 
L 184.343024 267.556584 
L 189.196071 266.663611 
L 194.049117 266.106384 
L 198.902164 265.869222 
L 203.75521 265.939934 
L 208.608257 266.310434 
L 213.461303 266.977066 
L 218.314349 267.940886 
L 223.167396 269.208077 
L 228.020442 270.790598 
L 232.873489 272.707199 
L 237.726535 274.984996 
L 242.579582 277.661851 
L 247.432628 280.790054 
L 249.859152 282.545054 
L 252.285675 284.442093 
L 254.712198 286.494912 
L 257.138721 288.719991 
L 259.565244 291.137259 
L 261.991768 293.771044 
L 264.418291 296.651353 
L 266.844814 299.815657 
L 269.271337 303.311359 
L 271.697861 307.199294 
L 274.124384 311.558639 
L 276.550907 316.493663 
L 278.97743 322.142069 
L 281.403954 328.681166 
L 283.830477 336.312016 
L 286.257 345.131021 
L 288.683523 354.524062 
L 291.110047 361.359454 
L 293.53657 360.293583 
L 295.963093 352.4503 
L 298.389616 343.196091 
L 300.816139 334.811865 
L 303.242663 327.619273 
L 305.669186 321.468294 
L 308.095709 316.158968 
L 310.522232 311.523832 
L 312.948756 307.434245 
L 315.375279 303.792832 
L 317.801802 300.525424 
L 320.228325 297.574793 
L 322.654849 294.896137 
L 325.081372 292.453877 
L 327.507895 290.21938 
L 329.934418 288.169322 
L 332.360942 286.284501 
L 337.213988 282.949283 
L 342.067034 280.114063 
L 346.920081 277.706991 
L 351.773127 275.675198 
L 356.626174 273.979295 
L 361.47922 272.589816 
L 366.332267 271.484875 
L 371.185313 270.648592 
L 376.03836 270.070039 
L 380.891406 269.74255 
L 385.744453 269.663313 
L 390.597499 269.833163 
L 395.450546 270.256593 
L 400.303592 270.94193 
L 405.156639 271.901728 
L 410.009685 273.153391 
L 414.862732 274.720082 
L 419.715778 276.63201 
L 424.568824 278.928168 
L 429.421871 281.658651 
L 431.848394 283.20615 
L 434.274917 284.887513 
L 436.701441 286.713423 
L 439.127964 288.695744 
L 441.554487 290.847454 
L 443.98101 293.182365 
L 446.407534 295.714456 
L 448.834057 298.456457 
L 451.26058 301.417064 
L 453.687103 304.595635 
L 458.54015 311.49181 
L 460.966673 315.034199 
L 463.393196 318.37951 
L 465.819719 321.177535 
L 468.246243 322.976688 
L 470.672766 323.371532 
L 473.099289 322.224418 
L 475.525812 319.757813 
L 477.952336 316.409563 
L 487.658429 301.134456 
L 490.084952 297.646801 
L 492.511475 294.38231 
L 494.937998 291.339121 
L 497.364522 288.507707 
L 499.791045 285.875402 
L 502.217568 283.428776 
L 504.644091 281.154815 
L 507.070614 279.041477 
L 511.923661 275.254476 
L 516.776707 271.995368 
L 521.629754 269.209328 
L 526.4828 266.856349 
L 531.335847 264.908761 
L 536.188893 263.349418 
L 541.04194 262.170575 
L 545.894986 261.373403 
L 550.748033 260.968119 
L 555.601079 260.974795 
L 558.027602 261.141955 
L 560.454126 261.42494 
L 562.880649 261.829905 
L 565.307172 262.364119 
L 567.733695 263.036136 
L 570.160219 263.855998 
L 572.586742 264.835485 
L 575.013265 265.988421 
L 577.439788 267.331024 
L 579.866312 268.882329 
L 582.23506 270.619329 
L 584.603809 272.600608 
L 586.972558 274.853959 
L 589.341307 277.410663 
L 591.710056 280.304308 
L 594.078805 283.567161 
L 596.447554 287.220966 
L 598.816303 291.255469 
L 603.5538 299.938619 
L 605.922549 303.759357 
L 608.291298 306.115854 
L 610.660047 306.086606 
L 613.028796 303.515276 
L 615.397545 299.174785 
L 622.503791 283.993331 
L 624.87254 279.51688 
L 627.241289 275.501961 
L 629.610038 271.947449 
L 631.871116 268.970503 
L 634.132195 266.38594 
L 636.393273 264.183932 
L 638.654352 262.360917 
L 640.91543 260.921246 
L 643.176509 259.878885 
L 645.437587 259.259751 
L 647.698666 259.105282 
L 649.959744 259.478119 
L 652.220823 260.471512 
L 654.481901 262.225581 
L 656.74298 264.957306 
L 659.004058 269.020786 
L 661.265136 275.043662 
L 663.526215 284.294709 
L 665.787293 299.995311 
L 668.048372 335.288883 
L 670.30945 330.135718 
L 672.570529 290.317063 
L 674.831607 269.115065 
L 677.092686 254.215596 
L 679.353764 242.536671 
L 679.353764 242.536671 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
   </g>
   <g id="line2d_46">
    <path d="M 82.429048 31.361323 
L 115.591533 31.422783 
L 528.909324 32.017427 
L 575.013265 32.225829 
L 596.447554 32.502578 
L 610.660047 32.870414 
L 622.503791 33.412457 
L 629.610038 33.918163 
L 636.393273 34.602974 
L 640.91543 35.216322 
L 645.437587 36.002436 
L 649.959744 37.018674 
L 654.481901 38.343806 
L 656.74298 39.154673 
L 659.004058 40.086361 
L 661.265136 41.158949 
L 663.526215 42.396012 
L 665.787293 43.825215 
L 668.048372 45.479015 
L 670.30945 47.395514 
L 672.570529 49.619498 
L 674.831607 52.203768 
L 677.092686 55.210876 
L 679.353764 58.715372 
L 679.353764 58.715372 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
    <g clip-path="url(#pf67e761123)">
     <use xlink:href="#mf021a1f644" x="82.429048" y="31.361323" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="117.960281" y="31.425567" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="153.491514" y="31.460915" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="189.196071" y="31.500748" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="225.593919" y="31.549536" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="261.991768" y="31.604272" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="298.389616" y="31.662101" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="334.787465" y="31.720556" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="371.185313" y="31.777712" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="407.583162" y="31.832217" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="443.98101" y="31.8836" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="480.378859" y="31.933686" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="516.776707" y="31.991729" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="553.174556" y="32.093796" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="589.341307" y="32.385043" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="624.87254" y="33.561894" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="659.004058" y="40.086361" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
    </g>
   </g>
   <g id="line2d_47">
    <path d="M 82.429048 31.401347 
L 194.049117 31.515133 
L 567.733695 32.165552 
L 594.078805 32.45168 
L 608.291298 32.787718 
L 620.135042 33.279786 
L 629.610038 33.925247 
L 636.393273 34.61494 
L 640.91543 35.231235 
L 645.437587 36.019578 
L 649.959744 37.036751 
L 654.481901 38.360782 
L 656.74298 39.170065 
L 659.004058 40.099324 
L 661.265136 41.168531 
L 663.526215 42.401166 
L 665.787293 43.824832 
L 668.048372 45.471979 
L 670.30945 47.380778 
L 672.570529 49.5962 
L 674.831607 52.171386 
L 677.092686 55.169427 
L 679.353764 58.665671 
L 679.353764 58.665671 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
   </g>
   <g id="line2d_48">
    <path d="M 82.429048 269.231358 
L 84.797797 280.257076 
L 87.166546 282.607585 
L 89.535295 284.303304 
L 91.904044 286.250439 
L 94.272793 288.525974 
L 96.641542 291.097083 
L 99.01029 293.908857 
L 103.747788 300.019305 
L 110.854035 309.468516 
L 113.222784 312.399674 
L 115.591533 315.070915 
L 117.960281 317.383879 
L 120.32903 319.249624 
L 122.697779 320.604796 
L 125.066528 321.425809 
L 127.435277 321.734425 
L 129.804026 321.591685 
L 132.172775 321.083168 
L 134.541523 320.302137 
L 139.279021 318.259899 
L 148.754017 313.819595 
L 153.491514 311.877773 
L 158.229012 310.231011 
L 162.96651 308.894815 
L 167.704008 307.864551 
L 172.441505 307.12767 
L 177.179003 306.669623 
L 181.916501 306.476619 
L 186.769547 306.541369 
L 191.622594 306.86175 
L 196.47564 307.430456 
L 201.328687 308.242907 
L 206.181733 309.297115 
L 211.03478 310.593493 
L 215.887826 312.134585 
L 220.740873 313.92466 
L 225.593919 315.969022 
L 230.446966 318.272795 
L 235.300012 320.838757 
L 240.153059 323.663471 
L 245.006105 326.730488 
L 252.285675 331.684409 
L 259.565244 336.732705 
L 264.418291 339.794335 
L 266.844814 341.110832 
L 269.271337 342.218657 
L 271.697861 343.069138 
L 274.124384 343.620818 
L 276.550907 343.845546 
L 278.97743 343.733186 
L 281.403954 343.293426 
L 283.830477 342.554029 
L 286.257 341.556148 
L 288.683523 340.348259 
L 293.53657 337.499423 
L 310.522232 326.628278 
L 315.375279 323.859343 
L 320.228325 321.332791 
L 325.081372 319.055578 
L 329.934418 317.025162 
L 334.787465 315.234608 
L 339.640511 313.675434 
L 344.493558 312.339159 
L 349.346604 311.218123 
L 354.199651 310.30591 
L 359.052697 309.597548 
L 363.905744 309.089615 
L 368.75879 308.780273 
L 373.611837 308.669308 
L 378.464883 308.758154 
L 383.317929 309.049937 
L 388.170976 309.54953 
L 393.024022 310.26361 
L 397.877069 311.200717 
L 402.730115 312.371255 
L 407.583162 313.787398 
L 412.436208 315.46275 
L 417.289255 317.411542 
L 422.142301 319.64689 
L 426.995348 322.177311 
L 431.848394 324.999929 
L 436.701441 328.087748 
L 451.26058 337.742408 
L 453.687103 339.046347 
L 456.113627 340.112318 
L 458.54015 340.873721 
L 460.966673 341.273099 
L 463.393196 341.272213 
L 465.819719 340.85986 
L 468.246243 340.054372 
L 470.672766 338.89972 
L 473.099289 337.456892 
L 475.525812 335.793775 
L 480.378859 332.064101 
L 490.084952 324.293344 
L 494.937998 320.677768 
L 499.791045 317.365233 
L 504.644091 314.385728 
L 509.497138 311.747782 
L 514.350184 309.450535 
L 519.203231 307.490285 
L 524.056277 305.86402 
L 528.909324 304.571386 
L 533.76237 303.615894 
L 538.615417 303.005816 
L 543.468463 302.755021 
L 548.321509 302.883899 
L 553.174556 303.420457 
L 555.601079 303.852754 
L 558.027602 304.401554 
L 560.454126 305.073043 
L 562.880649 305.874071 
L 565.307172 306.812122 
L 567.733695 307.895213 
L 570.160219 309.131669 
L 572.586742 310.529694 
L 575.013265 312.096594 
L 577.439788 313.83742 
L 579.866312 315.752653 
L 584.603809 319.951299 
L 591.710056 326.677649 
L 594.078805 328.570203 
L 596.447554 329.940776 
L 598.816303 330.541067 
L 601.185051 330.187952 
L 603.5538 328.842938 
L 605.922549 326.632315 
L 608.291298 323.790177 
L 613.028796 317.197107 
L 617.766294 310.55699 
L 620.135042 307.464984 
L 622.503791 304.588157 
L 624.87254 301.952124 
L 627.241289 299.574485 
L 629.610038 297.469743 
L 631.871116 295.728853 
L 634.132195 294.26546 
L 636.393273 293.099113 
L 638.654352 292.255384 
L 640.91543 291.767991 
L 643.176509 291.681504 
L 645.437587 292.055 
L 647.698666 292.967046 
L 649.959744 294.522338 
L 652.220823 296.859422 
L 654.481901 300.154745 
L 656.74298 304.599915 
L 659.004058 310.251985 
L 661.265136 316.379396 
L 663.526215 319.727346 
L 665.787293 315.426598 
L 668.048372 305.064852 
L 670.30945 293.672996 
L 672.570529 283.373323 
L 674.831607 274.536837 
L 677.092686 267.127265 
L 679.353764 261.10982 
L 679.353764 261.10982 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
   </g>
   <g id="line2d_49">
    <path d="M 82.429048 31.871748 
L 103.747788 32.300875 
L 117.960281 32.842272 
L 132.172775 33.60755 
L 148.754017 34.753211 
L 165.335259 36.130007 
L 184.343024 37.930351 
L 208.608257 40.461346 
L 293.53657 49.553625 
L 315.375279 51.57644 
L 334.787465 53.172463 
L 354.199651 54.55049 
L 373.611837 55.693566 
L 393.024022 56.595385 
L 412.436208 57.260673 
L 431.848394 57.70458 
L 453.687103 57.969476 
L 477.952336 58.024258 
L 507.070614 57.848157 
L 545.894986 57.365442 
L 575.013265 56.816213 
L 591.710056 56.280906 
L 601.185051 55.798142 
L 610.660047 55.07683 
L 617.766294 54.294519 
L 622.503791 53.613034 
L 627.241289 52.768829 
L 631.871116 51.754232 
L 636.393273 50.550386 
L 640.91543 49.106295 
L 645.437587 47.397312 
L 649.959744 45.408932 
L 654.481901 43.142909 
L 659.004058 40.623693 
L 665.787293 36.494212 
L 674.831607 30.855269 
L 679.353764 28.277219 
L 679.353764 28.277219 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
    <g clip-path="url(#pf67e761123)">
     <use xlink:href="#mf021a1f644" x="82.429048" y="31.871748" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="117.960281" y="32.842272" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="153.491514" y="35.125047" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="189.196071" y="38.41959" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="225.593919" y="42.317613" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="261.991768" y="46.30786" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="298.389616" y="50.021697" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="334.787465" y="53.172463" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="371.185313" y="55.563853" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="407.583162" y="57.115863" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="443.98101" y="57.879955" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="480.378859" y="58.018307" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="516.776707" y="57.748512" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="553.174556" y="57.250057" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="589.341307" y="56.376201" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="624.87254" y="53.213316" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="659.004058" y="40.623693" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
    </g>
   </g>
   <g id="line2d_50">
    <path d="M 82.429048 31.986168 
L 94.272793 32.07946 
L 108.485286 32.433542 
L 122.697779 33.038664 
L 136.910272 33.872896 
L 153.491514 35.097988 
L 170.072757 36.545812 
L 191.622594 38.673808 
L 220.740873 41.81007 
L 276.550907 47.878638 
L 300.816139 50.276925 
L 322.654849 52.211423 
L 342.067034 53.715592 
L 361.47922 54.994434 
L 380.891406 56.036495 
L 400.303592 56.841005 
L 419.715778 57.417712 
L 439.127964 57.785718 
L 460.966673 57.983357 
L 485.231905 57.986745 
L 516.776707 57.751524 
L 555.601079 57.224418 
L 579.866312 56.701293 
L 594.078805 56.186111 
L 603.5538 55.651876 
L 610.660047 55.082979 
L 617.766294 54.301219 
L 622.503791 53.621232 
L 627.241289 52.779623 
L 631.871116 51.768692 
L 636.393273 50.569414 
L 640.91543 49.130553 
L 645.437587 47.426839 
L 649.959744 45.442754 
L 654.481901 43.178627 
L 659.004058 40.657103 
L 665.787293 36.511885 
L 674.831607 30.820838 
L 679.353764 28.203235 
L 679.353764 28.203235 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
   </g>
   <g id="line2d_51">
    <path d="M 82.429048 230.351535 
L 84.797797 236.466592 
L 87.166546 246.244834 
L 89.535295 253.522683 
L 91.904044 258.158995 
L 94.272793 260.972157 
L 96.641542 262.633404 
L 99.01029 263.593584 
L 101.379039 264.142844 
L 106.116537 264.68194 
L 113.222784 265.308159 
L 117.960281 266.003738 
L 122.697779 267.032919 
L 127.435277 268.4261 
L 132.172775 270.195603 
L 136.910272 272.349419 
L 141.64777 274.896401 
L 146.385268 277.846922 
L 151.122766 281.209817 
L 155.860263 284.984094 
L 160.597761 289.141081 
L 167.704008 295.860559 
L 172.441505 300.26901 
L 174.810254 302.263445 
L 177.179003 303.99569 
L 179.547752 305.36297 
L 181.916501 306.270813 
L 184.343024 306.655273 
L 186.769547 306.465825 
L 189.196071 305.736112 
L 191.622594 304.545912 
L 194.049117 303.000497 
L 196.47564 301.209068 
L 201.328687 297.261095 
L 208.608257 291.338229 
L 213.461303 287.749167 
L 218.314349 284.549702 
L 223.167396 281.754035 
L 228.020442 279.351176 
L 232.873489 277.321325 
L 237.726535 275.643409 
L 242.579582 274.298387 
L 247.432628 273.270594 
L 252.285675 272.548217 
L 257.138721 272.123438 
L 261.991768 271.992478 
L 266.844814 272.155696 
L 271.697861 272.617811 
L 276.550907 273.388323 
L 281.403954 274.482191 
L 286.257 275.920899 
L 291.110047 277.734042 
L 295.963093 279.961712 
L 300.816139 282.658096 
L 303.242663 284.204152 
L 305.669186 285.897024 
L 308.095709 287.750438 
L 310.522232 289.780705 
L 312.948756 292.007349 
L 315.375279 294.453902 
L 317.801802 297.148948 
L 320.228325 300.127455 
L 322.654849 303.432461 
L 325.081372 307.117069 
L 327.507895 311.246432 
L 329.934418 315.898477 
L 332.360942 321.159353 
L 334.787465 327.101062 
L 342.067034 346.66628 
L 344.493558 349.422157 
L 346.920081 347.147382 
L 349.346604 341.256925 
L 354.199651 327.462609 
L 356.626174 321.323767 
L 359.052697 315.871835 
L 361.47922 311.040386 
L 363.905744 306.743375 
L 366.332267 302.901571 
L 368.75879 299.448376 
L 371.185313 296.329449 
L 373.611837 293.500778 
L 376.03836 290.926662 
L 378.464883 288.577982 
L 380.891406 286.430822 
L 383.317929 284.465402 
L 385.744453 282.665241 
L 388.170976 281.016531 
L 393.024022 278.128682 
L 397.877069 275.728399 
L 402.730115 273.762502 
L 407.583162 272.192452 
L 412.436208 270.990746 
L 417.289255 270.138588 
L 422.142301 269.624426 
L 426.995348 269.4431 
L 431.848394 269.595473 
L 436.701441 270.088486 
L 441.554487 270.935648 
L 446.407534 272.158028 
L 451.26058 273.785912 
L 456.113627 275.861396 
L 458.54015 277.08432 
L 460.966673 278.442406 
L 463.393196 279.946394 
L 465.819719 281.608991 
L 468.246243 283.445299 
L 470.672766 285.473383 
L 473.099289 287.714991 
L 475.525812 290.196502 
L 477.952336 292.950144 
L 480.378859 296.015566 
L 482.805382 299.441769 
L 485.231905 303.289277 
L 487.658429 307.631845 
L 490.084952 312.555327 
L 492.511475 318.146054 
L 494.937998 324.445144 
L 499.791045 337.966327 
L 502.217568 342.478609 
L 504.644091 342.315634 
L 507.070614 337.512181 
L 514.350184 317.032845 
L 516.776707 311.240783 
L 519.203231 306.118587 
L 521.629754 301.58051 
L 524.056277 297.541755 
L 526.4828 293.92976 
L 528.909324 290.68524 
L 531.335847 287.760469 
L 533.76237 285.117119 
L 536.188893 282.724353 
L 538.615417 280.557273 
L 541.04194 278.595727 
L 543.468463 276.823385 
L 545.894986 275.227047 
L 548.321509 273.796107 
L 550.748033 272.522155 
L 553.174556 271.398674 
L 555.601079 270.42082 
L 558.027602 269.58526 
L 560.454126 268.890067 
L 562.880649 268.334669 
L 565.307172 267.919822 
L 567.733695 267.647646 
L 570.160219 267.521694 
L 572.586742 267.547068 
L 575.013265 267.730599 
L 577.439788 268.081094 
L 579.866312 268.609667 
L 582.23506 269.310684 
L 584.603809 270.210605 
L 586.972558 271.329211 
L 589.341307 272.691051 
L 591.710056 274.326805 
L 594.078805 276.275188 
L 596.447554 278.585653 
L 598.816303 281.322321 
L 601.185051 284.569759 
L 603.5538 288.441561 
L 605.922549 293.092858 
L 608.291298 298.736739 
L 610.660047 305.65534 
L 613.028796 314.143032 
L 615.397545 324.032667 
L 617.766294 332.458777 
L 620.135042 331.754656 
L 622.503791 321.997544 
L 624.87254 310.858773 
L 627.241289 301.175846 
L 629.610038 293.124817 
L 631.871116 286.71044 
L 634.132195 281.30717 
L 636.393273 276.749522 
L 638.654352 272.924733 
L 640.91543 269.760625 
L 643.176509 267.216778 
L 645.437587 265.279498 
L 647.698666 263.960234 
L 649.959744 263.29741 
L 652.220823 263.362549 
L 654.481901 264.273062 
L 656.74298 266.21727 
L 659.004058 269.505226 
L 661.265136 274.682231 
L 663.526215 282.825474 
L 665.787293 296.55091 
L 670.30945 356.624439 
L 672.570529 298.266346 
L 674.831607 274.355524 
L 677.092686 258.49001 
L 679.353764 246.411187 
L 679.353764 246.411187 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
   </g>
   <g id="line2d_52">
    <path d="M 82.429048 31.361323 
L 115.591533 31.422783 
L 528.909324 32.017427 
L 575.013265 32.225829 
L 596.447554 32.502578 
L 610.660047 32.870414 
L 622.503791 33.412457 
L 629.610038 33.918163 
L 636.393273 34.602974 
L 640.91543 35.216322 
L 645.437587 36.002436 
L 649.959744 37.018674 
L 654.481901 38.343806 
L 656.74298 39.154673 
L 659.004058 40.086361 
L 661.265136 41.158949 
L 663.526215 42.396012 
L 665.787293 43.825215 
L 668.048372 45.479015 
L 670.30945 47.395514 
L 672.570529 49.619498 
L 674.831607 52.203768 
L 677.092686 55.210876 
L 679.353764 58.715372 
L 679.353764 58.715372 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
    <g clip-path="url(#pf67e761123)">
     <use xlink:href="#mf021a1f644" x="82.429048" y="31.361323" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="117.960281" y="31.425567" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="153.491514" y="31.460915" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="189.196071" y="31.500748" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="225.593919" y="31.549536" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="261.991768" y="31.604272" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="298.389616" y="31.662101" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="334.787465" y="31.720556" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="371.185313" y="31.777712" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="407.583162" y="31.832217" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="443.98101" y="31.8836" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="480.378859" y="31.933686" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="516.776707" y="31.991729" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="553.174556" y="32.093796" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="589.341307" y="32.385043" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="624.87254" y="33.561894" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="659.004058" y="40.086361" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
    </g>
   </g>
   <g id="line2d_53">
    <path d="M 82.429048 31.401347 
L 194.049117 31.515133 
L 567.733695 32.165552 
L 594.078805 32.45168 
L 608.291298 32.787718 
L 620.135042 33.279786 
L 629.610038 33.925247 
L 636.393273 34.61494 
L 640.91543 35.231235 
L 645.437587 36.019578 
L 649.959744 37.036751 
L 654.481901 38.360782 
L 656.74298 39.170065 
L 659.004058 40.099324 
L 661.265136 41.168531 
L 663.526215 42.401166 
L 665.787293 43.824832 
L 668.048372 45.471979 
L 670.30945 47.380778 
L 672.570529 49.5962 
L 674.831607 52.171386 
L 677.092686 55.169427 
L 679.353764 58.665671 
L 679.353764 58.665671 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
   </g>
   <g id="line2d_54">
    <path d="M 82.429048 269.231358 
L 84.797797 280.257076 
L 87.166546 282.607585 
L 89.535295 284.303304 
L 91.904044 286.250439 
L 94.272793 288.525974 
L 96.641542 291.097083 
L 99.01029 293.908857 
L 103.747788 300.019305 
L 110.854035 309.468516 
L 113.222784 312.399674 
L 115.591533 315.070915 
L 117.960281 317.383879 
L 120.32903 319.249624 
L 122.697779 320.604796 
L 125.066528 321.425809 
L 127.435277 321.734425 
L 129.804026 321.591685 
L 132.172775 321.083168 
L 134.541523 320.302137 
L 139.279021 318.259899 
L 148.754017 313.819595 
L 153.491514 311.877773 
L 158.229012 310.231011 
L 162.96651 308.894815 
L 167.704008 307.864551 
L 172.441505 307.12767 
L 177.179003 306.669623 
L 181.916501 306.476619 
L 186.769547 306.541369 
L 191.622594 306.86175 
L 196.47564 307.430456 
L 201.328687 308.242907 
L 206.181733 309.297115 
L 211.03478 310.593493 
L 215.887826 312.134585 
L 220.740873 313.92466 
L 225.593919 315.969022 
L 230.446966 318.272795 
L 235.300012 320.838757 
L 240.153059 323.663471 
L 245.006105 326.730488 
L 252.285675 331.684409 
L 259.565244 336.732705 
L 264.418291 339.794335 
L 266.844814 341.110832 
L 269.271337 342.218657 
L 271.697861 343.069138 
L 274.124384 343.620818 
L 276.550907 343.845546 
L 278.97743 343.733186 
L 281.403954 343.293426 
L 283.830477 342.554029 
L 286.257 341.556148 
L 288.683523 340.348259 
L 293.53657 337.499423 
L 310.522232 326.628278 
L 315.375279 323.859343 
L 320.228325 321.332791 
L 325.081372 319.055578 
L 329.934418 317.025162 
L 334.787465 315.234608 
L 339.640511 313.675434 
L 344.493558 312.339159 
L 349.346604 311.218123 
L 354.199651 310.30591 
L 359.052697 309.597548 
L 363.905744 309.089615 
L 368.75879 308.780273 
L 373.611837 308.669308 
L 378.464883 308.758154 
L 383.317929 309.049937 
L 388.170976 309.54953 
L 393.024022 310.26361 
L 397.877069 311.200717 
L 402.730115 312.371255 
L 407.583162 313.787398 
L 412.436208 315.46275 
L 417.289255 317.411542 
L 422.142301 319.64689 
L 426.995348 322.177311 
L 431.848394 324.999929 
L 436.701441 328.087748 
L 451.26058 337.742408 
L 453.687103 339.046347 
L 456.113627 340.112318 
L 458.54015 340.873721 
L 460.966673 341.273099 
L 463.393196 341.272213 
L 465.819719 340.85986 
L 468.246243 340.054372 
L 470.672766 338.89972 
L 473.099289 337.456892 
L 475.525812 335.793775 
L 480.378859 332.064101 
L 490.084952 324.293344 
L 494.937998 320.677768 
L 499.791045 317.365233 
L 504.644091 314.385728 
L 509.497138 311.747782 
L 514.350184 309.450535 
L 519.203231 307.490285 
L 524.056277 305.86402 
L 528.909324 304.571386 
L 533.76237 303.615894 
L 538.615417 303.005816 
L 543.468463 302.755021 
L 548.321509 302.883899 
L 553.174556 303.420457 
L 555.601079 303.852754 
L 558.027602 304.401554 
L 560.454126 305.073043 
L 562.880649 305.874071 
L 565.307172 306.812122 
L 567.733695 307.895213 
L 570.160219 309.131669 
L 572.586742 310.529694 
L 575.013265 312.096594 
L 577.439788 313.83742 
L 579.866312 315.752653 
L 584.603809 319.951299 
L 591.710056 326.677649 
L 594.078805 328.570203 
L 596.447554 329.940776 
L 598.816303 330.541067 
L 601.185051 330.187952 
L 603.5538 328.842938 
L 605.922549 326.632315 
L 608.291298 323.790177 
L 613.028796 317.197107 
L 617.766294 310.55699 
L 620.135042 307.464984 
L 622.503791 304.588157 
L 624.87254 301.952124 
L 627.241289 299.574485 
L 629.610038 297.469743 
L 631.871116 295.728853 
L 634.132195 294.26546 
L 636.393273 293.099113 
L 638.654352 292.255384 
L 640.91543 291.767991 
L 643.176509 291.681504 
L 645.437587 292.055 
L 647.698666 292.967046 
L 649.959744 294.522338 
L 652.220823 296.859422 
L 654.481901 300.154745 
L 656.74298 304.599915 
L 659.004058 310.251985 
L 661.265136 316.379396 
L 663.526215 319.727346 
L 665.787293 315.426598 
L 668.048372 305.064852 
L 670.30945 293.672996 
L 672.570529 283.373323 
L 674.831607 274.536837 
L 677.092686 267.127265 
L 679.353764 261.10982 
L 679.353764 261.10982 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
   </g>
   <g id="line2d_55">
    <path d="M 82.429048 58.689182 
L 89.535295 58.627668 
L 96.641542 58.320866 
L 103.747788 57.723677 
L 110.854035 56.877906 
L 120.32903 55.467353 
L 132.172775 53.428311 
L 170.072757 46.686634 
L 184.343024 44.465102 
L 198.902164 42.45518 
L 213.461303 40.700969 
L 228.020442 39.184129 
L 242.579582 37.88044 
L 259.565244 36.595264 
L 276.550907 35.528343 
L 295.963093 34.534035 
L 317.801802 33.653262 
L 342.067034 32.91478 
L 368.75879 32.336506 
L 397.877069 31.927653 
L 431.848394 31.680115 
L 470.672766 31.630268 
L 514.350184 31.798607 
L 560.454126 32.187535 
L 591.710056 32.633484 
L 610.660047 33.103538 
L 622.503791 33.592327 
L 631.871116 34.195129 
L 638.654352 34.836527 
L 645.437587 35.753819 
L 649.959744 36.59073 
L 654.481901 37.682333 
L 659.004058 39.119054 
L 661.265136 40.003754 
L 663.526215 41.023959 
L 665.787293 42.201792 
L 668.048372 43.562741 
L 670.30945 45.136037 
L 672.570529 46.955054 
L 674.831607 49.057744 
L 677.092686 51.487134 
L 679.353764 54.291945 
L 679.353764 54.291945 
" clip-path="url(#pf67e761123)" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
    <g clip-path="url(#pf67e761123)">
     <use xlink:href="#mf021a1f644" x="82.429048" y="58.689182" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="117.960281" y="55.843931" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="153.491514" y="49.54425" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="189.196071" y="43.766061" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="225.593919" y="39.421449" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="261.991768" y="36.430309" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="298.389616" y="34.424604" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="334.787465" y="33.112851" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="371.185313" y="32.294307" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="407.583162" y="31.834248" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="443.98101" y="31.640851" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="480.378859" y="31.649381" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="516.776707" y="31.813761" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="553.174556" y="32.110625" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="589.341307" y="32.589757" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="624.87254" y="33.721583" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     <use xlink:href="#mf021a1f644" x="659.004058" y="39.119054" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
    </g>
   </g>
   <g id="line2d_56">
    <path d="M 82.429048 59.01874 
L 89.535295 58.873709 
L 96.641542 58.448627 
L 103.747788 57.770296 
L 110.854035 56.878184 
L 120.32903 55.43618 
L 132.172775 53.381512 
L 167.704008 47.037202 
L 181.916501 44.793792 
L 196.47564 42.750544 
L 211.03478 40.96437 
L 225.593919 39.417854 
L 240.153059 38.086855 
L 257.138721 36.772437 
L 274.124384 35.678596 
L 293.53657 34.655886 
L 315.375279 33.745971 
L 339.640511 32.978913 
L 366.332267 32.374756 
L 395.450546 31.945439 
L 429.421871 31.684565 
L 468.246243 31.630022 
L 511.923661 31.795787 
L 560.454126 32.194156 
L 591.710056 32.628685 
L 610.660047 33.093253 
L 622.503791 33.581136 
L 631.871116 34.185159 
L 638.654352 34.828469 
L 645.437587 35.74822 
L 649.959744 36.586807 
L 654.481901 37.679887 
L 659.004058 39.117773 
L 661.265136 40.002984 
L 663.526215 41.023753 
L 665.787293 42.202365 
L 668.048372 43.564576 
L 670.30945 45.140033 
L 672.570529 46.962732 
L 674.831607 49.071517 
L 677.092686 51.510669 
L 679.353764 54.330627 
L 679.353764 54.330627 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
   </g>
   <g id="line2d_57">
    <path d="M 82.429048 191.2648 
L 84.797797 192.868243 
L 87.166546 194.15988 
L 89.535295 196.141885 
L 91.904044 198.631326 
L 94.272793 201.450755 
L 101.379039 210.635465 
L 106.116537 216.689462 
L 110.854035 222.437312 
L 115.591533 227.830322 
L 120.32903 232.869272 
L 125.066528 237.570743 
L 129.804026 241.954996 
L 134.541523 246.042566 
L 139.279021 249.854044 
L 144.016519 253.410741 
L 148.754017 256.735287 
L 153.491514 259.851848 
L 160.597761 264.192758 
L 167.704008 268.212176 
L 177.179003 273.222629 
L 191.622594 280.484015 
L 208.608257 289.054621 
L 220.740873 295.455192 
L 235.300012 303.437215 
L 245.006105 308.713946 
L 252.285675 312.414341 
L 257.138721 314.641191 
L 261.991768 316.591496 
L 266.844814 318.196608 
L 271.697861 319.404126 
L 276.550907 320.189148 
L 281.403954 320.560526 
L 286.257 320.559365 
L 291.110047 320.250474 
L 295.963093 319.710423 
L 303.242663 318.633973 
L 320.228325 315.933824 
L 327.507895 315.01901 
L 334.787465 314.359329 
L 342.067034 313.994414 
L 349.346604 313.949336 
L 354.199651 314.105363 
L 359.052697 314.414525 
L 363.905744 314.87959 
L 368.75879 315.502578 
L 373.611837 316.284621 
L 378.464883 317.225598 
L 383.317929 318.323504 
L 390.597499 320.252837 
L 397.877069 322.486519 
L 407.583162 325.797419 
L 417.289255 329.133504 
L 422.142301 330.612507 
L 426.995348 331.822271 
L 431.848394 332.647616 
L 434.274917 332.884337 
L 436.701441 332.991258 
L 439.127964 332.963069 
L 441.554487 332.798286 
L 443.98101 332.499394 
L 446.407534 332.072665 
L 451.26058 330.876642 
L 456.113627 329.31341 
L 460.966673 327.502595 
L 482.805382 318.951121 
L 487.658429 317.347567 
L 492.511475 315.931745 
L 497.364522 314.723909 
L 502.217568 313.740554 
L 507.070614 312.996435 
L 511.923661 312.506146 
L 516.776707 312.285402 
L 521.629754 312.352148 
L 526.4828 312.727607 
L 531.335847 313.437312 
L 536.188893 314.512149 
L 541.04194 315.989331 
L 545.894986 317.912963 
L 548.321509 319.057744 
L 550.748033 320.33332 
L 553.174556 321.746243 
L 555.601079 323.302478 
L 558.027602 325.006585 
L 560.454126 326.860399 
L 565.307172 330.996937 
L 575.013265 340.045424 
L 577.439788 341.945022 
L 579.866312 343.364595 
L 582.23506 344.095321 
L 584.603809 344.040848 
L 586.972558 343.165481 
L 589.341307 341.55029 
L 591.710056 339.361835 
L 594.078805 336.797091 
L 603.5538 325.833519 
L 605.922549 323.366511 
L 608.291298 321.097865 
L 610.660047 319.047796 
L 613.028796 317.231305 
L 615.397545 315.661345 
L 617.766294 314.351107 
L 620.135042 313.315801 
L 622.503791 312.574228 
L 624.87254 312.150379 
L 627.241289 312.075296 
L 629.610038 312.389486 
L 631.871116 313.101337 
L 634.132195 314.277602 
L 636.393273 315.999998 
L 638.654352 318.379658 
L 640.91543 321.57149 
L 643.176509 325.797897 
L 645.437587 331.388286 
L 647.698666 338.838996 
L 649.959744 348.840399 
L 652.220823 361.624702 
L 654.481901 371.342976 
L 656.74298 364.120932 
L 659.004058 350.130138 
L 661.265136 338.820862 
L 663.526215 330.825575 
L 665.787293 325.669662 
L 668.048372 322.896924 
L 670.30945 321.429116 
L 672.570529 317.644629 
L 674.831607 306.223695 
L 677.092686 288.782845 
L 679.353764 270.434251 
L 679.353764 270.434251 
" clip-path="url(#pf67e761123)" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
   </g>
   <g id="patch_3">
    <path d="M 52.582813 390.84375 
L 52.582813 10.8 
" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_4">
    <path d="M 709.2 390.84375 
L 709.2 10.8 
" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_5">
    <path d="M 52.582812 390.84375 
L 709.2 390.84375 
" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="patch_6">
    <path d="M 52.582812 10.8 
L 709.2 10.8 
" style="fill: none; stroke: #ffffff; stroke-width: 0.8; stroke-linejoin: miter; stroke-linecap: square"/>
   </g>
   <g id="legend_1">
    <g id="patch_7">
     <path d="M 638.182813 385.84375 
L 702.2 385.84375 
Q 704.2 385.84375 704.2 383.84375 
L 704.2 340.253125 
Q 704.2 338.253125 702.2 338.253125 
L 638.182813 338.253125 
Q 636.182813 338.253125 636.182813 340.253125 
L 636.182813 383.84375 
Q 636.182813 385.84375 638.182813 385.84375 
z
" style="opacity: 0.8; stroke: #cccccc; stroke-linejoin: miter"/>
    </g>
    <g id="line2d_58">
     <path d="M 640.182813 346.351562 
L 650.182813 346.351562 
L 660.182813 346.351562 
" style="fill: none; stroke: #8dd3c7; stroke-width: 1.5; stroke-linecap: square"/>
     <g>
      <use xlink:href="#mf021a1f644" x="650.182813" y="346.351562" style="fill: #8dd3c7; stroke: #8dd3c7; stroke-linejoin: miter"/>
     </g>
    </g>
    <g id="text_19">
     <!-- H -->
     <g style="fill: #ffffff" transform="translate(668.182813 349.851562)scale(0.1 -0.1)">
      <use xlink:href="#DejaVuSans-48"/>
     </g>
    </g>
    <g id="line2d_59">
     <path d="M 640.182813 361.029688 
L 650.182813 361.029688 
L 660.182813 361.029688 
" style="fill: none; stroke-dasharray: 7.4,3.2; stroke-dashoffset: 0; stroke: #feffb3; stroke-width: 2"/>
    </g>
    <g id="text_20">
     <!-- H_fit -->
     <g style="fill: #ffffff" transform="translate(668.182813 364.529688)scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-5f" d="M 3263 -1063 
L 3263 -1509 
L -63 -1509 
L -63 -1063 
L 3263 -1063 
z
" transform="scale(0.015625)"/>
       <path id="DejaVuSans-74" d="M 1172 4494 
L 1172 3500 
L 2356 3500 
L 2356 3053 
L 1172 3053 
L 1172 1153 
Q 1172 725 1289 603 
Q 1406 481 1766 481 
L 2356 481 
L 2356 0 
L 1766 0 
Q 1100 0 847 248 
Q 594 497 594 1153 
L 594 3053 
L 172 3053 
L 172 3500 
L 594 3500 
L 594 4494 
L 1172 4494 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-48"/>
      <use xlink:href="#DejaVuSans-5f" x="75.195312"/>
      <use xlink:href="#DejaVuSans-66" x="125.195312"/>
      <use xlink:href="#DejaVuSans-69" x="160.400391"/>
      <use xlink:href="#DejaVuSans-74" x="188.183594"/>
     </g>
    </g>
    <g id="line2d_60">
     <path d="M 640.182813 375.985937 
L 650.182813 375.985937 
L 660.182813 375.985937 
" style="fill: none; stroke-dasharray: 2,3.3; stroke-dashoffset: 0; stroke: #bfbbd9; stroke-width: 2"/>
    </g>
    <g id="text_21">
     <!-- err_rel -->
     <g style="fill: #ffffff" transform="translate(668.182813 379.485937)scale(0.1 -0.1)">
      <defs>
       <path id="DejaVuSans-6c" d="M 603 4863 
L 1178 4863 
L 1178 0 
L 603 0 
L 603 4863 
z
" transform="scale(0.015625)"/>
      </defs>
      <use xlink:href="#DejaVuSans-65"/>
      <use xlink:href="#DejaVuSans-72" x="61.523438"/>
      <use xlink:href="#DejaVuSans-72" x="100.886719"/>
      <use xlink:href="#DejaVuSans-5f" x="142"/>
      <use xlink:href="#DejaVuSans-72" x="192"/>
      <use xlink:href="#DejaVuSans-65" x="230.863281"/>
      <use xlink:href="#DejaVuSans-6c" x="292.386719"/>
     </g>
    </g>
   </g>
  </g>
 </g>
 <defs>
  <clipPath id="pf67e761123">
   <rect x="52.582813" y="10.8" width="656.617188" height="380.04375"/>
  </clipPath>
 </defs>
</svg>



## References

[1] Gustavsen, B. and Adam Semlyen. “Rational approximation of frequency domain responses by vector fitting.” IEEE Transactions on Power Delivery 14 (1999): 1052-1061.

[2] B. Gustavsen, "Improving the pole relocating properties of vector fitting," in IEEE Transactions on Power Delivery, vol. 21, no. 3, pp. 1587-1592, July 2006, doi: 10.1109/TPWRD.2005.860281.

[3] D. Deschrijver, M. Mrozowski, T. Dhaene and D. De Zutter, "Macromodeling of Multiport Systems Using a Fast Implementation of the Vector Fitting Method," in IEEE Microwave and Wireless Components Letters, vol. 18, no. 6, pp. 383-385, June 2008, doi: 10.1109/LMWC.2008.922585.



