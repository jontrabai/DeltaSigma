# -*- coding: utf-8 -*-
"""
    Created on Mon Mar 30 16:03:07 2015

    @author: jbaird
    """
from math import ceil, log, pi

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from tqdm import tqdm

# %% cell
SNDR = 60       # Signal to Noise + Distortion Ratio
dSNDR = 0.5     # Desired SNDR Accuracy
OSR = 32        # Oversampling ratio
fb = 0.390618e6  # Input signal bandwidith
Ain = 0.6       # Input signal amplitude (Volts)
alpha1 = 0.5    # 1st integrator gain
alpha2 = 0.5    # 2nd integrattor gain
levels = 3      # number of levels in the modulator
vrefp = 1       # D/A maximum output level (Volts)
vrefn = -1      # D/A minimum output level (Volts)

# %% calculations
print("Calculation results basied on inputs\n")
fs = fb*2*OSR   # Sampling frequency
Order = 2       # Order of the modulator
fin = fb/2      # Input signal frequency
bits = ceil(log(levels)/log(2))  # Number of sample points
print("Sampling frequency:     ", fs/1e6, "MHz")
print("Input signal frequency: ", fin/1e3, "KHz")
print("Number of output bits:  ", bits)

# Number of sample points
# ========================
num = 0
for i in tqdm(range(5, 21)):
    if 2**i < SNDR/dSNDR:
        num = 2**i
    else:
        num = num
num = OSR * num
# ========================

print("Number of sample points:", num)
ind = np.arange(0, num)  # Index vector
t1 = ind/fs            # Time vector
vin1 = Ain*np.sin(2*np.pi*fin*t1)  # input signal vector
dv = (vrefp-vrefn)/(levels-1)  # D/A step size
print("D/A step size:          ", dv, "V")
th = np.arange(0, levels)  # threshold index vector
vadcth = vrefn + th*dv+dv/2   # A/D converter thresholds
lev = np.arange(levels)  # level index vector
vdaclev = vrefn + dv*lev   # D/A converter output levels

# A/D - D/A converter routine
# ==========================


def vdac(vi):
    vo = vdaclev[0]
    for i in tqdm(range(levels-1)):
        if vi < vadcth[i]:
            vo = vo
        else:
            vo = vdaclev[i+1]
    return vo
# ==========================

numpts = 100 # number of points for plotting A/D - D/A
# j = range(0,numpts) # index vector for A/D - D/A

# input waveform for A/D - D/A plot
# =================================
vsw = [vrefn + j*((vrefp-vrefn)/(numpts)) for j in tqdm(range(numpts))]
dac = [vdac(vsw[j]) for j in tqdm(range(numpts))]
# =================================

# plot the A/D input and D/A output
# =================================
plt.figure(1)
plt.plot(vsw, dac)
plt.plot(vsw, vsw)
plt.ylim(-1.1, 1.1)
plt.show(block=False)
# =================================

# plot the time domain input and output waveforms

# initialize the waveforms
vint1 = []
vint2 = []
vo = []
vint1.append(0.0)
vint2.append(0.0)
vo.append(vdac(vint2[0]))

# iterate over 'num' samples
for i in tqdm(range(1, num)):
    vint1.append(vint1[i-1]+alpha1*(vin1[i-1]-vo[i-1]))
    vint2.append(vint2[i-1]+alpha2*(vint1[i-1]-vo[i-1]))
    vo.append(vdac(vint2[i]))

# plot the resulting waveforms
plt.figure(2)
plt.plot(t1, vin1)
plt.step(t1, vo)
plt.plot(t1, vint2)
# plt.xlim(0,4e-6)
plt.ylim(-1.1, 1.1)
plt.show(block=False)
# ===============================================

print("The mean output voltage:", np.mean(vo))
Vo = np.fft.fft(vo)  # frequency spectrum
# scale the fft vector by 1 over the square root of its length
# then only use the second half of the fft vector
# the result matches the Mathcad fft algorithm
Vo = (1.0/np.sqrt(len(Vo)))*Vo[:int(num/2)]
f = [(j*(0.5*fs)/(int(num/2))) for j in range(int(num/2))]
z = [np.exp(1j*2*pi*f[j]/fs) for j in range(int(num/2))]
vo_ideal = [(dv/np.sqrt(12))*((1-z[i]**-1)**2) / (((1-z[i]**-1)**2) +
                                                  (alpha2*z[i]**-1*(1-z[i]**-1
                                                   + alpha1*z[i]**-1)))
            for i in range(int(num/2))]
S = [alpha1*alpha2*((z[i])**-2) / (((1-z[i]**-1)**2) +
                                   (alpha2*z[i]**-1*(1-z[i]**-1
                                    + alpha1*z[i]**-1)))
     for i in range(int(num/2))]
plt.figure(3)
# plt.ylim(1e-6, 100)
plt.loglog(f, np.abs(S))
plt.plot(f, np.abs(Vo))
plt.plot(f, np.abs(vo_ideal))
plt.show(block=False)

# Filter
Order_sinc = Order + 1  # Number of stages of the sinc filter


def D(f):
    if f == 0:
        return 0
    else:
        return ((1.0/OSR)*(np.sin(pi*(f/fs)*OSR))/(np.sin(pi*(f/fs)) *
                                                   np.exp(-1j*pi*(f/fs) *
                                                   (OSR-1))))**Order_sinc
f = [j/((num/2)-1)*fs/2 for j in range(int(num/2))]
Hsinc = [D(f[j]) for j in range(int(num/2))]

# Rectangle window


def R(n, N):
    return int(n>=0*n<=(N-1))


def K(n, N, beta):
    return (sp.i0(beta*np.sqrt(1-((N-1-2*n)/(N-1))**2)/(sp.i0(beta))))*R(n, N)


def r(m, M, omegac):
    odd = np.mod(M, 2)
    if m == odd*((M-1)/2):
        return omegac/pi
    else:
        return np.sin(omegac*(m-((M-1)/2)))/(pi*(m-(M-1)/2))


def KFIR(fs, fstop, fpass, A):
    omegac = pi/fs*(fpass+fstop)
    if A < 21:
        beta = 0
    elif A <= 50:
        beta = 0.5842*(A-21)**4+0.07886*(A-21)
    else:
        beta = 0.1102*(A-8.7)
    M = ceil(1+(A-8)/(2.285*(2*pi)/fs*(fstop-fpass)))
    return M, beta, omegac

ffir = 2*fs/OSR
fpass = 0.8*ffir/4
fstop = 1.2*ffir/4
[M, beta, omegac] = KFIR(ffir, fstop, fpass, SNDR)
print("M:", M)
print("beta:", beta)
print("omegac:", omegac)
m = range(0, int(M))
a = [r(i, M, omegac)*K(i, M, beta) for i in m]

Hfir = [np.array(a).sum()*np.exp(-1j*2*pi*f[j]/ffir)
        for j in range(int(num/2))]
