# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:27:50 2021

@author: utric
"""

import numpy as np
from numpy import pi as π
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as μ0

import pycoilib


plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rc('lines', linewidth=2)
plt.rc('font', size=9)



if True:
    print("---------------------------------------------------------------------")
    print("----1\t- CODE VALIDATION : 2 lines")
    print("----1.1\t- A wire divided in two")#   1.1         
    p0 = np.array([0. , 0., 0.])
    p1 = np.array([0.5, 0., 0.])
    p2 = np.array([1., 0., 0.])
    line1 = pycoilib.segment.Line(p0,p1)
    line2 = pycoilib.segment.Line(p1,p2)
    
    M, tmp = pycoilib.calc_M(line1, line2)
    Mth =  1/2 * μ0/(2*π)*(line1.ell+line2.ell)*np.log(2)
    print(f"---- \tM: {M:.9e}\t Mth: {Mth:.9e}")


if False:
    print("---------------------------------------------------------------------")
    print("----1.2\t-  A pair of parallel lines")#   1.1
    ell = np.arange(0.5, 10.1, 0.5)*1e-2
    ell = np.logspace(np.log10(0.005),0,15)
    
    d0 = 5.*1e-2
    MM=[]
    for ell_i in ell:
        p0 = np.array([0.   , 0., 0.])
        p1 = np.array([ell_i, 0., 0.])
        p2 = np.array([0.   , 0., d0])
        p3 = np.array([ell_i, 0., d0])
        line1 = pycoilib.segment.Line(p0,p1)
        line2 = pycoilib.segment.Line(p2,p3)
        M, tmp = pycoilib.inductance.calc_M(line1, line2)
        MM.append(M)
    M1 = np.array(MM)
    Mth = 2*(ell*np.log( ((ell+np.sqrt(ell**2+d0**2))/d0) ) 
             -np.sqrt(ell**2+d0**2)+d0) *1e-7
    
    fig = plt.figure(figsize=(7.5/2.54, 5.5/2.54))
    ax = plt.gca()
    ax.loglog(ell*1e3,M1*1e9,"o", markersize=3)
    ax.loglog(ell*1e3,Mth*1e9,alpha=0.8 )
    ax.legend(["Our approach", "Analytical"])
    ax.set_xlabel(r"Line length [mm]")
    ax.set_ylabel(r"Mutual inductance [nH]")
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.99, r"$d = 50$ mm", transform=ax.transAxes, fontsize=8,
            verticalalignment='top', )
    fig.tight_layout()
    #fig.savefig("paralel_Lines.png", dpi=300)
    plt.show()

# print("---------------------------------------------------------------------")
# wire = pycoilib.wire.Wire_circ(1.e-3)
# coil = pycoilib.coil.Coil([line1,line2.flip_current_direction()], wire)
# coil.draw()
# print(coil.calc_I())
# print(pycoilib.inductance.calc_M(line1, line2))
if False:
    print("---------------------------------------------------------------------")
    print("----1.2\t-  Antiparalle pair of lines")#   1.1
    
    
    p0 = np.array([ 0., 0. , 0.])
    p1 = np.array([10., 0. , 0.])
    p2 = np.array([1., -1, 0.])
    p3 = np.array([ -11., 0, 0.])
    
    line1 = pycoilib.segment.Line(p0,p1)
    line2 = pycoilib.segment.Line(p2,p3)
    
    Lp, Ls = line1.ell, line2.ell 
    
    
    p_u, s_u = line1.vec_n, line2.vec_n
    
    s0 = np.sqrt( (line2.vec_r0-line1.vec_r0) @ (line2.vec_r0-line1.vec_r0) )
    s0_u = (line2.vec_r0-line1.vec_r0)/s0
    
    print("s0_u:\t", s0_u)
    print("p_u:\t", p_u)
    print("s_u:\t", s_u)
    
    print("s0@p:\t", s0_u@p_u )
    print("s0@s:\t", s0_u@s_u )
    print("s@p:\t", s_u@p_u )
    
    p = np.linspace(0, Ls, 10000)
    
    zero = s0*(p_u@s0_u - (s0_u@s_u)*(p_u@s_u))/(1-(p_u@s_u)**2)
    
    β = (s0**2 * (1-(s0_u@s_u)**2) 
         + p**2*(1-(p_u@s_u)**2) 
         + 2*s0*p*((p_u@s0_u) - (p_u@s_u)*(s0_u@s_u)) )
    
    
    σ1 = Ls + (s0*(s0_u@s_u) - p* (p_u@s_u))
    σ0 = 0  + (s0*(s0_u@s_u) - p* (p_u@s_u))
    
    plt.plot(p, β)
    plt.show()
    
    print(p[np.argmin(β)], np.min(β))
    print(zero)
    
    coil = pycoilib.coil.Coil([line1, line2]).draw()


if True:
    print("---------------------------------------------------------------------")
    print("----1.2\t-  Parralel of lines")#   1.1
    ρ = 2e-3
    p0 = np.array([0., 0. , 0.])
    p1 = np.array([1., 0. , 0.])
    p2 = np.array([1., 0., 0.])
    p3 = np.array([2., 0., 0.])
    p4 = np.array([1., ρ, 0.])
    p5 = np.array([2., ρ, 0.])
    
    line1 = pycoilib.segment.Line(p0,p1)
    line2 = pycoilib.segment.Line(p2,p3)
    line3 = pycoilib.segment.Line(p4,p5)
    
    i1, tmp = pycoilib.calc_M(line1, line2)
    print("ok")
    
    i2, tmp = pycoilib.calc_M(line1, line3)
    print("ok")
    
    print(i1)
    print(i2)
    

