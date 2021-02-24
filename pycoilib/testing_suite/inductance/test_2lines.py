# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:59:58 2021

@author: Aime Labbe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as μ0, pi as π
import random

import pycoilib as pycoil

def rand(n=1):
    return np.array([random.random() for i in range(n)])

def random_vec_on_sphere(n=1):
    φ = rand(n) *2*π
    θ = np.arccos( rand(n)*2-1 )
    
    n_vec = np.array([ [np.cos(θ_i)*np.sin(φ_i),
                        np.sin(θ_i)*np.sin(φ_i),
                        np.cos(φ_i)] for θ_i, φ_i in zip(θ, φ) ] )
    return n_vec

if False:
    print("-------Autoinductance d'un fil----------")
    I1 = []
    I2 = []
    _ell = np.logspace(-3, 2, 101)
    
    for ell in _ell:
        a =   0.001
        p0, p1 = np.array([0.,0.,0.]), np.array([ell,0.,0.])
        line1 = pycoil.segment.Line(p0, p1)
        wire = pycoil.wire.Wire_circ(a)
        
        coil = pycoil.coil.Coil([line1], wire )
        
        I1.append( coil.calc_I() )
        I2.append( μ0*ell/(2*π)*(np.log(2*ell/a)-1) )
    I1 = np.array(I1)
    I2 = np.array(I2)
    
    plt.loglog(_ell/a, I1*1e6)
    plt.loglog(_ell/a, I2*1e6)
    plt.legend(["Our", "Analytical"])
    plt.xlabel('Wire length/radius ratio')
    plt.ylabel("Self inductance [μH]")
    plt.title("Self inductance of a wire")
    plt.show()


if False:
    print("-------Mutuelle cas colinéaires-------")
    I1 = []
    I2 = []
    a = 1e-3
    ell_1 = 1
    ell_2 = 2
    _d = np.logspace(-3,1,51)
    for d in _d:
        p0, p1 = np.array([0.,0.,0.]), np.array([ell_1,0.,0.])
        p2, p3 = np.array([ell_1+d,0.,0.]), np.array([ell_1+d+ell_2, 0.,0.])
        line1 = pycoil.segment.Line(p0,p1)
        line2 = pycoil.segment.Line(p2, p3)
        
        p4, p5 = np.array([ell_1+d, a,0.]), np.array([ell_1+d+ell_2, a,0.])
        line3 = pycoil.segment.Line(p4, p5)
        
        i1, tmp = pycoil.calc_M(line1, line2)
        i2, tmp = pycoil.calc_M(line1, line3)
        I1.append(i1)
        I2.append(i2)
        
    I1 = np.array(I1)
    I2 = np.array(I2)
    
    plt.semilogx(_d*1e3, I1*1e9, "o", fillstyle='none')
    plt.semilogx(_d*1e3, I2*1e9, "+")
    plt.legend(['Case cosllinear', "Other"])
    plt.xlabel('Distance between wire [mm]')
    plt.ylabel("Self inductance [nH]")
    plt.title("Mutual between two collinear wires")
    plt.show()
    
    
if False: 
    print("-------Mutuelle cas coplanaire-------")
    I1 = []
    I2 = [] 
    a = 0.5
    ell_1 = 1 
    ell_2 = 2 
    z_axis = np.array([0.,0.,1.])
    vec_0 =  np.array([0.,0.,0.])
    _angle = np.linspace(0, 3*π/2, 301)[:-1]
    for angle in _angle:
        p0, p1 = np.array([0.,0., 0.]), np.array([ell_1, 0., 0.])
        p2, p3 = np.array([0,  a, 0.]), np.array([ell_2,  a, 0.])
        line1 = pycoil.segment.Line(p0,p1)
        
        line2 = pycoil.segment.Line(p2, p3)
        line2.rotate(z_axis, angle,)
        
        i1, tmp = pycoil.calc_M(line1, line2)
        
        I1.append(i1)

    I1 = np.array(I1)

    plt.plot(_angle/π, I1*1e6, "-", fillstyle='none')
 
    plt.xlabel('Angle [π]')
    plt.ylabel("Self inductance [μH]")
    plt.title("Mutual between two coplanar wires")
    plt.show()
    
if True: 
    print("-------Mutual between noncoplanar wires-----")
    I1 = []
    a = 0.5
    ell_1 = 1
    ell_2 = 2
    z_axis = np.array([0.,0.,1.])
    vec_0 =  np.array([0.,0.,0.])
   
    θ = np.linspace(0, 16*π, 400  )
    φ = np.linspace(0, π, 81)
    
    n_vec = np.array([ [np.cos(θ_i)*np.sin(φ_i),
                        np.sin(θ_i)*np.sin(φ_i),
                        np.cos(φ_i)] for θ_i, φ_i in zip(θ, φ) ] )
    
    p0, p1 = np.array([0.,0., 0.]), np.array([0., 0., ell_1])
    line1 = pycoil.segment.Line(p0,p1)
    for pos in n_vec:
        for orientation in n_vec: 
        
            p2, p3 = a*pos, a*pos + ell_2*orientation
            
            line2 = pycoil.segment.Line(p2, p3)
            
            i1, tmp = pycoil.calc_M(line1, line2)
            
            if np.isnan(i1):
                pycoil.coil.Coil([line1,line2]).draw()
                break
            
            I1.append(i1)
        if np.isnan(i1):
            break
    I1 = np.array(I1)
    
    plt.plot(I1*1e6, "-", )
    plt.legend([f"{θ.max()/π:.2f} π"])
    plt.xlabel('Angle [π]')
    plt.ylabel("Self inductance [μH]")
    plt.title("Mutual between two coplanar wires")
    plt.show()

    
    
