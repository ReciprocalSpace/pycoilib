# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:36:07 2021

@author: utric
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, arctan2 as atan, sqrt, pi as π, sign, log
from scipy.integrate import quad, dblquad
from scipy.special import ellipk as ellK,  ellipe as ellE
from scipy.special import ellipkinc as ellK_inc,  ellipeinc as ellE_inc
from scipy.constants import mu_0 as μ0

_vec_0 = np.array([0.,0.,0.])
_vec_x = np.array([1.,0.,0.])
_vec_y = np.array([0.,1.,0.])
_vec_z = np.array([0.,0.,1.])

import pycoilib as pycoil

z0, r0 = 0., 1000e-3
φ1 =  π/10 

a0 = 0.5e-3
z, r = 0, r0-a0

def get_A_φ(φ):
    m = (4*r0*r / ( (r0+r)**2 +(z-z0)**2 ) )
    
    def f2(φ0):
        ψ = (φ0-φ-π)/2
        
        t1 = (r0**2 + r**2 +(z-z0)**2) /sqrt( (r0+r)**2 +(z-z0)**2)
        
        t2 = sqrt( (r0+r)**2+(z-z0)**2 )
        
        return  ( t1*ellK_inc(ψ, m) - t2*ellE_inc(ψ, m) )
    
    A_φ = μ0/(4*π*r) * ( f2(φ1)-f2(0) )
    return A_φ


arc = pycoil.shape.Arc(_vec_0, r0, φ1, _vec_x, _vec_y, _vec_z)
wire = pycoil.wire.WireCircular(a0)
coil = pycoil.coil.Coil([arc], wire)

res, err = quad(get_A_φ, 0, φ1 )

print(coil.get_inductance(), r * res)
print(abs(coil.get_inductance() / (r * res) - 1) * 100, "%")




