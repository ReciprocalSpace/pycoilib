# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:16:50 2021

@author: Aime Labbe
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, arctan2 as atan, sqrt, pi as π, sign, log
from scipy.integrate import quad
from scipy.special import ellipk as ellK,  ellipe as ellE
from scipy.special import ellipkinc as ellK_inc,  ellipeinc as ellE_inc
from scipy.constants import mu_0 as μ0

from pycoilib.lib.misc.geometry import _vec_0

def Ψ_p(k):
    return (1-k**2/2)*ellK(k**2) - ellE(k**2)


def Ψ(φ1,φ0,k):
    res = ( (1-k**2/2)*(ellK_inc(φ1,k**2)-ellK_inc(φ0,k**2))/2
           -(ellE_inc(φ1,k**2)-ellE_inc(φ0,k**2))/2 )
    return res

def Θ(φ1,φ0,k):
    return ( sqrt(1-k**2*sin(φ1)**2)-sqrt(1-k**2*sin(φ0)**2) )/2


def calc_M_2arcs(arc1, arc2):
    Rp = arc1.radius
    Rs = arc2.radius

    s0_u = (arc2.vec_r0-arc1.vec_r0)
    s0 = sqrt(s0_u@s0_u)
    if s0==0:
        s0_u = _vec_0.copy()
    else:
        s0_u = s0_u/s0

    x_u = arc1.vec_x
    y_u = arc1.vec_y

    u_u = arc2.vec_x
    v_u = arc2.vec_y
    
    sφ = arc2.theta
    
    def integrand(θ):
        A = Rp**2 + Rs**2 + s0**2 - 2*Rp*s0*( s0_u@x_u*cos(θ) + s0_u@y_u*sin(θ) )
        B = -2*Rs*( Rp*(u_u@x_u*cos(θ) + u_u@y_u*sin(θ)) - s0*s0_u@u_u )
        C = -2*Rs*( Rp*(v_u@x_u*cos(θ) + v_u@y_u*sin(θ)) - s0*s0_u@v_u )

        d = sqrt(B**2 + C**2)
        k = np.clip(sqrt(2*d/(A+d)),0.,1.)

        κ = atan(C, B)
        φ1, φ0 = (sφ-κ)/2, -κ/2

        pre = 2*μ0*Rp*Rs/π
        fct1 = 1/(k**2*d*sqrt(A+d))
        fct2 = (-u_u@x_u*B-v_u@x_u*C)*sin(θ) + (u_u@y_u*B+v_u@y_u*C)*cos(θ)
        fct3 = (-u_u@x_u*C+v_u@x_u*B)*sin(θ) + (u_u@y_u*C-v_u@y_u*B)*cos(θ)
        
        return pre*fct1*(fct2*Θ(φ1,φ0,k) + fct3*Ψ(φ1,φ0,k))
    
    mutual, err = quad(integrand, 1e-8, arc1.theta-1e-8,epsabs=1e-15,limit=200 )

    return mutual, err