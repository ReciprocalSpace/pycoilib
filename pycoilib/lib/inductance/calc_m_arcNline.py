# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:33:02 2021

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

def calc_M_arcNline(loop, line):
    Rp = loop.radius
    x_u = loop.vec_x
    y_u = loop.vec_y
    θ1 = loop.theta
    
    s_u = line.vec_n
    Ls=line.ell
    
    
    s0 = sqrt( (line.vec_r0-loop.vec_r0)@(line.vec_r0-loop.vec_r0))
    if s0 != 0.:
        s0_u= (line.vec_r0-loop.vec_r0)/s0
    else:
        s0_u = _vec_0.copy()


    def integrand(θ):
        β =( Rp**2*( 1 - ( (x_u@s_u)*cos(θ) + (y_u@s_u)*sin(θ))**2)
            +s0**2*( 1 - (s0_u@s_u)**2 )
            -2*Rp*s0*( ((s0_u@x_u)-(s0_u@s_u)*(x_u@s_u))*cos(θ)
                      +((s0_u@y_u)-(s0_u@s_u)*(y_u@s_u))*sin(θ) ))
        
        β = sqrt(np.clip(β, 1e-54, np.inf))
        
        σ1 = Ls + s0*s_u@s0_u - Rp*((s_u@x_u)*cos(θ)+(s_u@y_u)*sin(θ)) 
        σ0 = 0. + s0*s_u@s0_u - Rp*((s_u@x_u)*cos(θ)+(s_u@y_u)*sin(θ)) 

        cst = μ0/(4*π)*Rp*( -(s_u@x_u)*sin(θ) +(s_u@y_u)*cos(θ))
        if isinstance(β, np.ndarray):
            fct = np.zeros_like(β)
            ind = np.logical_and( np.absolute(β)*1e18>σ0**2, 
                                  np.absolute(β)*1e18>σ1**2 )
            not_ind = np.logical_not(ind)
            fct[ind] =cst[ind]*(np.arctanh(σ1[ind]/sqrt(σ1[ind]**2+β[ind]))
                               -np.arctanh(σ0[ind]/sqrt(σ0[ind]**2+β[ind])))
            fct[not_ind] = cst[not_ind]*( 
                 sign(σ1[not_ind])*log(np.absolute(σ1[not_ind])) 
                -sign(σ0[not_ind])*log(np.absolute(σ0[not_ind])) )
        else:
            
            if abs(β)*1e18>σ0**2 and abs(β)*1e18>σ1**2:
                fct = cst * ( np.arctanh(σ1/sqrt(σ1**2+β))
                             -np.arctanh(σ0/sqrt(σ0**2+β)))
            else:
                fct = cst*(  sign(σ1)*log(np.absolute(σ1)) 
                           - sign(σ0)*log(np.absolute(σ0)) )

        return fct
    
    mutual, err = quad(integrand, 1e-5, θ1-1e-5, epsrel=1e-5,limit=200)
    if np.isnan(mutual):
        theta = np.linspace( 1e-8, θ1-1e-8, 201)
        res = integrand( theta )
        plt.plot(theta, res)
        plt.show()
        print(res)
    
    return mutual, err