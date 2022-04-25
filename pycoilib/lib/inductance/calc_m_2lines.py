# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:32:43 2021

@author: Aime Labbé
"""

import numpy as np
from numpy import sqrt, pi as π, log
from scipy.integrate import quad
from scipy.constants import mu_0 as μ0

from ..segment.segment import Line


def calc_M_2lines(line1: Line, line2: Line):
    """Compute the mutual inductance between a pair of lines,"""
    # Todo : implement as analytical solutions using Grover's approach
    Lp, Ls = line1.ell, line2.ell  # Line lengths
    p_u, s_u = line1.vec_n, line2.vec_n  # Line orientations
    s0 = sqrt( (line2.vec_r0-line1.vec_r0) @ (line2.vec_r0-line1.vec_r0))  # Distance between segments

    if np.isclose(s_u@p_u,  0.):  # Lines are orthogonal -> Mutual is 0.
        return 0., 0.
    
    if not np.isclose(s0,  0.):  # Lines don't have the same p0 origin
        s0_u = (line2.vec_r0-line1.vec_r0)/s0  # Unit vector
    else:
        s0_u = p_u.copy()  # Lines have the same origin -> s0_u won't matter, but we don't want a devide by zero
    
    cst = μ0*(p_u@s_u)/(4*π)
    
    def σ(s: float, p: float):
        return s + (s0*(s0_u@s_u) - p*(p_u@s_u))
    
    # Two cases : 
    # 1) s_u, s0_u and p_u are collinear
    if all(np.isclose([abs(p_u@s_u), abs(s0_u@p_u), abs(s0_u@s_u)], np.ones(3))):
        # Endpoints in p are removed from the integral to avoid σ = 0
        σ11, σ10 = σ(Ls, Lp*(1-1e-7)), σ(Ls, 1e-7)
        σ01, σ00 = σ(0, Lp*(1-1e-7)), σ(0, 1e-7)
        
        return -(s0_u@s_u)*cst*( 
              σ11*log(np.abs(σ11)) - σ10*log(np.abs(σ10)) 
            - σ01*log(np.abs(σ01)) + σ00*log(np.abs(σ00))), 0
    
    # 2) s_u, s0_u and p_u are not collinear
    def integrand(p):
        β2 = s0**2 * (1-(s0_u@s_u)**2) + p**2*(1-(p_u@s_u)**2) - 2*s0*p*((p_u@s0_u) - (p_u@s_u)*(s0_u@s_u))
        
        β = np.sqrt(np.clip(β2, 0, np.inf))  # Clip to avoid numerical errors
        
        σ1 = σ(Ls, p)
        σ0 = σ( 0, p)
        
        return cst*(np.arcsinh(σ1/β)-np.arcsinh(σ0/β))
    
    points = [0, Lp]
    # Check if coplanar
    # If all three vectors lie in the same plane, β(p) can have a zero. 
    if not np.isclose(abs(p_u@s_u), 1) and np.isclose(np.cross(p_u,s_u) @ s0_u, 0):
        root = s0*( p_u@s0_u - (s0_u@s_u)*(p_u@s_u) ) / (1-(p_u@s_u)**2)
        if np.isclose(root, 0):
            root = 0.
        points.append(root)
        
    mutual, err = quad(integrand, 0, Lp, epsrel=1e-8, limit=100, points=points)
    
    return mutual, err

