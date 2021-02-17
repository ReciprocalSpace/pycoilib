# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:33:29 2021

@author: Aime Labbe
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, sqrt, pi as π, sign
from scipy.integrate import quad
from scipy.special import ellipk as ellK,  ellipe as ellE
from scipy.special import ellipkinc as ellK_inc,  ellipeinc as ellE_inc
from scipy.constants import mu_0 as μ0




def calc_M_Babic(Rp,Rs,C=(0.,0.,0.),n=(0.,0.,1.)):
    xc,yc,zc = tuple(C)
    a,b,c = tuple(n)
    
    α, β, γ, δ  = Rs/Rp, xc/Rp, yc/Rp, zc/Rp

    l = sqrt(a**2+c**2)
    L = sqrt(a**2+b**2+c**2)

    if l==0.: #a==0. and c==0.:
        p1 = 0.
        p2 = -γ*sign(b)
        p3 = 0.
        p4 = -β*sign(b)
        p5 = δ
    else:
        p1 = γ*c/l
        p2 = -(β*l**2+γ*a*b)/(l*L)
        p3 = α*c/L
        p4 = -(β*a*b - γ*l**2 + δ*b*c)/(l*L)
        p5 = -(β*c-δ*a)/l

    def integrand(this, φ,l,L,p1,p2,p3,p4,p5):
        A0 = 1+α**2+β**2+γ**2+δ**2 + 2*α*(p4*cos(φ)+p5*sin(φ))
        if l==0.:
            print(True,"l==0")
            V0 =  sqrt( β**2+γ**2+α**2*cos(φ)**2-2*α*β*sign(b)*cos(φ) )
        else:
            V0 =  sqrt( α**2*( (1-b**2*c**2/(l**2*L**2))*cos(φ)**2 
                                 + c**2/l**2*sin(φ)**2 
                                 + a*b*c/(l**2*L)*sin(2*φ) )
                       +β**2+γ**2-2*α*(β*a*b-γ*l**2)/(l*L)*cos(φ)
                       -2*α*β*c/l*sin(φ) )
        k =  np.clip( sqrt( 4*V0/(A0+2*V0) ) , 0., 1.)
        Ψ = (1-k**2/2)*ellK(k**2)-ellE(k**2)
        cst = μ0*Rs/π
        
        
        V02 = (α**2*( (1-b**2*c**2/(l**2*L**2))*cos(φ)**2 
                     + c**2/l**2*sin(φ)**2 
                     + a*b*c/(l**2*L)*sin(2*φ) )
               +β**2 + γ**2 - 2*α*(β*a*b-γ*l**2)/(l*L)*cos(φ)
               -2*α*β*c/l*sin(φ) )
        this.arg = A0, V0,V02, k, Ψ
        
        return cst*(p1*cos(φ)+p2*sin(φ)+p3)*Ψ/(k*sqrt(V0**3))

    mutual, err = quad(integrand,0,2*π, args=(l,L,p1,p2,p3,p4,p5), epsrel=1e-7, limit=200)

    return mutual, err