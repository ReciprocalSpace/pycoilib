# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:32:14 2021

@author: Aime Labbe
"""


def calc_M_arcNloop(loop1, loop2):
    Rp = loop1.R
    Rs = loop2.R
    
    s0_u = (loop2.vec_r0-loop1.vec_r0)
    s0 = sqrt(s0_u@s0_u)
    if s0 != 0.:
        s0_u = s0_u/s0
    else:
        s0_u = _vec_0.copy()
    
    θ_max = loop1.theta

    x_u = loop1.vec_x
    y_u = loop1.vec_y
    
    u_u = loop2.vec_x
    v_u = loop2.vec_y
    
    def integrand(θ):
        A = Rp**2 + Rs**2 + s0**2 - 2*Rp*s0*( s0_u@x_u*cos(θ) + s0_u@y_u*sin(θ) )
        B = -2*Rs*( Rp*(u_u@x_u*cos(θ) + u_u@y_u*sin(θ)) - s0*s0_u@u_u )
        C = -2*Rs*( Rp*(v_u@x_u*cos(θ) + v_u@y_u*sin(θ)) - s0*s0_u@v_u )

        d = sqrt(B**2 + C**2)
        k = np.clip(sqrt(2*d/(A+d)),0.,1.)
        
        # k in [0,1], but rounding errors can make it k > 1
        k = np.clip(sqrt(2*d/(A+d)), 0., 1.)

        pre = 2*μ0*Rp*Rs/π 
        
        fct1 = 1/(k**2*d*sqrt(A+d))
        fct3 = (-u_u@x_u*C+v_u@x_u*B)*sin(θ) + (u_u@y_u*C-v_u@y_u*B)*cos(θ)

        return pre*fct1*(fct3*Ψ_p(k))
    
    mutual, err = quad(integrand, 0, θ_max, epsrel=1e-5, limit=200 )
    
    return mutual, err