# # -*- coding: utf-8 -*-
# """
# Created on Thu Dec  3 13:02:54 2020
#
# @author:
# """
#
# import numpy as np
# from numpy import cos, sin, sqrt, sign
# from scipy.integrate import quad
# from scipy.constants import mu_0 as μ0, pi as π
#
# from scipy.special import ellipe as ellE, ellipk as ellipK
# import pycoillib.geometry as geo
#
#
# def calc_M_2wires(wire1, wire2):
#     """Compute the mutual inductance between a pair of wires.
#
#     Parameters
#     ----------
#     wire1 : 2darray
#         Endpoint coordinates of the first wire.
#     wire2 : 2darray
#         Endpoint coordinates of the second wire
#
#     Returns
#     -------
#     Tuple
#         Tuple containing the mutual inductance and its estimated error.
#
#     """
#     p, s = np.copy(wire1), np.copy(wire2) # p : primary; s : secondary
#     Lp, Ls = geo.length(p), geo.length(s)
#
#     z_u = (p[1]-p[0])/Lp
#     n_u = (s[1]-s[0])/Ls
#
#     r0 = sqrt( (s[0]-p[0])@(s[0]-p[0]) )
#     r_u = (s[0]-p[0])/r0
#
#     def integrand(x):
#         ε = Ls*1e-11
#         β = (r0**2 * (1-(r_u@n_u)**2)
#              + x**2*(1-(z_u@n_u)**2)
#              + 2*r0*x*((z_u@n_u)*(r_u@n_u)-(z_u@r_u)) )
#         σ1 = Ls-ε + (r0*(r_u@n_u) - x* (z_u@n_u))
#         σ0 = 0+ε + (r0*(r_u@n_u) - x* (z_u@n_u))
#         cst = μ0*(z_u@n_u)/(4*π)
#         if isinstance(β, np.ndarray):
#             fct = np.zeros_like(β)
#             ind = np.logical_and(β*1e18>σ0**2, β*1e18>σ1**2)
#             not_ind = np.logical_not(ind)
#             fct[ind] = cst[ind]*(
#                 np.arctanh(σ1[ind]/sqrt(σ1[ind]**2+β[ind]))
#                -np.arctanh(σ0[ind]/sqrt(σ0[ind]**2+β[ind])) )
#             fct[not_ind] = cst[not_ind]*(
#                 sign(σ1[not_ind])*np.log(np.absolute(σ1[not_ind]))
#                -sign(σ0[not_ind])*np.log(np.absolute(σ0[not_ind])) )
#         else:
#             if β*1e6>σ0**2 and β*1e6>σ1**2:
#                 fct = cst * ( np.arctanh(σ1/sqrt(σ1**2+β))
#                              -np.arctanh(σ0/sqrt(σ0**2+β)) )
#             else:
#                 fct = cst*( sign(σ1)*np.log(np.absolute(σ1))
#                            -sign(σ0)*np.log( p.absolute(σ0)) )
#         return fct
#     ε = Lp*1e-11
#     mutual, err = quad(integrand, ε, Lp-ε )
#
#     return mutual, err
#
# def calc_M_loopNwire(radius, wire):
#     ρ0 = radius
#     wire = np.copy(wire)
#     Ls = geo.length(wire)
#     n_u = (wire[1]-wire[0])/Ls
#     s0 = sqrt(wire[0]@wire[0])
#     s0_u = wire[0]/s0
#
#     def integrand(θ):
#         ε = Ls*1e-7
#         ρ_u  = np.array([ cos(θ), sin(θ), np.zeros_like(θ)])
#         dρ_u = np.array([-sin(θ), cos(θ), np.zeros_like(θ)])
#
#         β = (  s0**2*(1 - (s0_u @n_u)**2)
#              + ρ0**2*(1 - (ρ_u.T@n_u)**2)
#              + 2*s0*ρ0*( (ρ_u.T@n_u)*(s0_u@n_u) - (s0_u@ρ_u) ) )
#         σ0 = ε + (s0*n_u@s0_u-ρ0*n_u@ρ_u)
#         σ1 = Ls-ε + (s0*n_u@s0_u-ρ0*n_u@ρ_u)
#
#         cst = μ0/(4*π)*(ρ0*n_u@dρ_u)
#         if isinstance(β, np.ndarray):
#             fct = np.zeros_like(β)
#             ind = np.logical_and( np.absolute(β)*1e18>σ0**2,
#                                   np.absolute(β)*1e18>σ1**2 )
#             not_ind = np.logical_not(ind)
#             fct[ind] =cst[ind]*(np.arctanh(σ1[ind]/sqrt(σ1[ind]**2+β[ind]))
#                                -np.arctanh(σ0[ind]/sqrt(σ0[ind]**2+β[ind])))
#             fct[not_ind] = cst[not_ind]*(
#                  sign(σ1[not_ind])*np.log(np.absolute(σ1[not_ind]))
#                 -sign(σ0[not_ind])*np.log(np.absolute(σ0[not_ind])) )
#         else:
#             if abs(β)*1e18>σ0**2 and abs(β)*1e18>σ1**2:
#                 fct = cst * ( np.arctanh(σ1/sqrt(σ1**2+β))
#                              -np.arctanh(σ0/sqrt(σ0**2+β)))
#             else:
#                 fct = cst*(  sign(σ1)*np.log(np.absolute(σ1))
#                            - sign(σ0)*np.log(np.absolute(σ0)) )
#
#         return fct
#     mutual, err = quad(integrand, 0, 2*π)
#     return mutual, err
#
# def calc_M_2loops(Rp,Rs,C=(0,0,0),n=(0,0,1)):
#     xc,yc,zc = tuple(C)
#     a,b,c = tuple(n)
#
#     α, β, γ, δ  = Rs/Rp, xc/Rp, yc/Rp, zc/Rp
#
#     l = sqrt(a**2+c**2)
#     L = sqrt(a**2+b**2+c**2)
#
#     if a==0 and c==0:
#         p1 = 0
#         p2 = -γ*sign(b)
#         p3 = 0
#         p4 = -β*sign(b)
#         p5 = δ
#     else:
#         p1 = γ*c/l
#         p2 = -(β*l**2+γ*a*b)/(l*L)
#         p3 = α*c/L
#         p4 = -(β*a*c-γ*l**2+δ*b*c)/(l*L)
#         p5 = -(β*c-δ*a)/l
#
#     def integrand(φ,l,L,p1,p2,p3,p4,p5):
#         A0 = 1+α**2+β**2+γ**2+δ**2 + 2*α*(p4*cos(φ)+p5*sin(φ))
#         if l==0:
#             V0 =  sqrt( β**2+γ**2+α**2*cos(φ)**2-2*α*β*cos(φ) )
#         else:
#             V0 =  sqrt( α**2 *( (1-b**2*c**2/(l**2*L**2))*cos(φ)**2
#                                   + c**2/l**2*sin(φ)**2
#                                   + a*b*c/(l**2*L)*sin(2*φ) )
#                        +β**2+γ**2-2*α*(β*a*b-γ*l**2)/(l*L)*cos(φ)
#                        -2*α*β*c/l*sin(φ) )
#         k = sqrt( 4*V0/(A0+2*V0) )
#         Ψ = (1-k**2/2)*ellipK(k**2)-ellE(k**2)
#         cst = μ0*Rs/π
#         return cst*(p1*cos(φ)+p2*sin(φ)+p3)*Ψ/(k*sqrt(V0**3))
#
#     mutual, err = quad(integrand,0,2*π, args=(l,L,p1,p2,p3,p4,p5))
#
#     return mutual, err
#
#
# def calc_M_2arcs():
#     pass
#
# def calc_M_2loops2():
#     pass
#
# def calc_M_arcNwire(wire, R, theta,):