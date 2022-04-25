# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:33:02 2021

@author: Aime Labbé
"""
import numpy as np
from numpy import pi as π
from scipy.integrate import quad
from scipy.constants import mu_0 as μ0

from scipy.special import ellipk, ellipe
from scipy.special import ellipkinc, ellipeinc

from ..segment.segment import Line, ArcAbstract

# TODO: clean this file. Verify which of these two functions works best
# TODO: documentation and comments


def calc_M_arc_line(arc: ArcAbstract, line: Line):
    """Comupute the mutual between an arc and a line."""
    VEC_0 = np.array([0., 0., 0.])

    Rp = arc.radius
    x_u = arc.vec_x
    y_u = arc.vec_y
    θ1 = arc.theta

    s_u = line.vec_n
    Ls = line.ell

    s0 = np.sqrt((line.vec_r0 - arc.vec_r0) @ (line.vec_r0 - arc.vec_r0))
    if s0 != 0.:
        s0_u = (line.vec_r0 - arc.vec_r0) / s0
    else:
        s0_u = VEC_0.copy()

    def integrand(θ):
        p_u = x_u*np.cos(θ) + y_u*np.sin(θ)
        dp_u = -x_u*np.cos(θ) + y_u*np.cos(θ)

        β2 = Rp**2 * (1 - (p_u@s_u)**2) + s0**2 * (1 - (s0_u@s_u)**2) - 2*Rp*s0 * (s0_u@p_u - (s0_u@s_u)*(p_u@s_u))

        β = np.sqrt(np.clip(β2, 0., np.inf))

        σ1 = Ls + s0*(s_u@s0_u) - Rp*(s_u@p_u)
        σ0 = 0. + s0*(s_u@s0_u) - Rp*(s_u@p_u)

        pre = μ0/(4*π) * Rp*(s_u @ dp_u)

        if isinstance(β, np.ndarray):
            fct = np.zeros_like(β)
            ind = β != 0.
            not_ind = np.logical_not(ind)
            fct[ind] = pre[ind] * (np.arcsinh(σ1[ind]/β[ind]) - np.arcsinh(σ0[ind]/β[ind]))
            fct[not_ind] = pre[not_ind] * (
                    np.sign(σ1[not_ind]) * np.log(np.absolute(σ1[not_ind]))
                    - np.sign(σ0[not_ind]) * np.log(np.absolute(σ0[not_ind]))
            )
        else:
            if β != 0.:
                fct = pre*(np.arcsinh(σ1/β) - np.arcsinh(σ0/β))
            else:
                fct = pre * (np.sign(σ1)*np.log(np.absolute(σ1)) - np.sign(σ0)*np.log(np.absolute(σ0)))

        return fct
    p0, p1 = line.get_endpoints()
    p2, p3 = arc.get_endpoints()

    points = []
    if all(np.isclose(p2, p0)) or all(np.isclose(p2, p1)):
        points.append(0.)
    if all(np.isclose(p3, p1)) or all(np.isclose(p3, p1)):
        points.append(θ1)
    print(points)

    output = quad(integrand, 0, θ1, epsabs=1e-10, limit=400, points=points)
    mutual, err = output[0], output[1:]

    return mutual, err

def Ψ_p(k):
    return (1 - k ** 2 / 2) * ellipk(k ** 2) - ellipe(k ** 2)


def Ψ(φ1, φ0, k):
    res = ((1 - k ** 2 / 2) * (ellipkinc(φ1, k ** 2) - ellipkinc(φ0, k ** 2)) / 2
           - (ellipeinc(φ1, k ** 2) - ellipeinc(φ0, k ** 2)) / 2)
    return res


def Θ(φ1, φ0, k):
    return (np.sqrt(1 - k**2 * np.sin(φ1)**2) - np.sqrt(1 - k**2 * np.sin(φ0)**2)) / 2


def calc_M_arc_line_2(arc: ArcAbstract, line: Line):
    Rs = arc.radius
    s0_u = (arc.vec_r0 - line.vec_r0)
    s0 = np.sqrt(s0_u @ s0_u)
    if np.isclose(s0, 0.):
        s0_u = np.array([0., 0., 0.])
    else:
        s0_u = s0_u / s0

    u_u = arc.vec_x
    v_u = arc.vec_y

    sφ = arc.theta

    p_u = line.vec_n

    def integrand(p: float):
        A = p**2 + Rs**2 + s0**2 - 2*p*s0 * (p_u @ s0_u)
        B = -2*Rs * (p*u_u@p_u - s0*s0_u@u_u)
        C = -2*Rs * (p*v_u@p_u - s0*s0_u@v_u)

        d = np.sqrt(B**2 + C**2)
        k = np.clip(np.sqrt(2*d/(A+d)), 0., 1.)

        κ = np.arctan2(C, B)
        φ1, φ0 = (sφ - κ) / 2, -κ / 2

        pre = 2 * μ0 * Rs / π
        fct1 = 1 / (k ** 2 * d * np.sqrt(A + d))
        fct2 = u_u@p_u*B + v_u@p_u*C
        fct3 = u_u@p_u*C - v_u@p_u*B

        return pre * fct1 * (fct2 * Θ(φ1, φ0, k) + fct3 * Ψ(φ1, φ0, k))

    output = quad(integrand, 1e-8, line.ell - 1e-8, epsabs=1e-15, limit=200)
    mutual, err = output[0], output[1:]

    return mutual, err
