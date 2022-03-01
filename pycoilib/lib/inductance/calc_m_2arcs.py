# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:16:50 2021

@author: Aimé Labbé
"""

import numpy as np
from numpy import pi as π
from scipy.integrate import quad
from scipy.special import ellipk, ellipe
from scipy.special import ellipkinc, ellipeinc
from scipy.constants import mu_0 as μ0

from typing import Tuple


# TODO: comment

def Ψ_p(k):
    return (1 - k ** 2 / 2) * ellipk(k ** 2) - ellipe(k ** 2)


def Ψ(φ1, φ0, k):
    res = ((1 - k ** 2 / 2) * (ellipkinc(φ1, k ** 2) - ellipkinc(φ0, k ** 2)) / 2
           - (ellipeinc(φ1, k ** 2) - ellipeinc(φ0, k ** 2)) / 2)
    return res


def Θ(φ1, φ0, k):
    return (np.sqrt(1 - k ** 2 * np.sin(φ1) ** 2) - np.sqrt(1 - k ** 2 * np.sin(φ0) ** 2)) / 2


def calc_M_2arcs(arc1, arc2) -> Tuple[float, float]:
    """
    Computes the mutual inductance between a pair of arcs.
    Parameters
    ----------
    arc1
    arc2

    Returns
    -------

    """
    Rp = arc1.radius
    Rs = arc2.radius

    s0_u = (arc2.vec_r0 - arc1.vec_r0)
    s0 = np.sqrt(s0_u @ s0_u)
    if s0 == 0:
        s0_u = np.array([0., 0., 0.])
    else:
        s0_u = s0_u / s0

    x_u = arc1.vec_x
    y_u = arc1.vec_y

    u_u = arc2.vec_x
    v_u = arc2.vec_y

    sφ = arc2.theta

    def integrand(θ):
        A = Rp ** 2 + Rs ** 2 + s0 ** 2 - 2 * Rp * s0 * (s0_u @ x_u * np.cos(θ) + s0_u @ y_u * np.sin(θ))
        B = -2 * Rs * (Rp * (u_u @ x_u * np.cos(θ) + u_u @ y_u * np.sin(θ)) - s0 * s0_u @ u_u)
        C = -2 * Rs * (Rp * (v_u @ x_u * np.cos(θ) + v_u @ y_u * np.sin(θ)) - s0 * s0_u @ v_u)

        d = np.sqrt(B ** 2 + C ** 2)
        k = np.clip(np.sqrt(2 * d / (A + d)), 0., 1.)

        κ = np.arctan2(C, B)
        φ1, φ0 = (sφ - κ) / 2, -κ / 2

        pre = 2 * μ0 * Rp * Rs / π
        fct1 = 1 / (k ** 2 * d * np.sqrt(A + d))
        fct2 = (-u_u @ x_u * B - v_u @ x_u * C) * np.sin(θ) + (u_u @ y_u * B + v_u @ y_u * C) * np.cos(θ)
        fct3 = (-u_u @ x_u * C + v_u @ x_u * B) * np.sin(θ) + (u_u @ y_u * C - v_u @ y_u * B) * np.cos(θ)

        return pre * fct1 * (fct2 * Θ(φ1, φ0, k) + fct3 * Ψ(φ1, φ0, k))



    output = quad(integrand, 1e-8, arc1.theta - 1e-8, epsabs=1e-15, limit=200)
    mutual, err = output[0], output[1:]

    return mutual, err
