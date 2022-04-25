# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 14:32:14 2021

@author: Aime Labbe
"""
import numpy as np
from numpy import sin, cos, sqrt
from scipy.constants import mu_0 as μ0, pi as π
from scipy.integrate import quad
from scipy.special import ellipk as ellK, ellipe as ellE
from scipy.special import ellipkinc as ellK_inc, ellipeinc as ellE_inc
from pycoilib.lib.misc.geometry import _vec_0

from ..segment.segment import Circle, ArcAbstract
from typing import Union


def Ψ_p(k):
    return (1 - k ** 2 / 2) * ellK(k ** 2) - ellE(k ** 2)


def Ψ(φ1, φ0, k):
    res = ((1 - k ** 2 / 2) * (ellK_inc(φ1, k ** 2) - ellK_inc(φ0, k ** 2)) / 2
           - (ellE_inc(φ1, k ** 2) - ellE_inc(φ0, k ** 2)) / 2)
    return res


def Θ(φ1, φ0, k):
    return (sqrt(1 - k ** 2 * sin(φ1) ** 2) - sqrt(1 - k ** 2 * sin(φ0) ** 2)) / 2


def calc_M_arcNcircle(arc: ArcAbstract, circle: Circle):
    """Compute the mutual inductance between an arc (or circle) and a circle."""
    Rp = arc.radius
    Rs = circle.radius

    s0_u = (circle.vec_r0 - arc.vec_r0)
    s0 = sqrt(s0_u @ s0_u)
    if s0 != 0.:
        s0_u = s0_u / s0
    else:
        s0_u = _vec_0.copy()

    θ_max = arc.theta

    x_u = arc.vec_x
    y_u = arc.vec_y

    u_u = circle.vec_x
    v_u = circle.vec_y

    def integrand(θ):
        A = Rp ** 2 + Rs ** 2 + s0 ** 2 - 2 * Rp * s0 * (s0_u @ x_u * cos(θ) + s0_u @ y_u * sin(θ))
        B = -2 * Rs * (Rp * (u_u @ x_u * cos(θ) + u_u @ y_u * sin(θ)) - s0 * s0_u @ u_u)
        C = -2 * Rs * (Rp * (v_u @ x_u * cos(θ) + v_u @ y_u * sin(θ)) - s0 * s0_u @ v_u)

        d = sqrt(B ** 2 + C ** 2)
        k = np.clip(sqrt(2 * d / (A + d)), 0., 1.)

        # k in [0,1], but rounding errors can make it k > 1
        k = np.clip(sqrt(2 * d / (A + d)), 0., 1.)

        pre = 2 * μ0 * Rp * Rs / π

        fct1 = 1 / (k ** 2 * d * sqrt(A + d))
        fct3 = (-u_u @ x_u * C + v_u @ x_u * B) * sin(θ) + (u_u @ y_u * C - v_u @ y_u * B) * cos(θ)

        return pre * fct1 * (fct3 * Ψ_p(k))

    mutual, err = quad(integrand, 0, θ_max, epsrel=1e-5, limit=200)

    return mutual, err
