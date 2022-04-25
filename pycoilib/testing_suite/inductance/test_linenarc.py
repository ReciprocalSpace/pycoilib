# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 11:53:00 2021

@author: Aime Labbé
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as π, sqrt, cos, sin

import pycoilib as pycoil
from pycoilib.lib.inductance.calc_m_arcNline import calc_M_arc_line, calc_M_arc_line_2
from pycoilib.lib.inductance.calc_m_2arcs import calc_M_2arcs

def test():
    R = 0.99
    n = 360 * 10 + 1
    θ = np.expand_dims(np.linspace(0, 2 * π, n), axis=1)

    s0_u = np.array([1., 0., 0.])
    s0 = 1

    s_u = np.array([1., -1., 0.])
    s_u = s_u / sqrt(s_u @ s_u)

    _p_u = np.array([[1., 0., 0.]] * n) * cos(θ) + np.array([[0., 1., 0.]] * n) * sin(θ)

    root = []
    for p_u in _p_u:
        if not np.isclose((p_u @ s_u) ** 2, 1):
            root.append(s0 * (p_u @ s0_u - (s0_u @ s_u) * (p_u @ s_u)) / (1 - (p_u @ s_u) ** 2))
        else:
            root.append(0)
    root = np.array(root)

    zero_crossings = np.where(np.diff(np.sign(root - R)))[0]

    def add_zero_bar():
        for z in zero_crossings:
            if np.absolute(root[z] - R) < (R / 10):
                plt.plot([θ[z] / π, θ[z] / π], [-1, 1])

    plt.plot(θ / π, root)
    add_zero_bar()
    plt.plot([0, 2], [R, R])
    plt.ylim([-5, 5])
    plt.show()

    β2 = []
    _σ = []

    res = []
    p = R

    def σ(s, p_u):
        return s + (s0 * (s0_u @ s_u) - p * (p_u @ s_u))

    for p_u in _p_u:
        tmp = max(s0 ** 2 * (1 - (s0_u @ s_u) ** 2)
                  + p ** 2 * (1 - (p_u @ s_u) ** 2)
                  - 2 * s0 * p * ((p_u @ s0_u) - (p_u @ s_u) * (s0_u @ s_u)), 1e-15)

        β2.append(tmp)

        σ1 = σ(2, p_u)
        σ0 = σ(0, p_u)
        _σ.append([σ1, σ0])

        res.append(np.arcsinh(σ1 / sqrt(tmp)) - np.arcsinh(σ0 / sqrt(tmp)))

    β = sqrt(np.clip(np.array(β2), 0, np.inf))

    plt.plot(θ / π, β)
    plt.plot(θ / π, _σ)
    add_zero_bar()
    plt.show()

    plt.plot(θ / π, res)
    add_zero_bar()
    plt.show()


def test_two_approaches():
    ell = 0.10
    w = 0.08

    p0, p1 = np.array([ell / 2, 0., w / 2]), np.array([-ell / 2, 0., w / 2])
    pos0, pos1 = np.array([-ell / 2, 0., 0.]), np.array([ell / 2, 0., 0.])

    line = pycoil.segment.Line(p0, p1)
    arc = pycoil.segment.Arc.from_rot(w / 2, π, π / 2, pos0, np.array([1., 0., 0.]), -π/2).flip_current_direction()

    print(line)
    print(arc)

    coil = pycoil.coil.Coil([line, arc])
    coil.draw()

    print(calc_M_arc_line(arc, line))
    print(calc_M_arc_line_2(arc, line))

    vec_x = np.array([1., 0., 0.])
    vec_y = np.array([0., 1., 0.])
    vec_z = np.array([0., 0., 1.])

    θ_i = π/200
    R = ell / θ_i

    p0, p1 = np.array([0., R, w / 2]), np.array([0, R, -w / 2])

    side1 = pycoil.segment.Arc.from_rot(R, θ_i, -θ_i / 2, p0, vec_z, -π / 2).flip_current_direction()
    side2 = pycoil.segment.Arc.from_rot(R, θ_i, -θ_i / 2, p1, vec_z, -π / 2)

    p0, p1 = side1.get_endpoints()
    p2, p3 = side2.get_endpoints()

    axis_2 = (p0 @ vec_x) * vec_x + (p0 @ vec_y) * vec_y - np.array([0., R, 0.])
    axis_1 = (p2 @ vec_x) * vec_x + (p2 @ vec_y) * vec_y - np.array([0., R, 0.])

    arc2 = pycoil.segment.Arc.from_endpoints(p1, p2, π, axis_1)
    arc1 = pycoil.segment.Arc.from_endpoints(p3, p0, π, axis_2)

    print(calc_M_2arcs(side1, arc2))
    coil = pycoil.coil.Coil([side1, arc2])
    coil.draw()


if __name__ == "__main__":
    test_two_approaches()
    test()


