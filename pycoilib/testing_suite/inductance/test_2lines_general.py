# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:20:39 2021

@author: Aime Labbé
"""

import numpy as np
from numpy import pi as π, cos, sin, arctanh, arccos, tan, sqrt, arctan
from scipy.constants import mu_0 as μ0
import pycoilib as pycoil
import pycoilib.lib.misc.geometry as geo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('lines', linewidth=2)
plt.rc('font', size=9)


def calc_M_general(line1, line2):
    Lp = line1.ell
    Ls = line2.ell

    p_u = line1.vec_n
    s_u = line2.vec_n

    # Orthogonal lines
    if np.isclose(p_u @ s_u, 0, 1e-8):
        return 0.

    s0 = np.linalg.norm(line2.vec_r0 - line1.vec_r0)
    s0_u = np.zeros(3, dtype=float) if s0 == 0. else (line2.vec_r0 - line1.vec_r0)/s0

    # Colinear lines
    if all(np.isclose([abs(p_u @ s_u), abs(s0_u @ p_u), abs(s0_u @ s_u)], np.ones(3))):
        def σ(s: float, p: float):
            return s + (s0 * (s0_u @ s_u) - p * (p_u @ s_u))
        # Endpoints in p are removed from the integral to avoid σ = 0
        σ11, σ10 = σ(Ls, Lp), σ(Ls, 0)
        σ01, σ00 = σ(0, Lp), σ(0, 0.)
        cst = μ0 * (p_u @ s_u) / (4 * π)

        partial_sum = 0.
        for σii in [σ11, -σ10, σ01, -σ00]:
            partial_sum += σii*np.log(np.abs(σii)) if σii != 0. else 0.

        # -(s0_u @ s_u) * cst * (σ11 * np.log(np.abs(σ11)) - σ10 * np.log(np.abs(σ10))
        #                        - σ01 * np.log(np.abs(σ01)) + σ00 * np.log(np.abs(σ00)))
        return -(s0_u @ s_u) * cst * partial_sum

    # General case
    vec_n = geo.normalize(np.cross(p_u, s_u))

    p0 = line1.vec_r0
    p2 = line2.vec_r0
    # p0 - α*p_u + d*vec_n + β*s_u = p2
    np.array([-p_u, vec_n, s_u])
    αdβ = np.linalg.solve(np.array([-p_u, -vec_n, s_u]).T, (p2 - p0))
    α, d, β = tuple(αdβ)

    θ = arccos(p_u @ s_u)
    if (θ % π/2) == 0.:
        return 0.

    R1 = sqrt(d ** 2 + (α + Lp) ** 2 + (β + Ls) ** 2 - 2 * (α + Lp) * (β + Ls) * cos(θ))
    R2 = sqrt(d ** 2 + (α + Lp) ** 2 + β ** 2 - 2 * (α + Lp) * β * cos(θ))
    R3 = sqrt(d ** 2 + α ** 2 + β ** 2 - 2 * α * β * cos(θ))
    R4 = sqrt(d ** 2 + α ** 2 + (β + Ls) ** 2 - 2 * α * (β + Ls) * cos(θ))

    Mp = (μ0 / (2 * π) * cos(θ) * ((α + Lp) * arctanh(np.clip(Ls / (R1 + R2), -1., 1.))
                                   + (β + Ls) * arctanh(np.clip(Lp / (R1 + R4), -1., 1.))
                                   - α * arctanh(np.clip(Ls / (R3 + R4), -1., 1.))
                                   - β * arctanh(np.clip(Lp / (R2 + R3), -1., 1.)))
          )

    if (θ % π) != 0. and R1 != 0. and d != 0.:
        Ω = (arctan((d ** 2 * cos(θ) + (α + Lp) * (β + Ls) * sin(θ) ** 2) / (d * R1 * sin(θ)))
             - arctan((d ** 2 * cos(θ) + (α + Lp) * β * sin(θ) ** 2) / (d * R2 * sin(θ)))
             + arctan((d ** 2 * cos(θ) + α * β * sin(θ) ** 2) / (d * R3 * sin(θ)))
             - arctan((d ** 2 * cos(θ) + α * (β + Ls) * sin(θ) ** 2) / (d * R4 * sin(θ)))
             )

        Mp -= μ0 / (4 * π) * Ω * d / tan(θ)

    # Mp = (μ0 / (2 * π) * cos(θ) * ((α + Lp) * arctanh(Ls / (R1 + R2))
    #                                + (β + Ls) * arctanh(Lp / (R1 + R4))
    #                                - α * arctanh(Ls / (R3 + R4))
    #                                - β * arctanh(Lp / (R2 + R3)))
    #       - μ0 / (4 * π) * Ω * d / tan(θ))

    return Mp


p0, p1 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
p2, p3 = np.array([0.25, 0.3, 0.]), np.array([1.25, 0.4, 0.4])

points = geo.fibonacci_sphere(64)

Mth = []
Mour = []
line1 = pycoil.segment.Line(p0, p1)
for p in points:
    for q in points:
        p2 = p * 0.5
        p3 = p2 + q * 2.
        line2 = pycoil.segment.Line(p2, p3)

        # pycoil.coil.Coil([line1,line2]).draw()
        Mth.append(calc_M_general(line1, line2))
        m, tmp = pycoil.calc_mutual(line1, line2)
        Mour.append(m)

Mth = np.array(Mth)
Mour = np.array(Mour)

# Affichage
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig = plt.figure(figsize=(7.5 / 2.54, 5.5 / 2.54))
ax = plt.gca()

# xy = [Mour.min() * 1e9, Mour.max() * 1e9]  # droite y=x
# print(xy)
# ax.plot(xy, xy, alpha=0.8, c=colors[1])

ax.plot((Mth - Mour)/(Mth+1e-18)*100, 'o', c=colors[0], markersize=2)  # data

ax.set_xlabel(r"Mutual (our approach) [nH]")
ax.set_ylabel(r"Error [%]")

fig.tight_layout()
fig.savefig("test-2lines-general.png", dpi=300)
plt.show()


def test_colinear():
    p0, p1 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
    p2, p3 = np.array([1., 0., 0.]), np.array([2., 0., 0.])
    line1 = pycoil.segment.Line(p0, p1)
    line2 = pycoil.segment.Line(p2, p3)
    print(calc_M_general(line1, line2))
    print(pycoil.calc_mutual(line1, line2))


# def test_coplanar():
if __name__ == "__main__":
    test_colinear()
