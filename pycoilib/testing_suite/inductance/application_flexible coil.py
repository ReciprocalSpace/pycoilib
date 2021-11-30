# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:26:05 2021

@author: Aime Labbe
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import pi as π
from pathlib import Path

import pycoilib as pycoil
from pycoilib.segment import Arc, Line, Circle

# plt.rc('xtick',labelsize=8)
# plt.rc('ytick',labelsize=8)
# plt.rc('lines', linewidth=2)
# plt.rc('font', size=9)

vec_x = np.array([1., 0., 0.])
vec_y = np.array([0., 1., 0.])
vec_z = np.array([0., 0., 1.])
vec_0 = np.array([0., 0., 0.])

ell = 0.10
a = 0.001
θ = np.linspace(0., 2 * π, round(360 / 5 + 1))

wire = pycoil.wire.Wire_circ(a)

I = []

# Premier cas : line
line = Line(np.array([0., 0., 0.]), np.array([0., 0., ell]))
coil = pycoil.coil.Coil([line], wire)
I.append(coil.get_inductance())

# Premier line : arc
for θ_i in θ[1:]:
    R = ell / θ_i
    arc = Arc.from_normal(R, arc_angle=θ_i, arc_angular_pos=0, position=vec_0, normal=VEC_Y)
    arc = Arc.from_normal(R, arc_angle=θ_i, arc_angular_pos=0, position=vec_0, normal=VEC_Y)
    arc = Arc(R, θ_i, vec_0, vec_x, vec_y, vec_z)
    coil = pycoil.coil.Coil([arc], wire)
    I.append(coil.get_inductance())
I = np.array(I)

loop = Circle(R)
I_loop = pycoil.coil.Coil([loop], wire).get_inductance()
I_line = I[0]

fig = plt.figure(figsize=(6.5 / 2.54, 5. / 2.54), dpi=300)
ax = plt.gca()

plt.plot(θ / (2 * π), I / 1e-9, )
plt.plot(θ / (2 * π), I ** 0 * I_line / 1e-9, "gray", lw=1, alpha=0.5)
plt.plot(θ / (2 * π), I ** 0 * I_loop / 1e-9, "gray", lw=1, alpha=0.5)

# ax.set_ylim([55,90])
ax.set_xlabel(r"Arc angle [$2\pi$]")
ax.set_ylabel(r"Inductance [nH]")

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.05, 0.18, r"Self of a Circle", transform=ax.transAxes, fontsize=8,
        verticalalignment='top', c="gray")

ax.text(0.60, 0.92, r"Self of a line", transform=ax.transAxes, fontsize=8,
        verticalalignment='top', c="gray")

ax.text(0.60, 0.92, r"Self of a line", transform=ax.transAxes, fontsize=8,
        verticalalignment='top', c="gray")

fig.tight_layout()
# fig.savefig(Path("output/Appli-fil-courbe.png"), dpi=300)
plt.show()
