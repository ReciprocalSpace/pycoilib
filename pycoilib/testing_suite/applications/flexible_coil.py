# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:56:43 2021

@author: Aimé Labbé
"""

import numpy as np
import matplotlib.pyplot as plt
import pycoilib as pycoil

from numpy import pi as π

vec_x = np.array([1., 0., 0.])
vec_y = np.array([0., 1., 0.])
vec_z = np.array([0., 0., 1.])
vec_0 = np.array([0., 0., 0.])

wire = pycoil.wire.Wire_circ(1e-3)

inductance = []

ell = 0.10
w = 0.08

n = 2
θ = np.linspace(0, π/100, n)

for θ_i in θ:
    print("*"*32)
    if θ_i == 0.:
        p0, p1 = np.array([ell/2, 0., w/2]), np.array([-ell/2, 0., w/2])
        p2, p3 = np.array([-ell/2, 0., -w/2]), np.array([ell/2, 0., -w/2])

        pos0, pos1 = np.array([-ell / 2, 0., 0.]), np.array([ell / 2, 0., 0.])

        side1 = pycoil.segment.Line(p0, p1)
        side2 = pycoil.segment.Line(p2, p3)

        arc1 = pycoil.segment.Arc.from_rot(w/2, π, -π/2, pos1, vec_x, -π/2).flip_current_direction()
        arc2 = pycoil.segment.Arc.from_rot(w/2, π, π/2, pos0, vec_x, -π/2).flip_current_direction()

    else:
        R = ell / θ_i

        p0, p1 = np.array([0., R, w / 2]), np.array([0, R, -w / 2])

        side1 = pycoil.segment.Arc.from_rot(R, θ_i, -θ_i/2, p0, vec_z, -π/2).flip_current_direction()
        side2 = pycoil.segment.Arc.from_rot(R, θ_i, -θ_i/2, p1, vec_z, -π/2)

        p0, p1 = side1.get_endpoints()
        p2, p3 = side2.get_endpoints()

        axis_2 = (p0 @ vec_x) * vec_x + (p0 @ vec_y) * vec_y - np.array([0., R, 0.])
        axis_1 = (p2 @ vec_x) * vec_x + (p2 @ vec_y) * vec_y - np.array([0., R, 0.])

        arc2 = pycoil.segment.Arc.from_endpoints(p1, p2, π, axis_1)
        arc1 = pycoil.segment.Arc.from_endpoints(p3, p0, π, axis_2)

    coil = pycoil.coil.Coil([side1, side2, arc1, arc2], wire)

    coil.draw(True)
    inductance.append(coil.get_inductance())

inductance = np.array(inductance)
#
fig = plt.figure(figsize=(7.5 / 2.54, 5.5 / 2.54))
ax = plt.gca()
ax.plot(θ/π, inductance * 1e6)
#
ax.set_xlabel(r"Bending angle [π]")
ax.set_ylabel(r"Self inductance [μH]")

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# ax.text(0.02, 0.99, r"$d = 50$ mm", transform=ax.transAxes, fontsize=8,
# verticalalignment='top', )
fig.tight_layout()
# # fig.savefig("Appli-boucle deformee.png", dpi=300)
plt.show()

#                         

# print("-----------")
# shape = pycoil.shape.Arc.from_endpoints(vec_x, -vec_x, π,  vec_z)
# shape.draw()
# print(shape)
