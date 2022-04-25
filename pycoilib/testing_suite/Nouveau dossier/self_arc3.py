# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 15:18:00 2021

@author: utric
"""
import numpy as np
from numpy import cos, sin, arctan2 as atan, sqrt, pi as π, sign, log
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
from scipy.integrate import quad
from scipy.special import ellipk as ellK,  ellipe as ellE
from scipy.special import ellipkinc as ellK_inc,  ellipeinc as ellE_inc
from scipy.constants import mu_0 as μ0

import pycoilib
from pycoilib.inductance import calc_M_2arcs
from pycoilib.shape import Arc, Line, Loop

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rc('lines', linewidth=2)
plt.rc('font', size=9)

vec_x=np.array([1.,0.,0.])
vec_y=np.array([0.,1.,0.])
vec_z=np.array([0.,0.,1.])
vec_0=np.array([0.,0.,0.])


ell = 0.10
a = 0.001

fig = plt.figure()
ax = plt.gca()

wire = pycoilib.wire.WireCircular(a)


θ = np.linspace(0.,2*π, round(360/5+1) )
I = []

# Premier cas : line
line = Line(np.array([0.,0.,0.]), np.array([0.,0.,ell]))
coil = pycoilib.coil.Coil([line], wire)
I.append(coil.get_inductance())

# Premier line : arc
for θ_i in θ[1:]:
    R = ell/θ_i
    arc = Arc.from_center(vec_0, R=R, arc_angle=θ_i, arc_rot=0)
    coil = pycoilib.coil.Coil([arc], wire)
    I.append(coil.get_inductance())
I = np.array(I)
plt.plot(θ, I, "+")
plt.show()

loop = Loop(R)
I_loop = coil.get_inductance()



fig = plt.figure(figsize=(7.5/2.54, 5.5/2.54))
ax = plt.gca()
ax.plot(ell*1e3,M1*1e9,"+")
ax.plot(ell*1e3,Mth*1e9,"--")
ax.legend(["Our approach", "Analytical"])
ax.set_xlabel(r"Line length [mm]")
ax.set_ylabel(r"Mutual inductance [nH]")

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.73, 0.14, r"$d = 5$ mm", transform=ax.transAxes, fontsize=8,
        verticalalignment='top', )
fig.tight_layout()
fig.savefig("paralel_Lines.png", dpi=300)
plt.show()