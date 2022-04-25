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
from pycoilib.shape import Arc

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rc('lines', linewidth=2)
plt.rc('font', size=9)

vec_x=np.array([1.,0.,0.])
vec_y=np.array([0.,1.,0.])
vec_z=np.array([0.,0.,1.])
vec_0=np.array([0.,0.,0.])

R = 0.2
a = 0.005
n = 10

fig = plt.figure()
ax = plt.gca()

N = [100, 50, 25, 10, 5, 2]

wire = pycoilib.wire.WireCircular(a)

for n in N:
    θ = 2*π/n
    I = []
    X = []
    for m in range(1, n+1):
        arc = Arc(R=R, arc_angle=m*θ, arc_rot=0)
        coil = pycoilib.coil.Coil([arc], wire)
        I.append(coil.get_inductance())
        X.append( 360*m/n )
    plt.plot(X, I, "+")
plt.show()




