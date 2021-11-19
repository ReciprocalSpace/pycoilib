# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:36:20 2021

@author: Aime Labbe
"""



import numpy as np
from numpy import pi as π, cos, sin, arctanh, arccos, tan, sqrt, arctan
from scipy.constants import mu_0 as μ0
import pycoilib as pycoil 
import pycoilib.lib.misc.geometry as geo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation

# fibonacci_spiral_sphere

points = geo.fibonacci_sphere(1000)
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')
ax.plot(points[:,0],points[:,1], points[:,2], "o", ms=1)
plt.show()

