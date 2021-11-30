# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 20:20:39 2021

@author: Aime Labbe
"""

import numpy as np
from numpy import pi as π, cos, sin, arctanh, arccos, tan, sqrt, arctan
from scipy.constants import mu_0 as μ0
import pycoilib as pycoil 
import pycoilib.lib.misc.geometry as geo
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rc('lines', linewidth=2)
plt.rc('font', size=9)


def calc_M_general(line1, line2):
    l = line1.ell
    m = line2.ell

    vec_p = line1.vec_n
    vec_s = line2.vec_n
    vec_n = geo.normalize( np.cross(vec_p, vec_s) )
    
    p0 = line1.vec_r0
    p2 = line2.vec_r0
    #p0 - α*vec_p + d*vec_n + β*vec_s = p2
    np.array([-vec_p, vec_n, vec_s])
    αdβ = np.linalg.solve(np.array([-vec_p, -vec_n,  vec_s]).T, (p2-p0) )
    α,d,β = tuple( αdβ )
    
    θ = arccos(vec_p@vec_s)

    R1 = sqrt( d**2 + (α+l)**2 + (β+m)**2 - 2*(α+l)*(β+m)*cos(θ) )
    R2 = sqrt( d**2 + (α+l)**2 + β**2 - 2*(α+l)*β*cos(θ) )
    R3 = sqrt( d**2 + α**2 + β**2 - 2*α*β*cos(θ) )
    R4 = sqrt( d**2 + α**2 + (β+m)**2 - 2*α*(β+m)*cos(θ) )
    
    Ω = ( arctan( (d**2*cos(θ) + (α+l)*(β+m)*sin(θ)**2) / (d*R1*sin(θ)) ) 
         -arctan( (d**2*cos(θ) + (α+l)* β*sin(θ)**2) / (d*R2*sin(θ)) )
         +arctan( (d**2*cos(θ) + α*β*sin(θ)**2) / (d*R3*sin(θ)) )
         -arctan( (d**2*cos(θ) + α*(β+m)*sin(θ)**2) / (d*R4*sin(θ)) ) 
         )
    
    Mp  = (μ0/(2*π)*cos(θ)*( (α+l)*arctanh( m/(R1+R2) ) 
                            +(β+m)*arctanh( l/(R1+R4) )
                            -α*arctanh( m/(R3+R4) )
                            -β*arctanh( l/(R2+R3) ) )
           - μ0/(4*π)*Ω*d/tan(θ) )
    return Mp

p0, p1 = np.array([0., 0., 0.]), np.array([1., 0., 0.])
#p2, p3 = np.array([0.25, 0.3 , 0.]), np.array([1.25, 0.4, 0.4])

points = geo.fibonacci_sphere(32)

Mth = []
Mour = []
line1 = pycoil.segment.Line(p0, p1)
for p in points:
    for q in points:
        p2 = p*0.5
        p3 = p2 + q*2.
        line2 = pycoil.segment.Line(p2,p3)

        #pycoil.coil.Coil([line1,line2]).draw()
        Mth.append( calc_M_general(line1,line2) )
        m, tmp= pycoil.calc_mutual(line1, line2)
        Mour.append(m)





Mth = np.array(Mth)
Mour = np.array(Mour)

# Affichage
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig = plt.figure(figsize=(7.5/2.54, 5.5/2.54))
ax = plt.gca()

xy = [Mour.min()*1e9, Mour.max()*1e9 ] # droite y=x
print(xy)
ax.plot(xy,xy, alpha=0.8, c=colors[1])

ax.plot(Mth*1e9, Mour*1e9, 'o', c=colors[0],  markersize=2) #data

ax.legend(["y=x"])
ax.set_xlabel(r"Mutual (our approach) [nH]")
ax.set_ylabel(r"Mutual (analytical) [nH]")


fig.tight_layout()
fig.savefig("test-2lines-general.png", dpi=300)
plt.show()

