# -*- coding: utf-8 -*-


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


if "M_dic" not in locals():
    print(True)
    M_dic = {}

def calc_I_arc(R, a, n, m):
    θ1 = 2*π/n
    arc1 = Arc(R=R, arc_angle=θ1, arc_rot=0)
    order = n+1
    if n not in M_dic:
        MM=[]
        for i in range(1,order+1):
                arc2 = Arc(R=R, arc_angle=θ1, arc_rot=θ1*i)
                tmp = calc_M_2arcs(arc1,arc2)
                MM.append(tmp[0])
        MM = np.array(MM)
        M_dic[n]=MM
    else:
        MM=M_dic[n]
    
    I = μ0*R*((1+3/16*a**2/R**2)*np.log(8*R/a)-a**2/(16*R**2)-2)
    θ1 = 2*π/n
    if (n%2)==0:
        k = int(n/2)-1
        I_n = I/n - 2*np.sum(MM[:k]) - 2*1/2*MM[k]
    else:
        k = int(n/2)
        I_n = I/n - 2*np.sum(MM[:k])
    
    I_nm = m*I_n
    for i in range(m-1):
        I_nm += 2*(m-i-1) * MM[i]
    return I_nm



#fig = plt.figure(figsize=(7.5/2.54, 5.5/2.54))
fig = plt.figure()
ax = plt.gca()

nn = [100,50,25, 10, 5, 2]
MM = []
mmm= []
for n in nn:
    I = μ0*R*((1+3/16*a**2/R**2)*np.log(8*R/a)-a**2/(16*R**2)-2)
    mm = np.arange(0,n+1)
    mmm.append(mm)
    M=[]
    for m in mm:
        M.append(calc_I_arc(R, a, n, m))
    M = np.array(M)
    MM.append(M)
    
    ax.plot(mm*2/n ,M,".")

ax.plot(mm*2/n, I*mm**0 )

ax.set_xlabel(r"Line length [mm]")
ax.set_ylabel(r"Mutual inductance [nH]")


fig.tight_layout()
fig.savefig("Self-arc.png", dpi=300)
plt.show()

####################################################

        
        



