"""
Created on Tue Jan 26 08:27:50 2021

@author: utric
"""

import numpy as np
from numpy import pi as π
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.constants import mu_0 as μ0

import pycoilib as pycoil
from pycoilib._set_axes_equal import _set_axes_equal
from pycoilib.shape import Arc, Line, Loop

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rc('lines', linewidth=2)
plt.rc('font', size=9)

_vec_0 = np.array([0.,0.,0.])
_vec_x = np.array([1.,0.,0.])
_vec_y = np.array([0.,1.,0.])
_vec_z = np.array([0.,0.,1.])

import random

R_p = 1

M_bab = []
M_our = []


def rand(n=1):
    return np.array([random.random() for i in range(n)])

def random_vec_on_sphere(n=1):
    φ = rand(n) *2*π
    θ = np.arccos( rand(n)*2-1 )
    
    n_vec = np.array([ [np.cos(θ_i)*np.sin(φ_i),
                        np.sin(θ_i)*np.sin(φ_i),
                        np.cos(φ_i)] for θ_i, φ_i in zip(θ, φ) ] )
    return n_vec


print("---------------------------------------------------------------------")
print("----1\t- CODE VALIDATION : ArcNloops")
print("----1.1\t- Complete randomness for secondary")#   1.1       
if False:    
    random.seed(1)    
    Rmax = 10
    n = 1000
    
    R_s = rand(n)*(10-0.1)*R_p + 0.1*R_p
    vec_n = random_vec_on_sphere(n)
    
    s0 = np.cbrt(rand(n))*Rmax
    
    vec_s0_u=random_vec_on_sphere(n)
    vec_s0 = np.array( [ s0_u_i*s0_i for s0_u_i, s0_i in zip(vec_s0_u, s0)] )

    loop1 = Loop(R_p)
    for R_si, vec_ni, vec_s0i in zip(R_s, vec_n, vec_s0):
        loop2 = Loop.from_normal(R_si, vec_s0i, vec_ni)
        
        m_our, err = pycoil.inductance.calc_M(loop1, loop2) 
        m_bab, err = pycoil.inductance.calc_M_Babic(R_p, R_si, vec_s0i, vec_ni)
        
        M_our.append(m_our)
        M_bab.append(m_bab)
        
    M_our = np.array(M_our)
    M_bab = np.array(M_bab) 
    
    # affichage
    fig = plt.figure(figsize=(7.5/2.54, 5.5/2.54))
    ax = plt.gca()
    
    
    xy = [min(M_our.min(), M_bab.min()), max(M_our.max(), M_bab.max()) ]
    
    ax.plot(xy,xy, alpha=0.8, c="g")
    ax.plot(M_our,M_bab, '+',c="b",  markersize=2)
    ax.legend(["y=x"])

    # ax.text(0.02, 0.99, r"$d = 50$ mm", transform=ax.transAxes, fontsize=8,
    #         verticalalignment='top', )
    fig.tight_layout()
    fig.savefig("BabicVSus.png", dpi=300)
    plt.show()



    plt.show()
    
    # plt.plot(M_our-M_bab,"*",markersize=2)

    
    # ax.set_xlabel(r"Inductance (our approach) [mm]")
    # ax.set_ylabel(r"Mutual inductance [μH]")
    # plt.show()
    
    # plt.hist(M_our-M_bab, bins=50)
    # plt.plot()
    
## Well-behaved cases 
## Coaxial coils - distance appart
if False:
    M_bab = []
    M_our = []
    loop1 = Loop(R_p)
    for z0 in np.logspace(-2, 1, 1000):
        R_s = 1
        vec_s0 = np.array([0., 0., z0])
        
        loop2 = Loop(R_s,vec_s0)
            
        m_our, err = pycoil.inductance.calc_M(loop1, loop2) 
        m_bab, err = pycoil.inductance.calc_M_Babic(R_p, R_s, vec_s0)
    
        M_our.append(m_our)
        M_bab.append(m_bab)
        
    M_our = np.array(M_our)
    M_bab = np.array(M_bab) 
        
    # affichage
    plt.plot(M_our,M_bab, '*', markersize=2)
    xy = [min(M_our.min(), M_bab.min()), max(M_our.max(), M_bab.max()) ]
    plt.plot(xy,xy)
    plt.title("Moving z")
    plt.show()
    
    plt.plot(M_our-M_bab,"*",markersize=2)
    plt.title("Moving z")
    plt.show()
    
    # plt.hist(M_our-M_bab, bins=50)
    # plt.title("Moving z")
    # plt.plot()
    
## Translation along x
if False:
    loop1 = Loop(R_p)
    z0 = 0.5
    R_s = 0.4
    M_bab = []
    M_our = []
    for x0 in np.logspace(-2, 1.2, 1001):
        vec_s0 = np.array([x0, 0., z0])        
        loop2 = Loop(R_s, vec_s0)
            
        m_our, err = pycoil.inductance.calc_M(loop1, loop2) 
        m_bab, err = pycoil.inductance.calc_M_Babic(R_p, R_s, vec_s0)
    
        M_our.append(m_our)
        M_bab.append(m_bab)
        
    M_our = np.array(M_our)
    M_bab = np.array(M_bab)
    
    
    # affichage
    plt.plot(M_our)
    plt.plot(M_bab)
    plt.legend(["Our","Babic"])
    plt.title("Moving x")
    plt.show()
    
    plt.plot(M_our,M_bab, '*', markersize=2)
    xy = [min(M_our.min(), M_bab.min()), max(M_our.max(), M_bab.max()) ]
    plt.plot(xy,xy)
    plt.show()
    
    plt.plot(M_our-M_bab,"*",markersize=2)
    plt.title("Moving x")
    plt.show()
    
    # plt.hist(M_our-M_bab, bins=50)
    # plt.plot()
    
## Random normal axis    
if False:
    n=201
    loop1 = Loop(R_p)
    z0 = 0.5
    y0 = 0.
    x0 = 0.5
    R_s = 0.4
    φ = np.arccos( np.linspace(-1,1,n) )
    θ = np.linspace(0,2*π*5, n)
    M_bab = []
    M_our = []
    VEC= []
    for φ_i, θ_i in zip(φ, θ):
        vec_n = np.array([np.cos(θ_i)*np.sin(φ_i),
                        np.sin(θ_i)*np.sin(φ_i),
                        np.cos(φ_i)])
        VEC.append(vec_n)
        
        vec_s0 = np.array([x0, y0, z0])
        
        loop2 = Loop.from_normal(R_s, vec_s0, vec_n)
            
        m_our, err = pycoil.inductance.calc_M(loop1, loop2) 
        m_bab, err = pycoil.inductance.calc_M_Babic(R_p, R_s, vec_s0,vec_n)
        if np.isnan(m_bab):
            break
        
        M_our.append(m_our)
        M_bab.append(m_bab)
        
    M_our = np.array(M_our)
    M_bab = np.array(M_bab)
        
    VEC = np.array(VEC)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x, y, z = VEC[:,0], VEC[:,1], VEC[:,2]
        
    ax.plot(x,y,z, "b")
    _set_axes_equal(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.title("Rotating n")
    plt.show()
    
    # affichage
    plt.plot(M_our)
    plt.plot(M_bab)
    plt.legend(["Our","Babic"])
    plt.title("Rotating n")
    plt.show()
    
    # affichage
    plt.plot(M_our,M_bab, '*', markersize=2)
    plt.title("Rotating n")
    xy = [min(M_our.min(), M_bab.min()), max(M_our.max(), M_bab.max()) ]
    plt.plot(xy,xy)
    plt.show()
    
    plt.plot(M_our-M_bab,"*",markersize=2)
    plt.title("Rotating n")
    plt.show()
    
    plt.hist(M_our-M_bab, bins=50)
    plt.title("Rotating n")
    plt.plot()
    
    
print("---------------------------------------------------------------------")
print("----1\t- CODE VALIDATION : ArcNloops")
print("----1.1\t- Complete randomness for secondary")#   1.1    

# Init 
R_p =1
Rmax = 10
n = 251

# Propriétés aléatoires du secondaire
random.seed(1)   
R_s = rand(n)*(10-0.1)*R_p + 0.1*R_p
vec_z = random_vec_on_sphere(n)
s0 = np.cbrt(rand(n))*Rmax

vec_s0_u=random_vec_on_sphere(n)
vec_s0 = np.array( [ s0_u_i*s0_i for s0_u_i, s0_i in zip(vec_s0_u, s0)] )

arc_angle = rand(n)*2*π

vec_t = random_vec_on_sphere(n)

vec_x = np.zeros_like(vec_t)
for i, (ti, zi) in enumerate(zip(vec_t, vec_z)):
    tmp = ti-(ti@zi)*zi
    vec_x[i] = tmp/np.sqrt(tmp@tmp)


vec_y = np.cross(vec_z, vec_x)
# Début calcul
M_arcs = []
M_loop = []

arc1 = Arc(_vec_0, R_p, 2*π, _vec_x, _vec_y, _vec_z)
loop = Loop(R_p)

for R_si, vec_zi, vec_s0i, vec_yi, vec_xi, arc_angle_i in zip(R_s, vec_z, vec_s0, vec_y, vec_x, arc_angle):
    arc2 = Arc(vec_s0i, R_si,arc_angle_i,vec_xi, vec_yi, vec_zi)

    m_arcs, err = pycoil.inductance.calc_M_2arcs(arc2, arc1) 
    m_loop, err = pycoil.inductance.calc_M_arcNloop(arc2, loop)

    M_arcs.append(m_arcs)
    M_loop.append(m_loop)

M_arcs = np.array(M_arcs)
M_loop = np.array(M_loop) 
    
# Affichage
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

fig = plt.figure(figsize=(7.5/2.54, 5.5/2.54))
ax = plt.gca()

xy = [min(M_loop.min(), M_arcs.min())*1e6, max(M_arcs.max(), M_loop.max())*1e6 ] # droite y=x
ax.plot(xy,xy, alpha=0.8, c=colors[1])

ax.plot(M_arcs*1e6, M_loop*1e6, 'o', c=colors[0],  markersize=2) #data
ax.legend(["y=x"])
ax.set_xlabel(r"Mutual (our approach) [μH]")
ax.set_ylabel(r"Mutual (Babic 2010) [μH]")

fig.tight_layout()
#fig.savefig("BabicVSus.png", dpi=300)
plt.show()

plt.plot(M_arcs-M_loop)
plt.show()