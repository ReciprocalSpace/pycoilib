# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:25:11 2021

@author: utric
"""

from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import cos, sin, arctan2 as atan, sqrt, pi as π, sign, log
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
#from scipy.integrate import quad
#from scipy.special import ellipk as ellK,  ellipe as ellE
#from scipy.special import ellipkinc as ellK_inc,  ellipeinc as ellE_inc
#from scipy.constants import mu_0 as μ0

from pycoilib._lib.misc._set_axes_equal import _set_axes_equal
import pycoilib._lib.misc.geometry as geo

plt.rc('xtick',labelsize=8)
plt.rc('ytick',labelsize=8)
plt.rc('lines', linewidth=2)
plt.rc('font', size=9)

_vec_x=np.array([1.,0.,0.])
_vec_y=np.array([0.,1.,0.])
_vec_z=np.array([0.,0.,1.])
_vec_0=np.array([0.,0.,0.])


class Segment():
    __metaclass__ = ABCMeta
    
    def __init__(self, pos, current):
        self.r0 = np.sqrt(pos@pos)
        self.vec_r0 = pos
        self.current = current
        
        
    def draw(self, ax=None, draw_current=True, transform=None):
        create_fig = (ax==None)
        if create_fig:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        
        x, y, z = self._coordinates2draw()
        x, y, z= x*1e3, y*1e3, z*1e3
        
        ax.plot(x,y,z, "k", lw=1)
        if draw_current:
            ax.plot(x[0],y[0],z[0], "bo", alpha=0.4, lw=2)
            ax.plot(x[-1],y[-1],z[-1], "ro",alpha=0.4, lw=2)
            
        if create_fig:
            _set_axes_equal(ax)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_zlabel("z [mm]")
            
            plt.show()
        
    @abstractmethod
    def _coordinates2draw(self):
        raise NotImplementedError()
    
    @abstractmethod
    def __str__(self):
        raise NotImplementedError()
    
    @abstractmethod
    def rotate(self, axis, angle, origin):
        raise NotImplementedError()
    
    def _rotate(self, axis, angle, origin, to_rotate):
        if origin=="self.vec_r0":
            origin = self.vec_r0
        rot = Rotation.from_rotvec(angle*axis).as_matrix()
        
        self.vec_r0 = rot @ (self.vec_r0-origin) + origin
        self.r0 = sqrt( self.vec_r0 @ self.vec_r0 )
        
        #This code is obscure : vec = self.a_ndarray
        #by doing vec[:] = ... the new values are stored in self.a_ndarray
        #this approach allows to encapsulate as many object properties in
        # to_rotate as necessary.
        for vec in to_rotate:
            vec[:] = rot @ vec
    
    @abstractmethod
    def flip_current_direction(self):
        raise NotImplementedError()
    
    
    def translate(self, translation):
        self.vec_r0 = translation - self.vec_r0
        self.r0 = np.sqrt(self.vec_r0 @ self.vec_r0)
        return self
    
    @abstractmethod
    def get_endpoints(self):
        raise NotImplementedError()

class Arc(Segment):
    def __init__(self, radius, arc_angle, pos, vec_x, vec_y, vec_z, current=1):
        super().__init__(pos, current)
        self.radius = radius
        self.theta = arc_angle
        self.vec_x = vec_x.copy()
        self.vec_y = vec_y.copy()
        self.vec_z = vec_z.copy()
    
    @staticmethod
    def get_vec_uvw(arc_rot, axis, angle):
        rot_arc = Rotation.from_rotvec(_vec_z*arc_rot).as_matrix()
        
        axis = axis/sqrt(axis@axis) # Normalization
        rot = Rotation.from_rotvec(axis*angle).as_matrix()
        
        vec_u = rot @ rot_arc @ _vec_x
        vec_v = rot @ rot_arc @ _vec_y
        vec_w = rot @ _vec_z
        
        return vec_u, vec_v, vec_w
    
    @classmethod
    def from_rot(cls, radius, arc_angle, center, arc_rot=0, 
                 axis=_vec_z, angle=0, current=1):
        vec_u, vec_v, vec_w = cls.get_vec_uvw(arc_rot, axis, angle)

        return cls(radius, arc_angle, center, vec_u, vec_v, vec_w, current)
    
    @classmethod
    def from_normal(cls, radius, arc_angle, center, arc_rot=0, normal=_vec_y,current=1):
        axis, angle = geo.get_rotation(_vec_z, normal)
        vec_u, vec_v, vec_w = cls.get_vec_uvw(arc_rot, axis, angle)
        
        return cls(radius, arc_angle, center, vec_u, vec_v, vec_w, current)
    
    @classmethod   
    def from_endpoints(cls, p0, p1, arc_angle, normal, current=1):
        # Bisection
        vec_n = p1 - p0
        len_n = sqrt(vec_n @ vec_n)
        θ = arc_angle
        
        R = len_n / ( 2*sin(θ/2) )
        
        vec_z = normal / sqrt(normal @ normal)
        
        vec_v = vec_n / sqrt( vec_n @ vec_n )
        vec_u = np.cross(vec_z, vec_v)
        
        vec_r0 = p0 + vec_n/2 + R*vec_u*cos(θ/2)
        
        vec_x = (p0-vec_r0)/R
        vec_y = np.cross(vec_z, vec_x)
        return cls(R, arc_angle, vec_r0, vec_x, vec_y, vec_z, current)
        
    def __str__(self):
        return (f"Segment : Arc\n"
                f"\tRad. r:\t{self.radius:.3}\n"
                f"\tangle θ:\t{self.theta*360/2/π:.1f}\n"
                f"\tpos:\t\t{self.vec_r0[0]},"
                f" {self.vec_r0[1]:.3}, {self.vec_r0[2]:.3}\n"
                f"\tvec x:\t{self.vec_x[0]:.3},"
                f" {self.vec_x[1]:.3}, {self.vec_x[2]:.3}\n"
                f"\tvec y:\t{self.vec_y[0]:.3}"
                f" {self.vec_y[1]:.3}, {self.vec_y[2]:.3}\n"
                f"\tvec z:\t{self.vec_z[0]:.3},"
                f" {self.vec_z[1]:.3}, {self.vec_z[2]:.3}")
    
    def _coordinates2draw(self):
        θ = np.linspace(0, self.theta, max( abs(int(self.theta*360/(2*π)/5))+1, 5))
        R = self.radius
        coord = np.array( [self.vec_r0 
                            + R*self.vec_x*cos(θ_i)
                            + R*self.vec_y*sin(θ_i) for θ_i in θ ]) 
        x, y, z = coord[:,0], coord[:,1], coord[:,2] 
        return x, y, z
    
    def flip_current_direction(self):
        rot = Rotation.from_rotvec(self.theta*self.vec_z).as_matrix()
        self.vec_x = rot @ self.vec_x
        self.vec_y =-rot @ self.vec_y
        self.vec_z = -self.vec_z
        return self
    
    def get_endpoints(self):
        p0 = self.vec_r0 + self.radius*self.vec_x
        
        p1 = self.vec_r0 + self.radius*( self.vec_x*cos(self.theta)
                               +self.vec_y*sin(self.theta))
        return p0, p1
    
    def rotate(self, axis, angle, origin="self.vec_r0"):
        to_rotate = [self.vec_x, self.vec_y, self.vec_z]
        super()._rotate(axis, angle, origin, to_rotate)
        return self
    
    
class Loop(Arc):
    def __init__(self, R, pos=_vec_0, axis=_vec_z, angle=0, current=1):
        vec_u, vec_v, vec_w = Arc.get_vec_uvw(0, axis, angle)
       
        super().__init__( R, 2*π, pos, vec_u, vec_v, vec_w, current)
    
    @classmethod
    def from_normal(cls, R, pos=_vec_0, vec_z=_vec_z, current=1):
        rot_axis, rot_angle = geo.get_rotation(_vec_z, vec_z)
        return cls(R, pos, rot_axis, rot_angle, current)
        
    def __str__(self):
            return (f"Segment : Loop\n"
                    f"\tradius r:\t{self.radius:.3e}\n"
                    f"\tposition:\t{self.vec_r0[0]},"
                    f" {self.vec_r0[1]}, {self.vec_r0[2]}\n"
                    f"\tvec x:\t{self.vec_x[0]},"
                    f" {self.vec_x[1]}, {self.vec_x[2]}\n"
                    f"\tvec y:\t{self.vec_y[0]},"
                    f" {self.vec_y[1]}, {self.vec_y[2]}\n"
                    f"\tvec z:\t{self.vec_z[0]},"
                    f" {self.vec_z[1]}, {self.vec_z[2]}")

class Line(Segment):
    def __init__(self, p0, p1, current=1):
        super().__init__(p0, current)
        self.ell = sqrt( (p1-p0)@(p1-p0) )
        self.vec_n = (p1-p0)/self.ell
        
    def _coordinates2draw(self):
        coord = np.array([self.vec_r0, self.vec_r0+self.ell*self.vec_n]) 
        x, y, z = coord[:,0], coord[:,1], coord[:,2] 
        return x, y, z
        
    def __str__(self):
            return (f"Segment : Line\n"
                    f"\tlength ell:\t{self.ell}\n"
                    f"\torientation:\t{self.vec_n[0]},"
                    f" {self.vec_n[1]}, {self.vec_n[2]}\n"
                    f"\tposition:\t{self.vec_r0[0]},"
                    f" {self.vec_r0[1]}, {self.vec_r0[2]}")
        
    def rotate(self, axis, angle, origin="self.vec_r0"):
        to_rotate = [self.vec_n]    
        super()._rotate(axis, angle, origin, to_rotate)
        return self
        
    def flip_current_direction(self):
        self.vec_r0 = self.vec_r0 + self.ell * self.vec_n
        self.vec_n = -self.vec_n
        return self
    
    def get_endpoints(self):
        p0 = self.r0
        p1 = self.r0 + self.ell*self.vec_n 
        return p0, p1

