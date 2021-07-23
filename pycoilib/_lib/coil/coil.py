# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:31:05 2021

@author: utric
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi as π, sqrt, cos, sin

from scipy.spatial.transform import Rotation

from pycoilib import segment
from pycoilib.wire import Wire
from pycoilib import calc_M

from pycoilib._lib.misc._set_axes_equal import _set_axes_equal
from pycoilib._lib.misc import geometry as geo
from pycoilib._lib.misc.geometry import _vec_0, _vec_x, _vec_y, _vec_z

class Coil():
    def __init__(self, shape_array, wire=Wire()):
        self.shape_array = shape_array
        self.wire = wire
        
        self.anchor = _vec_0.copy()
        
        # self.pos = pos.copy()
        # self.anchor = _vec_0.copy()
        # self.rotation= Rotation.from_rotvec(geo.normalize(axis)*angle)
    
    @classmethod
    def from_magpylib(cls, magpy_object, wire=Wire(), anchor="_vec_0"):
        return
    
    def _magpy2pycoil(self, magpy_object):
        pass
    
    def _pycoil2magpy(self, coil_array):
        pass
    
    def move_to(self, new_pos):
        # self.pos = new_pos.copy()
        return self
    
    def translate(self, translation):
        # self.pos += translation
        return self
    
    def rotate(self, axis, angle, anchor="self.anchor"):
        anchor = self.anchor if self.anchor=='self.anchor' else anchor
        
        # Rotation.from_rotvec(geo.normalize(axis)*angle)
        # new_rot = Rotation.from_rotvec(geo.normalize(axis)*angle)
        
        # self.rotation = new_rot @ self.rotation
        
        return self
        
    def reset_rotation(self):
        # self.rotation = np.eye(3)
        return self
    
    def draw(self, draw_current=True, savefig=False):            
        fig = plt.figure(figsize=(7.5/2.4, 7.5/2.4), dpi=300,)
        ax = fig.add_subplot(111, projection='3d')
        
        for shape in self.shape_array:
            shape.draw(ax, draw_current)
        
        _set_axes_equal(ax)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        
        if savefig:
            i=0
            while True:
                path="Fig_"+str(i)+".png"
                if os.path.exists(path):
                    i+=1
                else:
                    break
            plt.savefig(path, dpi=300, transparent=True)
        plt.show()
    
    def calc_I(self):
        I = 0
        n = len(self.shape_array)
        for i in range(n-1):
            
            for j in range(i+1,n):
                tmp = calc_M(self.shape_array[i], self.shape_array[j])
                I += tmp[0]
                if np.isnan(tmp[0]):
                    print(i,j)
                    print(self.shape_array[i])
                    print(self.shape_array[j])
        for i in range(n):
            res = self.wire.self_inductance(self.shape_array[i]) 
            I += res
        return I
    
class Loop(Coil):
    def __init__(self, radius, pos=_vec_0, axis=_vec_z, angle=0, wire=Wire() ):
        loop = segment.Loop(radius,pos, axis,angle)
        
        return super().__init__([loop], wire)
    
    @classmethod
    def from_normal(cls, radius, pos=_vec_0, normal=_vec_y, wire=Wire()):
        axis, angle = geo.get_rotation(geo.z_vector, normal)
        
        return cls(radius, pos, axis, angle, wire)
    
class Solenoid(Coil):
    def __init__(self, radius, length, nturns, 
                 pos=_vec_0, axis=_vec_z, angle = 0,
                 wire=Wire()):
        
        segments = []
        
        Z = np.linspace(-length/2,length/2,nturns)
        for zi in Z:
            pos = np.array([0,0,zi])
            segments.append( segment.Loop(radius, pos) )
            
        super().__init__(segments, wire)
        self.rotate(axis, angle)
        
    
    @classmethod
    def from_normal(cls, radius, length, nturns, 
                    pos=_vec_0, normal=_vec_z, wire=Wire()):
        
        axis, angle = geo.get_rotation(geo.z_vector, normal)
        
        return cls(radius, length, nturns, pos, axis, angle, wire)
    
class Polygon(Coil):
    def __init__(self, polygon, wire):
        lines = []
        for p0, p1 in zip(polygon[:-1],polygon[1:]):
            lines.append( segment.Line(p0, p1) )
        
        super().__init__(lines, wire)


class Helmholtz(Coil):
    def __init__(self, radius, position=_vec_0, axis=_vec_z, angle=0, wire=Wire() ):
        
        #axis, angle = geo.get_rotation(_vec_z, normal)
        
        segments = []
        
        segments.append( segment.Loop(radius, np.array([0,0,-radius/2])) )
        segments.append( segment.Loop(radius, np.array([0,0, radius/2])) )
        
        super().__init__(segments, wire)
        
        self.rotate(axis, angle)


# class Birdcage(Coil):
#     def __init__(self,
#                  radius, length, nwires, position=_vec_0, axis=_vec_z, angle=0,
#                  wire=Wire() ):
        
#         segments = []
        
#         θ_0 = 2*π/(nwires-1)/2 # Angular position of the first wire
#         Θ = np.linspace(θ_0, 2*π-θ_0, nwires) # Vector of angular positions
        
#         # Linear segments
#         p0, p1 = _vec_0, np.array([0,0,length] )
#         positions = np.array( [radius*cos(Θ), radius*sin(Θ), -length/2 ] )
#         currents = cos(Θ) # Current in each segment
        
#         for curr, pos in zip(currents, positions):
#             segments.append( segment.Line(p0+pos, p1+pos, curr))
        
#         # Arc segments
#         integral_matrix = np.zeros( (nwires, nwires) )
        
#         for i, line in enumerate(integral_matrix.T):
#             line[i:] = 1
#         currents = integral_matrix @ segments_current
#         currents -= np.sum(arcs_currents)
        
        
#         #arcs_pos # to be implemeted
#         #arcs_angle  # to be implemented
            
            
            
#         magpy_collection = magpy.collection(sources)
        
#         angle, axis = geo.get_rotation(geo.z_vector, normal)
#         magpy_collection.rotate(angle*180/π, axis)
#         magpy_collection.move(position)
#         vmax = norm(magpy_collection.getB(position))*1.2
#         super().__init__(magpy_collection, position, vmax)



class MTLR(Coil):
    def __init__(self, Rext, Rint, Nturns, thickness, 
                 position=_vec_0, axis=_vec_z, angle=0, wire=Wire()):
        #self.N = Ntours
        #self.espace = espace # Distance entre deux cercles concentriques
        self.width = width # Largeur piste supraconductrice
        self.thickness = thickness # Epaisseur du substrat
        self.R = np.array( [Rext-width/2 - n*(width+espace) for n in range(Ntours)] )
        self.εr = εr # Constante dielectrique relative du matériaux 
        self.ell = 2*π*np.sum(self.R)
        self.Cth = (0,0,-thickness)
        self._L = None
        
        segments = []
