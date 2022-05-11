# -*- coding: utf-8 -*-
"""
Coil module

Created on Tue Jan 26 08:31:05 2021

@author: Aimé Labbé
"""

from __future__ import annotations
from typing import List

import numpy as np
import os
import matplotlib.pyplot as plt

from ..segment.segment import Segment, Arc, Circle, Line
from ..wire.wire import Wire, WireRect, WireCircular
from ..inductance.inductance import calc_mutual
from ..misc.set_axes_equal import set_axes_equal
from ..misc import geometry as geo


class Coil:
    """General Coil object.

    A coil is defined as the combination of a segment array and a wire type.

    :param List[segment] segment_array: TODO description or delete
    :param function wire: TODO description or delete
    :param numpy.ndarray anchor: TODO description or delete
    """
    VEC_0 = np.array([0., 0., 0.])
    VEC_X = np.array([1., 0., 0.])
    VEC_Y = np.array([0., 1., 0.])
    VEC_Z = np.array([0., 0., 1.])
    def __init__(self, segment_array: List[Segment], wire=Wire(), anchor: np.ndarray = None):
        """the constructor"""
        self.segment_array = segment_array
        self.wire = wire
        self.anchor = self.VEC_0.copy() if anchor is None else anchor.copy()
    
     
    def from_magpylib(cls, magpy_object, wire=Wire(), anchor: np.ndarray = None):
        """Construct a coil from a collection of magpy sources and a Wire object

        .. WARNING::

            Not implemented
        """
        raise NotImplementedError

    def to_magpy(self):
        """Return a list of segments as collection of magpy sources

        .. WARNING::

            Not implemented
        """
        raise NotImplementedError
    
    def _magpy2pycoil(self, magpy_object):
        raise NotImplementedError
    
    def _pycoil2magpy(self, coil_array):
        raise NotImplementedError
    
    def move_to(self, new_position: np.ndarray) -> Coil:
        """Move the coil to a new position.

        :param numpy.ndarray new_position: TODO description or delete"""
        translation = new_position - self.anchor
        for segment in self.segment_array:
            segment.translate(translation)
        self.anchor = new_position.copy()
        return self
    
    def translate(self, translation: np.ndarray):
        """Translate the coil by a specific translation vector.

        :param numpy.ndarray translation: TODO description or delete"""
        for segment in self.segment_array:
            segment.translate(translation)
        self.anchor += translation
        return self

    def rotate(self, angle: float, axis: np.ndarray = None):
        """Rotate the coil around an axis by a specific angle.

        :param float angle: TODO description or delete
        :param numpy.ndarray axis: TODO description or delete"""
        axis = self.VEC_Z if axis is None else axis
        for segment in self.segment_array:
            segment.rotate(angle, axis, self.anchor)
        return self
    
    def draw(self, draw_current=True, savefig=False):
        """Draw the coil in a 3D plot.

        :param bool draw_current: TODO description or delete
        :param bool savefig: TODO description or delete"""
        fig = plt.figure(figsize=(7.5/2.4, 7.5/2.4), dpi=300,)
        ax = fig.add_subplot(111, projection='3d')
        
        for shape in self.segment_array:
            shape.draw(ax, draw_current)

        set_axes_equal(ax)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        
        if savefig:
            i = 0
            while True:
                path = "Fig_"+str(i)+".png"
                if os.path.exists(path):
                    i += 1
                else:
                    break
            plt.savefig(path, dpi=300, transparent=True)
        plt.show()

    def get_inductance(self):
        """Compute the coil self-inductance.

        :returns: self inductance
        :rtype: float"""
        inductance = 0
        n = len(self.segment_array)

        # Mutual between segment pairs
        for i, segment_i in enumerate(self.segment_array[:-1]):
            for j, segment_j in enumerate(self.segment_array[i + 1:]):
                res = calc_mutual(segment_i, segment_j)
                inductance += 2*res[0]

        # Self of segments
        for i, segment_i in enumerate(self.segment_array):
            res = self.wire.self_inductance(segment_i)

            inductance += res
        return inductance


class Loop(Coil):
    """Loop class TODO describe it

    :param float radius: TODO description or delete
    :param numpy.ndarray position: TODO description or delete
    :param numpy.ndarray axis: TODO description or delete
    :param float angle: TODO description or delete
    """
    def __init__(self, radius: float, position: np.ndarray = None, axis: np.ndarray = None, angle: float = 0.,
                 wire=Wire()):
        """The constructor"""
        position = self.VEC_0 if position is None else position
        axis = self.VEC_Z if axis is None else axis
        circle = Circle.from_rot(radius, position, axis, angle)

        super().__init__([circle], wire)
    
     
    def from_normal(cls, radius: float, position: np.ndarray = None, normal: np.ndarray = None, wire=Wire()):
        """ TODO describe methode
        
        :param float radius: TODO description or delete
        :param numpy.ndarray position: TODO description or delete
        :param numpy.ndarray normal: TODO description or delete
        :param function wire: TODO description or delete
        :returns: TODO description or delete
        :rtype: TODO description or delete
        """
        position = cls.VEC_0 if position is None else position
        normal = cls.VEC_Y if normal is None else normal

        axis, angle = geo.get_rotation(cls.VEC_Z, normal)
        return cls(radius, position, axis, angle, wire)


class Solenoid(Coil):
    """Solenoid class TODO describe it

    :param float radius: TODO description or delete
    :param float length: TODO description or delete
    :param int n_turns: TODO description or delete
    :param numpy.ndarray position: TODO description or delete
    :param numpy.ndarray axis: TODO description or delete
    :param float angle: TODO description or delete
    :param function wire: TODO description or delete
    """
    def __init__(self, radius: float, length: float, n_turns: int,
                 position: np.ndarray = None, axis: np.ndarray = None, angle: float = 0.,
                 wire=Wire()):
        """constructor"""
        segments = [Circle(radius, np.array([0., 0., z])) for z in np.linspace(-length/2, length/2, n_turns)]
        super().__init__(segments, wire)

        position = self.VEC_0 if position is None else position
        axis = self.VEC_Z if axis is None else axis
        self.move_to(position)
        self.rotate(axis, angle)

     
    def from_normal(cls, radius, length, n_turns, position, normal, wire=Wire()):
        """ TODO describe methode
        
        :param function cls: TODO description or delete
        :param float radius: TODO description or delete
        :param float length: TODO description or delete
        :param int n_turns: TODO description or delete
        :param numpy.ndarray position: TODO description or delete
        :param numpy.ndarray normal: TODO description or delete
        :param function wire: TODO description or delete
        :returns: TODO description or delete
        :rtype: TODO description or delete
        """
        axis, angle = geo.get_rotation(cls.VEC_Z, normal)
        return cls(radius, length, n_turns, position, axis, angle, wire)


class Polygon(Coil):
    """Polygon class TODO describe it

    :param TYPE polygon: TODO description or delete + type
    :param function wire: TODO description or delete
    """
    def __init__(self, polygon, wire):
        """the constructor"""
        lines = []
        for p0, p1 in zip(polygon[:-1], polygon[1:]):
            lines.append(Line(p0, p1))
        super().__init__(lines, wire)


class Helmholtz(Coil):
    """Helmholtz class TODO describe it

    :param float radius: TODO description or delete
    :param numpy.ndarray position: TODO description or delete
    :param numpy.ndarray axis: TODO description or delete
    :param float angle: TODO description or delete
    :param function wire: TODO description or delete
    """
    def __init__(self, radius: float, position: np.ndarray = None, axis: np.ndarray = None, angle:float = 0.,
                 wire=Wire()):

        segments = [Circle(radius, np.array([0, 0, -radius/2])),
                    Circle(radius, np.array([0, 0, radius/2]))]
        super().__init__(segments, wire)

        position = self.VEC_0 if position is None else position
        axis = self.VEC_Z if axis is None else axis
        self.move_to(position)
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
    """MTLR class TODO describe it

    :param float inner_radius: TODO description or delete
    :param float delta_radius: TODO description or delete
    :param float line_width: TODO description or delete
    :param int n_turns: TODO description or delete
    :param float dielectric_thickness: TODO description or delete
    :param numpy.ndarray anchor: TODO description or delete
    :param numpy.ndarray axis: TODO description or delete
    :param float angle: TODO description or delete
    """
    def __init__(self, inner_radius: float, delta_radius: float, line_width: float, n_turns,
                 dielectric_thickness: float,
                 anchor: np.ndarray = None, axis: np.ndarray = None, angle: float = 0.):
        """the constructor"""
        radii = np.array([inner_radius + n * delta_radius for n in range(n_turns)])

        segments = []
        for radius in radii:
            segments.append(Circle.from_normal(radius))
            segments.append(Circle.from_normal(radius, position=np.array([0., 0., -dielectric_thickness])))

        wire = WireRect(line_width, )
        super().__init__(segments, wire)

        if anchor:
            self.translate(anchor)

        if axis:
            self.rotate(angle, axis)

