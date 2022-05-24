# -*- coding: utf-8 -*-
"""
Segment module

Created on Tue Jan 26 08:25:11 2021

@author: Aimé Labbé
"""
from __future__ import annotations
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from abc import abstractmethod
# from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation

from pycoilib.lib.misc.set_axes_equal import set_axes_equal
from pycoilib.lib.misc.exceptions import PycoilibWrongShapeVector
import pycoilib.lib.misc.geometry as geo


class Segment:
    """Segment generic class

    .. note:: 
       This generic class is not available directly by the user.
       Arc, Circle and Line are to be used.
    
    :param numpy.ndarray vec_r0: Position vector of the segment, 1D of shape (3,)
    :param float current: Value of the current flowing in the Segment.

    """
    # __metaclass__ = ABCMeta

    VEC_X = np.array([1., 0., 0.])
    VEC_Y = np.array([0., 1., 0.])
    VEC_Z = np.array([0., 0., 1.])
    VEC_0 = np.array([0., 0., 0.])

    def __init__(self, position: np.ndarray, current: float):
        """Create an instance of a Segment object"""
        # Input shape validation
        if position.shape != (3,):
            raise PycoilibWrongShapeVector
        #:Position vector of the center of the arc.
        self.vec_r0 = position.copy()
        #:Current flowing in the arc.
        self.current = current

    def draw(self, ax=None, draw_current=True) -> None:
        """Plot the segment in 3-dimension

        This method draws the segment on an 3D-axis specified by ax, or a new figure otherwise. If the draw_current
        parameter is True, then the plot contains visual indicators of the the current orientation in the wire.

        :param Axes ax: Axis on which the segment is drawn. If no axis if provided, a new figure is created. Default is None.
        :param bool draw_current: Specify if indicators are added to the figure to display the current orientation in the wire. Default is True.

        """
        # TODO: add options for passing keyword arguments to pyplot

        create_fig = (ax is None)
        if create_fig:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

        x, y, z = self._coordinates2draw()
        x, y, z = x * 1e3, y * 1e3, z * 1e3  # Converting to mm

        ax.plot(x, y, z, "k", lw=1)
        if draw_current:
            c = np.linspace(0, 1, len(x))
            ax.scatter(x, y, z, c=c, cmap="plasma", marker="o")

        if create_fig:
            # It might be relevant to remove the following 4 lines from the if statement
            set_axes_equal(ax)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_zlabel("z [mm]")

            plt.show()

    def _coordinates2draw(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return a list of coordinates

        This method computes the coordinates of the segment in the laboratory frame for plotting.

        
        :returns: Tuple of three 1D
        :rtype: numpy.ndarray
        """
        raise NotImplementedError()

    def translate(self, translation: np.ndarray) -> Segment:
        """Translate the segment
        This method moves the segment to a position defined by its current position plus a translation.

        
        :param numpy.ndarray translation: Translation vector. 1D of shape (3,)
        :returns: itself
        """
        # Input segment validation
        if translation.shape != (3,):
            raise PycoilibWrongShapeVector

        self.vec_r0 = translation + self.vec_r0
        return self

    def move_to(self, new_position: np.ndarray):
        """Move the segment to a new position

        :param numpy.ndarray new_position: New position of the segment. 1D of shape (3,)
        :returns: itself
        """
        if new_position.shape != (3,):
            raise PycoilibWrongShapeVector

        self.vec_r0 = new_position

    def rotate(self, angle: float, axis: np.ndarray, origin: np.ndarray) -> Segment:
        """Rotate the segment
        
        """
        raise NotImplementedError()

    def _rotate(self, vec_to_rotate: List[np.ndarray], angle: float,
                axis: np.ndarray, origin: np.ndarray = None) -> None:
        """Rotate a list of vectors

        This method rotates a list of vector by a specific angle around an axis passing through an origin.

        :param List(numpy.ndarray) vec_to_rotate: List of vectors to rotate. List of 1D of shape (3,)
            
        :param float angle: Rotation angle, in rad.
        :param numpy.ndarray axis: Rotation axis vector. 1D of shape (3,)
        :param numpy.ndarray origin: Vector representing the reference origin used for the rotation. Default is centered on the segment position. 1D of shape (3,)
        """
        if origin is None:
            origin = self.vec_r0

        if axis.shape != (3,) or origin.shape != (3,):
            raise PycoilibWrongShapeVector

        axis = axis / np.sqrt(axis @ axis)  # Normalization

        rot = Rotation.from_rotvec(angle * axis).as_matrix()

        self.vec_r0 = rot @ (self.vec_r0 - origin) + origin

        # This code is a bit obscure : vec = self.a_ndarray
        # by doing vec[:] = ... the new values are stored in self.a_ndarray
        # this approach allows to encapsulate as many object properties in
        # to_rotate as necessary.
        for vec in vec_to_rotate:
            vec[:] = rot @ vec

    def flip_current_direction(self) -> Segment:
        """Inverse current orientation

        This method inverses the orientation of the current in the segment.


        :returns: itself
        """
        raise NotImplementedError()


    def get_endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return segment endpoint coordinates

        This method returns the beginning and the end points of the segment.

        :returns: tuple of two 1D array of shape (3,)
        :rtype: numpy.ndarray

        """
        raise NotImplementedError()


class ArcAbstract(Segment):
    """Superclass for arc and circle type segments

    .. note:: 
        This superclass is not available directly by the user.
        Arc, Circle and Line are to be used.
    
    :param float radius: Radius or the arc or circle segment.
    :param float arc_angle: Angle of the arc.
    :param numpy.ndarray position: Position vector of the center of the arc. 1D of shape (3,)
    :param numpy.ndarray vec_x: Unit vector corresponding to the x-axis in the arc referential. By construction, this vector also defines the beginning of the arc. 1D of shape (3,)
    :param numpy.ndarray vec_y: Unit vector corresponding to the y-axis in the arc referential. By construction, this vector also defines the orientation of the arc in the plane. 1D of shape (3,)
    :param numpy.ndarray vec_z: Unit vector corresponding to the z-axis in the arc referential. By construction, this vector is orthogonal to arc plane. 1D of shape (3,)
    :param float current: Current flowing in the arc. Must be positive.

    """
    def __init__(self, radius: float, arc_angle: float, position: np.ndarray,
                 vec_x: np.ndarray, vec_y: np.ndarray, vec_z: np.ndarray, current: float):
        """Create an ArcAbstract object."""
        super().__init__(position, current)

        # Input segment validation
        if any([vec.shape != (3,) for vec in [vec_x, vec_y, vec_z]]):
            raise PycoilibWrongShapeVector

        #: Radius or the arc or circle segment.
        self.radius = radius
        #: Angle of the arc.
        self.theta = arc_angle
        #:Unit vector corresponding to the x-axis in the arc referential. By construction, this vector also defines the beginning of the arc.
        self.vec_x = vec_x.copy()
        #: Unit vector corresponding to the y-axis in the arc referential. By construction, this vector also defines the orientation of the arc in the plane.
        self.vec_y = vec_y.copy()
        #: Unit vector corresponding to the z-axis in the arc referential. By construction, this vector is orthogonal to arc plane.
        self.vec_z = vec_z.copy()

    def get_vec_uvw(cls, arc_rot_angle: float, axis_rot_angle: float,
                    axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return an orthonormal basis (u,v,w) = R_axis(axis_rot_angle) R_z(arc_rot_angle) (x,y,z)

        This method transform the usual basis (x,y,z) in the arc segment reference frame into the laboratory frame
        (u,v,w). First, a rotation arc_rot_angle is applied around the z_axis (in the arc frame), and then a second
        rotation of axis_rot_angle around the specified axis is applied.

        :param float arc_rot_angle: Angular position of the arc around the z-axis, with the x-axis the origin.
        :param float axis_rot_angle: Rotation angle of the second operation around the axis.
        :param numpy.ndarray axis: Rotation axis for the second operation. 1D of shape (3,)
            
        :returns: Arc usual basis represented in the laboratory frame. 
        :rtype: Tuple of three 1D-numpy.ndarray of shape (3,)
        """

        if axis.shape != (3,):
            raise PycoilibWrongShapeVector

        # First rotation -> along z
        rot_arc = Rotation.from_rotvec(cls.VEC_Z * arc_rot_angle).as_matrix()

        # Second rotation -> along axis
        axis = axis / np.sqrt(axis @ axis)  # Normalization
        rot = Rotation.from_rotvec(axis * axis_rot_angle).as_matrix()

        vec_u = rot @ rot_arc @ cls.VEC_X
        vec_v = rot @ rot_arc @ cls.VEC_Y
        vec_w = rot @ cls.VEC_Z

        return vec_u, vec_v, vec_w

    def __str__(self):
        return (f"Segment : Arc\n"
                f"\tradius r:\t{self.radius:.3}\n"
                f"\tangle θ: \t{self.theta * 360 / 2 / np.pi:.1f}\n"
                f"\tposition:\t{self.vec_r0[0]},"
                f" {self.vec_r0[1]:.3}, {self.vec_r0[2]:.3}\n"
                f"\tvec x:\t{self.vec_x[0]:.3},"
                f" {self.vec_x[1]:.3}, {self.vec_x[2]:.3}\n"
                f"\tvec y:\t{self.vec_y[0]:.3}"
                f" {self.vec_y[1]:.3}, {self.vec_y[2]:.3}\n"
                f"\tvec z:\t{self.vec_z[0]:.3},"
                f" {self.vec_z[1]:.3}, {self.vec_z[2]:.3}")

    def _coordinates2draw(self):
        θ = np.linspace(0, self.theta, max(abs(int(self.theta * 360 / (2 * np.pi) / 5)) + 1, 5))
        R = self.radius
        coord = np.array([self.vec_r0
                          + R * self.vec_x * np.cos(θ_i)
                          + R * self.vec_y * np.sin(θ_i) for θ_i in θ])
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        return x, y, z

    def flip_current_direction(self):
        rot = Rotation.from_rotvec(self.theta * self.vec_z).as_matrix()
        self.vec_x = rot @ self.vec_x
        self.vec_y = -rot @ self.vec_y
        self.vec_z = -self.vec_z
        return self

    def get_endpoints(self):
        p0 = self.vec_r0 + self.radius * self.vec_x

        p1 = self.vec_r0 + self.radius * (self.vec_x * np.cos(self.theta)
                                          + self.vec_y * np.sin(self.theta))
        return p0, p1

    def rotate(self, angle: float, axis: np.ndarray, origin: np.ndarray = None):
        to_rotate = [self.vec_x, self.vec_y, self.vec_z]
        super()._rotate(to_rotate, angle, axis, origin)
        return self


class Arc(ArcAbstract):
    """Arc segment class

    This class defines several constructors for the arc segment class.

    """
    def __init__(self, radius, arc_angle, position: np.ndarray = None,
                 vec_x: np.ndarray = None, vec_y: np.ndarray = None, vec_z: np.ndarray = None, current=1.):
        position = self.VEC_0 if position is None else position
        vec_x = self.VEC_X if vec_x is None else vec_x
        vec_y = self.VEC_Y if vec_y is None else vec_y
        vec_z = self.VEC_Z if vec_z is None else vec_z

        super().__init__(radius, arc_angle, position, vec_x, vec_y, vec_z, current)

    def from_rot(cls, radius: float, arc_angle: float, arc_angular_pos: float, position: np.ndarray, axis: np.ndarray,
                 angle: float, current=1.) -> Arc:
        """Instantiate an arc in the xy-plane and rotate it around an axis

        
        :param float radius: Radius or the arc or circle segment.
        :param float arc_angle: Angle of the arc.
        :param float arc_angular_pos: Angular position of the arc around the z-axis, with the x-axis the origin.
        :param numpy.ndarray position: Position vector of the center of the arc. 1D of shape (3,)
        :param numpy.ndarray axis: Rotation axis for the arc. 1D of shape (3,)
        :param float angle: Rotation angle around the axis.
        :param float current: Current flowing in the arc. Must be positive.

        :returns: Arc
        :rtype: Arc
        """
        if any([vec.shape != (3,) for vec in [axis]]):
            raise PycoilibWrongShapeVector

        vec_u, vec_v, vec_w = cls.get_vec_uvw(arc_angular_pos, angle, axis)

        return cls(radius, arc_angle, position, vec_u, vec_v, vec_w, current)

    
    def from_normal(cls, radius: float, arc_angle: float, arc_angular_pos=0, position: np.ndarray = None,
                    normal: np.ndarray = None, current=1.) -> Arc:
        """Instantiate an arc from the orientation of its normal

        
        :param float radius: Radius or the arc or circle segment.
        :param float arc_angle: Angle of the arc.
        :param float arc_angular_pos: Angular position of the arc around the z-axis, with the x-axis the origin. Default is 0.
        :param numpy.ndarray position: Position vector of the center of the arc. Default is origin (0,0,0). 1D of shape (3,).
        :param numpy.ndarray normal: Orientation of the arc normal. Default is the z-axis (0,0,1). 1D of shape (3,).
        :param float current: Current flowing in the arc. Default is 1. Must be positive.

        :returns: Arc
        :rtype: Arc
        """

        position = cls.VEC_0 if position is None else position
        normal = cls.VEC_Z if normal is None else normal

        if any([vec.shape != (3,) for vec in [normal]]):
            raise PycoilibWrongShapeVector

        axis, angle = geo.get_rotation(cls.VEC_Z, normal)
        vec_u, vec_v, vec_w = cls.get_vec_uvw(arc_angular_pos, angle, axis)

        return cls(radius, arc_angle, position, vec_u, vec_v, vec_w, current)

    def from_endpoints(cls, p0: np.ndarray, p1: np.ndarray, arc_angle: float, normal: np.ndarray, current=1.) -> Arc:
        """Instantiate an arc from its endpoints and normal axis

        :param numpy.ndarray p0: Beginning of the arc. 1D of shape (3,)
        :param numpy.ndarray p1: End of the arc. 1D of shape (3,)
        :param float arc_angle: Angle of the arc.
        :param numpy.ndarray normal: Orientation of the arc normal. Default is the z-axis (0,0,1) .1D of shape (3,)
        :param float current: Current flowing in the arc. Default is 1. Must be povitive

        :returns: Arc
        :rtype: Arc
        """

        if any([vec.shape != (3,) for vec in [p0, p1, normal]]):
            raise PycoilibWrongShapeVector

        # Bisection
        vec_n = p1 - p0
        len_n = np.sqrt(vec_n @ vec_n)
        θ = arc_angle

        R = len_n / (2 * np.sin(θ / 2))

        vec_z = normal / np.sqrt(normal @ normal)

        vec_v = vec_n / np.sqrt(vec_n @ vec_n)
        vec_u = np.cross(vec_z, vec_v)

        vec_r0 = p0 + vec_n / 2 + R * vec_u * np.cos(θ / 2)

        vec_x = (p0 - vec_r0) / R
        vec_y = np.cross(vec_z, vec_x)
        return cls(R, arc_angle, vec_r0, vec_x, vec_y, vec_z, current)


class Circle(ArcAbstract):
    """Circle segment class

    This class defines several constructors for the circle segment class.
    """
    def __init__(self, radius: float, position: np.ndarray = None,
                 vec_x: np.ndarray = None, vec_y: np.ndarray = None, vec_z: np.ndarray = None,
                 current: float = 1.):
        position = self.VEC_0 if position is None else position
        vec_x = self.VEC_X if vec_x is None else vec_x
        vec_y = self.VEC_Y if vec_y is None else vec_y
        vec_z = self.VEC_Z if vec_z is None else vec_z

        super().__init__(radius, 2 * np.pi, position, vec_x, vec_y, vec_z, current)

    @classmethod
    def from_rot(cls, radius: float, position: np.ndarray, axis: np.ndarray, angle: float, current=1.) -> Circle:
        """Instantiate an arc in the xy-plane and rotate it around an axis

        :param float radius: Radius or the arc or circle segment.
        :param numpy.ndarray position: Position vector of the center of the arc. 1D of shape (3,)
        :param numpy.ndarray axis: Rotation axis for the arc. 1D of shape (3,)
        :param float angle: Rotation angle around the axis.
        :param float current: Current flowing in the arc.

        :returns: Circle
        :rtype: Circle
        """
        if axis.shape != (3,):
            raise PycoilibWrongShapeVector

        vec_u, vec_v, vec_w = Arc.get_vec_uvw(0., angle, axis)

        return cls(radius, position, vec_u, vec_v, vec_w, current)

    @classmethod
    def from_normal(cls, radius, position: np.ndarray = None, normal: np.ndarray = None, current=1.):
        """Instantiate a circle from the orientation of its normal

        :param float radius: Radius or the arc or circle segment.
        :param numpy.ndarray position: Position vector of the center of the arc. Default is origin (0,0,0). 1D of shape (3,)
        :param numpy.ndarray normal: Orientation of the arc normal. Default is the z-axis (0,0,1). 1D of shape (3,).
        :param float current: Current flowing in the arc. Default is 1.

        :returns: Circle
        :rtype: Circle
        """
        normal = cls.VEC_Z if normal is None else normal
        if normal.shape != (3,):
            raise PycoilibWrongShapeVector

        rot_axis, rot_angle = geo.get_rotation(cls.VEC_Z, normal)
        vec_x, vec_y, vec_z = Arc.get_vec_uvw(0., rot_angle, rot_axis)

        return cls(radius, position, vec_x, vec_y, vec_z, current)

    def __str__(self):
        return (f"Segment : Circle\n"
                f"\tradius r:\t{self.radius:11.3e}\n"
                f"\tposition:\t{self.vec_r0[0]:10.3e}, {self.vec_r0[1]:10.3e}, {self.vec_r0[2]:10.3e}\n"
                f"\tvec x:\t\t{self.vec_x[0]:10.3e}, {self.vec_x[1]:10.3e}, {self.vec_x[2]:10.3e}\n"
                f"\tvec y:\t\t{self.vec_y[0]:10.3e}, {self.vec_y[1]:10.3e}, {self.vec_y[2]:10.3e}\n"
                f"\tvec z:\t\t{self.vec_z[0]:10.3e}, {self.vec_z[1]:10.3e}, {self.vec_z[2]:10.3e}")


class Line(Segment):
    """Line segment class

    :param numpy.ndarray p0: Beginning of the line segment. 1D of shape (3,)
    :param numpy.ndarray p1: End of the line segment. 1D of shape (3,)
    :param float current: Current flowing in the arc. Default is 1. Must be positive
    
    """
    def __init__(self, p0: np.ndarray, p1: np.ndarray, current=1.):
        """Initialize a line segment from its endpoints
        """
        if p0.shape != (3,) or p1.shape != (3,):
            raise PycoilibWrongShapeVector

        super().__init__(p0, current)
        #: Length of the line object.
        self.ell = np.sqrt((p1 - p0) @ (p1 - p0))  
        #: Unit vector parallel to the orientation of the line object.
        self.vec_n = (p1 - p0) / self.ell

    def _coordinates2draw(self):
        coord = np.array([self.vec_r0 + alpha*self.vec_n for alpha in np.linspace(0, self.ell, 51)])
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        return x, y, z

    def __str__(self):
        return (f"Segment : Line\n"
                f"\tlength ell:\t\t{self.ell:8.3f}\n"
                f"\torientation:\t{self.vec_n[0]:8.3f}, {self.vec_n[1]:8.3f}, {self.vec_n[2]:8.3f}\n"
                f"\tposition:\t\t{self.vec_r0[0]:8.3f}, {self.vec_r0[1]:8.3f}, {self.vec_r0[2]:8.3f}")

    def rotate(self, angle: float, axis: np.ndarray, origin: np.ndarray = None):
        to_rotate = [self.vec_n]
        super()._rotate(to_rotate, angle, axis, origin)
        return self

    def flip_current_direction(self):
        self.vec_r0 = self.vec_r0 + self.ell * self.vec_n
        self.vec_n = -self.vec_n
        return self

    def get_endpoints(self):
        p0 = self.vec_r0
        p1 = self.vec_r0 + self.ell * self.vec_n
        return p0, p1
