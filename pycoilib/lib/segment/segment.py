# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:25:11 2021

@author: Aimé Labbé
"""
from __future__ import annotations
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from abc import ABCMeta, abstractmethod
from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation

from pycoilib.lib.misc._set_axes_equal import _set_axes_equal
from pycoilib.lib.misc.exceptions import PycoilibWrongShapeVector
import pycoilib.lib.misc.geometry as geo


_vec_x = np.array([1., 0., 0.])
_vec_y = np.array([0., 1., 0.])
_vec_z = np.array([0., 0., 1.])
_vec_0 = np.array([0., 0., 0.])


class Segment:
    """Segment generic class

    Attributes
    ----------
    r0 : float
        Length of the position vector of the segment.
    vec_r0 : 1D numpy.ndarray of shape (3,)
        Position vector of the segment
    current : float
        Value of the current flowing in the Segment.

    Methods
    -------
    draw(ax, draw_current):
        Plot the segment in 3-dimension
    translate(translation):
        Translate the shape
    rotate(angle, axis, origin):
        Rotate the shape
    flip_current_direction():
        Inverse current orientation
    get_endpoints():
        Return shape endpoint coordinates

    Other methods
    -------------
    _coordinates2draw():
        Return a list of coordinates
    _rotate(vec_to_rotate, angle, axis, origin):
        Rotate a list of vectors
    """
    __metaclass__ = ABCMeta

    def __init__(self, position: np.ndarray, current: float):
        """Create an instance of a Shape object

        Parameters
        ----------
        position: 1D numpy.ndarray of shape (3,)
            Position at which the shape is anchored. Specific to each shape child-class.
        current: float
            Value of the current in flowing in the shape.
        """
        if position.shape != (3,):
            raise PycoilibWrongShapeVector
        self.r0 = np.sqrt(position @ position)
        self.vec_r0 = position
        self.current = current

    def draw(self, ax=None, draw_current=True) -> None:
        """Plot the segment in 3-dimension

        This method draws the segment on an 3D-axis specified by ax, or a new figure otherwise. If the draw_current
        parameter is True, then the plot contains visual indicators of the the current orientation in the wire.

        Parameters
        ----------
        ax: Axes, optional
            Axis on which the segment is drawn. If no axis if provided, a new figure is created. Default is None.
        draw_current: bool, optional
            Specify if indicators are added to the figure to display the current orientation in the wire. Default is
            True.

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
            # TODO : display the current with more points. Use continuous color gradient ?
            ax.plot(x[0], y[0], z[0], "bo", alpha=0.4, lw=2)
            ax.plot(x[-1], y[-1], z[-1], "ro", alpha=0.4, lw=2)

        if create_fig:
            # It might be relevant to remove the following 4 lines from the if statement
            _set_axes_equal(ax)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_zlabel("z [mm]")

            plt.show()

    @abstractmethod
    def _coordinates2draw(self):
        """Return a list of coordinates"""
        raise NotImplementedError()

    def translate(self, translation: np.ndarray) -> Segment:
        """Translate the shape
        This method moves the shape to a position defined by its current position plus a translation.

        Parameters
        ----------
        translation: 1D numpy.ndarray of shape (3,)
            Translation vector.

        Returns
        -------
        itself
        """
        if translation.shape != (3,):
            raise PycoilibWrongShapeVector

        self.vec_r0 = translation + self.vec_r0
        self.r0 = np.sqrt(self.vec_r0 @ self.vec_r0)
        return self

    @abstractmethod
    def rotate(self, angle: float, axis: np.ndarray, origin: np.ndarray):
        """Rotate the shape"""
        raise NotImplementedError()

    def _rotate(self, vec_to_rotate: List[np.ndarray], angle: float, axis: np.ndarray, origin: np.ndarray = None) -> None:
        """Rotate a list of vectors

        This method rotates a list of vector by a specific angle around an axis passing through an origin.

        Parameters
        ----------
        vec_to_rotate: List of 1D-numpy.ndarray of shape (3,)
            List of vectors to rotate.
        angle: float
            Rotation angle, in rad.
        axis: 1D-numpy.ndarray of shape (3,)
            Rotation axis vector.
        origin: 1D-numpy.ndarray of shape (3,), optional
            Vector representing the reference origin used for the rotation. Default is centered on the shape position.
        Returns
        -------
        None

        """
        if origin is None:
            origin = self.vec_r0

        if axis.shape != (3,) or origin.shape != (3,) or any([vec_i.shape != (3,) for vec_i in vec_to_rotate]):
            raise PycoilibWrongShapeVector

        rot = Rotation.from_rotvec(angle * axis).as_matrix()

        self.vec_r0 = rot @ (self.vec_r0 - origin) + origin
        self.r0 = np.sqrt(self.vec_r0 @ self.vec_r0)

        # This code is a bit obscure : vec = self.a_ndarray
        # by doing vec[:] = ... the new values are stored in self.a_ndarray
        # this approach allows to encapsulate as many object properties in
        # to_rotate as necessary.
        for vec in vec_to_rotate:
            vec[:] = rot @ vec

    @abstractmethod
    def flip_current_direction(self):
        """Inverse current orientation"""
        raise NotImplementedError()

    @abstractmethod
    def get_endpoints(self):
        """Return shape endpoint coordinates"""
        raise NotImplementedError()


class ArcAbstract(Segment):
    def __init__(self, radius, arc_angle, pos, vec_x, vec_y, vec_z, current=1.):
        super().__init__(pos, current)
        self.radius = radius
        self.theta = arc_angle
        self.vec_x = vec_x.copy()
        self.vec_y = vec_y.copy()
        self.vec_z = vec_z.copy()

    @staticmethod
    def get_vec_uvw(arc_rot, axis, angle):
        rot_arc = Rotation.from_rotvec(_vec_z * arc_rot).as_matrix()

        axis = axis / np.sqrt(axis @ axis)  # Normalization
        rot = Rotation.from_rotvec(axis * angle).as_matrix()

        vec_u = rot @ rot_arc @ _vec_x
        vec_v = rot @ rot_arc @ _vec_y
        vec_w = rot @ _vec_z

        return vec_u, vec_v, vec_w

    def __str__(self):
        return (f"Segment : Arc\n"
                f"\tRad. r:\t{self.radius:.3}\n"
                f"\tangle θ:\t{self.theta * 360 / 2 / np.pi:.1f}\n"
                f"\tpos:\t\t{self.vec_r0[0]},"
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

    def rotate(self, angle, axis, origin):
        to_rotate = [self.vec_x, self.vec_y, self.vec_z]
        super()._rotate(to_rotate, angle, axis, origin)
        return self


class Arc(ArcAbstract):
    def __init__(self, radius, arc_angle, pos, vec_x, vec_y, vec_z, current=1.):
        super().__init__(radius, arc_angle, pos, vec_x, vec_y, vec_z, current)

    @classmethod
    def from_rot(cls, radius, arc_angle, center, arc_rot=0,
                 axis=_vec_z, angle=0, current=1):
        vec_u, vec_v, vec_w = cls.get_vec_uvw(arc_rot, axis, angle)

        return cls(radius, arc_angle, center, vec_u, vec_v, vec_w, current)

    @classmethod
    def from_normal(cls, radius, arc_angle, center, arc_rot=0, normal=_vec_y, current=1.):
        axis, angle = geo.get_rotation(_vec_z, normal)
        vec_u, vec_v, vec_w = cls.get_vec_uvw(arc_rot, axis, angle)

        return cls(radius, arc_angle, center, vec_u, vec_v, vec_w, current)

    @classmethod
    def from_endpoints(cls, p0, p1, arc_angle, normal, current=1.):
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


class Loop(ArcAbstract):
    def __init__(self, radius, pos=_vec_0, axis=_vec_z, angle=0., current=1.):
        vec_u, vec_v, vec_w = Arc.get_vec_uvw(0, axis, angle)

        super().__init__(radius, 2 * np.pi, pos, vec_u, vec_v, vec_w, current)

    @classmethod
    def from_normal(cls, radius, pos=_vec_0, vec_z=_vec_z, current=1.):
        rot_axis, rot_angle = geo.get_rotation(_vec_z, vec_z)
        return cls(radius, pos, rot_axis, rot_angle, current)

    def __str__(self):
        return (f"Segment : Loop\n"
                f"\tRadius r:\t{self.radius:11.3e}\n"
                f"\tPosition:\t{self.vec_r0[0]:10.3e}, {self.vec_r0[1]:10.3e}, {self.vec_r0[2]:10.3e}\n"
                f"\tvec x:\t\t{self.vec_x[0]:10.3e}, {self.vec_x[1]:10.3e}, {self.vec_x[2]:10.3e}\n"
                f"\tvec y:\t\t{self.vec_y[0]:10.3e}, {self.vec_y[1]:10.3e}, {self.vec_y[2]:10.3e}\n"
                f"\tvec z:\t\t{self.vec_z[0]:10.3e}, {self.vec_z[1]:10.3e}, {self.vec_z[2]:10.3e}")


class Line(Segment):
    def __init__(self, p0: np.ndarray, p1: np.ndarray, current=1.):
        super().__init__(p0, current)
        self.ell = np.sqrt((p1 - p0) @ (p1 - p0))  # ell: length
        self.vec_n = (p1 - p0) / self.ell

    def _coordinates2draw(self):
        coord = np.array([self.vec_r0, self.vec_r0 + self.ell * self.vec_n])
        x, y, z = coord[:, 0], coord[:, 1], coord[:, 2]
        return x, y, z

    def __str__(self):
        return (f"Segment : Line\n"
                f"\tLength ell:\t\t{self.ell:8.3f}\n"
                f"\tOrientation:\t{self.vec_n[0]:8.3f}, {self.vec_n[1]:8.3f}, {self.vec_n[2]:8.3f}\n"
                f"\tPosition:\t\t{self.vec_r0[0]:8.3f}, {self.vec_r0[1]:8.3f}, {self.vec_r0[2]:8.3f}")

    def rotate(self, angle: float, axis: np.ndarray, origin):
        to_rotate = [self.vec_n]
        super()._rotate(to_rotate, angle, axis, origin)
        return self

    def flip_current_direction(self):
        self.vec_r0 = self.vec_r0 + self.ell * self.vec_n
        self.vec_n = -self.vec_n
        return self

    def get_endpoints(self):
        p0 = self.vec_r0
        p1 = self.r0 + self.ell * self.vec_n
        return p0, p1
