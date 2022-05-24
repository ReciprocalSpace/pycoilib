# -*- coding: utf-8 -*-
"""
Wire module


Created on Tue Jan 26 08:25:25 2021

@author: utric
"""

import numpy as np
from scipy.constants import mu_0 as μ0

from pycoilib.lib.misc.geometry import _vec_0, _vec_x, _vec_y, _vec_z
from ..inductance.inductance import calc_mutual
from ..segment.segment import Segment, ArcAbstract, Line, Circle, Arc


class Wire:
    """ Wire class, TODO description or delete"""
    def self_inductance(self, segment):
        """
        .. WARNING
            Not implemented
        """
        raise NotImplementedError()


class WireCircular(Wire):
    """WireCircular class, 
    
    :param float radius: TODO description or delete
    """
    def __init__(self, radius):
        """constructor"""
        #: TODO description or delete
        self.radius = radius

    def self_inductance(self, segment: Segment):
        """self_inductance method, TODO description or delete

        :param Segment segment: TODO description or delete
        """
        if isinstance(segment, Line):
            ell = segment.ell
            a = self.radius
            
            p0, p1 = _vec_0, np.array([ell, 0., 0.])
            p2, p3 = np.array([0., a, 0.]), np.array([ell, a, 0.])
            
            line1 = Line(p0, p1)
            line2 = Line(p2, p3)
            
            I, tmp = calc_mutual(line1, line2)
            
            return I
        
        elif isinstance(segment, ArcAbstract):
            R1 = segment.radius
            arc_angle = segment.theta
            R2 = R1 - self.radius
            # radius, arc_angle, pos, vec_x, vec_y, vec_z, current=1
            arc1 = Arc(R1, arc_angle, _vec_0, _vec_x, _vec_y, _vec_z)
            arc2 = Arc(R2, arc_angle, _vec_0, _vec_x, _vec_y, _vec_z)
            I, tmp = calc_mutual(arc1, arc2)
            return I


class WireRect(Wire):
    """WireRect class, TODO description or delete

    :param float width: TODO description or delete
    :param float thickness: TODO description or delete
    """
    def __init__(self, width, thickness=0):
        self.width = width
        self.thickness = thickness

    def self_inductance(self, shape):
        """self_inductance method, TODO description or delete

        :param TYPE shape: TODO description or delete + type
        """
        if isinstance(shape, Line):
            raise NotImplementedError()
                    
        elif isinstance(shape, Circle):
            # Grover1912, p. 116-117 eq. 70
            # Formule empirique de Reyleigh et Niven
            wt = self.width
            d = 2*shape.radius
            I = μ0 *d/2*( np.log(4*d/wt)-1/2+(wt/d)**2/24*(np.log(4*d/wt)+43/12))
            return I
            