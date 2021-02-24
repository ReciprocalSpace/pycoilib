# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:25:25 2021

@author: utric
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin, arctan2 as atan, sqrt, pi as π, sign, log
from scipy.constants import mu_0 as μ0

from pycoilib.segment import Arc, Loop, Line

from pycoilib._lib.misc.geometry import _vec_0, _vec_x, _vec_y, _vec_z

from pycoilib import calc_M




class Wire():
    def __init__(self):
        pass
        #self.arc_res = arc_res
        #self._arc_self_inductance_dic = {} # key :(radius, n)
    
    def self_inductance(self, shape):
        raise NotImplementedError()
        
class Wire_circ(Wire):
    def __init__(self, radius, arc_res=100):
        self.radius = radius
        #super().__init__(arc_res)
 
    def self_inductance(self, shape):
        if isinstance(shape, Line):
            ell=shape.ell 
            a = self.radius
            
            p0, p1 = _vec_0, np.array([ell,0., 0.])
            p2, p3 = np.array([0., a, 0.]), np.array([ell, a, 0.])
            
            line1 = Line(p0,p1)
            line2 = Line(p2,p3)
            
            I, tmp = calc_M(line1, line2)
            
            return I
            
            #return μ0*ell/(2*π)*(log(2*ell/a)-1)
        
        # elif isinstance(shape, Arc):
        #     R1 = shape.R
        #     a2 = self.radius
            
            
        #     return μ0*R*((1+3/16*a**2/R**2)*np.log(8*R/a)-a**2/(16*R**2)-2)
        
        elif isinstance(shape, Arc):
            R1 = shape.radius
            arc_angle = shape.theta
            R2 = R1 - self.radius
            #radius, arc_angle, pos, vec_x, vec_y, vec_z, current=1
            arc1 = Arc(R1, arc_angle, _vec_0, _vec_x, _vec_y, _vec_z )
            arc2 = Arc(R2, arc_angle, _vec_0, _vec_x, _vec_y, _vec_z )
            I, tmp = calc_M(arc1, arc2)
            
            return I
            # Old version : complicated
            # R = shape.R 
            # a = self.radius
            
            # m, n = diophantine_approx(shape.theta/(2*π), self.arc_res )
            
            # if (R,a,n) in self._arc_self_inductance_dic:
            #     M = self._arc_self_inductance_dic[(R,a,n)]
            # else: # (R,n) not in M_dic:
            #     M=[]
            #     θ1 = 2*π/n
            #     arc1 = Arc.from_center(_vec_0,R=R, arc_angle=θ1, arc_rot=0)
            #     for i in range(1, n):
            #         arc2 = Arc.from_center(_vec_0, R=R, arc_angle=θ1, arc_rot=θ1*i)
            #         tmp = calc_M_2arcs(arc1,arc2)
            #         M.append(tmp[0])
            #     M = np.array(M)
            #     self._arc_self_inductance_dic[(R,a,n)] = M
            
            # I = μ0*R*((1+3/16*a**2/R**2)*np.log(8*R/a)-a**2/(16*R**2)-2)
            
            # if (n%2)==0:
            #     k = int(n/2)-1
            #     I_n = I/n - 2*np.sum(M[:k]) - 2*1/2*M[k]
            # else:
            #     k = int(n/2)
            #     I_n = I/n - 2*np.sum(M[:k])

            # I_nm = m*I_n

            # for i in range(m-1):
            #     I_nm += 2*(m-i-1) * M[i]
            # return I_nm
        
class Wire_rect(Wire):
    def __init__(self, width, thickness, arc_res=100):
        self.width = width
        self.thickness = thickness
        super().__init__(arc_res)