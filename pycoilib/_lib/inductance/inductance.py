# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:25:36 2021

@author: utric
"""

from pycoilib.segment import Arc, Loop, Line

from pycoilib._lib.inductance.calc_m_2lines import calc_M_2lines
from pycoilib._lib.inductance.calc_m_arcNline import calc_M_arcNline
from pycoilib._lib.inductance.calc_m_arcNloop import calc_M_arcNloop
from pycoilib._lib.inductance.calc_m_2arcs import calc_M_2arcs

def calc_M(shape1, shape2):
    if isinstance(shape1, Line):
        if isinstance(shape2, Line):
            return calc_M_2lines(shape1,shape2)
        if isinstance(shape2, Arc):
            return calc_M_arcNline(shape2,shape1)
    if isinstance(shape2, Line):
        if isinstance(shape1, Arc):
            return calc_M_arcNline(shape1,shape2)
    if isinstance(shape1, Loop):
        if isinstance(shape2,Arc):
            return calc_M_arcNloop(shape2,shape1)
    if isinstance(shape2, Loop):
        if isinstance(shape1, Arc):
            return calc_M_arcNloop(shape1,shape2)
    if isinstance(shape1, Arc):
        if isinstance(shape2,Arc):
            return calc_M_2arcs(shape1,shape2)
    raise NotImplementedError()
