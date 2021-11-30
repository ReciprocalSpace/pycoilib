# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:25:36 2021

@author: Aimé Labbé
"""

from pycoilib.segment import Arc, Circle, Line

from pycoilib.lib.inductance.calc_m_2lines import calc_M_2lines
from pycoilib.lib.inductance.calc_m_arcNline import calc_M_arcNline
from pycoilib.lib.inductance.calc_m_arcNloop import calc_M_arcNloop
from pycoilib.lib.inductance.calc_m_2arcs import calc_M_2arcs


def calc_mutual(shape1, shape2):
    if isinstance(shape1, Line):
        if isinstance(shape2, Line):
            return calc_M_2lines(shape1, shape2)
        if isinstance(shape2, Arc):
            return calc_M_arcNline(shape2, shape1)
    if isinstance(shape2, Line):
        if isinstance(shape1, Arc):
            return calc_M_arcNline(shape1, shape2)
    if isinstance(shape1, Circle):
        if isinstance(shape2, Arc):
            return calc_M_arcNloop(shape2, shape1)
    if isinstance(shape2, Circle):
        if isinstance(shape1, Arc):
            return calc_M_arcNloop(shape1, shape2)
    if isinstance(shape1, Arc):
        if isinstance(shape2, Arc):
            return calc_M_2arcs(shape1, shape2)
    raise NotImplementedError()
