# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:25:36 2021

@author: Aimé Labbé
"""

from ..segment.segment import ArcAbstract as Arc, Circle, Line, Segment

from .calc_m_2arcs import calc_M_2arcs
from .calc_m_2lines import calc_M_2lines
from .calc_m_arcNline import calc_M_arc_line
from .calc_m_arcNcircle import calc_M_arcNcircle

from typing import Tuple


def calc_mutual(segment1: Segment, segment2: Segment) -> Tuple[float, float]:
    """
    Computes the mutual intuctance between a pair of segments.

    This method checks the types of the two input segments and selects the appropriate function.

    Parameters
    ----------
    segment1: Segment
    segment2: Segment

    Returns
    -------
    inductance: float
    uncertainty: float
    """

    if isinstance(segment1, Line):
        if isinstance(segment2, Line):
            return calc_M_2lines(segment1, segment2)
        if isinstance(segment2, Arc):
            return calc_M_arc_line(segment2, segment1)
    if isinstance(segment2, Line):
        if isinstance(segment1, Arc):
            return calc_M_arc_line(segment1, segment2)
    if isinstance(segment1, Circle):
        if isinstance(segment2, Arc):
            return calc_M_arcNcircle(segment2, segment1)
    if isinstance(segment2, Circle):
        if isinstance(segment1, Arc):
            return calc_M_arcNcircle(segment1, segment2)
    if isinstance(segment1, Arc):
        if isinstance(segment2, Arc):
            return calc_M_2arcs(segment1, segment2)
    else:
        error_message = f"calc_M() not implemented for pair of segments of "\
                        f"type(segment1) {type(segment1)} and type(segment2) {type(segment2)}."
        raise NotImplementedError(error_message)

