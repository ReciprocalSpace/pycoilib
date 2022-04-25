# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:25:36 2021

@author: Aimé Labbé
"""

from typing import Union, List
from .inductance import calc_mutual
from ..segment.segment import Segment
from ..coil.coil import Coil


def get_mutual(element1: Union[Segment, Coil, List[Segment]], element2: Union[Segment, Coil, List[Segment]]) -> float:
    """Compute the mutual inductance between two a pair of objects containing segments.

    Parameters
    ----------
    element1: segment, coil, or list of segments
    element2: segment, coil, or list of segments

    Returns
    -------
    mutual-inductance: float
        Computed mutual inductance between the two elements.
    """
    segments = []
    for el in [element1, element2]:
        if isinstance(el, Segment):
            segments.append([el])
        elif isinstance(el, Coil):
            segments.append(el.segment_array)
        elif isinstance(el, list):
            segments.append(el)

    s1, s2 = tuple(segments)

    mutual = 0
    for s1_i in s1:
        for s2_j in s2:
            m, _ = calc_mutual(s1_i, s2_j)
            mutual += m

    return mutual
