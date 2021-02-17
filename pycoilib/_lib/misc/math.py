# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 16:11:25 2021

@author: utric
"""
import numpy as np
# from numpy import floor
# from numpy import cos, sin, arctan2 as atan, sqrt, pi as π, sign, log
# from scipy.special import ellipk as ellK,  ellipe as ellE
# from scipy.special import ellipkinc as ellK_inc,  ellipeinc as ellE_inc


def diophantine_approx(v, Q):
    Qinv = 1. / Q
    un_moins_Qinv = 1 - Qinv
    for q in range(1, Q+1):
        q_v = q*v
        frac_q_v =  q_v - np.floor(q_v)
        if (frac_q_v <= Qinv or un_moins_Qinv <= frac_q_v):
            p =  round(q_v)
            return p, q
    else:
        raise RuntimeError('Did not find diophantine approximation of vector '
                'v={} with parameter Q={}'.format(v, Q))


