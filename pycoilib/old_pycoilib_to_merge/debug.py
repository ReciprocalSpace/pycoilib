# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 11:05:04 2020

@author: utric
"""

import numpy as np
import pycoillib as pycoil

#polygon = np.array( [[0,0,0], [0.5,0,0.5], [0,0,1]] )
#poly = pycoil.coil.Polygon(polygon)
#poly.quickBmap(field="xyz",planes="xyz")

#circular = pycoil.coil.Circular(radius=10,position=(5,1,6), normal=(1,1,1))
#circular.quickBmap(field="xyz",planes="xyz")

#solenoid = pycoil.coil.Solenoid(radius=5.1,length=10,nturns= 10, 
#                                position=(0,0,0), normal=(0,0,1))
#solenoid.quickBmap(field="xyz",planes="xyz")



helmholtz = pycoil.coil.Helmholtz(radius=15, position=(0,0,0), normal=(0,1,0))
helmholtz.quickBmap(field="xyz",planes="xyz")