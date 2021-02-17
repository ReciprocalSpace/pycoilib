# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:31:05 2021

@author: utric
"""

import os
import matplotlib.pyplot as plt
import numpy as np

from pycoilib.wire import Wire
from pycoilib import calc_M

from pycoilib._lib.misc._set_axes_equal import _set_axes_equal

class Coil():
    def __init__(self, shape_array, wire=Wire()):
        self.shape_array = shape_array
        self.wire = wire
    
    def draw(self, draw_current=True, savefig=False):            
        fig = plt.figure(figsize=(7.5/2.4, 7.5/2.4))
        ax = fig.add_subplot(111, projection='3d')
        
        for shape in self.shape_array:
            shape.draw(ax, draw_current)
        
        _set_axes_equal(ax)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        
        if savefig:
            i=0
            while True:
                path="Fig_"+str(i)+".png"
                if os.path.exists(path):
                    i+=1
                else:
                    break
            plt.savefig(path, dpi=300, transparent=True)
        plt.show()
        
    def calc_I(self):
        I = 0
        n = len(self.shape_array)
        for i in range(n-1):
            
            for j in range(i+1,n):
                tmp = calc_M(self.shape_array[i], self.shape_array[j])
                I += tmp[0]
                if np.isnan(tmp[0]):
                    print(i,j)
                    print(self.shape_array[i])
                    print(self.shape_array[j])
        for i in range(n):
            res = self.wire.self_inductance(self.shape_array[i]) 
            I += res
        return I