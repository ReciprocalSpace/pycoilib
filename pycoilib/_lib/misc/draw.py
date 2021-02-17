# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 10:25:41 2021

@author: utric
"""


def draw(coil_or_shapes):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    coil_or_shape.draw()
    
    for shape in self.shape_array:
        shape.draw(ax)
    
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    plt.show()