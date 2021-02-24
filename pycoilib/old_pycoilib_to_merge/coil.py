# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 20:55:17 2020

@author: utric
"""

import numpy as np
from numpy import pi as π, cos, sin
from numpy import ma # masked array
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.constants import mu_0 as μ0
from scipy.spatial.transform import Rotation
import sys

import magpylib as magpy
from magpylib.source import current

import pycoillib.geometry as geo


class Coil():
    def __init__(self, magpy_collection, center=(0,0,0), 
                 vmax='norm(self.getB(center))*1.4'):
        
        assert isinstance(magpy_collection, 
                          magpy._lib.classes.collection.Collection),(
                'magpy_collection: invalid type - magpylib Collection object',
                'expected')
        
        self.current_source_collection = magpy_collection
        
        self.center = center
        
        if vmax == 'norm(self.getB(center))*1.4':
            vmax = norm(self.getB(center))*1.4
        
        self.vmax = vmax*1000
        
    
    def getB(self, POS):
        return self.current_source_collection.getB(POS)
    
    def getL(self):
        print(f"getL() has not been implemented for coil of type {type(self)}")
        
    def quickBmap(self, 
                  planes="xyz", 
                  field="xyz",
                  points=101,
                  projectcoil=True,
                  showcontour=True,
                  vmin=0,
                  vmax='self.vmax',
                  cmap="viridis"):
        
        if vmax == 'self.vmax':
            vmax = self.vmax
        
        # --------------------------------------------------------------------
        # Input validation
        # planes
        assert len(planes) == len(set(planes)),(
                "planes: each char must appear at most one time - Ex: 'xyz' ")
        
        assert (set(planes)-set("xyz"))==set(),(
                "planes: enter a combination of x,y,z - Ex: 'xyz'")
        
        # fields 
        assert len(field) == len(set(field)),(
                "fields: each char must appear at most one time - Ex: 'xyz' ")
        
        assert (set(field)-set("xyz"))==set(),(
                "fields: enter a combination of x,y,z - Ex: 'xyz'")
        
        # --------------------------------------------------------------------
        
        fig, axes = plt.subplots(1, len(planes))
        if not isinstance(axes, np.ndarray):
            axes=[axes] 
        
        domain = self._getDomain()
        
        r0 = np.mean(domain, axis=1)
        dr = np.max( [domain[:,1]-domain[:,0]] )*1.5/2
    
        X0 = np.linspace(r0[0]-dr, r0[0]+dr, points)
        Y0 = np.linspace(r0[1]-dr, r0[1]+dr, points)
        Z0 = np.linspace(r0[2]-dr, r0[2]+dr, points)
        
        print(domain)
        print(f"r0: \t{r0}")
        print(f"dr: \t{dr:.2f}")
        print(f"X0: \t{X0.min():.2f},\t{X0.max():.2f}")
        print(f"Y0: \t{Y0.min():.2f},\t{Y0.max():.2f}")
        print(f"Z0: \t{Z0.min():.2f},\t{Z0.max():.2f}")
        
        for ax, letter in zip(axes, planes.lower()):
            X, Y, Z = X0, Y0, Z0
            if letter == "x":
                X = np.linspace(r0[0], r0[0]+ 1, 1)
                extent =[Z.min(), Z.max(), Y.min(), Y.max()]
                ax.set_xlabel("Z [mm]")
                ax.set_ylabel("Y [mm]")
            
            elif letter == "y":
                Y = np.linspace(r0[1], r0[1]+ 1, 1)
                extent =[X.min(), X.max(), Z.min(), Z.max()]
                ax.set_xlabel("X [mm]")
                ax.set_ylabel("Z [mm]")
            
            elif letter =="z":
                Z = np.linspace(r0[2], r0[2]+ 1, 1 )
                extent =[X.min(), X.max(), Y.min(), Y.max()]
                ax.set_xlabel("X [mm]")
                ax.set_ylabel("Y [mm]")
            
            POS = np.array( [[x, y, z] for x in X for y in Y for z in Z] )
            
            B = self.getB(POS)*1000 # mT -> uT
            
            B2plot = self._getB2plot(B, field).reshape(points, points)
            B2plot = B2plot.T if letter in "yz" else B2plot
            
            ax.imshow(B2plot, origin="lower", cmap=cmap, extent=extent, 
                      vmin=0, vmax=vmax)
            
            if projectcoil:
                for source in self.current_source_collection.sources:
                    x,y = self._getProjection(source, letter)
                    
                    ax.plot(x,y,"-", c="w",linewidth=2,alpha=0.7)
                    
            if showcontour and B2plot.max()>0:
                lnext = vmax/256
                
                lmax = vmax*1.5
                levels=[]
                while True:
                    levels.append(lnext)
                    lnext*=2
                    if lnext>lmax:
                        break
                levels = np.array(levels)
                
                ax.contour(np.flip(B2plot, axis=0), 
                           extent=extent, levels=levels,
                           vmin=-levels.max()*0.2, vmax=levels.max()*0.8,
                           origin='image'
                           )
        plt.show()
        
    def _getDomain(self):
        # Domain definition : coordinates that emcompass the coil
        domain = np.array([[float('inf'), -float('inf')],
                           [float('inf'), -float('inf')],
                           [float('inf'), -float('inf')]] )
        
        for source in self.current_source_collection.sources:
            if isinstance(source, current.Line):
                for i in range(3):  
                    ri = source.position[i] + source.vertices[:,i]
                    domain[i,0] = min(domain[i,0], np.min(ri))
                    domain[i,1] = max(domain[i,1], np.max(ri))
                    
            
            elif isinstance(source, current.Circular):
                C = source.position
                R = source.dimension/2
                ω = source.angle*π/180*source.axis
                n =  geo.z_vector @ Rotation.from_rotvec(ω).as_matrix().T
                pts = geo.circle_in_3D(C,R,n)
                
                for i in range(3):
                    ri = pts[:,i]
                    domain[i,0] = min(domain[i,0], np.min(ri))
                    domain[i,1] = max(domain[i,1], np.max(ri))
            else:
                import warnings
                warnings.warn(
                    "called _getProjection is not implemented in this current"
                    "source, returning np.zeros(3,2)", RuntimeWarning)
                return np.zeros(3,2)
            
        return domain
        
    def _getB2plot(self, B,field):
        field = field.lower()
        Bx = B[:,0] if "x" in field else 0
        By = B[:,1] if "y" in field else 0
        Bz = B[:,2] if "z" in field else 0
        
        return np.sqrt(Bx**2 + By**2 + Bz**2)
    
    def _getProjection(self, source, normal):
        
        if isinstance(source, current.Line):
            vertices = source.vertices
            
        elif isinstance(source, current.Circular):
            C = source.position
            R = source.dimension/2
            ω = source.angle*π/180*source.axis
            n =  geo.z_vector @ Rotation.from_rotvec(ω).as_matrix().T
            vertices = geo.circle_in_3D(C,R,n)
            
        else:
            import warnings
            warnings.warn(
                "called _getProjection is not implemented in this current"
                "source, returning ([],[])", RuntimeWarning)
            return [],[]
        
        if normal=='x':
            x = vertices[:,2]
            y = vertices[:,1]
            
        elif normal=='y':
            x = vertices[:,0]
            y = vertices[:,2]
            
        elif normal=='z':
            x = vertices[:,0]
            y = vertices[:,1]
            
        return x, y
        
                
# class Circular(Coil):
#     def __init__(self, 
#                  radius, 
#                  position=(0,0,0), 
#                  normal=(0,1,0)
#                  ):
#         angle, axis = geo.get_rotation(geo.z_vector, normal)
        
#         sources = [ current.Circular(curr=1, dim=2*radius, pos=position,
#                              angle=angle*180/π, axis=axis) ]
        
#         magpy_collection = magpy.Collection(sources)

#         POS = geo.circle_in_3D(position,radius*0.85,normal, npoints=1)
#         vmax = norm(magpy_collection.getB(POS))*1.1                
        
#         super().__init__(magpy_collection, position, vmax)
        

# class Solenoid(Coil):
#     def __init__(self,
#                  radius,
#                  length,
#                  nturns,
#                  position=(0,0,0),
#                  normal=(0,1,0),
#                  ):
#         angle, axis = geo.get_rotation(geo.z_vector, normal)
        
#         sources = []
        
#         Z = np.linspace(-length/2,length/2,nturns)
#         for zi in Z:
#             pos = np.array([0,0,zi])
#             sources.append( current.Circular(curr=1,dim=2*radius,pos=pos) )
            
#         magpy_collection = magpy.Collection(sources)
#         magpy_collection.rotate(angle*180/π, axis)
#         magpy_collection.move(position)
        
#         vmax = norm(magpy_collection.getB(position))*1.2
        
#         super().__init__(magpy_collection, position, vmax)
        
        
# class Polygon(Coil):
#     def __init__(self,
#                  poly,
#                  position=(0,0,0),
#                  normal=(0,0,1)
#                  ):
#         angle, axis = geo.get_rotation(geo.z_vector, normal)
        
#         source = [ current.Line(1, poly) ]
#         magpy_collection = magpy.Collection(source)
#         magpy_collection.rotate(angle, axis)
#         magpy_collection.move(position)
        
#         r = norm( np.std(poly, axis=0) ) / 2
#         I = poly.shape[0]-1 # Number of linear segments
#         vmax = μ0*I/(2*π*r) *1e6
        
#         super().__init__(magpy_collection, position, vmax)
        

# class Helmholtz(Coil):
#     def __init__(self,
#                  radius,
#                  position=(0,0,0),
#                  normal=(0,1,0),
#                  ):
        
#         angle, axis = geo.get_rotation(geo.z_vector, normal)
        
#         sources = []
#         sources.append( current.Circular(curr=1,dim=2*radius,pos=[0,0,-radius/2]) )
#         sources.append( current.Circular(curr=1,dim=2*radius,pos=[0,0, radius/2]) )
        
#         magpy_collection = magpy.Collection(sources)
#         magpy_collection.rotate(angle, axis)
#         magpy_collection.move(position)
        
#         vmax = norm(magpy_collection.getB(position))*1.2
        
#         super().__init__(magpy_collection, position, vmax)
        

class Birdcage(Coil):
    def __init__(self,
                 radius,
                 length,
                 nwires,
                 position=(0,0,0),
                 normal=(0,0,1)
                 ):
        
        θ_0 = 2*π/(nwires-1)/2
        Θ = np.linspace(θ_0, 2*π-θ_0, nwires) 
        segments_current = cos(Θ)
        segment = np.array( [[0,0,0], [0,0,length]] )
        position = np.array( [radius*cos(Θ), radius*sin(Θ), -length/2 ] )
        
        sources = []
        for curr, seg, pos in zip(segments_current, segment, position):
            sources.append( magpy.source.current.Line(curr, seg, pos) )
            
        
        ## If magpylib develops an arc segment -> uncomment the following code
        #integral_matrix = np.zeros((nwires,nwires))
        #for i, line in enumerate(integral_matrix.T):
        #    line[i:] = 1
        #arcs_currents = integral_matrix @ segments_current
        #arcs_currents -= np.sum(arcs_currents)
        #arcs_pos # to be implemeted
        #arcs_angle  # to be implemented
            
            
            
        magpy_collection = magpy.collection(sources)
        
        angle, axis = geo.get_rotation(geo.z_vector, normal)
        magpy_collection.rotate(angle*180/π, axis)
        magpy_collection.move(position)
        vmax = norm(magpy_collection.getB(position))*1.2
        super().__init__(magpy_collection, position, vmax)
        
class Saddlecoil(Coil):
    pass

class MTLR(Coil):
    pass


    
    
    


