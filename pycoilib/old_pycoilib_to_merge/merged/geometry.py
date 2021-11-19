# # -*- coding: utf-8 -*-
# """
# Created on Tue Dec 15 15:28:30 2020
#
# @author: utric
# """
#
# import numpy as np
# from numpy import sqrt, arctan2, pi as π, cos, sin
# from scipy.spatial.transform import Rotation
# import sys
#
# _0vector = np.array([0.,0.,0.])
# x_vector = np.array([1.,0.,0.])
# y_vector = np.array([0.,1.,0.])
# z_vector = np.array([0.,0.,1.])
#
# def length(wire_or_vector):
#     """Return the length of a wire of a vector.
#
#     Parameters
#     ----------
#     wire_or_vector : ndarray 1xN or 2xN
#         Array containing the position of one (vector) of two points (wire) in
#         a N dimensional space.
#
#     Raises
#     ------
#     ValueError
#         Input was of dimension other than 1 or 2.
#
#     Returns
#     -------
#     float
#         Computed length or the wire of vector.
#
#     """
#     if wire_or_vector.ndim == 1:
#         # A vector
#         wire = np.array( [wire_or_vector, _0vector])
#     elif wire_or_vector.ndim == 2:
#         wire = wire_or_vector
#     else:
#         d = wire_or_vector.dim
#         raise ValueError("Wrong number of dimensions for input argument",
#                          "'wire_or_vector'. One or two was expected,",
#                          f"but input had {d}.")
#     return sqrt( (wire[1]-wire[0])@(wire[1]-wire[0]) )
#
# def translate(wire_or_vector, r0):
#     """Translate a vector or a list of vectors by r0.
#
#     Translation operation on a single vector (1xN) or a list of vectors (mxN)
#     in a N dimensional space.
#
#     Parameters
#     ----------
#     wire_or_vector : ndarray
#         Vector or list of vectors.
#     r0 : vector
#         Translation vector.
#
#     Raises
#     ------
#     ValueError
#         Input was of dimension other than 1 or 2.
#
#     Returns
#     -------
#     w : ndarray
#         Translated vector or list of vectors.
#
#     """
#     w = np.copy(wire_or_vector)
#     if w.dim == 1:
#         w += r0
#     if w.dim==2:
#         for wi in w:
#             wi += r0
#     else:
#         d = wire_or_vector.dim
#         raise ValueError("Wrong number of dimensions for input argument",
#                          "'wire_or_vector'. One or two was expected,",
#                          f"but input had {d}.")
#     return w
#
# def get_rotation(init_axis,target_axis):
#     init_axis = np.array(init_axis)
#     target_axis = np.array( target_axis)
#     z = target_axis/length(target_axis)
#     A = init_axis / length(init_axis) # Normalization
#     if abs(z@A) != 1:
#         # General case
#         # The rotation axis is defined as A x z
#         # The angle is found using basic trigonometry
#         adj, opp = (A@z)*z, A-(A@z)*z
#         rotation_angle = arctan2(opp@opp/length(opp), adj@z)
#         rotation_axis = np.cross(A,z)/length(np.cross(A,z)) # Rotation vector
#     elif z@A == -1:
#         # np.cross(A,z) won't work as A||z, but we still need a π rotation
#         rotation_angle = π
#         rotation_axis = x_vector # Rotation vector
#     else:
#         # No rotation is needed
#         rotation_angle = 0
#         rotation_axis = z_vector
#
#     return rotation_angle, rotation_axis
#
# def change_2loop_ref(loop_primary, loop_or_wire):
#     """Set the wire coordinates in the loop referential.
#
#     Changes the wire coordinates so that it is represented in the loop
#     referential, with the loop axis towards +z and centered at the origin.
#
#     Parameters
#     ----------
#     loop_primary : 2darray
#         First component is the coordinates vector for the loop center. Second
#         vector is the axis of the loop, aka the normal to the loop plane
#     loop_or_wire : 2darray
#         Wire or loop to be transformed in the new referential .
#
#     Returns
#     -------
#     wire_out : 2darray
#         Wire in the loop referential
#
#     """
#     loop_pos = loop_primary[1]
#     loop_axis = loop_primary[2]
#     loop_or_wire = translate(loop_or_wire, -loop_pos) # center on the loop center
#     z = z_vector
#     A = loop_axis / length(loop_axis) # Normalization
#
#     rot_angle, rot_axis = get_rotation(A, z)
#
#     ω = rot_angle*rot_axis
#     rot = Rotation.from_rotvec(ω)
#     loop_or_wire_out = loop_or_wire @ rot.as_matrix().T
#
#     return loop_or_wire_out
#
# def check_intersection(wire1,wire2):
#     # The objective is to determine if the two wires cross each other
#     ε = sys.float_info.epsilon
#     p0, p1 = tuple(wire1)
#     s0, s1 = tuple(wire2)
#
#     Lp, Ls = length(wire1), length(wire2)
#
#     z = (p1-p0)/Lp
#     n = (s1-s0)/Ls
#
#     # TODO : verification of geometry should be performed outside this function
#     # If segments are colinear, must verify if they overlap : error!
#     r0 = (s0-p0)
#     if sqrt(r0@r0)<ε:
#         r0 = z
#     else:
#         r0=r0/sqrt(r0@r0)
#     is_colin = sqrt(1-(r0@z)**2)<ε and sqrt(1-(r0@n)**2)<ε
#     if is_colin:
#         r0 = (s0-p0)
#         i0,i1 = (0,Lp)
#         j0,j1 = tuple( np.sort( [r0@z,r0@z+Ls*n@z] ) ) # Sorted
#         if ( i0<j0<i1
#             or i0<j1<i1
#             or j0<i0<j1
#             or j0<i1<j1
#             or (i0==j0 and i1==j1) ):
#             # The two domains overlap!
#             print('error')
#
#     A = np.array( [(p1-p0)/Lp, -(s1-s0)/Ls]).T
#     B = s0-p0
#     ATA = A.T@A
#     if np.linalg.cond(ATA) < 1/ε:
#         pseudo_inv_A = np.linalg.inv(ATA)@A.T
#         lp, ls = tuple( pseudo_inv_A @ B )
#
#         if 0<lp<Lp and 0<ls<Ls:
#             lp = min(max(lp, 0), Lp)
#             ls = min(max(ls, 0), Ls)
#             intersection = np.array([lp,ls]) #
#         else:
#             intersection =  None
#     else:
#         intersection = None
#     return intersection
#
# def circle_in_3D(pos,radius,normal, npoints=73):
#     # npoints = 73 is a 5° resolution
#     r = radius
#     xc,yc,zc = tuple(pos)
#     a, b, c = tuple(normal)
#
#     # if "normal" || y-axis, then l=0 and x0,x1,z0 are not analytical
#     if a!=0 or c!=0:
#         L, l    = np.sqrt(a**2+b**2+c**2), np.sqrt(a**2+c**2)
#         x0,y0,z0 = xc-r*a*b/l/L, yc+r*l/L, zc-r*b*c/l/L
#         u = np.array( [(x0-xc)/r, (y0-yc)/r, (z0-zc)/r] )
#         ux,uy,uz = tuple(u)
#         n = np.array( normal )
#         vx,vy,vz = tuple( np.cross( n, u ) )
#     else: # if normal is toward
#         ux,uy,uz = 0, 0, 1
#         vx,vy,vz = 1, 0, 0
#
#     φ = np.linspace(0,2*π, npoints)
#
#     coordinates = np.zeros( (len(φ),3) )
#     coordinates[:,0] = xc + r*ux*cos(φ)+r*vx*sin(φ)
#     coordinates[:,1] = yc + r*uy*cos(φ)+r*vy*sin(φ)
#     coordinates[:,2] = zc + r*uz*cos(φ)+r*vz*sin(φ)
#
#     return coordinates
#
#