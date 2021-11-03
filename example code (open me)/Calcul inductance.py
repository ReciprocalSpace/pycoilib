# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:50:57 2021

@author: Aimé Labbé
"""

import pycoilib as pycoil
import numpy as np
from numpy import pi as π

# Une antenne = collection de segments géométriques + un fil (wire)
wire = pycoil.wire.Wire_circ(0.3e-3)  # Fil circulaire de 0.3 mm de rayon

####################################################
# Géométrie Magnétomètre (c'est la géo la plus compliquée dans ce script)
r = 0.15  # rayon des arcs
l = 0.3  # longueur des segments linéaires
x = r * np.sin(π / 4)  # position des segments : 90° entre chaque segment

#  Points de coordonnées fin/début des segments
p0 = np.array([x, x, -0.151])  # -0.151 : pour l'amenée de courant
p1 = np.array([x, x, 0.001])  # !=0 pour éviter les superpositions de fils
p2 = np.array([x, x, l])
p3 = np.array([-x, x, l])
p4 = np.array([-x, x, 0])
p5 = np.array([x, -x, 0])
p6 = np.array([x, -x, l])
p7 = np.array([-x, -x, l])
p8 = np.array([-x, -x, -0.001])
p9 = np.array([x, x, -0.001])

p10 = np.array([x + 1.5e-3, x + 1.5e-3, -0.001])
p11 = np.array([x + 1.5e-3, x + 1.5e-3, -0.151])
z = np.array([0., 0., 1.])  # Vecteur unitaire selon z

# Liste des segments de l'antenne
# Constructeur pour ligne 
#     pycoil.segment.Line(p_begin, p_end)
# Constructeur pour un arc (il en existe d'autres, from_normal, from rot)
#     pycoil.segment.Arc.from_endpoints(p_begin, p_end, arc_angle, normal_axis)

segments = [pycoil.segment.Line(p1, p2), pycoil.segment.Arc.from_endpoints(p2, p3, π / 2, z),
            pycoil.segment.Line(p3, p4), pycoil.segment.Arc.from_endpoints(p4, p5, π, -z), pycoil.segment.Line(p5, p6),
            pycoil.segment.Arc.from_endpoints(p6, p7, π / 2, -z), pycoil.segment.Line(p7, p8),
            pycoil.segment.Arc.from_endpoints(p8, p9, π, z), pycoil.segment.Line(p0, p1), pycoil.segment.Line(p9, p10),
            pycoil.segment.Line(p10, p11)]

# Antenne (constructeur)
coil = pycoil.coil.Coil(segments, wire)

# Affichage -> les points aux extrémités des segments montrent le début (bleu)
# et la fin (rouge) des segments. Il faut penser à bien orienter les segments
# de telle manière que la fin d'un segment (rouge) soit superposée au début 
# (rouge) du segment suivant, de manière à avoir la couleur (mauve). C'est un
# solution temporaire en attendant de trouver quelque chose de plus 
# robuste.
coil.draw()

L = coil.calc_I()
print("L:", L, "(0th-order gradiometer - entire geometry)")

############################################################
# First-order gradiometer
b = 0.3
a = 0.15

θ = π / 180 * 1
y = a * np.sin(θ)
x = a * np.cos(θ)

p1 = np.array([a, 0, 0.15])
p2 = np.array([a, 0, 0])
p3 = np.array([x, -y, 0])
p4 = np.array([x, -y, -b])
p5 = np.array([x, y, -b])
p6 = np.array([x, y, 0.])
p6 = np.array([x, y, 0.15])

# Liste des segments de l'antenne
segments = []
# segments.append( pycoil.segment.Line(p1, p2) )
segments.append(pycoil.segment.Arc.from_endpoints(p2, p3, 2 * π - θ, z.copy()))
segments.append(pycoil.segment.Line(p3, p4))
segments.append(pycoil.segment.Arc.from_endpoints(p4, p5, 2 * π - 2 * θ, -z.copy()))
segments.append(pycoil.segment.Line(p5, p6))

# Antenne
# coil = pycoil.coil.Coil([segments[1],segments[3]], wire)
coil = pycoil.coil.Coil(segments, wire)
# Affichage
coil.draw()

L = coil.calc_I()
print("L:", L, "(1st-order gradiometer - entire geometry)")

#####################################################
# Second-order gradiometer
b = 0.3
a = 0.15

θ = π / 180 * 1
y = a * np.sin(θ)
x = a * np.cos(θ)

p1 = np.array([x, y, 0.15])
# p1 = np.array( [x, y,  0.] )
p2 = np.array([x, y, -2 * b])
p3 = np.array([x, -y, -2 * b])
# p4 = np.array( [x,-y,  0.] )
p4 = np.array([x, -y, 0.15])

# Liste des segments de l'antenne
segments = [pycoil.segment.Line(p1, p2), pycoil.segment.Line(p3, p4),
            pycoil.segment.Loop(a, np.array([0., 0., 0.]), axis=z),
            pycoil.segment.Loop(a, np.array([0., 0., -b + 1e-3]), axis=-z),
            pycoil.segment.Loop(a, np.array([0., 0., -b - 1e-3]), axis=-z),
            pycoil.segment.Loop(a, np.array([0., 0., -2 * b]), axis=+z)]

# Antenne
coil = pycoil.coil.Coil(segments, wire)
coil.draw()

L = coil.calc_I()
print("L:", L, "(2nd-order gradiometer)")
