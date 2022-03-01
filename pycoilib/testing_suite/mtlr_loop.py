import pycoilib as pycoil
import numpy as np

mtlr = pycoil.coil.MTLR(inner_radius=5.21e-3,
                        delta_radius=450e-6,
                        line_width=250e-5,
                        n_turns=6,
                        dielectric_thickness=330e-6)


loop = pycoil.coil.Loop(radius=15e-3, position=np.array([0., 0e-3, 4e-3]), wire=pycoil.wire.WireCircular(radius=0.5e-3))

L_mtlr = mtlr.get_inductance()
L_loop = loop.get_inductance()
mutual = pycoil.get_mutual(loop, mtlr)

k = (np.sqrt(L_loop*L_mtlr)/mutual)**(-1)


print("L mtlr:\t", L_mtlr)
print("L loop:\t", L_loop)

print('mutual:\t', mutual)
print("k:\t\t", k)
print("k**2:\t", k**2)


