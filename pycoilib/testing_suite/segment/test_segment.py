
import numpy as np
from numpy import pi as π
import matplotlib.pyplot as plt

from pycoilib.segment import Segment, Arc, Line, Loop


def test_line():
    p0 = np.array([0., 0., 0.])
    p1 = np.array([0., 1., 1.2])
    line = Line(p0, p1)
    print(line)
    line.rotate(np.array([0., 1., 0.]), π/3)
    print(line)
    line.flip_current_direction()
    print(line)
    p0, p1 = line.get_endpoints()
    print(f"Endpoints:\t{p0}, {p1}")
    line.translate(np.array([1., 0., 1.]))
    print(line)
    line.draw()


def test_loop():
    # Base constructor
    loop = Loop(0.5, np.array([0., 0., 0.]), np.array([0., 0., 1.]), 0.)
    print(loop)
    loop = Loop.from_normal(0.5, np.array([0., 1., 0.5]), np.array([0.5, 0.1, 0.]))
    loop.draw()
    print(loop)


if __name__ == '__main__':
    test_line()
    test_loop()
