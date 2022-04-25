
import numpy as np
from numpy import pi as π
import matplotlib.pyplot as plt

from pycoilib.segment import Arc, Line, Circle

ST24 = "*"*24
ST12 = "*"*12


def test_line():
    print(ST24 + "Test line" + ST24)
    print(ST12 + "Test constructor: ")
    p0 = np.array([0., 1., 0.])
    p1 = np.array([0., 2., 0.])
    line = Line(p0, p1)
    print(line)

    print(ST12 + "rotate(): π/2 around xaxis centered on line p0")
    line = Line(p0, p1)
    line.draw()
    line.rotate(angle=π/2, axis=np.array([1., 0., 0.]), )
    line.draw()
    print(line)

    print(ST12 + "rotate() π/2 around x-axis centered at (0,-1,0)")
    line = Line(p0, p1)
    line.rotate(angle=π/2, axis=np.array([1., 0., 0.]), origin=np.array([0., -1., 0.]))
    print(line)

    print(ST12 + "flip_current_direction()")
    line = Line(p0, p1)
    line.flip_current_direction()
    print(line)

    print(ST12 + "get_endpoints")
    line = Line(p0, p1)
    p_beg, p_end = line.get_endpoints()
    print(f"Endpoints:\t{p_beg}, {p_end}")

    print(ST12 + "translate() (1,0,1)")
    line = Line(p0, p1)
    print(line.vec_r0)
    line.translate(np.array([1., 0., 1.]))
    print(line)
    print("line.r0:", line.r0)

    print(ST12 + "move_to() (1,0,1)")
    line = Line(p0, p1)
    print(line.vec_r0)
    line.move_to(np.array([1., 0., 1.]))
    print(line)
    print("line.r0:", line.r0)

    print(ST12 + "draw()")
    line = Line(p0, p1)
    line.draw()


def test_arc():
    radius = 1.
    arc_angle = 2*π
    position = np.array([0., 0., 1.])

    print(ST24 + "Test Arc" + ST24)
    print(ST12 + "Arc()")
    arc = Arc(radius, arc_angle, position,)
    print(arc)

    print(ST12 + "Arc.from_normal()")
    arc_angular_pos = 0.
    normal = np.array([0., 1., 0.])
    arc = Arc.from_normal(radius, arc_angle, arc_angular_pos, position, normal)
    print(arc)
    # arc.draw()

    print(ST12 + "Arc.from_rot()")
    axis = np.array([0., 1., 0.])
    angle = π/2
    arc = Arc.from_rot(radius, arc_angle, arc_angular_pos, position, axis, angle)
    print(arc)
    # arc.draw()

    print(ST12 + "arc.rotate(π/2,np.array([0.,1.,0.]))")
    arc = Arc(radius, arc_angle, position,)
    arc.rotate(π/2, np.array([0., 1., 0.]))
    print(arc)

    print(ST12 + "arc.rotate(π/2, np.array([0., 1., 0.]), np.array([0.,-1.,0.])")
    arc = Arc(radius, arc_angle, position, )
    arc.rotate(π / 2, axis=np.array([0., 1., 0.]), origin=np.array([0., -1., 0.]))
    print(arc)

    print(ST12 + "arc.translate(np.array([0., 1., 5.]))")
    arc = Arc(radius, arc_angle, position, )
    arc.translate(np.array([0., 1., 5.]))
    print(arc)
    print(arc.vec_r0)
    print(arc.r0)

    print(ST12 + "arc.move_to(np.array([0., 1., 5.]))")
    arc = Arc(radius, arc_angle, position, )
    arc.move_to(np.array([0., 1., 5.]))
    print(arc)
    print(arc.vec_r0)
    print(arc.r0)

    print(ST12 + "arc.get_endpoints()")
    arc = Arc(radius, arc_angle, position, )
    p0, p1 = arc.get_endpoints()
    print(p0, p1)
    print()

    print(ST12 + "arc.flip_current_direction()")
    arc = Arc(radius, arc_angle, position, )
    arc.flip_current_direction()
    print(arc)


def test_circle():
    radius = 1.
    position = np.array([0., 0., 1.])

    print(ST24 + "Test Circle" + ST24)
    print(ST12 + "Circle()")
    circle = Circle(radius, position, )
    print(circle)

    print(ST12 + "Circle.from_normal()")
    circle_angular_pos = 0.
    normal = np.array([0., 1., 0.])
    circle = Circle.from_normal(radius, position, normal)
    print(circle)

    print(ST12 + "Circle.from_rot()")
    axis = np.array([0., 1., 0.])
    angle = π / 2
    circle = Circle.from_rot(radius, position, axis, angle)
    print(circle)
    # circle.draw()

    print(ST12 + "circle.rotate(π/2,np.array([0.,1.,0.]))")
    circle = Circle(radius, position)
    circle.rotate(π / 2, np.array([0., 1., 0.]))
    print(circle)

    print(ST12 + "circle.rotate(π/2, np.array([0., 1., 0.]), np.array([0.,-1.,0.])")
    circle = Circle(radius, position)
    circle.rotate(π / 2, axis=np.array([0., 1., 0.]), origin=np.array([0., -1., 0.]))
    print(circle)

    print(ST12 + "circle.translate(np.array([0., 1., 5.]))")
    circle = Circle(radius, position)
    circle.translate(np.array([0., 1., 5.]))
    print(circle)
    print(circle.vec_r0)
    print(circle.r0)

    print(ST12 + "circle.move_to(np.array([0., 1., 5.]))")
    circle = Circle(radius, position)
    circle.move_to(np.array([0., 1., 5.]))
    print(circle)
    print(circle.vec_r0)
    print(circle.r0)

    print(ST12 + "circle.get_endpoints()")
    circle = Circle(radius, position)
    p0, p1 = circle.get_endpoints()
    print(p0, p1)
    print()

    print(ST12 + "circle.flip_current_direction()")
    circle = Circle(radius, position)
    circle.flip_current_direction()
    print(circle)


if __name__ == '__main__':
    test_line()
    test_arc()
    test_circle()


