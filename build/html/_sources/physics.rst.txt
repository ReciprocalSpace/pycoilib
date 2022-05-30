Physics
==============


Self/mutual inductance
----------------------

The idea behind pycoilib is to compute the inductance of a coil by using the concept of "partial inductance". In this approach, the coil geometry is divided into a collection of :math:`N` simple current segments and the total inductance :math:`L` of the coil is computed using the following equation: 

.. math::

     L = \sum_{i=1}^{N} L_i + 2 \sum_{i=1}^{N-1} \sum_{j=i+1}^{N} M_{i,j}, 

where :math:`L_i` is the self-inductance of the i<sup>th</sup> segment, and :math:`M_{i,j}` is the mutual-inductance between the segments :math:`i` and :math:`j`.

To compute the mutual :math:`M_{p,s}` betweenn a primary and a secondary, we use the Neumann double integral formula:

.. math::

    M_{p,s} = \int_{C_p} d \vec p \left( \frac{\mu_0}{4 \pi} \int_{C_s} \frac{d\vec s}{|\vec p - \vec s|}\right)


where :math:`C_p` and :math:`C_s` are the oriented paths along the primary and secondary segment, respectively, and :math:`\mu_0` is the vacuum permeability. This equation corresponds to the integral over :math:`C_s` of the magnetic vector potential induced by a unitary current flowing over :math:`C_p`. The same expression can also be used to compute a segment self-inductance. 


Furthermore, the Neumann double integral is perfectly equivalent to the better known surface integral:

.. math::

    L_I = \Phi = \int_{S_p} \vec B_s \cdot d\vec S_p,

with :math:`\vec B_s` the magnetic field produced by the secondary over :math:`S_p` the primary coil surface. However, this latter expression is impractical when working with complex coil geometries, as a coil surface becomes difficult the properly define.



Segments parametric equations
-----------------------------

To design a coil geometry, Pycoilib includes three kinds of segments: lines, arc, and loops, as shown in the figure below. The segments can touch at their respective endpoints, but they do not cross each other. 

In this section (and the followings), we will make extensive use of vector notation. So here's a few notation conventions:

* vectors of arbitrary length are denoted with the arrow symbol: :math:`\vec x`
* unit vectors are topped with a hat symbol: :math:`\hat x \vec y \, (\equiv \hat x \cdot \vec y)` 
* scalar products between two vectors are noted without the " :math:`\cdot`": :math:`\vec x \hat y`
* cross products are designated with the wedge symbol: :math:`\hat x \wedge \vec y`


Without any loss of generality, we will assume the primary is centered at the origin :math:`O(0,0,0)` and the secondary is centered at an arbitrary position :math:`\vec s_0 = s_0 \hat s_0 ` with :math:`s_0= \|\vec s_0|` in the laboratory frame of reference.

Current line
------------


Test
----

.. figure:: ../phy/Figures/Fig-2.png
   :width: 400px
   :align: center


