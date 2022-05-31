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

.. figure:: ../phy/Figures/Fig-1.png
   :width: 400px
   :align: center

   Fig. 1 – Schematic view of the segments and their relevant parameters for arbitrary orientations. Current line as (a) the primary circuit and (b) the secondary circuit. Current arc as (c) the primary and (d) the secondary.




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

When the primary is a current line of length :math:`l_{p}` (Fig. 1a), its parametric equation can be defined as:

.. math::

    \begin{matrix}
    \overrightarrow{p} = \hat{p}p, \\
    d\overrightarrow{p} = \hat{p}dp, \\
    (2.2) \\
    \end{matrix}

for :math:`p \in \left\lbrack 0,l_{p} \right\rbrack` and where :math:`\hat{p}` is the line direction.

Similarly, when the secondary is a line segment of length :math:`l_{s}` beginning at :math:`{\overrightarrow{s}}_{0}` (Fig. 1b), its parametric equation reads:

.. math::

    \begin{matrix}
    \overrightarrow{s} = {\hat{s}}_{0}s_{0} + \hat{s}s, \\
    d\overrightarrow{s} = \hat{s}ds, \\
    (2.3) \\
    \end{matrix}

for :math:`s \in \left\lbrack 0,l_{s} \right\rbrack` and where
:math:`\hat{s}` is the line direction.

Current arcs and loops
----------------------

For the primary, we consider an arc of a circle or radius :math:`R_{p}` and subtending an angle :math:`\theta_{1}` (Fig.~1c). A loop is a sub-type of arcs, when :math:`\theta_{1} = 2\pi.` The
arc parametric equations are:

.. math::

    \begin{matrix}
    \overrightarrow{p} = R_{p}\left( \hat{x}\cos\theta + \hat{y}\sin\theta \right), \\
    d\overrightarrow{p} = R_{p}\left( - \hat{x}\sin\theta + \hat{y}\cos\theta \right)d\theta, \\
    (2.4) \\
    \end{matrix}

for :math:`\theta \in \left\lbrack 0,\theta_{1} \right\rbrack`. The unit
vectors :math:`\hat{x}` and :math:`\hat{y}` are orthogonal, their orientation in
the laboratory frame is arbitrary, and they define the arc plane.

Similarly, for the secondary, we consider an arc of a circle or radius
:math:`R_{s}` and subtending an angle :math:`\varphi_{1}` (Fig.~1d) and centered
at :math:`{\overrightarrow{s}}_{0}`. The parametric equations are:

.. math::

    \begin{matrix}
    \overrightarrow{s} = {\hat{s}}_{0}s_{0} + R_{s}\left( \hat{u}\cos\varphi + \hat{v}\sin\varphi \right), \\
    d\overrightarrow{s} = R_{s}\left( - \hat{u}\sin\varphi + \hat{v}\cos\varphi \right), \\
    (2.5) \\
    \end{matrix}

for :math:`\varphi \in \left\lbrack 0,\varphi_{1} \right\rbrack` . The
vectors :math:`\hat{u}` and :math:`\hat{v}` play the same role as :math:`\hat{x}` 
and :math:`\hat{y}` for the primary.

General solutions for the first integral
----------------------------------------

An important aspect of the double integral of Eq. (2.1) is that the
parametric equations of the two segments :math:`C_{p}` and :math:`C_{s}` are
independent. Therefore, we can solve the first integral of Eq.~(2.1),
which corresponds to the magnetic vector potential produced by unitary
current circulating along a specific secondary segment :math:`C_{s},` while
keeping the details of the primary :math:`C_{p}` hidden in the
:math:`\overrightarrow{p}` and :math:`d\overrightarrow{p}` parameters. This
allows a solution for Eq.~(2.1) that can be more easily generalized to
any type of primary :math:`C_{p}`. 

In this section, we will solve the first integral of Eq.~(2.1):

.. math::

    \begin{matrix}
    \frac{ \mu_{0} }{ 4\pi } \int_{C_{s}} \frac{ d\overrightarrow{s} }{\left| \overrightarrow{p} - \overrightarrow{s} \right|}, \\
    (3.1) \\
    \end{matrix}

for each specific secondary segment :math:`C_{s}` described previously.
Then, in the next section, we will present the solutions to Eq.~(2.1)
for the different pairs of segments under study.

When the secondary is a straight wire
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To solve Eq.~(3.1) when the secondary is a current line, we begin by
replacing :math:`\overrightarrow{s}` and :math:`d\overrightarrow{s}` by their
respective definition of Eq.~(2.3). The denominator squared of Eq.~(3.1)
becomes :

.. math::
    
    \left| \overrightarrow{p} - \overrightarrow{s} \right|^{2} = \left| \overrightarrow{p} - {\hat{s}}_{0}s_{0} - \hat{s}s \right|^{2} = \left| \overrightarrow{p} \right|^{2} + s_{0}^{2} + s^{2} - 2\overrightarrow{p}{\hat{s}}_{0}s_{0} - 2\overrightarrow{p}\hat{s}s + 2{\hat{s}}_{0}\hat{s}s_{0}s


.. math::

    \begin{matrix}
    \left( s + {\hat{s}}_{0}\hat{s}s_{0} - \overrightarrow{p}\hat{s} \right)^{2} + \left| \overrightarrow{p} \right|^{2}\left( 1 - \left( \hat{p}\hat{s} \right)^{2} \right) + s_{0}^{2}\left( 1 - \left( {\hat{s}}_{0}\hat{s} \right)^{2} \right) - 2\left| \overrightarrow{p} \right|s_{0}\left( \hat{p}{\hat{s}}_{0} - \left( {\hat{s}}_{0}\hat{s} \right)\left( \hat{p}\hat{s} \right) \right), \\
    (3.2) \\
    \end{matrix}

After a change in variable, the denominator can be written in a more
compact form:

.. math::

    \begin{matrix}
    \left| \overrightarrow{p} - \overrightarrow{s} \right| = \sqrt{\sigma^{2}\left( s,\overrightarrow{p} \right) + \beta^{2}\left( \overrightarrow{p} \right)}, \\
    (3.3) \\
    \end{matrix}

with,

.. math::
    
    \sigma\left( s,\overrightarrow{p} \right) = s + {\hat{s}}_{0}\hat{s}s_{0} - \overrightarrow{p}\hat{s},


.. math::

    \begin{matrix}
    \beta^2 \left( \overrightarrow{p} \right ) = \left | \overrightarrow{p} \right |^2 \left( 1 - \left( \hat{p}\hat{s} \right)^2 \right) + s_0^2 \left( 1 - \left( \hat{s}_0\hat{s} \right)^{2} \right) - 2\left | \overrightarrow{p} \right | s_0\left( \hat{p} \hat{s}_0 - \left( \hat{s}_0 \hat{s} \right)\left( \hat{p}\hat{s} \right) \right). \\
    (3.4) \\
    \end{matrix}

The proof that :math:`\beta^{2} \geq 0` , and therefore that :math:`\beta` is a
real number, is provided in annex.

If :math:`\beta \neq 0` , this leads to the following solution:

.. math::
    
    \begin{matrix}
    \frac{ \mu_0 }{ 4 \pi } \int_{C_s} \frac{ d \overrightarrow{s} }{ \left | \overrightarrow{p} - \overrightarrow{s} \right | } = \hat{s} \frac{\mu_0}{4\pi} \int_{\sigma \left ( 0,\overrightarrow{p} \right ) }^{\sigma \left ( l_s, \overrightarrow{p} \right ) } \frac{{d\sigma}}{\sqrt{\sigma^{2} + \beta^{2} } } = \hat{s} \frac{ \mu_0 }{4\pi}\left\lbrack {asinh}\left( \frac{\sigma\left( l_p,\overrightarrow{p} \right)}{\beta\left( \overrightarrow{p} \right)} \right) - {asinh}\left( \frac{\sigma\left( 0,\overrightarrow{p} \right)}{\beta\left( \overrightarrow{p} \right)} \right) \right\rbrack, \\
    (3.5) \\
    \end{matrix}

and if :math:`\beta = 0` , we have:

.. math::

    \begin{matrix}
    \frac{\mu_{0}}{4\pi}\int_{C_{s}} \frac{d\overrightarrow{s}}{\left| \overrightarrow{p} - \overrightarrow{s} \right|} = \hat{s}\frac{\mu_{0}}{4\pi}\int_{\sigma\left( 0,\overrightarrow{p} \right)}^{\sigma\left( l_{s},\overrightarrow{p} \right)}\frac{{d\sigma}}{|\sigma|} = \hat{s}\frac{\mu_{0}}{4\pi}\left( {sign}\sigma\left( l_{s},\overrightarrow{p} \right)\ln\left| \sigma\left( l_{s},\overrightarrow{p} \right) \right| - {sign}\sigma\left( 0,\overrightarrow{p} \right)\ln\left| \sigma\left( 0,\overrightarrow{p} \right) \right| \right). \\
    (3.6) \\
    \end{matrix}

When the secondary is an arc
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the same manner, for the arc segment, we inject the parametric
equations of Eq. (2.5) into Eq.~(3.1). The denominator squared is:

.. math::
    
    \left | \overrightarrow{p} - \overrightarrow{s} \right |^{2} = \left| \overrightarrow{p} - {\hat{s}}_{0}s_{0} - R_{s}\left( \hat{u}\cos\varphi + \hat{v}\sin\varphi \right) \right|^{2} = \left| \overrightarrow{p} \right|^{2} + s_{0}^{2} + R_{s}^{2} - 2\overrightarrow{p}{\hat{s}}_{0}s_{0} - 2R_{s}\left( \overrightarrow{p}\hat{u} - s_{0}{\hat{s}}_{0}\hat{u} \right)\cos\varphi - 2R_{2}\left( \overrightarrow{p}\hat{v} - s_{0}{\hat{s}}_{0}\hat{v} \right)\sin\varphi


.. math::

    \begin{matrix}
    A + B\cos\varphi + C\sin\varphi = (A + d)\left( 1 - k^{2}\sin^{2}\psi \right) \\
    (3.7) \\
    \end{matrix}

Where,

:math:`A = \left| \overrightarrow{p} \right|^{2} + s_{0}^{2} + R_{s}^{2} - 2\overrightarrow{p}{\hat{s}}_{0}s_{0},` 

:math:`B = - 2R_{s}\left( \overrightarrow{p}\hat{u} - s_{0}{\hat{s}}_{0}\hat{u} \right),` 

:math:`C = - 2R_{S}\left( \overrightarrow{p}\hat{v} - s_{0}{\hat{s}}_{0}\hat{v} \right),` 

:math:`d = \sqrt{B^{2} + C^{2}},` 

.. math::

    \begin{matrix}
    k^{2} = \frac{2d}{A + d}. \\
    (3.8) \\
    \end{matrix}

And using the following transformation:

:math:`\varphi = 2\psi + \kappa` 

:math:`\tan\kappa = \frac{C}{B}` 

Eq. (3.1) now reads:

.. math::
    
    \int_{C_{s}} \frac{d \overrightarrow{s} d \overrightarrow{p} }{ \left | \overrightarrow{p} - \overrightarrow{s} \right|} = \int_{\psi_{0}}^{\psi_{1}} \frac{2R_{s} \left( - \hat{u} d\overrightarrow{p} \sin( 2 \psi + \kappa) + \hat{v} d \overrightarrow{p} \cos(2 \psi + \kappa) \right) }{ \sqrt{A + d}\sqrt{1 - k^{2}\sin^{2}\psi}} d\Psi

.. math::

    \begin{matrix}
    \frac{- 4R_{s}}{\sqrt{A + d}}\left( \hat{u}d\overrightarrow{p}\cos\kappa + \hat{v}d\overrightarrow{p}\sin\kappa \right)\int_{\psi_{0}}^{\psi_{1}} \frac{\sin\psi\cos\psi}{\sqrt{1 - k^{2}\sin^{2}\psi}}{d\Psi} \\
    \frac{+ {- 2R}_{s}}{\sqrt{A + d}}\left( \hat{u}d \overrightarrow{p}\sin\kappa - \hat{v}d\overrightarrow{p}\cos\kappa \right)\int_{\psi_{0}}^{\psi_{1}} \frac{1 - 2\sin^{2}\psi}{\sqrt{1 - k^{2}\sin^{2}\psi}}{d\Psi}. \\
    (3.9) \\
    \end{matrix}

Where :math:`\psi_{1} = \frac{\left( \varphi_{1} - \kappa \right)}{2},` 

.. math::

    \begin{matrix}
    \psi_{0} = \frac{- \kappa}{2.} \\
    (3.10) \\
    \end{matrix}

To solve the two integrals of Eq.~(3.9), we further define the following
transformation:

.. math::

    \begin{matrix}
    \Delta = \sqrt{1 - k^{2}\sin^{2}\psi}. \\
    (3.11) \\
    \end{matrix}

The first integral leads to:

.. math::

    \begin{matrix}
    \int_{\psi_{0}}^{\psi_{1}} \frac{\sin\psi\cos\psi}{\sqrt{1 - k^{2}\sin^{2}\psi}}{d\Psi} = \int_{\Delta_{0}}^{\Delta_{1}} \frac{- 1}{k^{2}}d\Delta = \frac{- 2}{k^{2}}\left( \frac{\Delta_{1} - \Delta_{0}}{2} \right) = \frac{- 2}{k^{2}}\Phi\left( \psi_{1},\psi_{0},k \right), \\
    (3.12) \\
    \end{matrix}

where,

.. math::

    \begin{matrix}
    \Phi\left( \psi_{1},\psi_{0},k \right) = \left( \frac{\sqrt{1 - k^{2}\sin^{2}\left( \psi_{1} \right)} - \sqrt{1 - k^{2}\sin^{2}\left( \psi_{0} \right)}}{2} \right). \\
    (3.13) \\
    \end{matrix}

The second integral of Eq.~(3.9) can be rewritten as:

.. math::
    \int_{\psi_{0}}^{\psi_{1}}{\frac{1 - 2\sin^{2}\psi}{\sqrt{1 - k^{2}\sin^{2}\psi}}{d\Psi}} = \left( 1 - \frac{k^{2}}{2} \right)\int_{\psi_{0}}^{\psi_{1}} \frac{1}{\sqrt{1 - k^{2}\sin^{2}\psi}}{d\Psi}  + \frac{2}{k^{2}}\int_{\psi_{0}}^{\psi_{1}} \sqrt{1 - k^{2}\sin^{2}\psi}{d\Psi}

.. math::

    \begin{matrix}
    \frac{- 4}{k^{2}}\left\lbrack \left( 1 - \frac{k^{2}}{2} \right)\left( \frac{F\left( \psi_{1},k \right) - F\left( \psi_{0},k \right)}{2} \right) - \left( \frac{E\left( \psi_{1},k \right) - E\left( \psi_{0},k \right)}{2} \right) \right\rbrack = \frac{- 4}{k^{2}}\Psi\left( \psi_{1},\psi_{2},k \right), \\
    (3.14) \\
    \end{matrix}

where,

.. math:: 
    
    \begin{matrix}
    \Psi\left( \psi_{1},\psi_{0},k \right) = \left( 1 - \frac{k^{2}}{2} \right)\left( \frac{F\left( \psi_{1},k \right) - F\left( \psi_{0},k \right)}{2} \right) - \left( \frac{E\left( \psi_{1},k \right) - E\left( \psi_{0},k \right)}{2} \right), \\
    (3.15) \\
    \end{matrix}

and,

:math:`F \left( \psi^{'},k \right ) = \int_0^{\psi^{'}} \sqrt{1 - k^2\sin^2\psi}^{-1}d\Psi,` 

.. math::

    \begin{matrix}
    E\left( \psi^{'},k \right) = \int_{0}^{\psi^{'}}{\sqrt{1 - k^{2}\sin^{2}\psi}} {d\Psi}, \\
    (3.16) \\
    \end{matrix}

are the well-known incomplete elliptic integrals of the first kind
:math:`F(\psi,k)` and second kind :math:`E(\psi,k)` .

Using the two following identities:

:math:`\cos\kappa = \frac{B}{d},` 

.. math::

    \begin{matrix}
    \sin\kappa = \frac{C}{d}, \\
    (3.17) \\
    \end{matrix}

The integral is:

.. math::

    \begin{matrix}
    \int_{C_{s}} \frac{d\overrightarrow{s}d\overrightarrow{p}}{\left| \overrightarrow{p} - \overrightarrow{s} \right|} = \frac{8R_{s}}{k^{2}d\sqrt{A + d}}\left\lbrack \left( \hat{u}d\overrightarrow{p}B + \hat{v}d\overrightarrow{p}C \right)\Phi(k) + \left( \hat{u}d\overrightarrow{p}C - \hat{v}d\overrightarrow{p}B \right)\Psi\left( \psi_{1},\psi_{0},k \right) \right\rbrack. \\
    (3.18) \\
    \end{matrix}

When the secondary is a loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A special case arises when the secondary is a complete loop which lead
to a simplified version of Eq. (3.19). In this case,

:math:`\psi_{1} = \pi - \frac{\kappa}{2},` 

:math:`\psi_{0} = \frac{- \kappa}{2},` 

.. math::

    \begin{matrix}
    \Phi\left( \psi_{1},\psi_{0},k \right) = 0. \\
    (3.19) \\
    \end{matrix}

Also, the function :math:`\Psi(k)` can be re-expressed as a function of the
complete elliptic integrals of the first kind :math:`K(k)` and second kind
:math:`E(k)` , namely solutions to Eq.~(3.16) when :math:`\psi^{'} = \pi` (see
Babic \emph{et al.} for a demonstration):

.. math::

    \begin{matrix}
    \Psi^{'}(k) = \left( 1 - \frac{k^{2}}{2} \right)K(k) - E(k). \\
    (3.20) \\
    \end{matrix}

This leads to the following equation:

.. math::

    \begin{matrix}
    \int_{C_{s}} \frac{d\overrightarrow{s}d\overrightarrow{p}}{\left| \overrightarrow{p} - \overrightarrow{s} \right|} = \frac{8R_{s}}{k^{2}d\sqrt{A + d}}\left( \hat{u}d\hat{p}C - \hat{v}d\hat{p}B \right)\Psi^{'}(k). \\
    (3.21) \\
    \end{matrix}

Integral expression for the mutual inductance
---------------------------------------------

We now proceed to give the solution of Eq.~(2.1) for the different
combinations of primary and secondary segments. As the mutual inductance
is symmetric, that is :math:`M_{a,b} = M_{b,a}` for any segment :math:`a` and
:math:`b` , we will prefer the simplest equation between two possible
representations.

Pair of lines
~~~~~~~~~~~~~

For a pair of current lines, we must treat two separate situations. (i)
:math:`\beta(p) = 0\forall p` : this corresponds to the special case where
:math:`\hat{p}` , :math:`\hat{s}` and :math:`{\hat{s}}_{0}` are three collinear
vectors, that is when
:math:`\left| \hat{p}\hat{s} \right| = \left| {\hat{s}}_{0}\hat{s} \right| = \left| {\hat{s}}_{0}\hat{p} \right| = 1` ;
and (ii) :math:`\beta(p) \neq 0\forall p` otherwise.

When :math:`\hat{p}` , :math:`\hat{s}` and :math:`{\hat{s}}_{0}` are collinear
______________________________________________________________________________

For the first case, we inject Eq.~(2.2) and Eq.~(3.6) into Eq.~(2.1),
leading to the following equation:

.. math::

    \begin{matrix}
    M_{p,s} = \hat{s}\hat{p}\frac{\mu_{0}}{4\pi}\int_{0}^{l_{p}}\left( {sign}\sigma\left( l_{s},p \right)\ln\left| \sigma\left( l_{s},p \right) \right| - {sign}\sigma(0,p)\ln\left| \sigma(0,p) \right| \right){dp}. \\
    (4.1) \\
    \end{matrix}

with

.. math::

    \begin{matrix}
    \sigma(s,p) = s + {\hat{s}}_{0}\hat{s}s_{0} - \hat{p}\hat{s}p. \\
    (4.2) \\
    \end{matrix}

Using the argument that the two wires cannot be superposed, it can be
proven that :math:`\sigma(s,p)` does not change sign for
:math:`s \in \left\lbrack 0,l_{s} \right\rbrack,p \in \left\lbrack 0,l_{p} \right\rbrack` .
Also, using the convention that when :math:`s_{0} = 0` , then
:math:`{\hat{s}}_{0} = \hat{p}` , we have the following properties :
:math:`{sign}\sigma = {\hat{s}}_{0}\hat{s}` and
:math:`\left( \hat{s}\hat{p} \right)\left( {\hat{s}}_{0}\hat{s} \right) = {\hat{s}}_{0}\hat{p}` .
Finally, noting that :math:`\frac{{dp}}{d\sigma} = -1` and that
:math:`\left\lbrack \sigma_{1}\left( l_{p} \right) - \sigma_{1}(0) - \sigma_{0}\left( l_{p} \right) + \sigma_{0}(0) \right\rbrack = 0` ,
the integral of Eq.~(4.1) takes this analytical form:

.. math::

    \begin{matrix}
    M_{p,s} = - {\hat{s}}_{0}\hat{p}\frac{\mu_{0}}{4\pi}\begin{bmatrix}
    \sigma\left( l_{s},l_{p} \right)\ln\left| \sigma\left( l_{s},l_{p} \right) \right| - \sigma\left( l_{s},0 \right)\ln\left| \sigma\left( l_{s},0 \right) \right| \\
     - \sigma\left( 0,l_{p} \right)\ln\left| \sigma\left( 0,l_{p} \right) \right| + \sigma(0,0)\ln\left| \sigma(0,0) \right| \\
    \end{bmatrix}. \\
    (4.3) \\
    \end{matrix}

As a remark, Eq.~(4.2) is undefined for :math:`s,p` such that
:math:`\sigma(s,p) = 0` . By posing :math:`f(\sigma) = \sigma\ln|\sigma|` for
:math:`\sigma \neq 0,f` can be extended to a continuous function at
:math:`\sigma = 0` by posing :math:`f(0) = 0.` 

General case
________________

For the second case, inject Eq.~(2.2) and Eq.~(3.5) into Eq.~(2.1),
leading to the following solution:

.. math::

    \begin{matrix}
    M_{p,s} = \hat{s}\hat{p}\frac{\mu_{0}}{4\pi}\int_{0}^{l_{p}}{\left( {asinh}\frac{\sigma\left( l_{s},p \right)}{\beta(p)} - {asinh}\frac{\sigma(0,p)}{\beta(p)} \right){dp}}, \\
    (4.4) \\
    \end{matrix}

where,

.. math::

    \begin{matrix}
    \beta^{2}(p) = p^{2}\left( 1 - \left( \hat{p}\hat{s} \right)^{2} \right) + s_{0}^{2}\left( 1 - \left( {\hat{s}}_{0}\hat{s} \right)^{2} \right) - 2ps_{0}\left( \hat{p}{\hat{s}}_{0} - \left( {\hat{s}}_{0}\hat{s} \right)\left( \hat{p}\hat{s} \right) \right). \\
    (4.5) \\
    \end{matrix}

The parameter :math:`\beta(p)` can have one root :math:`p_{r}` at:

.. math::

    \begin{matrix}
    p_{r} = s_{0}\frac{\left\lbrack \hat{p}{\hat{s}}_{0} - \left( {\hat{s}}_{0}\hat{s} \right)\left( \hat{p}\hat{s} \right) \right\rbrack}{1 - \left( \hat{p}\hat{s} \right)^{2}}, \\
    (4.6) \\
    \end{matrix}

for which the integrand in Eq.~(4.3) is undefined (c.f. annex), yet
continuous. This problem can be avoided either by extending the
integrand of Eq.~(4.4) to a continuous function at :math:`p_{R}` . For a
given :math:`\sigma` , this problem can be avoided by setting a lower bound
:math:`\epsilon` on :math:`\beta` such that :math:`0 < \epsilon \ll |\sigma|` .

A loop and an arc or a loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We assume the loop to be the secondary coil. Injecting Eq.~(2.4) and
Eq.~(3.22) into Eq.~(2.1) gives:

.. math::

    \begin{matrix}
    M_{p,s} = \frac{{2\mu}_{0}R_{s}R_{p}}{\pi}\int_{0}^{\theta_{1}}{\frac{1}{k^{2}d\sqrt{A + d}}\left\lbrack \left( - \hat{u}\hat{x}C + \hat{v}\hat{x}B \right)\sin\theta + \left( \hat{u}\hat{y}C - \hat{v}\hat{y}B \right)\cos\theta \right\rbrack\Psi^{'}(k){d\theta}}, \\
    (4.7) \\
    \end{matrix}

where :math:`\Psi^{'}(k)` was defined in Eq.~(3.21) and:

:math:`A = R_{p}^{2} + s_{0}^{2} + R_{s}^{2} - 2R_{p}s_{0}\left( {\hat{s}}_{0}\hat{x}\cos\theta + {\hat{s}}_{0}\hat{y}\sin\theta \right),` 

:math:`B = - 2R_{s}\left( R_{p}\left( \hat{u}\hat{x}\cos\theta + \hat{u}\hat{y}\sin\theta \right) + s_{0}{\hat{s}}_{0}\hat{u} \right),` 

:math:`C = - 2R_{s}\left( R_{p}\left( \hat{v}\hat{x}\cos\theta + \hat{v}\hat{y}\sin\theta \right) + s_{0}{\hat{s}}_{0}\hat{v} \right),` 

:math:`d = \sqrt{B^{2} + C^{2}},` 

.. math::

    \begin{matrix}
    k^{2} = \frac{2d}{A + d}. \\
    (4.8) \\
    \end{matrix}

If :math:`\hat{x} = \lbrack 1,0,0\rbrack` ,
:math:`\hat{y} = \lbrack 0,1,0\rbrack` , and :math:`\theta_{1} = 2\pi` , this
equation corresponds to the case treated by Babic \emph{et al.} in their
2010 paper.

Pair of arcs
~~~~~~~~~~~~

In this situation, we need Eq.~(2.4) and Eq.~(3.19). This mutual
inductance is:

.. math::

    \begin{matrix}
    M_{p,s} = \frac{2\mu_{0}R_{s}R_{p}}{\pi}\int_{0}^{\theta_{1}}{\frac{1}{k^{2}d\sqrt{A + d}}\begin{bmatrix}
    \left( \left( - \hat{u}\hat{x}B - \hat{v}\hat{x}C \right)\sin\theta + \left( \hat{u}\hat{y}B + \hat{v}\hat{y}C \right)\cos\theta \right)\Phi\left( \psi_{1},\psi_{0},k \right) \\
     + \left( \left( - \hat{u}\hat{x}C + \hat{v}\hat{x}B \right)\sin\theta + \left( \hat{u}\hat{y}C - \hat{v}\hat{y}B \right)\cos\theta \right)\Psi\left( \psi_{1},\psi_{0},k \right) \\
    \end{bmatrix}{d\theta}}, \\
    (4.9) \\
    \end{matrix}

where the parameters :math:`A` , \emph{B}, \emph{C}, \emph{d} and \emph{k}
are the same as in the previous case (Eq.~(4.4)), the functions
:math:`\Phi\left( \psi_{1},\psi_{0},k \right)` and
:math:`\Psi\left( \psi_{1},\psi_{0},k \right)` were define in Eq.~(3.13) and
Eq.~(3.15) and :

:math:`\psi_{1} = \frac{\left( \varphi_{1} - \kappa \right)}{2},` 

:math:`\psi_{0} = \frac{- \kappa}{2},` 

.. math::

    \begin{matrix}
    \tan\kappa = \frac{C}{B}. \\
    (4.10) \\
    \end{matrix}

A straight wire and an arc or a loop
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we assume the current line is the secondary and the arc is the
primary. Using Eq.~(2.4) and Eq.~(3.5) in Eq.~(2.1) gives the following
result:

.. math::

    \begin{matrix}
    M_{p,s} = \frac{\mu_{0}R_{p}}{4\pi}\int_{0}^{\theta_{1}}{\left( - \hat{s}\hat{x}\sin\theta + \hat{s}\hat{y}\cos\theta \right)\left( {asinh}\frac{\sigma\left( l_{s},\theta \right)}{\beta(\theta)} - {asinh}\frac{\sigma(0,\theta)}{\beta(\theta)} \right)}{d\theta}, \\
    (4.11) \\
    \end{matrix}

with the parameters:

:math:`\sigma(s,\theta) = s + {\hat{s}}_{0}\hat{s}s_{0} - R_{p}\left( \hat{x}\hat{s}\cos\theta + \hat{y}\hat{s}\sin\theta \right)` 

:math:`\beta^{2}(\theta) = R_{p}^{2}\left( 1 - \left( \hat{x}\hat{s}\cos\theta + \hat{y}\hat{s}\sin\theta \right)^{2} \right) + s_{0}^{2}\left( 1 - \left( {\hat{s}}_{0}\hat{s} \right)^{2} \right)` 

.. math::

    \begin{matrix}
     - 2R_{p}s_{0}\left( \left\lbrack \hat{x}{\hat{s}}_{0} - \left( {\hat{s}}_{0}\hat{s} \right)\left( \hat{x}\hat{s} \right) \right\rbrack\cos\theta + \left\lbrack \hat{y}{\hat{s}}_{0} - \left( {\hat{s}}_{0}\hat{s} \right)\left( \hat{y}\hat{s} \right) \right\rbrack\sin\theta \right). \\
    (4.12) \\
    \end{matrix}

As with the case for two lines, :math:`\beta(\theta)` can have roots,
leading to non-analytical points in the integrand of Eq.~(4.7). This can
also be avoided by setting a lower bound :math:`\epsilon` on :math:`\beta` such
that :math:`0 < \epsilon \ll |\sigma|` for these points :math:`\sigma` .
