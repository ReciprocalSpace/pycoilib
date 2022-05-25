Pycoilib
========

pycoilib
========

Pycoilib is a python library developed to compute the self-inductance of a coil or the 
mutual-inductance between a pair of coils of arbitrary geometry. Self/mutual 
inductance is computed by solving integrals of analytical expressions of the mutual inductance.

***

Quickstart
----------
Pycoilib is still in development, and not yet accessible on PyPl. However, it can be downloaded 
from https://github.com/ReciprocalSpace/pycoilib. To use the library, add the path to a configuration 
*.pth file in your environment. See https://docs.python.org/3/library/site.html for more information.

The idea behind pycoilib is simple: design the geometry of your coil using segments, select a wire type, 
and compute the inductance. <i>Et voilà!</i> You're all set! See for yourself:


    import pycoilib as pycoil
    
    geometry = [pycoil.segment.Circle(radius=0.05)]  
    wire = pycoil.wire.WireCircular(radius=0.001)
    coil = pycoil.coil.Coil(geometry, wire)
    
    inductance = coil.get_inductance()
    print(inductance)
    
    >>> 2.476691129653111e-07


***

Contribute
----------

- Issue Tracker: github.com/$project/$project/issues
- Source Code: github.com/$project/$project

***

Support
-------
If you are having issues, please let us know.

***

License
-------

MIT License