# HelicalLattice

**Convert between a 2D lattice and a helical lattice.**

A helical structure can be seen as a 2D lattice rolled up into a tube. HelicalLattice shows both views
of the same structure side by side, and converts from one to the other.

```{image} ../images/helicallattice.png
:alt: The HelicalLattice app on its example parameters
:width: 100%
```

## Helical → 2D

With **Helical→2D** selected, enter the helical parameters of the structure: *Twist (°)*, *Rise (Å)*,
*Axial symmetry*, *Helical diameter (Å)* and *Helical length (Å)*. The app draws

- the **helical lattice** as a helix of points around the axis;
- the same lattice **unrolled into a 2D lattice** (twist against rise);
- the **2D lattice from which the helix is built**, with its two lattice vectors a and b and the
  unit cell that rolls up into the tube; the equator vector and the circumference are given above it.

Options: *Use primitive unit cell*, *Set unit cell vector a along x-axis*, the *2D lattice size factor*,
*Marker size* and *Plot height*.

## 2D ⇒ Helical

With **2D⇒Helical** selected, enter the 2D lattice by its two unit cell vectors, *a* (x and y, in Å) and
*b*, and the way it is rolled up: *# units along unit cell vector a* and *# units along unit cell
vector b* give the roll-up (equator) vector, the one that becomes the circumference of the tube. The
app draws the same three views and works out the **twist, rise, symmetry and diameter** of the helix
that this lattice makes. The *Helical length*, the *2D lattice size factor*, the *Marker size* and the
*Plot height* set what is drawn.
