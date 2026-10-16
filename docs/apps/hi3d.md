# HI3D

**Helical indexing using the cylindrical projection of a 3D map.**

HI3D considers a helical structure as a 2D crystal that has been rolled up into a cylindrical tube while
keeping the original lattice. The indexing computationally reverses this: the 3D helical structure is
first **unrolled** into a 2D image by a cylindrical projection, and then the 2D lattice is found
automatically, from which the helical parameters (**twist, rise and cyclic symmetry**) follow.

```{image} ../images/hi3d.png
:alt: The HI3D app on EMD-10499
:width: 100%
```

## Steps

1. Give the 3D map: *How to obtain the input map* offers *server*, *upload*, *url* and *emd-xxxxx* (an
   EMDB ID; the app starts on EMD-10499, and links to the list of helical structures of the EMDB).
2. Choose the **section** of the map to display along an axis, and use *Transform the map* if the map
   needs to be rotated, shifted or cropped so that the helical axis is where the app expects it.
3. Look at the **cylindrical projection** (the unrolled map) and its **auto-correlation**. The big plot
   shows the lattice found, with the twist, pitch, rise and symmetry in its title.
4. The numbers of **Twist (°)**, **Rise (Å)** and **Csym** can be edited; **Refine twist/rise** improves
   them from the data.

## Tips

If the helical parameters found with the default settings are not consistent, change the *rmin* and *rmax*
(the range of radii to project), the number of peaks, and the axial step size on the *Parameters* tab.
