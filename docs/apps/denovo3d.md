# Denovo3D

**De novo helical indexing and 3D reconstruction from a single 2D image.**

Give Denovo3D one 2D class average (or a projection image) of a helical filament, and it searches
the twist and rise that explain it, scoring a 3D reconstruction for each pair. The best pair, and the
map made with it, come out. This works from the data alone; it needs no starting model.

```{image} ../images/denovo3d.png
:alt: The Denovo3D app on its example data, before a search
:width: 100%
```

## Tutorial: a class average of EMPIAR-10940

The app starts on the 2D classes deposited in the public data set EMPIAR-10940, so you can follow this
without any file of your own.

1. **Start the app** with `helicon` and open the **Denovo3D** tab.
2. **Input.** *How to obtain the input images* offers *server*, *upload*, *url* and *emdb*. The default
   *url* is the `run_it020_classes.mrcs` of EMPIAR-10940. (To use your own, upload the file of a RELION
   or cryoSPARC Class2D job, or give its path with *server*.) The classes appear in the gallery on the
   left.
3. **Select an image** in the gallery. It shows above as a strip, with a dashed line through its middle.
4. **Align the filament with the dashed line.** The filament must lie along it, as in the picture above.
   Click **Auto Transform**, or set it by hand: *Rotation*, *Vertical shift* and *Vertical crop* move and
   crop the image, *Threshold* removes the background, *Pixel size* corrects the scale, and *Transpose*,
   *Flip* and *Invert contrast* change its orientation and contrast. **Reset Transform** goes back.
5. **Choose the search range.** *Twist (deg)* and *Rise (Å)* each have a minimum, a maximum and a step.
   For a first try keep the rise at 4.75 Å (the rise of most amyloids) and search the twist from 0.1° to 2°. Set *Csym* for a cyclic
   symmetry around the axis (1 for none).
6. **Click the run button.** It says **Search Parameters** while a range is searched and **Reconstruct 3D
   map** when twist and rise are single values. **Stop** ends a search.
7. **Read the result.** If one of twist and rise is fixed, the result is a 2D curve of the score against the
   other; if both are searched, it is a heat map. Below the curve, the map reconstructed with the best
   twist is shown.

## Tips

- **The handedness cannot be told from one image.** Search the positive and the negative twist, then
  decide which is right (see [Notes](../notes.md)). The *Parameters* tab can force one sign.
- **A very low twist** (a long pitch) can need the **L1 regularization ratio** on the *Parameters* tab at
  1, to use only the L1 term; the default 0.5 weights L1 and L2 equally.
- **Keep the filament straight** and the image clean: crop away neighbouring filaments and ice.
- Left- and right-arrow keys (or `A` and `D`) move a slider one step after you click its handle.

## A second example: a low-twist class average

A 2D class average of a filament of low twist, with its known twist, is in the archive
<https://zenodo.org/records/15800626/files/helicon.zip>. Unzip it and upload `run_it025_classes.mrcs`.
Select image 29, align it, set the search range for rise and twist, set the L1 regularization ratio to 1
on the *Parameters* tab, and search. The estimated twist is close to the true value, −0.46°, and the
reconstructed density matches the true one.
