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
   left. *emdb* instead takes the EMDB ID of an amyloid structure (or click *Randomize*) and shows the
   X, Y and Z projections of its map; *Generate projection* symmetrizes the map with the twist, rise and
   Csym given there and makes one projection of it, with noise if you ask, as the input image: a test
   with a known answer.
3. **Select an image** in the gallery. It shows above as a strip, with a dashed line through its middle.
   (Untick *Show the central dashed line in the images* on the *Parameters* tab to hide the line.)
   A shift-click adds more images; see [Two or more images](#two-or-more-images).
4. **Align the filament with the dashed line.** The filament must lie along it, as in the picture above.
   Click **Auto Transform**, or set it by hand: *Rotation*, *Vertical shift*, *Vertical crop* and
   *Horizontal crop* move and crop the image, *Threshold* removes the background, *Pixel size* corrects
   the scale, and *Transpose*, *Flip* and *Invert contrast* change its orientation and contrast.
   **Reset Transform** goes back to no rotation, no shift, no crop and no threshold.
5. **Choose the search range.** *Twist (deg)* and *Rise (Å)* each have a minimum, a maximum and a step.
   For a first try keep the rise at 4.75 Å (the rise of most amyloids) and search the twist from 0.1° to 2°. Set *Csym* for a cyclic
   symmetry around the axis (1 for none).
6. **Click the run button.** It says **Search Parameters** while a range is searched and **Reconstruct 3D
   map** when twist and rise are single values. **Stop** ends a search.
7. **Read the result.** If one of twist and rise is fixed, the result is a 2D curve of the score against the
   other; if both are searched, it is a heat map. Below the curve, the map reconstructed with the best
   twist is shown. Click another point of the plot to reconstruct the map of that twist and rise.

## Two or more images

Shift-click more images in the gallery to use them together. *Multiple images selected — what to do
with them:* then offers four ways:

- *Joint parameter search* scores every twist against each image on its own, then combines the score
  curves. It is fast. Use it for several class averages of the same filament type. Each image has its
  own transform card (click an image to edit it), and **Auto Transform** aligns each image.
- *Joint search by projection matching* scores every twist by fitting one reconstruction to all the
  images at once, each placed at its own position along the filament. It is several times slower, and
  on the data measured so far it picks the same twist as the joint parameter search, so treat it as a
  second opinion. It also shows the images joined at the positions that each twist implies.
- *Stitch automatically* is for class averages that are overlapping segments of one filament.
  **Auto Stitch** registers the images to each other and joins them into one longer image, which covers
  more of the helical pitch than one class average does. The stitched image is then searched as a single
  image. A verdict follows: *Registration looks consistent.*, or *Registration is NOT reliable - see
  below.*, with the length of the joined image, how many image pairs registered and how many images were
  placed. Images that cannot be placed are left out. **Undo stitch** brings the separate images back, and
  **Save stitched image (mrc)** saves the stitched image.
- *Stitch manually* lets you place each image yourself: click an image, set its *Rotation (deg)*,
  *Horizontal shift (pixel)*, *Vertical shift (pixel)*, *Flip polarity (x)* and *Flip side (y)*, then
  click **Stitch Images**. It starts from the layout of an automatic stitch or of a search by projection
  matching, if one was made.

*Extend the auto-stitched image as far as it goes* on the *Parameters* tab (on by default) places the
images end to end: it starts from the best-matching pair and adds the image that fits best at either
end, until none fits. The stitched image is then as long as the images allow. Untick it to average the
images over the stretch they share instead. The twist search does not need the extension; it is for
showing a long filament, or for finer layer lines in its Fourier transform.

## Tips

- **The handedness cannot be told from one image.** Search the positive and the negative twist, then
  decide which is right (see [Notes](../notes.md)). When twist and rise are single values, *Reconstruct
  with:* in the twist box chooses *Left-handed twisting (force negative twist)* or *Right-handed
  twisting (force positive twist)*.
- **If every calculation of a search fails**, the app says so and names its log file,
  `helicon.denovo3D.log` in the `logs` folder of the cache folder of Helicon (`~/.cache/helicon` unless
  `HELICON_CACHE_DIR` says otherwise). The reason is in that file.
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
