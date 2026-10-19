# HelicalPitch

**Determine the helical pitch and twist from 2D classification results.**

Filaments in a micrograph are cut into overlapping segments, and a 2D classification gives each
segment a class. Segments of one filament that fall into the same class are one pitch (or a whole
number of pitches) apart. HelicalPitch measures the distances between such segments of the same
class on the same filament and plots them in a histogram: the peaks are at multiples of the pitch.

```{image} ../images/helicalpitch.png
:alt: The HelicalPitch app on its example data
:width: 100%
```

## What you need

- the **class averages** (`*_classes.mrcs` of RELION, or the `.mrc` of cryoSPARC), and
- the **particle table** (`*_data.star` of RELION, or the `.cs` file of cryoSPARC) of a 2D
  classification of **filament segments**.

The app starts on the public data of EMPIAR-10940, so you can look around before you load your own.

## Steps

1. On the **Inputs** tab, choose how to obtain the *Class2D parameter file* and the *class average
   images* (server, upload or url), and click **Run**. The classes appear below, sorted by the number of
   segments each has.
2. **Click a class** (or several: a shift-click adds one). The app shows how many filaments and segments
   the selection has, the lengths of the filaments, and the table of filaments with the classes they
   contain.
3. Read the **histogram of pair distances** on the right. A good class gives clear peaks with equal
   spacing. Hover over the first major peak, away from the origin, to align the vertical lines with the
   peaks; the hover text then gives the **twist for each cyclic symmetry**, assuming that peak is the
   helical pitch. Decide which symmetry and twist is right.
4. If the peaks are not clear, the classes may be poor. Raise **Minimal length (Å)** under the filament
   length plot, to keep only the longer filaments (for example 1000 Å), which often sharpens the peaks.
5. **Download selected helices** saves the table of the filaments that match the length limits.

## Settings

**Helical rise (Å)** is 4.75 by default, the rise of most amyloid filaments along the axis. For
another kind of helical assembly, change it.

## Next

With the pitch and the twist, make a map with [Denovo3D](denovo3d.md) or [AbInitio3D](abinitio3d.md).
