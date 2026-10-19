# HILL

**Helical indexing using layer lines.**

HILL considers a helical structure as the product of a continuous helix and a set of parallel planes.
By the convolution theorem, the Fourier transform of the structure is the convolution of the transform
of the continuous helix and that of the planes. The transform of a continuous helix is a set of equally
spaced layer planes (3D) or layer lines (2D projection), described by Bessel functions of increasing
order (0, ±1, ±2, …) from the Fourier origin. You match these layer lines to the peaks in the power
spectrum of a 2D class average or of a 3D map by moving the **twist**, **pitch** (or **rise**) and
**symmetry** controls, and read the helical parameters from the match.

```{image} ../images/hill/overview.png
:alt: The HILL app on its example data
:width: 100%
```

## The screen

**On the left**, three tabs. *Inputs* has the *Input Mode* (*server*, *upload*, *url* or *emd-xxxxx*, an
EMDB id), the gallery of the images (or the sections of the map) to choose from, and *Image Parameters*
for the selected image. *Parameters* has the settings that rarely need a change: the out-of-plane tilt and
whether twist or pitch is kept when the rise changes (*Helical symmetry*), the resolution limits and the
size of the Fourier transform (*Fourier transform*), the peak detection of the profile (*Y profile*), what
the *Layer lines* show, and what is *Display*ed. *Advanced* has more tools.

**On the right**, above the plots, the helical parameters: **Csym** (the cyclic symmetry), and the sliders
for the **Twist**, the **Pitch** and the **Rise**. Change one, and the others that follow from it change
(pitch = 360 × rise / twist). Below them, from left to right:

- the **power spectrum**, with the layer lines of the helical parameters drawn on it, labelled with their
  Bessel orders, in a colour for each order *m* of the repeat;
- the **profile along the meridian** (the vertical axis of the transform), with its peaks marked;
- the **phase difference across the meridian** (PD): dark blue where the two sides of a layer line are in
  phase (0°) and bright yellow where they are opposite (180°).

The helical axis of HILL is vertical.

## Import data

HILL reads 2D class averages, or a 3D map, as the input. For your own data, use the `mrc` or `mrcs`
file from a Class2D job of cryoSPARC or RELION (*upload*, *url*, or *server* for a file on the computer
that runs Helicon). To start from a map of the EMDB, choose *emd-xxxxx* and give its id.

```{image} ../images/hill/input.png
:alt: The input panel of HILL
:width: 45%
```

## Data quality assessment

Go through the class averages to judge their quality. A good class average shows the features of
the filament and is not blurred or noisy. Such an image gives clear horizontal layer lines in the power
spectrum, and the PD plot shows consistently dark blue or yellow layer lines.

Many things make an image poor. If the helical axis is not vertical, the layer lines are slanted instead
of horizontal. Turn the image with the **Rotate (°)** slider under *Image Parameters*, or click
**Auto-transform**, until the layer lines are horizontal. Here the rotation was set 25° off:

```{image} ../images/hill/slanted.png
:alt: Slanted layer lines
```

and here it is right:

```{image} ../images/hill/straight.png
:alt: Horizontal layer lines
```

*Image Parameters* also has the *Pixel size*, the *Shift X* and *Shift Y* that centre the filament, the
*Mask length* along the filament, and *Invert image contrast*. Below it, the **Transformed Image** shows what
the transform is made from.

```{image} ../images/hill/image-parameters.png
:alt: The image parameters of HILL, and the transformed image
:width: 45%
```

Another factor that disturbs the indexing is the **out-of-plane tilt** of the filaments. In ideal
conditions of grid preparation and uniform ice thickness, filaments are perpendicular to the electron
beam. In practice, some grids are bent or undulated or have non-uniform ice, so the filaments may tilt out
of the plane of the image. A large tilt complicates indexing: avoid such images. The PD plot is a
sensitive gauge of it: an image with out-of-plane tilt shows a mosaic of blue and yellow, and you should
avoid it for indexing; look for the images that give dark blue or bright yellow layer lines. In the two
pictures above (image 1) and below (image 2) of the example data, the PD of image 2 has the clearer
dark layer lines. If you know the tilt, enter it under *Parameters*, *Helical symmetry*, *Out-of-plane tilt (°)*.

After you have chosen a class image, or a projection of a map with the least out-of-plane tilt, that
shows clear layer lines in both the power spectrum and the PD plot, you can start to index the layer
lines.

Helical assemblies can have an additional cyclic symmetry: two-fold, three-fold, four-fold and so on.
The examples below show one with an odd symmetry and one with an even symmetry.

## Example 1: VipA/VipB

**Example data:** in *Input Mode* select *url*, with the address

```text
https://tinyurl.com/y5tq9fqa
```

(the default of the app), and choose the 2nd of the 13 images.

The power spectrum shows clear horizontal layer lines, and the PD plot the corresponding dark blue
bands. The layer lines of the current parameters are drawn on both, and labelled with their Bessel
orders. Under *Parameters*, *Layer lines*, *Layer line text* switches the labels off, so that the layer
lines show as ellipses only, and *Max M* sets how many orders *m* are drawn.

```{image} ../images/hill/example1.png
:alt: The power spectrum, the profile along the meridian and the PD plot of image 2
```

Under the image on the left you see the **radial profile**, the distribution of the density across the
filament. The **diameter** of the filament is a parameter of the indexing: it is used to estimate the
Bessel order of the layer lines. Do not use the diameter measured at the edge of the filament or at a peak
of the radial profile. HILL estimates it by default, from a model of a core and a shell, using the centre
of mass of the density along the radius. Use that estimate, unless the Bessel orders are unclear.

```{image} ../images/hill/radial-profile.png
:alt: The transformed image, its radial profile and the diameter
:width: 45%
```

Another useful setting is the **resolution limit** on X and Y (*Parameters*, *Fourier transform*). A
lower limit lets you look at the region close to the centre of the power spectrum and the PD plot.

### Estimation of the helical rise

In this example, a precise **rise** is found by moving the *Rise* slider until the centres of the X
patterns (the groups of labelled layer lines with m = 1 or −1) match the peaks on the meridian of the
power spectrum. Increasing the rise decreases the distance between the centres of the X patterns. Here
the rise is 24 Å, a little too large; compare the 21.92 Å above.

```{image} ../images/hill/rise.png
:alt: The rise set to 24 Å
```

Note that the peak on the meridian of the power spectrum does not always correspond to the rise.
Out-of-plane tilt can also make artificial peaks on the meridian, so assess the tilt in the PD plot before
you start the indexing.

### Estimation of the helical pitch and the symmetry

With the rise known, keep it and adjust the **pitch** and the **Csym**. Change the *Pitch* slider (or the
*Twist*): increasing the pitch decreases the distance along the meridian between the labelled layer lines.
Here the twist is 24° instead of 29.4°, which makes the pitch 328.8 Å instead of 268.4 Å:

```{image} ../images/hill/pitch.png
:alt: The twist set to 24 degrees
```

For the symmetry, look first at the PD plot. If the phase difference along a layer line is consistently
blue (PD = 0°), the Bessel order of the layer line is even, as in this example: try only even symmetries
(2, 4, 6, 8, …). If it is consistently yellow (PD = 180°), the Bessel order is odd (1, 3, 5, …): you will see
an example in the next section.

Start with C2. The aim is to match the adjacent ellipses (the layer lines whose Bessel orders differ by
2, the symmetry you set) with the peaks in the power spectrum. After adjusting the pitch, look at whether the
first labelled peaks overlap the first peaks of the layer lines in the power spectrum. When you hover over
the first peak in the power spectrum, the hover tip gives the Bessel order (Jn) of that layer line, if the
cursor is on the first peak.

You may find a range of Bessel orders that suit when you move over a region that you think is a peak (for
example 4, 5, 6). The PD plot helps to decide whether the order is even or odd. Adjust the *Csym* to match
the ellipse to the first peak of the layer line.

The first peak off the meridian of each layer line in the power spectrum corresponds to the first peak of
the Bessel function. The distance $r$ from the meridian to the first peak of the Bessel function
$J_{n}(2\pi Rr)$ is set by the Bessel order $n$ and the diameter $R$ of the filament in real space. The
position $X_{0}$ of the first peak of a Bessel function is fixed by its order (a common approximation is
$X_{0} = n + 2$), so $X_{0} = 2\pi Rr$ is fixed. The position of the first peak in the power spectrum
(which is $r$) is therefore set by the Bessel order, and is inversely proportional to the filament diameter
in real space.

In this case the first peaks fit the ellipses best at Csym = 6, after the pitch is adjusted whenever the
Csym changes. This gives a good estimate of the parameters: record the rise, the pitch or twist, and the
symmetry, and go on.

## Example 2: EMD-26987, F-actin

In *Input Mode* select *emd-xxxxx* and give the EMDB id `emd-26987`. The app takes the helical
parameters reported in the EMDB entry, and plots the data with them:

```{image} ../images/hill/example2.png
:alt: HILL on EMD-26987, F-actin
```

The PD plot shows alternating blue and yellow layer lines: the symmetry is odd (here C1). From the power
spectrum alone it may be hard to match the labelled layer lines with the peaks, but the PD plot separates
the blue and the yellow layer lines clearly. With the reported parameters, the labelled layer lines of even
Bessel order fall into the blue regions of the PD plot and those of odd order into the yellow regions, and
the centres of the X patterns fall into the centre of the blue regions on the meridian. This is what an odd
symmetry looks like, and it shows how the clear patterns of the PD plot help to determine the helical
parameters when the power spectrum is not easy to read.

## Example 3: EMD-10129, TMV

Give the EMDB id `emd-10129`. With the images and the helical parameters reported in the EMDB entry, the
plots are these, with the resolution limit on X and Y set to 5 Å:

```{image} ../images/hill/example3.png
:alt: HILL on EMD-10129, TMV
```

There are very many layer lines. The reported helix has a very small rise, which puts the centres of the X
patterns very far apart: they cannot be matched with any layer line you can see in the plot, so you cannot
determine the rise first. Instead, match the layer lines of the many distant X patterns with the peaks, from
the Bessel orders of the layer lines and the PD pattern.

## Example 4: PHF tau, EMD-0259

Give the EMDB id `emd-0259`. PHF tau has a 2₁ screw symmetry, which is practically a kind of C1 symmetry
with a rise of 2.375 Å (half of the 4.75 Å of the usual amyloid) and a twist of about 180°. Such parameters
put the centres of the X patterns very far apart and make the pitch very small, so adjacent layer lines
are very close. Here the layer lines are drawn as ellipses (*Layer line text* off, *Max M* = 1), because
the labels of so many layer lines cover each other:

```{image} ../images/hill/example4.png
:alt: HILL on EMD-0259, PHF tau
```

Layer lines from several X patterns far from the equator, with odd Bessel orders and with even orders,
form separate regions of the power spectrum. In the PD plot the equator should be consistently blue, and
the other regions consistently yellow.

## Example 5: EMD-0260, SF tau

Give the EMDB id `emd-0260`. With the helical parameters of the EMDB entry the plots are these:

```{image} ../images/hill/example5.png
:alt: HILL on EMD-0260, SF tau
```

Here the centres of the X patterns match the peaks of the power spectrum with a rise of about 4.75 Å. The
reported helical parameters give the filament a very large pitch, so the adjacent layer lines are close and
the X patterns are flat, with dense layer lines. The PD pattern tells SF tau from PHF tau: in PHF tau, the
layer lines off the equator (m = 1 or −1) are consistently yellow. In SF tau, with C1 symmetry, the layer
line of Bessel order 0 is on the meridian, so the peak on the meridian matches a blue region of the PD
plot, and the remaining adjacent layer lines, with alternating odd and even Bessel orders, alternate blue
and yellow.
