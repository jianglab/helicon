# Notes and answers

## Twist, rise and pitch

A helix is described by its **twist** (the rotation, in degrees, from one subunit to the next along
the helix) and its **rise** (the shift along the axis, in Å). The **pitch**, the length of one full
turn, follows from them:

> pitch = 360 × rise / |twist|

Helical assemblies often have a **cyclic symmetry** (Csym), the same arrangement repeated around
the axis. The twist and rise are then those of the asymmetric unit; the apps ask for Csym when it
matters, and show the twist for each possible symmetry where it cannot be known.

## Handedness

Helicon, like RELION, writes a **left-handed** helix as a **negative twist** and a right-handed one
as a positive twist.

The handedness of a helical structure cannot be told from a single 2D image or from the 2D
classes: the projections of a left-handed and a right-handed helix look the same. Apps that search
the twist therefore try both signs, and you decide which is right from what you know. In a
reconstruction at a resolution where the secondary structure shows, α-helices are right-handed:
the hand in which they are is the correct one. At a lower resolution, use what is known about the
structure. If the map you made has the wrong hand, flip it.

## Align the filament with the equator

For a 3D reconstruction from one image (Denovo3D) and for the layer-line apps, the filament must
lie horizontally (Denovo3D) or vertically (HILL) in the image, as the app's picture shows. Use *Auto
Transform*, or the *Rotation*, *Vertical shift* and *Transpose* controls, until the filament sits on
the dashed line. A tilted axis shows up as slanted layer lines in the power spectrum.

## What makes a good 2D class

A good class average shows the features of the filament and is not blurred or noisy: clear layer
lines in its power spectrum, and consistently dark blue or yellow layer lines in the phase
difference plot of HILL. Classes of ice, carbon edges and other non-filament particles, or of
filaments that tilt out of the plane of the image, give poor results; choose the clearest classes
for indexing.

## Units

All distances are in Å, angles in degrees.

## Where do I report a problem or ask for a feature?

On the [issue tracker](https://github.com/jianglab/helicon/issues/new). Please include the output of
`helicon --version`, how you installed Helicon, your operating system, which app or command, and what you
did. [Report a problem](report-a-problem.md) lists what helps most.
