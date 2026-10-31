# AbInitio3D

**Make an ab initio 3D map from the 2D classes of helical filaments.**

AbInitio3D uses what the 2D classification already knows: which class each segment of a filament
was put in, and in what order the classes follow each other along the filament. The classes that
repeat along a filament sit at different *phases* of its helical repeat; the app finds the repeat
(the pitch) and the phase of each class from them, and uses the pitch estimate for the 3D map. It can
also **group filaments by type**: filaments that walk through different sets of classes, because they
are different polymorphs, fall into different groups.

```{image} ../images/abinitio3d.png
:alt: AbInitio3D after estimating the pitch from eight classes
:width: 100%
```

## What you need

The same as for [HelicalPitch](helicalpitch.md): the class averages and the particle table of a 2D
classification of filament segments.

## Steps

1. On the **Inputs** tab, give the *Class2D parameter file* and the *class average images*, and click
   **Run**.
2. **Select the classes** of one type: click a class, shift-click to add more. Leave out junk, and the
   classes of a different kind of filament. Three or more are needed. The *Selected classes* card shows
   what you picked; **Select all** and **Remove picked** change the selection in it.
3. Set **Smallest |twist| (°)** (0.3 by default) and click **Estimate pitch**. The app shows
   - the **repeat** with its error, and the twist for each cyclic symmetry;
   - the **repeat scan** (the peak of the top eigenvalue is the repeat), and the **phase of each
     class** on a ring;
   - the **per-filament repeat**, and for each class how well it **fits** the ring and how well its
     pixels agree with the others. Warnings say which classes fit the ring badly and could be removed.
4. **Suggest classes** proposes more classes of the same type: those that look like a selected class
   turned by 180°, and those on the same filaments as the selected classes.
5. If the data hold more than one type, the **Types** card groups the filaments. Choose the *Method*
   and click **Group filaments**:
   - *Marker classes (class averages)* is fast. It splits the filaments by the classes one group uses
     and the other never does.
   - *Filament context (sequence of classes)* fits a model of the order of the classes along each
     filament, as RELION's helix_filament_context does. It takes minutes on a large data set, and
     needs the segments in order along their filaments.

   The *Groups* slider sets how many groups there are. Pick a group in the list below it, then:
   **Use group** selects the group's classes and no others, and restricts the analysis to the
   group's filaments; **Add group** adds all the group's classes to the selection; **Add selected**
   adds only the classes you picked in the group's gallery. Estimate the pitch again for the group.
   The restriction is the Parameters option *Only the filaments of the type group in use*: untick it
   to use the selected classes on all filaments again.

   With *Filament context*, **Download the N types of every segment (.star)** saves the segments with
   the columns helix_filament_context writes: `rlnHelicalType` and `rlnHelicalTypeProbability`,
   `rlnHelicalSectionID` and `rlnHelicalContextProbability`.
6. **Select the filaments.** Under the per-filament repeat plot, the *Repeat (Å)* and *Length (Å)*
   ranges choose the filaments to keep (blue in the plot, the others grey). Double-click a number to
   type it. **Download N filaments / M segments (.star)** saves those segments, in the selected
   classes, as a RELION star file with their estimated `rlnAngleRot`, `rlnAngleTilt` (90),
   `rlnAnglePsi` and origins. You can reconstruct a starting map from it with `relion_reconstruct`
   (hover over the link for the command).
7. **Make the 3D map.** Set *C symmetry* (the pitch is C × the repeat) and *Hand* (*Left* or
   *Right*; the images cannot tell, the two give mirror images), and click **3D map from classes**.
   The class averages are placed at their azimuths on the ring and put together into one helical
   map. The app shows the averages at their azimuths, the side projection of the map and its central
   z section. **Download map (.mrc)** saves it; **View in browser** opens it in Mol\*.

   The settings of this map are on the **Parameters** tab, under *3D map from classes*:
   - *Reconstruction Method*: *Linear regression - elasticnet* (the default), *Linear regression -
     gauss* or *Back-projection* (with its own *Helical sym order*).
   - *Impose C symmetry*: off by default, so that the C symmetry seen in the z section is an
     independent check.
   - *Two rounds*: align the averages to the first map and build it again.
8. When the app runs on your own computer and finds RELION, a **relion_reconstruct** button appears
   above the C symmetry row. It makes a map from the segments in the selected ranges, with their
   angles and origins (no helical symmetry, and no C symmetry unless *Impose C symmetry* is ticked).
   *RELION project* is the folder that the paths in the star file start from; the app guesses it, and
   runs RELION in it. The row is not shown on hosted copies, or when `relion_reconstruct` is not
   installed.

In the picture, eight classes of EMPIAR-10940 give a repeat of about 712 Å, which is a twist of 2.4° for
a rise of 4.75 Å (360 × 4.75 / 712.5), and the warnings name the classes that fit the ring badly.

## Parameters

Besides the map settings above, the **Parameters** tab has:

- *Helical rise (Å)* (4.75 by default): turns the repeat into a twist, and with *Smallest |twist|*
  sets the longest repeat the estimate searches.
- *Ignore blank classes*, *Sort the classes by abundance* and *Show class labels on the
  azimuthal-angle ring*.
- *Dihedral symmetry check*: see the tips below.
- *Merge classes that are 180°-rotated copies of each other*: on by default; slow for many classes.
- *Split tube ids holding distinct filaments (Å)* (50 by default): segments of one tube id that lie
  farther than this from each other's axis get tube ids of their own, also in the star file. 0 turns
  it off.
- *Only the filaments of the type group in use*: see step 5.
- *Bootstrap resamples* (20 by default): how many times the filaments are resampled for the error of
  the repeat.
- *Decide the repeat with maps from the class averages too (minutes)*: see the tips below.

## Tips

- Select the classes of **one type** at a time. Mixed types blur the ring.
- A class that does not fit the ring, or whose pixels disagree with the others, is likely junk or from
  another type: remove it and estimate again. The *Fit threshold* and *Pixel agreement threshold*
  sliders beside the ranked fits set which classes are picked to remove.
- When the filaments are shorter than a repeat, tick *Decide the repeat with maps from the class
  averages too* (off by default) before **Estimate pitch**. At every candidate repeat, the app makes a
  map from two thirds of the selected averages and scores how well the rest fit it. This decides the
  repeat where the segments cannot, and also tells whether the pitch is one repeat or two (C1 or
  C2). It takes minutes.
- Tick *Dihedral symmetry check* to test for a two-fold axis perpendicular to the helix axis. The
  plot shows, for each azimuth of such an axis, how well each class matches its mirrored partner. A
  sharp peak close to 1, well above the rest of the curve, means a two-fold axis at that azimuth; a
  flat curve with its peak below about 0.75 means plain helical symmetry. The check needs classes
  spread evenly over the azimuths, and it changes nothing in the map.
- A warning **No segment is turned by 180°** means that all segments were aligned to the class
  averages as picked. Either all filaments were picked from the same end, or the filament has no
  polarity because of such a two-fold axis. In the second case the ring is complete with half the
  classes you might expect.
- Hover over the info marks (ⓘ) in the app: they explain each number.
