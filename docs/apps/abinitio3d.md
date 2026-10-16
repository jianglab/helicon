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
4. **Suggest classes** proposes the classes to keep, and **Types** with **Group filaments** groups the
   filaments by type (the *Method* can use marker classes or the order of the classes along the
   filament). Pick a group and use it to restrict the analysis to its filaments.
5. The **Parameters** tab has the settings of the 3D map made from the classes.

In the picture, eight classes of EMPIAR-10940 give a repeat of about 712 Å, which is a twist of 2.4° for
a rise of 4.75 Å (360 × 4.75 / 712.5), and the warnings name the classes that fit the ring badly.

## Tips

- Select the classes of **one type** at a time. Mixed types blur the ring.
- A class that does not fit the ring, or whose pixels disagree with the others, is likely junk or from
  another type: remove it and estimate again.
- Hover over the info marks (ⓘ) in the app: they explain each number.
