# HelicalProjection

**Compare 2D images with projections of helical structures.**

Given a 2D class average or image of a filament, HelicalProjection helps you find which known helical
structure it looks like: it compares the image with projections of helical structure maps: your own, those of the amyloid
atlas, or those of the EMDB.
Use it when you have an image and no idea yet of its helical parameters, or to check the parameters you
found.

```{image} ../images/helicalprojection.png
:alt: The HelicalProjection app on its example data
:width: 100%
```

## Steps

1. On **Input 2D Images**, give the images (*server*, *upload* or *url*). The classes appear in the
   gallery; click the one to use.
2. Bring the filament to the dashed line in the strip above: **Auto Transform**, or set the *Rotation*,
   *Threshold*, *Vertical crop size* and *Vertical shift* by hand. **Reset Transform** undoes it.
3. On **Input 3D Maps**, choose where the maps to compare with come from: *upload*, *url*, the
   *amyloid_atlas*, the helical structures of the EMDB (*EMDB-helical*), or all of the *EMDB*. For the
   last three a table lists the entries.
4. Click **Compare projections**.

## Settings

The *Parameters* tab can ignore blank images, show PDB ids in the EMDB table, use the curated helical
parameters of the EMDB entries (from the project
[EMDB_helical_parameter_curation](https://github.com/jianglab/EMDB_helical_parameter_curation)) instead
of those reported in them, and show the twist adjusted for the symmetry.
