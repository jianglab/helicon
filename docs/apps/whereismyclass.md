# WhereIsMyClass

**Map 2D classes to the helical tube or filament images they come from.**

After a 2D classification of filament segments, you may want to see where the segments of a class
are: on which filaments, and where along them. WhereIsMyClass lists the filaments (helices) that contain
the classes you select, shows the classes assigned to the selected filaments, and marks them on the
micrograph.

```{image} ../images/whereismyclass.png
:alt: The WhereIsMyClass app before a file is loaded
:width: 100%
```

## What you need

The **particle table** of a 2D classification of filament segments: a RELION `*_data.star` file or a
cryoSPARC `.cs` file. The class averages are found beside it (RELION: `*_classes.mrcs` of the same
job). To see the filaments on their micrographs you also need the **micrographs** (see below); the table, the
classes of each filament and the distances between segments work without them.

## Steps

1. On **Inputs**, choose *server* (a file on the computer that runs Helicon: *Browse…*, or type its
   path) or *url*, give the file, and click **Run**. The classes appear in the gallery, sorted by
   abundance, and the filaments in a table (its *length*, and the *classes* of its segments).
2. **Click a class** to select it: the histogram of the distances between the segments of that class on
   the same filament is plotted (see [HelicalPitch](helicalpitch.md)).
3. **Select a filament** in the table. The classes assigned to it appear under the table, and its micrograph
   on the right, with the segments of the filament marked in a colour for each class. *Mark these classes*
   chooses which classes are marked; *Select all* and *Unselect all* change them all at once. Use shift and click on the displayed micrograph to measure a disance manually.

```{image} ../images/whereismyclass.png
:alt: WhereIsMyClass on the public data of EMPIAR-10940: a filament on its micrograph, marked by class
:width: 100%
```

## The micrographs

- With a **local star file** the micrographs are read from the RELION project that the file is in
  (`<project>/<name of the micrograph in the star file>`). For a cryoSPARC `.cs` file, from the project folder.
- With a **star file given by url**, the project is the one three levels up in the address
  (`<project>/<JobType>/jobNNN/<name>.star`), so that a deposition of a whole RELION project, such as
  EMPIAR-10940, works: the micrograph of the selected filament is **downloaded** from
  `<project>/<its name in the star file>` (30 to 300 MB, depending on the camera), shown, and kept in the cache
  folder for the next visit; the ones not used for the longest time are deleted when they take more than
  2 GB. If the star file is not in a job folder of a project, or its micrographs are not on the server,
  the table and the classes are shown without the micrograph.
- On a **hosting service** (such as the online copy of the apps) there is only *url*: the star file by its address, and its micrographs downloaded from the same project.
- Untick **Download the micrographs of a star file given by url** under *Parameters* to stop the
  downloads (for a slow connection, or one paid for by the MB).

## Settings

On the *Parameters* tab: ignore blank classes, sort the classes by abundance, the download of
micrographs, the helical rise (4.75 Å by default), the pixel size to scale the micrograph down to, low-pass
and high-pass filters, the longest filament and pair distance to use, the number of histogram bins, and the
height of the plot.
