# The file browser

`helicon display` is a file browser for the files of cryo-EM work: images and image stacks, 3D
maps, RELION star files and CryoSPARC `.cs` files, and also bild, eps, pdf, html and text files. It
needs the `gui` extra (`pip install "helicon[gui]"`; see [Install](install.md)).

```sh
helicon display             # browse the current folder
helicon display <folder>    # browse a folder
```

```{image} images/display.png
:alt: The file browser on a RELION Class2D job
:width: 100%
```

## What is in the window

- **The path bar** at the top shows where you are; click any part of it to go there. The *Filter* box
  next to it limits the list to the names that match (`*` is any text; tick *Regex* for a regular
  expression).
- **The list** shows the files with their size, type, and, for an image or a stack, the number of
  images, their dimensions and the pixel size. Click a column to sort. A **link** (a RELION job linked
  from another folder, say) has a small arrow on its icon, and its tooltip says where it points. A
  linked folder opens at the path of the link, not of its target; to go to the target, right-click it
  (*Go to Linked Folder*, or *Show Linked File in Its Folder* for a file).
- **The sidebar** has *Places*, the *Bookmarks* you keep (menu *Bookmarks*), and the folders you opened
  last (*Recent*). In a **RELION or CryoSPARC project** it also shows the project: *Project folder*, and
  below it the job types and their jobs, so that you can move between jobs with a click.
  A **bookmark** is called by the name of its folder; to give it a name of your own, right-click it in
  the sidebar (*Rename bookmark…*), or use *Bookmarks* → *Manage Bookmarks…* → *Rename…* (a double click
  on an entry does the same). The folder is still the tooltip, and what the *Bookmarks* menu shows after the
  name; an empty name gives back the name of the folder.
- **The preview** on the right shows what the selected file holds: a picture for an image, a stack or a
  map, the table for a star file, the text for a text file, with its size, pixel size and other facts.
  Below the facts is a button for each way to open the selected file, and the *New window* box. Which
  buttons there are depends on the file (below). When you hide the preview pane, the buttons go to the
  bottom of the window.
  Right-click a picture of the preview (*Save image as…*) to save it as an image file, at its own size,
  not as it is scaled in the pane.
- **The preview of a star file** shows more than its table:
  - a table of micrographs (`micrographs.star` of an import, `corrected_micrographs.star` of a motion
    correction, `micrographs_ctf.star` of a CTF estimation) shows its first micrograph, with its defocus and
    the resolution of its CTF fit when the table has them;
  - the particle table of a RELION Class2D, Class3D or Refine3D job (`run_data.star`,
    `run_it025_data.star`) shows the micrograph with the most filaments, with a dot at each segment, one
    colour per filament, and says how many of its filaments are fragmented (have gaps wider than 1.5 times
    the spacing of the segments).

  Both pictures need more reading than the table, so they come a moment after it.

The window stays responsive while a long folder is listed or a large file is read: the work is done in
the background, and a folder of tens of thousands of files appears in a moment.

## Jobs of a RELION or CryoSPARC project

In a RELION or CryoSPARC project the sidebar lists the jobs by type, and a click on a job goes to it. The
preview of the job says what it is and how it ended, shows **what the job made** in place of the list of
its files (the files are in the list of the window), and has what is needed to move around the project:

- **The outcome**: what the job made, as pictures and numbers. The files of the job are in the list of
  the window, so the preview does not list them again. A job whose outcome is not known (a job of
  another type, or one that has not made anything yet) shows only its state and its links.

  | Job | What the preview shows |
  |---|---|
  | Import | the number of movies, micrographs or particles imported |
  | MotionCorr (motion correction) | the histogram of the length of the motion trajectories, and the trajectories of a few micrographs, from the least to the most motion |
  | CtfFind, CTF estimation | the histograms of the defocus and of the maximum resolution of the CTF fit |
  | ManualPick, AutoPick, pickers | the number of particles (or filaments, when picked by their start and end), of micrographs, and the average per micrograph; for filaments picked in RELION (by their start and end, or as helical segments), the histogram of their lengths |
  | Extract | the number of particles and micrographs, the average per micrograph, and a few of the particles |
  | Select | the classes and the particles kept, of all of them (2D or 3D), with the percentage of the particles, and the selected 2D class averages |
  | Class2D, 2D classification | the class averages of the last iteration, the most populated first, with the share of the particles of the first few |
  | Class3D, InitialModel, 3D classification | the central section of each map; for a RELION Class3D, the histogram of the tilt angles of the particles, with their median |
  | Refine3D, refinement | the central section of the map, the **FSC** curve with the resolution at FSC 0.143 (RELION only: CryoSPARC keeps the curve in its database, not in files), the histogram of the **tilt angles** with their median (RELION), the **views**: how many particles there are at each rotation and tilt angle (the first 300,000 particles) |
  | CtfRefine | the histogram of the change of the defocus of the particles (compared with the input of the job), and the fitted aberrations: beam tilt, higher-order aberrations and anisotropic magnification |
  | Polish (RELION), reference-based motion correction (CryoSPARC) | how much each movie frame contributes at some resolutions: early frames count at high resolution, late ones only at low resolution |
  | PostProcess | the central section of the sharpened map, the FSC curve, the resolution and the B-factor |
  | MaskCreate | the central section of the mask, and of the map it was made from multiplied by the mask; *Open masked map* writes that map (to the cache of Helicon) and opens it |
  | Reconstruct3D, reconstruction | the central section of the map |
  | LocalRes | the central section of the local resolution map, and the histogram of its values |
  | ModelAngelo | the number of chains, residues and atoms of the model |
  | Sharpen (CryoSPARC) | the central section of the sharpened map |
  | Curate exposures, particle subsets, downsampling (CryoSPARC) | the numbers: exposures accepted of all, subsets and their sizes, particles and box size |

  The number of filaments in a table of many GB (CryoSPARC stores a table row by row, so counting needs
  all of the file) is first an estimate from a few blocks of it, marked "about"; the preview counts them
  exactly in the background, replaces the estimate when it is done, and keeps the number in the cache
  of the user (in the cache folder of Helicon, `~/.cache/helicon` unless `HELICON_CACHE_DIR` says otherwise), so that it is there at once the next time and
  until the table is written again.

  A CryoSPARC project shows it for the jobs that import, correct the motion (also the reference-based
  one), estimate the CTF, pick, extract, select, classify (2D and 3D), refine, reconstruct and sharpen;
  the change of the defocus and the fitted aberrations of a CTF refinement, polishing, post-processing,
  mask creation, local resolution and ModelAngelo are RELION jobs. A CryoSPARC refinement has no FSC curve here,
  because CryoSPARC keeps it in its database.

- **Inputs from** lists the jobs whose files this job uses, and **Used by** the jobs that use its files.
  Each is a link: click it to go to that job. (RELION reads this from `default_pipeline.star`, CryoSPARC
  from the `parents` and `children` of `job.json`.)

```{image} images/display-job.png
:alt: The preview of a job, with the jobs before and after it as links
:width: 100%
```

- **Project folder** (the first entry of the project in the sidebar) shows the jobs of the project as a
  **graph** that runs down: a box for each job, below the jobs whose files it uses, coloured by how it
  ended (green: done, red: failed or aborted, yellow: running, blue: queued, grey: not run), with an arrow
  from each job to the jobs that use its files. A click on a box goes to that job. A large graph opens on
  the newest jobs. **Open in a window** puts it in a window of its own:
  - the **wheel scrolls** up and down (with `Shift`, sideways) and **`Ctrl` + wheel zooms**; a drag moves
    the graph, and a double click on the background shows all of it again (also the *Fit* button and `F`);
  - with the **pointer on a job**, the job, the jobs it takes its files from (its inputs) and the jobs that
    use its files are marked in three colours, with their arrows; the others step back;
  - **right-click** a job for a menu: *Highlight … and its ancestors* marks the job and every job it
    descends from (in two colours, with the arrows between them) and lets the others step back, until
    *Clear the highlight*; *Save the highlighted graph as PDF…* writes those jobs, laid out on their own,
    with the job marked, and *Save the entire graph as PDF…* writes all the jobs (a very long graph is cut
    into bands side by side, and a page is never larger than a PDF reader opens). The files are vector
    drawings with the labels as text, on a white page whatever the theme of the window. A right-click on
    the background offers the entire graph only;
  - a graph that is much longer than the window is wide is cut into **bands side by side**, to use the
    width of the window (an arrow that goes on in the next band is dashed), and is cut again when you
    resize the window;
  - the box of the job that the browser is in is marked, and follows you as you go from job to job.

```{image} images/display-job-graph.png
:alt: The jobs of a RELION project as a graph, in its window
:width: 80%
```

The graph of a large project is read and drawn in the background, so the window keeps answering; the
labels are left out when you zoom out too far to read them.

A RELION job can have an **alias**, a name of your own for it. RELION makes it a link to the job in the
folder of the job type (`Class2D/thin_filaments` for `Class2D/job010`) and a line in
`default_pipeline.star`. The sidebar and the list show the job and its alias in one row, `job010(thin_filaments)`,
and not as two. To add, edit or remove one, right-click the job in the sidebar (*Add alias…*, *Edit alias…*,
*Remove alias*); both the link and the pipeline are changed, as RELION does. An alias cannot have spaces or
slashes, cannot look like a job name, and cannot be one that is in use; a project that RELION is writing
at that moment is left alone, and you can try again.

## Open a file

Select the file and click one of its buttons in the preview pane (below the facts), or double-click the
file. The buttons offer only what fits the file you selected. The buttons that open a viewer window are drawn plain; those
that open a web app or a panel of tools are outlined in colour. Tick **New window** to open the file in a
window of its own instead of the one already open. A file that Helicon cannot read gets no buttons.

### Viewing images and maps

**2D Image**, **2D Slice**, **PDF**, **EPS**, or **Image Slice**
: The first button of an image file; its name tells what the file is: *2D Image* for an image or a stack
  of images (including the particles of a star file), *2D Slice* for a 3D map (one section at a time, which
  you move through), *PDF* and *EPS* for those documents. The file opens in a viewer window.

**Gallery**
: A grid of thumbnails of all the images of a stack, loaded as you scroll, so that a stack of tens of
  thousands of particles opens at once. For a star file it shows the particles it lists.

**3D Volume**
: The map as a 3D volume that you can turn and zoom.

**Ortho Slice**
: An interactive viewer of the three orthogonal sections (XY, XZ and YZ) of a 3D map, with controls to
  move each section.

**Projection**
: For a 3D map: its projections along Z, Y and X, side by side in a gallery window.

**ChimeraX**
: Opens the file in ChimeraX, which must be installed; the button is greyed out, with a note, when Helicon
  cannot find it (add it to your `PATH`).

**XYZ Slice**
: For the `optimiser.star` of a RELION job: the centre sections (Z, Y and X) of the maps that it refers to.

**2D Classes**
: The 2D class averages of a Class2D job. The panel at the side sorts them by abundance (the number
  of particles of each), by name, or not at all (the order of the `model.star`); when the `model.star` has a
  `rlnHelicalType` column, also by helical type (the default then: grouped by type, the most probable
  class first within a type). **Label columns** picks the columns of the `model.star` shown on each image;
  rest the mouse on an image to see all of them.

**3D Plot**
: For a `.bild` file: the 3D coordinates it holds, drawn.

**Browser**
: For an `.html` file: opens it in your web browser.

### Looking at tables and text

**Text**
: Opens the file as text. It is the way to read a star file, or any text file Helicon recognizes.

**Stats**
: For the particle table of a RELION 3D classification or refinement: estimates the **variance of the
  tilt, psi and rot angles** of each filament and shows it, a quick check on the quality of the helical
  alignment.

**FSC**
: For the `model.star` of a RELION job: the **FSC curve** of a refinement, or the SSNR-derived W_MAP curve
  of a 3D classification. If the job folder holds several `model.star` files, you pick the iterations to
  overlay.

### Particles on micrographs

The preview of a RELION star file whose particles have positions on the micrographs (an extraction, a
selection, a classification, a refinement) has a button, **View particles on micrographs…**. So do the
listing of a picking job (`manualpick.star`, `autopick.star`), the coordinate file of one micrograph
(the window then opens at that micrograph), and the folder of a ManualPick, AutoPick or Extract job.
It opens a window with:

- on the left, the **list of the micrographs**, with the number of particles of each (for helical
  segments, of segments and filaments; for filaments picked by their start and end, of filaments). Click a
  column to sort; the *Filter micrographs* box limits the list. Below it, the total, and for a picking job
  how the picks were made, the *Micrographs read* box (see below) and, for filaments, the histogram of
  their lengths;
- on the right, the **selected micrograph with its picks** on it, one colour per filament. Single particles
  are circles as large as the particle diameter of the job, filaments picked by their start and end are
  bands as wide as the filament, and helical segments are dots. The *Show particles* box (*Show picks* for
  a picking job) hides them, to see the micrograph alone.

A micrograph that cannot be found is shown as the positions alone, on a plain background.

**Micrographs read.** A picking job of many micrographs is read from some of them, spread over the job:
3,000 unless set otherwise. The counts of the others are estimated from them ("about"), and the
histogram of the filament lengths and the list of the window show only those read. The
*Micrographs read* box, under the pictures of the preview of a picking job and under the list of its
window, sets how many (*All* reads every one). The number is kept, and used by both. A table of
particles is always read whole.

### Working on the data

**Images2Star**
: Opens a panel of tools, with a preview, to **select and transform the particles** of a star file (or a
  cryoSPARC `.cs` file), and saves the result as a new RELION star file. It is the window version of
  [`helicon images2star`](commands/images2star.md).

**Proc3D**
: Opens a panel of tools for a 3D map: flip the hand, clip, resample, denoise (*--denoiseCurvelet*) and
  more, with the result shown next to the original, and saved as a new MRC file. It is the window version of
  [`helicon proc3d`](commands/proc3d.md).

**TrueFSC**
: For the `model.star` of a RELION refinement: computes the **True FSC curve** from its two half maps
  (see [`helicon trueFSC`](commands/trueFSC.md)). *Optimize mask* searches the width of the soft edge of
  the mask, and its threshold too when the threshold is automatic; *Central Z (A)* or *Z fraction* keep only
  the central Z sections in the automatic mask, for a helical map. Besides the resolution, the
  *Information* reports the area under the True FSC curve, a measure of the whole curve.

### A folder of EPU XML files

**Beam Shift**
: Shown, with nothing selected, for a folder that holds the XML files EPU writes for its exposures
  (`FoilHole_*_Data_*.xml`), in it or in a folder below it: a copy of them all in one folder (often called
  `xml`), or an EPU session with its `Images-Disc1/GridSquare_*/Data` folders. It plots the **beam shift** of
  every exposure in the folder and all the folders below it, as a scatter plot: each hole of the pattern
  collected around a stage position is a cluster of points. Hover over a point for its file and beam
  shift. It is a check before grouping the exposures by beam shift
  (`--assignOpticGroupByBeamShiftXY` of [`helicon images2star`](commands/images2star.md)).

  The name of the folder does not count: it is the files that do. The top folder of a session is known
  by the `EpuSession.dm` that EPU leaves in it. Below a folder, only the folders of a session
  (`Images-Disc*`, the first grid squares, `Data`) are looked into, so that the check stays quick on a
  network disk: in a project with an `xml` folder somewhere inside, open that folder. The XML files of the survey images
  (`GridSquare_*.xml`, and `FoilHole_*.xml` without `_Data_`) are left out; XML files renamed by hand are
  taken if they are EPU `MicroscopeImage` records.

### Opening a web app

These buttons open one of the [Web apps](apps/index.md) in your browser, with the file already loaded. They
appear only for the files of a **helical** job (RELION marks it with `rlnIsHelix`, and Helicon knows
cryoSPARC's), because that is what the apps are for.

| Button | Offered for | Opens |
|---|---|---|
| **HelicalPitch** | the particle table (`*_data.star`) of a helical Class2D job, or the particle file of a helical cryoSPARC 2D classification (with its class averages beside it) | [HelicalPitch](apps/helicalpitch.md): the pitch and twist from the 2D classification |
| **AbInitio3D** | the same | [AbInitio3D](apps/abinitio3d.md): a 3D map from the 2D classes |
| **WhereIsMyClass** | the particle table of a helical Class2D job | [WhereIsMyClass](apps/whereismyclass.md): where the particles of a class are |
| **HelicalProjection** | the class averages (`*_classes.mrcs`) of a helical Class2D job | [HelicalProjection](apps/helicalprojection.md): compare with projections of known structures |
| **HILL** | the same | [HILL](apps/hill.md): indexing with Fourier layer lines |
| **denovo3D** | the same, and the particles of a helical cryoSPARC job | [Denovo3D](apps/denovo3d.md): indexing and a 3D map from one image |
| **HI3D** | a 3D map of a helical Class3D or Refine3D job, or of a helical cryoSPARC job | [HI3D](apps/hi3d.md): indexing from the cylindrical projection of the map |

### Beside the list

**Open RELION**
: In the path bar, when you are in a RELION project: opens the project in RELION. Set the RELION
  program up once with *Apps* → *Configure RELION…*.

## Tips

- The **Apps** menu has a *Terminal*, *Configure RELION…*, and the other Helicon programs.
- Press the *refresh* button after a job has finished writing files.
- To keep a folder: *Bookmarks* → *Bookmark This Folder* (`Ctrl+D`). *File* → *Recent Folders* lists the
  last ones you opened.
- *File* → *Quit* (`Ctrl+Q`) closes all the windows of Helicon. The size and place of each window, and the
  panes, columns and sort of the browser, are saved and come back the next time.
- The first image viewer or gallery window takes a few seconds to appear, because the viewer is
  loaded when it is first needed. Opening further files is faster.
