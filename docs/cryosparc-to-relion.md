# Converting cryoSPARC files to RELION star files

CryoSPARC keeps the particles of a job in a `.cs` file (a NumPy table with one row for each particle).
RELION uses a `.star` file. `helicon images2star` reads a `.cs` file and writes a star file, with the
names and conventions of RELION. The same command does the three jobs below; what differs is where the
particles, and above all their **helical** information, come from.

| Your particles | Go to |
|---|---|
| any particles: a 2D classification, a refinement, the picks | [1. Generic conversion](#generic) |
| **helical segments extracted in cryoSPARC** | [2. Helical segments from cryoSPARC](#helical-cryosparc) |
| segments **extracted in RELION, imported into cryoSPARC, and exported back** | [3. Segments that went through cryoSPARC](#helical-roundtrip) |

You need `helicon` (any [install](install.md)); this needs no extra. Run the commands in a terminal.

(generic)=
## 1. Generic format conversion

```sh
helicon images2star J20/J20_particles.cs J20_particles.star
```

The first argument is the `.cs` file of the job (the output `*_particles.cs` of a cryoSPARC job), the
second the star file to write. The `.cs` file of a refinement or a classification keeps part of its
information in a second *passthrough* file, which holds what the job took over from earlier jobs: the CTF,
the location in the micrograph, and so on. Helicon finds it by itself when it is in the same folder as the
`.cs` file (`J20_passthrough_particles.cs`, or `J20_passthrough_<name>.cs` for a file such as
`particles_selected.cs`). If it is somewhere else, or there are several, name them:

```sh
helicon images2star J20_particles.cs J20_particles.star \
    --csparcPassthroughFiles J20_passthrough_particles.cs
```

If the CTF is missing from the result, Helicon says *"CTF info not found. You should also provide the
passthrough file that has CTF info"*: give that file.

**What is converted.** The particle image (`blob/path` and `blob/idx` become `rlnImageName`), the
micrograph and the movie, the pixel size, the CTF (defocus, astigmatism angle, voltage, spherical
aberration, amplitude contrast, phase shift, B-factor and scale, beam tilt and anisotropic
magnification), the location in the micrograph (`rlnCoordinateX` and `Y`), the 2D class and its
alignment (class number, shift, in-plane angle), the 3D alignment (the pose becomes the three Euler
angles `rlnAngleRot`, `Tilt` and `Psi`, the shift becomes `rlnOriginXAngst` and `Y`), the half set
(`rlnRandomSubset`), and the exposure groups, which become RELION optics groups. The trefoil and
tetrafoil of the CTF are not converted: Helicon warns about it when they are not zero, and RELION's CtfRefine
can fit them.

**Image files.** A star file for RELION must name stacks that end in `.mrcs`. CryoSPARC writes `.mrc`
stacks: Helicon makes a `.mrcs` symbolic link next to each one (or, when it cannot write there, in a
folder `mrc2mrcs` beside the input file, else in the current folder) and uses the link in the star file. The images themselves are not
copied.

**Paths.** The `--path` option chooses how the files are named in the star file (by default, as they were read): `relative` (to the
RELION project folder of the output file, or to the folder of the star file), `absolute`, `real` (links
resolved), or `shortest`. Use `--path absolute` if you are not sure; use `--path relative` when you write
the star file inside the RELION project and the images are below it.

**Check the result.** Helicon prints a summary of what it read. For helical data it names the number of
segments, filaments and micrographs, the pixel size and the distance between segments. Open the star file in
[the file browser](display.md) to see the columns and the images.

(helical-cryosparc)=
## 2. Helical segments extracted in cryoSPARC

When the segments were picked and extracted in cryoSPARC (the *Filament Tracer* and *Extract From
Micrographs*), every particle in the `.cs` file knows its filament: `filament/filament_uid` says which filament it
is on, `filament/position_A` how far along the filament it is, and `filament/filament_pose` the direction of the
filament in the micrograph. The command is the same as in the first situation, and the helical columns of
RELION follow from these three fields:

```sh
helicon images2star J20/J20_particles.cs J20_particles.star
```

| From the `.cs` file | In the star file |
|---|---|
| `filament/filament_uid` | `rlnHelicalTubeID`, the filament number: renumbered 1, 2, … in each image file when cryoSPARC's numbers are large (as they usually are), otherwise kept as they are |
| `filament/position_A` | `rlnHelicalTrackLengthAngst`, the distance along the filament from its first segment, in Å |
| `filament/filament_pose` | `rlnAnglePsiPrior`, the in-plane direction of the filament, in degrees |
| | `rlnAngleRotPrior` = 0, `rlnAngleTiltPrior` = 90 and `rlnAnglePsiFlipRatio` = 0.5, the values RELION expects for filaments lying in the plane of the micrograph |

The location of the segments (`rlnCoordinateX`, `rlnCoordinateY`) comes from their fractional position in
the micrograph and the size of the micrograph, so the `.cs` file, or its passthrough file, must have the
micrograph shape; the usual outputs of a cryoSPARC extraction do.

Look at the first lines of the summary Helicon prints: *"Read in N segments (extracted with X Å
inter-segment shift) in M helices from K micrographs"*. M must be the number of filaments you expect, and X
the distance between the segments of your extraction; if M is one per micrograph or one per segment, the
filament information did not come across (check that the file is the extraction's or a later job's output,
not a table of picks). You can also check the columns:

```sh
grep -c "_rlnHelicalTubeID" J20_particles.star      # 1 when the column is there
```

The star file is ready for RELION's helical jobs (2D classification, 3D classification and
refinement of helical data). Set the helical parameters (tube diameter, rise, twist, number of
asymmetric units) in RELION as you would for segments extracted there.

(helical-roundtrip)=
## 3. Segments extracted in RELION, imported into cryoSPARC, and exported back

CryoSPARC's jobs do not write RELION's helical columns (`rlnHelicalTubeID`, the track length, the
priors), so the `.cs` file of a job that ran on particles imported from RELION cannot be relied on to carry
them. A conversion that looked only at that `.cs` file could give a star file that has lost the helical
information.

Helicon avoids this by going back to the star file that you imported. CryoSPARC keeps it in the import job
(`J1/particles.star`, with the particles in `J1/imported_particles.cs`). When the images named in the `.cs` file all
live in such import jobs, Helicon reads the **original star file** as the source of the particles, selects the ones
that are in your `.cs` file (the job may have kept only some of them), and puts the information that
cryoSPARC changed on top of it. The command is again the same:

```sh
helicon images2star J20/J20_particles.cs J20_particles.star
```

and Helicon says so:

```text
Detected .cs from RELION import. Using original STAR: .../J1/particles.star (total=…, selected=…)
```

What you get is the original star file with these changes, for the selected particles only:

- **kept from the original:** everything RELION wrote, including the helical columns (`rlnHelicalTubeID`,
  `rlnHelicalTrackLengthAngst`, the priors), the coordinates, the image and micrograph names, the optics
  table;
- **taken from cryoSPARC:** the 2D class (`rlnClassNumber`, `rlnMaxValueProbDistribution`) and 2D alignment
  (shift as `rlnOriginXAngst` and `Y`, in-plane angle as `rlnAnglePsi`) of a 2D classification; the 3D class,
  half set (`rlnRandomSubset`), score (`rlnLogLikeliContribution`), pose (the three Euler angles
  `rlnAngleRot`, `rlnAngleTilt` and `rlnAnglePsi`) and shift of a 3D classification or refinement; and the CTF
  values (defocus, astigmatism angle, phase shift, B-factor, scale). They are converted as in the first
  situation, so the star file has the alignment of your cryoSPARC job together with everything RELION knew
  about the particles.

If several imports were combined in one cryoSPARC job, the filament numbers of each import are shifted by a
multiple of 10, 100 or 1000 so that filament 1 of one import is not taken for filament 1 of another one on the
same micrograph.

**Another way, often the better one: select from your original star file.** If what you need from the
cryoSPARC job is *which particles it kept* (after a 2D classification in which you chose the good classes, or a
refinement that dropped particles), start from the original star file of the extraction and select the
particles that are in the `.cs` file:

```sh
helicon images2star original_extraction.star selected.star --selectFile J20/J20_particles.cs
```

`--selectFile` keeps the particles of the input star file whose images are also in the file you name. For a
`.cs` file of imported particles Helicon reads the file as above, through the star file that was imported,
so the images match those of your star file. The result is your original star file, row by row, for the
selected particles only: every column exactly as RELION wrote it (the helical columns, the CTF, the poses,
the optics groups), and no column that cryoSPARC changed. Use it when RELION should go on from its own values
with the cryoSPARC selection; use the route above when you want cryoSPARC's classes, alignment and CTF as well.

**When it does not apply.** The route needs the `.cs` file to be in its cryoSPARC project folder (so that Helicon
finds `J1/particles.star` next to it) and every particle image to belong to an import job. If you
*extracted* the particles again in cryoSPARC, the images are new and the situation is the
[second one](#helical-cryosparc). If you copied the `.cs` file out of the project, or the import job was
deleted, Helicon converts the `.cs` file alone. Then copy the helical
information from your original star file:

```sh
helicon images2star J20_particles.cs J20_particles.star \
    --copyParm original_extraction.star rlnHelicalTubeID rlnHelicalTrackLengthAngst \
        rlnAngleRotPrior rlnAngleTiltPrior rlnAnglePsiPrior rlnAnglePsiFlipRatio
```

`--copyParm` takes a star file and the names of the columns to copy (all of them when none is named; `~name`
leaves one out). It matches the particles by the image file and the position in it, so it needs the
particles to be the same images as in the original star file, and the original star file to have all of the
particles of the `.cs` file.
