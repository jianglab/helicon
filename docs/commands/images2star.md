# images2star

Analyze and transform datasets and save them as a RELION star file.

The workhorse for particle datasets. It reads the particles of RELION (`.star`), cryoSPARC
(`.cs`) and other programs, applies the operations you ask for (selecting, splitting, changing or
copying parameters, assigning optics groups, working on helical filaments, and more), and writes a RELION
star file.

```sh
helicon images2star input.star output.star [options]
helicon images2star J20/J20_particles.cs output.star     # a cryoSPARC file becomes a star file
```

To bring cryoSPARC particles to RELION, including helical segments, read
[Converting cryoSPARC files to RELION star files](../cryosparc-to-relion.md).

## Options

```text
usage: helicon images2star [-h] [--csparcPassthroughFiles <filename> [<filename> ...]]
                           [--first <n>] [--last <n>] [--subset <n>] [--sets <n>]
                           [--splitNumSets <n>]
                           [--splitMode <evenodd|random|micrograph|helicaltube>]
                           [--ignoreBadParticlePath <0|1|2|3>]
                           [--ignoreBadMicrographPath <0|1>] [--tag <str>]
                           [--folder [<dirname> ...]] [--verbose <0|1|2>] [--cpu <n>]
                           [--micrographStar <starfile>]
                           [--joinFilaments <0|1>[:epsilon=<1.0>]] [--force <0|1>]
                           [--ppid <n>] [--addParm <var> <val> <var> <val>] [--apix <A/pixel>]
                           [--assignOpticGroupByBeamShiftLabel <no|auto|EPU|serialEM_pncc>]
                           [--assignOpticGroupByBeamShiftXY 0|1|xml_folder=<path>:min_micrographs_per_group=<n>]
                           [--assignOpticGroupByTime <n>]
                           [--assignOpticGroupPerMicrograph <0|1>]
                           [--averagePowerSpectra groupby=<col1,col2,...>:fftX=<nx>:fftY=<ny>:batchSize=<n>:cpu=<n>:outdir=<path>]
                           [--breakFilaments maxSegments=<50>]
                           [--calibratePixelSize <graphene|graphene_oxide|go|gold|ice>]
                           [--copyCtf <starfile>]
                           [--copyParm <starfile< var ~var ...>> [<starfile< var ~var ...>> ...]]
                           [--createStack output.mrcs:rescale2size=<n>:float16=<0|1>:force=<0|1>]
                           [--delParm <var> [<var> ...]]
                           [--denoiseCurvelet [[transform=<mct|udct|fdct>[:sigma=<float>][:numScales=<auto>][:wedgesPerDir=<3>][:gpu=<true|false>][:tileSize=<N>][:overlap=<N>][:outdir=<path>]]]]
                           [--duplicateParm <from> <to> <from> <to>]
                           [--estimateHelicalAngleVariance <0|1>]
                           [--estimateHelicalTubeLength <0|1>]
                           [--excludeFile starFile:col1=<name>:col2=<name>:pattern=<str>]
                           [--extractHelices width=300:outPath=./helicon.helices/:topLength=<10>[:topLengthFraction=<0.1>][lengthCutoffAngst=<1000>]]
                           [--fullStack <0|1>] [--keepOneParticlePerHelicalTube <0|1>]
                           [--keepOneParticlePerMicrograph <0|1>]
                           [--keepParm <var> [<var> ...]]
                           [--maskGold value_sigma=<n>:gradient_sigma=<Å>:min_area=<Å^2>:both_sides=<0|1>:outdir=<str>:force=<0|1>:cpu=<n>]
                           [--minDuplicates <n>] [--minStack <0|1>]
                           [--multParm <var> <val> <var> <val>]
                           [--normEulerDist <angle bin size> <nkeep>]
                           [--path <absolute|relative|real|shortest|current>]
                           [--process processor_name:param1=value1:param2=value2 [processor_name:param1=value1:param2=value2 ...]]
                           [--psiPrior180 <0|1>] [--randomSample <n>]
                           [--recoverFullFilaments minFraction=<0.5>[:forcePickJob=<0|1>][:fullStarFile=<filename>][:matchBy=<image|ends|coords>][:maxDistance=<pixels>]]
                           [--removeDuplicates <var> [<var> ...]]
                           [--renameParm <old> <new> <old> <new>]
                           [--replaceImageName <new mrcs file>]
                           [--replaceStr <var> <original str> <new str>]
                           [--resetInterSegmentDistance <Å>] [--resetOpticGroup <0|1>]
                           [--rsortby <parameter> [<parameter> ...]]
                           [--saveStartEnd <dirname>] [--select <var> <val1<,val2>...>]
                           [--selectByParticleLocation starFile:maxDist=<pixel>]
                           [--selectFile starFile:col1=<name>:col2=<name>:pattern=<str>]
                           [--selectCommonHelices starFile]
                           [--selectRatioRange <var> <ratio min> <ratio max>]
                           [--selectValueRange <var> <valmin> <valmax>]
                           [--setBeamTiltClass <0|1>] [--setCTF <filename>]
                           [--setParm <var> <val> [<var> <val> ...]] [--showTime <attr>]
                           [--sortby <parameter> [<parameter> ...]]
                           [--splitByMicrograph <0|1>] [--summary2D] [--jobs JOB [JOB ...]]
                           input_imageFiles [input_imageFiles ...] output_starFile

A command line tool that analyzes/transforms dataset(s) and saves the dataset in a RELION star
file

positional arguments:
  input_imageFiles      input image file(s), or a Class2D folder with --summary2D
  output_starFile       output star file name, or PDF with --summary2D

options:
  -h, --help            show this help message and exit
  --csparcPassthroughFiles <filename> [<filename> ...]
                        input cryosparc v2 passthrough file(s)
  --first <n>           first image to process. default to 0
  --last <n>            last image to process. default to the last image in the file
  --subset <n>          which subset to keep. default to 0
  --sets <n>            number of subsets to split into. default to 1
  --splitNumSets <n>    number of subsets to split into. default to 1
  --splitMode <evenodd|random|micrograph|helicaltube>
                        how to split image set
  --ignoreBadParticlePath <0|1|2|3>
                        ignore bad particle image file path: 1-check but ignore missing files,
                        2-skip file checking, 3-skip file checking and recursive tracing of
                        lst files. default: 0
  --ignoreBadMicrographPath <0|1>
                        ignore bad micrograph image file path. default: 1
  --tag <str>           add this tag to new binary image files
  --folder [<dirname> ...]
                        Search these folders if the images cannot be found. default: ""
  --verbose <0|1|2>     verbose mode. default to 2
  --cpu <n>             number of cpus to use. default to 1
  --micrographStar <starfile>
                        Reference RELION STAR file with rlnMicrographName entries. CryoSPARC
                        micrograph names are cleaned (hash + _patch_aligned_doseweighted
                        stripped) and mapped to matching entries from this STAR file.
  --joinFilaments <0|1>[:epsilon=<1.0>]
                        if multiple input files all have helical particles, join the filaments
                        on the same micrograph that lie on overlapping straight lines (RELION
                        start-end picking) into one helical tube, removing the particles that
                        are closer than half of the inter-box distance to particles kept from
                        a larger piece of the same tube. epsilon (pixels) is the tolerance of
                        the |ab|-|pa|-|pb|<epsilon test of point p on line segment ab. default
                        to 1
  --force <0|1>         force overwrite output images. default to 0
  --ppid <n>            Set the PID of the parent process, used for cross platform PPID
  --addParm <var> <val> <var> <val>
                        modify parameter: var+=val
  --apix <A/pixel>      set mag to have this sampling
  --assignOpticGroupByBeamShiftLabel <no|auto|EPU|serialEM_pncc>
                        assign images to optic groups according to beam shift labels in
                        filenames, one group per beam shift position. default to no
  --assignOpticGroupByBeamShiftXY 0|1|xml_folder=<path>:min_micrographs_per_group=<n>
                        assign images to optic groups by beam shift XY coordinates from EPU
                        FoilHole XML files. Requires EPU_old FoilHole XML files. disabled by
                        default
  --assignOpticGroupByTime <n>
                        assign images to optic groups according to data collection time, n
                        movies per group. disabled by default
  --assignOpticGroupPerMicrograph <0|1>
                        assign images to optic groups, one group per micrograph. default to 0
  --averagePowerSpectra groupby=<col1,col2,...>:fftX=<nx>:fftY=<ny>:batchSize=<n>:cpu=<n>:outdir=<path>
                        Compute average 2D power spectra (and phase differences) for particle
                        groups. Saves ``.mrcs`` files per group. Example:
                        ``--averagePowerSpectra groupby=class:fftX=512:fftY=1024``
  --breakFilaments maxSegments=<50>
                        break long filaments into shorter filaments with at most maxSegments
                        segments per filament. Backs up original rlnHelicalTubeID to
                        rlnHelicalTubeIDOriginal
  --calibratePixelSize <graphene|graphene_oxide|go|gold|ice>
                        calibrate pixel size for common samples. default no
  --copyCtf <starfile>  Star file to copy CTF parameters from. Should be a CTF refinement
                        output file.
  --copyParm <starfile< var ~var ...>> [<starfile< var ~var ...>> ...]
                        copy the specified parameters or all parameters if no var is
                        specified. ~var will skip copying var
  --createStack output.mrcs:rescale2size=<n>:float16=<0|1>:force=<0|1>
                        create a new mrcs file to store all particles
  --delParm <var> [<var> ...]
                        remove parameter var for each image
  --denoiseCurvelet [[transform=<mct|udct|fdct>[:sigma=<float>][:numScales=<auto>][:wedgesPerDir=<3>][:gpu=<true|false>][:tileSize=<N>][:overlap=<N>][:outdir=<path>]]]
                        apply curvelet-based denoising to particle images in parallel, or
                        micrographs when rlnImageName is absent. Transform defaults to MCT.
                        Use tileSize=N to enable tiled processing (parallel within each large
                        image). Defaults: transform=mct, sigma=3, numScales=auto,
                        wedgesPerDir=3, gpu=false, outdir=./denoised/, overlap=32. CPU count
                        from --cpu flag. Requires curvepy-fdct (fdct) or curvelets (udct/mct).
                        disabled by default
  --duplicateParm <from> <to> <from> <to>
                        duplicate parameter
  --estimateHelicalAngleVariance <0|1>
                        estimate the variance of the tilt, psi, rot angles of segments in the
                        same helical tube/filament
  --estimateHelicalTubeLength <0|1>
                        estimate the length of each helical filament/tube
  --excludeFile starFile:col1=<name>:col2=<name>:pattern=<str>
                        exclude images whose <col1> is present in the specified star file's
                        <col2>. Example:
                        ref.star:col1=rlnImageName:col2=rlnImageName:pattern=(.+)
  --extractHelices width=300:outPath=./helicon.helices/:topLength=<10>[:topLengthFraction=<0.1>][lengthCutoffAngst=<1000>]
                        extract helical filaments from input files. disabled by default
  --fullStack <0|1>     generate a star stack including all particles in the referenced image
                        files. default to 0
  --keepOneParticlePerHelicalTube <0|1>
                        keep only one segment of each helical filament/tube
  --keepOneParticlePerMicrograph <0|1>
                        keep only one segment of each micrograph
  --keepParm <var> [<var> ...]
                        keep parameter var for each image, remove other parameters
  --maskGold value_sigma=<n>:gradient_sigma=<Å>:min_area=<Å^2>:both_sides=<0|1>:outdir=<str>:force=<0|1>:cpu=<n>
                        mask out electron dense (gold, ferritin, ice) pixels in images.
                        disabled by default
  --minDuplicates <n>   only keep images >=n duplicates. disabled by default
  --minStack <0|1>      generate a new set of mrcs files including only the subset of
                        particles in this stack. default to 0
  --multParm <var> <val> <var> <val>
                        modify parameter: var*=val
  --normEulerDist <angle bin size> <nkeep>
                        reduce Euler (view) preference by removing the worst particles in
                        over-populated angular bins
  --path <absolute|relative|real|shortest|current>
                        which type of file path is used for the images. default to current
  --process processor_name:param1=value1:param2=value2 [processor_name:param1=value1:param2=value2 ...]
                        no longer supported: it needed EMAN2's image processors.
  --psiPrior180 <0|1>   duplicate data by adding 180 degrees to rlnAnglePsiPrior
  --randomSample <n>    take random n images per rlnRandomSubset. disabled by default
  --recoverFullFilaments minFraction=<0.5>[:forcePickJob=<0|1>][:fullStarFile=<filename>][:matchBy=<image|ends|coords>][:maxDistance=<pixels>]
                        recover the whole filaments if current filament has >=minFraction of
                        the segments in the whole filament. matchBy selects how the particles
                        are matched to the filaments of the full star file: image (default) by
                        rlnImageName, requiring the particles to be a subset of the full star
                        file; ends (default if forcePickJob=1) by the start and end particles
                        of each filament lying on a filament of the full star file; coords by
                        the nearest particle (within maxDistance pixels, default to the inter-
                        box distance of the full star file) on the same micrograph, for
                        particles extracted separately from the full star file, e.g. from
                        multiple picking jobs whose filaments were joined into the full star
                        file (images2star --joinFilaments --saveStartEnd). In the coords mode,
                        the fraction of a filament is that of its particles matched by any
                        particle
  --removeDuplicates <var> [<var> ...]
                        remove images with duplicate parameters. disabled by default
  --renameParm <old> <new> <old> <new>
                        rename parameter
  --replaceImageName <new mrcs file>
                        replace rlnImageName column by the provided mrcs file that has the
                        same number of particles
  --replaceStr <var> <original str> <new str>
                        replace substr in the variable with new str
  --resetInterSegmentDistance <Å>
                        reset inter-segment distance by adding/removing 'particles' with
                        updated 'rlnCoordinateX rlnCoordinateY rlnHelicalTrackLengthAngst'
                        parameters. Warning: the output star file is meaningful only for
                        particle extraction and it should NOT be used for 2d/3d classification
                        or 3d refinement
  --resetOpticGroup <0|1>
                        reset all optics groups to a single group. disabled by default
  --rsortby <parameter> [<parameter> ...]
                        reverse sort (large to small) by the specified parameter(s). disabled
                        by default
  --saveStartEnd <dirname>
                        save the start-end coordinates of the helical filaments to this
                        folder, in the format of the output of a RELION ManualPick job: one
                        <micrograph>_manualpick.star file per micrograph and the coordinate
                        list file <dirname>/manualpick.star. Use the coordinate list file as
                        the input coordinates of a RELION Extract job (with 'Coordinates are
                        start-end only?' = Yes) to re-extract the filaments with a new inter-
                        box distance, e.g. after joining the filaments from multiple input
                        files. The filaments are saved as they are at the position of this
                        option among the other options
  --select <var> <val1<,val2>...>
                        select images with exact matching of the specified variable value(s).
                        disabled by default
  --selectByParticleLocation starFile:maxDist=<pixel>
                        select particles that are at the same locations in the micrograph
                        (example: x.star:maxDist=10). disabled by default
  --selectFile starFile:col1=<name>:col2=<name>:pattern=<str>
                        select images whose <col1> is present in the specified star file's
                        <col2>. Example:
                        ref.star:col1=rlnImageName:col2=rlnImageName:pattern=(.+)
  --selectCommonHelices starFile
                        select helices in the specified file (example: x.star) based on
                        rlnMicrographName and rlnHelicalTubeID. disabled by default
  --selectRatioRange <var> <ratio min> <ratio max>
                        select images with the variable value in the specified ratio range.
                        disabled by default
  --selectValueRange <var> <valmin> <valmax>
                        select images with the variable value in the specified range. disabled
                        by default
  --setBeamTiltClass <0|1>
                        set rlnBeamTiltClass column, one group per micrograph
  --setCTF <filename>   set ctf parameters stored in this file (EMAN1 ctfparm.txt) to the
                        output star file
  --setParm <var> <val> [<var> <val> ...]
                        set parameter var val pair for each image
  --showTime <attr>     include file create time of the attr in the output star file. disabled
                        by default
  --sortby <parameter> [<parameter> ...]
                        sort (small to large) by the specified parameter(s). disabled by
                        default
  --splitByMicrograph <0|1>
                        split the output into separate star files, one per micrograph. default
                        to 0
  --summary2D           write a PDF gallery of the latest iteration of successful RELION
                        Class2D jobs
  --jobs JOB [JOB ...]  jobs to summarize: folder names/aliases, numbers, or inclusive numeric
                        ranges (e.g. --jobs job001 good_classes or --jobs 1,3,5-9). Default:
                        all jobs; requires --summary2D
```
