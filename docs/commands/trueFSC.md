# trueFSC

Compute the True FSC curve with an optimal mask and phase randomization.

Computes the True FSC curve of two half maps, with a mask made from the maps and the phases
randomized beyond a resolution, so that the effect of the mask is taken out. The plot is written as a pdf,
png or jpeg file.

```sh
helicon trueFSC half_map_1.mrc half_map_2.mrc fsc.pdf [options]
```

## Options

```text
usage: helicon trueFSC [-h] [--apix <Angstrom/pixel>] [--cutoffRes <Angstrom>]
                       [--oneMask <0|1>] [--showPlot <0|1>] [--maskFile [<filename> ...] |
                       --maskFractionThresh <0-1> | --maskThresh [<pixel value> ...] |
                       --maskMass <kDa>] [--maskZlen <Angstrom> | --maskZfraction <0-1>]
                       [--maskSoft <Angstrom> | --refineMask <0|1>] [--force <0|1>]
                       [--verbose <0-2>]
                       <map 1> <map 2> [<plot file>]

A command line tool to compute True FSC curve with optimal mask and phase randomization

positional arguments:
  <map 1>               input half-map 1
  <map 2>               input half-map 2
  <plot file>           output plot file (pdf, png, jpeg format). Default: trueFSC.pdf

options:
  -h, --help            show this help message and exit
  --apix <Angstrom/pixel>
                        pixel size (Angstrom/pixel). Read from map header if not provided
  --cutoffRes <Angstrom>
                        starting resolution for phase randomization. Default: FSC=0.8 of
                        unmasked maps
  --oneMask <0|1>       use the same mask for both maps (1) or separate masks (0)
  --showPlot <0|1>      show plots on screen (1) or not (0)
  --maskFile [<filename> ...]
                        mask file(s)
  --maskFractionThresh <0-1>
                        mask threshold as fraction of max pixel value
  --maskThresh [<pixel value> ...]
                        mask threshold pixel value
  --maskMass <kDa>      total mass of structure in kDa
  --maskZlen <Angstrom>
                        keep only the central Z sections of this length (Angstrom) in the
                        automatic mask, for helical maps whose ends are poorer. Default: all
  --maskZfraction <0-1>
                        keep only this central fraction of the Z sections in the automatic
                        mask, for helical maps. Default: 1 (all)
  --maskSoft <Angstrom>
                        mask slope width in Angstrom. Default: auto
  --refineMask <0|1>    optimize the mask (0 or 1): its edge width, and its threshold when
                        that is automatic. Default: 1 (ignored if --maskSoft is set)
  --force <0|1>         delete the cached mask searches and search again (1). Default: 0
  --verbose <0-2>       verbose level (0-2). Default: 1
```
