# proc3d

Analyze and transform 3D maps.

Reads a 3D map in MRC format, applies the operations you give (change the pixel size, clip,
resample in Fourier space, flip the hand, impose helical symmetry, average along z, denoise, and more), and
writes the map. The file browser offers it for a map (see [The file browser](../display.md)).

```sh
helicon proc3d input.mrc output.mrc [options]
```

## Options

```text
usage: helicon proc3d [-h] [--force <0|1>] [--verbose <0|1|2>] [--cpu <n>]
                      [--apix <param>=<val>:...] [--clip <param>=<val>:...]
                      [--denoiseCurvelet [[strength=<auto|x>][:halfMaps=<half1,half2>][:precision=<single|double>][:tileSize=<auto|n>]]]
                      [--fft_resample <param>=<val>:...] [--flip_hand <param>=<val>:...]
                      [--helical_sym <param>=<val>:...] [--z_moving_average <param>=<val>:...]
                      <inputMapFile> [<outputMapFile>]

A command line tool that anaylzes/transforms 3D maps

positional arguments:
  <inputMapFile>        input 3D map file in MRC format
  <outputMapFile>       output 3D map file (optional, auto-generated from input if omitted)

options:
  -h, --help            show this help message and exit
  --force <0|1>         force overwrite the output file. default to 0
  --verbose <0|1|2>     verbose mode. default to 2
  --cpu <n>             number of cpus to use. default to -1 and use all idle cpus
  --apix <param>=<val>:...
                        apix parameter
  --clip <param>=<val>:...
                        clip parameter
  --denoiseCurvelet [[strength=<auto|x>][:halfMaps=<half1,half2>][:precision=<single|double>][:tileSize=<auto|n>]]
                        denoise the 3D map automatically: monogenic curvelets (sheets and
                        lines) and wavelets (spots), with every sub-band's threshold set from
                        its own noise, averaged. strength scales all thresholds. auto
                        (default): tuned so that denoised half map 1 best predicts half map 2
                        when the half maps are found by their RELION, cryoSPARC or EMDB names
                        next to the input map (or given as halfMaps) and this is the first
                        option; otherwise 1.5, the best on amyloid maps (others wanted up to
                        6); or a number. Tuning takes 4-7 extra denoising runs (one fewer when
                        the map is half map 1). When the map is a half map or the average of
                        the two, also FSC-weighted or sharpened (but not masked), every sub-
                        band's noise is measured on their difference rather than guessed from
                        the map: closer to the truth, with the best strengths near 1.
                        precision: single (default) gave the same maps as double, 2-3 times
                        faster in half the memory. tileSize: auto (default) denoises the whole
                        map when it fits in the free memory, otherwise in the largest tiles
                        that fit; n uses tiles of n voxels a side, 0 the whole map. CPU count
                        from --cpu. disabled by default
  --fft_resample <param>=<val>:...
                        fft_resample parameter
  --flip_hand <param>=<val>:...
                        flip_hand parameter
  --helical_sym <param>=<val>:...
                        helical_sym parameter
  --z_moving_average <param>=<val>:...
                        apply a moving average filter along the z-axis
```
