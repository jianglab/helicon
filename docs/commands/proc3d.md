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
usage: helicon proc3d [-h] [--force <0|1>] [--verbose <0|1|2>] [--cpu <n>] [--apix <Å>]
                      [--clip <param>=<val>:...]
                      [--denoiseCurvelet [[strength=<auto|x>][:halfMaps=<half1,half2>][:precision=<single|double>][:tileSize=<auto|n>]]]
                      [--fft_resample <param>=<val>:...] [--flip_hand <x|y|z>]
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
  --apix <Å>            set the pixel size of the map (Å) without changing its voxels: to
                        correct a wrong header. Use --fft_resample to change the sampling
  --clip <param>=<val>:...
                        cut the map to a new box, or pad it with zeros where the box is
                        larger: new_nx, new_ny, new_nz (voxels; by default the current size)
                        around center_x, center_y, center_z (voxels; by default the centre of
                        the map). Example: --clip new_nx=200:new_ny=200
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
                        resample the map to new_nx, new_ny, new_nz voxels (by default the
                        current size) by cropping or padding its Fourier transform; the pixel
                        size changes to match (keep the three ratios equal for an even
                        sampling). Example: --fft_resample new_nx=128:new_ny=128:new_nz=128
  --flip_hand <x|y|z>   flip the hand of the map by mirroring it along this axis
  --helical_sym <param>=<val>:...
                        impose helical symmetry: twist (°), rise (Å) and csym (default 1) from
                        a central part of the map, given by exactly one of center_len (Å),
                        center_n_rise (number of rises) or center_fraction (of the map
                        length). Optional new_apix, new_nz and new_nxy (a square cross-
                        section) set the output sampling and box. Example: --helical_sym
                        twist=-1.2:rise=4.75:center_fraction=0.3
  --z_moving_average <param>=<val>:...
                        apply a moving average filter along the z-axis
```
