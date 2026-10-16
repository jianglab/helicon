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
                      [--denoiseCurvelet [<sigma=<float>[:numScales=<3>][:wedgesPerDir=<3>][:gpu=<true|false>][:transform=<udct|mct>][:]tileSize=<int>[:overlap=<32>]>]]
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
  --denoiseCurvelet [<sigma=<float>[:numScales=<3>][:wedgesPerDir=<3>][:gpu=<true|false>][:transform=<udct|mct>][:]tileSize=<int>[:overlap=<32>]>]
                        apply curvelet-based denoising to the 3D volume (UDCT or MCT).
                        Defaults: sigma=3.0, numScales=3, wedgesPerDir=3, gpu=false,
                        transform=mcttileSize=0 auto-detects from available memory. Requires
                        curvelets package. disabled by default
  --fft_resample <param>=<val>:...
                        fft_resample parameter
  --flip_hand <param>=<val>:...
                        flip_hand parameter
  --helical_sym <param>=<val>:...
                        helical_sym parameter
  --z_moving_average <param>=<val>:...
                        apply a moving average filter along the z-axis
```
