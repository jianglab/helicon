# cryosparc

Talk to a CryoSPARC server and run image analysis tasks.

Works on the `.cs` files of a cryoSPARC project, or on a running cryoSPARC server (it needs the
project and job ids, and your connection details), for example to assign exposure groups or to extract
particles again. To convert particles for RELION, use `helicon images2star` instead; see
[Converting cryoSPARC files to RELION star files](../cryosparc-to-relion.md).

## Options

```text
usage: helicon cryosparc [-h] [--csFile <filename> [<filename> ...]] [-p <pid>] [-j <jid>]
                         [-g <n>] [-w <wid>] [--saveLocal <0|1>] [--cpu <n>]
                         [--verbose <0|1|2>] [--assignExposureGroupByBeamShiftLabel 0|1]
                         [--assignExposureGroupByBeamShiftXY 0|1|xml_folder=<path>:min_micrographs_per_group=<n>]
                         [--assignExposureGroupByTime <n>]
                         [--assignExposureGroupPerMicrograph <0|1>]
                         [--changePixelSize <Angstrom>]
                         [--copyExposureGroupAssignments <star file>]
                         [--copyExposureGroupParameters source_cs_file=<filename>|source_job_id=<Jxx>[:beam_tilt=<0|1>:trefoil=<0|1>:tetrafoil=<0|1>:cs=<0|1>:anisomag=<0|1>]]
                         [--extractParticles box_size=<n>:fft_crop_size=<n>[:recenter=<0|1>][replace_ctf=<0|1>][normalize=<0|1>][fill_mode=<mean|random>][sign=<-1|1>][n_micrographs=<-1|n>][fp16=<0|1>][:micrographs_cs_file=<filename>|micrographs_job_id=<JXXX>][reuse_job_id=<JXXX>]]
                         [--resetExposureGroups <0|1>] [--splitByMicrograph <0|1>]

A command line tool that interacts with a CryoSPARC server and performs image analysis tasks

options:
  -h, --help            show this help message and exit
  --csFile <filename> [<filename> ...]
                        input local .cs file(s)
  -p, --projectID <pid>
                        CryoSPARC project ID
  -j, --jobID <jid>     CryoSPARC job ID(s)
  -g, --groupIndex <n>  output group index for each job (default: 0)
  -w, --outputWorkspaceID <wid>
                        output workspace ID (e.g. W1)
  --saveLocal <0|1>     save output data as local .cs file(s) instead of uploading to
                        CryoSPARC server
  --cpu <n>             number of CPUs. default to -1 (all available)
  --verbose <0|1|2>     verbose mode. default to 2
  --assignExposureGroupByBeamShiftLabel 0|1
                        assign images to exposure groups by beam shift label from filenames.
                        One group per distinct beam shift value. disabled by default
  --assignExposureGroupByBeamShiftXY 0|1|xml_folder=<path>:min_micrographs_per_group=<n>
                        assign images to exposure groups by beam shift XY coordinates from EPU
                        XML files. Requires EPU_old FoilHole XML files. disabled by default
  --assignExposureGroupByTime <n>
                        assign images to exposure groups according to data collection time, n
                        movies per group. disabled by default
  --assignExposureGroupPerMicrograph <0|1>
                        assign images to exposure groups, one group per micrograph. disabled
                        by default
  --changePixelSize <Angstrom>
                        change the pixel size to this value. Adjust defocus and Cs
                        accordingly. disabled by default
  --copyExposureGroupAssignments <star file>
                        copy the optics group assignments from this star file.
                        rlnMicrographMovieName and rlnOpticsGroup must be in this star file.
                        disabled by default
  --copyExposureGroupParameters source_cs_file=<filename>|source_job_id=<Jxx>[:beam_tilt=<0|1>:trefoil=<0|1>:tetrafoil=<0|1>:cs=<0|1>:anisomag=<0|1>]
                        copy exposure group parameters (beam tilt, trefoil, tetrafoil, cs,
                        anisotropic distortion, etc.). disabled by default
  --extractParticles box_size=<n>:fft_crop_size=<n>[:recenter=<0|1>][replace_ctf=<0|1>][normalize=<0|1>][fill_mode=<mean|random>][sign=<-1|1>][n_micrographs=<-1|n>][fp16=<0|1>][:micrographs_cs_file=<filename>|micrographs_job_id=<JXXX>][reuse_job_id=<JXXX>]
                        split the dataset by micrograph. disabled by default
  --resetExposureGroups <0|1>
                        reset all exposure groups to a single group. disabled by default
  --splitByMicrograph <0|1>
                        split the dataset by micrograph. disabled by default
```
