# Command-line programs

`helicon` runs the programs below. Run any of them with `-h` for its options, for example
`helicon images2star -h`; the options on the pages are what that prints. A program that needs an
[extra](../install.md) is marked; without it the program is not in `helicon --help`.

```sh
helicon --help               # the programs that are installed
helicon <program> --help     # the options of one
```

The options of `images2star`, `proc3d` and `cryosparc` that change the data are applied one after
the other, in the order given, and the same option may be given more than once, each time with
its own value.

| Program | What it does | Needs |
|---|---|---|
| [images2star](images2star.md) | Analyze and transform datasets and save them as a RELION star file |  |
| [proc3d](proc3d.md) | Analyze and transform 3D maps |  |
| [cryosparc](cryosparc.md) | Talk to a CryoSPARC server and run image analysis tasks |  |
| [trueFSC](trueFSC.md) | Compute the True FSC curve with an optimal mask and phase randomization |  |
| [update](update.md) | Update Helicon to the newest release |  |
| [webCalEM](webCalEM.md) | Open the web app WebCalEM, which calibrates the magnification (pixel size) of a TEM |  |
| [webApps](webApps.md) | Start the web apps (the same as `helicon` alone) | `shiny` |
| [display](display.md) | A file browser for images, maps, star, bild, eps, pdf, html and text files | `gui` |
| [ctfSimulation](ctfSimulation.md) | Simulate 1D and 2D TEM contrast transfer functions | `streamlit` |
| [map2seq](map2seq.md) | Find the protein sequence that best explains a 3D density map | `streamlit` |
| [procart](procart.md) | Plot cartoon illustrations of residue properties of amyloid atomic models | `streamlit` |

```{toctree}
:hidden:

images2star
proc3d
cryosparc
trueFSC
update
webCalEM
webApps
display
ctfSimulation
map2seq
procart
```
