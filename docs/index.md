# Helicon

**Helicon** is a set of tools for cryo-EM analysis of helical structures, such as amyloid
filaments and other helical assemblies. It finds the helical parameters (twist, rise,
symmetry) from 2D images or 3D maps, relates 2D classes to the filaments they came from, and
reconstructs a 3D map from a single 2D image or from a set of 2D classes.

It comes as web apps that run in your browser (on your own computer or on a server), a file
viewer, and command-line programs.

```{image} images/home.png
:alt: The Home tab of the Helicon web apps
:width: 90%
:align: center
```

## Where to start

- **New to Helicon?** [Install it](install.md), then follow [Getting started](getting-started.md).
- **Which app do I need?** See [the Web apps](apps/index.md): one page for each, with a short
  walk-through on a public data set.
- **Looking at your files?** [The file browser](display.md) shows images, maps, star files and more,
  and opens the right app for a 2D classification.
- **Moving particles from cryoSPARC to RELION?** [Convert `.cs` files to star files](cryosparc-to-relion.md), including helical segments.
- **Scripting?** [The command-line programs](commands/index.md) and their options.
- **Something is unclear or not working?** See [Notes and answers](notes.md), or [report a problem or ask for a
  feature](report-a-problem.md).

```{toctree}
:hidden:
:caption: Using Helicon

install
getting-started
apps/index
display
cryosparc-to-relion
commands/index
notes
report-a-problem
```

```{toctree}
:hidden:
:caption: Project

Source code and issues <https://github.com/jianglab/helicon>
Project page <https://jianglab.science.psu.edu/helicon/>
Contributing <https://github.com/jianglab/helicon/blob/main/docs/DEVELOPMENT.md>
```

## Citation

If Helicon is useful in your work, please cite:

> Li, D., Zhang, X., Jiang, W., 2025. Helicon: Helical parameter determination and 3D
> reconstruction from one image. *Journal of Structural Biology* 217, 108256.
> [doi:10.1016/j.jsb.2025.108256](https://doi.org/10.1016/j.jsb.2025.108256)
