# Helicon

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)

Tools for cryo-EM analysis of helical structures, such as amyloid filaments and
other helical assemblies. Helicon finds the helical parameters (twist, rise,
symmetry) from 2D images or 3D maps, relates 2D classes to the filaments they came
from, and reconstructs a 3D map from a single 2D image or from a set of 2D classes.

It comes as command-line programs, a file viewer, and a set of web apps that run in
your browser, on your own computer or on a cloud server. You can 
[try the web apps online](https://01a01bc0-00db-fb4f-bfe0-f0ba9f0dd427.share.connect.posit.cloud)
without installing anything.

<p align="center">
  <a href="https://01a01bc0-00db-fb4f-bfe0-f0ba9f0dd427.share.connect.posit.cloud">
    <img src="https://raw.githubusercontent.com/jianglab/helicon/main/docs/images/helicon-home.jpeg"
         alt="The Home tab of the Helicon web apps: a diagram of the helical data processing workflow, with one box for each app" width="720">
  </a>
  <br>
  <em>The Home tab of the web apps. In the app, hover over a box to see what that app does, and click it to open it.</em>
</p>

## Quick start

Helicon needs **Python 3.13 or newer**. Use a fresh virtual or conda environment, as
it pulls in many packages.

```sh
pip install "helicon[shiny]"     # the command-line programs and the web apps
helicon                          # opens the web apps, on their Home tab
helicon --help                   # lists all programs
```

To get the newest development code instead of the latest release:

```sh
pip install "helicon[shiny] @ git+https://github.com/jianglab/helicon"
```

### Choosing what to install

The plain `pip install helicon` gives the command-line programs. Add what you need in
square brackets, separated by commas, e.g. `helicon[shiny,gui]`.

| Extra | Adds |
|---|---|
| `shiny` | the web apps (`helicon webApps`, or just `helicon`) |
| `gui` | the `display` file viewer (napari, PySide6, pyqtgraph) |
| `streamlit` | the Streamlit apps (`ctfSimulation`, `map2seq`, `procart`) |
| `all` | `shiny`, `streamlit`, `gui` and the curvelet extras |
| `test` | what the test suite needs, for developers |

### Keeping up to date

```sh
helicon --version        # the release and the git commit it was built from
helicon update --check   # is there a newer release?
helicon update           # install it
```

Only releases count, not the commits between them. `helicon update` also works for a
git checkout (`pip install -e`): it moves the checkout forward to the newest release
tag, and stops if you have uncommitted changes.

## What is included

### Web apps

Start them with `helicon` (or `helicon webApps`). Each tab works on its own.

| Tab | What it does |
|---|---|
| **AbInitio3D** | Reconstruct a 3D map from a set of 2D classes, with the twist found by a phase fit |
| **Denovo3D** | De novo helical indexing and 3D reconstruction from a single 2D image |
| **HelicalPitch** | Determine the helical pitch and twist from 2D classification results |
| **HILL** | Helical indexing with Fourier-Bessel layer lines (power spectra and phase differences) |
| **HI3D** | Helical indexing by cylindrical projection of a 3D map |
| **HelicalProjection** | Compare 2D images with projections of helical structures from the EMDB |
| **HelicalLattice** | Convert between a 2D lattice and a helical lattice |
| **WhereIsMyClass** | Map 2D classes back to the helical tube/filament images they contain |

### Command-line programs

| Program | What it does |
|---|---|
| `cryosparc` | Talk to a CryoSPARC server and run image analysis tasks |
| `images2star` | Analyze and transform datasets and save them as a RELION star file |
| `proc3d` | Analyze and transform 3D maps |
| `trueFSC` | Compute the True FSC curve with an optimal mask and phase randomization |
| `update` | Update Helicon to the newest release |

### Viewer and other apps

| Program | Needs | What it does |
|---|---|---|
| `display` | `gui` | A file browser for images, maps, star, bild, eps, pdf, html and text files |
| `ctfSimulation` | `streamlit` | Simulate 1D and 2D TEM contrast transfer functions |
| `map2seq` | `streamlit` | Find the protein sequence that best explains a 3D density map |
| `procart` | `streamlit` | Plot cartoon illustrations of residue properties of amyloid atomic models |

Run any of them with `-h` for its options, for example `helicon images2star -h`.

## Documentation

- [Documentation at Read the Docs](https://helicon.readthedocs.io): for users
- [Helicon on DeepWiki](https://deepwiki.com/jianglab/helicon): for developers
- [Project page](https://jianglab.science.psu.edu/helicon/)

## Citation

If Helicon is useful in your work, please cite:

> Li, D., Zhang, X., Jiang, W., 2025. Helicon: Helical parameter determination and 3D
> reconstruction from one image. *Journal of Structural Biology* 217, 108256.
> [doi:10.1016/j.jsb.2025.108256](https://doi.org/10.1016/j.jsb.2025.108256)

```bibtex
@article{li2025helicon,
  author  = {Li, D. and Zhang, X. and Jiang, W.},
  title   = {Helicon: Helical parameter determination and 3D reconstruction from one image},
  journal = {Journal of Structural Biology},
  volume  = {217},
  pages   = {108256},
  year    = {2025},
  doi     = {10.1016/j.jsb.2025.108256}
}
```

## Development

```sh
git clone https://github.com/jianglab/helicon.git
cd helicon
pip install -e ".[all,test]"
pytest
```

The version comes from the release tags (`v2026.10`), so `helicon --version` in a
checkout shows the commits since the last tag and the commit hash. See
[AGENTS.md](AGENTS.md) for the code layout and conventions. Bugs and feature requests
go to the [issue tracker](https://github.com/jianglab/helicon/issues).

## License

[MIT](LICENSE)
