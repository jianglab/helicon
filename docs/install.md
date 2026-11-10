# Install

Helicon needs **Python 3.13 or newer**. Use a fresh virtual or conda environment: Helicon pulls
in many packages, and a new environment keeps them apart from your other work.

## With conda

```sh
conda create -n helicon python=3.13
conda activate helicon
pip install "helicon[shiny]"
```

(Install conda from <https://docs.conda.io/en/latest/miniconda.html> if you do not have it.)

## With pip alone

```sh
python3.13 -m venv helicon-env
source helicon-env/bin/activate
pip install "helicon[shiny]"
```

Then check that it works:

```sh
helicon --version
helicon --help
```

## Choosing what to install

The plain `pip install helicon` gives only the command-line programs. Add what you need in
square brackets, separated by commas, for example `helicon[shiny,gui]`.

| Extra | Adds |
|---|---|
| `shiny` | the web apps (`helicon webApps`, or just `helicon`) |
| `gui` | the `display` file browser (napari, PySide6, pyqtgraph) |
| `streamlit` | the Streamlit apps (`ctfSimulation`, `map2seq`, `procart`) |
| `all` | `shiny`, `streamlit` and `gui` |

A program whose extra is missing is left out of `helicon --help`. Install the extra and it
appears.

## The newest development version

```sh
pip install "helicon[shiny] @ git+https://github.com/jianglab/helicon"
```

## Keeping up to date

```sh
helicon --version        # the release, and the git commit it was built from
helicon update --check   # is there a newer release?
helicon update           # install it
```

Only releases count, not the commits between them. `helicon update` also works for a git
checkout (`pip install -e`): it moves the checkout forward to the newest release tag, and stops
if you have uncommitted changes.

## Without installing anything

You can [try the web apps online](https://jianglab.github.io/helicon/)
on example data. It runs entirely in your browser (no server), so there is no server
filesystem to browse: upload a file, give a URL, or give an EMDB ID.
