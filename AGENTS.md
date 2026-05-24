# Helicon — Agent Guide

## Project structure

```
src/helicon/
  __init__.py         # Version + public API re-exports
  helicon.py          # CLI entrypoint (argparse)
  commands/           # One .py per subcommand
  lib/                # Core library modules (well-factored)
  plugins/            # Auto-discovered handlers for cryosparc/images2star/proc3d
tests/
```

## CLI system

- Entry: `helicon` command — routes to `commands/<name>.py` via `helicon.py:main`
- Each command module must export `add_args(parser)` and `main(args)`. Optionally `check_args(args, parser)` for validation.
- Three command groups in `helicon.py:12-25`:

| Group | Commands |
|-------|----------|
| `cli_commands` | `cryosparc`, `images2star`, `proc3d`, `symmetry_mismatch` |
| `shiny_commands` | `denovo3D`, `helicalPitch`, `helicalProjection`, `whereIsMyClass` |
| `streamlit_commands` | `ctfSimulation`, `helicalLattice`, `hi3d`, `hill`, `map2seq`, `procart` |

- Commands missing from these lists won't appear in `helicon --help`.
- Shiny/streamlit commands are hidden if the optional dependency is missing (`helicon.has_shiny()` / `helicon.has_streamlit()`).
- Unregistered command files (exist but not in any list): `HOM_containerC.py`

## Plugin architecture

The `plugins/` directory contains auto-discovered handler modules for CLI options:

```
plugins/
  __init__.py              # Empty (marker)
  cryosparc/               # Handlers for helicon cryosparc --* options
    __init__.py            # Auto-discovers handlers + dispatch()
    extractparticles.py
    splitbymicrograph.py
    changepixelsize.py
    assignexposuregroup*.py
    copyexposuregroup*.py
    resetexposuregroups.py
  images2star/             # Handlers for helicon images2star --* options
    __init__.py            # Same plugin pattern as cryosparc
    apix.py, path.py, select.py, sets.py
    process.py, fullstack.py, minstack.py, createstack.py
    calibratepixelsize.py, estimatehelical*.py
    assignopticgroup*.py, setctf.py, copyctf.py
    addparm.py, setparm.py, multparm.py, copyparm.py, keepparm.py, delparm.py, renameparm.py, duplicateparm.py
    replacestr.py, replaceimagename.py
    removeduplicates.py, minduplicates.py
    randomsample.py
    selectcommonhelices.py, extracthelices.py
    keeponeparticlepermicrograph.py, keeponeparticleperhelicaltube.py
    resetintersegmentdistance.py
    normeulerdist.py, psiprior180.py
    selectvaluerange.py, selectratiorange.py
    maskgold.py
    showtime.py
    recoverfullfilaments.py
  proc3d/                  # Handlers for helicon proc3d --* options
    __init__.py
    apix.py, clip.py, flip_hand.py, z_moving_average.py
```

**Pattern:** Each plugin module exports:
- `option_name: str` — the CLI option name (e.g., `"--extract-particles"`)
- `handle(...)` — the handler function with signature matching the dispatcher
- Optional: `add_args(parser)` — to register CLI flags

**Discovery:** `plugins/<category>/__init__.py` uses `pkgutil.iter_modules()` to auto-discover all modules at import time and builds a registry keyed by `option_name`.

**Calling pattern** (in commands like `cryosparc.py`):
- `plugins.cryosparc.add_plugin_args(parser)` — add CLI args from all plugins
- `plugins.cryosparc.dispatch(option_name, data, args, ...)` — invoke matching handler

## Library modules (`lib/`)

The library has been refactored (Phase 1 complete from `REFACTORING_PLAN.md`):

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Re-exports selected public symbols |
| `analysis.py` | FSC, correlation metrics, elbow point, helix geometry, masks |
| `alignment.py` | Image alignment functions |
| `angular.py` | Angular difference, periodic range utilities |
| `cache.py` | Caching decorator, cache dir setup, `Timer`, `DotDict` |
| `clustering.py` | `AgglomerativeClusteringWithMinSize`, beam shift clustering |
| `collections.py` | List/array utilities: `flatten`, `unique`, `split_array`, etc. |
| `dataset.py` | `MockDataset`, `EMDBMirror`, `EMDBProject` |
| `epu.py` | EPU/SerialEM XML parsing, beam shift extraction, timestamps |
| `euler.py` | Euler angle ↔ quaternion conversions, RELION ↔ EMAN |
| `exceptions.py` | `HeliconError`, `HeliconExit` (proper exception hierarchy) |
| `filters.py` | Structural factors, low/high-pass, normalization, tapering, downsampling |
| `groups.py` | Per-micrograph mapping, timestamp extraction, group combination |
| `io.py` | STAR/CS/DataFrame I/O, conventions, filename normalization |
| `io_mrc.py` | MRC file utilities: axes order, orthoslice display, image size |
| `logging.py` | `get_logger`, `log_command_line`, `color_print` |
| `path_utils.py` | File path conversion, file readiness, EMDB ID extraction |
| `point_group.py` | Point group symmetry utilities |
| `ptycho.py` | Ptychography utilities (optional: `py4dstem`) |
| `shiny.py` | Shiny app helpers |
| `system.py` | CPU counting, OpenMP thread control, `available_cpu` |
| `transforms.py` | `apply_helical_symmetry`, FFT crop/rescale, rotations, hand flipping |
| `util.py` | Remaining utilities: option parsing, param validation, auto-install |

**Note:** `src/helicon/__init__.py` explicitly re-exports the most commonly used functions from these modules (no wildcard imports).

## Build & install

- Use conda env `helicon`

```sh
pip install -e ".[all]"         # editable install with shiny + streamlit extras
pip install -e ".[streamlit]"   # streamlit only
pip install -e ".[shiny]"       # shiny only
```

Package layout: `src/` directory (set in `pyproject.toml` `package-dir`).

## Tests

```sh
pytest tests/                      # all tests
pytest tests/test_io.py            # single file
pytest -k test_euler               # keyword match
```

- Tests use pytest with plain class grouping (no `unittest.TestCase` inheritance) and plain `assert` statements.
- Fixtures in `tests/conftest.py`: `star_df`, `cs_array`, `star_file`, `cs_file`, `clean_tmp_path`
- Property-based tests use `hypothesis` (see `test_angular_property.py`)
- Test coverage:

| Test file | Tests |
|-----------|-------|
| `test_io.py` | STAR/CS I/O |
| `test_cryosparc.py` | `helicon cryosparc` handlers + plugins |
| `test_images2star.py` | `helicon images2star` handlers + plugins |
| `test_proc3d.py` | `helicon proc3d` handlers + plugins |
| `test_symmetry_mismatch.py` | `symmetry_mismatch` command |
| `test_HOM_containerC.py` | `HelicalSegmentConsistency` |
| `test_denovo3D_*.py` | denovo3D app, solver, utils, pipeline |
| `test_whereIsMyClass_*.py` | whereIsMyClass app + compute |
| `test_filters.py`, `test_transforms.py`, `test_analysis.py` | Core lib modules |
| `test_angular_property.py` | Hypothesis-based property tests |
| `test_emdb_mirror.py`, `test_dataset.py` | Dataset/EMDB |
| `test_util.py`, `test_groups.py`, `test_point_group.py`, `test_shiny.py` | Other lib modules |

**Policy:** every new option added to `proc3d.py`, `images2star.py`, or `cryosparc.py` must be coupled with tests covering both argparse wiring and handler logic. Use in-memory data (numpy arrays, DataFrames, `MockDataset`) rather than file I/O.

## Lint & format

```sh
black src/helicon/ tests/                    # format everything
pre-commit run --all-files                   # same via pre-commit
```

Only `black` is configured (`.pre-commit-config.yaml`).

**Always match Black's formatting.**

## Key dependencies

From `pyproject.toml`:

**Required:**
- `starfile` — reading/writing RELION STAR files
- `cryosparc-tools` — CryoSPARC server interaction
- `mrcfile` — MRC image I/O
- `numba` — JIT acceleration (used in `transforms.py`, `io.py`)
- `finufft` — non-uniform FFT
- `numpy`, `scipy`, `pandas[html]`, `scikit-learn`, `scikit-image`
- `joblib`, `tqdm`, `rich`, `psutil`, `uptime`
- `pylops` — linear operators

**Optional extras:**
- `shiny`: `shiny`, `shinywidgets`, `plotly`, `jupyter_bokeh`, `ipywidgets`, `itk-montage`, `requests`, `ipyfilechooser`
- `streamlit`: `streamlit`, `streamlit-bokeh`, `streamlit-drawable-canvas`, `st-clickable-images`, `atomium`, `bokeh`, `kneebow`, `qrcode`, `shapely`, `trackpy`, `xmltodict`, `uptime`, `psutil`, `numpy`
- `ptycho`: `py4dstem` (not included in `[all]`)

## Version

Dynamic — read from `helicon/__init__.py:__version__` (currently `"2026.05"`).

`pyproject.toml` uses `[tool.setuptools.dynamic] version = {attr = "helicon.__version__"}`.

`__init__.py` also contains NumPy 2.x compatibility aliases for `numba`.

## Numba notes

- Used in `lib/transforms.py`, `lib/io.py`
- The package works without numba but runs slower. Imports are guarded.
- Thread count controlled via `helicon.available_cpu()` (respects SLURM, numba config, memory).
- `lib/system.py` also contains OpenMP helper functions.

## Exception hierarchy

Defined in `lib/exceptions.py`:
- `HeliconError(Exception)` — base exception for domain errors
- `HeliconExit(SystemExit)` — clean exit without error

Caught centrally in `helicon.py:104-115`.

## PyPI publishing

```sh
python -m build                    # build sdist + wheel (requires build package)
twine upload dist/helicon-*        # upload to PyPI
```

- Version comes from `helicon/__init__.py:__version__` — bump there before release.
- Update `HISTORY.md` with each release.

## Docstring style

Use **NumPy-style** docstrings everywhere (the `Parameters` / `Returns` / `Raises` sections with underlined headers).

Example:
```python
def func(param1: int, param2: str = "default") -> bool:
    """Short description.

    Parameters
    ----------
    param1 : int
        Description of param1.
    param2 : str, optional
        Description of param2. Defaults to ``"default"``.

    Returns
    -------
    bool
        Description of return value.
    """
```

## Memory system (session persistence)

- **Proactively memorize** context for seamless cross-session resumption:
  - Use `memory add` for project state, key decisions, current TODOs, and noteworthy findings
  - Use `memory profile` for user preferences (flags, format, workflow habits)
  - Tag memories with technical keywords for targeted `search` later
- **On session start**, run `memory list` or `memory search` to pick up where we left off
- **On session end**, save a summary of what was done and what's next

## Documentation files in repo

| File | Purpose |
|------|---------|
| `README.md` | Main project README |
| `HISTORY.md` | Release history |
| `AGENTS.md` | This file — agent/LLM coding guide |

## What's not here

- No CI workflows (no `.github/workflows/`).
- Sphinx docs configured but `docs/` directory is empty.
