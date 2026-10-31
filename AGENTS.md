# Helicon — Agent Guide

## CLI system

- Entry: `helicon` command — routes to `commands/<name>.py` via `helicon.py:main`
- Each command module must export `add_args(parser)` and `main(args)`. Optionally `check_args(args, parser)` for validation.
- Commands are registered in the group lists at the top of `helicon.py`.
- Commands missing from these lists won't appear in `helicon --help`.
- Shiny/streamlit commands are hidden if the optional dependency is missing (`helicon.has_shiny()` / `helicon.has_streamlit()`).

## Plugin architecture

CLI options of `cryosparc.py`, `images2star.py` and `proc3d.py` are handled by auto-discovered modules under `plugins/<category>/` (each exports `option_name` and `handle`); see `plugins/<category>/__init__.py` for the registry and `dispatch()`.

### Repeated options are a pipeline (do not remove)

`cryosparc`, `images2star` and `proc3d` run their plugin options as an ordered
pipeline: options are applied in the order given, and the same option may be
given more than once, each occurrence carrying its own value.

- Every plugin option of these commands is declared `action="append"`, so
  `args.<dest>` holds one value per occurrence in command-line order.
  `check_args` collects them in `args.append_options` (dests whose parser
  action is `_AppendAction`) and `args.all_options` (`plugin_options_in_argv`:
  the plugin dests in argv order, repeats included).
- The dispatch loop (`<command>.main`, `proc3d_engine.apply_options`,
  `images2star_engine.apply_options`) takes occurrence *n*'s value from the
  append dest and owns the count: it assigns `index_d[option_name] = n + 1`
  after each dispatch, so a handler that also bumps `index_d` cannot make the
  pipeline skip a value. Nothing reads `index_d`; it is only passed through
  the handler signature.
- The GUI builds the same pipeline: `stack_to_namespace` accumulates repeated
  options, and the widget passes its specs' append dests to `apply_options`.
  Repeating an option is refused only when the option is declared non-append,
  i.e. genuinely single-valued (some `images2star` options taking a list).

None of this is dead code: dropping `args.append_options`, the per-occurrence
lookup or the append declarations makes every repeat apply the last value to
all occurrences.

## Web app bookmarks

Every tab of `helicon webApps` is bookmarked the same way (`webApps/bookmark.py`):
the URL is `?tab=<Tab>&<short key>=<value>...`, carrying only the parameters
that differ from their defaults. A tab declares its parameters once, in
`BOOKMARK_DEFAULTS` (`short key: (input id, default[, bookmark.DERIVED])`); the
page script, the restore and the file-browser launchers all come from it. The
default must be what the input shows on the tab's example data; mark values
the tab computes from the loaded data `DERIVED`. Do not parse the query string
in a tab.

## Build & install

- Use conda env `helicon`

## Tests

- Tests use pytest with plain class grouping (no `unittest.TestCase` inheritance) and plain `assert` statements.
- Everyday run: `pytest tests -n auto -m "not slow and not browser"` (needs `pytest-xdist`).
  The whole suite, in parallel, is `pytest tests -n auto`; run in turn it takes about 17
  minutes, in parallel about 7.
- `slow` marks the tests of 8 s or more, listed in `tests/slow_tests.txt`; refresh the
  list from a run with `--durations=0` using `tests/update_slow_tests.py`. `browser` marks
  the tests that drive a web page with Playwright (`playwright install chromium`).
- `tests/conftest.py` gives the run its own `QSettings` folder and sets
  `QT_QPA_PLATFORM=offscreen` unless it is set: a test must never change the user's saved
  settings or open a window on their screen. A test that needs its own settings file can
  still patch `QSettings` as before.
- Test behaviour, not source text: asserting that a source file contains a string breaks
  with every harmless refactor. To test the layout of a tab, render its page and ask what the
  user would see (`tests/_ui.py`: `page(tab_ui).comes_before(...)`, `.columns(...)`,
  `.same_row(...)`); a source check is for what only the code shows (a rule about every
  decorated function, say).
- `tests/test_quality_baselines.py` keeps the accuracy of the main methods on synthetic data
  with known truth (`tests/quality_scenarios.py`, numbers in `tests/quality_baselines.json`).
  A change that makes one worse by more than its tolerance fails; after a change that is meant
  to move a number, run `python tests/update_quality_baselines.py` and commit the diff of the
  file with the change. A new method gets a scenario there.
- CI (`.github/workflows/ci.yml`): Black, ruff and the private-data check, the fast tests,
  the docs build (`-W`) and a no-extras install on every push; the whole suite nightly. Install
  the local hooks once with `pre-commit install --hook-type pre-commit --hook-type commit-msg`
  (the second checks commit messages for private names, `tools/check_private_names.py`).

**Policy:** every new option added to `proc3d.py`, `images2star.py`, or `cryosparc.py` must be coupled with tests covering both argparse wiring and handler logic. Use in-memory data (numpy arrays, DataFrames, `MockDataset`) rather than file I/O.

## Lint & format

**Always match Black's formatting.**

CI fails on a single file that Black would change, so check before every commit, with the
version CI pins (`.pre-commit-config.yaml`, now 24.8.0), from the `helicon` env:

```
python -m black --check src tests tools && python -m ruff check src tests tools
```

- Use `python -m black`: a bare `black` may not be on `PATH`, or may be another version
  that formats differently. Format only the files you changed (`python -m black <files>`);
  a run over the whole tree with another version rewrites files you did not touch.
- Black leaves a long trailing comment alone but wraps the code before it into an ugly
  `x = (\n {}\n)  # comment`: put a long comment on its own line above the code.
  It also moves the closing `"""` of a long one-line docstring to its own line.
- Check again after merging or pulling other people's commits: their files may not pass.
- `.git/hooks/pre-commit` (from `pre-commit install`) does nothing unless it is executable
  (`chmod +x`); git only prints a hint when it skips it.

## Path handling

Use `pathlib.Path` instead of `os.path` for all filesystem path operations
(join, exists, basename, dirname, splitext, relpath, etc.). pathlib is
already imported as `Path` in most modules.

**Exception:** `os.path.expandvars()` has no pathlib equivalent — keep it
when expanding environment variables in paths.

## Imports and load time

**Delay imports as much as possible.** Start-up time is a feature: `helicon --help`,
`helicon display` and `helicon webApps` should show something quickly.

- Import heavy packages (numba, SciPy submodules, matplotlib, bokeh,
  IPython/ipywidgets/shinywidgets, Qt dialogs, plugin packages) inside the function that
  uses them, not at the top of the module.
- A package `__init__` must not import all its submodules: use a module-level
  `__getattr__` (see `webApps/lib/__init__.py`, `lib/gui/__init__.py`) so that
  `from pkg import sub` and `pkg.sub` still work.
- `helicon.py` imports only the command that is run; the others are listed from the
  docstrings in their source files (`_source_doc`).
- A web app tab that needs heavy modules is imported when first opened (`_LAZY_TABS` in
  `webApps/app.py`); what the page needs at start-up (its `BOOKMARK_DEFAULTS`) lives in a
  small module of its own (`<tab>_bookmark.py`).
- numba: decorate with `lib/numba_compat.lazy_jit("njit", ...)` (compiled on the first
  call), never import numba at module level; `import helicon` must not load numba
  or h5py (`tests/test_numba_compat.py` checks it).
- Worker processes started from the web app use the `forkserver` context with the
  module preloaded (`layerline_index._pool_context`): spawned workers re-import helicon
  each, and forking a threaded server can hang. Keep heavy imports of a script that
  starts a pool inside `if __name__ == "__main__":`, or every worker re-imports them.
- Check what a change saves with `python -X importtime -c "import helicon.webApps.app"`
  (or the module concerned); wall-clock times on a busy computer are noisy.

## Numba notes

- The package works without numba but runs slower. Imports are guarded.
- Thread count controlled via `helicon.available_cpu()` (respects SLURM, numba config, memory).
- `lib/system.py` also contains OpenMP helper functions.

## Releases

Use the `release` skill (`.claude/skills/release/SKILL.md`) for PyPI publishing.

## Docstring style

Use **NumPy-style** docstrings everywhere (the `Parameters` / `Returns` / `Raises` sections with underlined headers).

## Git workflow

**AI should NOT commit or push changes unless explicitly told to do so by the user.**
