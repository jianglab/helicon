# Development guide

For people who change Helicon. Users: see the [README](../README.md).

## Set-up

```sh
git clone https://github.com/jianglab/helicon.git
cd helicon
conda env create -f environment.yml && conda activate helicon   # or: pip install -e ".[all,test]" pytest-xdist
pytest -n auto -m "not slow and not browser"   # the everyday run, a few minutes
pytest -n auto                                 # everything (browser tests need `playwright install chromium`)
pre-commit install --hook-type pre-commit --hook-type commit-msg   # Black, and the names of private data
```

## Private data

Keep what identifies private or unpublished data out of the history (once pushed, it stays):
home and cluster folders, RELION job folders with a three-digit job number, CryoSPARC project and
job numbers, and your own user name and host name. The hooks, CI, and `python tools/check_private_names.py
--range origin/main..HEAD` refuse such a message, path or added line; nothing to set up. A line
about EMPIAR, or one that ends in `private-ok`, is skipped. Names that no rule can know can go,
one a line, in `~/.config/helicon/private_names.txt` (outside the repository; optional).

## Tests

Test behaviour, not source text: a test that looks for a string in a source file breaks with
every harmless refactor. To test the layout of a web-app tab, render its page and ask what a
user would see (`tests/_ui.py`: `page(tab_ui).comes_before(...)`, `.columns(...)`,
`.same_row(...)`).

`tests/test_quality_baselines.py` keeps the accuracy of the main methods on synthetic data with
known truth (`tests/quality_scenarios.py`; the numbers are in `tests/quality_baselines.json`). A
change that makes one worse by more than its tolerance fails. After a change that is meant to
move a number, run `python tests/update_quality_baselines.py` and commit the diff of the file
with the change; a new method gets a scenario there.

The tests that take 8 s or more are marked `slow` (the list is `tests/slow_tests.txt`,
made by `tests/update_slow_tests.py`), and those that drive a web page are marked
`browser`. A test run has its own settings folder and never opens a window on your
screen. On GitHub, every push runs Black, the fast tests and an install with no extras,
and the whole suite runs every night.

The version comes from the release tags (`v2026.10`), so `helicon --version` in a
checkout shows the commits since the last tag and the commit hash. Bugs and feature
requests go to the [issue tracker](https://github.com/jianglab/helicon/issues).
