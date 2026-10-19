# update

Update Helicon to the newest release.

See [Install](../install.md#keeping-up-to-date): `helicon update --check` tells whether there is
a newer release, and `helicon update` installs it.

## Options

```text
usage: helicon update [-h] [--check] [--to <release>] [--yes]

A command that updates helicon to the newest release (a version tag such as v2026.10)

options:
  -h, --help      show this help message and exit
  --check         only report whether a newer release exists, change nothing
  --to <release>  update to this release (for example v2026.10) instead of the newest
  --yes           do not ask before updating

Only releases count, not the commits between them. The newest release is looked up on PyPI
when helicon is published there, otherwise among the version tags of the public GitHub
repository. How it is then installed depends on how helicon was installed: from PyPI, from a
git URL, or as a git checkout (pip install -e).
```
