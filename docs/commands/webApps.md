# webApps

Start the web apps (the same as `helicon` alone).

(Needs `helicon[shiny]`; see [Install](../install.md).)

Starts the web apps and opens them in your browser. See [Getting started](../getting-started.md) and
[the Web apps](../apps/index.md).

## Options

```text
usage: helicon webApps [-h] [--reload]

A Helicon Web app with eight analytical tools as tabs: whereIsMyClass, helicalProjection,
HILL, helicalPitch, denovo3D, abinitio3D, helicalLattice, and HI3D

options:
  -h, --help  show this help message and exit
  --reload    Restart the app when its source files change (for developing the app). Off by
              default: the reloader imports the whole app twice, which doubles the start-up
              time
```
