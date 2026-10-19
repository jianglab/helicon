# display

A file browser for images, maps, star, bild, eps, pdf, html and text files.

(Needs `helicon[gui]`; see [Install](../install.md).)

See [The file browser](../display.md) for what the window shows and how to use it.

## Options

```text
usage: helicon display [-h] [<folder>]

A file browser for viewing image, map, star, bild, eps, pdf, html, and text files.

positional arguments:
  <folder>    folder to browse (default: current directory)

options:
  -h, --help  show this help message and exit

This module is the coordinator for the ``helicon display`` command. The implementation lives
in ``helicon.lib.gui``; every moved name is re-exported here so existing callers that import
from ``helicon.commands.display`` keep working unchanged.
```
