"""Write ``docs/commands/``, the reference of the command-line programs, from their own help.

    python docs/tools/gen_commands.py            # rewrite the pages
    python docs/tools/gen_commands.py --check    # exit 1 if a page is out of date

There is one page for each program, and an index. The options of a program are what
``helicon <program> -h`` prints, so the reference cannot disagree with the program; the short text
above them says what the program is for and where to read more. Programs whose optional packages
are not installed are left out of the run; with ``--check`` only the programs that could be run are
compared. Regenerate after changing an option.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent.parent / "commands"

# program: (what it is for, the extra it needs, the text above its options)
PROGRAMS = {
    "images2star": (
        "Analyze and transform datasets and save them as a RELION star file",
        None,
        """The workhorse for particle datasets. It reads the particles of RELION (`.star`), cryoSPARC
(`.cs`) and other programs, applies the operations you ask for (selecting, splitting, changing or
copying parameters, assigning optics groups, working on helical filaments, and more), and writes a RELION
star file.

```sh
helicon images2star input.star output.star [options]
helicon images2star J20/J20_particles.cs output.star     # a cryoSPARC file becomes a star file
```

To bring cryoSPARC particles to RELION, including helical segments, read
[Converting cryoSPARC files to RELION star files](../cryosparc-to-relion.md).""",
    ),
    "proc3d": (
        "Analyze and transform 3D maps",
        None,
        """Reads a 3D map in MRC format, applies the operations you give (change the pixel size, clip,
resample in Fourier space, flip the hand, impose helical symmetry, average along z, denoise, and more), and
writes the map. The file browser offers it for a map (see [The file browser](../display.md)).

```sh
helicon proc3d input.mrc output.mrc [options]
```""",
    ),
    "cryosparc": (
        "Talk to a CryoSPARC server and run image analysis tasks",
        None,
        """Works on the `.cs` files of a cryoSPARC project, or on a running cryoSPARC server (it needs the
project and job ids, and your connection details), for example to assign exposure groups or to extract
particles again. To convert particles for RELION, use `helicon images2star` instead; see
[Converting cryoSPARC files to RELION star files](../cryosparc-to-relion.md).""",
    ),
    "trueFSC": (
        "Compute the True FSC curve with an optimal mask and phase randomization",
        None,
        """Computes the True FSC curve of two half maps, with a mask made from the maps and the phases
randomized beyond a resolution, so that the effect of the mask is taken out. The plot is written as a pdf,
png or jpeg file.

```sh
helicon trueFSC half_map_1.mrc half_map_2.mrc fsc.pdf [options]
```""",
    ),
    "update": (
        "Update Helicon to the newest release",
        None,
        """See [Install](../install.md#keeping-up-to-date): `helicon update --check` tells whether there is
a newer release, and `helicon update` installs it.""",
    ),
    "webCalEM": (
        "Open the web app WebCalEM, which calibrates the magnification (pixel size) of a TEM",
        None,
        """Opens <https://jianglab.github.io/WebCalEM/> in your browser. With `--printUrl` it only prints the
address.""",
    ),
    "webApps": (
        "Start the web apps (the same as `helicon` alone)",
        "shiny",
        """Starts the web apps and opens them in your browser. See [Getting started](../getting-started.md) and
[the Web apps](../apps/index.md).""",
    ),
    "display": (
        "A file browser for images, maps, star, bild, eps, pdf, html and text files",
        "gui",
        """See [The file browser](../display.md) for what the window shows and how to use it.""",
    ),
    "ctfSimulation": (
        "Simulate 1D and 2D TEM contrast transfer functions",
        "streamlit",
        """Starts a Streamlit app in your browser.""",
    ),
    "map2seq": (
        "Find the protein sequence that best explains a 3D density map",
        "streamlit",
        """Starts a Streamlit app in your browser.""",
    ),
    "procart": (
        "Plot cartoon illustrations of residue properties of amyloid atomic models",
        "streamlit",
        """Starts a Streamlit app in your browser.""",
    ),
}

INDEX = """# Command-line programs

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
{rows}

```{{toctree}}
:hidden:

{names}
```
"""


def help_of(program: str) -> str | None:
    env = dict(os.environ, COLUMNS="96", NO_COLOR="1")
    done = subprocess.run(
        [str(Path(sys.executable).parent / "helicon"), program, "-h"],
        capture_output=True,
        text=True,
        env=env,
    )
    if done.returncode != 0 or not done.stdout.strip():
        return None
    text = done.stdout.rstrip()
    # the usage line names the interpreter's script; show the name people type
    return re.sub(r"^usage: \S+", "usage: helicon", text, flags=re.M)


def build() -> tuple[dict[str, str], list[str]]:
    """The text of every page, by file name, and the programs that could not be run."""
    pages, missing = {}, []
    for program, (what, extra, intro) in PROGRAMS.items():
        text = help_of(program)
        if text is None:
            missing.append(program)
            continue
        needs = (
            f"\n\n(Needs `helicon[{extra}]`; see [Install](../install.md).)"
            if extra
            else ""
        )
        pages[f"{program}.md"] = (
            f"# {program}\n\n{what}.{needs}\n\n{intro}\n\n"
            f"## Options\n\n```text\n{text}\n```\n"
        )
    rows = [
        f"| [{p}]({p}.md) | {PROGRAMS[p][0]} | "
        + (f"`{PROGRAMS[p][1]}`" if PROGRAMS[p][1] else "")
        + " |"
        for p in PROGRAMS
    ]
    pages["index.md"] = INDEX.format(rows="\n".join(rows), names="\n".join(PROGRAMS))
    return pages, missing


def main(argv: list[str]) -> int:
    pages, missing = build()
    if missing:
        print(f"not run (missing packages?): {', '.join(missing)}", file=sys.stderr)
    if "--check" in argv:
        if missing:
            return 0  # cannot compare a partial run with the whole set
        old = (
            {p.name: p.read_text() for p in TARGET.glob("*.md")}
            if TARGET.exists()
            else {}
        )
        if old != pages:
            print(
                "docs/commands/ is out of date: run python docs/tools/gen_commands.py"
            )
            return 1
        return 0
    TARGET.mkdir(exist_ok=True)
    for stale in TARGET.glob("*.md"):
        if stale.name not in pages:
            stale.unlink()
    for name, text in pages.items():
        (TARGET / name).write_text(text)
    print(f"wrote {len(pages)} pages in {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
