"""Rewrite ``slow_tests.txt`` from the output of a run with ``--durations``.

    pytest tests -n 8 --durations=0 --durations-min=1.0 -p no:cacheprovider > durations.txt
    python tests/update_slow_tests.py durations.txt --min 8

The tests whose own time (the ``call`` phase) is at least ``--min`` seconds are listed,
by test id without its parameters (so every case of a parametrised test is marked), and
``conftest.py`` marks them ``slow``. Everyday and CI runs leave them out
(``pytest -m "not slow and not browser"``); the nightly run has them all.
"""

import argparse
import re
from pathlib import Path

LINE = re.compile(r"^\s*([0-9.]+)s\s+call\s+(\S+)")


def slow_tests(text: str, minimum: float) -> list[str]:
    """The ids (without parameters) of the tests that took at least ``minimum`` seconds."""
    found = set()
    for line in text.splitlines():
        m = LINE.match(line)
        if m and float(m.group(1)) >= minimum:
            found.add(m.group(2).split("[")[0])
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("durations", help="the output of pytest --durations=0")
    parser.add_argument("--min", type=float, default=8.0, help="seconds (default 8)")
    args = parser.parse_args()
    ids = slow_tests(Path(args.durations).read_text(errors="replace"), args.min)
    out = Path(__file__).with_name("slow_tests.txt")
    out.write_text(
        "# Tests marked slow (see update_slow_tests.py); one id per line, parameters left out.\n"
        + "".join(f"{i}\n" for i in ids)
    )
    print(f"{len(ids)} tests of {args.min:g} s or more written to {out}")


if __name__ == "__main__":
    main()
