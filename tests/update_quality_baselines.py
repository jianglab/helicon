"""Refresh ``quality_baselines.json`` from the current code.

    python tests/update_quality_baselines.py [scenario ...]

Run it after a change that is meant to move a number (better or worse, knowingly), and read
the diff of the file before committing it: that diff is the record of what the change did to
the accuracy of the methods. With scenario names, only those are refreshed.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = HERE / "quality_baselines.json"


def main(names: list[str]) -> int:
    sys.path.insert(0, str(HERE))
    import quality_scenarios as q

    unknown = [n for n in names if n not in q.SCENARIOS]
    if unknown:
        print(f"unknown scenario {unknown}; the scenarios are {list(q.SCENARIOS)}")
        return 2
    old = json.loads(BASELINES.read_text()) if BASELINES.exists() else {}
    new = dict(old)
    for name in names or list(q.SCENARIOS):
        for key, value in q.SCENARIOS[name]().items():
            new[key] = round(float(value), 4)
            before = old.get(key)
            print(f"{key:48s} {before!s:>10} -> {new[key]:>10}")
    BASELINES.write_text(json.dumps(dict(sorted(new.items())), indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
