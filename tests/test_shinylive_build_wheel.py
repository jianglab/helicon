"""tools/shinylive/build.py's build_wheel() picks the right wheel, even
across repeated calls into the same out_dir at different checkout versions.

Fast and shinylive-free (unlike tests/test_shinylive_build.py): build_wheel()
only ever shells out to `pip wheel`, which this fakes, so there is nothing
here that needs the shinylive package or a real build.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "shinylive"))
import build as shinylive_build  # noqa: E402


def _fake_pip_wheel(version, out_dir):
    """A subprocess.run replacement: writes a wheel named for *version*,
    touched to a fresh mtime, as `pip wheel` would."""

    def run(cmd, **kwargs):
        (out_dir / f"helicon-{version}-py3-none-any.whl").touch()

    return run


class TestBuildWheel:
    def test_returns_the_one_wheel_it_just_built(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            shinylive_build.subprocess, "run", _fake_pip_wheel("1.0.dev1", tmp_path)
        )
        wheel = shinylive_build.build_wheel(out_dir=tmp_path)
        assert wheel == tmp_path / "helicon-1.0.dev1-py3-none-any.whl"

    def test_a_stale_wheel_from_an_earlier_version_is_removed(
        self, tmp_path, monkeypatch
    ):
        # as when the checkout's setuptools-scm version changed between two
        # calls into the same out_dir (a commit, or just the working tree
        # changing): an earlier call's wheel is still sitting there
        stale = tmp_path / "helicon-1.0.dev1-py3-none-any.whl"
        stale.touch()

        monkeypatch.setattr(
            shinylive_build.subprocess, "run", _fake_pip_wheel("1.0.dev2", tmp_path)
        )
        wheel = shinylive_build.build_wheel(out_dir=tmp_path)

        assert wheel == tmp_path / "helicon-1.0.dev2-py3-none-any.whl"
        assert not stale.exists()
        assert list(tmp_path.glob("helicon-*.whl")) == [wheel]

    def test_several_stale_wheels_are_all_removed(self, tmp_path, monkeypatch):
        for v in ("1.0.dev1", "1.0.dev2", "1.0.dev3"):
            (tmp_path / f"helicon-{v}-py3-none-any.whl").touch()

        monkeypatch.setattr(
            shinylive_build.subprocess, "run", _fake_pip_wheel("1.0.dev4", tmp_path)
        )
        wheel = shinylive_build.build_wheel(out_dir=tmp_path)

        assert list(tmp_path.glob("helicon-*.whl")) == [wheel]

    def test_rebuilding_the_same_version_still_returns_it(self, tmp_path, monkeypatch):
        # pip wheel overwrites a same-named file rather than erroring on it;
        # build_wheel must not mistake that for "nothing built"
        existing = tmp_path / "helicon-1.0.dev1-py3-none-any.whl"
        existing.touch()

        monkeypatch.setattr(
            shinylive_build.subprocess, "run", _fake_pip_wheel("1.0.dev1", tmp_path)
        )
        wheel = shinylive_build.build_wheel(out_dir=tmp_path)

        assert wheel == existing
        assert list(tmp_path.glob("helicon-*.whl")) == [wheel]
