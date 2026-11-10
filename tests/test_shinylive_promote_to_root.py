"""tools/shinylive/build.py's promote_to_root() makes an exported bundle's
own page the site's landing page, in place of write_index()'s plain link list
-- used for "home" only, so Home itself is what a visitor lands on.

Fast and shinylive-free (like test_shinylive_build_wheel.py): promote_to_root()
is plain file moving, nothing here needs a real export.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "tools" / "shinylive"))
import build as shinylive_build  # noqa: E402


class TestPromoteToRoot:
    def test_the_bundles_own_files_land_at_the_root(self, tmp_path):
        bundle_dir = tmp_path / "home"
        bundle_dir.mkdir()
        (bundle_dir / "index.html").write_text("<html>home</html>")
        (bundle_dir / "shinylive").mkdir()
        (bundle_dir / "shinylive" / "shinylive.js").write_text("// js")

        shinylive_build.promote_to_root(bundle_dir, out_dir=tmp_path)

        assert (tmp_path / "index.html").read_text() == "<html>home</html>"
        assert (tmp_path / "shinylive" / "shinylive.js").exists()

    def test_the_bundles_own_directory_is_removed_once_empty(self, tmp_path):
        bundle_dir = tmp_path / "home"
        bundle_dir.mkdir()
        (bundle_dir / "index.html").write_text("<html>home</html>")

        shinylive_build.promote_to_root(bundle_dir, out_dir=tmp_path)

        assert not bundle_dir.exists()

    def test_it_does_not_disturb_sibling_bundles_already_at_the_root(self, tmp_path):
        (tmp_path / "helical_lattice").mkdir()
        (tmp_path / "helical_lattice" / "index.html").write_text("sibling")
        bundle_dir = tmp_path / "home"
        bundle_dir.mkdir()
        (bundle_dir / "index.html").write_text("<html>home</html>")

        shinylive_build.promote_to_root(bundle_dir, out_dir=tmp_path)

        assert (tmp_path / "helical_lattice" / "index.html").read_text() == "sibling"
        assert (tmp_path / "index.html").read_text() == "<html>home</html>"

    def test_a_same_named_file_already_at_the_root_is_replaced(self, tmp_path):
        # a rebuild: the previous run's own landing page is already there
        (tmp_path / "index.html").write_text("stale")
        bundle_dir = tmp_path / "home"
        bundle_dir.mkdir()
        (bundle_dir / "index.html").write_text("<html>fresh</html>")

        shinylive_build.promote_to_root(bundle_dir, out_dir=tmp_path)

        assert (tmp_path / "index.html").read_text() == "<html>fresh</html>"
