"""Take the screenshot of the file browser (``helicon display``) for the user documentation.

    python docs/tools/screenshot_display.py

Makes a small RELION-like project in a temporary folder (the 2D classes of the public data set
EMPIAR-10940, and a short particle table), opens the file browser on its Class2D job without
showing it on the screen (Qt's offscreen platform) and saves ``docs/images/display.png``.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# the file browser remembers bookmarks and recent folders: a picture for the documentation must
# not show the ones of the person who takes it, so it runs with a settings folder of its own
_SETTINGS = tempfile.mkdtemp(prefix="helicon-docs-settings-")
os.environ["XDG_CONFIG_HOME"] = _SETTINGS
os.environ["XDG_DATA_HOME"] = _SETTINGS

IMAGES = Path(__file__).resolve().parent.parent / "images"
CLASSES = (
    "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/"
    "run_it020_classes.mrcs"
)


def _project(root: Path) -> Path:
    job = root / "Class2D" / "job010"
    job.mkdir(parents=True)
    (root / "default_pipeline.star").write_text(
        "data_pipeline_general\n\n_rlnPipeLineJobCounter 11\n"
    )
    urllib.request.urlretrieve(CLASSES, job / "run_it020_classes.mrcs")
    # what marks a RELION job as helical
    (job / "run_it020_model.star").write_text("data_model_general\n\n_rlnIsHelix 1\n")
    rows = []
    for i in range(300):
        rows.append(
            f"{i % 40 + 1:06d}@Extract/job009/mic{i // 30 + 1}.mrcs mic{i // 30 + 1}.mrc "
            f"{100.0 + 4 * (i % 30):.1f} {200.0:.1f} {i // 30 + 1} {i % 40 + 1} 0.0 0.0 0.0 "
            f"-1 {i % 30}"
        )
    (job / "run_it020_data.star").write_text(
        "data_particles\n\nloop_\n_rlnImageName\n_rlnMicrographName\n_rlnCoordinateX\n"
        "_rlnCoordinateY\n_rlnHelicalTubeID\n_rlnClassNumber\n_rlnAnglePsi\n_rlnOriginXAngst\n"
        "_rlnOriginYAngst\n_rlnAngleTiltPrior\n_rlnHelicalTrackLengthAngst\n"
        + "\n".join(rows)
        + "\n"
    )
    return job


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from helicon.lib.gui import file_browser as fb

    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "myproject"
        root.mkdir()
        job = _project(root)
        widget = fb.FolderBrowserWidget(start_dir=str(job))
        widget.resize(1280, 720)
        widget.show()

        def pump(seconds: float) -> None:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                app.processEvents()
                time.sleep(0.01)

        pump(1.5)
        # the particle table of the Class2D job: the preview shows its columns, and the action bar
        # offers the apps that fit a 2D classification
        for row in range(widget._model.rowCount()):
            if widget._model.item(row, 0).text() == "run_it020_data.star":
                widget._tree.setCurrentIndex(widget._model.index(row, 0))
        widget.wait_for_selection()
        widget._preview.wait()
        pump(4)
        IMAGES.mkdir(exist_ok=True)
        widget.grab().save(str(IMAGES / "display.png"))
        widget._preview.wait()
        widget.close()
    print(f"wrote {IMAGES / 'display.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
