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
    import math

    # three micrographs of five filaments, twenty segments each, spread over a 4k micrograph:
    # the preview draws them on the micrograph with the most filaments
    rows = []
    for i in range(300):
        mic, tube, k = i // 100 + 1, i // 20 + 1, i % 20
        angle = math.radians(25 * tube)
        x = 600 + 600 * (tube % 5) + 90 * k * math.cos(angle)
        y = 700 + 500 * (tube % 5) + 90 * k * math.sin(angle)
        rows.append(
            f"{i % 40 + 1:06d}@Extract/job009/mic{mic}.mrcs mic{mic}.mrc "
            f"{x:.1f} {y:.1f} {tube} {i % 40 + 1} 0.0 0.0 0.0 -1 {90.0 * k:.1f}"
        )
    (job / "run_it020_data.star").write_text(
        "data_particles\n\nloop_\n_rlnImageName\n_rlnMicrographName\n_rlnCoordinateX\n"
        "_rlnCoordinateY\n_rlnHelicalTubeID\n_rlnClassNumber\n_rlnAnglePsi\n_rlnOriginXAngst\n"
        "_rlnOriginYAngst\n_rlnAngleTiltPrior\n_rlnHelicalTrackLengthAngst\n"
        + "\n".join(rows)
        + "\n"
    )
    return job


PIPELINE = "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/default_pipeline.star"


def _pipeline_project(root: Path) -> None:
    """The jobs of the RELION project of EMPIAR-10940: its pipeline, and an empty folder for each job."""
    import re

    pipeline = root / "default_pipeline.star"
    urllib.request.urlretrieve(PIPELINE, pipeline)
    for line in pipeline.read_text().splitlines():
        m = re.match(r"^(\w+)/(job\d+)/\s", line)
        if m:
            job = root / m.group(1) / m.group(2)
            job.mkdir(parents=True, exist_ok=True)
            # what RELION leaves in a job folder, so that the job has files and a state
            state = {"Aborted": "ABORTED", "Failed": "FAILURE"}.get(
                line.split()[-1], "SUCCESS"
            )
            (job / f"RELION_JOB_EXIT_{state}").write_text("")
            (job / "note.txt").write_text(f"{m.group(1)}/{m.group(2)}/\n")
            (job / "job.star").write_text("data_job\n")
    # the outcome of one job: the class averages of its last iteration and their shares
    job = root / "Class2D" / "job010"
    urllib.request.urlretrieve(CLASSES, job / "run_it020_classes.mrcs")
    urllib.request.urlretrieve(
        CLASSES.replace("classes.mrcs", "model.star"), job / "run_it020_model.star"
    )


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

        # the jobs of a project as a graph, in its window, and the preview of a job
        project = Path(tmp) / "pipeline-project"
        project.mkdir()
        _pipeline_project(project)
        widget = fb.FolderBrowserWidget(start_dir=str(project / "Class2D" / "job010"))
        widget.resize(1280, 720)
        widget.show()
        pump(3)
        widget._preview.wait()
        pump(2)
        widget.grab().save(str(IMAGES / "display-job.png"))
        widget._navigate_to(str(project))
        pump(3)
        widget._preview.wait()
        widget._preview.open_graph_window()
        window = widget._preview._graph_window
        window.resize(760, 640)
        window.view.set_current(str(project / "Class2D" / "job010"))
        window.view.zoom(0.9)
        pump(3)
        # the pointer on a job: it, its inputs and the jobs that use its files are marked
        graph = window.view.graph
        window.view._set_hover(
            next(i for i, n in enumerate(graph.nodes) if n.key == "Refine3D/job017")
        )
        pump(1)
        window.grab().save(str(IMAGES / "display-job-graph.png"))
        window.close()
        widget._preview.wait()
        widget.close()
    print(f"wrote {IMAGES / 'display.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
