"""The EPU XML folders of ``helicon display``: finding them, reading their beam shifts,
and the Beam Shift button and plot of the file browser."""

import os

import pytest

from helicon.lib.epu import (
    find_epu_xml_files,
    is_epu_data_xml,
    is_epu_xml_folder,
    read_epu_beamshifts,
)

EPU_XML = """<?xml version="1.0" encoding="utf-8"?>
<MicroscopeImage xmlns="http://schemas.datacontract.org/2004/07/Fei.SharedObjects"
    xmlns:a="http://schemas.datacontract.org/2004/07/Fei.Types">
  <microscopeData>
    <optics>
      <BeamShift><a:_x>{x}</a:_x><a:_y>{y}</a:_y></BeamShift>
    </optics>
  </microscopeData>
</MicroscopeImage>
"""


def _data_xml(n: int) -> str:
    return (
        f"FoilHole_{28788144 + n}_Data_28764755_{n % 9}_20240328_19{n % 60:02d}16.xml"
    )


def _write(path, x=0.0, y=0.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EPU_XML.format(x=x, y=y))
    return path


@pytest.fixture
def xml_folder(tmp_path):
    """A folder of the exposure XML files copied together, as is often done."""
    folder = tmp_path / "xml"
    for n, (x, y) in enumerate([(0.1, 0.2), (-0.1, 0.2), (0.1, -0.2)]):
        _write(folder / _data_xml(n), x, y)
    return folder


@pytest.fixture
def session(tmp_path):
    """An EPU session: Images-Disc1/GridSquare_*/Data, with the survey XML files."""
    root = tmp_path / "session"
    (root / "EpuSession.dm").parent.mkdir(parents=True)
    (root / "EpuSession.dm").write_text("<EpuSessionXml/>")
    for g in (1, 2):
        square = root / "Images-Disc1" / f"GridSquare_{g}"
        _write(square / f"GridSquare_20240328_19{g}016.xml")
        _write(square / "FoilHoles" / f"FoilHole_2878814{g}_20240328_192016.xml")
        for n in range(2):
            _write(square / "Data" / _data_xml(10 * g + n), g, n)
    return root


class TestEpuDataXml:
    def test_exposure_xml(self):
        assert is_epu_data_xml("FoilHole_28788144_Data_28764755_46_20240328_192116.xml")
        assert is_epu_data_xml(
            "/a/b/FoilHole_1464933_Data_427288_427290_20250502_213110.xml"
        )

    def test_survey_and_other_files(self):
        assert not is_epu_data_xml("FoilHole_28788144_20240328_192116.xml")
        assert not is_epu_data_xml("GridSquare_20240328_192116.xml")
        assert not is_epu_data_xml(
            "FoilHole_28788144_Data_28764755_46_20240328_192116_fractions.tiff"
        )


class TestIsEpuXmlFolder:
    def test_folder_of_xml_files(self, xml_folder):
        assert is_epu_xml_folder(xml_folder)

    def test_session_and_its_folders(self, session):
        assert is_epu_xml_folder(session)
        assert is_epu_xml_folder(session / "Images-Disc1")
        assert is_epu_xml_folder(session / "Images-Disc1" / "GridSquare_1" / "Data")

    def test_epu_session_file(self, tmp_path):
        # the top folder of a session, before EPU has written an exposure, or on a disk
        # the XML files were not copied to: the plot says there is nothing to show
        (tmp_path / "EpuSession.dm").write_text("<EpuSessionXml/>")
        assert is_epu_xml_folder(tmp_path)

    def test_survey_xml_files_only(self, session):
        assert not is_epu_xml_folder(
            session / "Images-Disc1" / "GridSquare_1" / "FoilHoles"
        )

    def test_folder_name_does_not_count(self, tmp_path):
        (tmp_path / "xml").mkdir()
        (tmp_path / "xml" / "notes.xml").write_text("<notes/>")
        assert not is_epu_xml_folder(tmp_path / "xml")
        assert not is_epu_xml_folder(tmp_path)

    def test_renamed_exposure_xml_files(self, tmp_path):
        _write(tmp_path / "movie_0001.xml")
        assert is_epu_xml_folder(tmp_path)

    def test_other_folders_below_are_not_looked_into(self, tmp_path):
        # a project with a folder of XML files somewhere in it: only that folder has
        # the button, so that opening the project reads no more than its own listing
        _write(tmp_path / "xml" / _data_xml(0))
        _write(tmp_path / "a" / "Data" / _data_xml(1))
        assert not is_epu_xml_folder(tmp_path)
        assert is_epu_xml_folder(tmp_path / "xml")

    def test_only_the_first_grid_squares(self, tmp_path):
        disc = tmp_path / "Images-Disc1"
        for g in range(1, 4):
            (disc / f"GridSquare_{g}" / "Data").mkdir(parents=True)
        _write(disc / "GridSquare_4" / "Data" / _data_xml(0))
        assert not is_epu_xml_folder(tmp_path, n_squares=3)
        assert is_epu_xml_folder(tmp_path, n_squares=4)

    def test_reads_few_folders(self, session, monkeypatch):
        # what is read for a session, however many grid squares and survey folders it has
        from helicon.lib import epu

        for g in range(3, 200):
            (session / "Images-Disc1" / f"GridSquare_{g}" / "FoilHoles").mkdir(
                parents=True
            )
        read = []
        listing = epu._visible_entries
        monkeypatch.setattr(
            epu, "_visible_entries", lambda f: read.append(f) or listing(f)
        )
        assert is_epu_xml_folder(session)
        assert len(read) == 1  # its EpuSession.dm says it all
        read.clear()
        (session / "EpuSession.dm").unlink()
        assert is_epu_xml_folder(session)
        assert len(read) == 4  # the session, Images-Disc1, GridSquare_1 and its Data
        read.clear()
        assert not is_epu_xml_folder(session / "Images-Disc1" / "GridSquare_150")
        assert len(read) <= 2

    def test_hidden_folders_are_skipped(self, tmp_path):
        _write(tmp_path / ".snapshot" / _data_xml(0))
        assert not is_epu_xml_folder(tmp_path)


class TestFindAndRead:
    def test_finds_the_exposures_below(self, session):
        files = find_epu_xml_files(session)
        assert len(files) == 4
        assert all(is_epu_data_xml(f.name) for f in files)

    def test_renamed_files_when_none_is_named_the_epu_way(self, tmp_path):
        _write(tmp_path / "a.xml")
        (tmp_path / "b.xml").write_text("<notes/>")
        assert [f.name for f in find_epu_xml_files(tmp_path)] == ["a.xml"]

    def test_reads_the_beam_shifts(self, xml_folder):
        (xml_folder / _data_xml(7)).write_text("<MicroscopeImage>")  # cut short
        files, shifts, failed = read_epu_beamshifts(find_epu_xml_files(xml_folder))
        assert len(files) == 3
        assert [f.name for f in failed] == [_data_xml(7)]
        by_name = {f.name: tuple(xy) for f, xy in zip(files, shifts)}
        assert by_name[_data_xml(1)] == pytest.approx((-0.1, 0.2))
        assert shifts.shape == (3, 2) and shifts.dtype == float

    def test_nothing_to_read(self):
        files, shifts, failed = read_epu_beamshifts([])
        assert files == [] and failed == []
        assert shifts.shape == (0, 2)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _select(widget, name):
    from PySide6.QtCore import QItemSelectionModel

    for r in range(widget._model.rowCount()):
        index = widget._model.index(r, 0)
        if widget._model.file_path(index).endswith(name):
            widget._tree.selectionModel().select(
                index, QItemSelectionModel.Select | QItemSelectionModel.Clear
            )
            widget.wait_for_selection()
            return
    raise AssertionError(f"{name} is not listed")


class TestBeamShiftButton:
    def test_shown_for_the_folder_of_xml_files(self, qapp, xml_folder):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(xml_folder))
        widget.wait_for_selection()
        assert not widget._action_bar.isHidden()
        assert not widget._btn_beamshift.isHidden()
        assert widget._btn_beamshift.text() == "Beam Shift"
        emitted = []
        widget.display_requested.connect(lambda *a: emitted.append(a))
        widget._btn_beamshift.click()
        assert emitted == [(str(xml_folder.resolve()), "epuBeamShift", False)]

    def test_not_for_another_folder_nor_a_file(self, qapp, xml_folder, tmp_path):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        other = tmp_path / "other"
        other.mkdir()
        (other / "notes.txt").write_text("hello")
        widget = FolderBrowserWidget(start_dir=str(other))
        widget.wait_for_selection()
        assert widget._action_bar.isHidden()

        widget._navigate_to(str(xml_folder))
        widget.wait_for_selection()
        assert not widget._btn_beamshift.isHidden()
        # a file selected has the buttons of the file
        _select(widget, _data_xml(0))
        assert widget._btn_beamshift.isHidden()
        assert not widget._btn_text.isHidden()

        widget._navigate_to(str(other))
        widget.wait_for_selection()
        assert widget._action_bar.isHidden()

    def test_shown_for_a_session_above_the_data(self, qapp, session):
        from helicon.lib.gui.file_browser import FolderBrowserWidget

        widget = FolderBrowserWidget(start_dir=str(session))
        assert widget._display_modes_for(str(session)) == ["epuBeamShift"]
        assert widget._display_modes_for(str(session / "Images-Disc1")) == [
            "epuBeamShift"
        ]

    def test_going_through_folders_quickly(self, qapp, tmp_path, monkeypatch):
        # a folder's buttons are worked out in a thread: going on to the next folder while
        # it ends must not destroy the thread while it still runs (that aborted helicon)
        import time

        from helicon.lib.gui.file_browser import FolderBrowserWidget

        # as slow as a folder of a network disk
        modes_for = FolderBrowserWidget._display_modes_for
        monkeypatch.setattr(
            FolderBrowserWidget,
            "_display_modes_for",
            lambda self, path: time.sleep(0.002) or modes_for(self, path),
        )

        folders = []
        for i in range(4):
            (tmp_path / f"f{i}").mkdir()
            folders.append(tmp_path / f"f{i}")
        _write(folders[0] / _data_xml(0))
        widget = FolderBrowserWidget(start_dir=str(tmp_path))
        for n in range(500):
            widget._navigate_to(str(folders[n % 4]))
            for _ in range(n % 4):
                qapp.processEvents()
        widget._navigate_to(str(folders[0]))
        widget.wait_for_selection()
        assert not widget._btn_beamshift.isHidden()


class TestBeamShiftPlot:
    def test_plots_every_exposure(self, qapp, session):
        pytest.importorskip("pyqtgraph")
        from helicon.lib.gui.epu_beamshift import (
            _open_beamshift_plot,
            read_folder_beamshifts,
        )

        result = read_folder_beamshifts(session)
        win = _open_beamshift_plot(result)
        try:
            x, y = win._scatter.getData()
            assert sorted(zip(x, y)) == [(1, 0), (1, 1), (2, 0), (2, 1)]
            assert "4 exposures" in win.windowTitle()
            assert all(
                is_epu_data_xml(p.data()) for p in win._scatter.points()
            ), "a point is named after its XML file"
        finally:
            win.close()

    def test_reuses_the_window(self, qapp, session, xml_folder):
        pytest.importorskip("pyqtgraph")
        from helicon.lib.gui.epu_beamshift import (
            _open_beamshift_plot,
            read_folder_beamshifts,
        )

        win = _open_beamshift_plot(read_folder_beamshifts(session))
        try:
            again = _open_beamshift_plot(
                read_folder_beamshifts(xml_folder), reuse_window=win
            )
            assert again is win
            assert len(win._scatter.getData()[0]) == 3
            assert "xml" in win.windowTitle()
        finally:
            win.close()

    def test_says_when_there_is_nothing_to_plot(self, qapp, tmp_path, monkeypatch):
        pytest.importorskip("pyqtgraph")
        from PySide6.QtWidgets import QMessageBox

        from helicon.lib.gui.epu_beamshift import (
            _open_beamshift_plot,
            read_folder_beamshifts,
        )

        said = []
        monkeypatch.setattr(
            QMessageBox, "information", lambda *a, **k: said.append(a[2])
        )
        assert _open_beamshift_plot(read_folder_beamshifts(tmp_path)) is None
        assert said and os.fspath(tmp_path) in said[0]
