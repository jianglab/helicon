import sqlite3

import pandas as pd
import pytest

from helicon.lib import io
from helicon.lib.epu import EPU_xml_2_beamshift
from helicon.lib.exceptions import HeliconIOError

EPU_XML = """<?xml version="1.0" encoding="utf-8"?>
<MicroscopeImage xmlns="http://schemas.fei.com/Applications/Epu/MicroscopeImage"
    xmlns:a="http://schemas.datacontract.org/2004/07/Fei.Types">
  <microscopeData>
    <optics>
      <BeamShift><a:_x>{x}</a:_x><a:_y>{y}</a:_y></BeamShift>
      <SpotIndex>4</SpotIndex>
    </optics>
  </microscopeData>
</MicroscopeImage>
"""


class TestEpuBeamShift:
    def test_reads_both_components(self, tmp_path):
        f = tmp_path / "a.xml"
        f.write_text(EPU_XML.format(x="-1.5e-06", y="2.25e-06"))
        assert EPU_xml_2_beamshift(f) == (-1.5e-06, 2.25e-06)

    def test_a_str_path_works_too(self, tmp_path):
        f = tmp_path / "a.xml"
        f.write_text(EPU_XML.format(x="1", y="2"))
        assert EPU_xml_2_beamshift(str(f)) == (1.0, 2.0)

    def test_a_file_without_beam_shift_is_an_io_error(self, tmp_path):
        f = tmp_path / "a.xml"
        f.write_text("<MicroscopeImage><optics/></MicroscopeImage>")
        with pytest.raises(HeliconIOError, match="BeamShift"):
            EPU_xml_2_beamshift(f)


def _cistem_db(path, n=3):
    """A minimal cisTEM project database with one refinement of n particles."""
    db = sqlite3.connect(path)
    pos = list(range(1, n + 1))
    frames = {
        "REFINEMENT_LIST": pd.DataFrame(
            dict(REFINEMENT_ID=[1, 2], REFINEMENT_PACKAGE_ASSET_ID=[7, 7])
        ),
        "REFINEMENT_PACKAGE_ASSETS": pd.DataFrame(
            dict(REFINEMENT_PACKAGE_ASSET_ID=[7], STACK_FILENAME=["stack.mrcs"])
        ),
        "REFINEMENT_PACKAGE_CONTAINED_PARTICLES_7": pd.DataFrame(
            dict(
                POSITION_IN_STACK=pos,
                PIXEL_SIZE=[2.0] * n,
                X_POSITION=[20.0 * p for p in pos],
                Y_POSITION=[10.0 * p for p in pos],
                SPHERICAL_ABERRATION=[2.7] * n,
                MICROSCOPE_VOLTAGE=[300.0] * n,
                AMPLITUDE_CONTRAST=[0.07] * n,
            )
        ),
        "REFINEMENT_RESULT_2_1": pd.DataFrame(
            dict(
                POSITION_IN_STACK=pos,
                PSI=[1.0 * p for p in pos],
                THETA=[90.0] * n,
                PHI=[0.0] * n,
                XSHIFT=[4.0] * n,
                YSHIFT=[-6.0] * n,
                DEFOCUS1=[15000.0] * n,
                DEFOCUS2=[14000.0] * n,
                DEFOCUS_ANGLE=[30.0] * n,
                PHASE_SHIFT=[0.0] * n,
                LOGP=[-5.0] * n,
            )
        ),
    }
    for name, frame in frames.items():
        frame.to_sql(name, db, index=False)
    db.close()


class TestCistemDatabase:
    def test_the_last_refinement_is_read_into_relion_columns(self, tmp_path):
        f = tmp_path / "project.db"
        _cistem_db(f)
        data = io.cistem2dataframe(str(f), ignore_bad_particle_path=2)
        assert len(data) == 3
        assert list(data["rlnImageName"]) == [
            "000001@stack.mrcs",
            "000002@stack.mrcs",
            "000003@stack.mrcs",
        ]
        assert list(data["rlnAnglePsi"]) == [1.0, 2.0, 3.0]
        assert list(data["rlnCoordinateX"]) == [10.0, 20.0, 30.0]  # pixels
        assert list(data["rlnOriginX"]) == [-2.0] * 3  # shifts are negated
        assert data.attrs["convention"] == "relion"

    def test_an_iteration_can_be_chosen_with_the_prefix(self, tmp_path):
        f = tmp_path / "project.db"
        _cistem_db(f)
        data = io.cistem2dataframe(f"2@{f}", ignore_bad_particle_path=2)
        assert len(data) == 3

    def test_a_refinement_without_results_is_an_error(self, tmp_path):
        f = tmp_path / "project.db"
        _cistem_db(f)
        with pytest.raises(Exception):
            io.cistem2dataframe(f"1@{f}", ignore_bad_particle_path=2)
