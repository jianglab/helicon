"""Tests for average_power_spectra module and images2star plugin."""

from __future__ import annotations

import argparse
from pathlib import Path
import mrcfile
import numpy as np
import pandas as pd
import pytest

from helicon.lib.average_power_spectra import (
    average_power_spectra_from_dataframe,
    _remap_columns,
    rotation_trans_align,
)


# ---------------------------------------------------------------------------
# _remap_columns tests
# ---------------------------------------------------------------------------


class TestRemapColumns:
    def test_maps_relion_columns(self):
        data = pd.DataFrame(
            {
                "rlnImageName": ["000001@a.mrcs", "000002@b.mrcs", "000003@c.mrcs"],
                "rlnClassNumber": [1, 1, 2],
                "rlnAnglePsiPrior": [90.0, 95.0, 85.0],
                "rlnPixelSize": [1.0, 1.0, 2.0],
                "rlnHelicalTubeID": [1, 1, 2],
            }
        )
        mapped = _remap_columns(data)
        assert list(mapped["filename"]) == ["a.mrcs", "b.mrcs", "c.mrcs"]
        assert list(mapped["pid"]) == [0, 1, 2]
        assert list(mapped["apix"]) == [1.0, 1.0, 2.0]
        assert list(mapped["class"]) == [1, 1, 2]
        assert list(mapped["phi0"]) == [0.0, 5.0, -5.0]
        assert list(mapped["helicaltube"]) == [0, 0, 1]

    def test_pass_through_when_already_mapped(self):
        data = pd.DataFrame({"filename": ["a.mrcs"], "pid": [0], "apix": [1.0]})
        mapped = _remap_columns(data)
        assert list(mapped.columns) == ["filename", "pid", "apix"]

    def test_phi0_from_rlnAnglePsi(self):
        data = pd.DataFrame(
            {
                "rlnImageName": ["000001@a.mrcs"],
                "rlnAnglePsiPrior": [100.0],
                "rlnPixelSize": [1.0],
            }
        )
        mapped = _remap_columns(data)
        assert mapped["phi0"].iloc[0] == 10.0  # 100 - 90

    def test_phi0_absent_when_no_psi_column(self):
        data = pd.DataFrame(
            {
                "rlnImageName": ["000001@a.mrcs"],
                "rlnPixelSize": [1.0],
            }
        )
        mapped = _remap_columns(data)
        assert "phi0" not in mapped

    def test_apix_fallback_from_param(self):
        data = pd.DataFrame(
            {
                "rlnImageName": ["000001@a.mrcs"],
            }
        )
        mapped = _remap_columns(data, apix=2.5)
        assert mapped["apix"].iloc[0] == 2.5


# ---------------------------------------------------------------------------
# average_power_spectra_from_dataframe end-to-end tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mrc_stack(tmp_path):
    """Create a small MRC stack with synthetic particles."""
    path = tmp_path / "particles.mrcs"
    np.random.seed(42)
    ny, nx = 64, 64
    stack = np.zeros((4, ny, nx), dtype=np.float32)
    for i in range(4):
        cy, cx = ny // 2, nx // 2
        ys, xs = np.ogrid[:ny, :nx]
        stack[i] = np.exp(
            -((ys - cy) ** 2 + (xs - cx) ** 2) / (2 * (6 + i * 2) ** 2)
        ).astype(np.float32)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(stack)
    return path


@pytest.fixture
def particles_df(mrc_stack):
    """DataFrame with RELION-style columns referencing the MRC stack."""
    rows = []
    for i in range(4):
        rows.append(
            {
                "rlnImageName": f"{i+1:06d}@{mrc_stack}",
                "rlnPixelSize": 1.5,
                "rlnClassNumber": (i % 2) + 1,
                "rlnAnglePsiPrior": 90.0,
            }
        )
    return pd.DataFrame(rows)


class TestAveragePowerSpectraFromDataframe:
    def test_groups_by_class(self, particles_df):
        results, apix, cutoff = average_power_spectra_from_dataframe(
            particles_df,
            groupby=["class"],
            batch_size=10,
            cpu=1,
            fft_x=32,
            fft_y=64,
        )
        assert len(results) == 2
        for r in results.values():
            assert r["ps_avg"].shape == (64, 32)
            assert "pd_avg" in r
            assert np.any(r["ps_avg"] > 0)

    def test_single_group_when_no_groupby(self, mrc_stack):
        """Without phi0/class, single MRC stack triggers per-PID grouping -> 4 groups."""
        df = pd.DataFrame(
            {
                "rlnImageName": [f"{i+1:06d}@{mrc_stack}" for i in range(4)],
                "rlnPixelSize": [1.5] * 4,
            }
        )
        results, _, _ = average_power_spectra_from_dataframe(
            df,
            groupby=None,
            batch_size=10,
            cpu=1,
            fft_x=32,
            fft_y=64,
        )
        assert len(results) == 4
        for r in results.values():
            assert r["#images"] == 1

    def test_auto_apix_from_column(self, particles_df):
        """apix is read from rlnPixelSize, not from parameter."""
        results, apix, _ = average_power_spectra_from_dataframe(
            particles_df,
            apix=0.0,
            groupby=None,
            batch_size=10,
            cpu=1,
            fft_x=32,
            fft_y=64,
        )
        assert apix == 1.5  # from rlnPixelSize

    def test_auto_cutoff_resolution(self, particles_df):
        """With cutoff_res=[0,0], it defaults to 2*apix."""
        _, _, (cy, cx) = average_power_spectra_from_dataframe(
            particles_df,
            apix=1.5,
            groupby=None,
            batch_size=10,
            cpu=1,
            cutoff_res=[0.0, 0.0],
            fft_x=32,
            fft_y=64,
        )
        assert cy == 3.0  # 2 * 1.5
        assert cx == 3.0

    def test_min_particles_filters_small_groups(self, particles_df):
        """With min_particles=3, only class with >=3 particles survives."""
        results, _, _ = average_power_spectra_from_dataframe(
            particles_df,
            groupby=["class"],
            batch_size=10,
            cpu=1,
            min_particles=3,
            fft_x=32,
            fft_y=64,
        )
        # Each class has 2 particles, so both should be filtered out
        assert len(results) == 0

    def test_no_phi0_no_pd_avg(self, particles_df):
        """Without phi0 column, pd_avg should not be computed."""
        # Drop angle column
        df = particles_df.drop(columns=["rlnAnglePsiPrior"])
        results, _, _ = average_power_spectra_from_dataframe(
            df,
            groupby=["class"],
            batch_size=10,
            cpu=1,
            fft_x=32,
            fft_y=64,
        )
        if len(results) > 0:
            for r in results.values():
                assert "pd_avg" not in r

    def test_force_phase_diff(self, particles_df):
        """force_phase_diff=True computes pd_avg even without phi0."""
        df = particles_df.drop(columns=["rlnAnglePsiPrior"])
        results, _, _ = average_power_spectra_from_dataframe(
            df,
            groupby=["class"],
            batch_size=10,
            cpu=1,
            force_phase_diff=True,
            fft_x=32,
            fft_y=64,
        )
        if len(results) > 0:
            for r in results.values():
                assert "pd_avg" in r


# ---------------------------------------------------------------------------
# images2star plugin tests
# ---------------------------------------------------------------------------


class TestAveragePowerSpectraPlugin:
    def test_registered_in_argparse(self):
        """--averagePowerSpectra should be registered as an argparse argument."""
        from helicon.plugins.images2star.averagepowerspectra import add_args

        parser = argparse.ArgumentParser()
        add_args(parser)
        # The action is 'append', so we can test via parse_known_args
        args = parser.parse_known_args(["--averagePowerSpectra", "groupby=class"])[0]
        assert args.averagePowerSpectra == ["groupby=class"]

    def test_handler_saves_mrcs(self, particles_df, tmp_path):
        from helicon.plugins.images2star.averagepowerspectra import handle

        args = argparse.Namespace(output_starFile=str(tmp_path / "output.star"))
        index_d = {"averagePowerSpectra": 0}
        outdir = tmp_path / "mrcs_out"
        param = f"groupby=class:fftX=32:fftY=64:cpu=1:outdir={outdir}"

        result_data, result_index = handle(particles_df, args, index_d, param)

        assert result_index["averagePowerSpectra"] == 1
        mrcs_files = list(outdir.glob("*.mrcs"))
        assert len(mrcs_files) > 0
        for mf in mrcs_files:
            with mrcfile.open(mf) as mrc:
                assert mrc.data.ndim == 2
                assert np.any(mrc.data != 0)

    def test_handler_returns_data_unchanged(self, particles_df, tmp_path):
        from helicon.plugins.images2star.averagepowerspectra import handle

        args = argparse.Namespace(output_starFile=str(tmp_path / "output.star"))
        index_d = {"averagePowerSpectra": 0}
        result_data, _ = handle(
            particles_df, args, index_d, "groupby=class:cpu=1:fftX=32:fftY=64"
        )

        pd.testing.assert_frame_equal(result_data, particles_df)

    def test_handler_empty_param_returns_early(self, particles_df, tmp_path):
        from helicon.plugins.images2star.averagepowerspectra import handle

        args = argparse.Namespace(output_starFile=str(tmp_path / "output.star"))
        index_d = {"averagePowerSpectra": 0}
        result_data, result_index = handle(particles_df, args, index_d, "")

        pd.testing.assert_frame_equal(result_data, particles_df)
        assert result_index["averagePowerSpectra"] == 0

    def test_discovered_by_plugin_system(self):
        from helicon.plugins.images2star import _plugins

        assert "averagePowerSpectra" in _plugins


# ---------------------------------------------------------------------------
# rotation_trans_align test
# ---------------------------------------------------------------------------


class TestRotationTransAlign:
    def test_returns_correct_shape(self):
        image = np.random.randn(32, 32).astype(np.float32)
        aligned, da, shift = rotation_trans_align(image, angle0=0, mask=None)
        assert aligned.shape == (32, 32)
        assert isinstance(da, float) or isinstance(da, np.floating)
        assert isinstance(shift, float) or isinstance(shift, np.floating)
