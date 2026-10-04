"""Regression tests for bugs fixed in helicon.lib (io, transforms, analysis, ...)."""

import io as pyio
import itertools
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import helicon
from helicon.lib import analysis, filters, io, io_mrc, transforms
from helicon.lib.exposure_groups import propagate_ctf_median


def _cs_particles():
    shifts = [np.array([float(i), 10.0 + i], dtype=np.float32) for i in range(4)]
    shapes = [np.array([1000 + i, 2000 + i], dtype=np.uint32) for i in range(4)]
    return pd.DataFrame(
        {
            "blob/path": [f"p{i}.mrcs" for i in range(4)],
            "blob/idx": np.arange(4),
            "blob/psize_A": [2.0] * 4,
            "alignments2D/shift": shifts,
            "location/center_x_frac": [0.5] * 4,
            "location/center_y_frac": [0.25] * 4,
            "location/micrograph_shape": shapes,
        }
    )


class TestCryosparcToRelionIndex:
    def test_subset_rows_keep_their_own_shift_and_shape(self):
        d = _cs_particles().iloc[[1, 3]]
        ret = io.dataframe_cryosparc_to_relion(d)
        assert list(ret.index) == [1, 3]
        assert ret.loc[3, "rlnOriginXAngst"] == pytest.approx(-3.0 * 2.0)
        assert ret.loc[3, "rlnOriginYAngst"] == pytest.approx(-13.0 * 2.0)
        assert ret.loc[1, "rlnOriginXAngst"] == pytest.approx(-1.0 * 2.0)
        assert ret.loc[3, "rlnCoordinateX"] == pytest.approx(0.5 * 2003)
        assert ret.loc[3, "rlnCoordinateY"] == pytest.approx(0.25 * 1003)
        assert not ret.isna().any().any()


class TestStarDissolveOpticsgroup:
    def test_int_groups_fill_every_particle(self):
        data = pd.DataFrame(
            {
                "rlnImageName": [f"{i + 1:06d}@a.mrcs" for i in range(4)],
                "rlnOpticsGroup": [1, 1, 2, 2],
            },
            index=[10, 11, 12, 13],
        )
        data.attrs["convention"] = "relion"
        data.attrs["optics"] = pd.DataFrame(
            {
                "rlnOpticsGroup": [1, 2],
                "rlnVoltage": [300.0, 200.0],
                "rlnImagePixelSize": [1.1, 2.2],
            }
        )
        io.star_dissolve_opticsgroup(data)
        assert list(data["rlnVoltage"]) == [300.0, 300.0, 200.0, 200.0]
        assert list(data["rlnImagePixelSize"]) == [1.1, 1.1, 2.2, 2.2]
        assert data.attrs["optics"] is None

    def test_unknown_group_raises(self):
        data = pd.DataFrame({"rlnOpticsGroup": [1, 3]})
        data.attrs["convention"] = "relion"
        data.attrs["optics"] = pd.DataFrame(
            {"rlnOpticsGroup": [1], "rlnVoltage": [300.0]}
        )
        with pytest.raises(io.HeliconValueError):
            io.star_dissolve_opticsgroup(data)


class TestDataframe2cs:
    def test_dtypes_preserved(self, tmp_path):
        uid = np.array([2**63 + 5, 17], dtype=np.uint64)
        df = pd.DataFrame(
            {
                "uid": uid,
                "blob/path": [b"J1/a.mrc", b"J1/bb.mrc"],
                "blob/idx": np.array([0, 1], dtype=np.uint32),
                "ctf/df1_A": np.array([12345.678901234, 1.0], dtype=np.float64),
                "big": np.array([2**40, -1], dtype=np.int64),
                "alignments2D/shift": [
                    np.array([1.5, -2.5], dtype=np.float32),
                    np.array([3.0, 4.0], dtype=np.float32),
                ],
            }
        )
        out = tmp_path / "out.cs"
        io.dataframe2cs(df, str(out))
        cs = np.load(out)
        assert cs.dtype["uid"] == np.uint64
        assert cs["uid"][0] == uid[0]
        assert cs.dtype["ctf/df1_A"] == np.float64
        assert cs["ctf/df1_A"][0] == 12345.678901234
        assert cs["big"][0] == 2**40
        assert cs.dtype["blob/idx"] == np.uint32
        assert cs["blob/path"][1] == b"J1/bb.mrc"
        assert cs.dtype["alignments2D/shift"].shape == (2,)
        assert cs.dtype["alignments2D/shift"].base == np.float32
        np.testing.assert_array_equal(cs["alignments2D/shift"][0], [1.5, -2.5])

    def test_str_column(self, tmp_path):
        df = pd.DataFrame({"blob/path": ["a.mrc", "bcd.mrc"], "x": [1.0, 2.0]})
        out = tmp_path / "out.cs"
        io.dataframe2cs(df, str(out))
        cs = np.load(out)
        assert list(cs["blob/path"]) == [b"a.mrc", b"bcd.mrc"]


class TestPropagateCtfMedian:
    def test_vector_columns_use_per_component_median(self):
        data = {
            "ctf/exp_group_id": np.array([1, 1, 1, 2]),
            "ctf/cs_mm": np.array([2.7, 2.7, 2.7, 0.01]),
            "ctf/tilt_A": np.array(
                [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [5.0, 50.0]]
            ),
        }
        propagate_ctf_median(data, "ctf/exp_group_id")
        np.testing.assert_allclose(data["ctf/tilt_A"][:3], [[2.0, 20.0]] * 3)
        np.testing.assert_allclose(data["ctf/tilt_A"][3], [5.0, 50.0])


def _blob(shape, sigma=3.0):
    from scipy.ndimage import gaussian_filter

    a = np.zeros(shape)
    a[tuple(n // 2 for n in shape)] = 1.0
    return gaussian_filter(a, sigma)


class TestFftRescale:
    @pytest.mark.parametrize("n", [30, 31])
    def test_identity_2d(self, n):
        a = np.random.default_rng(0).random((n, n))
        b = np.fft.ifft2(transforms.fft_rescale(a))
        np.testing.assert_allclose(b.real, a, atol=1e-4)
        assert np.abs(b.imag).max() < 1e-4

    @pytest.mark.parametrize("n", [16, 17])
    def test_identity_3d(self, n):
        a = np.random.default_rng(0).random((n, n, n))
        b = np.fft.ifftn(transforms.fft_rescale(a))
        np.testing.assert_allclose(b.real, a, atol=1e-4)
        assert np.abs(b.imag).max() < 1e-4

    @pytest.mark.parametrize("n,m", [(64, 32), (63, 31), (64, 33), (31, 64)])
    def test_resample_keeps_centred_peak_centred(self, n, m):
        a = _blob((n, n))
        c = 2.0 * n / m
        b = np.fft.ifft2(
            transforms.fft_rescale(a, apix=1.0, cutoff_res=(c, c), output_size=(m, m))
        )
        assert np.unravel_index(np.argmax(b.real), b.shape) == (m // 2, m // 2)
        assert np.abs(b.imag).max() < 1e-3 * np.abs(b.real).max()


class TestFftCrop:
    @pytest.mark.parametrize(
        "shape,out",
        [
            ((64, 64), (32, 32)),
            ((64, 64), (31, 33)),
            ((63, 65), (31, 32)),
            ((32, 32, 32), (16, 15, 17)),
        ],
    )
    def test_shape_and_mean(self, shape, out):
        a = np.random.default_rng(1).random(shape) + 5.0
        b = transforms.fft_crop(a, out)
        assert b.shape == out
        assert b.mean() == pytest.approx(a.mean())

    def test_centred_peak_stays_centred_odd(self):
        b = transforms.fft_crop(_blob((64, 64), 4.0), (31, 31))
        assert np.unravel_index(np.argmax(b), b.shape) == (15, 15)


class TestCropCenterZ:
    @pytest.mark.parametrize("nz,n", [(20, 5), (20, 6), (21, 7)])
    def test_exactly_n_slices(self, nz, n):
        data = np.arange(nz)[:, None, None] * np.ones((1, 2, 2))
        out = transforms.crop_center_z(data, n)
        assert out.shape == (n, 2, 2)
        assert out[n // 2, 0, 0] == nz // 2


class TestApplyHelicalSymmetryDefaults:
    def test_default_new_size(self):
        a = np.random.default_rng(0).random((16, 16, 16)).astype(np.float32)
        out = transforms.apply_helical_symmetry(a, 1.0, -1.0, 4.75)
        assert out.shape == a.shape


class TestCalcFrc2d:
    def test_frequency_axis_matches_shells(self):
        n, apix = 64, 2.0
        rng = np.random.default_rng(0)
        img1 = rng.standard_normal((n, n))
        # img2 shares img1 below 0.25 cycles/pixel and is independent above
        ky = np.fft.fftfreq(n)[:, None]
        kx = np.fft.fftfreq(n)[None, :]
        high = np.sqrt(ky**2 + kx**2) > 0.25
        F = np.fft.fft2(img1)
        F[high] = np.fft.fft2(rng.standard_normal((n, n)))[high]
        img2 = np.fft.ifft2(F).real
        saxis, frc = analysis.calc_frc_2d(img1, img2, apix)
        s_pix = saxis * apix  # cycles/pixel
        assert s_pix[-1] == pytest.approx(0.5)
        assert np.all(frc[(s_pix > 0) & (s_pix < 0.22)] > 0.99)
        assert np.all(np.abs(frc[s_pix > 0.28]) < 0.4)
        assert np.abs(frc[s_pix > 0.28]).mean() < 0.15

    def test_zero_power_shells_are_not_perfect(self):
        n = 32
        img = np.ones((n, n))  # only the DC term has power
        saxis, frc = analysis.calc_frc_2d(img, img, 1.0)
        assert np.isnan(frc[1:]).all()
        assert analysis.frc_score(img, img, 1.0) == pytest.approx(1.0)
        rng = np.random.default_rng(0)
        noise = rng.standard_normal((n, n))
        assert analysis.frc_score(noise, rng.standard_normal((n, n)), 1.0) < 0.3


class TestCalcFscUnits:
    def test_small_apix_keeps_all_shells(self):
        n, apix = 16, 0.5
        rng = np.random.default_rng(0)
        m = rng.standard_normal((n, n, n))
        fsc = analysis.calc_fsc(m, m, apix)
        assert len(fsc) == n // 2 + 1
        assert fsc[-1, 0] == pytest.approx(1 / (2 * apix))
        F = np.fft.rfftn(m)
        fsc2 = analysis.calc_fsc_from_fft(F, F, n, apix)
        assert len(fsc2) == n // 2 + 1


class _Header:
    def __init__(self, mapc, mapr, maps):
        self.mapc, self.mapr, self.maps = mapc, mapr, maps

    def copy(self):
        return _Header(self.mapc, self.mapr, self.maps)


class TestChangeMapAxesOrder:
    @pytest.mark.parametrize("order", list(itertools.permutations([1, 2, 3])))
    def test_all_permutations(self, order):
        nx, ny, nz = 2, 3, 4
        # f(x, y, z) is a unique value per voxel
        x, y, z = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        f = x + 10 * y + 100 * z  # indexed [x, y, z]
        # stored map: columns = axis order[0], rows = order[1], sections = order[2]
        cols, rows, secs = (o - 1 for o in order)
        stored = np.transpose(f, (secs, rows, cols))  # numpy (sections, rows, cols)
        data, header = io_mrc.change_map_axes_order(stored, _Header(*order))
        expected = np.transpose(f, (2, 1, 0))  # (z, y, x)
        np.testing.assert_array_equal(data, expected)
        assert (header.mapc, header.mapr, header.maps) == (1, 2, 3)


class TestReadImage2d:
    def test_single_2d_image(self, tmp_path):
        import mrcfile

        img = np.arange(12, dtype=np.float32).reshape(3, 4)
        f = tmp_path / "one.mrc"
        with mrcfile.new(str(f)) as mrc:
            mrc.set_data(img)
        np.testing.assert_array_equal(io_mrc.read_image_2d(str(f), 0), img)


class TestAvailableCpu:
    def test_never_below_one(self):
        import psutil

        class _Mem:
            available = 1024**3  # 1 GB

        with patch.object(psutil, "virtual_memory", return_value=_Mem()):
            from helicon.lib.system import available_cpu

            assert available_cpu(mem_gb_per_cpu=100) == 1


class TestTerminalEnvFile:
    def test_private_location_and_permissions(self, tmp_path, monkeypatch):
        if os.name == "nt":
            pytest.skip("POSIX permissions")
        from helicon.lib import terminal

        monkeypatch.setattr(helicon, "cache_dir", tmp_path / "cache")
        envfile = terminal._env_file()
        assert str(envfile).startswith(str(tmp_path))
        assert "/tmp/helicon_terminal" not in str(envfile)
        path = terminal._write_env_file({"PATH": "/usr/bin"})
        assert path == envfile
        assert path.read_text() == "export PATH=/usr/bin\n"
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    def test_linux_fallback_sources_a_written_file(self, tmp_path, monkeypatch):
        if os.name == "nt":
            pytest.skip("POSIX shells")
        from helicon.lib import terminal

        monkeypatch.setattr(helicon, "cache_dir", tmp_path / "cache")
        with (
            patch.object(terminal, "platform") as mock_platform,
            patch.object(terminal, "_is_wsl", return_value=False),
            patch.object(terminal, "_linux_terminal_candidates", return_value=[]),
            patch.object(terminal, "_spawn_detached", return_value=True) as spawn,
        ):
            mock_platform.system.return_value = "Linux"
            terminal._open_terminal(str(tmp_path))
        cmd = spawn.call_args[0][0][-1]
        assert str(terminal._env_file()) in cmd
        assert terminal._env_file().is_file()
        assert "[ -f " in cmd


class TestNormalize:
    def test_min_max_offset(self):
        out = filters.normalize_min_max(np.array([0.0, 5.0, 10.0]), min=-1, max=1)
        np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])

    def test_mean_std_args(self):
        rng = np.random.default_rng(0)
        out = filters.normalize_mean_std(rng.random(1000), mean=5, std=2)
        assert out.mean() == pytest.approx(5)
        assert out.std() == pytest.approx(2)


class _FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestDownloadFileFromUrl:
    def test_small_file_readable_and_timeout(self):
        import requests

        from helicon.lib.path_utils import download_file_from_url

        calls = []

        def fake_get(url, **kwargs):
            calls.append(kwargs)
            return _FakeResponse(b"hello")

        with patch.object(requests, "get", side_effect=fake_get):
            fileobj = download_file_from_url("https://example.org/x.mrc")
            name = download_file_from_url(
                "https://example.org/y.mrc", return_filename=True
            )
        assert Path(fileobj.name).read_bytes() == b"hello"
        assert fileobj.read() == b"hello"
        try:
            assert Path(name).read_bytes() == b"hello"
        finally:
            Path(name).unlink()
        assert all(c.get("timeout") for c in calls)


class TestIoModuleMisc:
    def test_no_global_pandas_option_change(self):
        src = Path(io.__file__).read_text()
        assert "copy_on_write" not in src

    def test_dataframe2star_does_not_mutate_input(self):
        data = pd.DataFrame(
            {
                "rlnImageName": ["000001@a.mrcs", "000002@a.mrcs"],
                "rlnVoltage": [300.0, 300.0],
                "rlnImagePixelSize": [1.0, 1.0],
                "rlnSphericalAberration": [2.7, 2.7],
            }
        )
        data.attrs["convention"] = "relion"
        cols = list(data.columns)
        io.dataframe2star(data, pyio.StringIO())
        assert list(data.columns) == cols

    def test_images2dataframe_mixed_star_and_cs(self):
        star = pd.DataFrame({"rlnImageName": ["000001@a.mrcs"]})
        star.attrs["convention"] = "relion"
        cs = pd.DataFrame({"blob/path": ["b.mrcs"], "blob/idx": [0]})
        cs.attrs["convention"] = "cryosparc"

        def fake(f, *args, **kwargs):
            return star.copy() if f.endswith(".star") else cs.copy()

        with patch.object(io, "image2dataframe", side_effect=fake):
            out = io.images2dataframe(["a.star", "b.cs"])
        assert list(out["rlnImageName"]) == ["000001@a.mrcs", "000001@b.mrcs"]

    def test_get_relion_project_folder_walks_up(self, tmp_path):
        (tmp_path / "default_pipeline.star").write_text("")
        job = tmp_path / "Class2D" / "job003"
        job.mkdir(parents=True)
        (job / "job_pipeline.star").write_text("")
        star = job / "run_it025_data.star"
        assert io.get_relion_project_folder(str(star)) == str(tmp_path.resolve())

    def test_get_relion_project_folder_none(self, tmp_path):
        star = tmp_path / "Class2D" / "job003" / "x.star"
        assert io.get_relion_project_folder(str(star)) is None
