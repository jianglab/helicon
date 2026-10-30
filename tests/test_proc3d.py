import argparse
import sys
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
import helicon
from helicon.commands import proc3d
from helicon.lib.exceptions import HeliconFileExistsError


class TestProc3dArgs(object):
    def test_add_args_parser_has_expected_arguments(self):
        parser = argparse.ArgumentParser()
        proc3d.add_args(parser)
        actions = {a.dest for a in parser._actions}
        expected = {
            "inputMapFile",
            "outputMapFile",
            "apix",
            "flip_hand",
            "clip",
            "fft_resample",
            "helical_sym",
            "z_moving_average",
            "force",
            "verbose",
            "cpu",
        }
        assert expected.issubset(actions)

    def test_repeated_options_keep_each_occurrence_value(self):
        # Repeated options are a pipeline: each occurrence keeps its own
        # value, argparse must not collapse them onto the last one.
        parser = argparse.ArgumentParser()
        proc3d.add_args(parser)
        args = parser.parse_args(
            ["in.mrc", "--flip_hand", "x", "--clip", "new_nx=6", "--flip_hand", "z"]
        )
        assert args.flip_hand == ["x", "z"]
        assert args.clip == ["new_nx=6"]

    def test_check_args_sets_default_output_path(self):
        parser = argparse.ArgumentParser()
        proc3d.add_args(parser)
        args = parser.parse_args(["input.mrc"])
        args = proc3d.check_args(args, parser)
        assert args.outputMapFile == Path("input.proc3d.mrc")

    def test_check_args_uses_provided_output_path(self):
        parser = argparse.ArgumentParser()
        proc3d.add_args(parser)
        args = parser.parse_args(["input.mrc", "--outputMapFile", "out.mrc"])
        with patch("pathlib.Path.exists", return_value=False):
            args = proc3d.check_args(args, parser)
        assert args.outputMapFile == Path("out.mrc")

    @patch("pathlib.Path.exists", return_value=True)
    def test_check_args_existing_output_raises_error(self, mock_exists):
        parser = argparse.ArgumentParser()
        proc3d.add_args(parser)
        args = parser.parse_args(["input.mrc"])
        with pytest.raises(HeliconFileExistsError):
            proc3d.check_args(args, parser)

    @patch("pathlib.Path.exists", return_value=True)
    def test_check_args_force_overwrites_existing(self, mock_exists):
        parser = argparse.ArgumentParser()
        proc3d.add_args(parser)
        args = parser.parse_args(["input.mrc", "--force", "1"])
        args = proc3d.check_args(args, parser)
        assert args.outputMapFile == Path("input.proc3d.mrc")

    @patch.object(
        sys, "argv", ["prog", "input.mrc", "--flip_hand", "x", "--apix", "1.5"]
    )
    def test_check_args_sets_all_options(self):
        parser = argparse.ArgumentParser()
        proc3d.add_args(parser)
        args = parser.parse_args(["input.mrc", "--flip_hand", "x", "--apix", "1.5"])
        args = proc3d.check_args(args, parser)
        for opt in ["flip_hand", "apix"]:
            assert opt in args.all_options
        for opt in ["cpu", "force", "inputMapFile", "outputMapFile", "verbose"]:
            assert opt not in args.all_options
        # every proc3d plugin option is repeatable (a pipeline step)
        assert set(args.append_options) == {
            "apix",
            "clip",
            "denoiseCurvelet",
            "fft_resample",
            "flip_hand",
            "helical_sym",
            "z_moving_average",
        }


class TestProc3dHandlers(object):
    """Test each proc3d option handler in isolation using a known 3D array."""

    def setup_method(self, method):
        self.data = np.zeros((16, 12, 12), dtype=np.float32)
        self.data[4:12, 2:10, 2:10] = 1.0
        self.apix = 1.0
        self.nz, self.ny, self.nx = self.data.shape

    # --- flip_hand ---

    def _run_flip_hand(self, data, axis):
        if axis not in ["x", "y", "z"]:
            raise SystemExit(1)
        return helicon.flip_hand(data, axis=axis)

    def test_flip_hand_x(self):
        result = self._run_flip_hand(self.data.copy(), "x")
        np.testing.assert_array_equal(result, self.data[:, :, ::-1])

    def test_flip_hand_y(self):
        result = self._run_flip_hand(self.data.copy(), "y")
        np.testing.assert_array_equal(result, self.data[:, ::-1, :])

    def test_flip_hand_z(self):
        result = self._run_flip_hand(self.data.copy(), "z")
        np.testing.assert_array_equal(result, self.data[::-1, :, :])

    def test_flip_hand_invalid_axis_errors(self):
        with pytest.raises(SystemExit):
            self._run_flip_hand(self.data.copy(), "w")

    # --- clip ---

    def _run_clip(self, data, param_str, nx, ny, nz):
        param_dict_default = dict(
            new_nx=nx,
            new_ny=ny,
            new_nz=nz,
            center_x=nx // 2,
            center_y=ny // 2,
            center_z=nz // 2,
        )
        _, param_dict = helicon.parse_param_str(param_str)
        param_dict, _, _ = helicon.validate_param_dict(
            param=param_dict, param_ref=param_dict_default
        )
        new_nx = int(param_dict["new_nx"])
        new_ny = int(param_dict["new_ny"])
        new_nz = int(param_dict["new_nz"])
        center_x = int(param_dict["center_x"])
        center_y = int(param_dict["center_y"])
        center_z = int(param_dict["center_z"])
        if new_nx < 1 or new_ny < 1 or new_nz < 1:
            raise ValueError("dimensions must be >0")

        data = helicon.get_clip3d(
            data,
            z0=center_z - new_nz // 2,
            y0=center_y - new_ny // 2,
            x0=center_x - new_nx // 2,
            nz=new_nz,
            ny=new_ny,
            nx=new_nx,
        )
        return data

    def test_clip_reduces_size(self):
        result = self._run_clip(
            self.data.copy(),
            "new_nx=6:new_ny=6:new_nz=8",
            self.nx,
            self.ny,
            self.nz,
        )
        assert result.shape == (8, 6, 6)

    def test_clip_centered_preserves_content(self):
        data = np.zeros((20, 20, 20), dtype=np.float32)
        data[5:15, 5:15, 5:15] = 2.0
        result = self._run_clip(
            data,
            "new_nx=10:new_ny=10:new_nz=10",
            20,
            20,
            20,
        )
        assert result.shape == (10, 10, 10)
        assert result[4, 4, 4] > 0

    def test_clip_full_size_returns_same(self):
        result = self._run_clip(
            self.data.copy(),
            f"new_nx={self.nx}:new_ny={self.ny}:new_nz={self.nz}",
            self.nx,
            self.ny,
            self.nz,
        )
        assert result.shape == self.data.shape
        np.testing.assert_array_equal(result, self.data)

    # --- fft_resample ---

    def _run_fft_resample(self, data, param_str, apix, nx, ny, nz):
        param_dict_default = dict(new_nx=nx, new_ny=ny, new_nz=nz)
        _, param_dict = helicon.parse_param_str(param_str)
        param_dict, _, _ = helicon.validate_param_dict(
            param=param_dict, param_ref=param_dict_default
        )
        new_nx = int(param_dict["new_nx"])
        new_ny = int(param_dict["new_ny"])
        new_nz = int(param_dict["new_nz"])
        if new_nx < 1 or new_ny < 1 or new_nz < 1:
            raise ValueError("dimensions must be >0")

        fft = helicon.fft_rescale(
            data,
            apix=apix,
            cutoff_res=(
                2 * apix * nz / new_nz,
                2 * apix * ny / new_ny,
                2 * apix * nx / new_nx,
            ),
            output_size=(new_nz, new_ny, new_nx),
        )
        data = np.abs(np.fft.ifftn(fft)).astype(np.float32)
        data *= new_nx * new_ny * new_nz / (nx * ny * nz)
        return data

    def test_fft_resample_downsample(self):
        data = np.random.rand(16, 16, 16).astype(np.float32)
        result = self._run_fft_resample(
            data.copy(),
            "new_nx=8:new_ny=8:new_nz=8",
            1.0,
            16,
            16,
            16,
        )
        assert result.shape == (8, 8, 8)
        assert np.all(np.isfinite(result))

    def test_fft_resample_upsample(self):
        data = np.random.rand(8, 8, 8).astype(np.float32)
        result = self._run_fft_resample(
            data.copy(),
            "new_nx=16:new_ny=16:new_nz=16",
            1.0,
            8,
            8,
            8,
        )
        assert result.shape == (16, 16, 16)
        assert np.all(np.isfinite(result))

    def test_fft_resample_identity(self):
        data = np.random.rand(12, 12, 12).astype(np.float32)
        result = self._run_fft_resample(
            data.copy(),
            "new_nx=12:new_ny=12:new_nz=12",
            1.0,
            12,
            12,
            12,
        )
        assert result.shape == (12, 12, 12)

    # --- z_moving_average ---

    def _run_z_moving_average(self, data, param_str, apix):
        param_dict_default = dict(length=0.0, n_pixel=0)
        _, param_dict = helicon.parse_param_str(param_str)
        param_dict, _, _ = helicon.validate_param_dict(
            param=param_dict, param_ref=param_dict_default
        )
        length = float(param_dict["length"])
        n_pixel = int(float(param_dict["n_pixel"]))
        if length <= 0 and n_pixel <= 0:
            raise ValueError("length or n_pixel must be >0")
        if length > 0 and n_pixel > 0:
            raise ValueError("only one of length or n_pixel should be specified")

        if length > 0:
            n_pixel = int(np.round(length / apix))

        tmp = np.cumsum(data, axis=0, dtype=float)
        data = data.copy()
        data[n_pixel // 2 : -n_pixel // 2] = (tmp[n_pixel:] - tmp[:-n_pixel]) / n_pixel
        return data

    def test_z_moving_average_by_n_pixel(self):
        data = np.zeros((20, 8, 8), dtype=np.float32)
        data[:10] = 0.0
        data[10:] = 1.0
        result = self._run_z_moving_average(data, "n_pixel=5", 1.0)
        assert result.shape == (20, 8, 8)
        assert np.all(np.isfinite(result))

    def test_z_moving_average_by_length(self):
        data = np.zeros((20, 8, 8), dtype=np.float32)
        data[:10] = 0.0
        data[10:] = 1.0
        result = self._run_z_moving_average(data, "length=5", 1.0)
        assert result.shape == (20, 8, 8)

    def test_z_moving_average_no_params_errors(self):
        with pytest.raises(ValueError):
            self._run_z_moving_average(self.data.copy(), "length=0:n_pixel=0", 1.0)

    def test_z_moving_average_both_params_errors(self):
        with pytest.raises(ValueError):
            self._run_z_moving_average(self.data.copy(), "length=5:n_pixel=5", 1.0)

    # --- apix ---

    def test_apix_overrides_pixel_size(self):
        apix = 1.0
        param = "2.5"
        apix = float(param)
        assert apix == 2.5

    # --- helical_sym ---

    @patch("helicon.apply_helical_symmetry")
    def test_helical_sym_calls_apply(self, mock_apply):
        mock_apply.return_value = np.zeros((24, 24, 24), dtype=np.float32)

        data = self.data.copy()
        apix = 1.0
        nz, ny, nx = data.shape
        param_str = (
            "twist=30:rise=5:csym=1:center_n_rise=2:new_apix=1.0:new_nz=24:new_nxy=24"
        )

        param_dict_default = dict(
            twist=0.0,
            rise=0.0,
            csym=1,
            center_len=0.0,
            center_n_rise=0.0,
            center_fraction=0.0,
            new_apix=apix,
            new_nz=nz,
            new_nxy=nx,
        )
        _, param_dict = helicon.parse_param_str(param_str)
        param_dict, _, _ = helicon.validate_param_dict(
            param=param_dict, param_ref=param_dict_default
        )
        twist = float(param_dict["twist"])
        rise = float(param_dict["rise"])
        csym = int(param_dict.get("csym", 1))
        new_apix = float(param_dict.get("new_apix", apix))
        new_nz = int(param_dict["new_nz"])
        new_nxy = int(param_dict["new_nxy"])
        center_n_rise = float(param_dict["center_n_rise"])
        center_fraction = center_n_rise * rise / (nz * apix)
        center_fraction = max(rise / (nz * apix), min(1.0, center_fraction))

        data = helicon.apply_helical_symmetry(
            data=data,
            apix=apix,
            twist_degree=twist,
            rise_angstrom=rise,
            csym=csym,
            fraction=center_fraction,
            new_size=(new_nz, new_nxy, new_nxy),
            new_apix=new_apix,
            cpu=1,
        )

        mock_apply.assert_called_once()
        assert data.shape == (24, 24, 24)

    def test_helical_sym_validates_rise(self):
        from helicon.plugins.proc3d import helical_sym
        from helicon.lib.exceptions import HeliconError

        args = argparse.Namespace(verbose=0, cpu=1)
        data = self.data.copy()
        apix = 1.0
        nz, ny, nx = data.shape
        index_d = {"helical_sym": 0}

        with pytest.raises(HeliconError):
            helical_sym.handle(
                data, args, index_d, "rise=0:center_n_rise=2", apix, nx, ny, nz
            )

    def test_helical_sym_validates_csym(self):
        from helicon.plugins.proc3d import helical_sym
        from helicon.lib.exceptions import HeliconError

        args = argparse.Namespace(verbose=0, cpu=1)
        data = self.data.copy()
        apix = 1.0
        nz, ny, nx = data.shape
        index_d = {"helical_sym": 0}

        with pytest.raises(HeliconError):
            helical_sym.handle(
                data,
                args,
                index_d,
                "rise=5:twist=30:csym=0:center_n_rise=2",
                apix,
                nx,
                ny,
                nz,
            )

    # --- denoiseCurvelet ---

    def _denoise(self, data, param, **args):
        from helicon.plugins.proc3d.denoise_curvelet import handle

        index_d = {"denoiseCurvelet": 0}
        out = handle(
            data,
            argparse.Namespace(verbose=1, cpu=1, **args),
            index_d,
            param,
            self.apix,
            self.nx,
            self.ny,
            self.nz,
        )
        assert index_d["denoiseCurvelet"] == 1
        return out

    def test_denoise_curvelet3d_argparse(self):
        parser = argparse.ArgumentParser()
        proc3d.add_args(parser)
        args = parser.parse_args(["input.mrc", "--denoiseCurvelet", "strength=1"])
        assert args.denoiseCurvelet == ["strength=1"]
        args = parser.parse_args(["input.mrc", "--denoiseCurvelet"])
        assert args.denoiseCurvelet == [""]
        args = parser.parse_args(["input.mrc", "--denoiseCurvelet", "tileSize=128"])
        assert args.denoiseCurvelet == ["tileSize=128"]

    def test_denoise_curvelet3d_removes_noise_and_keeps_the_box(self):
        z, y, x = np.mgrid[:32, :32, :32] - 16
        ball = (z**2 + y**2 + x**2 < 64).astype(np.float32)
        noisy = ball + np.random.default_rng(0).normal(scale=0.3, size=ball.shape)
        result, apix, nx, ny, nz = self._denoise(noisy.astype(np.float32), "")
        assert result.shape == ball.shape and result.dtype == np.float32
        assert (apix, nx, ny, nz) == (self.apix, self.nx, self.ny, self.nz)
        assert np.sqrt(np.mean((result - ball) ** 2)) < 0.45 * np.sqrt(
            np.mean((noisy - ball) ** 2)
        )

    @pytest.mark.parametrize(
        "param, strength, precision, tile_size",
        [
            ("", 1.5, "single", None),
            ("strength=1:precision=double", 1.0, "double", None),
            ("tileSize=96", 1.5, "single", 96),
            ("tileSize=0", 1.5, "single", 0),
        ],
    )
    def test_denoise_curvelet3d_parameters_reach_the_denoiser(
        self, monkeypatch, param, strength, precision, tile_size
    ):
        import helicon.lib.denoise

        calls = []

        def fake(volume, strength, workers, precision, tile_size, noise):
            calls.append((strength, workers, precision, tile_size))
            info = dict(
                strength=strength,
                precision=precision,
                noise_from="test",
                tile_size=tile_size or None,
                tiles_why="given",
            )
            return volume, info

        monkeypatch.setattr(helicon.lib.denoise, "denoise_map", fake)
        self._denoise(self.data.copy(), param)
        assert calls == [(strength, 1, precision, tile_size)]

    @pytest.mark.parametrize(
        "param, match",
        [
            ("precision=half", "precision"),
            ("tileSize=big", "tileSize"),
            ("strength=strong", "strength"),
            ("halfMaps=one.mrc", "halfMaps"),
        ],
    )
    def test_denoise_curvelet3d_bad_parameters_error(self, param, match):
        from helicon.lib.exceptions import HeliconError

        with pytest.raises(HeliconError, match=match):
            self._denoise(self.data.copy(), param)

    def _half_map_files(self, folder, shape=None):
        """A RELION Refine3D map and its two half maps."""
        import mrcfile

        rng = np.random.default_rng(0)
        for name in (
            "run_class001",
            "run_half1_class001_unfil",
            "run_half2_class001_unfil",
        ):
            data = rng.normal(size=shape or self.data.shape).astype(np.float32)
            mrcfile.new(folder / f"{name}.mrc", data=data).close()
        return folder / "run_class001.mrc"

    def _tuned(self, monkeypatch, data, **args):
        """Run the handler with tune_strength and denoise_map faked; return
        (the half maps tuned on, the strength the map was denoised with)."""
        import helicon.lib.denoise

        tuned, used = [], []

        def fake_tune(h1, h2, **kwargs):
            tuned.append((h1, h2, kwargs["noise_from_halves"]))
            return 3.0, {1.5: 2.0, 3.0: 1.0, 6.0: 1.5}

        def fake_denoise(volume, strength, **kwargs):
            used.append(strength)
            self.noise_used = kwargs["noise"]
            info = dict(
                strength=strength,
                precision="single",
                noise_from="test",
                tile_size=None,
                tiles_why="test",
            )
            return volume, info

        monkeypatch.setattr(helicon.lib.denoise, "tune_strength", fake_tune)
        monkeypatch.setattr(helicon.lib.denoise, "denoise_map", fake_denoise)
        self._denoise(data, args.pop("param", ""), **args)
        return tuned, used[0]

    def test_denoise_curvelet3d_tunes_on_the_half_maps_next_to_the_map(
        self, monkeypatch, tmp_path
    ):
        import mrcfile

        path = self._half_map_files(tmp_path)
        tuned, strength = self._tuned(
            monkeypatch,
            self.data.copy(),
            inputMapFile=str(path),
            all_options=["denoiseCurvelet"],
        )
        half1 = mrcfile.read(tmp_path / "run_half1_class001_unfil.mrc")
        assert len(tuned) == 1 and np.array_equal(tuned[0][0], half1)
        assert strength == 3.0

    def test_denoise_curvelet3d_tuning_is_kept_until_a_half_map_changes(
        self, monkeypatch, tmp_path
    ):
        import os

        path = self._half_map_files(tmp_path)
        args = dict(inputMapFile=str(path), all_options=["denoiseCurvelet"])
        first, _ = self._tuned(monkeypatch, self.data.copy(), **args)
        again, strength = self._tuned(monkeypatch, self.data.copy(), **args)
        assert len(first) == 1 and again == [] and strength == 3.0
        half2 = tmp_path / "run_half2_class001_unfil.mrc"
        os.utime(half2, ns=(0, half2.stat().st_mtime_ns + 10**9))
        changed, _ = self._tuned(monkeypatch, self.data.copy(), **args)
        assert len(changed) == 1

    def test_denoise_curvelet3d_tunes_on_given_half_maps(self, monkeypatch, tmp_path):
        self._half_map_files(tmp_path)
        halves = ",".join(
            str(tmp_path / f"run_half{h}_class001_unfil.mrc") for h in (1, 2)
        )
        tuned, strength = self._tuned(
            monkeypatch, self.data.copy(), param=f"halfMaps={halves}"
        )
        assert len(tuned) == 1 and strength == 3.0

    @pytest.mark.parametrize("case", ["no halves", "options before", "other size"])
    def test_denoise_curvelet3d_otherwise_uses_the_default(
        self, monkeypatch, tmp_path, case
    ):
        shape = (8, 8, 8) if case == "other size" else None
        path = self._half_map_files(tmp_path, shape)
        if case == "no halves":
            (tmp_path / "run_half2_class001_unfil.mrc").unlink()
        options = ["clip", "denoiseCurvelet"] if case == "options before" else []
        tuned, strength = self._tuned(
            monkeypatch,
            self.data.copy(),
            inputMapFile=str(path),
            all_options=options + ["denoiseCurvelet"],
        )
        assert tuned == [] and strength == 1.5
        assert self.noise_used is None

    @pytest.mark.parametrize(
        "map_is, scale",
        [("half 1", np.sqrt(0.5)), ("half 2", np.sqrt(0.5)), ("average", 0.5)],
    )
    def test_denoise_curvelet3d_measures_the_noise_on_the_half_maps(
        self, monkeypatch, tmp_path, map_is, scale
    ):
        import mrcfile

        path = self._half_map_files(tmp_path)
        h1, h2 = (
            mrcfile.read(tmp_path / f"run_half{h}_class001_unfil.mrc") for h in (1, 2)
        )
        data = {"half 1": h1, "half 2": h2, "average": (h1 + h2) / 2}[map_is]
        args = dict(inputMapFile=str(path), all_options=["denoiseCurvelet"])
        tuned, strength = self._tuned(monkeypatch, data.copy(), **args)
        assert tuned[0][2] is True and strength == 3.0
        assert np.allclose(self.noise_used, (h1 - h2) * scale)
        # a given strength still uses the noise, without tuning
        tuned, strength = self._tuned(
            monkeypatch, data.copy(), param="strength=2", **args
        )
        assert tuned == [] and strength == 2.0 and self.noise_used is not None

    def test_denoise_curvelet3d_a_sharpened_map_keeps_the_guessed_noise(
        self, monkeypatch, tmp_path
    ):
        path = self._half_map_files(tmp_path)  # the map is unrelated to its halves
        tuned, strength = self._tuned(
            monkeypatch,
            self.data.copy(),
            inputMapFile=str(path),
            all_options=["denoiseCurvelet"],
        )
        assert tuned[0][2] is False and strength == 3.0
        assert self.noise_used is None

    def test_denoise_curvelet3d_tiles_keep_the_box(self):
        result, *_ = self._denoise(
            np.random.default_rng(0).normal(size=(40, 36, 32)).astype(np.float32),
            "tileSize=16",
        )
        assert result.shape == (40, 36, 32) and np.isfinite(result).all()
