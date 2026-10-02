"""Regression tests for bugs fixed in the command modules and their plugins."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import mrcfile
import numpy as np
import pandas as pd
import pytest

import helicon
from helicon.lib.exceptions import HeliconError, HeliconFileExistsError


def _write_mrc(path, data, apix=1.0):
    with mrcfile.new(
        str(path), data=np.asarray(data, dtype=np.float32), overwrite=True
    ) as mrc:
        mrc.voxel_size = apix


# ---------------------------------------------------------------------------
# plugin option dispatch
# ---------------------------------------------------------------------------


class TestPluginOptionsInArgv:
    def test_only_plugin_options_in_order(self):
        from helicon.plugins import plugin_options_in_argv

        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", type=int)
        parser.add_argument("--foo", type=int)
        parser.add_argument("--bar-opt", dest="bar", action="append")
        argv = ["in", "--verbose=2", "--bar-opt", "1", "--foo", "1", "--bar-opt=2"]
        assert plugin_options_in_argv(argv, parser, ["foo", "bar"]) == [
            "bar",
            "foo",
            "bar",
        ]


class TestImages2starCheckArgsOptions:
    def test_infrastructure_options_are_not_dispatched(self, tmp_path):
        from helicon.commands import images2star

        out = str(tmp_path / "out.star")
        argv = [
            "in.star",
            out,
            "--csparcPassthroughFiles",
            "p.cs",
            "--subset",
            "1",
            "--sets",
            "2",
            "--ppid",
            "3",
            "--select",
            "rlnClassNumber",
            "1",
        ]
        parser = argparse.ArgumentParser()
        images2star.add_args(parser)
        args = parser.parse_args(argv)
        with patch.object(sys, "argv", ["helicon", "images2star"] + argv):
            args = images2star.check_args(args, parser)
        # --sets is a plugin option (it keeps every n-th particle); --subset is not
        assert args.all_options == ["sets", "select"]


class TestProc3dCheckArgsOptions:
    def test_output_map_file_flag_is_not_dispatched(self, tmp_path):
        from helicon.commands import proc3d

        out = str(tmp_path / "out.mrc")
        argv = ["in.mrc", "--outputMapFile", out, "--flip_hand", "x"]
        parser = argparse.ArgumentParser()
        proc3d.add_args(parser)
        args = parser.parse_args(argv)
        with patch.object(sys, "argv", ["helicon", "proc3d"] + argv):
            args = proc3d.check_args(args, parser)
        assert args.all_options == ["flip_hand"]
        assert args.outputMapFile == Path(out)


# ---------------------------------------------------------------------------
# cryosparc
# ---------------------------------------------------------------------------


def _make_cs_particles(project_dir):
    from cryosparc.dataset import Dataset

    job_dir = project_dir / "J1"
    job_dir.mkdir(parents=True)
    n = 4
    data = Dataset(
        [
            ("uid", np.arange(n, dtype="u8")),
            ("location/micrograph_uid", np.array([1, 1, 2, 2], dtype="u8")),
            (
                "location/micrograph_path",
                np.array(["J1/a.mrc"] * 2 + ["J1/b.mrc"] * 2, dtype=object),
            ),
            ("location/micrograph_shape", np.array([[64, 64]] * n, dtype="u4")),
            ("location/micrograph_psize_A", np.full(n, 1.0, dtype="f4")),
            ("location/center_x_frac", np.array([0.3, 0.6, 0.4, 0.5], "f4")),
            ("location/center_y_frac", np.array([0.3, 0.6, 0.5, 0.4], "f4")),
        ]
    )
    cs_file = job_dir / "particles.cs"
    data.save(str(cs_file))
    rng = np.random.default_rng(0)
    for name in ["a.mrc", "b.mrc"]:
        _write_mrc(job_dir / name, rng.normal(size=(64, 64)))
    return cs_file


def _run_cryosparc(argv):
    from helicon.commands import cryosparc

    parser = argparse.ArgumentParser()
    cryosparc.add_args(parser)
    args = parser.parse_args(argv)
    with patch.object(sys, "argv", ["helicon", "cryosparc"] + argv):
        args = cryosparc.check_args(args, parser)
    cryosparc.main(args)
    return args


class TestCryosparcSaveOutput:
    def test_cs_file_result_is_saved(self, tmp_path, monkeypatch):
        from cryosparc.dataset import Dataset

        cs_file = _make_cs_particles(tmp_path / "P1")
        outdir = tmp_path / "out"
        outdir.mkdir()
        monkeypatch.chdir(outdir)
        args = _run_cryosparc(
            ["--csFile", str(cs_file), "--splitByMicrograph", "1", "--verbose", "0"]
        )
        assert args.all_options == ["splitByMicrograph"]
        output = outdir / "particles_per-micrograph-split.cs"
        assert output.exists()
        result = Dataset.load(str(output))
        assert len(result) == 4
        assert sorted(np.unique(result["alignments3D/split"])) == [0, 1]

    def test_output_cs_filename(self):
        from helicon.commands import cryosparc

        args = argparse.Namespace(
            csFile=["/x/J1/a.cs", "/x/J2/b.cs"], projectID=None, jobID=[]
        )
        assert cryosparc.output_cs_filename(args, "") == "a-b.output.cs"
        assert cryosparc.output_cs_filename(args, "->2 group") == "a-b_2-group.cs"
        args = argparse.Namespace(csFile=[], projectID="P1", jobID=["J5"])
        assert cryosparc.output_cs_filename(args, "->x/y") == "P1_J5_x_y.cs"

    def test_extract_particles_from_cs_file(self, tmp_path, monkeypatch):
        from cryosparc.dataset import Dataset

        cs_file = _make_cs_particles(tmp_path / "P1")
        outdir = tmp_path / "out"
        outdir.mkdir()
        monkeypatch.chdir(outdir)
        _run_cryosparc(
            [
                "--csFile",
                str(cs_file),
                "--saveLocal",
                "1",
                "--cpu",
                "1",
                "--verbose",
                "0",
                "--extractParticles",
                "box_size=16:fft_crop_size=8",
            ]
        )
        outputs = list(outdir.glob("*.cs"))
        assert len(outputs) == 1
        result = Dataset.load(str(outputs[0]))
        assert len(result) == 4
        for f in np.unique(result["blob/path"]):
            with mrcfile.open(str(outdir / f)) as mrc:
                assert mrc.data.shape == (2, 8, 8)

    def test_copy_exposure_group_parameters_uses_server_client(self):
        from helicon.plugins.cryosparc import copyexposuregroupparameters as mod

        class Client:
            def find_job(self, project_id, job_id):
                raise RuntimeError(f"find_job {project_id} {job_id}")

        args = argparse.Namespace(verbose=0, projectID="P1", cryosparc_client=Client())
        with pytest.raises(RuntimeError, match="find_job P1 J7"):
            mod.handle(None, args, {}, "source_job_id=J7", "", set(), "", "", [])


# ---------------------------------------------------------------------------
# trueFSC
# ---------------------------------------------------------------------------


def _make_half_maps(tmp_path, seed=0, n=32):
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[:n, :n, :n] - n / 2
    signal = np.exp(-(x**2 + y**2 + z**2) / (2 * 4.0**2))
    signal += 0.5 * np.exp(-((x - 5) ** 2 + y**2 + (z - 3) ** 2) / (2 * 2.0**2))
    m1 = signal + 0.05 * rng.normal(size=signal.shape)
    m2 = signal + 0.05 * rng.normal(size=signal.shape)
    f1, f2 = tmp_path / "half1.mrc", tmp_path / "half2.mrc"
    _write_mrc(f1, m1, apix=2.0)
    _write_mrc(f2, m2, apix=2.0)
    return f1, f2


class TestTrueFSC:
    def test_file_identity_changes_when_overwritten(self, tmp_path):
        from helicon.commands import trueFSC

        f = tmp_path / "a.mrc"
        _write_mrc(f, np.zeros((4, 4, 4)))
        id1 = trueFSC._file_identity(f)
        _write_mrc(f, np.zeros((6, 6, 6)))
        id2 = trueFSC._file_identity(f)
        assert id1[0] == str(f.resolve())
        assert id1 != id2

    def test_outputs_written_for_each_requested_name(self, tmp_path):
        from helicon.commands import trueFSC

        f1, f2 = _make_half_maps(tmp_path)
        for name in ["first.pdf", "second.pdf"]:
            result = trueFSC.compute_truefsc(
                f1, f2, str(tmp_path / name), mask_soft=4, refine_mask=0
            )
            assert (tmp_path / name).exists()
            assert result["plot_file"] == str(tmp_path / name)
            assert (tmp_path / name).with_suffix(".true.txt").exists()

    def test_overwritten_half_map_is_not_stale(self, tmp_path):
        from helicon.commands import trueFSC

        f1, f2 = _make_half_maps(tmp_path)
        plot = str(tmp_path / "fsc.pdf")
        r1 = trueFSC.compute_truefsc(f1, f2, plot, mask_soft=4, refine_mask=0)
        rng = np.random.default_rng(1)
        _write_mrc(f2, rng.normal(size=(32, 32, 32)), apix=2.0)  # pure noise
        r2 = trueFSC.compute_truefsc(f1, f2, plot, mask_soft=4, refine_mask=0)
        assert r2["resolution_unmasked"] != r1["resolution_unmasked"]

    def test_mask_slope_search_is_keyed_by_file_identity(self, tmp_path):
        from helicon.commands import trueFSC

        f1, f2 = _make_half_maps(tmp_path)
        calls = []

        def fake_slope(input_files, params, *args):
            calls.append(input_files)
            return 3.0

        with patch.object(trueFSC, "_optimal_mask_slope", fake_slope):
            trueFSC.compute_truefsc(f1, f2, str(tmp_path / "fsc.pdf"))
        assert calls == [[trueFSC._file_identity(f1), trueFSC._file_identity(f2)]]

    def test_optimal_mask_slope_search_runs(self, tmp_path):
        from helicon.commands import trueFSC

        f1, f2 = _make_half_maps(tmp_path)
        search = getattr(trueFSC._optimal_mask_slope, "__wrapped__", None)
        if search is None:
            pytest.skip("cache wrapper does not expose the function")
        with patch.object(trueFSC, "_optimal_mask_slope", search):
            result = trueFSC.compute_truefsc(f1, f2, str(tmp_path / "fsc.pdf"))
        assert np.isfinite(result["resolution"])

    def test_user_mask_files_are_closed(self, tmp_path):
        from helicon.commands import trueFSC

        f1, f2 = _make_half_maps(tmp_path)
        mask = tmp_path / "mask.mrc"
        _write_mrc(mask, np.ones((32, 32, 32)), apix=2.0)
        opened = []
        real_open = mrcfile.open

        def spy_open(*args, **kwargs):
            mrc = real_open(*args, **kwargs)
            opened.append(mrc)
            return mrc

        with patch.object(mrcfile, "open", spy_open):
            trueFSC.compute_truefsc(
                f1, f2, str(tmp_path / "fsc.pdf"), mask_file=[str(mask), str(mask)]
            )
        assert opened
        assert all(m._iostream is None or m._iostream.closed for m in opened)

    def test_soft_mask_falls_to_zero_at_width(self):
        from helicon.commands import trueFSC
        from scipy.ndimage import distance_transform_edt

        n, w = 48, 8.0
        mask = np.zeros((n, n, n))
        mask[16:32, 16:32, 16:32] = 1
        soft = trueFSC._soft_mask(mask, w)
        dist = distance_transform_edt(mask == 0)
        near_outer = (dist > 0.85 * w) & (dist <= w)
        mid = (dist > 0.45 * w) & (dist < 0.55 * w)
        assert soft[near_outer].max() < 0.1  # no hard drop from 0.5 to 0
        assert abs(soft[mid].mean() - 0.5) < 0.1

    def test_import_does_not_set_matplotlib_backend(self):
        code = (
            "import matplotlib; matplotlib.use('pdf'); "
            "import helicon.commands.trueFSC; print(matplotlib.get_backend())"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip().splitlines()[-1] == "pdf"


# ---------------------------------------------------------------------------
# images2star plugins
# ---------------------------------------------------------------------------


def _particles(n_per_mgraph=2, mgraphs=("m1.mrc", "m2.mrc")):
    rows = []
    for mi, m in enumerate(mgraphs):
        for i in range(n_per_mgraph):
            rows.append(
                {
                    "rlnImageName": f"{mi * n_per_mgraph + i + 1:06d}@stack.mrcs",
                    "rlnMicrographName": m,
                    "rlnDefocusU": 10000.0,
                    "rlnDefocusV": 9000.0,
                    "rlnDefocusAngle": 0.0,
                    "rlnOpticsGroup": 1,
                }
            )
    data = pd.DataFrame(rows)
    data.attrs["optics"] = pd.DataFrame(
        {"rlnOpticsGroup": [1], "rlnImageSize": [32], "rlnImagePixelSize": [1.0]}
    )
    return data


class TestCopyCtf:
    def test_astigmatism_angle_average_has_180_degree_period(self):
        from helicon.plugins.images2star.copyctf import average_ctf_per_micrograph

        data = pd.DataFrame(
            {
                "rlnMicrographName": ["m1", "m1"],
                "rlnDefocusU": [11000.0, 11000.0],
                "rlnDefocusV": [9000.0, 9000.0],
                "rlnDefocusAngle": [89.0, -89.0],  # -89 is the same as 91
                "rlnImageName": ["1@a", "2@a"],  # a non-numeric column
            }
        )
        ret = average_ctf_per_micrograph(data)
        assert abs(abs(ret.loc["m1", "mean_astig_angle"]) - 90) < 1e-6
        assert abs(ret.loc["m1", "mean_astig"] - 1000 * np.cos(np.deg2rad(2))) < 1e-6
        assert abs(ret.loc["m1", "mean_defocus"] - 10000) < 1e-6

    def test_handle_copies_ctf_and_optics(self):
        from helicon.plugins.images2star import copyctf

        data = _particles()
        data2 = _particles()
        data2["rlnDefocusU"] = 21000.0
        data2["rlnDefocusV"] = 19000.0
        data2["rlnDefocusAngle"] = 30.0
        data2["rlnCtfBfactor"] = 5.0
        data2.attrs["optics"] = data2.attrs["optics"].assign(rlnBeamTiltX=0.3)
        args = argparse.Namespace(
            folder=[], ignoreBadParticlePath=0, ignoreBadMicrographPath=1, verbose=0
        )
        with patch.object(helicon, "images2dataframe", return_value=data2):
            out, _ = copyctf.handle(data, args, {}, "ref.star")
        assert np.allclose(out["rlnDefocusU"], 21000.0)
        assert np.allclose(out["rlnDefocusV"], 19000.0)
        assert np.allclose(out["rlnDefocusAngle"], 30.0)
        assert np.allclose(out["rlnCtfBfactor"], 5.0)
        assert out.attrs["optics"]["rlnBeamTiltX"].iloc[0] == 0.3


class TestCreateStackResize:
    @pytest.mark.parametrize("newsize", [16, 48])
    def test_resize_keeps_smooth_image(self, newsize):
        from helicon.plugins.images2star.createstack import resize_image

        def gauss(n):
            y, x = (np.mgrid[:n, :n] - n // 2) / n
            return 1.0 + np.exp(-(x**2 + y**2) / (2 * 0.1**2))

        out = resize_image(gauss(32), newsize)
        assert out.shape == (newsize, newsize)
        np.testing.assert_allclose(out, gauss(newsize), atol=1e-3)


class TestCreateStack:
    def test_create_rescaled_stack(self, tmp_path):
        from helicon.plugins.images2star import createstack

        stack = tmp_path / "stack.mrcs"
        images = np.random.default_rng(0).normal(size=(4, 32, 32)) + 2.0
        _write_mrc(stack, images, apix=1.0)
        data = _particles()
        data["rlnImageName"] = [f"{i + 1:06d}@{stack}" for i in range(4)]
        out = tmp_path / "new.mrcs"
        args = argparse.Namespace(verbose=0)
        ret, index_d = createstack.handle(
            data, args, {"createStack": 0}, f"{out}:rescale2size=16:float16=0"
        )
        with mrcfile.open(str(out)) as mrc:
            assert mrc.data.shape == (4, 16, 16)
            assert abs(float(mrc.voxel_size.x) - 2.0) < 1e-4
            assert abs(mrc.data[1].mean() - images[1].mean()) < 1e-3
        assert list(ret["rlnImageName"]) == [f"{i}@{out}" for i in range(1, 5)]
        assert ret.attrs["optics"]["rlnImageSize"].iloc[0] == 16
        assert ret.attrs["optics"]["rlnImagePixelSize"].iloc[0] == 2.0
        assert index_d["createStack"] == 1


class TestMinStack:
    def test_min_stack_keeps_particle_order(self, tmp_path):
        from helicon.plugins.images2star import minstack

        stack = tmp_path / "stack.mrcs"
        images = np.arange(5)[:, None, None] * np.ones((5, 4, 4))
        _write_mrc(stack, images)
        data = pd.DataFrame({"rlnImageName": [f"4@{stack}", f"2@{stack}"]})
        args = argparse.Namespace(output_starFile=str(tmp_path / "out.star"), verbose=0)
        ret, _ = minstack.handle(data, args, {"minStack": 0}, 1)
        new_file = tmp_path / "out" / "stack.mrcs"
        with mrcfile.open(str(new_file)) as mrc:
            assert list(mrc.data[:, 0, 0]) == [3.0, 1.0]
        assert list(ret["rlnImageName"]) == [
            f"000001@{new_file}",
            f"000002@{new_file}",
        ]


class TestSetCtf:
    def test_read_ctfparm_file_and_set_ctf(self, tmp_path):
        from helicon.plugins.images2star import setctf

        ctfparm = tmp_path / "ctfparm.txt"
        ctfparm.write_text(
            "mic1.mrc\t-2.0,0.1,30,100,1,0.07,0,0,0,0,300,2.7,1.5,0\n"
            "mic2\t-1.5,100,1,0.1,0,0,0,0,200,2.0,1.2,0\n"
        )
        parms = setctf.read_ctfparm_file(str(ctfparm))
        assert parms["mic1"]["defocus"] == 2.0
        assert parms["mic1"]["dfdiff"] == 0.1
        assert abs(parms["mic1"]["ampcont"] - 7) < 1e-9
        assert parms["mic2"]["voltage"] == 200

        data = pd.DataFrame({"rlnImageName": ["1@mic1.mrcs", "1@mic2.mrcs"]})
        with patch.object(
            helicon,
            "eman_astigmatism_to_relion",
            lambda defocus, dfdiff, dfang: (defocus * 1e4, defocus * 1e4, dfang),
        ):
            ret, _ = setctf.handle(data, argparse.Namespace(), {"setCTF": 0}, ctfparm)
        assert list(ret["rlnVoltage"]) == [300, 200]
        assert list(ret["rlnSphericalAberration"]) == [2.7, 2.0]
        assert ret["rlnDefocusU"].iloc[1] == 15000


class TestMaskGold:
    def test_mask_gold_micrographs(self, tmp_path, monkeypatch):
        pytest.importorskip("skimage")
        from helicon.plugins.images2star import maskgold

        rng = np.random.default_rng(0)
        mgraph = rng.normal(size=(64, 64))
        mgraph[20:30, 20:30] += 20  # a dense "gold" particle
        mfile = tmp_path / "m1.mrc"
        _write_mrc(mfile, mgraph)
        data = pd.DataFrame({"rlnMicrographName": [str(mfile)]})
        args = argparse.Namespace(output_starFile=str(tmp_path / "o.star"), verbose=0)
        outdir = tmp_path / "masked"
        ret, _ = maskgold.handle(
            data,
            args,
            {"maskGold": 0},
            f"gradient_sigma=-1:min_area=4:outdir={outdir}",
        )
        out_file = outdir / "m1.mrc"
        assert ret["rlnMicrographName"].iloc[0] == out_file.as_posix()
        with mrcfile.open(str(out_file)) as mrc:
            assert mrc.data[25, 25] < 5  # the dense pixels are replaced


class TestOtherImages2starPlugins:
    def test_replace_image_name(self, tmp_path):
        from helicon.plugins.images2star import replaceimagename

        stack = tmp_path / "s.mrcs"
        _write_mrc(stack, np.zeros((2, 4, 4)))
        data = pd.DataFrame({"rlnImageName": ["1@a.mrcs", "2@a.mrcs"]})
        ret, _ = replaceimagename.handle(
            data, argparse.Namespace(), {"replaceImageName": 0}, str(stack)
        )
        assert list(ret["rlnImageName"]) == [f"000001@{stack}", f"000002@{stack}"]


# ---------------------------------------------------------------------------
# images2star --splitNumSets
# ---------------------------------------------------------------------------


def _run_images2star_main(argv, data):
    from helicon.commands import images2star

    parser = argparse.ArgumentParser()
    images2star.add_args(parser)
    args = parser.parse_args(argv)
    with patch.object(sys, "argv", ["helicon", "images2star"] + argv):
        args = images2star.check_args(args, parser)
    written = {}

    def fake_save(df, filename):
        written[filename] = df
        Path(filename).write_text("")

    with (
        patch.object(helicon, "images2dataframe", return_value=data),
        patch.object(helicon, "dataframe2file", fake_save),
        patch.object(helicon, "log_command_line"),
    ):
        images2star.main(args)
    return written


class TestSplitNumSets:
    def test_subset_filenames_in_output_folder(self):
        from helicon.commands.images2star import split_subset_filenames

        assert split_subset_filenames("d/x.star", 2, "evenodd") == [
            Path("d/x.e.star"),
            Path("d/x.o.star"),
        ]
        assert split_subset_filenames("d/x.star", 3, "random") == [
            Path(f"d/x.subset-{i}.star") for i in range(3)
        ]

    def test_split_writes_next_to_output_and_respects_force(self, tmp_path):
        data = pd.DataFrame({"rlnMicrographName": [f"m{i}.mrc" for i in range(4)]})
        out = tmp_path / "sub" / "out.star"
        out.parent.mkdir()
        argv = [
            "in.star",
            str(out),
            "--splitNumSets",
            "2",
            "--verbose",
            "0",
            "--path",
            "absolute",
        ]
        written = _run_images2star_main(argv, data.copy())
        assert sorted(written) == [
            str(out.parent / "out.e.star"),
            str(out.parent / "out.o.star"),
        ]
        assert list(written[str(out.parent / "out.e.star")]["rlnRandomSubset"]) == [
            1,
            1,
        ]
        with pytest.raises(HeliconFileExistsError):
            _run_images2star_main(argv, data.copy())
        written = _run_images2star_main(argv + ["--force", "1"], data.copy())
        assert len(written) == 2

    def test_no_global_copy_on_write_setting(self):
        import inspect
        from helicon.commands import images2star

        assert "copy_on_write" not in inspect.getsource(images2star)


# ---------------------------------------------------------------------------
# proc3d plugins
# ---------------------------------------------------------------------------


class TestFftResample:
    def test_negative_values_are_kept(self):
        pytest.importorskip("finufft")
        from helicon.plugins.proc3d import fft_resample

        z, y, x = np.mgrid[:16, :16, :16] - 8
        data = (-1.0 - np.exp(-(x**2 + y**2 + z**2) / 8.0)).astype(np.float32)
        args = argparse.Namespace(verbose=0)
        out, apix, nx, ny, nz = fft_resample.handle(
            data, args, {}, "new_nx=32:new_ny=32:new_nz=32", 2.0, 16, 16, 16
        )
        assert out.shape == (32, 32, 32) and (nx, ny, nz) == (32, 32, 32)
        assert apix == 1.0
        assert out.max() < 0
        assert abs(out.mean() - data.mean()) < 0.05


class TestZMovingAverage:
    def _run(self, data, param, apix=1.0):
        from helicon.plugins.proc3d import z_moving_average as mod

        nz, ny, nx = data.shape
        index_d = {mod.option_name: 0}
        out, *_ = mod.handle(
            data, argparse.Namespace(verbose=0), index_d, param, apix, nx, ny, nz
        )
        return out

    def test_length_of_one_slice_is_identity(self):
        data = np.zeros((9, 2, 2), dtype=np.float32)
        data[4] = 1
        out = self._run(data, "length=1")
        np.testing.assert_allclose(out, data)

    def test_window_is_centred(self):
        data = np.zeros((9, 2, 2), dtype=np.float32)
        data[4] = 3
        out = self._run(data, "n_pixel=3")
        np.testing.assert_allclose(out[:, 0, 0], [0, 0, 0, 1, 1, 1, 0, 0, 0])

    def test_length_in_angstrom(self):
        data = np.zeros((9, 2, 2), dtype=np.float32)
        data[4] = 5
        out = self._run(data, "length=10", apix=2.0)  # 5 slices
        np.testing.assert_allclose(out[:, 0, 0], [0, 0, 1, 1, 1, 1, 1, 0, 0])

    def test_window_longer_than_map(self):
        with pytest.raises(HeliconError):
            self._run(np.zeros((3, 2, 2)), "n_pixel=5")


# ---------------------------------------------------------------------------
# helicon entry point and the launcher commands
# ---------------------------------------------------------------------------


class TestHeliconCheckArgsErrors:
    def test_unexpected_check_args_error_is_reported(self, monkeypatch, caplog):
        from helicon import helicon as entry
        from helicon.commands import proc3d

        def broken_check_args(args, parser):
            raise ValueError("boom in check_args")

        monkeypatch.setattr(proc3d, "check_args", broken_check_args)
        monkeypatch.setattr(sys, "argv", ["helicon", "proc3d", "in.mrc"])
        with caplog.at_level(logging.ERROR, logger="helicon.helicon"):
            with pytest.raises(SystemExit):
                entry._get_commands(["proc3d"], [], [], [])
        assert "boom in check_args" in caplog.text


class TestHOMContainerCImport:
    def test_import_without_pytz(self):
        code = (
            "import sys; sys.modules['pytz'] = None; "
            "import helicon.commands.HOM_containerC"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


class TestStreamlitLaunchers:
    @pytest.mark.parametrize("name", ["procart", "ctfSimulation"])
    def test_failure_is_reported(self, name):
        import importlib

        mod = importlib.import_module(f"helicon.commands.{name}")
        with patch("subprocess.call", return_value=1) as call:
            with pytest.raises(HeliconError, match="streamlit"):
                mod.main(argparse.Namespace())
        assert call.call_args[0][0][:3] == [sys.executable, "-m", "streamlit"]

    @pytest.mark.parametrize("name", ["procart", "ctfSimulation"])
    def test_success(self, name):
        import importlib

        mod = importlib.import_module(f"helicon.commands.{name}")
        with patch("subprocess.call", return_value=0):
            mod.main(argparse.Namespace())
