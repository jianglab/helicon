"""Successful-job discovery and exact, paginated Class2D PDF summaries."""

import argparse
import logging
from pathlib import Path
import sys

import mrcfile
import numpy as np
import pandas as pd
import pytest
import starfile

from helicon.commands import images2star
from helicon.plugins.images2star import summary2d as summary
from helicon.lib.exceptions import (
    HeliconError, HeliconFileExistsError, HeliconValidationError,
)


def make_stack(job, prefix="run_it025", classes=3, assignments=(1, 1, 3), legacy=False):
    job.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[-1:1:32j, -1:1:32j]
    pixels = np.stack([
        np.exp(-((x - 0.2 * np.sin(i)) ** 2 + 2 * y ** 2) * (i + 2))
        for i in range(classes)
    ]).astype(np.float32)
    with mrcfile.new(job / f"{prefix}_classes.mrcs", overwrite=True) as mrc:
        mrc.set_data(pixels)
        mrc.set_image_stack()
    table = pd.DataFrame({"rlnClassNumber": list(assignments)})
    tables = {"": table} if legacy else {
        "optics": pd.DataFrame({"rlnOpticsGroup": [1]}), "particles": table,
    }
    starfile.write(tables, job / f"{prefix}_data.star", overwrite=True)
    return job


def make_job(root, name="job001", success=True, **kwargs):
    job = make_stack(root / name, **kwargs)
    if success:
        (job / "RELION_JOB_EXIT_SUCCESS").touch()
    return job


def symlink(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if sys.platform == "win32":
            # Junctions exercise real directory alias resolution without the
            # Windows privilege required to create symbolic links.
            import _winapi

            try:
                _winapi.CreateJunction(str(target), str(link))
                return
            except OSError:
                pass
        pytest.skip(f"Directory links unavailable: {exc}")


@pytest.mark.parametrize("selectors", [
    None, ["job001", "job003", "job010"], ["1", "003", "10"],
    ["1,3,10"], ["1-10"], ["job001", "3-10", "1"],
])
def test_selection_and_natural_order(tmp_path, selectors):
    for number in (10, 3, 1):
        (tmp_path / f"job{number:03d}").mkdir()
    jobs = summary.discover_jobs(tmp_path, selectors)
    assert [j.path.name for j in jobs] == ["job001", "job003", "job010"]


@pytest.mark.parametrize("selector", ["job999", "999", "11-20", "10-1", "../job001"])
def test_invalid_selection(tmp_path, selector):
    (tmp_path / "job001").mkdir()
    with pytest.raises(HeliconValidationError):
        summary.discover_jobs(tmp_path, [selector])


def test_aliases_deduplicated_in_mixed_selection(tmp_path):
    job = make_job(tmp_path)
    symlink(tmp_path / "good_classes", job)
    symlink(tmp_path / "another_alias", tmp_path / "good_classes")
    selected = summary.collect_summary(tmp_path, ["good_classes", "1-3", "job001"])
    assert len(selected.jobs) == 1
    assert selected.jobs[0].path == job.resolve()
    assert selected.jobs[0].aliases == ["another_alias", "good_classes"]
    assert not selected.skipped


def test_quoted_folder_name(tmp_path):
    make_job(tmp_path, "good classes")
    assert summary.discover_jobs(tmp_path, ["good classes"])[0].path.name == "good classes"


def test_alias_only_target(tmp_path):
    root = tmp_path / "Class2D"
    root.mkdir()
    job = make_job(tmp_path / "elsewhere", "job021")
    symlink(root / "external", job)
    assert summary.discover_jobs(root, ["21"])[0].path == job.resolve()
    result = summary.collect_summary(root)
    assert len(result.jobs) == 1


def test_broken_link_is_reported(tmp_path, monkeypatch):
    # Simulate lstat visibility on systems that cannot create dangling links.
    missing = tmp_path / "job099"
    real_iterdir = Path.iterdir
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(Path, "iterdir", lambda p: iter([missing]) if p == tmp_path
                        else real_iterdir(p))
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == missing or real_is_symlink(p))
    result = summary.collect_summary(tmp_path)
    assert len(result.skipped) == 1
    assert result.skipped[0][0] == "job099"


def test_latest_numeric_iteration_continuation_and_exact_counts(tmp_path):
    job = make_job(tmp_path, prefix="run_it9", assignments=(2,))
    make_stack(job, "run_it025", assignments=(3,))
    make_stack(job, "run_ct25_it100", assignments=(1, 3, 3, 3))
    result = summary.collect_summary(tmp_path)
    assert result.jobs[0].stacks[0].path.name == "run_ct25_it100_classes.mrcs"
    assert result.jobs[0].stacks[0].counts.tolist() == [1, 0, 3]


def test_all_stacks_at_latest_iteration_and_legacy_unnumbered(tmp_path):
    job = make_job(tmp_path)
    make_stack(job, "other_it025", assignments=(2,))
    make_job(tmp_path, "job002", prefix="run", legacy=True)
    result = summary.collect_summary(tmp_path)
    assert len(result.jobs[0].stacks) == 2
    assert result.jobs[1].stacks[0].counts.tolist() == [2, 0, 1]


def test_success_flag_required_and_no_fallback(tmp_path):
    make_job(tmp_path)
    failed = make_job(tmp_path, "job002", success=False)
    (failed / "RELION_JOB_EXIT_FAILURE").touch()
    make_job(tmp_path, "job003", success=False)
    job = make_job(tmp_path, "job004")
    make_stack(job, "run_it100")
    (job / "run_it100_data.star").unlink()
    result = summary.collect_summary(tmp_path)
    assert [j.path.name for j in result.jobs] == ["job001"]
    assert [name for name, reason in result.skipped] == ["job002", "job003", "job004"]
    assert "RELION_JOB_EXIT_SUCCESS" in result.skipped[0][1]
    assert "run_it100_data.star" in result.skipped[2][1]


@pytest.mark.parametrize("assignments", [(0,), (4,), (1.5,), (float("nan"),), ("bad",)])
def test_invalid_class_assignments_skip_job(tmp_path, assignments):
    make_job(tmp_path, assignments=assignments)
    result = summary.collect_summary(tmp_path)
    assert not result.jobs
    assert "rlnClassNumber" in result.skipped[0][1]


def test_missing_column_and_corrupt_stack_skip_jobs(tmp_path):
    job = make_job(tmp_path)
    starfile.write({"particles": pd.DataFrame({"rlnOther": [1]})},
                   job / "run_it025_data.star", overwrite=True)
    bad = make_job(tmp_path, "job002")
    (bad / "run_it025_classes.mrcs").write_bytes(b"broken")
    result = summary.collect_summary(tmp_path)
    assert not result.jobs
    assert len(result.skipped) == 2


def test_single_image_and_empty_particle_table(tmp_path):
    job = make_job(tmp_path, classes=1, assignments=())
    with mrcfile.new(job / "run_it025_classes.mrcs", overwrite=True) as mrc:
        mrc.set_data(np.zeros((32, 32), dtype=np.float32))
    result = summary.collect_summary(tmp_path)
    assert result.jobs[0].stacks[0].counts.tolist() == [0]


def parse_args(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["helicon", "images2star", *argv])
    parser = images2star.add_args(argparse.ArgumentParser())
    return images2star.check_args(parser.parse_args(argv), parser)


def test_cli_summary_dispatch_bypasses_dataset_pipeline(tmp_path, monkeypatch):
    make_job(tmp_path)
    args = parse_args(monkeypatch, [str(tmp_path), str(tmp_path / "summary.pdf"),
                                   "--summary2D", "--jobs", "1-3", "--jobs", "job001"])
    calls = []
    monkeypatch.setattr(summary, "summarize_class2d", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(images2star.helicon, "log_command_line", lambda: None)
    monkeypatch.setattr(images2star.helicon, "images2dataframe",
                        lambda *a, **kw: pytest.fail("Summary loaded a particle dataset"))
    images2star.main(args)
    assert calls[0][1]["jobs"] == ["1-3", "job001"]
    assert args.all_options == []


def test_summary_plugin_registration_and_handler(monkeypatch):
    from helicon.plugins.images2star import _plugins, dispatch
    from helicon.lib.images2star_engine import gui_operation_specs, operation_specs

    assert _plugins["summary2D"] is summary
    assert operation_specs()["summary2D"]["option_string"] == "--summary2D"
    assert "summary2D" not in gui_operation_specs()
    calls = []
    monkeypatch.setattr(summary, "summarize_class2d", lambda *a, **kw: calls.append((a, kw)))
    args = argparse.Namespace(input_imageFiles=["Class2D"], output_starFile="out.pdf",
                              jobs=["1-3"], force=1)
    index_d = {"summary2D": 0}
    assert dispatch("summary2D", None, args, index_d, False) == (None, index_d)
    assert not calls
    assert dispatch("summary2D", None, args, index_d, True) == (None, index_d)
    assert index_d["summary2D"] == 1
    assert calls[0][0] == ("Class2D", "out.pdf")
    assert calls[0][1]["overwrite"] is True


@pytest.mark.parametrize("extra", [["--sortby", "rlnClassNumber"], ["--first", "0"],
                                   ["--splitNumSets", "2"]])
def test_summary_rejects_transform_options(tmp_path, monkeypatch, extra):
    with pytest.raises(HeliconValidationError, match="cannot be combined"):
        parse_args(monkeypatch, [str(tmp_path), "out.pdf", "--summary2D", *extra])


def test_cli_validation_and_conversion_compatibility(tmp_path, monkeypatch):
    with pytest.raises(HeliconValidationError, match="requires --summary2D"):
        parse_args(monkeypatch, ["input.star", "out.star", "--jobs", "1"])
    with pytest.raises(HeliconValidationError, match=".pdf"):
        parse_args(monkeypatch, [str(tmp_path), "out.star", "--summary2D"])
    with pytest.raises(HeliconValidationError, match="one Class2D"):
        parse_args(monkeypatch, [str(tmp_path), str(tmp_path), "out.pdf", "--summary2D"])
    args = parse_args(monkeypatch, ["input.star", str(tmp_path / "out.star")])
    assert not args.summary2D


def test_pdf_pages_labels_counts_and_skips(tmp_path, caplog):
    pypdf = pytest.importorskip("pypdf")
    make_job(tmp_path, classes=21, assignments=(1, 1, 21))
    make_job(tmp_path, "job002", success=False)
    output = tmp_path / "summary.pdf"
    with caplog.at_level(logging.WARNING):
        result = summary.summarize_class2d(tmp_path, output)
    reader = pypdf.PdfReader(output)
    assert len(reader.pages) == 3
    overview, gallery, continuation = [p.extract_text() for p in reader.pages]
    assert "RELION_JOB_EXIT_SUCCESS" in overview
    assert "job002" in caplog.text
    assert "job001 / class 1\n2 particles" in gallery
    assert "job001 / class 2\n0 particles" in gallery
    assert "job001 / class 21\n1 particles" in continuation
    assert "3 / 3" in continuation
    # Matplotlib shares the image resource dictionary across PDF pages. Count
    # actual draw operations, not resources visible from each page.
    assert [sum(op == b"Do" for _, op in p.get_contents().operations)
            for p in reader.pages] == [0, 20, 1]
    assert len(result.jobs) == 1


def test_overwrite_and_failed_render_preserve_existing_pdf(tmp_path, monkeypatch):
    make_job(tmp_path)
    output = tmp_path / "summary.pdf"
    output.write_bytes(b"original")
    with pytest.raises(HeliconFileExistsError):
        summary.summarize_class2d(tmp_path, output)
    def fail(*args):
        raise OSError("render failed")
    monkeypatch.setattr(summary, "_render_pdf", fail)
    with pytest.raises(OSError, match="render failed"):
        summary.summarize_class2d(tmp_path, output, overwrite=True)
    assert output.read_bytes() == b"original"
    assert list(tmp_path.glob("*.pdf")) == [output]


def test_no_successful_jobs_writes_nothing(tmp_path):
    make_job(tmp_path, success=False)
    output = tmp_path / "summary.pdf"
    with pytest.raises(HeliconError, match="No successful"):
        summary.summarize_class2d(tmp_path, output)
    assert not output.exists()
