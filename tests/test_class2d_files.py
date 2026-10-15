"""Finding the other file of a 2D classification beside the one given."""

import pytest
from _source import contains

from helicon.webApps import class2d_files as c2


def _touch(folder, *names):
    for n in names:
        (folder / n).write_text("x")
    return [str(folder / n) for n in names]


class TestCompanion:
    def test_relion_iterations_pair_up(self, tmp_path):
        star, mrcs = _touch(tmp_path, "run_it025_data.star", "run_it025_classes.mrcs")
        _touch(tmp_path, "run_it024_data.star", "run_it024_classes.mrcs")
        assert c2.companion(star) == mrcs
        assert c2.companion(mrcs) == star

    def test_relion_without_iteration(self, tmp_path):
        star, mrcs = _touch(tmp_path, "run_data.star", "run_classes.mrcs")
        assert c2.companion(star) == mrcs

    def test_cryosparc_outputs_pair_up(self, tmp_path):
        cs, mrc = _touch(tmp_path, "J63_020_particles.cs", "J63_020_class_averages.mrc")
        _touch(tmp_path, "J63_passthrough_particles.cs", "J63_020_class_averages.cs")
        assert c2.companion(cs) == mrc
        assert c2.companion(mrc) == cs

    def test_nothing_for_urls_missing_files_or_other_names(self, tmp_path):
        (other,) = _touch(tmp_path, "particles.star")
        assert c2.companion("https://ftp.ebi.ac.uk/x/run_it020_data.star") is None
        assert c2.companion(str(tmp_path / "run_it001_data.star")) is None
        assert c2.companion(other) is None
        assert c2.companion("") is None

    def test_a_missing_partner_is_not_invented(self, tmp_path):
        (star,) = _touch(tmp_path, "run_it025_data.star")
        assert c2.companion(star) is None

    def test_the_folder_is_kept_as_given(self, tmp_path):
        real = tmp_path / "real"
        real.mkdir()
        _touch(real, "run_it025_data.star", "run_it025_classes.mrcs")
        link = tmp_path / "link"
        link.symlink_to(real)
        out = c2.companion(str(link / "run_it025_data.star"))
        assert out == str(link / "run_it025_classes.mrcs")


class TestNeedsFilling:
    def test_example_urls_and_empty_fields_are_replaced(self, tmp_path):
        (star,) = _touch(tmp_path, "run_it025_data.star")
        assert c2.needs_filling("https://ftp.ebi.ac.uk/x/run_it020_classes.mrcs", star)
        assert c2.needs_filling("", star)

    def test_a_file_chosen_in_the_same_folder_is_kept(self, tmp_path):
        star, mrcs = _touch(tmp_path, "run_it025_data.star", "other_classes.mrcs")
        assert not c2.needs_filling(mrcs, star)

    def test_a_file_from_another_folder_is_replaced(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (star,) = _touch(tmp_path / "a", "run_it025_data.star")
        (mrcs,) = _touch(tmp_path / "b", "run_it025_classes.mrcs")
        assert c2.needs_filling(mrcs, star)


@pytest.mark.parametrize("tab", ["abinitio3d_tab", "helical_pitch_tab"])
class TestTheTabsUseIt:
    def test_both_fields_fill_each_other(self, tab):
        import importlib
        import inspect

        src = inspect.getsource(importlib.import_module(f"helicon.webApps.tabs.{tab}"))
        # files on the server only: a URL has no folder to look in
        assert "@reactive.event(input.server_params)" in src
        assert "@reactive.event(input.server_classes)" in src
        assert "@reactive.event(input.url_params)" not in src
        assert "class2d_files.companion(" in src

    def test_tube_ids_are_split_on_load(self, tab):
        import importlib
        import inspect

        mod = importlib.import_module(f"helicon.webApps.tabs.{tab}")
        src = inspect.getsource(mod)
        assert "@reactive.event(params_raw, input.split_axis_distance)" in src
        assert "compute.split_distinct_filaments(raw, distance)" in src
        assert mod.BOOKMARK_DEFAULTS["split"] == ("split_axis_distance", 50)

    def test_the_server_mode_reads_its_own_fields(self, tab):
        import importlib
        import inspect

        mod = importlib.import_module(f"helicon.webApps.tabs.{tab}")
        src = inspect.getsource(mod)
        assert "helicon.shiny.source_modes(deployment.is_cloud())" in src
        assert "\"input.input_mode_params === 'server'\"" in src
        assert contains(src, 'file_picker_fill( "params_browse", "server_params"')
        assert mod.BOOKMARK_DEFAULTS["server_params"] == ("server_params", "")
        assert mod.BOOKMARK_DEFAULTS["server_classes"] == ("server_classes", "")


class TestProjectFolder:
    """The folder a parameter file's image paths start from, for
    relion_reconstruct."""

    def test_relion_project_by_its_pipeline(self, tmp_path):
        job = tmp_path / "proj" / "Class2D" / "job010"
        job.mkdir(parents=True)
        (tmp_path / "proj" / "default_pipeline.star").write_text("x")
        star = job / "run_it020_data.star"
        star.write_text("x")
        assert c2.project_folder(star) == str(tmp_path / "proj")
        # also given as a file:// URL in the url mode
        assert c2.project_folder(star.as_uri()) == str(tmp_path / "proj")

    def test_the_folder_the_images_exist_from_wins(self, tmp_path):
        # images extracted under another project, the parameters copied here
        star = tmp_path / "a" / "b" / "c" / "run_data.star"
        star.parent.mkdir(parents=True)
        star.write_text("x")
        image = "Extract/job005/mic1.mrcs"
        (tmp_path / "a" / "Extract" / "job005").mkdir(parents=True)
        (tmp_path / "a" / image).write_text("x")
        assert c2.project_folder(star, image) == str(tmp_path / "a")

    def test_cryosparc_project_by_its_marker(self, tmp_path):
        project = tmp_path / "CS-tau"
        job = project / "J63"
        job.mkdir(parents=True)
        (project / "project.json").write_text("{}")
        (job / "job.json").write_text("{}")
        cs = job / "J63_020_particles.cs"
        cs.write_text("x")
        assert c2.project_folder(cs) == str(project)

    def test_without_markers_the_usual_depth(self, tmp_path):
        star = tmp_path / "P" / "Class2D" / "job010" / "run_data.star"
        star.parent.mkdir(parents=True)
        star.write_text("x")
        assert c2.project_folder(star) == str(tmp_path / "P")
        cs = tmp_path / "Q" / "J9" / "J9_particles.cs"
        cs.parent.mkdir(parents=True)
        cs.write_text("x")
        assert c2.project_folder(cs) == str(tmp_path / "Q")

    def test_symlinked_files_stay_in_their_project(self, tmp_path):
        # cryoSPARC links imported data into its jobs
        elsewhere = tmp_path / "raw" / "J1_particles.cs"
        elsewhere.parent.mkdir()
        elsewhere.write_text("x")
        project = tmp_path / "CS-p"
        (project / "J1").mkdir(parents=True)
        (project / "project.json").write_text("{}")
        link = project / "J1" / "J1_particles.cs"
        link.symlink_to(elsewhere)
        assert c2.project_folder(link) == str(project)

    def test_web_urls_and_missing_files_give_nothing(self, tmp_path):
        assert c2.project_folder("https://ftp.ebi.ac.uk/x_data.star") == ""
        assert c2.project_folder(tmp_path / "missing_data.star") == ""


class TestPairProblem:
    """Files of one Class2D job are named alike; a pair that is not is named."""

    def test_the_files_of_one_relion_job_are_a_pair(self):
        assert (
            c2.pair_problem(
                "/p/Class2D/job010/run_it020_data.star",
                "/p/Class2D/job010/run_it020_classes.mrcs",
            )
            is None
        )

    def test_a_url_and_a_local_file_of_the_same_job_are_a_pair(self):
        assert (
            c2.pair_problem(
                "/p/Class2D/job010/run_it020_data.star",
                "https://ftp.example.org/x/Class2D/job010/run_it020_classes.mrcs",
            )
            is None
        )

    def test_the_job_and_the_iteration_that_differ_are_named(self):
        why = c2.pair_problem(
            "/p/Class2D/job007/run_it025_data.star",
            "https://ftp.example.org/x/Class2D/job010/run_it020_classes.mrcs",
        )
        assert "iterations differ (run_it025 and run_it020)" in why
        assert "folders differ (Class2D/job007 and Class2D/job010)" in why

    def test_only_the_iteration_can_differ(self):
        why = c2.pair_problem(
            "/p/Class2D/job010/run_it025_data.star",
            "/p/Class2D/job010/run_it020_classes.mrcs",
        )
        assert "iterations differ" in why and "folders differ" not in why

    def test_relion_and_cryosparc_files_are_not_a_pair(self):
        why = c2.pair_problem("run_it025_data.star", "J63_020_class_averages.mrc")
        assert "RELION" in why and "cryoSPARC" in why

    def test_cryosparc_jobs_that_differ_are_named(self):
        why = c2.pair_problem(
            "J63/J63_020_particles.cs", "J64/J64_020_class_averages.mrc"
        )
        assert "J63_020 and J64_020" in why and "J63 and J64" in why
        assert "J63_020_particles.cs" in why  # the example is cryoSPARC's

    def test_the_files_of_one_cryosparc_job_are_a_pair(self):
        assert (
            c2.pair_problem(
                "J63/J63_020_particles.cs", "J63/J63_020_class_averages.mrc"
            )
            is None
        )

    def test_uploads_are_judged_by_their_names(self):
        assert c2.pair_problem("run_it025_data.star", "run_it025_classes.mrcs") is None
        assert c2.pair_problem("run_it025_data.star", "run_it020_classes.mrcs")

    def test_a_file_named_otherwise_is_reported_as_not_named_alike(self):
        why = c2.pair_problem("run_it025_data.star", "my_classes.mrcs")
        assert "names differ (run_it025 and my)" in why

    def test_a_name_that_follows_no_pattern_cannot_be_judged(self):
        assert c2.pair_problem("run_it025_data.star", "classes.mrc") is None
        assert c2.pair_problem("", "run_it020_classes.mrcs") is None
        assert c2.pair_problem(None, None) is None


class TestAbundanceErrorMessage:
    def test_it_gives_the_class_counts_and_the_pair_problem(self):
        text = c2.abundance_error_message(
            "/p/Class2D/job007/run_it025_data.star",
            "/p/Class2D/job010/run_it020_classes.mrcs",
            n_param_classes=100,
            n_image_classes=80,
            highest_class=100,
        )
        assert text.startswith("Failed to get class abundance")
        assert "100 classes (numbered up to 100); the image file has 80" in text
        assert "do not look like a pair" in text
        assert text.endswith("for the same Class2D job.")

    def test_without_a_pair_problem_only_the_counts_are_added(self):
        text = c2.abundance_error_message(
            "run_it020_data.star",
            "run_it020_classes.mrcs",
            n_param_classes=3,
            n_image_classes=2,
            highest_class=3,
        )
        assert "the image file has 2" in text and "pair" not in text

    def test_without_counts_it_is_the_old_message(self):
        text = c2.abundance_error_message("a.star", "b.mrcs")
        assert text == (
            "Failed to get class abundance from the provided Class2D parameter and "
            "image files. Make sure that the two files are for the same Class2D job."
        )


class TestTabsGiveTheCounts:
    def test_the_three_tabs_use_the_message(self):
        import inspect

        from helicon.webApps.tabs import (
            abinitio3d_tab,
            helical_pitch_tab,
            where_is_my_class_tab,
        )

        for tab in (abinitio3d_tab, helical_pitch_tab, where_is_my_class_tab):
            src = inspect.getsource(tab)
            assert "class2d_files.abundance_error_message(" in src, tab.__name__
            assert (
                "Make sure that the two files are for the same Class2D job" not in src
            )
