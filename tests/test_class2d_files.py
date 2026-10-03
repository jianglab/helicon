"""Finding the other file of a 2D classification beside the one given."""

import pytest

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
        assert (
            'file_picker_fill(\n            "params_browse",\n            "server_params"'
            in src
        )
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
