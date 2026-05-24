import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from helicon.webApps.whereIsMyClass import compute


class TestGetProjectRootDir(object):
    def test_star_file(self):
        result = compute.get_project_root_dir("/a/b/c/JobName/run_it020_data.star")
        assert result is not None
        assert result.name == "b"

    def test_cs_file(self):
        result = compute.get_project_root_dir("/a/b/c/J123/J456/run_it020.cs")
        assert result is not None
        assert result.name == "J123"

    def test_unknown_extension(self):
        result = compute.get_project_root_dir("/a/b/c/file.txt")
        assert result is None


class TestGetClassFile(object):
    def test_star_file_class2d(self):
        result = compute.get_class_file("/a/b/c/JobName/run_it020_data.star")
        assert result is not None
        assert result.suffix == ".mrcs"
        assert "classes" in result.name

    def test_star_file_class3d(self):
        files = compute.get_class_file("/a/b/c/Class3D/run_it020_data.star")
        if files is not None:
            assert isinstance(files, list)

    def test_cs_file(self):
        result = compute.get_class_file("/a/b/c/J123/run_it020.cs")
        assert result is not None
        assert result.suffix == ".mrc"
        assert "class_averages" in result.name

    def test_unknown_extension(self):
        result = compute.get_class_file("/a/b/c/file.txt")
        assert result is None


class TestGetFilamentLength(object):
    def test_basic_length(self):
        helices = [
            (
                ("micrograph1", 1),
                pd.DataFrame(
                    {
                        "rlnHelicalTrackLengthAngst": [100.0, 200.0, 300.0],
                    }
                ),
            ),
        ]
        lengths = compute.get_filament_length(helices)
        assert lengths == [200.0]

    def test_multiple_helices(self):
        helices = [
            (
                ("micrograph1", 1),
                pd.DataFrame({"rlnHelicalTrackLengthAngst": [50.0, 150.0]}),
            ),
            (
                ("micrograph1", 2),
                pd.DataFrame({"rlnHelicalTrackLengthAngst": [10.0, 20.0, 30.0]}),
            ),
        ]
        lengths = compute.get_filament_length(helices)
        assert lengths == [100.0, 20.0]

    def test_with_particle_box_length(self):
        helices = [
            (
                ("micrograph1", 1),
                pd.DataFrame({"rlnHelicalTrackLengthAngst": [100.0, 200.0]}),
            ),
        ]
        lengths = compute.get_filament_length(helices, particle_box_length=50)
        assert lengths == [150.0]


class TestSelectClasses(object):
    def setup_method(self, method):
        self.params = pd.DataFrame(
            {
                "rlnClassNumber": [1, 1, 2, 2, 3],
                "rlnMicrographName": ["m1", "m1", "m1", "m2", "m2"],
                "rlnHelicalTubeID": [1, 2, 1, 1, 1],
            }
        )

    def test_selects_single_class(self):
        helices = compute.select_classes(self.params, class_indices=[0])
        assert len(helices) > 0
        for _, h in helices:
            assert all(h["rlnClassNumber"] == 1)

    def test_selects_multiple_classes(self):
        helices = compute.select_classes(self.params, class_indices=[0, 1])
        all_classes = set()
        for _, h in helices:
            all_classes.update(h["rlnClassNumber"].unique())
        assert all_classes == {1, 2}

    def test_empty_when_no_match(self):
        helices = compute.select_classes(self.params, class_indices=[99])
        assert len(helices) == 0


class TestSelectHelicesFromHelixID(object):
    def setup_method(self, method):
        self.params = pd.DataFrame(
            {
                "helixID": [1, 1, 2, 2, 3],
                "rlnMicrographName": ["m1", "m1", "m1", "m1", "m1"],
                "rlnHelicalTubeID": [1, 1, 1, 2, 1],
            }
        )

    def test_selects_single_id(self):
        helices = compute.select_helices_from_helixID(self.params, ids=[1])
        assert len(helices) > 0
        for _, h in helices:
            assert all(h["helixID"] == 1)

    def test_selects_multiple_ids(self):
        helices = compute.select_helices_from_helixID(self.params, ids=[1, 2])
        ids_found = set()
        for _, h in helices:
            ids_found.update(h["helixID"].unique())
        assert ids_found == {1, 2}


class TestComputePairDistances(object):
    def setup_method(self, method):
        self.helices = [
            (
                ("m1", 1),
                pd.DataFrame(
                    {
                        "rlnClassNumber": [1, 1, 1],
                        "rlnHelicalTrackLengthAngst": [0.0, 100.0, 200.0],
                        "rlnAnglePsi": [0.0, 0.0, 0.0],
                    }
                ),
            ),
        ]

    def test_returns_sorted_distances(self):
        dists, min_len = compute.compute_pair_distances(self.helices)
        assert len(dists) > 0
        assert np.all(dists[:-1] <= dists[1:])
        assert min_len == 0

    def test_with_same_polarity_same_class(self):
        helices = [
            (
                ("m1", 1),
                pd.DataFrame(
                    {
                        "rlnClassNumber": [1, 1, 1],
                        "rlnHelicalTrackLengthAngst": [0.0, 100.0, 200.0],
                        "rlnAnglePsi": [10.0, 10.0, 190.0],
                    }
                ),
            ),
        ]
        dists, min_len = compute.compute_pair_distances(helices)
        # Segments with psi=10 and psi=190 differ by 180, so |(190-10+180)%360-180|=0 < 90
        # Only pairs where both segments have |psi_diff| < 90 are included
        assert len(dists) > 0

    def test_with_lengths_filter(self):
        dists, min_len = compute.compute_pair_distances(
            self.helices, lengths=[100], target_total_count=1
        )
        assert len(dists) > 0

    def test_empty_input_returns_empty(self):
        dists, min_len = compute.compute_pair_distances([])
        assert dists == []
        assert min_len == 0


class TestEstimateInterSegmentDistance(object):
    def test_returns_median_distance(self):
        data = pd.DataFrame(
            {
                "rlnMicrographName": ["m1", "m1", "m1"],
                "rlnHelicalTubeID": [1, 1, 1],
                "rlnHelicalTrackLengthAngst": [0.0, 100.0, 300.0],
            }
        )
        result = compute.estimate_inter_segment_distance(data)
        # distances: [100, 200], median = 150
        assert result == 150.0

    def test_single_segment_raises_value_error(self):
        data = pd.DataFrame(
            {
                "rlnMicrographName": ["m1"],
                "rlnHelicalTubeID": [1],
                "rlnHelicalTrackLengthAngst": [100.0],
            }
        )
        with pytest.raises(ValueError):
            compute.estimate_inter_segment_distance(data)

    def test_empty_raises_value_error(self):
        data = pd.DataFrame(
            columns=[
                "rlnMicrographName",
                "rlnHelicalTubeID",
                "rlnHelicalTrackLengthAngst",
            ]
        )
        with pytest.raises(ValueError):
            compute.estimate_inter_segment_distance(data)


class TestGetClassAbundance(object):
    def test_counts_correctly(self):
        params = pd.DataFrame(
            {
                "rlnClassNumber": [1, 1, 2, 2, 2, 3],
            }
        )
        abundance = compute.get_class_abundance(params, nClass=3)
        np.testing.assert_array_equal(abundance, [2, 3, 1])

    def test_more_classes_than_data(self):
        params = pd.DataFrame(
            {
                "rlnClassNumber": [1, 2],
            }
        )
        abundance = compute.get_class_abundance(params, nClass=5)
        np.testing.assert_array_equal(abundance, [1, 1, 0, 0, 0])

    def test_empty_input(self):
        params = pd.DataFrame({"rlnClassNumber": []})
        abundance = compute.get_class_abundance(params, nClass=3)
        np.testing.assert_array_equal(abundance, [0, 0, 0])


class TestGetOneMapXyzProjects(object):
    def test_returns_2d_image(self):
        data = np.ones((8, 8, 8), dtype=np.float32)
        result = compute.get_one_map_xyz_projects(data, nx=8)
        assert result.shape == (8, 26)
        assert np.all(np.isfinite(result))

    def test_normalization_by_min_max(self):
        data = np.zeros((4, 4, 4), dtype=np.float32)
        data[1:3, 1:3, 1:3] = 10.0
        result = compute.get_one_map_xyz_projects(data, nx=4)
        assert result.shape == (4, 14)
        assert np.all(result >= 0) and np.all(result <= 4)

    def test_constant_data_does_not_divide_by_zero(self):
        data = np.ones((4, 4, 4), dtype=np.float32)
        result = compute.get_one_map_xyz_projects(data, nx=4)
        assert result.shape == (4, 14)


class TestGetClass2dFromFile(object):
    def test_reads_mrc_data(self):
        mock_data = np.random.rand(4, 8, 8).astype(np.float32)
        mock_mrc = MagicMock()
        mock_mrc.voxel_size.x = 2.5
        mock_mrc.data = mock_data
        mock_mrc.__enter__.return_value = mock_mrc

        with patch("mrcfile.open", return_value=mock_mrc):
            data, apix = compute.get_class2d_from_file("classes.mrcs")

        np.testing.assert_array_equal(data, mock_data)
        assert apix == 2.5


class TestStarToDataframe(object):
    def test_returns_dataframe_with_optics(self):
        mock_data = pd.DataFrame({"rlnClassNumber": [1, 2]})
        mock_data.attrs["optics"] = pd.DataFrame()
        mock_data.attrs["starFile"] = "test.star"

        with patch(
            "starfile.read",
            return_value={"optics": pd.DataFrame(), "particles": mock_data},
        ):
            result = compute.star_to_dataframe("test.star")

        assert isinstance(result, pd.DataFrame)
        assert "optics" in result.attrs
        assert "starFile" in result.attrs

    def test_raises_on_missing_optics_or_particles(self):
        with patch("starfile.read", return_value={"only_table": pd.DataFrame()}):
            with pytest.raises(AssertionError):
                compute.star_to_dataframe("test.star")


class TestCsToDataframe(object):
    def setup_method(self, method):
        n = 5
        self.cs_data = np.zeros(
            n,
            dtype=[
                ("blob/idx", "<i8"),
                ("blob/path", "S128"),
                ("filament/filament_uid", "<i8"),
                ("filament/arc_length_A", "<f8"),
                ("alignments2D/class", "<i8"),
                ("alignments2D/pose", "<f8", (3,)),
                ("location/center_x_frac", "<f8"),
                ("location/center_y_frac", "<f8"),
                ("location/micrograph_shape", "<f8", (2,)),
                ("micrograph_blob/path", "S128"),
            ],
        )
        for i in range(n):
            self.cs_data[i] = (
                i,
                b"/path/to/micrograph.mrc",
                i,
                float(i * 100),
                0,
                (0.0, 0.0, 0.0),
                0.5,
                0.5,
                (100.0, 100.0),
                b"/path/to/micrograph.mrc",
            )

    def test_converts_to_dataframe(self):
        with patch("numpy.load", return_value=self.cs_data):
            result = compute.cs_to_dataframe("test.cs")

        assert isinstance(result, pd.DataFrame)
        assert "rlnImageName" in result.columns
        assert "rlnClassNumber" in result.columns
        assert "rlnHelicalTubeID" in result.columns
        assert "rlnCoordinateX" in result.columns

    def test_rln_class_number_is_1_indexed(self):
        with patch("numpy.load", return_value=self.cs_data):
            result = compute.cs_to_dataframe("test.cs")
        assert all(result["rlnClassNumber"] == 1)

    def test_missing_required_attrs_raises_error(self):
        bad_data = np.zeros(1, dtype=[("blob/idx", "<i8")])
        with patch("numpy.load", return_value=bad_data):
            with pytest.raises(ValueError):
                compute.cs_to_dataframe("test.cs")


class TestGetClass2dParamsFromFile(object):
    def setup_method(self, method):
        self.star_data = pd.DataFrame(
            {
                "rlnImageName": ["001@test.mrcs"],
                "rlnHelicalTubeID": [1],
                "rlnHelicalTrackLengthAngst": [100.0],
                "rlnClassNumber": [1],
                "rlnCoordinateX": [50.0],
                "rlnCoordinateY": [100.0],
            }
        )

    def test_accepts_star_file(self):
        with patch(
            "helicon.webApps.whereIsMyClass.compute.star_to_dataframe",
            return_value=self.star_data,
        ):
            result = compute.get_class2d_params_from_file("test.star")
        assert isinstance(result, pd.DataFrame)

    def test_accepts_cs_file(self):
        with patch(
            "helicon.webApps.whereIsMyClass.compute.cs_to_dataframe",
            return_value=self.star_data,
        ):
            result = compute.get_class2d_params_from_file("test.cs")
        assert isinstance(result, pd.DataFrame)

    def test_rejects_unknown_extension(self):
        with pytest.raises(ValueError):
            compute.get_class2d_params_from_file("test.txt")

    def test_missing_required_attrs_raises(self):
        bad_data = pd.DataFrame({"rlnClassNumber": [1]})
        with patch(
            "helicon.webApps.whereIsMyClass.compute.star_to_dataframe",
            return_value=bad_data,
        ):
            with pytest.raises(ValueError):
                compute.get_class2d_params_from_file("test.star")


class TestPlotMicrograph(object):
    def test_returns_figure_widget(self):
        data = np.random.rand(64, 64).astype(np.float32)
        fig = compute.plot_micrograph(data, title="Test", apix=1.0)
        assert fig is not None
        assert len(fig.data) > 0
        assert "image" in fig.data[0].name

    def test_with_plot_dimensions(self):
        data = np.random.rand(64, 64).astype(np.float32)
        fig = compute.plot_micrograph(data, title="Test", apix=1.0, plot_height=400)
        assert fig.layout.height == 400


class TestMarkClassesOnHelices(object):
    def test_adds_traces_to_figure(self):
        import plotly.graph_objects as go

        fig = go.FigureWidget()
        fig.add_trace(go.Heatmap(z=[[0, 1], [1, 0]], name="image"))

        helices = {(1, 2): {"x": [10.0, 20.0], "y": [30.0, 40.0]}}
        compute.mark_classes_on_helices(fig, helices, marker_size=10)

        traces = [d for d in fig.data if d.name.startswith("class_")]
        assert len(traces) == 1
        assert traces[0].name == "class_2"

    def test_empty_helices_removes_class_traces(self):
        import plotly.graph_objects as go

        fig = go.FigureWidget()
        fig.add_trace(go.Heatmap(z=[[0, 1], [1, 0]], name="image"))

        compute.mark_classes_on_helices(fig, {}, marker_size=10)
        traces = [d for d in fig.data if d.name.startswith("class_")]
        assert len(traces) == 0


class TestDrawDistanceMeasurement(object):
    def test_adds_line_when_both_points_present(self):
        import plotly.graph_objects as go

        fig = go.FigureWidget()
        fig.add_trace(go.Heatmap(z=[[0, 1], [1, 0]], name="image"))

        compute.draw_distance_measurement(
            fig, first_point=(0, 0), second_point=(10, 10)
        )
        lines = [d for d in fig.data if d.name == "distance_line"]
        assert len(lines) == 1

    def test_removes_line_when_no_points(self):
        import plotly.graph_objects as go

        fig = go.FigureWidget()
        fig.add_trace(go.Heatmap(z=[[0, 1], [1, 0]], name="image"))
        compute.draw_distance_measurement(
            fig, first_point=(0, 0), second_point=(10, 10)
        )

        compute.draw_distance_measurement(fig, first_point=None, second_point=None)
        lines = [d for d in fig.data if d.name == "distance_line"]
        assert len(lines) == 0


class TestPlotHistogram(object):
    def test_returns_figure_with_histogram(self):
        data = np.random.rand(100) * 100
        fig = compute.plot_histogram(
            data=data,
            title="Test",
            xlabel="Distance (Å)",
            ylabel="Count",
            bins=20,
        )
        assert fig is not None
        assert len(fig.data) > 0

    def test_filters_by_max_pair_dist(self):
        data = np.array([10, 20, 100, 200])
        fig = compute.plot_histogram(
            data=data,
            title="Test",
            xlabel="Distance (Å)",
            ylabel="Count",
            max_pair_dist=50,
            bins=10,
        )
        assert fig is not None

    def test_updates_existing_figure(self):
        import plotly.graph_objects as go

        existing_fig = go.FigureWidget()
        existing_fig.add_trace(go.Bar(x=[1, 2, 3], y=[1, 2, 3]))

        data = np.random.rand(50) * 100
        fig = compute.plot_histogram(
            data=data,
            title="Updated",
            xlabel="Distance (Å)",
            ylabel="Count",
            bins=20,
            fig=existing_fig,
        )
        assert fig is existing_fig
        assert fig.layout.title.text == "Updated"
