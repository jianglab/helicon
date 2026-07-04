import logging
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from helicon.webApps.denovo3D import pipeline


class TestGetImagesFromFile(object):
    def test_reads_mrc_file(self):
        mock_data = np.ones((16, 16), dtype=np.float32)
        mock_mrc = MagicMock()
        mock_mrc.voxel_size.x = 1.5
        mock_mrc.data = mock_data
        mock_mrc.__enter__.return_value = mock_mrc

        with patch("mrcfile.open", return_value=mock_mrc):
            data, apix = pipeline.get_images_from_file("test.mrc")

        np.testing.assert_array_equal(data, mock_data)
        assert apix == 1.5

    def test_rounds_apix(self):
        mock_mrc = MagicMock()
        mock_mrc.voxel_size.x = 1.23456
        mock_mrc.data = np.zeros((4, 4), dtype=np.float32)
        mock_mrc.__enter__.return_value = mock_mrc

        with patch("mrcfile.open", return_value=mock_mrc):
            data, apix = pipeline.get_images_from_file("test.mrc")

        assert apix == 1.2346


class TestProcessOneTask(object):
    def setup_method(self, method):
        np.random.seed(42)
        self.data = np.random.rand(16, 16).astype(np.float32)

        self.base_params = dict(
            ti=0,
            ntasks=1,
            data=None,
            imageFile="test.mrc",
            imageIndex=1,
            twist=30,
            rise=10,
            rise_range=(5, 15),
            csym=1,
            tilt=0,
            tilt_range=(-5, 5),
            psi=0,
            psi_range=0,
            dy=0,
            dy_range=0,
            apix2d_orig=1.0,
            denoise="",
            low_pass=0,
            transpose=0,
            horizontalize=0,
            target_apix3d=2.0,
            target_apix2d=1.0,
            thresh_fraction=-1,
            positive_constraint=-1,
            tube_length=-1,
            tube_diameter=40,
            tube_diameter_inner=0,
            reconstruct_length=20,
            sym_oversample=1,
            interpolation="nn",
            fsc_test=0,
            return_3d=False,
            score_metric="cosine",
            algorithm=dict(model="lsq"),
            verbose=0,
        )

    def test_returns_tuple(self):
        params = dict(self.base_params, data=self.data)
        result = pipeline.process_one_task(**params)
        assert result is not None
        score, return_data, param_tuple = result
        assert isinstance(score, (float, np.floating))
        assert isinstance(return_data, tuple)
        assert len(return_data) >= 5

    def test_blank_image_returns_none(self):
        blank_data = np.zeros((16, 16), dtype=np.float32)
        params = dict(self.base_params, data=blank_data)
        result = pipeline.process_one_task(**params)
        assert result is None

    def test_return_data_contains_projections(self):
        params = dict(self.base_params, data=self.data)
        result = pipeline.process_one_task(**params)
        score, return_data, param_tuple = result
        proj_x, proj_y, proj_z, vol_data, *_ = return_data
        assert isinstance(proj_x, np.ndarray)
        assert isinstance(proj_y, np.ndarray)
        assert isinstance(proj_z, np.ndarray)
        assert len(proj_x.shape) == 2

    def test_param_tuple_contains_parameters(self):
        params = dict(self.base_params, data=self.data)
        result = pipeline.process_one_task(**params)
        score, return_data, param_tuple = result
        assert param_tuple[1] == "test.mrc"
        assert param_tuple[2] == 1

    def test_with_return_3d(self):
        params = dict(self.base_params, data=self.data, return_3d=True)
        result = pipeline.process_one_task(**params)
        assert result is not None
        score, return_data, param_tuple = result
        vol_data = return_data[3]
        assert vol_data is not None
        rec3d, rec3d_h1, rec3d_h2 = vol_data
        assert isinstance(rec3d, np.ndarray)
        assert len(rec3d.shape) == 3

    def test_with_fsc_test(self):
        params = dict(self.base_params, data=self.data, fsc_test=1)
        result = pipeline.process_one_task(**params)
        assert result is not None

    def test_with_csym(self):
        params = dict(self.base_params, data=self.data, csym=2)
        result = pipeline.process_one_task(**params)
        assert result is not None

    def test_with_horizontalize(self):
        params = dict(self.base_params, data=self.data, horizontalize=1)
        result = pipeline.process_one_task(**params)
        assert result is not None

    def test_with_thresh_fraction(self):
        params = dict(self.base_params, data=self.data, thresh_fraction=0.5)
        result = pipeline.process_one_task(**params)
        assert result is not None

    def test_with_tilt_and_psi(self):
        params = dict(
            self.base_params,
            data=self.data,
            tilt=5,
            psi=10,
            dy=2,
        )
        result = pipeline.process_one_task(**params)
        assert result is not None

    @patch("helicon.read_image_2d")
    def test_loads_data_when_none(self, mock_read):
        mock_read.return_value = np.random.rand(16, 16).astype(np.float32)
        unique_image = f"test_loads_{id(self)}.mrc"
        params = dict(self.base_params, data=None, imageFile=unique_image)
        result = pipeline.process_one_task(**params)
        if result is not None:
            mock_read.assert_called_once()
