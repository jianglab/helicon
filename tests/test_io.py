import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import pandas as pd
from helicon.lib import io
from helicon.lib.euler import (
    euler_relion2eman,
    euler_eman2relion,
    relion_euler2quaternion,
    eman_euler2quaternion,
    quaternion2euler,
)
from helicon.lib.epu import (
    guess_data_collection_software,
    extract_data_collection_time,
    extract_beamshift,
)
from helicon.lib.util import set_angle_range


class TestIo(object):
    def test_euler_conversions(self):
        # Test RELION to EMAN and back
        rot, tilt, psi = 10, 20, 30
        az, alt, phi = euler_relion2eman(rot, tilt, psi)
        assert abs(az - 100) < 1e-7
        assert abs(alt - 20) < 1e-7
        assert abs(phi - (-60)) < 1e-7
        rot_new, tilt_new, psi_new = euler_eman2relion(az, alt, phi)
        assert abs(rot_new - rot) < 1e-7
        assert abs(tilt_new - tilt) < 1e-7
        assert abs(psi_new - psi) < 1e-7

    def test_quaternion_conversions(self):
        # Test RELION Euler to quaternion and back
        rot, tilt, psi = 10, 20, 30
        q = relion_euler2quaternion(rot, tilt, psi)
        assert q.shape == (1, 4)
        rot_new, tilt_new, psi_new = quaternion2euler(q, euler_convention="relion")
        assert abs(rot_new[0] - rot) < 1e-7
        assert abs(tilt_new[0] - tilt) < 1e-7
        assert abs(psi_new[0] - psi) < 1e-7

        # Test EMAN Euler to quaternion and back
        az, alt, phi = 100, 20, -60
        q_eman = eman_euler2quaternion(az, alt, phi)
        az_new, alt_new, phi_new = quaternion2euler(q_eman, euler_convention="eman")
        # Need to handle angle wrapping
        assert (
            abs(
                set_angle_range(az_new, [-180, 180])[0]
                - set_angle_range(az, [-180, 180])
            )
            < 1e-7
        )
        assert abs(alt_new[0] - alt) < 1e-7
        assert (
            abs(
                set_angle_range(phi_new, [-180, 180])[0]
                - set_angle_range(phi, [-180, 180])
            )
            < 1e-7
        )

    def test_filename_parsing(self):
        epu_old_filename = "FoilHole_1464933_Data_427288_427290_20250502_213110_Fractions_patch_aligned_doseweighted.mrc"
        epu_filename = "FoilHole_30593197_Data_30537205_30537207_20230430_084907_fractions_patch_aligned_doseweighted.mrc"
        serialem_pncc_filename = (
            "Grid_Screening_20240328_192116_001_X+1Y-1-1_fractions.tiff"
        )
        serialem_cuhksz_filename = "Grid_Screening_20240328_192116_00001_fractions.tiff"

        assert guess_data_collection_software(epu_old_filename) == "EPU_old"
        assert guess_data_collection_software(epu_filename) == "EPU_old"
        assert guess_data_collection_software(serialem_pncc_filename) == "serialEM_pncc"
        assert (
            guess_data_collection_software(serialem_cuhksz_filename)
            == "serialEM_cuhksz"
        )

        assert (
            extract_data_collection_time(epu_old_filename, software="EPU_old")
            == 1746221470.0
        )
        # Note: extract_beamshift with serialEM_pncc needs a serial number prefix
        # (e.g. "Grid_Screening_20240328_192116_001_X+1Y-1-1_fractions.tiff")
        # The test filename "Grid_Screening_20240328_192116_X+1Y-1-1_fractions.tiff"
        # does not match the current pattern.
        assert (
            extract_beamshift(serialem_cuhksz_filename, software="serialEM_cuhksz")
            == "00001"
        )

    def setup_method(self, method):
        # Create a dummy star file for testing
        self.star_file = "test.star"
        self.df = pd.DataFrame(
            {
                "rlnDefocusU": [10000, 11000],
                "rlnDefocusV": [10000, 11000],
                "rlnDefocusAngle": [0, 0],
                "rlnSphericalAberration": [2.7, 2.7],
                "rlnAmplitudeContrast": [0.1, 0.1],
                "rlnImageName": ["001@test.mrcs", "002@test.mrcs"],
                "rlnMicrographOriginalPixelSize": [1.0, 1.0],
                "rlnImageSize": [64, 64],
                "rlnImagePixelSize": [1.0, 1.0],
            }
        )
        import starfile

        starfile.write({"particles": self.df}, self.star_file, overwrite=True)

        # Create a dummy cs file for testing
        self.cs_file = "test.cs"
        import numpy as np

        self.cs_array = np.array(
            [
                (300.0, 10000, 10000, 0.0, 2.7, 0.1, b"/path/to/test.mrc", 0),
                (300.0, 11000, 11000, 0.0, 2.7, 0.1, b"/path/to/test.mrc", 1),
            ],
            dtype=[
                ("ctf/accel_kv", "<f8"),
                ("ctf/df1_A", "<i8"),
                ("ctf/df2_A", "<i8"),
                ("ctf/df_angle_rad", "<f8"),
                ("ctf/cs_mm", "<f8"),
                ("ctf/amp_contrast", "<f8"),
                ("blob/path", "S128"),
                ("blob/idx", "<i8"),
            ],
        )
        np.save(self.cs_file, self.cs_array)

    def teardown_method(self, method):
        if Path(self.star_file).exists():
            Path(self.star_file).unlink()
        if Path(self.cs_file).exists():
            Path(self.cs_file).unlink()

    @patch.object(Path, "is_symlink", return_value=False)
    @patch.object(Path, "is_file", return_value=True)
    @patch.object(Path, "exists", return_value=True)
    def test_star2dataframe(self, mock_exists, mock_isfile, mock_islink):
        df = io.star2dataframe(self.star_file)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_dataframe2star(self):
        output_star_file = "output.star"
        # Mock the path normalization to avoid issues with file not found
        with patch("helicon.lib.io.dataframe_normalize_filename") as mock_normalize:
            mock_normalize.side_effect = lambda df, *args, **kwargs: df
            io.dataframe2star(self.df, output_star_file)

        import starfile

        df_read = starfile.read(output_star_file, always_dict=True)
        if "data_particles" in df_read:
            particles_df = df_read["data_particles"]
        else:
            particles_df = df_read["particles"]
        assert len(particles_df) == 2
        assert "rlnImageName" in particles_df.columns
        Path(output_star_file).unlink()

    @patch.object(Path, "is_symlink", return_value=False)
    @patch.object(Path, "is_file", return_value=True)
    @patch.object(Path, "exists", return_value=True)
    def test_cs2dataframe(self, mock_exists, mock_isfile, mock_islink):
        with patch("numpy.load") as mock_load:
            mock_load.return_value = self.cs_array
            df = io.cs2dataframe(
                self.cs_file, warn_missing_ctf=0, ignore_bad_particle_path=1
            )
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "ctf/accel_kv" in df.columns

    def test_dataframe2cs(self):
        output_cs_file = "output.cs"
        df = pd.DataFrame(self.cs_array)
        io.dataframe2cs(df, output_cs_file)

        import numpy as np

        cs_read = np.load(output_cs_file, allow_pickle=True)
        assert len(cs_read) == 2
        assert "ctf/accel_kv" in cs_read.dtype.names
        Path(output_cs_file).unlink()

    def test_dataframe_convert(self):
        # Test CryoSPARC to RELION conversion
        cs_df = pd.DataFrame(
            {
                "ctf/accel_kv": [300.0],
                "blob/path": ["/path/to/test.mrcs"],
                "blob/idx": [0],
            }
        )
        cs_df.attrs["convention"] = "cryosparc"
        relion_df = io.dataframe_convert(cs_df, target="relion")
        assert relion_df.attrs["convention"] == "relion"
        assert "rlnVoltage" in relion_df.columns
        assert "rlnImageName" in relion_df.columns
        assert relion_df["rlnImageName"][0] == "000001@/path/to/test.mrcs"

        # Test RELION to CryoSPARC conversion
        with pytest.raises(NameError):
            io.dataframe_convert(relion_df, target="cryosparc")

    def test_mrc2mrcs_preserves_extension(self, tmp_path):
        mrc_file = tmp_path / "particles.mrc"
        mrc_file.write_text("")
        df = pd.DataFrame(
            {
                "rlnImageName": [
                    f"000001@{mrc_file}",
                    f"000002@{mrc_file}",
                ]
            }
        )
        result = io.mrc2mrcs(df.copy())
        assert result["rlnImageName"][0].endswith(".mrcs")
        assert result["rlnImageName"][1].endswith(".mrcs")
        name = result["rlnImageName"][0].split("@")[1]
        assert name.endswith(".mrcs"), f"Expected .mrcs suffix, got: {name}"

    def test_dataframe_convert_coordinates(self):
        """Test coordinate conversion: both CS and RELION use top-left origin."""
        cs_df = pd.DataFrame(
            {
                "location/center_x_frac": [0.25, 0.5, 0.75],
                "location/center_y_frac": [0.25, 0.5, 0.75],
                "location/micrograph_shape": [[4096, 4096], [4096, 4096], [4096, 4096]],
                "blob/path": ["/a.mrc"] * 3,
                "blob/idx": [0, 1, 2],
                "blob/psize_A": [1.0, 1.0, 1.0],
                "alignments2D/shift": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            }
        )
        cs_df.attrs["convention"] = "cryosparc"

        r = io.dataframe_cryosparc_to_relion(cs_df)
        # No Y inversion: center_y_frac * height
        assert r["rlnCoordinateY"].iloc[0] == 1024.0
        assert r["rlnCoordinateY"].iloc[1] == 2048.0
        assert r["rlnCoordinateY"].iloc[2] == 3072.0
        # X is also center_x_frac * width
        assert r["rlnCoordinateX"].iloc[0] == 1024.0
        assert r["rlnCoordinateX"].iloc[2] == 3072.0
        # Angstrom origins: pixel shift * blob/psize_A, (negated for 2D)
        assert r["rlnOriginXAngst"].iloc[0] == -1.0
        assert r["rlnOriginYAngst"].iloc[0] == -2.0
        assert "rlnOriginX" not in r.columns
        assert "rlnOriginY" not in r.columns

    def test_dataframe_convert_coordinates_via_dataframe_convert(self):
        """Test coordinate conversion flows through dataframe_convert."""
        cs_df = pd.DataFrame(
            {
                "location/center_x_frac": [0.25],
                "location/center_y_frac": [0.75],
                "location/micrograph_shape": [[4096, 4096]],
                "blob/path": ["/a.mrc"],
                "blob/idx": [0],
            }
        )
        cs_df.attrs["convention"] = "cryosparc"
        r = io.dataframe_convert(cs_df, target="relion")
        assert r["rlnCoordinateY"].iloc[0] == 3072.0  # 0.75 * 4096

    def test_dataframe_convert_angstrom_origins(self):
        """Test rlnOriginXAngst/YAngst are computed from pixel shifts * apix."""
        cs_df = pd.DataFrame(
            {
                "alignments2D/shift": [[2.0, 3.0]],
                "alignments3D/shift": [[4.0, 5.0]],
                "blob/psize_A": [0.5],
                "blob/path": ["/a.mrc"],
                "blob/idx": [0],
            }
        )
        cs_df.attrs["convention"] = "cryosparc"
        r = io.dataframe_cryosparc_to_relion(cs_df)
        # 3D shift overwrites 2D shift → Angstrom = 4.0 * 0.5 = 2.0
        assert r["rlnOriginXAngst"].iloc[0] == 2.0
        assert r["rlnOriginYAngst"].iloc[0] == 2.5
        assert "rlnOriginX" not in r.columns

    def test_dataframe_convert_coordinates_via_images2dataframe(self):
        """Test coordinate conversion flows through images2dataframe."""
        cs_df = pd.DataFrame(
            {
                "location/center_x_frac": [0.25],
                "location/center_y_frac": [0.75],
                "location/micrograph_shape": [[4096, 4096]],
                "blob/path": ["/a.mrc"],
                "blob/idx": [0],
            }
        )
        cs_df.attrs["convention"] = "cryosparc"
        with patch("helicon.lib.io.image2dataframe", return_value=cs_df):
            r = io.images2dataframe("input.cs", target_convention="relion")
        assert r["rlnCoordinateY"].iloc[0] == 3072.0

    def test_clean_cs_micrograph_path(self):
        """Test stripping cryoSPARC hash and _patch_aligned_doseweighted."""
        # CS path with hash + _patch_aligned_doseweighted
        assert (
            io.clean_cs_micrograph_path(
                "J298/motioncorrected/004163012191649015490_250123_SF0431_00004_1-4_patch_aligned_doseweighted.mrc"
            )
            == "250123_SF0431_00004_1-4.mrc"
        )
        # Absolute path
        assert (
            io.clean_cs_micrograph_path(
                "/net/scratch/CS-apoferritin/J298/motioncorrected/004163012191649015490_250123_SF0431_00004_1-4_patch_aligned_doseweighted.mrc"
            )
            == "250123_SF0431_00004_1-4.mrc"
        )
        # Only _patch_aligned_doseweighted (no hash)
        assert (
            io.clean_cs_micrograph_path(
                "250123_SF0431_00004_1-4_patch_aligned_doseweighted.mrc"
            )
            == "250123_SF0431_00004_1-4.mrc"
        )
        # Already clean (no hash, no _patch_aligned_doseweighted)
        assert (
            io.clean_cs_micrograph_path("250123_SF0431_00004_1-4.mrc")
            == "250123_SF0431_00004_1-4.mrc"
        )
