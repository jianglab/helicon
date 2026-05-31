from __future__ import annotations

from typing import Any
import numpy as np

__all__ = [
    "load_h5_file",
    "reconstruct_ptychography",
]


def load_h5_file(filepath: str) -> tuple[Any, Any, float, float, float]:
    """Load a ptychography dataset from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file.

    Returns
    -------
    tuple of (ndarray, ndarray, float, float, float)
        - data: The ptychography data.
        - vacuumProbe: The vacuum probe.
        - AccVoltage: Acceleration voltage in volts.
        - SemiConvAngle: Convergence semi-angle in milliradians.
        - R_pixel: Probe scan step size in Angstroms.

    Raises
    ------
    AssertionError
        If the file extension is not .h5.
    """
    # based on empiar-12236 apoferritin ptychography dataset
    from pathlib import Path
    import h5py

    extension = Path(filepath).suffix.lower()
    assert extension == ".h5", f"ERROR: only hdf5 (.h5) files are supported"

    file = h5py.File(filepath, "r")
    data = file["data"]
    vacuumProbe = file["vacuumProbe"]

    AccVoltage = (
        float(file["data"].attrs["Acceleration voltage [kV]"]) * 1e3
    )  # 300,000 [V]
    SemiConvAngle = float(file["data"].attrs["Convergence semi-angle [mrad]"])  # [mrad]
    R_pixel = float(file["data"].attrs["STEM step-size [A]"])  # 20.00 # A

    return data, vacuumProbe, AccVoltage, SemiConvAngle, R_pixel


def reconstruct_ptychography(
    filepath: str,
    defocus_initial_guess: float = -15000,
    iteration_times: int = 30,
    com_rotation_force: float = 89.8,
    com_transpose_force: bool = False,
    dataset_scan_size: tuple[int, int] = (128, 128),
    batch_size: int = 256,
    step_size: float = 0.5,
    crop_margin: int = 16,
    num_iter: int = 5,
) -> np.ndarray:
    """Reconstruct a ptychography image from an h5 file.

    Parameters
    ----------
    filepath : str
        Path to the h5 file containing the data.
    defocus_initial_guess : float, optional
        Initial defocus guess in Angstroms. Defaults to -15000.
    iteration_times : int, optional
        Number of iterations. Defaults to 30.
    com_rotation_force : float, optional
        Center of mass rotation force in degrees. Defaults to 89.8.
    com_transpose_force : bool, optional
        Whether to transpose COM. Defaults to False.
    dataset_scan_size : tuple of int, optional
        Scan size for the dataset ``(rows, cols)``. Defaults to (128, 128).
    batch_size : int, optional
        Batch size for reconstruction. Defaults to 256.
    step_size : float, optional
        Step size for reconstruction. Defaults to 0.5.
    crop_margin : int, optional
        Margin to crop from the reconstructed object. Defaults to 16.
    num_iter : int, optional
        Number of iterations for reconstruction. Defaults to 5.

    Returns
    -------
    numpy.ndarray
        The reconstructed image (cropped object).
    """

    import py4DSTEM
    from py4DSTEM.process.phase import Parallax, SingleslicePtychography
    from py4DSTEM.process.calibration import get_probe_size
    from py4DSTEM import DataCube
    from pathlib import Path
    import numpy as np

    assert py4DSTEM is not None, "py4DSTEM must be installed to run this function"
    assert (
        isinstance(filepath, str) and len(filepath) > 0
    ), "filepath must be a non-empty string"
    assert Path(filepath).exists(), f"File {filepath} does not exist"
    assert Path(filepath).suffix.lower() == ".h5", "Only .h5 files are supported"
    assert isinstance(
        defocus_initial_guess, (int, float)
    ), "defocus_initial_guess must be a number"
    assert (
        isinstance(iteration_times, int) and iteration_times > 0
    ), "iteration_times must be a positive integer"
    assert isinstance(
        com_rotation_force, (int, float)
    ), "com_rotation_force must be a number"
    assert isinstance(
        com_transpose_force, bool
    ), "com_transpose_force must be a boolean"
    assert (
        isinstance(dataset_scan_size, tuple) and len(dataset_scan_size) == 2
    ), "dataset_scan_size must be a tuple of two integers"
    assert (
        isinstance(batch_size, int) and batch_size > 0
    ), "batch_size must be a positive integer"
    assert (
        isinstance(step_size, (int, float)) and step_size > 0
    ), "step_size must be a positive number"
    assert (
        isinstance(num_iter, int) and num_iter > 0
    ), "num_iter must be a positive integer"
    assert (
        isinstance(dataset_scan_size, tuple) and len(dataset_scan_size) == 2
    ), "dataset_scan_size must be a tuple of two integers"
    assert (
        dataset_scan_size[0] > 0 and dataset_scan_size[1] > 0
    ), "dataset_scan_size must be positive integers"
    assert (
        batch_size <= dataset_scan_size[0] * dataset_scan_size[1]
    ), "batch_size must be less than or equal to the total number of pixels in the dataset"
    assert step_size > 0, "step_size must be a positive number"
    assert num_iter > 0, "num_iter must be a positive integer"
    assert (
        defocus_initial_guess < 0
    ), "defocus_initial_guess should be negative for defocus estimation"
    assert com_rotation_force >= 0, "com_rotation_force should be non-negative"
    assert com_transpose_force in [
        True,
        False,
    ], "com_transpose_force should be a boolean value"

    # Load data
    data, vacuumProbe, AccVoltage, SemiConvAngle, R_pixel = load_h5_file(filepath)
    dataset = py4DSTEM.DataCube(data=data)

    # Get mean diffraction pattern and calibrations
    dataset.get_dp_mean()
    probe_semiangle, probe_qx0, probe_qy0 = py4DSTEM.process.calibration.get_probe_size(
        dataset.tree("dp_mean").data
    )

    dataset.calibration.set_R_pixel_size(R_pixel)
    dataset.calibration.set_R_pixel_units("A")
    dataset.calibration.set_Q_pixel_size(SemiConvAngle / probe_semiangle)
    dataset.calibration.set_Q_pixel_units("mrad")

    # Create cropped dataset for parallax
    dataset_cropped = py4DSTEM.DataCube(data=data[0:64, :]).bin_Q(2)
    dataset_cropped.get_dp_mean()
    probe_semiangle_cropped, probe_qx0_cropped, probe_qy0_cropped = (
        py4DSTEM.process.calibration.get_probe_size(
            dataset_cropped.tree("dp_mean").data
        )
    )

    dataset_cropped.calibration.set_R_pixel_size(R_pixel)
    dataset_cropped.calibration.set_R_pixel_units("A")
    dataset_cropped.calibration.set_Q_pixel_size(
        SemiConvAngle / probe_semiangle_cropped
    )
    dataset_cropped.calibration.set_Q_pixel_units("mrad")

    # Parallax for defocus estimation
    parallax = py4DSTEM.process.phase.Parallax(
        energy=AccVoltage,
        datacube=dataset_cropped,
        verbose=False,
        device="cpu",
    ).preprocess(
        plot_average_bf=False,
        defocus_guess=defocus_initial_guess,
        rotation_guess=com_rotation_force,
    )
    parallax.reconstruct(
        min_alignment_bin=16,
        num_iter_at_min_bin=16,
    )
    parallax.aberration_fit()
    parallax.aberration_correct()

    # Single-slice ptychography
    ptycho = py4DSTEM.process.phase.SingleslicePtychography(
        verbose=False,
        datacube=dataset,
        device="cpu",
        energy=AccVoltage,
        vacuum_probe_intensity=vacuumProbe,
        defocus=parallax.aberration_C1,
        object_padding_px=(16, 16),
        object_type="potential",
    ).preprocess(
        plot_center_of_mass=False,
        plot_rotation=False,
        plot_probe_overlaps=False,
        force_com_rotation=parallax.rotation_Q_to_R_rads * 180 / np.pi + 180,
        force_com_transpose=com_transpose_force,
    )

    # Reconstruction
    ptycho = ptycho.reconstruct(
        reset=True,
        store_iterations=True,
        step_size=step_size,
        num_iter=num_iter,
        q_lowpass=None,
        fix_positions=True,
        global_affine_transformation=False,
        fix_probe_aperture=True,
        fit_probe_aberrations=True,
        fit_probe_aberrations_max_angular_order=4,
        fit_probe_aberrations_max_radial_order=4,
        max_batch_size=batch_size,
        object_positivity=False,
    )

    # Return the cropped reconstructed object
    return ptycho.object_cropped[crop_margin:-crop_margin, crop_margin:-crop_margin]
