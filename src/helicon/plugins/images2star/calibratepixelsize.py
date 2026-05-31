"""Handler for the calibratePixelSize option."""

from __future__ import annotations
import helicon
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mrcfile
from helicon.lib.io import getPixelSize, setPixelSize
import logging

logger = logging.getLogger(__name__)


option_name = "calibratePixelSize"


def add_args(parser):
    choices = "graphene graphene_oxide go gold ice".split()
    parser.add_argument(
        "--calibratePixelSize",
        type=str,
        metavar="<%s>" % ("|".join(choices)),
        choices=choices,
        help="calibrate pixel size for common samples. default no",
        default="no",
    )


def handle(data, args, index_d, param):
    """Handle the calibratePixelSize option.

    Parameters
    ----------
    data : pd.DataFrame
        The particle data DataFrame.
    args : argparse.Namespace
        CLI arguments.
    index_d : dict
        Option index tracker.
    param : object
        The parameter value for this option.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (data, index_d) after processing.
    """
    if param:
        standard_sample = param
        # ice ring at 3.661Å: https://journals.iucr.org/d/issues/2021/04/00/tz5104/index.html
        supported_standards = dict(
            graphene=2.13, graphene_oxide=2.13, go=2.13, gold=2.355, ice=3.661
        )  # unit: Angstrom
        target_res = supported_standards[standard_sample.lower()]
        apix, pixelSize_source = getPixelSize(data, return_pixelSize_source=True)
        if apix is None:
            raise HeliconError(
                '\\tERROR: cannot find "rlnImagePixelSize" or "rlnMicrographPixelSize"'
            )
        half_corner_res = 1.0 / (1 / (2 * apix) * (1 + np.sqrt(2)) / 2)
        if target_res <= half_corner_res:
            raise HeliconError(
                "\\tERROR: target resolution {target_res} Å for {param} is beyond the limit ({half_corner_res:.2f} Å = (1+sqrt(2))/2 * Nyquist resolution)"
            )

        search_range = 0.05  #  # default range: +/- 5%
        corner_res = 2 * apix / np.sqrt(2)
        res_low = target_res * (1 + search_range)
        res_high = max(corner_res, target_res * (1 - search_range))
        r_samples = 100  # 0.1% stepsize
        theta_samples = (
            int(
                np.pi
                / ((1 / res_high - 1 / res_low) / (r_samples - 1) / (1 / target_res))
            )
            + 1
        )  # equal sampling in radial and angular directions
        if args.verbose > 1:
            if args.verbose > 2:
                logger.info(
                    f"\tCurrent {pixelSize_source}: {apix} Å (Nyquist={2*apix} Å)"
                )
            logger.info(
                f"\tResolution range (±{search_range*100}%) to search for diffraction peak ({target_res} Å) of {param}: {res_low:.2f} -> {res_high:.2f} Å"
            )

        def fft_resolution_range(
            images,
            apix,
            res_low=0,
            res_high=0,
            r_samples=-1,
            theta_samples=180,
            return_R_only=False,
        ):
            import finufft
            import numpy as np

            R0 = 1 / res_low if res_low > 0 else 0
            R1 = 1 / res_high if res_high > 0 else 1 / (2 * apix)
            nr = r_samples if r_samples > 0 else min(images.shape[-2:]) // 2
            R = np.linspace(start=R0, stop=R1, num=nr, endpoint=True)
            if return_R_only:
                return R

            Theta = np.linspace(start=0, stop=np.pi, num=theta_samples, endpoint=False)
            Theta, R = np.meshgrid(Theta, R, indexing="ij")
            Y = (2 * np.pi * apix * R * np.sin(Theta)).flatten(order="C")
            X = (2 * np.pi * apix * R * np.cos(Theta)).flatten(order="C")
            from finufft import nufft2d2

            if len(images.shape) > 2:
                if len(images) > 1:
                    fft = nufft2d2(x=X, y=Y, f=images.astype(np.complex128), eps=1e-6)
                else:
                    fft = nufft2d2(
                        x=X, y=Y, f=images[0].astype(np.complex128), eps=1e-6
                    )
            else:
                fft = nufft2d2(x=X, y=Y, f=images.astype(np.complex128), eps=1e-6)
            if len(images.shape) > 2:
                new_shape = list(images.shape[:-2]) + list(R.shape)
            else:
                new_shape = R.shape
            fft = fft.reshape(new_shape)
            return fft

        def calibrateMag_process_one_micrograph(
            imageFile, apix, res_low, res_high, r_samples, theta_samples
        ):
            import mrcfile

            with mrcfile.open(imageFile) as mrc:
                images = mrc.data
            if len(images.shape) == 2:
                images = np.expand_dims(images, axis=0)
            fft = fft_resolution_range(
                images,
                apix,
                res_low,
                res_high,
                r_samples,
                theta_samples,
                return_R_only=False,
            )
            pwr = np.abs(fft)
            pwr_1d = pwr.max(axis=tuple(range(len(pwr.shape) - 1)))
            # pwr_1d = pwr.max(axis=1)
            # pwr_1d = pwr_1d.mean(axis=0)
            pwr_1d -= np.median(pwr_1d)
            from scipy.stats import median_abs_deviation

            pwr_curve = pwr_1d / median_abs_deviation(pwr_1d)
            n_ptcl = len(images)
            return (pwr_curve, n_ptcl)

        mapping = dict(
            rlnImagePixelSize="rlnImageName",
            rlnMicrographPixelSize="rlnMicrographName",
        )
        imageFiles = (
            data[mapping[pixelSize_source]]
            .str.split("@", expand=True)
            .iloc[:, -1]
            .unique()
        )

        from tqdm import tqdm
        from joblib import Parallel, delayed

        results = list(
            tqdm(
                Parallel(
                    return_as="generator",
                    n_jobs=args.cpu if len(imageFiles) > 1 else 1,
                )(
                    delayed(calibrateMag_process_one_micrograph)(
                        imageFile, apix, res_low, res_high, r_samples, theta_samples
                    )
                    for imageFile in imageFiles
                ),
                unit="micrograph",
                total=len(imageFiles),
                disable=len(imageFiles) < 2
                or (args.cpu > 1 and args.verbose < 2)
                or (args.cpu == 1 and args.verbose != 2),
            )
        )

        pwr_curves = []
        n_ptcls = []
        for result in results:
            pwr_curve, n_ptcl = result
            pwr_curves.append(pwr_curve)
            n_ptcls.append(n_ptcl)
        pwr_curves = np.vstack(pwr_curves)
        n_ptcls = np.array(n_ptcls)
        pwr_mean = (
            np.sum(pwr_curves * np.expand_dims(n_ptcls, axis=1), axis=0) / n_ptcls.sum()
        )
        from scipy.signal import detrend

        pwr_mean = detrend(pwr_mean)

        import mrcfile

        with mrcfile.open(imageFiles[0]) as mrc:
            images = mrc.data
        R = fft_resolution_range(
            images[0],
            apix,
            res_low,
            res_high,
            r_samples,
            theta_samples=180,
            return_R_only=True,
        )
        res_peak = 1 / R[np.argmax(pwr_mean)]
        apix_new = round(apix * target_res / res_peak, 3)  # precision: 0.1%

        if args.verbose > 1:
            outputFile = f"{Path(args.output_starFile).with_suffix('')}.calibrateMag.{pixelSize_source}={apix}.txt"
            np.savetxt(
                outputFile,
                np.hstack((R.reshape((len(R), 1)), pwr_mean.reshape((len(R), 1)))),
            )
            logger.info(
                f"\tAverage power spectra saved to {outputFile} using the original {pixelSize_source} {apix}"
            )
            import matplotlib.pyplot as plt

            plt.plot(R, pwr_mean, label=f"original pixel size={apix}")
            plt.axvline(x=1 / target_res, color="r", linestyle="dashed")
            plt.xlabel(f"Spatial Frequency (1/Å)")
            plt.ylabel("Power Spectra")
            plt.title(f"{standard_sample}: expected peak at {target_res} Å")
        if apix_new != apix:
            if args.verbose > 1:
                R2 = R * apix / apix_new
                outputFile2 = f"{Path(args.output_starFile).with_suffix('')}.calibrateMag.{pixelSize_source}={apix_new}.txt"
                np.savetxt(
                    outputFile2,
                    np.hstack((R2.reshape((len(R), 1)), pwr_mean.reshape((len(R), 1)))),
                )
                logger.info(
                    f"\tAverage power spectra also saved to {outputFile2} using the new, calibrated {pixelSize_source} {apix_new}"
                )
                plt.plot(R2, pwr_mean, label=f"calibrated pixel size={apix_new}")
            setPixelSize(data, apix_new=apix_new, update_defocus=True)
            if args.verbose > 0:
                logger.info(
                    f"\tCalibrated {pixelSize_source}: {apix_new} ({100*(apix_new-apix)/apix:.1f}% from the original {pixelSize_source} {apix}). {pixelSize_source}, rlnDefocusU, and rlnDefocusV have been updated to use the calibrated {pixelSize_source} {apix_new}"
                )
        else:
            if args.verbose > 0:
                logger.info(
                    f"\tCongratulations! Your original {pixelSize_source} {apix} is accurate without a need to adjust"
                )
        if args.verbose > 1:
            plt.legend(loc="upper right")
            plt.show()
    return data, index_d
