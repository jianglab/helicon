from pathlib import Path
import numpy as np
import pandas as pd
import helicon


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "denovo3D"), expires_after=7, verbose=0
)  # 7 days
def get_images_from_url(url):
    url_final = helicon.get_direct_url(
        url
    )  # convert cloud drive indirect url to direct url
    fileobj = helicon.download_file_from_url(url_final)
    if fileobj is None:
        raise ValueError(
            f"ERROR: {url} could not be downloaded. If this url points to a cloud drive file, make sure the link is a direct download link instead of a link for preview"
        )
    data, apix = get_images_from_file(fileobj.name)
    return data, apix


def get_images_from_file(imageFile):
    import mrcfile

    with mrcfile.open(imageFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, round(apix, 4)


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "denovo3D"), expires_after=7, verbose=0
)  # 7 days
def process_one_task(
    ti,
    ntasks,
    data,
    imageFile,
    imageIndex,
    twist,
    rise,
    rise_range,
    csym,
    tilt,
    tilt_range,
    psi,
    dy,
    apix2d_orig,
    denoise,
    low_pass,
    transpose,
    horizontalize,
    target_apix3d,
    target_apix2d,
    thresh_fraction,
    positive_constraint,
    tube_length,
    tube_diameter,
    tube_diameter_inner,
    reconstruct_length,
    sym_oversample,
    interpolation,
    fsc_test,
    return_3d,
    algorithm,
    verbose,
    logger,
):

    from helicon.commands import denovo3DBatch

    return denovo3DBatch.process_one_task(
        ti,
        ntasks,
        data,
        imageFile,
        imageIndex,
        twist,
        rise,
        rise_range,
        csym,
        tilt,
        tilt_range,
        psi,
        dy,
        apix2d_orig,
        denoise,
        low_pass,
        transpose,
        horizontalize,
        target_apix3d,
        target_apix2d,
        thresh_fraction,
        positive_constraint,
        tube_length,
        tube_diameter,
        tube_diameter_inner,
        reconstruct_length,
        sym_oversample,
        interpolation,
        fsc_test,
        return_3d,
        algorithm,
        verbose,
        logger,
    )
