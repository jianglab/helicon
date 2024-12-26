from pathlib import Path
import numpy as np
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


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "denovo3D"), expires_after=7, verbose=0
)  # 7 days
def get_images_from_emdb(emdb_id):
    emdb = helicon.dataset.EMDB()
    data, apix = emdb(emdb_id)
    if data is None:
        raise IOError(f"ERROR: failed to download {emdb_id} from EMDB")

    return data, round(apix, 4)


def get_images_from_file(imageFile):
    import mrcfile

    with mrcfile.open(imageFile) as mrc:
        apix = float(mrc.voxel_size.x)
        data = mrc.data
    return data, round(apix, 4)


def generate_xyz_projections(map3d, is_amyloid=False, apix=None):
    proj_xyz = [map3d.sum(axis=i) for i in [2, 1, 0]]
    if is_amyloid:
        nz = map3d.shape[0]
        nz_center = int(round(4.75 / apix))
        z0 = nz // 2 - nz_center // 2
        proj_xyz[-1] = map3d[z0 : z0 + nz_center].sum(axis=0)
    return proj_xyz


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "denovo3D"), expires_after=7, verbose=0
)  # 7 days
def symmetrize_project(
    data,
    apix,
    twist_degree,
    rise_angstrom,
    csym=1,
    fraction=1.0,
    new_size=None,
    new_apix=None,
    axial_rotation=0,
    tilt=0,
):
    if new_apix > apix:
        data_work = helicon.low_high_pass_filter(
            data, low_pass_fraction=apix / new_apix
        )
    else:
        data_work = data
    m = helicon.apply_helical_symmetry(
        data=data_work,
        apix=apix,
        twist_degree=twist_degree,
        rise_angstrom=rise_angstrom,
        csym=csym,
        new_size=new_size,
        new_apix=new_apix,
        fraction=1 / 3,
        cpu=helicon.available_cpu(),
    )
    if axial_rotation or tilt:
        m = helicon.transform_map(m, rot=axial_rotation, tilt=tilt)
    proj = np.transpose(m.sum(axis=-1))[:, ::-1]
    proj = proj[np.newaxis, :, :]
    return proj


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
