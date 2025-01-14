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
def symmetrize_transform_map(
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
        fraction=fraction,
        cpu=helicon.available_cpu(),
    )
    if axial_rotation or tilt:
        m = helicon.transform_map(m, rot=axial_rotation, tilt=tilt)
    return m


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

def itk_stitch(temp_dir):
    # ==========================================================================
    #
    #   Copyright NumFOCUS
    #
    #   Licensed under the Apache License, Version 2.0 (the "License");
    #   you may not use this file except in compliance with the License.
    #   You may obtain a copy of the License at
    #
    #          https://www.apache.org/licenses/LICENSE-2.0.txt
    #
    #   Unless required by applicable law or agreed to in writing, software
    #   distributed under the License is distributed on an "AS IS" BASIS,
    #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    #   See the License for the specific language governing permissions and
    #   limitations under the License.
    #
    # ==========================================================================*/
    import sys
    import os
    import itk
    import numpy as np
    import tempfile
    from pathlib import Path
    
    input_path = Path(temp_dir)
    output_path = Path(temp_dir)
    out_file = Path(temp_dir+"/itk_stitched.mrc")
    if not out_file.is_absolute(): 
        out_file = (output_path / out_file).resolve()
    
    
    dimension = 2
    
    stage_tiles = itk.TileConfiguration[dimension]()
    stage_tiles.Parse(str(input_path / "TileConfiguration.txt"))
    
    color_images = [] # for mosaic creation
    grayscale_images = [] # for registration
    for t in range(stage_tiles.LinearSize()):
        origin = stage_tiles.GetTile(t).GetPosition()
        filename = str(input_path / stage_tiles.GetTile(t).GetFileName())
        image = itk.imread(filename)
        spacing = image.GetSpacing()
    
      # tile configurations are in pixel (index) coordinates
      # so we convert them into physical ones
        for d in range(dimension):
            origin[d] *= spacing[d]
    
        image.SetOrigin(origin)
        color_images.append(image)

        image = itk.imread(filename, itk.F) # read as grayscale
        image.SetOrigin(origin)
        grayscale_images.append(image)
    
    # only float is wrapped as coordinate representation type in TileMontage
    montage = itk.TileMontage[type(grayscale_images[0]), itk.F].New()
    montage.SetMontageSize(stage_tiles.GetAxisSizes())
    for t in range(stage_tiles.LinearSize()):
        montage.SetInputTile(t, grayscale_images[t])
    
    print("Computing tile registration transforms")
    montage.Update()
    
    print("Writing tile transforms")
    actual_tiles = stage_tiles # we will update it later
    for t in range(stage_tiles.LinearSize()):
        index = stage_tiles.LinearIndexToNDIndex(t)
        regTr = montage.GetOutputTransform(index)
        tile = stage_tiles.GetTile(t)
        itk.transformwrite([regTr], str(output_path / (tile.GetFileName() + ".tfm")))
      
        # calculate updated positions - transform physical into index shift
        pos = tile.GetPosition()
        for d in range(dimension):
            pos[d] -= regTr.GetOffset()[d] / spacing[d]
        tile.SetPosition(pos)
        actual_tiles.SetTile(t, tile)
    actual_tiles.Write(str(output_path / "TileConfiguration.registered.txt"))
    
    print("Producing the mosaic")
    
    input_pixel_type = itk.template(color_images[0])[1][0]
    try:
        input_rgb_type = itk.template(input_pixel_type)[0]
        accum_type = input_rgb_type[itk.F] # RGB or RGBA input/output images
    except KeyError:
        accum_type = itk.D # scalar input / output images
    
    resampleF = itk.TileMergeImageFilter[type(color_images[0]), accum_type].New()
    resampleF.SetMontageSize(stage_tiles.GetAxisSizes())
    for t in range(stage_tiles.LinearSize()):
        resampleF.SetInputTile(t, color_images[t])
        index = stage_tiles.LinearIndexToNDIndex(t)
        resampleF.SetTileTransform(index, montage.GetOutputTransform(index))
    resampleF.Update()
    #itk.imwrite(resampleF.GetOutput(), str(out_file))
    print("Resampling complete")
    return np.array(resampleF.GetOutput())