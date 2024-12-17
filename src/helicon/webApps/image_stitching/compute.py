import os, pathlib
import numpy as np
import pandas as pd
import helicon


class MapInfo:
    def __init__(self, data=None, filename=None, url=None, emd_id=None, label="", apix=None, twist=None, rise=None, csym=1):
        non_nones = [p for p in [data, filename, url, emd_id] if p is not None]
        if len(non_nones)>1:
            raise ValueError(f"MapInfo(): only one of these parameters can be set: data, filename, url, emd_id")
        elif len(non_nones)<1:
            raise ValueError(f"MapInfo(): one of these parameters must be set: data, filename, url, emd_id")
        self.data = data
        self.filename = filename
        self.url = url
        self.emd_id = emd_id
        self.label = label
        self.apix = apix
        self.twist = twist
        self.rise = rise
        self.csym = csym

    def __repr__(self):
        return (f"MapInfo(label={self.label}, emd_id={self.emd_id}, "
                f"twist={self.twist}, rise={self.rise}, csym={self.csym}, "
                f"apix={self.apix})")
        
    def get_data(self):
        if self.data is not None:
            return self.data, self.apix
        if isinstance(self.filename, str) and len(self.filename) and pathlib.Path(self.filename).exists():
            self.data, self.apix = get_images_from_file(self.filename)
            return self.data, self.apix
        if isinstance(self.url, str) and len(self.url):
            self.data, self.apix = get_images_from_url(self.url)
            return self.data, self.apix
        if isinstance(self.emd_id, str) and len(self.emd_id):
            emdb = helicon.dataset.EMDB()
            self.data, self.apix = emdb(self.emd_id)
            return self.data, self.apix
        raise ValueError(f"MapInfo.get_data(): failed to obtain data")

def estimate_helix_rotation_center_diameter(data):
    """
    Returns:
        rotation (float): The rotation (degrees) needed to rotate the helix to horizontal direction.
        shift_y (float): The post-rotation vertical shift (pixels) needed to shift the helix to the box center in vertical direction.
        diameter (int): The estimated diameter (pixels) of the helix.
    """
    from skimage.measure import label, regionprops
    from skimage.morphology import closing
    import helicon

    thresh = np.max(data) * 0.2
    bw = closing(data > thresh, mode="ignore")
    label_image = label(bw)
    props = regionprops(label_image=label_image, intensity_image=data)
    props.sort(key=lambda x: x.area, reverse=True)
    angle = (
        np.rad2deg(props[0].orientation) + 90
    )  # relative to +x axis, counter-clockwise
    if abs(angle) > 90:
        angle -= 180
    rotation = helicon.set_to_periodic_range(angle, min=-180, max=180)

    # changed: images=data -> images=data.copy()
    # ValueError: buffer source array is read-only
    data_rotated = helicon.transform_image(image=data.copy(), rotation=rotation)
    bw = closing(data_rotated > thresh, mode="ignore")
    label_image = label(bw)
    props = regionprops(label_image=label_image, intensity_image=data_rotated)
    props.sort(key=lambda x: x.area, reverse=True)
    minr, minc, maxr, maxc = props[0].bbox
    diameter = maxr - minr + 1
    center = props[0].centroid
    shift_y = center[0] - data.shape[0] // 2

    return rotation, shift_y, diameter


@helicon.cache(
    cache_dir=str(helicon.cache_dir / "helicalProjection"), expires_after=7, verbose=0
)  # 7 days
def get_images_from_url(url):
    url_final = helicon.get_direct_url(url)  # convert cloud drive indirect url to direct url
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


@helicon.cache(expires_after=7, cache_dir=helicon.cache_dir / "helicalProjection", verbose=0)
def get_one_map_xyz_projects(map_info, length_z, map_projection_xyz_choices):
    label = map_info.label
    try:
        data, apix = map_info.get_data()
    except Exception as e:
        if map_info.filename:
            msg = f"Failed to obtain uploaded map {label}"
        elif map_info.url:
            msg = f"Failed to download the map from {map_info.url}"
        elif map_info.emd_id:
            msg = f"Failed to download the map from EMDB for {map_info.emd_id}"
        raise ValueError(msg)
    
    images = []
    image_labels = []
    if 'z' in map_projection_xyz_choices:
        rise = map_info.rise
        if rise>0:
            images += [helicon.crop_center_z(data, n=max(1, int(0.5 + length_z * rise / apix))).sum(axis=0)]
        else:
            images += [data.sum(axis=0)]
        image_labels += [label + ':Z']
    if 'y' in map_projection_xyz_choices:
        images += [data.sum(axis=1)]
        image_labels += [label + ':Y']
    if 'x' in map_projection_xyz_choices:
        images += [data.sum(axis=2)]
        image_labels += [label + ':X']
        
    return images, image_labels


@helicon.cache(expires_after=7, cache_dir=helicon.cache_dir / "helicalProjection", verbose=0)
def symmetrize_project_align_one_map(map_info, image_query, image_query_label, image_query_apix, rescale_apix, length_xy_factor, match_sf, angle_range, scale_range):
    if abs(map_info.twist) < 1e-3:
        return None
    
    try:
        data, apix = map_info.get_data()
    except:
        return None

    twist = map_info.twist
    rise = map_info.rise
    csym = map_info.csym
    label = map_info.label
    
    nz, ny, nx = data.shape
    if rescale_apix:
        image_ny, image_nx = image_query.shape
        new_apix = image_query_apix
        twist_work = helicon.set_to_periodic_range(twist, min=-180, max=180)
        if abs(twist_work)<90:
            pitch = 360/abs(twist_work) * rise 
        elif abs(twist_work)<180:
            pitch = 360/(180-abs(twist_work)) * rise
        else:
            pitch = image_nx * new_apix
        length = int(pitch / new_apix + image_nx * length_xy_factor)//2*2
        new_size = (length, image_ny, image_ny)

        data_work = helicon.low_high_pass_filter(data, low_pass_fraction=apix/new_apix)
    else:
        new_apix = apix
        new_size = (nz, ny, nx)
        data_work = data

    fraction = 5 * rise / (nz * apix)
    
    data_sym = helicon.apply_helical_symmetry(
        data = data_work,
        apix = apix,
        twist_degree = twist,
        rise_angstrom = rise,
        csym = csym,
        fraction = fraction,
        new_size = new_size,
        new_apix = new_apix,
        cpu = helicon.available_cpu()
    )
    proj = data_sym.sum(axis=2).T
        
    flip, scale, rotation_angle, shift_cartesian, similarity_score, aligned_image_moving = helicon.align_images(image_moving=image_query, image_ref=proj, scale_range=scale_range, angle_range=angle_range, check_polarity=True, check_flip=True, return_aligned_moving_image=True) 

    if match_sf:
        proj = helicon.match_structural_factors(data=proj, apix=new_apix, data_target=aligned_image_moving, apix_target=new_apix)

    return (flip, scale, rotation_angle, shift_cartesian, similarity_score, aligned_image_moving, image_query_label, proj, label)

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
	
