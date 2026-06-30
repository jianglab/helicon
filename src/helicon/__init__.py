__version__ = "2026.05"

import numpy as np

# NumPy 2.x removed several names that numba still references
# during JIT compilation. Alias them for compatibility.
_numba_np_compat = {
    "trapz": "trapezoid",
    "in1d": "isin",
    "cumproduct": "cumprod",
}
for _old, _new in _numba_np_compat.items():
    if not hasattr(np, _old) and hasattr(np, _new):
        setattr(np, _old, getattr(np, _new))

from .lib.analysis import (
    calc_fsc,
    calc_fsc_from_fft,
    calc_fsc_per_shell,
    calc_frc_2d,
    cosine_similarity,
    cross_correlation_coefficient,
    estimate_helix_rotation_center_diameter,
    find_elbow_point,
    frc_score,
    get_cylindrical_mask,
    is_3d,
    is_amyloid,
    line_fit_projection,
    ms_ssim_score,
    mutual_information_score,
    r_factor_score,
    ssim_score,
    twist2pitch,
    estimate_inter_segment_distance,
    reset_inter_segment_distance,
    estimate_helicalTube_length,
)
from .lib.epu import (
    EPU_micrograph_path_2_movie_xml_path,
    EPU_xml_2_beamshift,
    assign_beamshift_groups,
    check_foilhole_xml_files,
    extract_beamshift,
    extract_data_collection_time,
    guess_data_collection_software,
    movie_filename_patterns,
    verify_data_collection_software,
)
from .lib.euler import (
    euler_relion2eman,
    euler_eman2relion,
    eman_euler2quaternion,
    relion_euler2quaternion,
    quaternion2euler,
    average_quaternions,
    average_relion_eulers,
    angular_distance,
)
from .lib.filters import (
    calculate_structural_factor,
    down_scale,
    generate_tapering_filter,
    low_high_pass_filter,
    match_structural_factors,
    normalize_mean_std,
    normalize_min_max,
    normalize_percentile,
    set_structural_factors,
    threshold_data,
)
from .lib.groups import (
    assign_time_groups,
    combine_groups,
    extract_timestamps,
    per_micrograph_ids,
    per_micrograph_mapping,
    propagate_ctf_median,
    sync_group_columns,
)
from .lib.io import (
    Relion_OpticsGroup_Parameters,
    assign_beamshifts_to_cluster,
    clean_cs_micrograph_path,
    cistem2dataframe,
    connect_cryosparc,
    cs2dataframe,
    dataframe2cs,
    dataframe2file,
    dataframe2star,
    dataframe_convert,
    dataframe_cryosparc_to_relion,
    dataframe_guess_data_type,
    dataframe_normalize_filename,
    eman_astigmatism_to_relion,
    get_dataframe_convention,
    get_relion_project_folder,
    guess_data_type,
    image2dataframe,
    images2dataframe,
    mrc2mrcs,
    relion_astigmatism_to_eman,
    star2dataframe,
    star_build_opticsgroup,
    star_dissolve_opticsgroup,
    getPixelSize,
    setPixelSize,
    pixelSizeAttrForImageAttr,
)
from .lib.io_mrc import (
    change_map_axes_order,
    display_map_orthoslices,
    get_image_number,
    get_image_size,
    read_image_2d,
)
from .lib.transforms import (
    apply_helical_symmetry,
    compute_phase_difference_across_meridian,
    compute_power_spectra,
    crop_center,
    crop_center_z,
    fft_crop,
    fft_rescale,
    flip_hand,
    get_clip,
    get_clip3d,
    get_rotated_clip,
    pad_to_size,
    rotate_shift_image,
    transform_image,
    transform_map,
)
from .lib.util import (
    setup_cache_dir,
    import_with_auto_install,
    DummyMemory,
    cache,
    color_print,
    getLogger,
    log_command_line,
    get_context_function_name,
    timedelta2string,
    Timer,
    unique,
    assign_to_groups,
    flatten,
    order_by_unique_counts,
    split_array,
    unique_attr_name,
    all_matched_attrs,
    first_matched_attr,
    DotDict,
    which,
    find_relion_project_folders,
    get_direct_url,
    get_file_size,
    download_file_from_url,
    get_emdb_id,
    is_file_readable,
    is_file_writable,
    file_ready,
    convert_file_path,
    convert_dataframe_file_path,
    check_required_columns,
    get_option_list,
    parse_param_str,
    validate_param_dict,
    has_shiny,
    has_streamlit,
    has_curvelet_fdct,
    has_curvelet_udct,
    has_curvelet_udct_gpu,
    available_cpu,
    omp_get_max_threads,
    omp_set_num_threads,
    get_terminal_size,
    bytes2units,
    ceil_power_of_10,
    encode_numpy,
    encode_PIL_Image,
    angular_difference,
    set_angle_range,
    set_to_periodic_range,
)

if has_curvelet_fdct():
    from .lib.curvelet import (
        curvelet_denoise_fdct,
        curvelet_denoise_batch_fdct,
        curvelet_denoise_udct,
        curvelet_denoise_batch_udct,
        curvelet_denoise_3d_udct,
        curvelet_denoise_3d_udct_tiled,
        curvelet_denoise_3d_mct,
        curvelet_denoise_3d_mct_tiled,
        curvelet_denoise_mct,
        curvelet_denoise_batch_mct,
        curvelet_denoise_udct_tiled,
        curvelet_denoise_fdct_tiled,
        curvelet_denoise_mct_tiled,
    )
try:
    from .lib.gauss import (
        AnisotropicGaussian,
        AnisotropicGaussianSet,
        IsotropicGaussian,
        IsotropicGaussianSet,
    )
except ImportError:
    pass

cache_dir = setup_cache_dir()

from .lib import dataset

try:
    from .lib import shiny
except ImportError:
    pass

try:
    from .lib import curvelet
except ImportError:
    pass
