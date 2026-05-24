import numpy as np

__all__ = [
    "align_images",
]


def align_images(
    image_moving,
    image_ref,
    scale_range,
    angle_range,
    check_polarity=True,
    check_flip=True,
    return_aligned_moving_image=False,
):
    """Align a moving image to a reference image using phase cross-correlation.

    Optimizes over rotation angle and optionally scale to maximize the
    cross-correlation coefficient between the images.

    Parameters
    ----------
    image_moving : ndarray
        Image to align.
    image_ref : ndarray
        Reference image.
    scale_range : float
        Range of scale to search (fraction, 0 to <1).
    angle_range : float
        Range of rotation angle to search (degrees).
    check_polarity : bool, optional
        If True, also check 180-degree rotated polarity. Defaults to True.
    check_flip : bool, optional
        If True, also check vertically flipped image. Defaults to True.
    return_aligned_moving_image : bool, optional
        If True, return the aligned moving image. Defaults to False.

    Returns
    -------
    tuple
        ``(scale, rotation_angle_degree, shift_cartesian, similarity_score)``
        or ``(scale, rotation_angle_degree, shift_cartesian, similarity_score,
        aligned_image)`` if *return_aligned_moving_image* is True.
    """
    from .. import (
        generate_tapering_filter,
        pad_to_size,
        threshold_data,
        transform_image,
    )
    from .analysis import cross_correlation_coefficient

    assert (
        0 <= scale_range < 1
    ), f"align_images(): {scale_range=} is out of valid range [0, 1)"

    if check_flip:
        result = align_images(
            image_moving=image_moving,
            image_ref=image_ref,
            scale_range=scale_range,
            angle_range=angle_range,
            check_flip=False,
            return_aligned_moving_image=return_aligned_moving_image,
        )

        image_moving_flip = image_moving[::-1, :]
        result_flip = align_images(
            image_moving=image_moving_flip,
            image_ref=image_ref,
            scale_range=scale_range,
            angle_range=angle_range,
            check_flip=False,
            return_aligned_moving_image=return_aligned_moving_image,
        )
        if result_flip[3] > result[3]:
            return (True, *result_flip)
        else:
            return (False, *result)

    from skimage.registration import phase_cross_correlation

    tapering_filter_moving = generate_tapering_filter(
        image_size=image_moving.shape, fraction_start=[0.8, 0.8]
    )
    padded_tapering_filter_moving = pad_to_size(tapering_filter_moving, image_ref.shape)

    padded_image_moving = pad_to_size(image_moving, image_ref.shape)

    padded_image_moving_work = threshold_data(
        padded_tapering_filter_moving * padded_image_moving, thresh_fraction=-1.0
    )

    tapering_filter_ref = generate_tapering_filter(
        image_size=image_ref.shape, fraction_start=[0.8, 0.8]
    )
    image_ref_work = threshold_data(
        tapering_filter_ref * image_ref, thresh_fraction=0.0
    )
    mask_image_ref_work = None

    mode = "wrap"

    best = [1e10, 1, 0, 0, None]

    def scale_rotation_score(x, angle0):
        if isinstance(x, np.ndarray):
            scale_log, angle = x
            scale = np.exp(scale_log)
        else:
            scale = 1.0
            angle = x
        angle += angle0

        rotated_scaled_padded_image_moving = transform_image(
            image=padded_image_moving_work, scale=scale, rotation=angle, mode="constant"
        )
        mask_rotated_scaled_padded_image_moving = None

        shift_cartesian, error, diffphase = phase_cross_correlation(
            reference_image=image_ref_work,
            moving_image=rotated_scaled_padded_image_moving,
            reference_mask=mask_image_ref_work,
            moving_mask=mask_rotated_scaled_padded_image_moving,
            overlap_ratio=0.5,
            disambiguate=False,
            normalization="phase",
        )
        shifted_rotated_scaled_padded_image_moving = transform_image(
            image=padded_image_moving_work,
            scale=scale,
            rotation=angle,
            post_translation=shift_cartesian,
            mode=mode,
        )
        shifted_rotated_scaled_padded_tapering_filter_moving = transform_image(
            image=padded_tapering_filter_moving,
            scale=scale,
            rotation=angle,
            post_translation=shift_cartesian,
            mode=mode,
        )
        mask = shifted_rotated_scaled_padded_tapering_filter_moving > 0
        score = -cross_correlation_coefficient(
            image_ref_work[mask], shifted_rotated_scaled_padded_image_moving[mask]
        )
        if score < best[0]:
            best[0] = score
            best[1] = scale
            best[2] = angle
            best[3] = shift_cartesian
            best[4] = shifted_rotated_scaled_padded_image_moving
        return score

    if scale_range > 0:
        from scipy.optimize import minimize

        result = minimize(
            scale_rotation_score,
            x0=[0, 0],
            args=(0),
            bounds=[
                (-np.log(1 + scale_range), np.log(1 + scale_range)),
                (-angle_range, angle_range),
            ],
            method="Nelder-Mead",
            options=dict(xatol=0.01),
        )
        if check_polarity:
            minimize(
                scale_rotation_score,
                x0=[0, 0],
                args=(180),
                bounds=[
                    (-np.log(1 + scale_range), np.log(1 + scale_range)),
                    (-angle_range, angle_range),
                ],
                method="Nelder-Mead",
                options=dict(xatol=0.01),
            )
    elif angle_range > 0:
        from scipy.optimize import minimize_scalar

        minimize_scalar(
            scale_rotation_score,
            args=(0),
            bounds=(-angle_range, angle_range),
            method="bounded",
        )
        if check_polarity:
            minimize_scalar(
                scale_rotation_score,
                args=(180),
                bounds=(-angle_range, angle_range),
                method="bounded",
            )

    (
        _,
        scale,
        rotation_angle_degree,
        shift_cartesian,
        shifted_rotated_scaled_padded_image_moving,
    ) = best

    if shifted_rotated_scaled_padded_image_moving is None:
        shifted_rotated_scaled_padded_image_moving = padded_image_moving_work

    shifted_rotated_scaled_padded_tapering_filter_moving = transform_image(
        image=padded_tapering_filter_moving,
        scale=scale,
        rotation=rotation_angle_degree,
        post_translation=shift_cartesian,
        mode=mode,
    )
    mask = shifted_rotated_scaled_padded_tapering_filter_moving > 0
    similarity_score = cross_correlation_coefficient(
        image_ref_work[mask], shifted_rotated_scaled_padded_image_moving[mask]
    )

    shifted_rotated_scaled_padded_image_moving = transform_image(
        image=padded_image_moving,
        scale=scale,
        rotation=rotation_angle_degree,
        post_translation=shift_cartesian,
        mode=mode,
    )

    if return_aligned_moving_image:
        return (
            scale,
            rotation_angle_degree,
            shift_cartesian,
            similarity_score,
            shifted_rotated_scaled_padded_image_moving,
        )
    else:
        return scale, rotation_angle_degree, shift_cartesian, similarity_score
