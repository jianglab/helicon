"""denovo3D tab — de novo 3D reconstruction from a single 2D image.

Ported from src/helicon/webApps/denovo3D/app.py (Shiny Express) into the
consolidated Helicon Lab Shiny module pattern.

Uses ``helicon.shiny.image_gallery`` (plain function) instead of
``helicon.shiny.image_select`` (express module), ``pio.to_html`` + ``render.ui``
instead of ``render_plotly``, and ``input_action_button`` instead of
``input_task_button``.
"""

import asyncio
import itertools
import logging
import mrcfile
import pathlib
import random
import tempfile
import traceback
import mrcfile
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

import helicon
from shiny import reactive, render, req, ui, module

from ..lib.shared_state import ProjectState

from ..lib import denovo3d_pipeline
from ..lib.helical_projection_utils import (
    _combine_images_for_display,
    _image_stitching_x_positions,
)

logger = logging.getLogger(__name__)

BOOKMARK_DEFAULTS = {
    "input_mode_images": ("dn_input_mode_images", "url"),
    "url_images": ("dn_url_images", ""),
    "show_emdb": ("dn_show_emdb_input_mode", False),
    "is_3d": ("dn_is_3d", False),
    "ignore_blank": ("dn_ignore_blank", True),
    "plot_scores": ("dn_plot_scores", True),
    "show_download": ("dn_show_download_buttons", False),
    "display_size": ("dn_selected_image_display_size", 128),
    "rec_length": ("dn_reconstruct_length_rise", 3),
    "target_apix2d": ("dn_target_apix2d", 5),
    "target_apix3d": ("dn_target_apix3d", 5),
    "sym_oversample": ("dn_sym_oversample", -1),
    "lr_alpha": ("dn_lr_alpha", -1),
    "lr_l1_ratio": ("dn_lr_l1_ratio", 0.5),
    "top_n": ("dn_top_n_results", 10),
    "lr_algorithm": ("dn_lr_algorithm", "elasticnet"),
    "positive": ("dn_positive_constraint", -1),
    "interpolation": ("dn_interpolation", "linear"),
    "score_metric": ("dn_score_metric", "cosine"),
    "input_ui_type": ("dn_input_ui_type", "Slider"),
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-14046/map/emd_14046.map.gz",
    ),
}
_url_key = "empiar-10940_job010"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fig_to_html(fig):
    """Convert a plotly figure to responsive HTML for render.ui."""
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displayModeBar": False},
        default_height=400,
    )
    return ui.HTML(html)


def _estimate_helix_rotation_center_diameter(
    data, estimate_rotation=True, estimate_center=True, threshold=0
):
    """Estimate the rotation, vertical center shift, and diameter of a helix.

    Returns
    -------
    tuple
        (rotation_deg, shift_y_px, diameter_px)
    """
    from skimage.morphology import closing

    ny, nx = data.shape

    def _weighted_params(mask, intensity):
        ys, xs = np.where(mask)
        if len(ys) < 2:
            return 0.0, 0.0, ny
        w = intensity[ys, xs].astype(np.float64)
        w = w - w.min() + 1e-8
        cw = w.sum()
        cy = (ys * w).sum() / cw
        cx = (xs * w).sum() / cw
        uy = ys - cy
        ux = xs - cx
        i_yy = (uy * uy * w).sum() / cw
        i_xx = (ux * ux * w).sum() / cw
        i_xy = (uy * ux * w).sum() / cw
        theta = 0.5 * np.arctan2(2.0 * i_xy, i_yy - i_xx)
        angle = np.rad2deg(theta) + 90.0
        if abs(angle) > 90.0:
            angle -= 180.0
        diameter = int(ys.max() - ys.min() + 1)
        if estimate_center:
            shift = ny // 2 - cy
        else:
            shift = 0.0
        return angle, shift, diameter

    bw = closing(data > threshold, mode="ignore")
    mask = bw > 0
    if not mask.any():
        return 0.0, 0.0, ny

    if estimate_rotation:
        rotation, _, _ = _weighted_params(mask, data)
        rotation = helicon.set_to_periodic_range(rotation, min=-180, max=180)
        data_rotated = helicon.transform_image(image=data, rotation=rotation)
    else:
        rotation = 0.0
        data_rotated = data

    bw_rot = closing(data_rotated > threshold, mode="ignore")
    mask_rot = bw_rot > 0
    if not mask_rot.any():
        return rotation, 0.0, ny

    _, shift_y, diameter = _weighted_params(mask_rot, data_rotated)
    return rotation, shift_y, diameter


# ═══════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════


@module.ui
def denovo3d_tab_ui():
    return ui.layout_sidebar(
        # ── Sidebar ──
        ui.sidebar(
            ui.navset_pill(
                # ── Input 2D Images panel ──
                ui.nav_panel(
                    "Inputs",
                    ui.div(
                        ui.input_radio_buttons(
                            "dn_input_mode_images",
                            "How to obtain the input images:",
                            choices=["upload", "url", "emdb"],
                            selected="url",
                            inline=True,
                        ),
                        ui.output_ui("dn_create_input_image_files_ui"),
                        ui.output_ui("dn_display_emdb_info_ui"),
                        id="dn_input_image_files",
                        style="display: flex; flex-direction: column; align-items: flex-start;",
                    ),
                    ui.output_ui("dn_map_xyz_projections_gallery"),
                    ui.output_ui("dn_generate_ui_symmetrize_projection"),
                    ui.div(
                        ui.output_ui("dn_select_image_gallery"),
                        id="dn_image_selection",
                        style="max-height: 80vh; overflow-y: auto; display: flex; flex-direction: column; align-items: center;",
                    ),
                ),
                # ── Parameters panel ──
                ui.nav_panel(
                    "Parameters",
                    ui.layout_columns(
                        col_widths=6,
                        style="align-items: flex-end;",
                    ),
                    ui.layout_columns(
                        # ui.input_checkbox("dn_show_emdb_input_mode", "Show EMDB input mode", value=False),
                        ui.input_checkbox(
                            "dn_is_3d", "The input is a 3D map", value=False
                        ),
                        ui.input_checkbox(
                            "dn_ignore_blank", "Ignore blank input images", value=True
                        ),
                        ui.input_checkbox("dn_plot_scores", "Plot scores", value=True),
                        ui.input_checkbox(
                            "dn_show_download_buttons",
                            "Show download buttons",
                            value=False,
                        ),
                        col_widths=6,
                        style="align-items: flex-end;",
                    ),
                    ui.layout_columns(
                        ui.input_numeric(
                            "dn_cpu",
                            "# CPUs",
                            min=1,
                            max=helicon.available_cpu(),
                            value=helicon.available_cpu(),
                            step=1,
                            update_on="blur",
                        ),
                        ui.input_numeric(
                            "dn_selected_image_display_size",
                            "Selected image display size (pixel)",
                            min=32,
                            max=512,
                            value=128,
                            step=32,
                            update_on="blur",
                        ),
                        ui.tooltip(
                            ui.input_numeric(
                                "dn_reconstruct_length_rise",
                                "Reconstruction length (rise)",
                                min=1,
                                value=3,
                                step=1,
                                update_on="blur",
                            ),
                            "Reconstruction length as the number of rises",
                        ),
                        ui.tooltip(
                            ui.input_numeric(
                                "dn_target_apix2d",
                                "Target image pixel size (A)",
                                min=-1,
                                value=5,
                                step=1,
                                update_on="blur",
                            ),
                            "Down-scale images to this pixel size. <=0 -> no down-scaling.",
                        ),
                        ui.tooltip(
                            ui.input_numeric(
                                "dn_target_apix3d",
                                "Target voxel size (A)",
                                min=-1,
                                value=5,
                                step=1,
                                update_on="blur",
                            ),
                            "Voxel size of 3D reconstruction. 0 -> set to target 2D. <0 -> auto.",
                        ),
                        ui.tooltip(
                            ui.input_numeric(
                                "dn_sym_oversample",
                                "Helical/Csym oversampling factor",
                                min=-1,
                                value=-1,
                                step=1,
                                update_on="blur",
                            ),
                            "Controls # of equations in A matrix. Larger -> slower but better. Negative = auto.",
                        ),
                        ui.tooltip(
                            ui.input_numeric(
                                "dn_lr_alpha",
                                "Weight of regularization",
                                min=0,
                                value=-1,
                                step=1e-4,
                                update_on="blur",
                            ),
                            "Only for elasticnet/lasso/ridge. Default: 1e-4 for elasticnet/lasso, 1 for ridge.",
                        ),
                        ui.tooltip(
                            ui.input_numeric(
                                "dn_lr_l1_ratio",
                                "L1 regularization ratio",
                                min=0.0,
                                max=1.0,
                                value=0.5,
                                step=0.1,
                                update_on="blur",
                            ),
                            "Ratio (0 to 1) of L1 in L1/L2 combined regularization.",
                        ),
                        ui.input_numeric(
                            "dn_top_n_results",
                            "# of results to show",
                            min=-1,
                            value=10,
                            step=1,
                            update_on="blur",
                        ),
                        col_widths=6,
                        style="align-items: flex-end;",
                    ),
                    ui.layout_columns(
                        col_widths=12,
                        style="align-items: flex-end;",
                    ),
                    ui.tooltip(
                        ui.input_radio_buttons(
                            "dn_lr_algorithm",
                            "Reconstruction algorithm",
                            ["elasticnet", "lasso", "ridge", "lsq"],
                            selected="elasticnet",
                            inline=True,
                        ),
                        "Choose the regularization algorithm for the least-squares reconstruction",
                    ),
                    ui.layout_columns(
                        ui.tooltip(
                            ui.input_radio_buttons(
                                "dn_positive_constraint",
                                "Positive constraint",
                                {-1: "Auto", 0: "No", 1: "Yes"},
                                selected=-1,
                                inline=True,
                            ),
                            "How positive constraint is used for the 3D reconstruction",
                        ),
                        ui.tooltip(
                            ui.input_radio_buttons(
                                "dn_interpolation",
                                "Interpolation method",
                                {"linear": "Linear", "nn": "Nearest Neighbor"},
                                selected="linear",
                                inline=True,
                            ),
                            "Interpolation method for reconstruction",
                        ),
                        ui.tooltip(
                            ui.input_select(
                                "dn_score_metric",
                                "Score metric",
                                {
                                    "cosine": "Cosine similarity",
                                    "ssim": "SSIM",
                                    "ms_ssim": "MS-SSIM",
                                    "mutual_information": "Mutual information",
                                    "composite": "Composite (mean of all 4)",
                                },
                                selected="cosine",
                            ),
                            "Metric used to rank reconstruction quality.",
                        ),
                        ui.input_radio_buttons(
                            "dn_input_ui_type",
                            "Image transformation parameters input type:",
                            ["Slider", "Input box"],
                            inline=True,
                        ),
                        col_widths=6,
                        style="align-items: flex-end;",
                    ),
                    ui.layout_columns(
                        ui.input_action_button(
                            "dn_clear_cache",
                            label="Clear joblib cache",
                            class_="btn-primary",
                            style="width: 200px;",
                        ),
                        col_widths=6,
                        style="align-items: flex-end;",
                    ),
                ),
            ),
            width="33vw",
            style="display: flex; flex-direction: column; height: 100%;",
        ),
        # ── Main content ──
        ui.h1(
            "Denovo3D: de novo helical indexing and 3D reconstruction",
            style="font-weight: bold;",
        ),
        ui.div(
            ui.output_ui("dn_generate_image_gallery_multiple"),
            ui.output_ui("dn_generate_image_transformation_multiple"),
            ui.output_ui("dn_image_stitching_transformed"),
            ui.output_ui("dn_display_stitched_image"),
            style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0",
        ),
        ui.div(
            ui.div(
                ui.output_ui("dn_generate_image_gallery_single"),
                ui.accordion(
                    ui.accordion_panel(
                        "Filtering options:",
                        ui.tooltip(
                            ui.input_numeric(
                                "dn_binning",
                                "Binning:",
                                value=1,
                                min=1,
                                max=100,
                                step=1,
                                update_on="blur",
                            ),
                            "Default binning makes the image smallest dimension ≤ 128 pixels.",
                        ),
                        ui.layout_columns(
                            ui.input_numeric(
                                "dn_lp_angst",
                                "Low pass filtering (Å):",
                                value=-1,
                                step=0.1,
                                update_on="blur",
                            ),
                            ui.input_numeric(
                                "dn_hp_angst",
                                "High pass filtering (Å):",
                                value=-1,
                                step=0.1,
                                update_on="blur",
                            ),
                            col_widths=(6, 6),
                            style="align-items: flex-end;",
                        ),
                    ),
                    id="dn_filtering_options",
                    open=False,
                    width="100%",
                ),
                style="display: flex; flex-flow: column wrap; align-items: flex-start; gap: 10px; margin-bottom: 0",
            ),
            ui.output_ui("dn_generate_image_transformation_single"),
            style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0",
        ),
        ui.div(
            ui.tooltip(
                ui.card(
                    style="height: 115px",
                ),
                "Will reconstruct 3D map when min. twist = max twist and min. rise = max rise",
            ),
            ui.output_ui("dn_twist_card"),
            ui.output_ui("dn_rise_card"),
            ui.card(
                ui.card_header("Csym"),
                ui.input_numeric(
                    "dn_csym",
                    "n",
                    value=1,
                    min=1,
                    step=1,
                    width="70px",
                    update_on="blur",
                ),
                style="height: 115px",
            ),
            ui.output_ui("dn_show_run_button"),
            ui.panel_conditional(
                "input['dn_twist_min']!==input['dn_twist_max'] || input['dn_rise_min']!==input['dn_rise_max']",
                ui.input_action_button(
                    "dn_stop_denovo3D",
                    label="Stop",
                    class_="btn-danger",
                    style="width: 115px; height: 115px;",
                ),
            ),
            style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0",
        ),
        ui.div(
            ui.output_ui("dn_scores_plot"),
            ui.output_ui("dn_reconstructed_projections"),
            ui.output_ui("dn_download_map_section"),
        ),
        ui.HTML(
            "<i><p style='margin:2px 0'>Developed by the <a href='https://jianglab.science.psu.edu/helicon' target='_blank'>Jiang Lab</a>. "
            "Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Server
# ═══════════════════════════════════════════════════════════════════════════


@module.server
def denovo3d_tab_server(input, output, session, project: ProjectState):
    # ── Reactive values ──────────────────────────────────────────────
    url_images_init = reactive.value("")
    input_data = reactive.value(None)
    map_symmetrized = reactive.value(None)
    map_xyz_projections = reactive.value([])
    all_images = reactive.value(None)

    displayed_image_ids = reactive.value([])
    displayed_images = reactive.value([])
    displayed_image_title = reactive.value("Select an image:")
    displayed_image_labels = reactive.value([])

    initial_selected_image_indices = reactive.value([0])
    selected_images_original = reactive.value([])
    selected_images_thresholded = reactive.value([])
    selected_images_thresholded_rotated_shifted = reactive.value([])
    selected_images_thresholded_rotated_shifted_cropped = reactive.value([])
    selected_images_title = reactive.value("Selected image:")
    selected_images_labels = reactive.value([])

    img_transpose_rv = reactive.value(False)
    img_flip_rv = reactive.value(False)
    img_negate_rv = reactive.value(False)
    new_initial_image = reactive.value(True)
    pre_rotation_rv = reactive.value(0)
    threshold_rv = reactive.value(0.0)
    apix_rv = reactive.value(1.0)
    shift_y_rv = reactive.value(0)
    vertical_crop_size_rv = reactive.value(32)
    horizontal_crop_size_rv = reactive.value(256)

    t_ui_counter = reactive.value(0)
    selected_images_rotated_shifted = reactive.value([])
    transformed_images_displayed = reactive.value([])
    transformed_images_labels = reactive.value([])
    transformed_images_links = reactive.value([])
    transformed_images_x_offsets = reactive.value([])

    stitched_image_displayed = reactive.value([])
    stitched_image_labels = reactive.value([])
    stitched_image_links = reactive.value([])

    initial_image = reactive.value([])
    display_initial_image_value = reactive.value([""])

    reconstruction_results = reactive.value([])
    reconstructed_projection_images = reactive.value([])
    reconstructed_projection_labels = reactive.value([])

    run_button_text = reactive.value("Search Parameters")
    abort_flag = [False]  # mutable container shared with the task (not reactive)

    # ── Helper: UI for single-image transformation controls ──────────

    def _transformation_ui_single():
        """Build the transformation control card for a single selected image."""
        if input.dn_input_ui_type() == "Slider":
            card_content = ui.card(
                ui.layout_columns(
                    ui.input_checkbox(
                        "dn_img_transpose", "Transpose", img_transpose_rv()
                    ),
                    ui.input_checkbox("dn_img_flip", "Flip", img_flip_rv()),
                    ui.input_checkbox(
                        "dn_img_negate", "Invert contrast", img_negate_rv()
                    ),
                    ui.input_slider(
                        "dn_pre_rotation",
                        "Rotation (deg)",
                        min=-20,
                        max=20,
                        value=pre_rotation_rv(),
                        step=0.1,
                    ),
                    ui.input_slider(
                        "dn_threshold",
                        "Threshold",
                        min=threshold_rv() - 1,
                        max=threshold_rv() + 1,
                        value=threshold_rv(),
                        step=0.001,
                    ),
                    ui.input_slider(
                        "dn_apix",
                        "Pixel size (A)",
                        min=0.0,
                        max=10.0,
                        value=apix_rv(),
                        step=0.001,
                    ),
                    ui.input_slider(
                        "dn_shift_y",
                        "Vertical shift (A)",
                        min=-100,
                        max=100,
                        value=shift_y_rv(),
                        step=0.1,
                    ),
                    ui.input_slider(
                        "dn_vertical_crop_size",
                        "Vertical crop (pixel)",
                        min=32,
                        max=256,
                        value=vertical_crop_size_rv(),
                        step=2,
                    ),
                    ui.input_slider(
                        "dn_horizontal_crop_size",
                        "Horizontal crop (pixel)",
                        min=32,
                        max=256,
                        value=horizontal_crop_size_rv(),
                        step=2,
                    ),
                    col_widths=4,
                ),
                ui.layout_columns(
                    ui.input_action_button(
                        "dn_auto_transform",
                        label="Auto Transform",
                        class_="btn-primary",
                    ),
                    ui.input_action_button(
                        "dn_reset_transform",
                        label="Reset Transform",
                        class_="btn-primary",
                    ),
                    col_widths=6,
                ),
                id="dn_single_card_ui",
            )
        else:
            card_content = ui.card(
                ui.layout_columns(
                    ui.input_checkbox(
                        "dn_img_transpose", "Transpose", img_transpose_rv()
                    ),
                    ui.input_checkbox("dn_img_flip", "Flip", img_flip_rv()),
                    ui.input_checkbox(
                        "dn_img_negate", "Invert contrast", img_negate_rv()
                    ),
                    ui.input_numeric(
                        "dn_pre_rotation",
                        "Rotation (deg)",
                        min=-20,
                        max=20,
                        value=pre_rotation_rv(),
                        step=0.1,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "dn_threshold",
                        "Threshold",
                        min=threshold_rv() - 1,
                        max=threshold_rv() + 1,
                        value=threshold_rv(),
                        step=0.001,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "dn_apix",
                        "Pixel size (A)",
                        min=0.0,
                        max=10.0,
                        value=apix_rv(),
                        step=0.001,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "dn_shift_y",
                        "Vertical shift (A)",
                        min=-100,
                        max=100,
                        value=shift_y_rv(),
                        step=0.1,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "dn_vertical_crop_size",
                        "Vertical crop (pixel)",
                        min=32,
                        max=256,
                        value=vertical_crop_size_rv(),
                        step=2,
                        update_on="blur",
                    ),
                    ui.input_numeric(
                        "dn_horizontal_crop_size",
                        "Horizontal crop (pixel)",
                        min=32,
                        max=256,
                        value=horizontal_crop_size_rv(),
                        step=2,
                        update_on="blur",
                    ),
                    col_widths=4,
                ),
                ui.layout_columns(
                    ui.input_action_button(
                        "dn_auto_transform",
                        label="Auto Transform",
                        class_="btn-primary",
                    ),
                    ui.input_action_button(
                        "dn_reset_transform",
                        label="Reset Transform",
                        class_="btn-primary",
                    ),
                    col_widths=6,
                ),
                id="dn_single_card_ui",
            )

        # Update ranges for new/existing images
        if new_initial_image():
            imgs = initial_image()
            if imgs:
                apix = round(all_images().apix, 4)
                ui.update_numeric("dn_apix", value=apix, max=apix * 2)
                ny, nx = np.shape(imgs[0])
                ui.update_numeric("dn_vertical_crop_size", min=32, max=ny)
                ui.update_numeric("dn_horizontal_crop_size", min=32, max=nx)
                ui.update_numeric("dn_shift_y", min=-ny // 2, max=ny // 2)
                if ny > nx:
                    ui.update_checkbox("dn_img_transpose", value=True)
                    ui.update_checkbox("dn_img_negate", value=True)
                else:
                    ui.update_checkbox("dn_img_transpose", value=False)
                    ui.update_checkbox("dn_img_negate", value=False)
            new_initial_image.set(False)
        else:
            if len(selected_images_thresholded()):
                ny, nx = np.shape(selected_images_thresholded()[0])
            else:
                imgs = initial_image()
                ny, nx = np.shape(imgs[0]) if imgs else (128, 128)
            ui.update_numeric("dn_vertical_crop_size", min=32, max=ny)
            ui.update_numeric("dn_horizontal_crop_size", min=32, max=nx)
            ui.update_numeric("dn_shift_y", min=-ny // 2, max=ny // 2)

            images = initial_image()
            if img_negate_rv():
                images = [-img for img in images]
            if images:
                min_val = float(np.min([np.min(img) for img in images]))
                max_val = float(np.max([np.max(img) for img in images]))
                step_val = (max_val - min_val) / 100
                ui.update_numeric(
                    "dn_threshold",
                    min=round(min_val, 3),
                    max=round(max_val, 3),
                    step=round(step_val, 3),
                )

        return card_content

    # ── Helper: UI for per-image transformation controls ─────────────

    def _transformation_ui_group(prefix, shift_scale=100):
        return ui.card(
            ui.layout_columns(
                ui.input_slider(
                    prefix + "_pre_rotation",
                    "Rotation (deg)",
                    min=-45,
                    max=45,
                    value=0,
                    step=0.1,
                ),
                ui.input_slider(
                    prefix + "_shift_x",
                    "Horizontal shift (pixel)",
                    min=-shift_scale,
                    max=shift_scale,
                    value=0,
                    step=1,
                ),
                ui.input_slider(
                    prefix + "_shift_y",
                    "Vertical shift (pixel)",
                    min=-100,
                    max=100,
                    value=0,
                    step=1,
                ),
                col_widths=4,
            ),
            id=f"{prefix}_card",
        )

    # ── Render: sidebar dynamic UI ───────────────────────────────────

    @render.ui
    def dn_create_input_image_files_ui():
        displayed_images.set([])
        ret = []
        mode = input.dn_input_mode_images()
        if mode == "upload":
            ret.append(
                ui.input_file(
                    "dn_upload_images",
                    "Upload the input images in MRC format (.mrcs, .mrc)",
                    accept=[".mrcs", ".mrc", ".star"],
                    placeholder="mrcs, mrc or star file",
                )
            )
        elif mode == "url":
            default_url = url_images_init() or _urls[_url_key][0]
            ret.append(
                ui.input_text(
                    "dn_url_images",
                    "Download URL for a RELION or cryoSPARC 2D class mrc(s) file",
                    value=default_url,
                )
            )
        elif mode == "emdb":
            ret.append(
                ui.div(
                    ui.input_text(
                        "dn_emdb_id",
                        "Specify an amyloid structure EMDB ID",
                        value="EMD-14046",
                        width="calc(100% - 110px)",
                    ),
                    ui.input_action_button(
                        "dn_randomize_emdb_id",
                        "Randomize",
                        class_="btn-primary",
                        style="width: 100px; height: 30px; margin-bottom: 14px; display: flex; align-items: center; justify-content: center;",
                    ),
                    style="display: flex; flex-wrap: wrap; width: 100%; justify-content: space-between; align-items: flex-end; gap: 10px;",
                )
            )
        return ret

    @render.ui
    def dn_display_emdb_info_ui():
        req(input_data() is not None)
        req(len(input_data().data))
        req(input_data().emdb_id)
        emdb = helicon.dataset.EMDB()
        emd_id_num = input.dn_emdb_id().split("-")[-1].split("_")[-1]
        req(emd_id_num in emdb.emd_ids)
        emd_id = f"EMD-{emd_id_num}"
        info = emdb.get_info(emd_id)
        nz, ny, nx = input_data().data.shape
        apix = input_data().apix
        s = (
            f"<p><a href='https://www.ebi.ac.uk/emdb/{emd_id}' target='_blank'>{emd_id}</a>"
            f": {info.title}"
            f"<br>{nx}x{ny}x{nz}|{apix}\u00c5/pixel|resolution={info.resolution}\u00c5|"
            f"twist={info.twist}\u00b0|pitch={info.pitch:,}\u00c5|rise={info.rise}\u00c5|{info.csym}</p>"
        )
        return ui.HTML(s)

    @render.ui
    def dn_map_xyz_projections_gallery():
        projs = map_xyz_projections()
        if not projs or len(projs) == 0:
            return ui.div()
        return helicon.shiny.image_gallery(
            id=session.ns("dn_xyz_proj"),
            label=reactive.value("XYZ Projections"),
            images=map_xyz_projections,
            image_labels=reactive.value("X Y Z".split()),
            image_size=reactive.value(128),
            enable_selection=False,
            style="margin-bottom: 20px;",
        )

    @render.ui
    def dn_generate_ui_symmetrize_projection():
        req(input_data() is not None)
        req(input_data().is_3d)
        req(len(input_data().data))
        twist = 0
        pitch = np.nan
        rise = 0
        csym = 1
        if input_data().emdb_id:
            try:
                emdb = helicon.dataset.EMDB()
                emd_id_num = input_data().emdb_id.split("-")[-1].split("_")[-1]
                if emd_id_num in emdb.emd_ids:
                    emd_id = f"EMD-{emd_id_num}"
                    info = emdb.get_info(emd_id)
                    twist = info.twist
                    rise = info.rise
                    csym = int(info.csym[1:])
                    pitch = info.pitch
            except Exception:
                logger.debug("EMDB info lookup failed, using defaults")
        width = int((input_data().data.shape[2] * input_data().apix) / 5) // 4 * 4
        length = (
            int(round(0.5 * pitch / 5)) // 4 * 4 if not np.isnan(pitch) else width * 2
        )

        params_row = ui.div(
            ui.tags.hr(),
            ui.input_numeric(
                "dn_input_twist",
                "Twist (deg)",
                value=twist,
                step=0.1,
                width="140px",
                update_on="blur",
            ),
            ui.input_numeric(
                "dn_input_rise",
                "Rise (A)",
                value=rise,
                step=0.1,
                width="140px",
                update_on="blur",
            ),
            ui.input_numeric(
                "dn_input_csym",
                "Csym",
                value=csym,
                min=1,
                step=1,
                width="140px",
                update_on="blur",
            ),
            ui.input_numeric(
                "dn_input_apix",
                "Input voxel size (A)",
                value=input_data().apix,
                min=0.1,
                step=0.1,
                width="140px",
                update_on="blur",
            ),
            ui.input_numeric(
                "dn_output_apix",
                "Output pixel size (A)",
                value=5,
                min=0.1,
                step=0.1,
                width="140px",
                update_on="blur",
            ),
            ui.input_numeric(
                "dn_output_axial_rotation",
                "Axial rotation (deg)",
                value=0,
                min=-20,
                max=20,
                step=1,
                width="140px",
                update_on="blur",
            ),
            ui.input_numeric(
                "dn_output_width",
                "Output width (pixels)",
                value=width,
                min=32,
                step=16,
                width="140px",
                update_on="blur",
            ),
            ui.input_numeric(
                "dn_output_length",
                "Output length (pixels)",
                value=length,
                min=32,
                step=16,
                width="140px",
                update_on="blur",
            ),
            ui.input_numeric(
                "dn_output_tilt",
                "Tilt out of plane (deg)",
                value=0,
                min=-90,
                max=90,
                step=1,
                width="140px",
                update_on="blur",
            ),
            ui.input_numeric(
                "dn_gauss_noise_std",
                "Gaussian noise standard deviation",
                value=1.0,
                width="140px",
                update_on="blur",
            ),
            style="display: flex; flex-wrap: wrap; flex-direction: row; gap: 4px; align-items: flex-end; justify-content: center;",
        )
        return ui.div(
            params_row,
            ui.input_action_button(
                "dn_symmetrization_projection",
                "Generate projection",
                class_="btn-primary",
                style="margin-bottom: 10px;",
            ),
            style="display: flex; flex-direction: column; justify-content: center;",
        )

    @render.ui
    def dn_select_image_gallery():
        imgs = displayed_images()
        if not imgs or len(imgs) == 0:
            return ui.div()
        return helicon.shiny.image_gallery(
            id=session.ns("dn_select_image"),
            label=displayed_image_title,
            images=displayed_images,
            image_labels=displayed_image_labels,
            image_size=reactive.value(128),
            initial_selected_indices=initial_selected_image_indices,
            enable_selection=True,
            allow_multiple_selection=True,
        )

    @render.ui
    def dn_generate_ui_symmetrize_projection_download():
        req(input_data() is not None)
        req(input_data().is_3d)
        req(map_symmetrized() is not None)

        dl = render.download(
            label="Download symmetrized input map",
            filename="helicon_denovo3d_input_map.mrc",
        )

        @dl
        def _download():
            with tempfile.NamedTemporaryFile(suffix=".mrc") as temp:
                with mrcfile.new(temp.name, overwrite=True) as mrc:
                    mrc.set_data(map_symmetrized())
                    mrc.voxel_size = input.dn_output_apix()
                with open(temp.name, "rb") as file:
                    yield file.read()

        return dl

    # ── Render: main area galleries ──────────────────────────────────

    @render.ui
    def dn_generate_image_gallery_multiple():
        imgs = displayed_images()
        if not imgs or len(imgs) == 0:
            return ui.div()
        sel = input.dn_select_image()
        if sel is None or len(sel) == 0:
            return ui.div()
        req(0 <= min(sel))
        req(max(sel) < len(imgs))
        n_images_selected = len(selected_images_original())
        if n_images_selected == 1:
            return ui.div()
        return helicon.shiny.image_gallery(
            id=session.ns("dn_display_selected_image_multi"),
            label=selected_images_title,
            images=selected_images_rotated_shifted,
            image_labels=selected_images_labels,
            image_size=reactive.value(128),
            justification="left",
            display_dashed_line=True,
            enable_selection=False,
        )

    @render.ui
    @reactive.event(selected_images_original)
    def dn_generate_image_transformation_multiple():
        imgs = displayed_images()
        if not imgs or len(imgs) == 0:
            return ui.div()
        sel = input.dn_select_image()
        if sel is None or len(sel) == 0:
            return ui.div()
        if not (0 <= min(sel) and max(sel) < len(imgs)):
            return ui.div()
        n_images_selected = len(selected_images_original())
        if n_images_selected == 1:
            return ui.div()

        dim = len(selected_images_original()[0])
        shift_scale = int(0.9 * dim) * n_images_selected
        container = ui.div(
            style="display: flex; flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 0"
        )

        for i, label in enumerate(selected_images_labels()):
            curr_counter = i
            container.append(
                ui.row(
                    _transformation_ui_group(
                        f"dn_t_ui_group_{curr_counter}", shift_scale=shift_scale
                    )
                )
            )

            id_rotation = f"dn_t_ui_group_{curr_counter}_pre_rotation"
            id_x_shift = f"dn_t_ui_group_{curr_counter}_shift_x"
            id_y_shift = f"dn_t_ui_group_{curr_counter}_shift_y"

            @reactive.effect
            @reactive.event(input.dn_select_image)
            def _update_multi_originals():
                selected_images_rotated_shifted.set(
                    [displayed_images()[j] for j in input.dn_select_image()]
                )
                transformed_images_x_offsets.set(np.zeros(len(input.dn_select_image())))

            def _make_transform_multi_img(_ii, _id_rot, _id_ys):
                @reactive.effect
                @reactive.event(input[_id_rot], input[_id_ys])
                def _fn():
                    req(len(selected_images_original()))
                    rotated = selected_images_rotated_shifted().copy()
                    if input[_id_rot]() != 0 or input[_id_ys]() != 0:
                        rotated[_ii] = helicon.transform_image(
                            image=selected_images_original()[_ii].copy(),
                            rotation=input[_id_rot](),
                            post_translation=(input[_id_ys](), 0),
                        )
                    selected_images_rotated_shifted.set(rotated)

                return _fn

            _make_transform_multi_img(i, id_rotation, id_y_shift)

            def _make_update_multi_displayed(_xi, _id_xs):
                @reactive.effect
                @reactive.event(selected_images_rotated_shifted, input[_id_xs])
                def _fn():
                    req(len(selected_images_rotated_shifted()))
                    curr_offsets = transformed_images_x_offsets().copy()
                    if len(curr_offsets) != len(selected_images_rotated_shifted()):
                        curr_offsets = np.zeros(len(selected_images_rotated_shifted()))
                    for img_i in range(len(selected_images_rotated_shifted())):
                        if img_i == _xi:
                            curr_offsets[_xi] = input[_id_xs]()
                        else:
                            curr_offsets[img_i] = input[
                                f"dn_t_ui_group_{img_i}_shift_x"
                            ]()
                    image_work = _combine_images_for_display(
                        selected_images_rotated_shifted(), curr_offsets
                    )
                    transformed_images_displayed.set([image_work])
                    transformed_images_labels.set(["Combined images"])
                    transformed_images_links.set([""])
                    transformed_images_x_offsets.set(curr_offsets)

                return _fn

            _make_update_multi_displayed(i, id_x_shift)

        container.append(
            ui.input_action_button(
                "dn_perform_stitching",
                label="Stitch Images",
                class_="btn-primary",
                style="width: 100%; margin-top: 10px;",
            )
        )
        t_ui_counter.set(t_ui_counter() + n_images_selected)
        return container

    @render.ui
    def dn_image_stitching_transformed():
        imgs = displayed_images()
        if not imgs or len(imgs) == 0:
            return ui.div()
        sel = input.dn_select_image()
        if sel is None or len(sel) == 0:
            return ui.div()
        req(0 <= min(sel))
        req(max(sel) < len(imgs))
        n_images_selected = len(selected_images_original())
        if n_images_selected == 1:
            return ui.div()
        req(len(transformed_images_displayed()))
        return helicon.shiny.image_gallery(
            id=session.ns("dn_display_transformed_images"),
            label=reactive.value("Transformed selected images:"),
            images=transformed_images_displayed,
            image_labels=transformed_images_labels,
            image_links=transformed_images_links,
            image_size=reactive.value(256),
            justification="left",
            display_dashed_line=True,
            enable_selection=False,
        )

    @render.ui
    def dn_display_stitched_image():
        imgs = displayed_images()
        if not imgs or len(imgs) == 0:
            return ui.div()
        sel = input.dn_select_image()
        if sel is None or len(sel) == 0:
            return ui.div()
        req(0 <= min(sel))
        req(max(sel) < len(imgs))
        n_images_selected = len(selected_images_original())
        if n_images_selected == 1:
            return ui.div()
        req(len(stitched_image_displayed()))
        return helicon.shiny.image_gallery(
            id=session.ns("dn_display_stitched_image"),
            label=reactive.value("Stitched image:"),
            images=stitched_image_displayed,
            image_labels=stitched_image_labels,
            image_links=stitched_image_links,
            image_size=reactive.value(128),
            display_dashed_line=True,
            justification="left",
            enable_selection=False,
        )

    @render.ui
    def dn_generate_image_gallery_single():
        init = initial_image()
        if not init:
            return ui.div()
        sel = input.dn_select_image()
        if sel is None or len(displayed_images()) == 0:
            return ui.div()
        req(0 <= min(sel))
        req(max(sel) < len(displayed_images()))
        n_images_selected = len(init)
        if n_images_selected == 1:
            return helicon.shiny.image_gallery(
                id=session.ns("dn_display_selected_image_single"),
                label=selected_images_title,
                images=selected_images_thresholded_rotated_shifted_cropped,
                image_labels=display_initial_image_value,
                image_size=input.dn_selected_image_display_size,
                justification="left",
                enable_selection=False,
                display_dashed_line=True,
            )
        return ui.div()

    @render.ui
    @reactive.event(initial_image)
    def dn_generate_image_transformation_single():
        init = initial_image()
        if not init:
            return ui.div()
        sel = input.dn_select_image()
        if sel is None or len(displayed_images()) == 0:
            return ui.div()
        if not (0 <= min(sel) and max(sel) < len(displayed_images())):
            return ui.div()
        n_images_selected = len(init)
        if n_images_selected == 1:
            return ui.div(
                _transformation_ui_single(),
                style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0",
            )
        return ui.div()

    @render.ui
    def dn_twist_card():
        return ui.card(
            ui.card_header("Twist (deg)"),
            ui.div(
                ui.input_numeric(
                    "dn_twist_min",
                    "min",
                    value=0.1,
                    step=0.1,
                    width="70px",
                    update_on="blur",
                ),
                ui.input_numeric(
                    "dn_twist_max",
                    "max",
                    value=2.0,
                    step=0.1,
                    width="70px",
                    update_on="blur",
                ),
                ui.input_numeric(
                    "dn_twist_step",
                    "step",
                    value=0.1,
                    step=0.1,
                    width="70px",
                    update_on="blur",
                ),
                ui.panel_conditional(
                    "input['dn_twist_min']===input['dn_twist_max'] && input['dn_rise_min']===input['dn_rise_max']",
                    ui.input_radio_buttons(
                        "dn_twisting_handedness",
                        "Reconstruct with:",
                        [
                            "Left-handed twisting (force negative twist)",
                            "Right-handed twisting (force positive twist)",
                        ],
                    ),
                ),
                style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0",
            ),
            style="height: 115px",
        )

    @render.ui
    def dn_rise_card():
        return ui.card(
            ui.card_header("Rise (A)"),
            ui.div(
                ui.input_numeric(
                    "dn_rise_min",
                    "min",
                    value=4.75,
                    step=0.1,
                    width="70px",
                    update_on="blur",
                ),
                ui.input_numeric(
                    "dn_rise_max",
                    "max",
                    value=4.75,
                    step=0.1,
                    width="70px",
                    update_on="blur",
                ),
                ui.input_numeric(
                    "dn_rise_step",
                    "step",
                    value=0.1,
                    step=0.01,
                    width="70px",
                    update_on="blur",
                ),
                style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0",
            ),
            style="height: 115px",
        )

    @render.ui
    def dn_show_run_button():
        return ui.input_action_button(
            "dn_run_denovo3D",
            run_button_text(),
            class_="btn-primary",
            style="width: 115px; height: 115px;",
        )

    @render.download(
        filename="helicon_denovo3d_reconstructed_map.mrc",
    )
    def dn_download_map():
        req(len(reconstruction_results()) == 1)
        result = reconstruction_results()[0]
        _score, return_data, _params = result
        *_projection_data, rec3d_map = return_data

        req(rec3d_map is not None)
        apix = input_data().apix
        rec3d_map = np.asarray(rec3d_map, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".mrc") as temp:
            with mrcfile.new(temp.name, overwrite=True) as mrc:
                mrc.set_data(rec3d_map)
                mrc.voxel_size = apix
            with open(temp.name, "rb") as file:
                yield file.read()

    @render.ui
    def dn_download_map_section():
        res = reconstruction_results()
        req(len(res) == 1)
        req(res[0][1][8] is not None)
        from htmltools import tags

        return ui.div(
            tags.a(
                "Download reconstructed map",
                id=session.ns("dn_download_map"),
                class_="btn btn-primary shiny-download-link",
                href="",
                target="_blank",
            ),
            style="display: flex; justify-content: center; margin-top: 10px;",
        )

    # ── Render: scores plot ──────────────────────────────────────────

    @render.ui
    def dn_scores_plot():
        res = reconstruction_results()
        if len(res) <= 1:
            return ui.div()
        results_arr = np.zeros((3, len(res)), dtype=float)
        for ri, result in enumerate(res):
            score, _projs, params = result
            twist, rise = params[5], params[6]
            results_arr[0, ri] = twist
            results_arr[1, ri] = rise
            results_arr[2, ri] = score

        n_twists = len(np.unique(results_arr[0, :]))
        n_rises = len(np.unique(results_arr[1, :]))

        if n_twists > 1 and n_rises > 1:
            x = results_arr[0, :]
            y = results_arr[1, :]
            scores = results_arr[2, :]
            vmin = np.min(scores)
            x_unique = np.sort(np.unique(x))
            y_unique = np.sort(np.unique(y))
            X, Y = np.meshgrid(x_unique, y_unique, indexing="ij")
            Z = np.zeros_like(X) + vmin
            for j in range(Z.shape[1]):
                for i in range(Z.shape[0]):
                    vals = [
                        scores[si]
                        for si in range(len(x))
                        if y[si] == Y[i, j] and x[si] == X[i, j]
                    ]
                    if vals:
                        Z[i, j] = np.max(vals)
            fig = px.imshow(
                Z.T,
                x=x_unique,
                y=y_unique,
                origin="lower",
                labels=dict(x="Twist (deg)", y="Rise (A)", color="Score"),
                color_continuous_scale="viridis",
            )
            fig.update_layout(coloraxis_colorbar_title="Score")
            max_idx = np.unravel_index(np.argmax(Z), Z.shape)
            max_x = x_unique[max_idx[0]]
            max_y = y_unique[max_idx[1]]
            fig.add_shape(
                type="rect",
                x0=max_x - (x_unique[1] - x_unique[0]) / 2,
                y0=max_y - (y_unique[1] - y_unique[0]) / 2,
                x1=max_x + (x_unique[1] - x_unique[0]) / 2,
                y1=max_y + (y_unique[1] - y_unique[0]) / 2,
                line=dict(color="red", width=2),
                fillcolor=None,
            )
        elif n_twists > 1 or n_rises > 1:
            if n_twists > 1:
                x = results_arr[0, :]
                x_title = "Twist (deg)"
                hovertemplate = "Twist: %{x}deg<br>Score: %{y}"
            else:
                x = results_arr[1, :]
                x_title = "Rise (A)"
                hovertemplate = "Rise: %{x}A<br>Score: %{y}"
            y = results_arr[2, :]
            sort_idx = np.argsort(x)
            x = np.array(x)[sort_idx]
            y = np.array(y)[sort_idx]
            fig = px.line(x=x, y=y, color_discrete_sequence=["blue"], markers=True)
            fig.update_layout(
                xaxis_title=x_title, yaxis_title="Score", showlegend=False
            )
            fig.update_traces(hovertemplate=hovertemplate)
        else:
            return ui.div()

        return _fig_to_html(fig)

    # ── Render: reconstructed projections ────────────────────────────

    @render.ui
    def dn_reconstructed_projections():
        req(len(reconstructed_projection_images()))
        img_list = reconstructed_projection_images()
        label_list = reconstructed_projection_labels()

        pairs = []
        for img, label_value in zip(img_list, label_list):
            img_enc = helicon.encode_numpy(img)
            label_value = str(label_value)
            pairs.extend(
                [
                    ui.div(
                        {"class": "label-row", "style": "margin: 10px 0;"},
                        ui.h4(label_value),
                    ),
                    ui.div(
                        {
                            "class": "image-row",
                            "style": "max-height: 100vh; overflow-y: auto; display: flex; flex-direction: column; align-items: left; margin-bottom: 5px",
                        },
                        ui.img(
                            {"src": img_enc, "style": "max-width: 100%; height: auto;"}
                        ),
                    ),
                ]
            )
        return ui.div(pairs)

    # ══════════════════════════════════════════════════════════════════
    # Reactive effects: data loading
    # ══════════════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(input.dn_input_mode_images)
    def _reset_input_data_ui():
        input_data.set(None)
        ui.update_checkbox("dn_is_3d", value=False)
        map_symmetrized.set(None)
        map_xyz_projections.set(None)
        selected_images_thresholded_rotated_shifted_cropped.set(None)

    @reactive.effect
    @reactive.event(input.dn_input_mode_images, input.dn_upload_images)
    def _get_image_from_upload():
        req(input.dn_input_mode_images() == "upload")
        fileinfo = input.dn_upload_images()
        req(fileinfo)
        image_file = fileinfo[0]["datapath"]

        if image_file.split(".")[-1] == "star":
            df = helicon.star2dataframe(str(image_file))
            indices = range(len(df))
            if "rlnHelixImageName" in df.columns:
                data = []

                for i in indices:
                    imageFile = pathlib.Path(df.loc[i, "rlnHelixImageName"])
                    with mrcfile.open(imageFile) as mrc:
                        apix = round(float(mrc.voxel_size.x), 4)
                        data.append(mrc.data)
                is_3d = False
                emdb_id = None
                is_amyloid = False
            else:
                try:
                    data, apix = denovo3d_pipeline.get_images_from_file(image_file)
                except Exception as e:
                    logger.error("Failed to read uploaded images", exc_info=True)
                    ui.modal_show(
                        ui.modal(
                            f"failed to read the uploaded 2D images from {fileinfo[0]['name']}",
                            title="File upload error",
                            easy_close=True,
                            footer=None,
                        )
                    )
                    return
                emdb_id = helicon.get_emdb_id(fileinfo[0]["name"])
                is_3d = emdb_id or helicon.is_3d(data)
                is_amyloid = helicon.is_amyloid(emdb_id)
        else:
            try:
                data, apix = denovo3d_pipeline.get_images_from_file(image_file)
            except Exception as e:
                logger.error("Failed to read uploaded images", exc_info=True)
                ui.modal_show(
                    ui.modal(
                        f"failed to read the uploaded 2D images from {fileinfo[0]['name']}",
                        title="File upload error",
                        easy_close=True,
                        footer=None,
                    )
                )
                return
            emdb_id = helicon.get_emdb_id(fileinfo[0]["name"])
            is_3d = emdb_id or helicon.is_3d(data)
            is_amyloid = helicon.is_amyloid(emdb_id)

        d = helicon.DotDict(
            data=data, apix=apix, emdb_id=emdb_id, is_3d=is_3d, is_amyloid=is_amyloid
        )
        input_data.set(d)
        ui.update_checkbox("dn_is_3d", value=is_3d)

    @reactive.effect
    @reactive.event(input.dn_input_mode_images, input.dn_url_images)
    def _get_images_from_url():
        req(input.dn_input_mode_images() == "url")
        req(len(input.dn_url_images()) > 0)
        url = input.dn_url_images()
        try:
            data, apix = denovo3d_pipeline.get_images_from_url(url)
        except Exception as e:
            logger.error("Failed to download images from URL", exc_info=True)
            ui.modal_show(
                ui.modal(
                    f"failed to download 2D images from {input.dn_url_images()}",
                    title="File download error",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        emdb_id = helicon.get_emdb_id(url)
        is_3d = emdb_id or helicon.is_3d(data)
        is_amyloid = helicon.is_amyloid(emdb_id)
        d = helicon.DotDict(
            data=data, apix=apix, emdb_id=emdb_id, is_3d=is_3d, is_amyloid=is_amyloid
        )
        input_data.set(d)
        ui.update_checkbox("dn_is_3d", value=is_3d)

    @reactive.effect
    @reactive.event(input.dn_randomize_emdb_id)
    def _randomize_emdb_id():
        emdb = helicon.dataset.EMDB()
        ids = emdb.amyloid_atlas_ids()
        ui.update_text("dn_emdb_id", value=f"EMD-{random.choice(ids)}")

    @reactive.effect
    @reactive.event(input.dn_input_mode_images, input.dn_emdb_id)
    def _get_images_from_emdb():
        req(input.dn_input_mode_images() == "emdb")
        emdb_id = input.dn_emdb_id()
        req(len(emdb_id) > 0)
        try:
            data, apix = denovo3d_pipeline.get_images_from_emdb(emdb_id=emdb_id)
        except Exception as e:
            logger.error("Failed to obtain map from EMDB", exc_info=True)
            ui.modal_show(
                ui.modal(
                    f"failed to obtain {emdb_id} map from EMDB",
                    title="File download error",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        is_amyloid = helicon.is_amyloid(emdb_id)
        d = helicon.DotDict(
            data=data, apix=apix, emdb_id=emdb_id, is_3d=True, is_amyloid=is_amyloid
        )
        input_data.set(d)
        ui.update_checkbox("dn_is_3d", value=True)

    @reactive.effect
    @reactive.event(input_data)
    def _update_all_images_from_2d():
        req(input_data())
        req(len(input_data().data))
        if input_data().is_3d:
            all_images.set(None)
        else:
            d = helicon.DotDict(data=input_data().data, apix=input_data().apix)
            all_images.set(d)

    @reactive.effect
    @reactive.event(input.dn_is_3d)
    def _update_input_data_is_3d():
        req(input_data())
        d = input_data()
        d.is_3d = input.dn_is_3d()
        d2 = helicon.DotDict(d)
        input_data.set(d2)

    @reactive.effect
    @reactive.event(input_data)
    def _get_xyz_projections():
        req(input_data())
        req(len(input_data().data))
        if input_data().is_3d:
            proj_xyz = denovo3d_pipeline.generate_xyz_projections(
                input_data().data,
                is_amyloid=input_data().is_amyloid,
                apix=input_data().apix,
            )
            map_xyz_projections.set(proj_xyz)
        else:
            map_xyz_projections.set(None)

    @reactive.effect
    @reactive.event(input.dn_symmetrization_projection)
    def _update_all_images_from_3d():
        req(input_data())
        req(len(input_data().data))
        req(input_data().is_3d)
        m = denovo3d_pipeline.symmetrize_transform_map(
            data=input_data().data,
            apix=input.dn_input_apix(),
            twist_degree=input.dn_input_twist(),
            rise_angstrom=input.dn_input_rise(),
            csym=input.dn_input_csym(),
            new_size=(
                input.dn_output_length(),
                input.dn_output_width(),
                input.dn_output_width(),
            ),
            new_apix=input.dn_output_apix(),
            axial_rotation=input.dn_output_axial_rotation(),
            tilt=input.dn_output_tilt(),
        )
        map_symmetrized.set(m)
        proj = np.transpose(m.sum(axis=-1))[:, ::-1]
        proj = proj[np.newaxis, :, :]

        def _add_noise(image, noise, thres=1e-3):
            sigma = np.std(image[image > thres])
            image += np.random.normal(scale=sigma * noise, size=image.shape)
            return image

        if input.dn_gauss_noise_std() > 0:
            proj[0, :, :] = _add_noise(proj[0, :, :], input.dn_gauss_noise_std())

        d = helicon.DotDict(data=proj, apix=input.dn_output_apix())
        all_images.set(d)

    # ══════════════════════════════════════════════════════════════════
    # Reactive effects: image display & selection
    # ══════════════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(all_images, input.dn_ignore_blank)
    def _get_displayed_images():
        if all_images() is None:
            displayed_images.set([])
            return
        req(len(all_images().data))
        data = all_images().data
        apix = all_images().apix
        if isinstance(data, np.ndarray):
            if len(data.shape) < 3:
                data = np.expand_dims(data, axis=0)
        n = len(data)
        if n:
            ny, nx = data[0].shape[:2]
            images = [data[i] for i in range(n)]
            display_seq_all = np.arange(n, dtype=int)
            if input.dn_ignore_blank():
                included = [
                    display_seq_all[i]
                    for i in range(n)
                    if np.max(images[display_seq_all[i]])
                    > np.min(images[display_seq_all[i]])
                ]
                images = [images[i] for i in included]
            else:
                included = list(display_seq_all)
            image_labels = [f"{i+1}" for i in included]
            title = f"{len(images)}/{n} images|{nx}x{ny}|{apix}\u00c5/pixel|length={round(nx*apix):,}\u00c5"
        else:
            included = []
            images = []
            image_labels = []
            title = ""
        displayed_image_ids.set(included)
        displayed_images.set(images)
        displayed_image_title.set(title)
        displayed_image_labels.set(image_labels)

    @reactive.effect
    @reactive.event(displayed_images)
    def _update_binning_default():
        req(len(displayed_images()))
        all_shapes = [img.shape for img in displayed_images()]
        max_dim = max([max(s) for s in all_shapes])
        suggested = max(1, int(np.ceil(max_dim / 256)))
        try:
            current = input.dn_binning()
        except Exception:
            logger.debug("dn_binning input not ready, defaulting to 1")
            current = 1
        if current != suggested:
            ui.update_numeric("dn_binning", value=suggested)

    @reactive.effect
    @reactive.event(
        input.dn_select_image,
        displayed_images,
        input.dn_lp_angst,
        input.dn_hp_angst,
        input.dn_binning,
    )
    def _on_image_selected():
        imgs = displayed_images()
        if not imgs:
            return
        sel = input.dn_select_image()
        if not sel:
            selected_images_original.set([])
            selected_images_labels.set([])
            return

        stitched_image_displayed.set([])
        stitched_image_labels.set([])
        stitched_image_links.set([])

        images = [imgs[i] for i in sel]

        try:
            apix = input.dn_apix()
        except Exception:
            logger.debug("dn_apix input not ready, falling back to all_images().apix")
            apix = round(all_images().apix, 4)

        try:
            binning = input.dn_binning()
        except Exception:
            logger.debug("dn_binning input not ready, defaulting to 1")
            binning = 1
        if binning and binning > 1:
            from skimage.transform import rescale

            images = [
                rescale(
                    img,
                    1.0 / binning,
                    anti_aliasing=True,
                    order=3,
                    preserve_range=True,
                )
                for img in images
            ]
            apix = apix * binning

        do_filtering = False
        low_pass_fraction = -1
        high_pass_fraction = -1
        try:
            lp = input.dn_lp_angst()
            hp = input.dn_hp_angst()
        except Exception:
            logger.debug(
                "dn_lp_angst/dn_hp_angst inputs not ready, disabling filtering"
            )
            lp = -1
            hp = -1
        if lp and lp > 0:
            low_pass_fraction = 2 * apix / lp
            do_filtering = True
        if hp and hp > 0:
            high_pass_fraction = 2 * apix / hp
            do_filtering = True
        if do_filtering:
            images = [
                helicon.low_high_pass_filter(
                    img,
                    low_pass_fraction=low_pass_fraction,
                    high_pass_fraction=high_pass_fraction,
                )
                for img in images
            ]

        selected_images_original.set(images)
        selected_images_labels.set([displayed_image_labels()[i] for i in sel])
        reconstruction_results.set([])

    @reactive.effect
    @reactive.event(selected_images_original, ignore_init=False)
    def _set_initial_image():
        req(len(selected_images_original()))
        n_images_selected = len(selected_images_original())
        if n_images_selected == 1:
            initial_image.set(selected_images_original())
            new_initial_image.set(True)
        else:
            if len(stitched_image_displayed()):
                initial_image.set(stitched_image_displayed())
            else:
                initial_image.set([])

    # ══════════════════════════════════════════════════════════════════
    # Reactive effects: thresholding & transformation
    # ══════════════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(initial_image, input.dn_img_negate)
    def _update_threshold_scale():
        req(len(initial_image()))
        images = initial_image()
        if input.dn_img_negate():
            images = [-img for img in images]
        min_val = float(np.min([np.min(img) for img in images]))
        max_val = float(np.max([np.max(img) for img in images]))
        step_val = (max_val - min_val) / 100
        from skimage.filters import threshold_otsu

        thresh_value = float(np.median([threshold_otsu(img) for img in images]))
        ui.update_numeric(
            "dn_threshold",
            value=round(thresh_value, 3),
            min=round(min_val, 3),
            max=round(max_val, 3),
            step=round(step_val, 3),
        )

    @reactive.effect
    @reactive.event(
        initial_image, input.dn_threshold, input.dn_img_transpose, input.dn_img_flip
    )
    def _threshold_selected_images():
        req(len(initial_image()))
        images = initial_image()
        if input.dn_img_negate():
            tmp = [
                helicon.threshold_data(-img, thresh_value=input.dn_threshold())
                for img in images
            ]
        else:
            tmp = [
                helicon.threshold_data(img, thresh_value=input.dn_threshold())
                for img in images
            ]
        if input.dn_img_transpose():
            tmp = [np.transpose(img) for img in tmp]
        if input.dn_img_flip():
            tmp = [np.fliplr(img) for img in tmp]
        selected_images_thresholded.set(tmp)

    # Sync checkbox/reactive values
    @reactive.effect
    @reactive.event(input.dn_img_transpose)
    def _sync_transpose():
        if img_transpose_rv() != input.dn_img_transpose():
            img_transpose_rv.set(input.dn_img_transpose())

    @reactive.effect
    @reactive.event(input.dn_img_flip)
    def _sync_flip():
        if img_flip_rv() != input.dn_img_flip():
            img_flip_rv.set(input.dn_img_flip())

    @reactive.effect
    @reactive.event(input.dn_img_negate)
    def _sync_negate():
        if img_negate_rv() != input.dn_img_negate():
            img_negate_rv.set(input.dn_img_negate())

    @reactive.effect
    @reactive.event(input.dn_pre_rotation)
    def _sync_pre_rotation():
        if pre_rotation_rv() != input.dn_pre_rotation():
            pre_rotation_rv.set(input.dn_pre_rotation())

    @reactive.effect
    @reactive.event(input.dn_threshold)
    def _sync_threshold():
        if threshold_rv() != input.dn_threshold():
            threshold_rv.set(input.dn_threshold())

    @reactive.effect
    @reactive.event(input.dn_apix)
    def _sync_apix():
        if apix_rv() != input.dn_apix():
            apix_rv.set(input.dn_apix())

    @reactive.effect
    @reactive.event(input.dn_shift_y)
    def _sync_shift_y():
        if shift_y_rv() != input.dn_shift_y():
            shift_y_rv.set(input.dn_shift_y())

    @reactive.effect
    @reactive.event(input.dn_vertical_crop_size)
    def _sync_crop_y():
        if vertical_crop_size_rv() != input.dn_vertical_crop_size():
            vertical_crop_size_rv.set(input.dn_vertical_crop_size())

    @reactive.effect
    @reactive.event(input.dn_horizontal_crop_size)
    def _sync_crop_x():
        if horizontal_crop_size_rv() != input.dn_horizontal_crop_size():
            horizontal_crop_size_rv.set(input.dn_horizontal_crop_size())

    @reactive.effect
    @reactive.event(input.dn_auto_transform, threshold_rv)
    def _auto_transform():
        req(all_images())
        req(len(selected_images_thresholded()))
        images = selected_images_thresholded()
        ny = int(np.max([img.shape[0] for img in images]))
        nx = int(np.max([img.shape[1] for img in images]))

        if input_data().is_3d:
            estimate_rotation = False
            estimate_center = False
        else:
            estimate_rotation = True
            estimate_center = True

        tmp = np.array(
            [
                _estimate_helix_rotation_center_diameter(
                    img,
                    threshold=np.max(img) * 0.2,
                    estimate_rotation=estimate_rotation,
                    estimate_center=estimate_center,
                )
                for img in images
            ]
        )
        rotation = np.mean(tmp[:, 0])
        shift_y = np.mean(tmp[:, 1]) * input.dn_apix()
        diameter = np.max(tmp[:, 2])

        if input_data().is_3d:
            crop_size = int(diameter * 1.2) // 4 * 4
        else:
            crop_size = int(diameter * 2) // 4 * 4

        apix = round(all_images().apix, 4)
        ui.update_numeric("dn_apix", value=apix, max=apix * 2)
        ui.update_numeric("dn_pre_rotation", value=round(rotation, 1))
        ui.update_numeric(
            "dn_shift_y",
            value=round(shift_y, 1),
            min=-crop_size * apix // 2,
            max=crop_size * apix // 2,
        )
        ui.update_numeric(
            "dn_vertical_crop_size",
            value=max(32, crop_size),
            min=min(32, int(diameter) // 2 * 2),
            max=ny // 2 * 2,
        )
        ui.update_numeric("dn_horizontal_crop_size", value=nx, min=32, max=nx // 2 * 2)

    @reactive.effect
    @reactive.event(input.dn_reset_transform)
    def _reset_transform():
        req(len(selected_images_thresholded()))
        images = selected_images_thresholded()
        ny = int(np.max([img.shape[0] for img in images]))
        nx = int(np.max([img.shape[1] for img in images]))
        ui.update_numeric("dn_pre_rotation", value=0.0)
        ui.update_numeric("dn_shift_y", value=0.0)
        ui.update_numeric("dn_vertical_crop_size", value=ny // 2 * 2)
        ui.update_numeric("dn_horizontal_crop_size", value=nx // 2 * 2)

    @reactive.effect
    @reactive.event(
        selected_images_thresholded, input.dn_pre_rotation, input.dn_shift_y
    )
    def _transform_selected_images():
        req(len(selected_images_thresholded()))
        if input.dn_pre_rotation() != 0 or input.dn_shift_y() != 0:
            rotated = []
            for img in selected_images_thresholded():
                rotated.append(
                    helicon.transform_image(
                        image=img,
                        rotation=input.dn_pre_rotation(),
                        post_translation=(input.dn_shift_y() / input.dn_apix(), 0),
                    )
                )
        else:
            rotated = selected_images_original()
        selected_images_thresholded_rotated_shifted.set(rotated)

    @reactive.effect
    @reactive.event(
        selected_images_thresholded_rotated_shifted,
        input.dn_vertical_crop_size,
        input.dn_horizontal_crop_size,
    )
    def _crop_selected_images():
        req(len(selected_images_thresholded_rotated_shifted()))
        crop_ny = int(input.dn_vertical_crop_size())
        crop_nx = int(input.dn_horizontal_crop_size())
        cropped = []
        for img in selected_images_thresholded_rotated_shifted():
            ny, nx = img.shape
            if crop_ny < ny or crop_nx < nx:
                cropped.append(
                    helicon.crop_center(img, shape=(min(ny, crop_ny), min(nx, crop_nx)))
                )
            else:
                cropped.append(img)
        selected_images_thresholded_rotated_shifted_cropped.set(cropped)

    # ══════════════════════════════════════════════════════════════════
    # Reactive effects: multi-image stitching
    # ══════════════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(selected_images_rotated_shifted)
    def _update_transformed_images_displayed():
        req(len(selected_images_rotated_shifted()))
        image_work = _combine_images_for_display(selected_images_rotated_shifted())
        transformed_images_displayed.set([image_work])
        transformed_images_labels.set(["Selected images:"])
        transformed_images_links.set([""])

    @reactive.effect
    @reactive.event(input.dn_perform_stitching)
    def _update_stitched_image_displayed():
        req(len(selected_images_rotated_shifted()))
        x_offsets = transformed_images_x_offsets()
        x_positions = _image_stitching_x_positions(
            selected_images_rotated_shifted(), x_offsets
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with open(temp_dir + "/TileConfiguration.txt", "w") as tc:
                tc.write("dim = 2\n\n")
                for i, img in enumerate(selected_images_rotated_shifted()):
                    tmp = np.uint8(
                        (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                    )
                    tmp_imf = Image.fromarray(tmp, "L")
                    tmp_imf.save(temp_dir + "/" + str(i) + ".png")
                    tc.write(str(i) + ".png; ; (" + str(x_positions[i]) + ", 0.0)\n")
            result = denovo3d_pipeline.itk_stitch(temp_dir)

        result = result.astype(np.float32)
        result = (result - result.mean()) / result.std()
        result = result / result.max()
        stitched_image_displayed.set([result])
        stitched_image_labels.set(["Stitched image:"])
        stitched_image_links.set([""])

    # ══════════════════════════════════════════════════════════════════
    # Reactive effects: reconstruction
    # ══════════════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(
        input.dn_twist_min,
        input.dn_twist_max,
        input.dn_rise_min,
        input.dn_rise_max,
        input.dn_select_image,
    )
    def _update_run_button_label():
        if (
            input.dn_twist_min() != input.dn_twist_max()
            or input.dn_rise_min() != input.dn_rise_max()
        ):
            run_button_text.set("Search Parameters")
        else:
            run_button_text.set("Reconstruct 3D map")

    @reactive.effect
    @reactive.event(input.dn_run_denovo3D)
    def _run_denovo3D_reconstruction():
        data = selected_images_thresholded_rotated_shifted_cropped()
        req(len(data) > 0)

        data = data[0]
        ny, nx = data.shape
        binning_factor = max(1, getattr(input, "dn_binning", lambda: 1)())
        apix_binned = input.dn_apix() * binning_factor

        imageFile = selected_images_title().strip(":")
        imageIndex = selected_images_labels()[0]

        _log_dir = pathlib.Path.home() / ".cache" / "helicon"
        _log_dir.mkdir(parents=True, exist_ok=True)
        log = helicon.getLogger(
            logfile=str(_log_dir / "helicon.denovo3D.log"), verbose=1
        )

        # Build twist/rise parameter grid
        if (
            input.dn_twisting_handedness()
            == "Left-handed twisting (force negative twist)"
            and input.dn_twist_max() == input.dn_twist_min()
        ):
            twists = [np.negative(np.abs(input.dn_twist_max()))]
        elif (
            input.dn_twisting_handedness()
            == "Right-handed twisting (force positive twist)"
            and input.dn_twist_max() == input.dn_twist_min()
        ):
            twists = [np.abs(input.dn_twist_max())]
        else:
            if input.dn_twist_min() < input.dn_twist_max():
                twists = np.arange(
                    input.dn_twist_min(),
                    input.dn_twist_max() + input.dn_twist_step() / 2,
                    input.dn_twist_step(),
                )
            else:
                twists = [input.dn_twist_min()]

        if input.dn_rise_min() < input.dn_rise_max():
            rises = np.arange(
                input.dn_rise_min(),
                input.dn_rise_max() + input.dn_rise_step() / 2,
                input.dn_rise_step(),
            )
        else:
            rises = [input.dn_rise_min()]

        tr_pairs = list(itertools.product(twists, rises))
        n_pairs = len(tr_pairs)
        return_3d = n_pairs == 1
        n_cpu = input.dn_cpu()
        n_threads_per_job = max(1, n_cpu // max(1, n_pairs))

        if input.dn_target_apix2d() > apix_binned:
            target_apix2d_overwrite = input.dn_target_apix2d()
        else:
            target_apix2d_overwrite = -1
        if input.dn_target_apix3d() > apix_binned:
            target_apix3d_overwrite = input.dn_target_apix3d()
        else:
            target_apix3d_overwrite = -1

        tasks = []
        for ti, t in enumerate(tr_pairs):
            twist, rise = t
            twist = np.round(helicon.set_to_periodic_range(twist, min=-180, max=180), 6)
            csym = input.dn_csym()
            apix = apix_binned
            tilt_range_val = 0
            tilt = 0
            tilt_min = 0
            tilt_max = 0
            psi = 0
            psi_range_val = 0
            dy = 0
            dy_range_val = 0
            tube_length = nx * apix
            tube_diameter = ny * apix
            reconstruct_length = input.dn_reconstruct_length_rise() * rise

            algorithm = dict(
                model=input.dn_lr_algorithm(), l1_ratio=input.dn_lr_l1_ratio()
            )
            if input.dn_lr_alpha() >= 0:
                algorithm["alpha"] = input.dn_lr_alpha()

            if abs(twist) < 0.01:
                log.warning(f"WARNING: twist={round(twist, 3)} ignored (too small)")
                continue
            if abs(rise) < 0.01:
                log.warning(f"WARNING: rise={round(rise, 3)} ignored (too small)")
                continue
            if abs(rise) >= tube_length / 2:
                log.warning(f"WARNING: rise={round(rise, 3)} ignored (too large)")
                continue

            tasks.append(
                (
                    ti,
                    len(tr_pairs),
                    data,
                    imageFile,
                    imageIndex,
                    twist,
                    rise,
                    (np.min(rises), np.max(rises)),
                    csym,
                    tilt,
                    (tilt_min, tilt_max),
                    psi,
                    psi_range_val,
                    dy,
                    dy_range_val,
                    apix,
                    "",
                    -1,
                    0,
                    0,
                    target_apix3d_overwrite,
                    target_apix2d_overwrite,
                    -1,
                    int(input.dn_positive_constraint()),
                    tube_length,
                    tube_diameter,
                    0.0,
                    reconstruct_length,
                    input.dn_sym_oversample(),
                    input.dn_interpolation(),
                    0,
                    return_3d,
                    input.dn_score_metric(),
                    algorithm,
                    2,
                    n_threads_per_job,
                )
            )

        if len(tasks) < 1:
            log.warning("Nothing to do. I will quit")
            return

        abort_flag[0] = False
        _reconstruction_task(tasks, n_cpu, abort_flag)

    @reactive.extended_task
    async def _reconstruction_task(tasks, cpu, abort_ref):
        _log_dir = pathlib.Path.home() / ".cache" / "helicon"
        _log_dir.mkdir(parents=True, exist_ok=True)
        log = helicon.getLogger(
            logfile=str(_log_dir / "helicon.denovo3D.log"), verbose=1
        )

        try:
            with ui.Progress(min=0, max=len(tasks)) as p:
                p.set(
                    message="Calculation in progress",
                    detail="This may take a while ...",
                )

                from time import time
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=cpu) as executor:
                    future_tasks = [
                        executor.submit(denovo3d_pipeline.process_one_task, *task)
                        for task in tasks
                    ]
                    t0 = time()
                    results = []
                    n_discarded = 0
                    update_interval = max(1, len(tasks) // 20)

                    for completed_task in as_completed(future_tasks):
                        await asyncio.sleep(0)
                        if abort_ref[0] is True:
                            log.warning("User aborted the denovo3D run early.")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

                        try:
                            result = completed_task.result()
                        except Exception:
                            log.error(
                                "Task raised an exception:\n%s", traceback.format_exc()
                            )
                            n_discarded += 1
                            continue
                        if result is None:
                            n_discarded += 1
                            continue

                        results.append(result)
                        t1 = time()
                        remaining = (
                            (len(tasks) - len(results) - n_discarded)
                            / max(len(results), 1)
                            * (t1 - t0)
                        )
                        p.set(
                            len(results) + n_discarded,
                            message=f"Completed {len(results) + n_discarded}/{len(tasks)}",
                            detail=f"{helicon.timedelta2string(remaining)} remaining",
                        )

                        if len(results) % update_interval == 0:
                            results.sort(key=lambda x: x[0], reverse=True)
                            reconstruction_results.set(list(results))

                    t_final = time()
                    log.info("reconstruction time: %s", t_final - t0)

            if n_discarded:
                log.info(
                    f"{n_discarded}/{len(tasks)} results are None and thus discarded"
                )
            if results:
                results.sort(key=lambda x: x[0], reverse=True)
            reconstruction_results.set(results)
        except Exception:
            log.error("Reconstruction task failed:\n%s", traceback.format_exc())

    @reactive.effect
    @reactive.event(input.dn_stop_denovo3D)
    def _on_stop_denovo3D():
        abort_flag[0] = True

    # ══════════════════════════════════════════════════════════════════
    # Reactive effects: display results
    # ══════════════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(reconstruction_results)
    def _display_denovo3D_projections():
        reconstructed_projection_labels.set([])
        reconstructed_projection_images.set([])
        req(len(reconstruction_results()))

        top_n = input.dn_top_n_results()
        if top_n <= 0:
            top_n = len(reconstruction_results())

        labels = []
        images = []
        for ri, result in enumerate(reconstruction_results()[:top_n]):
            (
                score,
                (rec3d_x_proj, _rec3d_y_proj, rec3d_z_sections, rec3d, *_rest1),
                (
                    _data,
                    _imageFile,
                    _imageIndex,
                    _apix3d,
                    _apix2d,
                    twist,
                    rise,
                    _csym,
                    _tilt,
                    _psi,
                    _dy,
                ),
            ) = result

            query_image = selected_images_thresholded_rotated_shifted_cropped()[0]
            query_image_padded = helicon.pad_to_size(
                query_image, shape=rec3d_x_proj.shape
            )

            pitch_val = int(round(rise * 360 / abs(twist))) if abs(twist) > 0.01 else 0
            label_x = f"{ri+1}: X|score={score:.4f}|pitch={pitch_val:,}A|twist={round(twist, 3)}deg|rise={round(rise, 6)}A"
            labels += [
                f"Input image: {selected_images_labels()[0]}",
                label_x,
                f"{ri+1}: Z",
            ]

            rec3d_z_sections_padded = helicon.pad_to_size(
                rec3d_z_sections, shape=rec3d_x_proj.shape
            )
            images += [query_image_padded, rec3d_x_proj, rec3d_z_sections_padded]

        reconstructed_projection_labels.set(labels)
        reconstructed_projection_images.set(images)

    # ══════════════════════════════════════════════════════════════════
    # Reactive effects: misc
    # ══════════════════════════════════════════════════════════════════

    @reactive.effect
    @reactive.event(input.dn_clear_cache)
    def _clear_joblib_cache():
        from joblib import Memory

        cache_dir_path = helicon.cache_dir / "denovo3D"
        if cache_dir_path.exists():
            mem = Memory(location=str(cache_dir_path), verbose=0)
            mem.clear()
            logger.info(f"Cleared joblib cache at {cache_dir_path}")
