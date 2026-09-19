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
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import helicon
from shiny import reactive, render, req, ui, module
from shiny.types import SilentException

from ..lib.shared_state import ProjectState

from ..lib import denovo3d_joint, denovo3d_pipeline, denovo3d_register
from ..lib import helix_transform
from ..lib.helix_transform import (
    estimate_helix_rotation_center_diameter as _estimate_helix_rotation_center_diameter,
    refine_helix_rotation_center as _refine_helix_rotation_center,
)
from ..lib.helical_projection_utils import (
    _combine_images_for_display,
    _image_stitching_x_positions,
)

logger = logging.getLogger(__name__)

# ── Multi-image workflows ───────────────────────────────────────────────
# Selecting several images offers two mutually exclusive routes, and they use
# separate transform chains, which is why only one set of controls is shown at
# a time. Stitching is a pre-step that turns N images into 1 (per-image
# rotation/shift, then compositing, and the result then feeds the normal
# single-image path). The joint search keeps the N images and applies one
# common threshold/rotation/crop to all of them, solving each and combining the
# score curves.
MODE_JOINT = "Joint parameter search"
# The projection-matching route, kept beside the curve-averaging one rather
# than replacing it. The two agree on every set measured so far -- 1.20, 1.25
# and 1.20 on two ten-class sets and all 33 of EMPIAR-10940 -- so there is no
# evidence for promoting this one over the method that is already validated;
# what it adds is a roughly doubled margin on the full set, placements that
# settle, and a composite you can look at. Leaving both in place means the
# agreement stays checkable on the user's own data instead of being asserted.
MODE_PROJMATCH = "Joint search by projection matching"
MODE_STITCH = "Stitch manually"
MODE_AUTOSTITCH = "Stitch automatically"


BOOKMARK_DEFAULTS = {
    "input_mode_images": ("dn_input_mode_images", "url"),
    "url_images": ("dn_url_images", ""),
    "show_emdb": ("dn_show_emdb_input_mode", False),
    "is_3d": ("dn_is_3d", False),
    "ignore_blank": ("dn_ignore_blank", True),
    "plot_scores": ("dn_plot_scores", True),
    "show_download": ("dn_show_download_buttons", False),
    "match_input_box": ("dn_match_input_box", True),
    "display_size": ("dn_selected_image_display_size", 128),
    "rec_length": ("dn_reconstruct_length_rise", 3),
    "target_apix2d": ("dn_target_apix2d", 5),
    "target_apix3d": ("dn_target_apix3d", 5),
    "sym_oversample": ("dn_sym_oversample", -1),
    "lr_alpha": ("dn_lr_alpha", -1),
    "lr_l1_ratio": ("dn_lr_l1_ratio", 0.5),
    "top_n": ("dn_top_n_results", 10),
    "lr_algorithm": ("dn_lr_algorithm", "elasticnet"),
    "rec_algorithm": ("dn_rec_algorithm", "elasticnet"),
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


def _display_key(param_tuple):
    """Identify one solved (image, twist, rise) combination.

    ``param_tuple`` is the third element of a solver result, laid out as
    ``(data, imageFile, imageIndex, apix3d, apix2d, twist, rise, ...)``. The
    twist and rise are rounded because they travel through the task tuple and
    back, and only exact equality would otherwise match.
    """
    return (
        round(float(param_tuple[5]), 6),
        round(float(param_tuple[6]), 6),
        param_tuple[2],
    )


def _display_redraw_plan(tasks, results, ranked, top_n, display_model):
    """Work out which tasks to re-solve so the displayed pictures come from
    ``display_model``, and where each answer belongs.

    Only the pairs that will actually be shown are re-solved -- the top ``top_n``
    ranked twist/rise pairs, across every image -- because a search may have
    scored hundreds of pairs and the rest are never drawn.

    Parameters
    ----------
    tasks : list of tuple
        The positional argument tuples that were handed to
        ``denovo3d_pipeline.process_one_task``. Index 4 is imageIndex, 5 twist,
        6 rise, and -3 the algorithm dict.
    results, ranked : list
        The raw results and their ranking, as ``(score, return_data, params)``.
    top_n : int
        How many ranked pairs are displayed; ``<= 0`` means all of them.
    display_model : str
        Solver to draw with.

    Returns
    -------
    redo : list of tuple
        Task tuples with the algorithm's model replaced, for the shown pairs.
    slot : dict
        ``_display_key`` -> index into ``results``, so a finished re-solve can
        be put back in the right place.
    """
    if top_n <= 0:
        top_n = len(ranked)
    wanted = {_display_key(r[2])[:2] for r in ranked[:top_n]}
    slot = {_display_key(r[2]): i for i, r in enumerate(results)}
    redo = []
    for task in tasks:
        key = (round(float(task[5]), 6), round(float(task[6]), 6), task[4])
        if key[:2] in wanted and key in slot:
            t = list(task)
            t[-3] = dict(t[-3], model=display_model)
            redo.append(tuple(t))
    return redo, slot


def _denovo3d_logger():
    """The tab's log, under whichever cache root helicon resolved.

    Not ``~/.cache/helicon`` spelled out, which is only one of the four places
    ``setup_cache_dir`` may choose: it also honours HELION_CACHE_DIR and
    prefers ``/fast-scratch`` when that exists. Hardcoding the home path sent
    the log somewhere other than the cache whenever either applied, which is
    the one thing a user looking for it would not expect.
    """
    log_dir = helicon.cache_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return helicon.getLogger(logfile=str(log_dir / "helicon.denovo3D.log"), verbose=1)


def _rank(results, n_images, log=None):
    """Order solver results best-first, jointly when several images were solved.

    With one image this is just a sort by score. With several, the scores are
    combined across images per twist/rise pair, because a single class average
    picks the twist unreliably -- only 5/10 good EMPIAR-10940 classes peaked at
    the right value on their own, whereas the combined curve was right with a
    ~50x larger margin. See ``denovo3d_joint`` for why the combination z-scores
    with a shrinkage floor rather than averaging raw scores.
    """
    if not results:
        return []
    if n_images < 2:
        return sorted(results, key=lambda x: x[0], reverse=True)
    joint, weights = denovo3d_joint.combine_results(results)
    if log is not None and weights:
        log.info(
            "joint weights: "
            + ", ".join(f"{k}={v:.2f}" for k, v in sorted(weights.items()))
        )
    return joint


def _prepare_download_map(
    result,
    *,
    match_input_box,
    input_image_shape,
    input_apix,
    compact_apix,
    cpu,
):
    """Return the reconstructed map and voxel size for download.

    RELION expects an initial reference to use the particle images' cubic box
    and pixel size.  The compact map remains available to avoid the memory and
    download-size cost of padding a long helical reconstruction to a cube.
    """
    _score, return_data, params = result
    rec3d = return_data[3]
    compact_map = return_data[8]

    if not match_input_box:
        if compact_map is None:
            raise ValueError("The compact reconstructed map is unavailable")
        return np.asarray(compact_map, dtype=np.float32), float(compact_apix)

    if rec3d is None:
        raise ValueError("The reconstructed map is unavailable")

    apix3d, twist, rise, csym = params[3], params[5], params[6], params[7]
    box_size = max(int(v) for v in input_image_shape[-2:])
    rec3d_map = helicon.apply_helical_symmetry(
        data=rec3d[0],
        apix=apix3d,
        twist_degree=twist,
        rise_angstrom=rise,
        csym=csym,
        fraction=1.0,
        new_size=(box_size, box_size, box_size),
        new_apix=input_apix,
        cpu=cpu,
    )
    return np.asarray(rec3d_map, dtype=np.float32), float(input_apix)


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
                    # Filtering lives here rather than beside the images: it is
                    # rarely touched, and in the main panel it had to be kept in
                    # step with the gallery's width, which took a resize
                    # observer to do at all reliably.
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
                        ),
                        id="dn_filtering_options",
                        open=False,
                        width="100%",
                    ),
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
                        ui.tooltip(
                            ui.input_checkbox(
                                "dn_match_input_box",
                                "Match input box and pixel size for downloaded map",
                                value=True,
                            ),
                            "Generate a cubic map that can be used directly as a RELION initial reference. Disable to download the smaller, compact map.",
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
                            "Search algorithm",
                            ["gauss", "elasticnet", "lasso", "ridge", "lsq"],
                            selected="gauss",
                            inline=True,
                        ),
                        (
                            "Used while scanning twist/rise, where only the score"
                            " matters. gauss solves the same regularized problem on a"
                            " basis of Gaussians rather than voxels; a Gaussian projects"
                            " to a Gaussian exactly, so its projections are built"
                            " analytically instead of by resampling a volume, which"
                            " makes it roughly ten times faster per twist/rise. It is"
                            " convex, so the answer does not depend on a starting guess."
                            " On the 32 good EMPIAR-10940 classes both solvers put the"
                            " joint peak on the true twist, but elasticnet's peak stands"
                            " out more clearly and is the more reliable of the two on a"
                            " single class average, so prefer gauss when the scan is"
                            " large and elasticnet when you have few images."
                        ),
                    ),
                    ui.tooltip(
                        ui.input_radio_buttons(
                            "dn_rec_algorithm",
                            "Reconstruction algorithm",
                            ["gauss", "elasticnet", "lasso", "ridge", "lsq"],
                            selected="elasticnet",
                            inline=True,
                        ),
                        (
                            "Used when a single twist/rise pair is requested, i.e. when"
                            " the 3D map is what you want, and to draw the projections"
                            " shown after a search, since those are reconstructions too."
                            " Prefer elasticnet here."
                            " gauss is built from non-negative Gaussians, which place"
                            " the strands correctly but cannot carve the hollow core of"
                            " a filament, so its map fills the dark lane between the"
                            " strands. Search with gauss, reconstruct with elasticnet."
                        ),
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
        helix_transform.card_switching_script(
            [
                {"input": "dn_active_image", "cards": ".dn-pi-card", "key": "pi"},
                {
                    "input": "dn_stitch_active_image",
                    "cards": ".dn-ms-card",
                    "key": "ms",
                },
            ]
        ),
        ui.output_ui("dn_multi_mode_ui"),
        ui.div(
            ui.div(
                ui.output_ui("dn_generate_image_gallery_multiple"),
                ui.output_ui("dn_stitch_button_ui"),
                style="display: flex; flex-direction: column;"
                " align-items: flex-start; gap: 6px; margin-bottom: 0",
            ),
            ui.output_ui("dn_generate_image_transformation_multiple"),
            # These two are the combined preview and the stitched result. Both
            # are as wide as all the images laid end to end, so they overflow
            # the window however the row is arranged; let them scroll on their
            # own rather than pushing the page sideways.
            ui.div(
                ui.output_ui("dn_image_stitching_transformed"),
                style="max-width: 100%; overflow-x: auto;",
            ),
            ui.div(
                ui.output_ui("dn_display_stitched_image"),
                style="max-width: 100%; overflow-x: auto;",
            ),
            # Four things abreast -- gallery, controls, combined preview and the
            # stitched result -- run off the right of the screen, and the
            # stitched image is both the widest and the last. Wrapping puts it
            # on the next row instead of out of view.
            style="display: flex; flex-direction: row; flex-wrap: wrap;"
            " align-items: flex-start; gap: 10px; margin-bottom: 0",
        ),
        ui.div(
            ui.div(
                ui.output_ui("dn_generate_image_gallery_single"),
                style="display: flex; flex-direction: column; align-items: flex-start;"
                " width: fit-content; gap: 10px; margin-bottom: 0",
            ),
            # Transform controls and their buttons share a row: the buttons
            # sit to the right of whichever card is showing rather than under
            # it, which keeps them beside the sliders they act on instead of
            # pushing the next control further down an already tall card. The
            # two card outputs are mutually exclusive by mode, so nesting them
            # in a column here means the buttons land beside either one.
            ui.div(
                ui.div(
                    ui.output_ui("dn_joint_per_image_transform_ui"),
                    ui.output_ui("dn_generate_image_transformation_single"),
                    style="display: flex; flex-direction: column;"
                    " align-items: flex-start; gap: 6px;",
                ),
                ui.output_ui("dn_transform_buttons_ui"),
                style="display: flex; flex-direction: row; align-items: flex-start;"
                " gap: 10px; margin-bottom: 0",
            ),
            # Its own column to the right of the transform card, so the two
            # sets of controls read as separate things rather than one stack.
            ui.div(
                ui.output_ui("dn_autostitch_ui"),
                style="display: flex; flex-direction: column; align-items: flex-start;"
                " max-width: 380px; gap: 6px; margin-bottom: 0",
            ),
            # Wrap rather than overflow: a stitched image can be many times
            # wider than a class average, and without this the transform card
            # is pushed off to the right of it. Narrow images still sit side by
            # side; a wide one takes the row and the controls move below.
            style="display: flex; flex-direction: row; flex-wrap: wrap;"
            " align-items: flex-start; gap: 10px; margin-bottom: 0",
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
            # Server-rendered like its two neighbours, not static. Built into
            # the page directly it shipped with the initial HTML while the
            # twist and rise cards were still waiting on the websocket, so on
            # every load the Csym card appeared on its own for a moment before
            # the row it belongs to filled in around it.
            ui.output_ui("dn_csym_card"),
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

    # Per-image (rotation_deg, shift_y_pixels) from auto-transform, one entry
    # per selected image. Empty means "nothing image-specific", which is the
    # single-image case where the shared Rotation/Vertical shift boxes hold the
    # absolute values instead.
    per_image_transforms = reactive.value([])

    # Mirror of the dynamically created mode radio; see _sync_multi_mode.
    multi_mode_rv = reactive.value(MODE_JOINT)

    # Diagnostics from the last automatic stitch, shown to the user.
    autostitch_report = reactive.value({})

    # The automatic stitch's per-image transforms, expressed in the manual
    # route's own terms so that switching to manual picks up where automatic
    # left off instead of starting from nothing. Possible only because the
    # manual card carries flips: without them a transferred state would still
    # not reproduce the composite.
    autostitch_transforms = reactive.value([])

    # (twist, rise) -> the composite of every image at the placement that twist
    # implies, produced by the projection-matching route and shown beside the
    # per-image reconstructions. Empty on every other route.
    projmatch_composites = reactive.value({})

    @reactive.calc
    def transformed_gallery_title():
        """Heading for the main gallery, which shows whatever the pipeline holds.

        Separate from ``selected_images_title`` because that one is also
        recorded as the imageFile in each reconstruction task, so it has to keep
        naming the source rather than what is currently on screen.
        """
        if len(stitched_image_displayed()):
            return "Stitched, transformed image:"
        return selected_images_title()

    @reactive.calc
    def transformed_image_labels():
        """One label per transformed image, so a multi-image set is legible."""
        n = len(selected_images_thresholded_rotated_shifted_cropped())
        labels = list(selected_images_labels())
        if n <= 1 or len(labels) != n:
            return [""] * n
        return [str(l) for l in labels]

    reconstruction_results = reactive.value([])
    # The unranked per-(image, twist, rise) results. reconstruction_results
    # keeps one entry per twist/rise pair -- which is what the scores plot and
    # the download path expect -- so the per-image reconstructions behind each
    # pair are kept here for the results gallery to expand.
    reconstruction_results_raw = reactive.value([])
    reconstructed_projection_images = reactive.value([])
    reconstructed_projection_labels = reactive.value([])

    run_button_text = reactive.value("Search Parameters")
    abort_flag = [False]  # mutable container shared with the task (not reactive)

    # ── Per-image transform parameters ───────────────────────────────
    # In joint mode every image carries its own full set of transform
    # settings. All the per-image inputs are created up front, one card per
    # image, and only the card for the clicked image is shown -- switching is
    # pure CSS. Rendering them all means every input always exists, which
    # matters because reading an input that was never created raises
    # SilentException and can silently kill an effect.

    def _pi_id(kind, i):
        return f"dn_pi_{kind}_{i}"

    # The single-image control that each per-image setting corresponds to.
    _PI_SHARED_ID = {
        "transpose": "dn_img_transpose",
        "flip": "dn_img_flip",
        "negate": "dn_img_negate",
        "rot": "dn_pre_rotation",
        "threshold": "dn_threshold",
        "apix": "dn_apix",
        "dy": "dn_shift_y",
        "vcrop": "dn_vertical_crop_size",
        "hcrop": "dn_horizontal_crop_size",
    }

    def _param(kind, i, default=0.0):
        """Effective value of one transform setting for image ``i``.

        Joint mode reads that image's own control; otherwise the shared one.
        """
        if _per_image_transform_ui_active():
            return _input_or(_pi_id(kind, i), default)
        return _input_or(_PI_SHARED_ID[kind], default)

    def _pi_defaults(i):
        """Starting values for image ``i``'s card."""
        per = per_image_transforms()
        rot, shift_px = per[i] if i < len(per) else (0.0, 0.0)
        return dict(
            transpose=img_transpose_rv(),
            flip=img_flip_rv(),
            negate=img_negate_rv(),
            rot=round(float(rot), 2),
            threshold=threshold_rv(),
            apix=apix_rv(),
            dy=round(float(shift_px) * (apix_rv() or 1.0), 2),
            vcrop=vertical_crop_size_rv(),
            hcrop=horizontal_crop_size_rv(),
        )

    def _transformation_card_per_image(i, label, ny, nx):
        """The full transform card for one image, with per-image input ids."""
        d = _pi_defaults(i)
        slider = input.dn_input_ui_type() == "Slider"
        num = ui.input_slider if slider else ui.input_numeric
        extra = {} if slider else {"update_on": "blur"}
        return ui.div(
            ui.card(
                ui.div(
                    f"Image {label}",
                    style="font-weight: bold; margin-bottom: 4px;",
                ),
                ui.layout_columns(
                    ui.input_checkbox(
                        _pi_id("transpose", i), "Transpose", d["transpose"]
                    ),
                    ui.input_checkbox(_pi_id("flip", i), "Flip", d["flip"]),
                    ui.input_checkbox(
                        _pi_id("negate", i), "Invert contrast", d["negate"]
                    ),
                    num(
                        _pi_id("rot", i),
                        "Rotation (deg)",
                        min=-90,
                        max=90,
                        value=d["rot"],
                        step=0.1,
                        **extra,
                    ),
                    num(
                        _pi_id("threshold", i),
                        "Threshold",
                        min=round(d["threshold"] - 1, 3),
                        max=round(d["threshold"] + 1, 3),
                        value=d["threshold"],
                        step=0.001,
                        **extra,
                    ),
                    num(
                        _pi_id("apix", i),
                        "Pixel size (A)",
                        min=0.0,
                        max=10.0,
                        value=d["apix"],
                        step=0.001,
                        **extra,
                    ),
                    num(
                        _pi_id("dy", i),
                        "Vertical shift (A)",
                        min=-200,
                        max=200,
                        value=d["dy"],
                        step=0.1,
                        **extra,
                    ),
                    num(
                        _pi_id("vcrop", i),
                        "Vertical crop (pixel)",
                        min=32,
                        max=max(32, ny),
                        value=min(d["vcrop"], max(32, ny)),
                        step=2,
                        **extra,
                    ),
                    num(
                        _pi_id("hcrop", i),
                        "Horizontal crop (pixel)",
                        min=32,
                        max=max(32, nx),
                        value=min(d["hcrop"], max(32, nx)),
                        step=2,
                        **extra,
                    ),
                    col_widths=4,
                ),
            ),
            class_="dn-pi-card",
            **{"data-pi": str(i)},
            # A minimum width so the three slider columns and their labels are
            # not squeezed onto several lines, and the buttons below fit on one.
            # Only the first card starts visible; the gallery click swaps them.
            style="min-width: 430px;" + ("" if i == 0 else " display: none;"),
        )

    # ── Helper: UI for single-image transformation controls ──────────

    def _transformation_ui_single(shared_only=False):
        """Build the transformation control card.

        ``shared_only`` drops the Rotation and Vertical shift controls, for the
        multi-image joint route where those are per-image and rendered
        alongside each image instead. Everything left -- transpose, flip,
        contrast, threshold, pixel size, crops -- genuinely applies to the
        whole set.
        """
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
                    *(
                        []
                        if shared_only
                        else [
                            ui.input_slider(
                                "dn_pre_rotation",
                                "Rotation (deg)",
                                min=-20,
                                max=20,
                                value=pre_rotation_rv(),
                                step=0.1,
                            )
                        ]
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
                    *(
                        []
                        if shared_only
                        else [
                            ui.input_slider(
                                "dn_shift_y",
                                "Vertical shift (A)",
                                min=-100,
                                max=100,
                                value=shift_y_rv(),
                                step=0.1,
                            )
                        ]
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
                    *(
                        []
                        if shared_only
                        else [
                            ui.input_numeric(
                                "dn_pre_rotation",
                                "Rotation (deg)",
                                min=-20,
                                max=20,
                                value=pre_rotation_rv(),
                                step=0.1,
                                update_on="blur",
                            )
                        ]
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
                    *(
                        []
                        if shared_only
                        else [
                            ui.input_numeric(
                                "dn_shift_y",
                                "Vertical shift (A)",
                                min=-100,
                                max=100,
                                value=shift_y_rv(),
                                step=0.1,
                                update_on="blur",
                            )
                        ]
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

    def _transformation_ui_group(
        prefix, shift_scale=100, index=0, label="", visible=True, initial=None
    ):
        """Stitching controls for one image: rotation, the two shifts, and flip.

        Only what stitching needs -- threshold, pixel size and cropping belong
        to the reconstruction and would be noise here. One card is shown at a
        time, chosen by clicking the image above, rather than stacking a set per
        image down the page.

        Every card is rendered and hidden with CSS rather than created on
        demand, so each image keeps whatever was typed into it and no effect
        can read an input that does not exist.

        ``initial`` seeds the controls from a previous automatic stitch, so
        switching to manual continues from that result rather than from zero.
        """
        d = initial or {}
        return ui.div(
            ui.card(
                ui.div(
                    f"Image {label}" if label else "",
                    style="font-weight: bold; margin-bottom: 4px;",
                ),
                ui.layout_columns(
                    ui.input_slider(
                        prefix + "_pre_rotation",
                        "Rotation (deg)",
                        min=-45,
                        max=45,
                        value=round(float(d.get("rotation", 0.0)), 2),
                        step=0.1,
                    ),
                    ui.input_slider(
                        prefix + "_shift_x",
                        "Horizontal shift (pixel)",
                        min=-shift_scale,
                        max=shift_scale,
                        value=int(round(float(d.get("shift_x", 0.0)))),
                        step=1,
                    ),
                    ui.input_slider(
                        prefix + "_shift_y",
                        "Vertical shift (pixel)",
                        min=-100,
                        max=100,
                        value=int(round(float(d.get("shift_y", 0.0)))),
                        step=1,
                    ),
                    col_widths=4,
                ),
                ui.layout_columns(
                    ui.input_checkbox(
                        prefix + "_flip_x",
                        "Flip polarity (x)",
                        bool(d.get("flip_x", False)),
                    ),
                    ui.input_checkbox(
                        prefix + "_flip_y",
                        "Flip side (y)",
                        bool(d.get("flip_y", False)),
                    ),
                    col_widths=6,
                ),
                id=f"{prefix}_card",
            ),
            class_="dn-ms-card",
            **{"data-ms": str(index)},
            style="min-width: 430px;" + ("" if visible else " display: none;"),
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

    # ── Multi-image workflow mode ────────────────────────────────────

    @reactive.effect
    def _sync_multi_mode():
        """Mirror the mode radio into a reactive value that is always set.

        The radio is created by a render.ui, so `input.dn_multi_mode` does not
        exist until a multi-selection renders it. Naming a missing input in a
        reactive.event list is fatal -- reactive.event calls every dependency
        up front, so the SilentException from an unset input aborts the whole
        effect -- so everything triggers off this mirror instead. Reading the
        input registers the dependency before raising, so this re-runs as soon
        as the radio appears.
        """
        try:
            mode = input.dn_multi_mode()
        except SilentException:
            # Not created yet, or briefly gone during a re-render. Keep what we
            # have: defaulting here would fight whatever the user just picked.
            return
        if mode and multi_mode_rv() != mode:
            multi_mode_rv.set(mode)

    def _multi_mode():
        """Which multi-image route is active; joint search unless told otherwise."""
        return multi_mode_rv()

    def _stitching_active():
        """True when the manual stitch route owns the display."""
        return len(selected_images_original()) > 1 and _multi_mode() == MODE_STITCH

    def _autostitch_active():
        """True when the automatic stitch route owns the display."""
        return len(selected_images_original()) > 1 and _multi_mode() == MODE_AUTOSTITCH

    def _projmatch_active():
        """True when the projection-matching search route is chosen.

        It solves the same per-image tasks as the plain joint search -- so the
        per-image views are unchanged -- and then re-ranks the twists by fitting
        one volume to all the images at once, each at its own azimuth.
        """
        return len(selected_images_original()) > 1 and _multi_mode() == MODE_PROJMATCH

    @render.ui
    @reactive.event(selected_images_original)
    def dn_multi_mode_ui():
        """The route selector, rebuilt only when the selection changes.

        Deliberately not sensitive to the mode itself: reading multi_mode_rv
        here made every mode change re-render the radio, destroying and
        recreating the very input that had just been clicked. While it was
        being recreated the input read as unset, the sync below fell back to
        the default, and the selector snapped straight back to the joint
        search -- so no other mode could be chosen at all.
        """
        if len(selected_images_original()) < 2:
            return ui.div()
        return ui.div(
            ui.input_radio_buttons(
                "dn_multi_mode",
                "Multiple images selected — what to do with them:",
                # Labels carry their own tooltip. The four routes do quite
                # different things to the same selection and the names alone do
                # not say which, least of all what the two joint searches cost
                # relative to each other.
                choices={
                    MODE_JOINT: ui.tooltip(
                        ui.span(MODE_JOINT),
                        "Score every twist against each image separately, then"
                        " combine the score curves. Fast, and the method the"
                        " published measurements were made with.",
                    ),
                    MODE_PROJMATCH: ui.tooltip(
                        ui.span(MODE_PROJMATCH),
                        "Score every twist by fitting ONE reconstruction to all"
                        " the images at once, each placed at its own azimuth."
                        " Several times slower, and it also draws the images"
                        " composited at the positions each twist implies. On"
                        " the data measured so far it agrees with the plain"
                        " joint search.",
                    ),
                    MODE_AUTOSTITCH: ui.tooltip(
                        ui.span(MODE_AUTOSTITCH),
                        "Register the images to each other by correlation and"
                        " combine them into one longer image, which covers more"
                        " of the helical pitch than any single class does.",
                    ),
                    MODE_STITCH: ui.tooltip(
                        ui.span(MODE_STITCH),
                        "Place each image by hand -- rotation, shift and flip"
                        " per image -- then combine them. An automatic stitch,"
                        " or a joint search, can hand its layout over as the"
                        " starting point.",
                    ),
                },
                selected=_multi_mode(),
                inline=True,
            ),
            style="margin-bottom: 0",
        )

    @render.ui
    def dn_transform_buttons_ui():
        """Auto / Reset Transform, rendered from one place only.

        Both branches of the shared card and the per-image block used to carry
        their own copies, which collided as duplicate input ids whenever two of
        them were in the DOM at the same time.
        """
        if not len(selected_images_thresholded()):
            return ui.div()
        # Hidden on the manual stitch route until something has been stitched.
        # Until then each image is placed by its own card, and these two act on
        # the shared transform -- so before a composite exists there is nothing
        # for them to auto-transform or reset, and offering them only invites a
        # click that appears to do nothing. Once the stitch has collapsed the
        # selection into one image they apply to it, and come back.
        if _stitching_active() and not len(stitched_image_displayed()):
            return ui.div()
        # Stacked, not side by side: they now sit in a narrow column beside the
        # transform card, where two buttons in a row would either overflow it
        # or squeeze their labels onto two lines each.
        return ui.div(
            ui.input_action_button(
                "dn_auto_transform", label="Auto Transform", class_="btn-primary"
            ),
            ui.input_action_button(
                "dn_reset_transform", label="Reset Transform", class_="btn-primary"
            ),
            style="display: flex; flex-direction: column; gap: 6px;"
            " width: 160px; flex: 0 0 auto; margin-top: 4px;",
        )

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
        if not _stitching_active():
            return ui.div()
        return helicon.shiny.image_gallery(
            id=session.ns("dn_stitch_active_image"),
            label=selected_images_title,
            images=selected_images_rotated_shifted,
            image_labels=selected_images_labels,
            image_size=reactive.value(128),
            justification="left",
            # Clicking picks which image's stitching card is shown, the same way
            # the joint route works, instead of stacking a card per image.
            enable_selection=True,
            allow_multiple_selection=False,
            initial_selected_indices=reactive.value([0]),
            display_dashed_line=True,
        )

    @render.ui
    def dn_stitch_button_ui():
        """The manual Stitch button, rendered apart from the transform cards.

        It acts on the whole selection rather than on the one image the card is
        editing, so it belongs under the gallery it applies to.
        """
        if not _stitching_active():
            return ui.div()
        # No button for seeding the layout from a search: a projection-matching
        # search already places every image, and hands that layout over on its
        # own, exactly as the automatic stitcher does with its own.
        return ui.input_action_button(
            "dn_perform_stitching",
            label="Stitch Images",
            class_="btn-primary",
            style="max-width: 200px; margin-top: 6px;",
        )

    @render.ui
    @reactive.event(selected_images_original, multi_mode_rv, autostitch_transforms)
    def dn_generate_image_transformation_multiple():
        imgs = displayed_images()
        if not imgs or len(imgs) == 0:
            return ui.div()
        sel = input.dn_select_image()
        if sel is None or len(sel) == 0:
            return ui.div()
        if not (0 <= min(sel) and max(sel) < len(imgs)):
            return ui.div()
        if not _stitching_active():
            return ui.div()

        n_images_selected = len(selected_images_original())
        dim = len(selected_images_original()[0])
        width = int(np.shape(selected_images_original()[0])[1])
        # Wide enough for a transferred layout: undoing the end-to-end tiling
        # puts the last image at roughly -(n-1) * width.
        shift_scale = max(int(0.9 * dim) * n_images_selected, width * n_images_selected)
        transferred = autostitch_transforms()
        if len(transferred) != n_images_selected:
            transferred = [None] * n_images_selected
        container = ui.div(
            style="display: flex; flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 0"
        )

        for i, label in enumerate(selected_images_labels()):
            curr_counter = i
            container.append(
                _transformation_ui_group(
                    f"dn_t_ui_group_{curr_counter}",
                    shift_scale=shift_scale,
                    index=i,
                    label=str(label),
                    visible=(i == 0),
                    initial=transferred[i],
                )
            )

            id_rotation = f"dn_t_ui_group_{curr_counter}_pre_rotation"
            id_x_shift = f"dn_t_ui_group_{curr_counter}_shift_x"
            id_y_shift = f"dn_t_ui_group_{curr_counter}_shift_y"
            id_flip_x = f"dn_t_ui_group_{curr_counter}_flip_x"
            id_flip_y = f"dn_t_ui_group_{curr_counter}_flip_y"

            @reactive.effect
            @reactive.event(input.dn_select_image)
            def _update_multi_originals():
                selected_images_rotated_shifted.set(
                    [displayed_images()[j] for j in input.dn_select_image()]
                )
                transformed_images_x_offsets.set(np.zeros(len(input.dn_select_image())))

            def _make_transform_multi_img(_ii, _id_rot, _id_ys, _id_fx, _id_fy):
                @reactive.effect
                @reactive.event(
                    input[_id_rot], input[_id_ys], input[_id_fx], input[_id_fy]
                )
                def _fn():
                    req(len(selected_images_original()))
                    rotated = selected_images_rotated_shifted().copy()
                    work = selected_images_original()[_ii].copy()
                    # Flips first, as lossless array operations, so only the
                    # rotation and shift cost an interpolation.
                    if input[_id_fx]():
                        work = work[:, ::-1]
                    if input[_id_fy]():
                        work = work[::-1, :]
                    work = np.ascontiguousarray(work)
                    if input[_id_rot]() != 0 or input[_id_ys]() != 0:
                        work = helicon.transform_image(
                            image=work,
                            rotation=input[_id_rot](),
                            post_translation=(input[_id_ys](), 0),
                        )
                    rotated[_ii] = work
                    selected_images_rotated_shifted.set(rotated)

                return _fn

            _make_transform_multi_img(i, id_rotation, id_y_shift, id_flip_x, id_flip_y)

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
                    transformed_images_labels.set([""])
                    transformed_images_links.set([""])
                    transformed_images_x_offsets.set(curr_offsets)

                return _fn

            _make_update_multi_displayed(i, id_x_shift)

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
        if not _stitching_active():
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
        # Either stitch route produces a stitched image to show.
        if not (_stitching_active() or _autostitch_active()):
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
        multi = _per_image_transform_ui_active()
        return helicon.shiny.image_gallery(
            id=session.ns("dn_active_image"),
            label=transformed_gallery_title,
            images=selected_images_thresholded_rotated_shifted_cropped,
            image_labels=transformed_image_labels,
            image_size=input.dn_selected_image_display_size,
            justification="left",
            # Clicking picks which image's transform card is shown, so the
            # controls take one card's worth of space instead of N.
            enable_selection=multi,
            allow_multiple_selection=False,
            initial_selected_indices=reactive.value([0] if multi else []),
            display_dashed_line=True,
        )

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
        # Shown for multi-selection too: the same threshold/rotation/crop applies
        # to the whole set, and the joint parameter search needs them transformed.
        if _per_image_transform_ui_active():
            # Every setting is per-image there, so this card would duplicate
            # controls that no longer drive anything.
            return ui.div()
        return ui.div(
            _transformation_ui_single(),
            style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0",
        )

    def _per_image_transform_ui_active():
        """True when rotation/shift are per-image rather than shared.

        Both multi-image routes that auto-transform need this -- the automatic
        stitch registers the auto-transformed images, so those must have been
        transformed individually too. Manual stitching is excluded because it
        already has its own per-image sliders, and a single image has nothing
        to distinguish.
        """
        # Counted on initial_image -- what the pipeline is actually carrying --
        # rather than the selection, so that once a stitch has produced a single
        # image it belongs to the shared card however many were selected to make
        # it. Deliberately not selected_images_thresholded: _threshold_selected_
        # images calls this through _param, so reading its own output here is a
        # reactive cycle, and the whole tab silently stops rendering.
        return len(initial_image()) > 1 and _multi_mode() != MODE_STITCH

    @render.ui
    @reactive.event(
        selected_images_original,
        multi_mode_rv,
        autostitch_report,
        stitched_image_displayed,
    )
    def dn_autostitch_ui():
        """The automatic stitch route: one button and an honest verdict.

        The diagnostics are shown rather than hidden because whether a dataset
        can be stitched at all is a property of the data -- how much of the
        pitch its classes happen to cover -- not something the algorithm can
        decide for the user.
        """
        if not _autostitch_active():
            return ui.div()
        rep = autostitch_report()
        rows = []
        if rep:
            trust = rep.get("trustworthy")
            colour = "#1a7f37" if trust else "#b35900"
            verdict = (
                "Registration looks consistent."
                if trust
                else "Registration is NOT reliable - see below."
            )
            span_note = (
                f"Combined length {rep['span_gain']:.2f}x a single image"
                f" ({rep['span_px']:.0f} px)."
            )
            detail = (
                f"{rep['n_pairs']} of {rep['n_possible']} image pairs registered"
                f" ({rep['n_rejected']} rejected as inconsistent),"
                f" {rep['n_connected']}/{rep['n_images']} images placed."
            )
            checks = []
            if rep["redundancy"] <= 0:
                checks.append(
                    "no redundant pairs, so the consistency check cannot verify anything"
                )
            elif not rep["closure_meaningful"]:
                checks.append("consistency check unavailable")
            else:
                checks.append(
                    f"consistency {rep['closure']['dx']:.2f} px,"
                    f" {rep['closure']['psi']:.2f} deg"
                )
            if rep.get("flip_conflicts"):
                checks.append(f"{rep['flip_conflicts']} pairs disagree on polarity")
            if rep.get("unconnected"):
                checks.append(
                    f"images {rep['unconnected']} could not be placed and were left out"
                )
            rows = [
                ui.div(verdict, style=f"color: {colour}; font-weight: bold;"),
                ui.div(span_note),
                ui.div(detail, style="font-size: 90%;"),
                ui.div("; ".join(checks), style="font-size: 90%; color: #555;"),
            ]
        if len(stitched_image_displayed()):
            # Stitched already: the pipeline now holds one image, so running it
            # again would do nothing. Offer the way back instead -- undoing
            # restores the individual images and their per-image controls, which
            # is what anyone unhappy with the result actually needs.
            return ui.div(
                *rows,
                ui.input_action_button(
                    "dn_undo_stitch",
                    label="Undo stitch",
                    class_="btn-primary",
                    style="max-width: 200px; margin-top: 6px;",
                ),
                ui.div(
                    "Undoing brings back the individual images so you can adjust"
                    " them and stitch again.",
                    style="font-size: 85%; color: #777;",
                ),
                style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 0;",
            )
        return ui.div(
            ui.div(
                "Register the selected images to each other and combine them into"
                " one longer image. A longer image covers more of the helical"
                " pitch, which is what the twist search needs.",
                style="font-size: 90%; color: #555;",
            ),
            ui.input_action_button(
                "dn_auto_stitch",
                label="Auto Stitch",
                class_="btn-primary",
                style="max-width: 200px; margin-top: 6px;",
            ),
            *rows,
            style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 0;",
        )

    @render.ui
    @reactive.event(selected_images_original, per_image_transforms, multi_mode_rv)
    def dn_joint_per_image_transform_ui():
        """One full transform card per image, only the clicked one visible.

        Every setting is per-image because every class average differs: the
        per-image rotations of ten good EMPIAR-10940 classes span -20.1 to
        +10.2 deg, and contrast and centring vary just as much. All the cards
        are rendered together and hidden with CSS rather than rendered on
        demand, so each image's inputs keep their values while you click
        between images and no effect can read an input that does not exist.
        """
        if not _per_image_transform_ui_active():
            return ui.div()
        labels = list(selected_images_labels())
        imgs = selected_images_thresholded() or initial_image() or []
        if imgs:
            ny, nx = np.shape(imgs[0])
        else:
            ny, nx = 128, 128
        return ui.div(
            ui.div(
                "Per-image transform (click an image to edit it):",
                style="font-weight: bold; margin-bottom: 4px;",
            ),
            *[
                _transformation_card_per_image(i, label, int(ny), int(nx))
                for i, label in enumerate(labels)
            ],
            style="display: flex; flex-direction: column; gap: 4px; margin-bottom: 0;",
        )

    @render.ui
    def dn_csym_card():
        """Csym, rendered server-side so it arrives with twist and rise."""
        return ui.card(
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
        )

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
        imgs = all_images()
        req(imgs is not None)
        if isinstance(imgs.data, np.ndarray):
            if len(imgs.data.shape) < 3:
                input_image_shape = imgs.data.shape
            else:
                input_image_shape = imgs.data.shape[-2:]
        else:
            image_index = int(result[2][2]) - 1
            input_image_shape = imgs.data[image_index].shape

        compact_apix = _input_or("dn_apix", apix_rv()) * max(1, input.dn_binning())
        rec3d_map, apix = _prepare_download_map(
            result,
            match_input_box=input.dn_match_input_box(),
            input_image_shape=input_image_shape,
            input_apix=imgs.apix,
            compact_apix=compact_apix,
            cpu=input.dn_cpu(),
        )

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
        req(res[0][1][3] is not None)
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

    @reactive.effect(priority=100)
    def _seed_crop_sizes_from_images():
        """Start the crop controls at the full image, not the 32/256 defaults.

        On the joint route the per-image cards read these reactive values for
        their initial Vertical/Horizontal crop, and there is no shared card to
        have corrected them first.
        """
        imgs = initial_image()
        req(len(imgs))
        ny, nx = np.shape(imgs[0])
        if vertical_crop_size_rv() > ny or vertical_crop_size_rv() < 32:
            vertical_crop_size_rv.set(int(ny) // 2 * 2)
        if horizontal_crop_size_rv() > nx:
            horizontal_crop_size_rv.set(int(nx) // 2 * 2)

    @reactive.effect(priority=100)
    def _seed_apix_from_images():
        """Keep apix_rv current from the loaded stack.

        Same reason as the threshold above: on the joint route there is no
        shared Pixel size control to sync this back, and the per-image cards
        read it for their starting value.
        """
        imgs = all_images()
        req(imgs is not None)
        apix = round(float(imgs.apix), 4)
        if apix > 0 and apix_rv() != apix:
            apix_rv.set(apix)

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
    @reactive.event(selected_images_original)
    def _clear_per_image_transforms():
        """Per-image transforms belong to one selection; drop them on a change.

        A new selection of the same size would otherwise silently inherit the
        previous images' rotations, which the length check cannot catch.

        Keyed on the selection only, never the mode: the joint and automatic
        stitch routes both use these values, so clearing them on a mode change
        threw away the auto-transform just as the automatic stitch was about to
        compose with it -- which left every image's rotation coming out as the
        small registration residual alone.
        """
        per_image_transforms.set([])

    @reactive.effect
    @reactive.event(selected_images_original)
    def _clear_autostitch_transfer():
        """Drop a transferred layout when the selection changes.

        Deliberately keyed on the selection alone, not on the mode: the whole
        point of the transfer is to survive the switch from automatic to manual
        stitching, so clearing it on a mode change would erase it moments before
        the manual card reads it.
        """
        autostitch_transforms.set([])

    @reactive.effect
    @reactive.event(multi_mode_rv)
    def _discard_stitch_on_mode_change():
        """Any change of route discards the stitched image.

        A stitch made by one route is not a valid starting state for another.
        Left in place it keeps standing in for the selection: the joint search
        would quietly run on one stitched image instead of the several that were
        picked, and switching between the stitch routes showed a composite that
        the other route's controls did not describe.

        Each route therefore starts again from the individual images.
        """
        if len(stitched_image_displayed()):
            stitched_image_displayed.set([])
            stitched_image_labels.set([])
            stitched_image_links.set([])
        if autostitch_report():
            autostitch_report.set({})

    @reactive.effect
    @reactive.event(
        selected_images_original,
        stitched_image_displayed,
        multi_mode_rv,
        ignore_init=False,
    )
    def _set_initial_image():
        req(len(selected_images_original()))
        # What the threshold/transform/crop chain operates on:
        #   - a stitched image, once made, replaces the selection entirely;
        #   - the joint and automatic-stitch routes both need every selected
        #     image auto-transformed, so the whole selection goes through;
        #   - manual stitching before stitching has nothing to show yet, since
        #     running the selection through this chain individually is not what
        #     that route does.
        if len(stitched_image_displayed()):
            initial_image.set(stitched_image_displayed())
        elif _stitching_active():
            initial_image.set([])
        else:
            initial_image.set(selected_images_original())
        new_initial_image.set(True)

    # ══════════════════════════════════════════════════════════════════
    # Reactive effects: thresholding & transformation
    # ══════════════════════════════════════════════════════════════════

    # Plain effect: dn_img_negate belongs to the shared card, which the joint
    # route replaces with per-image cards, so naming it here would abort this
    # effect and leave threshold_rv at its 0.0 default -- which the per-image
    # cards then read as their starting Threshold.
    @reactive.effect(priority=100)
    def _update_threshold_scale():
        req(len(initial_image()))
        images = initial_image()
        if _input_or("dn_img_negate", img_negate_rv()):
            images = [-img for img in images]
        min_val = float(np.min([np.min(img) for img in images]))
        max_val = float(np.max([np.max(img) for img in images]))
        step_val = (max_val - min_val) / 100
        from skimage.filters import threshold_otsu

        thresh_value = float(np.median([threshold_otsu(img) for img in images]))
        # Set the reactive value directly, not only the input: the shared card
        # that used to sync it back does not exist on the joint route, and the
        # per-image cards seed their Threshold from this value.
        threshold_rv.set(round(thresh_value, 3))
        ui.update_numeric(
            "dn_threshold",
            value=round(thresh_value, 3),
            min=round(min_val, 3),
            max=round(max_val, 3),
            step=round(step_val, 3),
        )

    def _prepare_image(i, img, threshold=True):
        """Apply image *i*'s input options: negate, threshold, transpose, flip.

        ``threshold=False`` skips only the thresholding and keeps everything
        else, which is what automatic stitching resamples from. Thresholding
        flattens the background to a constant, so compositing thresholded
        images yields a stitched image with a dead flat background instead of
        the raw noise -- unlike every other view in the tab, and unlike manual
        stitching, which composites the untouched originals. The geometric
        options have to stay either way: the auto-transform's rotation and
        shift were measured through them, so dropping them would invalidate
        the transform being composed on top.
        """
        work = -img if _param("negate", i, False) else img
        if threshold:
            work = helicon.threshold_data(
                work, thresh_value=_param("threshold", i, threshold_rv())
            )
        if _param("transpose", i, False):
            work = np.transpose(work)
        if _param("flip", i, False):
            work = np.fliplr(work)
        return work

    # Plain effect, not reactive.event: in joint mode these settings live on
    # per-image controls, and naming a not-yet-created input in an event list
    # aborts the effect outright.
    @reactive.effect(priority=90)
    def _threshold_selected_images():
        images = initial_image()
        req(len(images))
        selected_images_thresholded.set(
            [_prepare_image(i, img) for i, img in enumerate(images)]
        )

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

        auto = helix_transform.auto_transform(
            images,
            estimate_rotation=estimate_rotation,
            estimate_center=estimate_center,
            crop_factor=1.2 if input_data().is_3d else 2.0,
        )
        tmp = np.array([(r, s, auto.diameter) for r, s in auto.per_image])
        diameter = auto.diameter
        crop_size = auto.crop_size

        if len(images) > 1:
            # Every class average sits at its own angle and height, and no
            # single value describes them: on ten good EMPIAR-10940 classes the
            # per-image rotations span -20.1 to +10.2 deg with a mean of -0.47,
            # so applying that mean leaves them 5.73 deg off horizontal on
            # average (worst 19.6) where per-image transforms give 0.06 (worst
            # 0.16). Keep the individual values and leave the shared Rotation
            # and Vertical shift boxes at zero, where they act as a common
            # nudge applied on top of each image's own transform.
            per_image_transforms.set([(float(r), float(s)) for r, s, _ in tmp])
            # The cards already exist, so update them in place rather than
            # re-rendering, which would discard the user's other edits.
            for i, (r, sh, _d) in enumerate(tmp):
                img_apix = _param("apix", i, apix_rv()) or 1.0
                ui.update_numeric(_pi_id("rot", i), value=round(float(r), 2))
                ui.update_numeric(_pi_id("dy", i), value=round(float(sh) * img_apix, 2))
                ui.update_numeric(_pi_id("vcrop", i), value=max(32, crop_size))
            rotation = 0.0
            shift_y = 0.0
        else:
            per_image_transforms.set([])
            rotation = float(tmp[0, 0])
            shift_y = float(tmp[0, 1]) * _input_or("dn_apix", apix_rv())

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
        per_image_transforms.set([])
        for i in range(len(images)):
            ui.update_numeric(_pi_id("rot", i), value=0.0)
            ui.update_numeric(_pi_id("dy", i), value=0.0)
            ui.update_numeric(_pi_id("vcrop", i), value=ny // 2 * 2)
            ui.update_numeric(_pi_id("hcrop", i), value=nx // 2 * 2)
        ui.update_numeric("dn_pre_rotation", value=0.0)
        ui.update_numeric("dn_shift_y", value=0.0)
        ui.update_numeric("dn_vertical_crop_size", value=ny // 2 * 2)
        ui.update_numeric("dn_horizontal_crop_size", value=nx // 2 * 2)

    def _input_or(name, default=0.0):
        """Read a dynamically created input, or a default before it exists.

        Reading an unset input registers the dependency and then raises
        SilentException, so catching it here still means this effect re-runs
        once the input appears or the user changes it.
        """
        try:
            val = input[name]()
        except SilentException:
            return default
        return default if val is None else val

    # A plain effect, deliberately not reactive.event: the per-image inputs are
    # created dynamically, and reactive.event calls every dependency up front,
    # so naming an input that does not exist yet raises SilentException and
    # silently aborts the whole effect -- which left the transformed images
    # never being set, and so no gallery at all. An effect tracks whatever it
    # actually reads, which also means there is no cap on the image count.
    @reactive.effect
    def _transform_selected_images():
        images = selected_images_thresholded()
        req(len(images))
        per = per_image_transforms()
        if len(per) != len(images):
            per = [(0.0, 0.0)] * len(images)
        apix = _input_or("dn_apix", apix_rv()) or 1.0

        rotated = []
        for i, img in enumerate(images):
            img_apix = _param("apix", i, apix) or apix
            rotation = _param("rot", i, per[i][0])
            shift = _param("dy", i, per[i][1] * img_apix) / img_apix
            if rotation or shift:
                rotated.append(
                    helicon.transform_image(
                        image=img, rotation=rotation, post_translation=(shift, 0)
                    )
                )
            else:
                # Must stay on the thresholded image: the originals would
                # silently drop the thresholding, and for a stitched image they
                # are the wrong images entirely.
                rotated.append(img)
        selected_images_thresholded_rotated_shifted.set(rotated)

    @reactive.effect
    def _crop_selected_images():
        images = selected_images_thresholded_rotated_shifted()
        req(len(images))
        cropped = []
        for i, img in enumerate(images):
            ny, nx = img.shape
            crop_ny = int(_param("vcrop", i, vertical_crop_size_rv()) or ny)
            crop_nx = int(_param("hcrop", i, horizontal_crop_size_rv()) or nx)
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
    @reactive.event(input.dn_undo_stitch)
    def _undo_auto_stitch():
        """Discard the stitched image and go back to the individual images."""
        stitched_image_displayed.set([])
        stitched_image_labels.set([])
        stitched_image_links.set([])
        autostitch_report.set({})

    @reactive.effect
    @reactive.event(input.dn_auto_stitch)
    def _auto_stitch_images():
        """Register the selected images to each other and composite them.

        Registration runs on the *auto-transformed* images, which are already
        near horizontal and centred, so only a small residual remains to be
        found and the search stays tight and reliable. The transforms are then
        composed with the auto-transform and applied once to the originals, so
        the composite is interpolated a single time rather than twice.
        """
        # The auto-transformed, cropped images -- not selected_images_thresholded,
        # which is thresholded but NOT yet rotated or centred. Registering those
        # would hand a tight rotation search images still tilted by up to ten
        # degrees, and then compose the auto-transform on top of a registration
        # that had already tried to absorb it.
        images = selected_images_thresholded_rotated_shifted_cropped()
        req(len(images) > 1)
        # Resample from the *unthresholded* images. Registration wants the
        # threshold -- it suppresses the background that would otherwise
        # dominate the correlation -- but the composite must not carry it, or
        # the stitched image comes out with a flat background instead of the
        # raw noise the inputs actually have. Manual stitching composites the
        # untouched originals, and this is the same thing.
        originals = [
            _prepare_image(i, img, threshold=False)
            for i, img in enumerate(initial_image())
        ]
        if len(originals) != len(images):
            originals = images

        per = per_image_transforms()
        if len(per) != len(images):
            per = [(0.0, 0.0)] * len(images)
        apix = _input_or("dn_apix", apix_rv()) or 1.0
        # What auto-transform applied, per image, so it can be folded back in.
        auto = [
            (
                _param("rot", i, per[i][0]),
                _param("dy", i, per[i][1] * apix) / apix,
            )
            for i in range(len(images))
        ]

        # Every pair is registered twice (polarity, then rotation and shifts)
        # and every image once per refinement round, so the count is knowable up
        # front and the bar can be honest about how far along it is.
        n_jobs = denovo3d_register.n_registration_jobs(len(images))
        with ui.Progress(min=0, max=n_jobs) as p:
            done = [0]

            def _tick(label):
                done[0] += 1
                p.set(
                    done[0],
                    message=f"Registering: {done[0]}/{n_jobs}",
                    detail=label,
                )

            p.set(0, message=f"Registering: 0/{n_jobs}", detail="starting ...")
            try:
                _stitched, transforms, diagnostics = denovo3d_register.auto_stitch(
                    images,
                    rot_range=2.0,
                    dy_range=3.0,
                    coarse_step=0.5,
                    progress=_tick,
                )
            except Exception:
                logger.error("automatic stitching failed", exc_info=True)
                ui.modal_show(
                    ui.modal(
                        "Automatic registration failed; see the log for details.",
                        title="Auto Stitch",
                        easy_close=True,
                        footer=None,
                    )
                )
                return

        # Compose with the auto-transform and resample the originals once.
        composed = []
        for (r, sy), t in zip(auto, transforms):
            c = denovo3d_register.compose_transforms(r, sy, t)
            c["connected"] = t.get("connected", True)
            c["dx"] = t["dx"] + c.pop("extra_dx", 0.0)
            composed.append(c)
        placed = [
            dict(
                psi=0.0,
                dy=0.0,
                dx=c["dx"],
                connected=c["connected"],
                flip_x=False,
                flip_y=False,
            )
            for c in composed
        ]
        oriented = [
            denovo3d_register.apply_composed(im, c)
            for im, c in zip(originals, composed)
        ]
        stitched, _coverage = denovo3d_register.composite(oriented, placed)
        if stitched is None:
            logger.warning("automatic stitching produced no image")
            return

        # Convert into what the manual sliders mean. Their x value is a
        # correction from an end-to-end tiled layout, not an absolute position,
        # so the tiled offset has to come back out.
        width = int(np.shape(originals[0])[1]) if len(originals) else 0
        placed_dx = [c["dx"] for c in composed if c["connected"]]
        base = min(placed_dx) if placed_dx else 0.0
        transferred = []
        for i, (c, t) in enumerate(zip(composed, transforms)):
            transferred.append(
                dict(
                    flip_x=bool(t.get("flip_x", False)),
                    flip_y=bool(t.get("flip_y", False)),
                    rotation=float(c["rotation"]),
                    shift_y=float(c["shift_y"]),
                    shift_x=float(c["dx"] - base - i * width),
                    connected=bool(c["connected"]),
                )
            )
        autostitch_transforms.set(transferred)

        report = dict(diagnostics)
        report.update(
            n_images=len(images),
            n_possible=len(images) * (len(images) - 1) // 2,
        )
        autostitch_report.set(report)
        logger.info(
            "auto stitch: %d/%d pairs, %d placed, span %.2fx, trustworthy=%s",
            diagnostics.get("n_pairs", 0),
            report["n_possible"],
            diagnostics.get("n_connected", 0),
            diagnostics.get("span_gain", 1.0),
            diagnostics.get("trustworthy"),
        )

        result = np.asarray(stitched, dtype=np.float32)
        if result.std() > 0:
            result = (result - result.mean()) / result.std()
            result = result / max(abs(result.max()), 1e-6)
        stitched_image_displayed.set([result])
        stitched_image_labels.set([""])
        stitched_image_links.set([""])

    @reactive.effect
    @reactive.event(input.dn_perform_stitching)
    def _update_stitched_image_displayed():
        """Composite the manually placed images.

        Placed directly rather than through ITK's montage: that expects a
        regular grid of tiles and infers the grid from the positions, so it
        fails outright once the images overlap substantially -- which is
        exactly what a layout transferred from the automatic stitch looks like
        (four 128 px images spanning 172 px). It raised

            ITK ERROR: Axis sizes: [2, 1] current index: [0, 1]

        for four images in a row. Compositing here also means the two stitch
        routes combine their images the same way, so a transferred layout
        reproduces the automatic result instead of merely approximating it.
        """
        images = selected_images_rotated_shifted()
        req(len(images))
        x_positions = _image_stitching_x_positions(
            images, transformed_images_x_offsets()
        )
        # The per-image effects have already applied flip, rotation and dy, so
        # only the placement is left.
        placed = [
            dict(
                psi=0.0, dy=0.0, dx=float(x), flip_x=False, flip_y=False, connected=True
            )
            for x in x_positions
        ]
        result, _coverage = denovo3d_register.composite(images, placed)
        if result is None:
            logger.warning("manual stitching produced no image")
            return

        result = np.asarray(result, dtype=np.float32)
        if result.std() > 0:
            result = (result - result.mean()) / result.std()
            result = result / max(abs(result.max()), 1e-6)
        stitched_image_displayed.set([result])
        stitched_image_labels.set([""])
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
        images = selected_images_thresholded_rotated_shifted_cropped()
        req(len(images) > 0)

        binning_factor = max(1, getattr(input, "dn_binning", lambda: 1)())
        # Pixel size is per-image in joint mode, so each task carries its own.
        apix_per_image = [
            _param("apix", i, apix_rv()) * binning_factor for i in range(len(images))
        ]
        apix_binned = apix_per_image[0]

        imageFile = selected_images_title().strip(":")
        labels = list(selected_images_labels())
        if len(labels) != len(images):
            # A stitched image collapses the selection into one image
            labels = [f"Stitched: {'+'.join(str(l) for l in labels)}"][: len(images)]

        log = _denovo3d_logger()

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
        n_jobs = n_pairs * len(images)
        n_threads_per_job = max(1, n_cpu // max(1, n_jobs))

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
            reconstruct_length = input.dn_reconstruct_length_rise() * rise

            # A search cares only about the score; a single pair means the user
            # wants the map, and the same solver is not best for both.
            algorithm = dict(
                model=(
                    input.dn_rec_algorithm() if return_3d else input.dn_lr_algorithm()
                ),
                l1_ratio=input.dn_lr_l1_ratio(),
            )
            if input.dn_lr_alpha() >= 0:
                algorithm["alpha"] = input.dn_lr_alpha()

            if abs(twist) < 0.01:
                log.warning(f"WARNING: twist={round(twist, 3)} ignored (too small)")
                continue
            if abs(rise) < 0.01:
                log.warning(f"WARNING: rise={round(rise, 3)} ignored (too small)")
                continue

            for img_i, (data, imageIndex) in enumerate(zip(images, labels)):
                apix = apix_per_image[img_i]
                ny, nx = data.shape
                tube_length = nx * apix
                tube_diameter = ny * apix
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

        if len(images) > 1:
            log.info(
                f"joint search over {len(images)} images x {n_pairs} twist/rise pairs"
            )

        # A search scores with the search solver, but the pictures it puts on
        # screen are reconstructions, and the user compares them against the
        # input image. Drawing them with the search solver shows a map the
        # reconstruction selector was never going to produce -- so when the two
        # differ, redraw the displayed pairs with the reconstruction solver.
        # The scores, and therefore the ranking, stay the search solver's.
        if return_3d or input.dn_rec_algorithm() == input.dn_lr_algorithm():
            display_model = None
        else:
            display_model = input.dn_rec_algorithm()
        log.info(
            f"{n_pairs} twist/rise pair(s) x {len(images)} image(s);"
            f" search={input.dn_lr_algorithm()}"
            f" reconstruction={input.dn_rec_algorithm()}"
            f" -> solving with {algorithm['model']}"
            + (f", redrawing the display with {display_model}" if display_model else "")
        )

        abort_flag[0] = False
        # Everything the task needs is read here, on this side of the boundary.
        # An extended task may not read reactive sources at all -- doing so
        # raises, the run dies inside the catch-all below, and the only symptom
        # is a search that produces nothing.
        _reconstruction_task(
            tasks,
            n_cpu,
            abort_flag,
            len(images),
            display_model,
            input.dn_top_n_results(),
            _projmatch_active(),
            input.dn_lr_algorithm(),
            list(images),
            input.dn_rec_algorithm(),
        )

    @reactive.extended_task
    async def _reconstruction_task(
        tasks,
        cpu,
        abort_ref,
        n_images=1,
        display_model=None,
        top_n=1,
        projmatch=False,
        lr_algorithm="elasticnet",
        images=None,
        rec_algorithm=None,
    ):
        log = _denovo3d_logger()

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
                            reconstruction_results_raw.set(list(results))
                            reconstruction_results.set(_rank(results, n_images))

                    t_final = time()
                    log.info("reconstruction time: %s", t_final - t0)

            if n_discarded:
                log.info(
                    f"{n_discarded}/{len(tasks)} results are None and thus discarded"
                )

            async def _rank_final(results, log=None):
                """Rank, by whichever joint method the route asks for.

                Used everywhere a *final* ranking is produced -- including after
                the display re-solve, which otherwise re-ranked with the plain
                joint method and silently undid the projection-matching order.
                The interim rankings inside the solve loop deliberately stay on
                the cheap method: projection matching costs a refinement per
                twist, which is worth paying once, not on every progress update.

                Run in a worker thread, for the same reason the solve above is:
                it takes minutes, and called straight from this coroutine it
                blocked the event loop for all of them. Nothing was serviced
                while it ran -- no progress, no other effects -- and a session
                left unattended that long can lose its connection, after which
                the *next* run looks stuck whichever route it uses.
                """
                from concurrent.futures import ThreadPoolExecutor

                ranked = _rank(results, n_images, log)
                if not (projmatch and n_images > 1 and images and not abort_ref[0]):
                    projmatch_composites.set({})
                    return ranked

                # The per-image results are kept exactly as solved, so the
                # per-image views are identical on both joint routes and only
                # the ordering -- and the composite -- differ.
                seen = dict(i=0, n=0, twist=0.0)

                def _on_pair(i, n, twist):
                    seen.update(i=i, n=n, twist=twist)
                    return not abort_ref[0]

                def _work():
                    return denovo3d_joint.rank_by_projection_matching(
                        results,
                        images,
                        algorithm=dict(model=lr_algorithm),
                        display_algorithm=(
                            dict(model=rec_algorithm)
                            if rec_algorithm and rec_algorithm != lr_algorithm
                            else None
                        ),
                        log=log,
                        progress=_on_pair,
                    )

                loop = asyncio.get_running_loop()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = loop.run_in_executor(pool, _work)
                    with ui.Progress(min=0, max=1) as pm:
                        pm.set(0, message="Matching projections across images")
                        while not future.done():
                            await asyncio.sleep(0.1)
                            if seen["n"]:
                                pm.set(
                                    seen["i"] / seen["n"],
                                    message="Matching projections across images",
                                    detail=f"twist {seen['twist']:.3f}"
                                    f" ({seen['i'] + 1}/{seen['n']})",
                                )
                        matched, composites = await future

                if abort_ref[0]:
                    projmatch_composites.set({})
                    return ranked
                projmatch_composites.set(composites)

                # Hand the winning twist's placements to the manual stitch, the
                # same way the automatic stitcher hands over its own layout. An
                # azimuth is an axial position, so the search has already
                # worked out where every image sits; making the user press a
                # button to have that computed again -- which is what this used
                # to do -- was redundant twice over, since the answer was
                # already in hand.
                best = matched[0][2] if matched else None
                found = (
                    composites.get((round(float(best[5]), 6), round(float(best[6]), 6)))
                    if best is not None
                    else None
                )
                if found and found.get("placed") is not None:
                    try:
                        autostitch_transforms.set(
                            denovo3d_joint.placements_as_transforms(
                                found["phis"],
                                found["placed"],
                                found["twist"],
                                found["rise"],
                                found["apix2d"],
                                int(np.shape(images[0])[1]),
                                two_fold=found["two_fold"],
                            )
                        )
                    except Exception:  # pragma: no cover - a layout is a bonus
                        logger.warning(
                            "could not express the placements as a manual"
                            " stitch layout:\n%s",
                            traceback.format_exc(),
                        )
                return matched

            ranked = await _rank_final(results, log)
            reconstruction_results_raw.set(list(results))
            reconstruction_results.set(ranked)

            if display_model and results and not abort_ref[0]:
                results = await _redraw_with(
                    tasks, results, ranked, top_n, display_model, cpu, abort_ref, log
                )
                reconstruction_results_raw.set(list(results))
                reconstruction_results.set(await _rank_final(results, log))
        except Exception:
            log.error("Reconstruction task failed:\n%s", traceback.format_exc())
            # Say so on screen. Swallowed silently, a failure here is
            # indistinguishable from a slow run: the progress bar goes, no
            # results appear, and the app looks hung rather than broken. That
            # is exactly how a RuntimeError raised inside this task -- for
            # reading a reactive source, which an extended task may not do --
            # presented itself.
            try:
                ui.notification_show(
                    "The search failed. See helicon.denovo3D.log under"
                    f" {helicon.cache_dir / 'logs'} for the details.",
                    type="error",
                    duration=15,
                )
            except Exception:  # pragma: no cover - no session to notify
                pass

    async def _redraw_with(
        tasks, results, ranked, top_n, display_model, cpu, abort_ref, log
    ):
        """Re-solve the displayed twist/rise pairs with another solver.

        Only the projections are taken from the second solve. Every score keeps
        the value the search produced, so the ranking the user is looking at is
        the one the search actually computed -- this changes the picture, not
        the answer.

        Returns
        -------
        list
            ``results`` with the displayed entries' return_data replaced. Pairs
            that were not displayed, and any re-solve that failed, are left as
            they were.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        redo, slot = _display_redraw_plan(tasks, results, ranked, top_n, display_model)
        if not redo:
            return results

        log.info(
            f"redrawing {len(redo)} displayed reconstructions with {display_model}"
        )
        results = list(results)
        with ui.Progress(min=0, max=len(redo)) as p:
            p.set(message=f"Reconstructing with {display_model}", detail="for display")
            with ThreadPoolExecutor(max_workers=cpu) as executor:
                futures = [
                    executor.submit(denovo3d_pipeline.process_one_task, *t)
                    for t in redo
                ]
                done = 0
                for completed_task in as_completed(futures):
                    await asyncio.sleep(0)
                    if abort_ref[0] is True:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    done += 1
                    p.set(done, message=f"Reconstructed {done}/{len(redo)}")
                    try:
                        result = completed_task.result()
                    except Exception:
                        log.error(
                            "Display reconstruction raised an exception:\n%s",
                            traceback.format_exc(),
                        )
                        continue
                    if result is None:
                        continue
                    i = slot.get(_display_key(result[2]))
                    if i is not None:
                        # Score and parameters from the search, images from here.
                        results[i] = (results[i][0], result[1], results[i][2])
        return results

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
        ranked = reconstruction_results()
        req(len(ranked))

        top_n = input.dn_top_n_results()
        if top_n <= 0:
            top_n = len(ranked)

        # Group the raw results by twist/rise so each ranked pair can show every
        # image's reconstruction. Ordered rank-major and image-minor: the best
        # twist for image 1, 2, ... n, then the second-best for image 1, 2, ...
        # so the same twist can be compared across images side by side. That is
        # n * top_n entries for n selected images.
        by_pair = {}
        for r in reconstruction_results_raw():
            key = (round(float(r[2][5]), 6), round(float(r[2][6]), 6))
            by_pair.setdefault(key, {})[r[2][2]] = r
        image_order = [str(l) for l in selected_images_labels()]
        n_solved = len(selected_images_thresholded_rotated_shifted_cropped())
        joint = n_solved > 1

        labels = []
        images = []
        for ri, ranked_result in enumerate(ranked[:top_n]):
            joint_score = ranked_result[0]
            pair = (
                round(float(ranked_result[2][5]), 6),
                round(float(ranked_result[2][6]), 6),
            )
            group = by_pair.get(pair, {})
            # Selection order where known, so the images stay in a stable,
            # recognisable sequence; anything unmatched is appended.
            ordered = [group[k] for k in image_order if k in group]
            ordered += [v for k, v in group.items() if k not in image_order]
            if not ordered:
                ordered = [ranked_result]

            # The whole-set views come first, before this twist's per-image
            # rows: every selected image laid onto one canvas at the axial
            # position its azimuth implies, and directly beneath it the long
            # side projection of the volume that placement produced. They share
            # a canvas -- one period plus an image width -- so they line up
            # column for column and can be read against each other.
            #
            # A picture, not a score. Measured, the composite's contrast peaks
            # at a different twist from the fit (see placement_composite), so
            # take the ranking from the score and this from the eye.
            pictures = projmatch_composites().get(pair) or {}
            twist_val, rise_val = pair
            stamp = f"|twist={round(twist_val, 3)}deg|rise={round(rise_val, 6)}A"
            composite = pictures.get("composite")
            model = pictures.get("model")
            if composite is not None and np.size(composite):
                labels.append(f"{ri+1}: all {len(ordered)} images placed{stamp}")
                images.append(np.asarray(composite, dtype=np.float32))
            if model is not None and np.size(model):
                labels.append(f"{ri+1}: model projection{stamp}")
                images.append(np.asarray(model, dtype=np.float32))
            zview = pictures.get("zview")
            if zview is not None and np.size(zview):
                labels.append(f"{ri+1}: model Z{stamp}")
                images.append(np.asarray(zview, dtype=np.float32))

            for result in ordered:
                (
                    score,
                    (rec3d_x_proj, _rec3d_y_proj, rec3d_z_sections, rec3d, *_rest1),
                    (
                        query_image,
                        _imageFile,
                        imageIndex,
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

                # The query image comes from the result itself, so it is always
                # the image this reconstruction was solved from.
                query_image_padded = helicon.pad_to_size(
                    query_image, shape=rec3d_x_proj.shape
                )
                rec3d_z_sections_padded = helicon.pad_to_size(
                    rec3d_z_sections, shape=rec3d_x_proj.shape
                )

                pitch_val = (
                    int(round(rise * 360 / abs(twist))) if abs(twist) > 0.01 else 0
                )
                if joint:
                    label_x = (
                        f"{ri+1}: X|{imageIndex}|score={score:.4f}"
                        f"|joint={joint_score:.4f}|pitch={pitch_val:,}A"
                        f"|twist={round(twist, 3)}deg|rise={round(rise, 6)}A"
                    )
                else:
                    label_x = (
                        f"{ri+1}: X|score={score:.4f}|pitch={pitch_val:,}A"
                        f"|twist={round(twist, 3)}deg|rise={round(rise, 6)}A"
                    )
                labels += [
                    f"Input image: {imageIndex}",
                    label_x,
                    f"{ri+1}: Z",
                ]
                images += [
                    query_image_padded,
                    rec3d_x_proj,
                    rec3d_z_sections_padded,
                ]

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
