"""The page of the denovo3d tab: its inputs, controls and plots, and what only they need."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.io as pio
import helicon
from shiny import ui, module
from .. import deployment
from ..lib import helix_transform
from .denovo3d_bookmark import _urls, _url_key  # noqa: E402,F401
from .. import auto_method_ui  # noqa: E402


# A click on a point that carries customdata (twist, rise) is sent to the app as an
# input of the given id. ``{plot_id}`` is filled with the figure's own id by Plotly.
_CLICK_TO_INPUT = (
    "var gd = document.getElementById('{plot_id}');"
    "if (gd && window.Shiny) {"
    "  gd.on('plotly_click', function (ev) {"
    "    var p = ev.points && ev.points[0];"
    "    if (!p || !p.customdata) return;"
    "    Shiny.setInputValue('%s', {twist: p.customdata[0], rise: p.customdata[1],"
    "      nonce: Date.now()}, {priority: 'event'});"
    "  });"
    "}"
)


def _fig_to_html(fig, click_input=None):
    """Convert a plotly figure to responsive HTML for render.ui.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    click_input : str, optional
        Full id of the Shiny input that receives ``{twist, rise}`` when a point
        carrying that pair as its customdata is clicked.
    """
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displayModeBar": False},
        default_height=400,
        post_script=None if click_input is None else _CLICK_TO_INPUT % click_input,
    )
    return ui.HTML(html)


# What the hover text of a candidate says a click does.
CLICK_HINT = "Click to reconstruct a 3D map with this twist and rise"


def _scores_figure(res, picked=None):
    """The scores of the candidate twist/rise solutions, each clickable.

    A heat map over twist and rise when both were searched, a line when one was.
    Every point carries its ``(twist, rise)`` as customdata, which is what a click
    sends back, and its hover text says what a click does (:data:`CLICK_HINT`).

    Parameters
    ----------
    res : list
        ``(score, return_data, params)`` of each candidate (params[5] twist,
        params[6] rise).
    picked : tuple of float, optional
        ``(twist, rise)`` whose 3D map was reconstructed: marked on the plot.

    Returns
    -------
    plotly.graph_objects.Figure or None
        None when there are fewer than two candidates, or one value of each.
    """
    if len(res) <= 1:
        return None
    results_arr = np.zeros((3, len(res)), dtype=float)
    for ri, result in enumerate(res):
        score, _projs, params = result
        results_arr[0, ri] = params[5]
        results_arr[1, ri] = params[6]
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
        fig.update_traces(
            customdata=np.stack([X.T, Y.T], axis=-1),
            hovertemplate=(
                "Twist: %{x:.3f} deg<br>Rise: %{y:.3f} A<br>Score: %{z:.3f}"
                f"<br><i>{CLICK_HINT}</i><extra></extra>"
            ),
        )
        max_idx = np.unravel_index(np.argmax(Z), Z.shape)
        max_x = x_unique[max_idx[0]]
        max_y = y_unique[max_idx[1]]
        dx = (x_unique[1] - x_unique[0]) / 2
        dy = (y_unique[1] - y_unique[0]) / 2
        fig.add_shape(
            type="rect",
            x0=max_x - dx,
            y0=max_y - dy,
            x1=max_x + dx,
            y1=max_y + dy,
            line=dict(color="red", width=2),
            fillcolor=None,
        )
        if picked is not None:
            fig.add_scatter(
                x=[picked[0]],
                y=[picked[1]],
                mode="markers",
                marker=dict(symbol="x", size=11, color="white", line=dict(width=2)),
                hoverinfo="skip",
                showlegend=False,
            )
    elif n_twists > 1 or n_rises > 1:
        if n_twists > 1:
            x = results_arr[0, :]
            x_title = "Twist (deg)"
            hovertemplate = (
                "Twist: %{x} deg<br>Rise: %{customdata[1]:.3f} A<br>Score: %{y}"
                f"<br><i>{CLICK_HINT}</i><extra></extra>"
            )
        else:
            x = results_arr[1, :]
            x_title = "Rise (A)"
            hovertemplate = (
                "Rise: %{x} A<br>Twist: %{customdata[0]:.3f} deg<br>Score: %{y}"
                f"<br><i>{CLICK_HINT}</i><extra></extra>"
            )
        y = results_arr[2, :]
        sort_idx = np.argsort(x)
        x = np.array(x)[sort_idx]
        y = np.array(y)[sort_idx]
        pairs = np.stack([results_arr[0, :], results_arr[1, :]], axis=-1)[sort_idx]
        fig = px.line(x=x, y=y, color_discrete_sequence=["blue"], markers=True)
        fig.update_layout(xaxis_title=x_title, yaxis_title="Score", showlegend=False)
        fig.update_traces(hovertemplate=hovertemplate, customdata=pairs)
        if picked is not None:
            at = [
                i
                for i, (t, r) in enumerate(pairs)
                if abs(t - picked[0]) < 1e-6 and abs(r - picked[1]) < 1e-6
            ]
            if at:
                fig.add_scatter(
                    x=[x[at[0]]],
                    y=[y[at[0]]],
                    mode="markers",
                    marker=dict(symbol="x", size=12, color="red", line=dict(width=2)),
                    hoverinfo="skip",
                    showlegend=False,
                )
    else:
        return None
    return fig


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
                            choices=helicon.shiny.source_modes(
                                deployment.is_cloud(), ("upload", "url", "emdb")
                            ),
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
                        ui.tooltip(
                            ui.input_select(
                                "dn_auto_method",
                                auto_method_ui.LABEL,
                                choices=auto_method_ui.CHOICES,
                                selected=auto_method_ui.DEFAULT,
                            ),
                            auto_method_ui.HELP,
                        ),
                        col_widths=6,
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
                "(input['dn_twist_range'] || [0, 0])[0] !== (input['dn_twist_range'] || [0, 0])[1] || (input['dn_rise_range'] || [0, 0])[0] !== (input['dn_rise_range'] || [0, 0])[1]",
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
