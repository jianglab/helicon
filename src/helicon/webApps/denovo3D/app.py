from pathlib import Path
import numpy as np
import pandas as pd

from shinywidgets import render_plotly

import shiny
from shiny import reactive, req
from shiny.express import input, ui, render

import helicon

from . import compute

images_all = reactive.value([])
image_size = reactive.value(0)
image_apix = reactive.value(0)

displayed_image_ids = reactive.value([])
displayed_images = reactive.value([])
displayed_image_title = reactive.value("Select an image:")
displayed_image_labels = reactive.value([])

initial_selected_image_indices = reactive.value([0])
selected_images_original = reactive.value([])
selected_images_thresholded = reactive.value([])
selected_images_thresholded_rotated_shifted = reactive.value([])
selected_image_diameter = reactive.value(0)
selected_images_thresholded_rotated_shifted_cropped = reactive.value([])
selected_images_title = reactive.value("Selected image:")
selected_images_labels = reactive.value([])

reconstrunction_results = reactive.value([])
reconstructed_projection_images = reactive.value([])
reconstructed_projection_labels = reactive.value([])
reconstructed_map = reactive.value(None)


ui.head_content(ui.tags.title("Helicon denovo3D"))
helicon.shiny.google_analytics(id="G-FDSYXQNKLX")
helicon.shiny.setup_ajdustable_sidebar()
ui.tags.style(
    """
    * { font-size: 10pt; padding:0; border: 0; margin: 0; }
    aside {--_padding-icon: 10px;}
    """
)
urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-14046/map/emd_14046.map.gz",
    )
}
url_key = "empiar-10940_job010"

with ui.sidebar(
    width="33vw", style="display: flex; flex-direction: column; height: 100%;"
):
    with ui.navset_pill(id="tab"):
        with ui.nav_panel("Input 2D Images"):
            with ui.div(
                id="input_image_files",
                style="display: flex; flex-direction: column; align-items: flex-start;",
            ):
                ui.input_radio_buttons(
                    "input_mode_images",
                    "How to obtain the input images:",
                    choices=["upload", "url"],
                    selected="url",
                    inline=True,
                )

                @render.ui
                @reactive.event(input.input_mode_images)
                def create_input_image_files_ui():
                    displayed_images.set([])
                    ret = []
                    if input.input_mode_images() == "upload":
                        ret.append(
                            ui.input_file(
                                "upload_images",
                                "Upload the input images in MRC format (.mrcs, .mrc)",
                                accept=[".mrcs", ".mrc"],
                                placeholder="mrcs or mrc file",
                            )
                        )
                    elif input.input_mode_images() == "url":
                        ret.append(
                            ui.input_text(
                                "url_images",
                                "Download URL for a RELION or cryoSPARC image output mrc(s) file",
                                value=urls[url_key][0],
                            )
                        )
                    return ret

            with ui.div(
                id="image-selection",
                style="max-height: 80vh; overflow-y: auto; display: flex; flex-direction: column; align-items: center;",
            ):
                helicon.shiny.image_select(
                    id="select_image",
                    label=displayed_image_title,
                    images=displayed_images,
                    image_labels=displayed_image_labels,
                    image_size=reactive.value(128),
                    initial_selected_indices=initial_selected_image_indices,
                    allow_multiple_selection=False,
                )

                @render.ui
                @reactive.event(input.show_download_print_buttons)
                def generate_ui_print_input_images():
                    req(input.show_download_print_buttons())
                    return ui.input_action_button(
                        "print_input_images",
                        "Print input images",
                        onclick=""" 
                                        var w = window.open();
                                        w.document.write(document.head.outerHTML);
                                        var printContents = document.getElementById('select_image-show_image_gallery').innerHTML;
                                        w.document.write(printContents);
                                        w.document.write('<script type="text/javascript">window.onload = function() { window.print(); w.close();};</script>');
                                        w.document.close();
                                        w.focus();
                                    """,
                        width="200px",
                    )

        with ui.nav_panel("Parameters"):
            with ui.layout_columns(col_widths=6, style="align-items: flex-end;"):
                ui.input_checkbox(
                    "ignore_blank", "Ignore blank input images", value=True
                )
                ui.input_checkbox("plot_scores", "Plot scores", value=True)
                ui.input_checkbox(
                    "show_download_print_buttons",
                    "Show download/print buttons",
                    value=False,
                )

            with ui.layout_columns(col_widths=6, style="align-items: flex-end;"):
                ui.input_numeric(
                    "cpu",
                    "# CPUs",
                    min=1,
                    max=helicon.available_cpu(),
                    value=helicon.available_cpu(),
                    step=1,
                )
                ui.input_numeric(
                    "selected_image_display_size",
                    "Selected image display size (pixel)",
                    min=32,
                    max=512,
                    value=128,
                    step=32,
                )
                with ui.tooltip():
                    ui.input_numeric(
                        "reconstruct_length_rise",
                        "Reconstruction length (rise)",
                        min=1,
                        value=3,
                        step=1,
                    )

                    "Reconstruction length as the number of rises"

                with ui.tooltip():
                    ui.input_numeric(
                        "target_apix2d",
                        "Target image pixel size (Å)",
                        min=-1,
                        value=-1,
                        step=1,
                    )

                    "Down-scale images to have this pixel size before 3D reconstruction. <=0 -> no down-scaling"

                with ui.tooltip():
                    ui.input_numeric(
                        "target_apix3d",
                        "Target voxel size (Å)",
                        min=-1,
                        value=-1,
                        step=1,
                    )

                    "Voxel size of 3D reconstruction. 0 -> Set to target image pixel size. <0 -> auto-decision."

                with ui.tooltip():
                    ui.input_numeric(
                        "sym_oversample",
                        "Helical/Csym oversampling factor",
                        min=-1,
                        value=-1,
                        step=1,
                    )

                    "Helical sym and csym oversampling factor that controls the number of equations when setting up the A matrix of the least square solution. larger values (e.g. 100) -> slower but better quality. A negative value means auto-decision"

                with ui.tooltip():
                    ui.input_numeric(
                        "lr_alpha",
                        "Weight of regularization",
                        min=0,
                        value=-1,
                        step=1e-4,
                    )

                    "Only used for elasticnet, lasso and ridge algorithms. default: 1e-4 for elasticnet/lasso and 1 for ridge"

                with ui.tooltip():
                    ui.input_numeric(
                        "lr_l1_ratio",
                        "L1 regularization ratio",
                        min=0.0,
                        max=1.0,
                        value=0.5,
                        step=0.1,
                    )

                    "The ratio (0 to 1) of L1 regularization in the L1/L2 combined regularization. Only used for the elasticnet algorithms"

            with ui.layout_columns(col_widths=12, style="align-items: flex-end;"):
                with ui.tooltip():
                    ui.input_radio_buttons(
                        "lr_algorithm",
                        "Linear regression algorithm",
                        "elasticnet lasso ridge lreg lsq".split(),
                        selected="elasticnet",
                        inline=True,
                    )

                    "Choose the algorithm that will be used to solve the linear equations"

            with ui.layout_columns(col_widths=6, style="align-items: flex-end;"):
                with ui.tooltip():
                    ui.input_radio_buttons(
                        "positive_constraint",
                        "Positive constraint",
                        {-1: "Auto", 0: "No", 1: "Yes"},
                        selected=-1,
                        inline=True,
                    )

                    "How positive constraint is used for the 3D reconstruction"

                with ui.tooltip():
                    ui.input_radio_buttons(
                        "interpolation",
                        "Interpolation method",
                        {
                            "linear": "Linear",
                            "nn": "Nearest Neighbor",
                        },
                        selected="linear",
                        inline=True,
                    )

                    "How positive constraint is used for the 3D reconstruction"

title = "Denovo3D: de novo helical indexing and 3D reconstruction"
ui.h1(title, style="font-weight: bold;")

with ui.div(
    style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0"
):
    helicon.shiny.image_select(
        id="display_selected_image",
        label=selected_images_title,
        images=selected_images_thresholded_rotated_shifted_cropped,
        image_labels=selected_images_labels,
        image_size=input.selected_image_display_size,
        justification="left",
        enable_selection=False,
    )

    with ui.layout_columns(col_widths=4):
        ui.input_slider(
            "apix", "Pixel size (Å)", min=0.0, max=10.0, value=1.0, step=0.001
        )

        ui.input_slider("threshold", "Threshold", min=0.0, max=1.0, value=0.0, step=0.1)

        ui.input_slider(
            "pre_rotation",
            "Rotation (°)",
            min=-45,
            max=45,
            value=0,
            step=0.1,
        )

        ui.input_slider(
            "shift_y",
            "Vertical shift (Å)",
            min=-100,
            max=100,
            value=0,
            step=0.1,
        )

        ui.input_slider(
            "vertical_crop_size",
            "Vertical crop (pixel)",
            min=32,
            max=256,
            value=0,
            step=2,
        )

        ui.input_slider(
            "horizontal_crop_size",
            "Horizontal crop (pixel)",
            min=32,
            max=256,
            value=0,
            step=2,
        )

with ui.div(
    style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0"
):
    with ui.card(style="height: 115px"):
        ui.card_header("Twist (°)")
        with ui.div(
            style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0"
        ):
            ui.input_numeric("twist_min", "min", value=0.1, step=0.1, width="70px")
            ui.input_numeric("twist_max", "max", value=2, step=0.1, width="70px")
            ui.input_numeric("twist_step", "step", value=0.1, step=0.1, width="70px")

    with ui.card(style="height: 115px"):
        ui.card_header("Rise (Å)")
        with ui.div(
            style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0"
        ):
            ui.input_numeric("rise_min", "min", value=4.75, step=0.1, width="70px")
            ui.input_numeric("rise_max", "max", value=4.75, step=0.1, width="70px")
            ui.input_numeric("rise_step", "step", value=0.1, step=0.01, width="70px")

    with ui.card(style="height: 115px"):
        ui.card_header("Csym")
        ui.input_numeric("csym", "n", value=1, min=1, step=1, width="70px")

    ui.input_task_button(
        "run_denovo3D", label="Reconstruct 3D Map", style="width: 115px; height: 115px;"
    )


@render_plotly
@reactive.event(reconstrunction_results)
def display_denovo3D_scores():
    req(len(reconstrunction_results()) > 1)

    scores = []
    twists = []
    for ri, result in enumerate(reconstrunction_results()):
        (
            score,
            (
                rec3d_x_proj,
                rec3d_y_proj,
                rec3d_z_sections,
                rec3d,
                reconstruct_diameter_2d_pixel,
                reconstruct_diameter_3d_pixel,
                reconstruct_length_2d_pixel,
                reconstruct_length_3d_pixel,
            ),
            (
                data,
                imageFile,
                imageIndex,
                apix3d,
                apix2d,
                twist,
                rise,
                csym,
                tilt,
                psi,
                dy,
            ),
        ) = result
        scores.append(score)
        twists.append(twist)
    sort_idx = np.argsort(twists)
    twists = np.array(twists)[sort_idx]
    scores = np.array(scores)[sort_idx]

    import plotly.express as px

    fig = px.line(x=twists, y=scores, color_discrete_sequence=["blue"], markers=True)
    fig.update_layout(xaxis_title="Twist (°)", yaxis_title="Score", showlegend=False)
    fig.update_traces(hovertemplate="Twist: %{x}°<br>Score: %{y}")

    return fig


with ui.div(
    style="max-height: 60vh; overflow-y: auto; display: flex; flex-direction: column; align-items: left; margin-bottom: 5px"
):
    helicon.shiny.image_select(
        id="display_reconstructed_projections",
        label=reactive.value("Reconstructed map projections:"),
        images=reconstructed_projection_images,
        image_labels=reconstructed_projection_labels,
        image_size=input.selected_image_display_size,
        justification="left",
        enable_selection=False,
    )


@render.ui
@reactive.event(input.show_download_print_buttons)
def generate_ui_print_reeconstructed_images():
    req(input.show_download_print_buttons())
    return ui.input_action_button(
        "print_reeconstructed_images",
        "Print reeconstructed images",
        onclick=""" 
                        var w = window.open();
                        w.document.write(document.head.outerHTML);
                        var printContents = document.getElementById('display_reconstructed_projections-show_image_gallery').innerHTML;
                        w.document.write(printContents);
                        w.document.write('<script type="text/javascript">window.onload = function() { window.print(); w.close();};</script>');
                        w.document.close();
                        w.focus();
                    """,
        width="256px",
    )


@render.download(label="Download map", filename="helicon_denovo3d_map.mrc")
@reactive.event(reconstrunction_results)
def download_denovo3D_map():
    req(len(reconstrunction_results()) == 1)
    (
        score,
        (
            rec3d_x_proj,
            rec3d_y_proj,
            rec3d_z_sections,
            rec3d,
            reconstruct_diameter_2d_pixel,
            reconstruct_diameter_3d_pixel,
            reconstruct_length_2d_pixel,
            reconstruct_length_3d_pixel,
        ),
        (
            data,
            imageFile,
            imageIndex,
            apix3d,
            apix2d,
            twist,
            rise,
            csym,
            tilt,
            psi,
            dy,
        ),
    ) = reconstrunction_results()[0]

    ny, nx = images_all()[0].shape
    apix = image_apix()
    rec3d_map = helicon.apply_helical_symmetry(
        data=rec3d[0],
        apix=apix3d,
        twist_degree=twist,
        rise_angstrom=rise,
        csym=csym,
        fraction=1.0,
        new_size=(nx, ny, ny),
        new_apix=apix,
        cpu=input.cpu(),
    ).astype(np.float32)

    import mrcfile
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mrc") as temp:
        with mrcfile.new(temp.name, overwrite=True) as mrc:
            mrc.set_data(rec3d_map)
            mrc.voxel_size = apix
        with open(temp.name, "rb") as file:
            yield file.read()


ui.HTML(
    "<i><p>Developed by the <a href='https://jiang.bio.purdue.edu/helicon' target='_blank'>Jiang Lab</a>. Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
)


@reactive.effect
@reactive.event(input.input_mode_images, input.upload_images)
def get_image_from_upload():
    req(input.input_mode_images() == "upload")
    fileinfo = input.upload_images()
    req(fileinfo)
    image_file = fileinfo[0]["datapath"]
    try:
        data, apix = compute.get_images_from_file(image_file)
    except Exception as e:
        print(e)
        data, apix = None, 0
        m = ui.modal(
            f"failed to read the uploaded 2D images from {fileinfo[0]['name']}",
            title="File upload error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
        return
    images_all.set(data)
    image_size.set(min(data.shape))
    image_apix.set(apix)


@reactive.effect
@reactive.event(input.input_mode_images, input.url_images)
def get_images_from_url():
    req(input.input_mode_images() == "url")
    req(len(input.url_images()) > 0)
    url = input.url_images()
    try:
        data, apix = compute.get_images_from_url(url)
    except Exception as e:
        print(e)
        data, apix = None, 0
        m = ui.modal(
            f"failed to download 2D images from {input.url_images()}",
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
        return
    images_all.set(data)
    image_size.set(min(data.shape))
    image_apix.set(apix)


@reactive.effect
@reactive.event(images_all, input.ignore_blank)
def get_displayed_images():
    req(len(images_all()))
    data = images_all()
    n = len(data)
    ny, nx = data[0].shape[:2]
    images = [data[i] for i in range(n)]
    image_size.set(max(images[0].shape))

    display_seq_all = np.arange(n, dtype=int)
    if input.ignore_blank():
        included = []
        for i in range(n):
            image = images[display_seq_all[i]]
            if np.max(image) > np.min(image):
                included.append(display_seq_all[i])
        images = [images[i] for i in included]
    else:
        included = display_seq_all
    image_labels = [f"{i+1}" for i in included]

    displayed_image_ids.set(included)
    displayed_images.set(images)
    displayed_image_title.set(
        f"{len(images)}/{n} images | {nx}x{ny} pixels | {image_apix()} Å/pixel"
    )
    displayed_image_labels.set(image_labels)


@reactive.effect
@reactive.event(selected_images_original)
def update_selected_image_rotation_shift_diameter():
    req(len(selected_images_original()))

    ny = int(np.max([img.shape[0] for img in selected_images_original()]))
    nx = int(np.max([img.shape[1] for img in selected_images_original()]))
    tmp = np.array(
        [
            helicon.estimate_helix_rotation_center_diameter(img)
            for img in selected_images_original()
        ]
    )
    rotation = np.mean(tmp[:, 0])
    shift_y = np.mean(tmp[:, 1]) * input.apix()
    diameter = np.max(tmp[:, 2])
    crop_size = int(diameter * 3) // 4 * 4
    min_val = float(np.min([np.min(img) for img in selected_images_original()]))
    max_val = float(np.max([np.max(img) for img in selected_images_original()]))
    step_val = (max_val - min_val) / 100

    selected_image_diameter.set(diameter)
    ui.update_numeric("apix", value=round(image_apix(), 4), max=round(image_apix() * 2))
    ui.update_numeric("pre_rotation", value=round(rotation, 1))
    ui.update_numeric(
        "shift_y",
        value=shift_y,
        min=-crop_size * input.apix() // 2,
        max=crop_size * input.apix() // 2,
    )
    ui.update_numeric(
        "vertical_crop_size",
        value=max(32, crop_size),
        min=max(32, int(diameter) // 2 * 2),
        max=ny,
    )
    ui.update_numeric("horizontal_crop_size", value=nx, min=32, max=nx)
    ui.update_numeric(
        "threshold",
        value=0,
        min=round(min_val, 3),
        max=round(max_val, 3),
        step=round(step_val, 3),
    )


@reactive.effect
@reactive.event(input.select_image)
def update_selecte_images_orignal():
    selected_images_original.set([displayed_images()[i] for i in input.select_image()])
    selected_images_labels.set(
        [displayed_image_labels()[i] for i in input.select_image()]
    )
    reconstrunction_results.set([])


@reactive.effect
@reactive.event(selected_images_original, input.threshold)
def threshold_selected_images():
    req(len(selected_images_original()))
    tmp = [
        helicon.threshold_data(img, thresh_value=input.threshold())
        for img in selected_images_original()
    ]
    selected_images_thresholded.set(tmp)


@reactive.effect
@reactive.event(selected_images_thresholded, input.pre_rotation, input.shift_y)
def transform_selected_images():
    req(len(selected_images_thresholded()))
    if input.pre_rotation != 0 or input.shift_y != 0:
        rotated = []
        for img in selected_images_thresholded():
            rotated.append(
                helicon.transform_image(
                    image=img,
                    rotation=input.pre_rotation(),
                    post_translation=(input.shift_y() / input.apix(), 0),
                )
            )
    else:
        rotated = selected_images_original()
    selected_images_thresholded_rotated_shifted.set(rotated)


@reactive.effect
@reactive.event(
    selected_images_thresholded_rotated_shifted,
    input.vertical_crop_size,
    input.horizontal_crop_size,
)
def crop_selected_images():
    req(len(selected_images_thresholded_rotated_shifted()))
    req(input.vertical_crop_size() > 0 or input.horizontal_crop_size)
    crop_ny = max(32, int(input.vertical_crop_size()))
    crop_nx = max(32, int(input.horizontal_crop_size()))
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


@reactive.effect
@reactive.event(input.run_denovo3D)
def run_denovo3D_reconstruction():
    data = selected_images_thresholded_rotated_shifted_cropped()
    req(len(data) > 0)

    data = data[0]
    ny, nx = data.shape
    tube_length = nx * input.apix()

    imageFile = selected_images_title()
    imageIndex = selected_images_labels()[0]

    logger = helicon.get_logger(
        logfile="helicon.denovo3D.log",
        verbose=1,
    )

    if input.twist_min() < input.twist_max():
        twists = np.arange(input.twist_min(), input.twist_max(), input.twist_step())
    else:
        twists = [input.twist_min()]
    if input.rise_min() < input.rise_max():
        rises = np.arange(input.rise_min(), input.rise_max(), input.rise_step())
    else:
        rises = [input.rise_min()]

    import itertools

    tr_pairs = list(itertools.product(twists, rises))
    return_3d = len(tr_pairs) == 1

    tasks = []
    for ti, t in enumerate(tr_pairs):
        twist, rise = t
        twist = np.round(helicon.set_to_periodic_range(twist, min=-180, max=180), 6)
        csym = input.csym()
        apix = input.apix()
        tilt = 0
        tilt_min = 0
        tilt_max = 0
        psi = 0
        dy = 0
        denoise = ""
        low_pass = -1
        transpose = 0
        horizontalize = 0
        target_apix2d = apix
        target_apix3d = input.target_apix3d()
        thresh_fraction = -1
        positive_constraint = int(input.positive_constraint())
        tube_diameter = ny * apix
        tube_diameter_inner = 0.0
        tube_length = nx * apix
        reconstruct_length = input.reconstruct_length_rise() * rise
        sym_oversample = input.sym_oversample()
        interpolation = input.interpolation()
        fsc_test = 0
        verbose = 2
        cpu = input.cpu()

        algorithm = dict(model=input.lr_algorithm(), l1_ratio=input.lr_l1_ratio())
        if input.lr_alpha() >= 0:
            algorithm["alpha"] = input.lr_alpha()

        if abs(twist) < 0.01:
            logger.warning(
                f"WARNING: (twist={round(twist, 3)}, rise={round(rise, 3)}) will be ignored due to very small twist value (twist={round(twist, 3)}°)"
            )
            continue
        if abs(rise) < 0.01:
            logger.warning(
                f"WARNING: (twist={round(twist, 3)}, rise={round(rise, 3)}) will be ignored due to very small rise value (rise={round(rise, 3)}Å)"
            )
            continue
        if abs(rise) >= tube_length / 2:
            logger.warning(
                f"WARNING: (twist={round(twist, 3)}, rise={round(rise, 3)}) will be ignored due to very large rise value (rise={round(rise, 3)}Å)"
            )
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
                dy,
                apix,
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
        )

    if len(tasks) < 1:
        logger.warning("Nothing to do. I will quit")
        return

    with ui.Progress(min=1, max=len(tasks)) as p:
        p.set(message="Calculation in progress", detail="This may take a while ...")

        from tqdm import tqdm
        from joblib import Parallel, delayed
        from time import time

        t0 = time()
        results = []
        for ti, task in enumerate(tasks):
            result = compute.process_one_task(*task)
            t1 = time()
            results.append(result)
            remaining = (len(tasks) - ti - 1) * (t1 - t0)
            p.set(
                ti + 1,
                message=f"Completed {ti+1}/{len(tasks)}",
                detail=f"{helicon.timedelta2string(remaining)} remaining",
            )
            t0 = time()

    results_none = [res for res in results if res is None]
    if len(results_none):
        logger.info(
            f"{len(results_none)}/{len(results)} results are None and thus discarded"
        )
        results = [res for res in results if res is not None]

    results.sort(key=lambda x: x[0], reverse=True)  # sort from high to low scores
    reconstrunction_results.set(results)


@reactive.effect
@reactive.event(reconstrunction_results)
def display_denovo3D_projections():
    reconstructed_projection_labels.set([])
    reconstructed_projection_images.set([])
    req(len(reconstrunction_results()))
    labels = []
    images = []
    for ri, result in enumerate(reconstrunction_results()):
        (
            score,
            (
                rec3d_x_proj,
                rec3d_y_proj,
                rec3d_z_sections,
                rec3d,
                reconstruct_diameter_2d_pixel,
                reconstruct_diameter_3d_pixel,
                reconstruct_length_2d_pixel,
                reconstruct_length_3d_pixel,
            ),
            (
                data,
                imageFile,
                imageIndex,
                apix3d,
                apix2d,
                twist,
                rise,
                csym,
                tilt,
                psi,
                dy,
            ),
        ) = result

        query_image = selected_images_thresholded_rotated_shifted_cropped()[0]
        query_image_padded = helicon.pad_to_size(query_image, shape=rec3d_x_proj.shape)

        label_x = f"{ri+1}: X|score={score:.4f}|pitch={int(round(rise*360/abs(twist))):,}Å|twist={round(twist,3)}°|rise={round(rise,6)}Å"
        labels += [f"Input image: {selected_images_labels()[0]}", label_x, "Z"]
        images += [query_image_padded, rec3d_x_proj, rec3d_z_sections]

    reconstructed_projection_labels.set(labels)
    reconstructed_projection_images.set(images)


@render.ui
@reactive.event(reconstrunction_results)
def toggle_map_download_button():
    if len(reconstrunction_results()) == 1:
        ret = ui.tags.style("#download_denovo3D_map {visibility: visible;}")
    else:
        ret = ui.tags.style("#download_denovo3D_map {visibility: hidden;}")
    return ret
