from pathlib import Path
import numpy as np
import pandas as pd

from shinywidgets import render_plotly

import shiny
from shiny import reactive, req
from shiny.express import input, ui, render

import helicon

from . import compute


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

t_ui_counter = reactive.value(0)
selected_images_rotated_shifted = reactive.value([])
transformed_images_displayed = reactive.value([])
transformed_images_title = reactive.value("Transformed selected images:")
transformed_images_labels = reactive.value([])
transformed_images_links = reactive.value([])
transformed_images_vertical_display_size = reactive.value(256)
transformed_images_x_offsets = reactive.value([])

stitched_image_displayed = reactive.value([])
stitched_image_title = reactive.value("Stitched image:")
stitched_image_labels = reactive.value([])
stitched_image_links = reactive.value([])
stitched_image_vertical_display_size = reactive.value(128)

initial_image = reactive.value([])
display_initial_image_value = reactive.value([''])

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

                @render.ui
                @reactive.event(input.show_emdb_input_mode)
                def create_input_modes_ui():
                    choices = ["upload", "url"]
                    if input.show_emdb_input_mode():
                        choices.append("emdb")
                    return ui.input_radio_buttons(
                        "input_mode_images",
                        "How to obtain the input images:",
                        choices=choices,
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
                                "Download URL for a RELION or cryoSPARC 2D class mrc(s) file",
                                value=urls[url_key][0],
                            )
                        )
                    elif input.input_mode_images() == "emdb":
                        ret.append(
                            ui.div(
                                ui.input_text(
                                    "emdb_id",
                                    "Specify an amyloid structure EMDB ID",
                                    value="EMD-14046",
                                    width="calc(100% - 110px)",
                                ),
                                ui.input_action_button(
                                    "randomize_emdb_id",
                                    "Randomize",
                                    style="width: 100px; height: 30px; margin-bottom: 14px; display: flex; align-items: center; justify-content: center;",
                                ),
                                style="""
                                    display: flex;
                                    flex-wrap: wrap;
                                    width: 100%;
                                    justify-content: space-between;
                                    align-items: flex-end;
                                    gap: 10px;
                                """,
                            )
                        )
                    return ret

                @render.ui
                @reactive.event(input_data)
                def display_emdb_info():
                    req(input_data() is not None)
                    req(len(input_data().data))
                    req(input_data().emdb_id)
                    emdb = helicon.dataset.EMDB()
                    emd_id_num = input.emdb_id().split("-")[-1].split("_")[-1]
                    req(emd_id_num in emdb.emd_ids)
                    emd_id = f"EMD-{emd_id_num}"
                    info = emdb.get_info(emd_id)
                    nz, ny, nx = input_data().data.shape
                    apix = input_data().apix
                    s = f"<p><a href='https://www.ebi.ac.uk/emdb/{emd_id}' target='_blank'>{emd_id}</a>"
                    s += f": {info.title}"
                    s += f"<br>{nx}x{ny}x{nz}|{apix}Å/pixel|resolution={info.resolution}Å|twist={info.twist}°|pitch={info.pitch:,}Å|rise={info.rise}Å|{info.csym}"
                    s += "</p>"
                    ret = ui.HTML(s)
                    return ret

            helicon.shiny.image_select(
                id="map_xyz_projections",
                label=reactive.value("XYZ Projections"),
                images=map_xyz_projections,
                image_labels=reactive.value("X Y Z".split()),
                image_size=reactive.value(128),
                enable_selection=False,
                style="margin-bottom: 20px;",
            )

            @render.ui
            @reactive.event(input_data)
            def generate_ui_symmetrize_projection():
                req(input_data().is_3d)
                req(input_data() is not None)
                req(len(input_data().data))
                twist = 0
                pitch = np.nan
                rise = 0
                csym = 1
                if input_data().emdb_id:
                    emdb = helicon.dataset.EMDB()
                    emd_id_num = input_data().emdb_id.split("-")[-1].split("_")[-1]
                    if emd_id_num in emdb.emd_ids:
                        emd_id = f"EMD-{emd_id_num}"
                        info = emdb.get_info(emd_id)
                        twist = info.twist
                        rise = info.rise
                        csym = int(info.csym[1:])
                        pitch = info.pitch
                width = (
                    int((input_data().data.shape[2] * input_data().apix) / 5) // 4 * 4
                )
                length = (
                    int(round(0.5 * pitch / 5)) // 4 * 4
                    if not np.isnan(pitch)
                    else width * 2
                )

                ret = ui.div(
                    ui.tags.hr(),
                    ui.input_numeric(
                        "input_twist", "Twist (°)", value=twist, step=0.1, width="140px"
                    ),
                    ui.input_numeric(
                        "input_rise", "Rise (Å)", value=rise, step=0.1, width="140px"
                    ),
                    ui.input_numeric(
                        "input_csym", "Csym", value=csym, min=1, step=1, width="140px"
                    ),
                    ui.input_numeric(
                        "input_apix",
                        "Input voxel size (Å)",
                        value=input_data().apix,
                        min=0.1,
                        step=0.1,
                        width="140px",
                    ),
                    ui.input_numeric(
                        "output_apix",
                        "Output pixel size (Å)",
                        value=5,
                        min=0.1,
                        step=0.1,
                        width="140px",
                    ),
                    ui.input_numeric(
                        "output_axial_rotation",
                        "Axial rotation (°)",
                        value=0,
                        min=-180,
                        max=180,
                        step=1,
                        width="140px",
                    ),
                    ui.input_numeric(
                        "output_width",
                        "Output width (pixels)",
                        value=width,
                        min=32,
                        step=16,
                        width="140px",
                    ),
                    ui.input_numeric(
                        "output_length",
                        "Output length (pixels)",
                        value=length,
                        min=32,
                        step=16,
                        width="140px",
                    ),
                    ui.input_numeric(
                        "output_tilt",
                        "Tilt out of plane (°)",
                        value=0,
                        min=-90,
                        max=90,
                        step=1,
                        width="140px",
                    ),
                    style="display: flex; flex-wrap: wrap; flex-direction: row; gap: 4px; align-items: flex-end; justify-content: center;",
                )
                ret = ui.div(
                    ret,
                    ui.input_task_button(
                        "symmetrization_projection",
                        "Generate proejction",
                        style="margin-bottom: 10px;",
                    ),
                    style="display: flex; flex-direction: column; justify-content: center;",
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
                    allow_multiple_selection=True,
                )

                @render.download(
                    label="Download symmetrized input map",
                    filename="helicon_denovo3d_input_map.mrc",
                )
                @reactive.event(map_symmetrized)
                def download_denovo3D_input_map():
                    req(map_symmetrized() is not None)

                    import mrcfile
                    import tempfile

                    with tempfile.NamedTemporaryFile(suffix=".mrc") as temp:
                        with mrcfile.new(temp.name, overwrite=True) as mrc:
                            mrc.set_data(map_symmetrized())
                            mrc.voxel_size = input.output_apix()
                        with open(temp.name, "rb") as file:
                            yield file.read()

                @render.ui
                @reactive.event(input.show_download_print_buttons, displayed_images)
                def generate_ui_print_input_images():
                    req(input.show_download_print_buttons())
                    req(len(displayed_images()))
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
                        width="270px",
                    )

        with ui.nav_panel("Parameters"):
            with ui.layout_columns(col_widths=6, style="align-items: flex-end;"):
                ui.input_checkbox(
                    "show_emdb_input_mode", "Show EMDB input mode", value=False
                )
                ui.input_checkbox("is_3d", "The input is a 3D map", value=False)
                ui.input_checkbox(
                    "ignore_blank", "Ignore blank input images", value=True
                )
                ui.input_checkbox("plot_scores", "Plot scores", value=True)
                ui.input_checkbox("image_stitching", "Image_stitching", value=False)
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
                        value=5,
                        step=1,
                    )

                    "Down-scale images to have this pixel size before 3D reconstruction. <=0 -> no down-scaling"

                with ui.tooltip():
                    ui.input_numeric(
                        "target_apix3d",
                        "Target voxel size (Å)",
                        min=-1,
                        value=5,
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
    
    @shiny.render.ui
    @reactive.event(selected_images_rotated_shifted, input.image_stitching, ignore_init=False)
    def generate_image_gallery_mutiple():
        req(len(displayed_images()))
        req(0 <= min(input.select_image()))
        req(max(input.select_image()) < len(displayed_images()))
        n_images_selected = len(selected_images_original())

        # Return None early if condition isn't met
        if n_images_selected == 1:
            return None
        

        # Only proceed with accessing other reactive values if we have exactly 1 image
        return helicon.shiny.image_gallery(
                    id="display_selected_image",
                    label=selected_images_title,
                    images=selected_images_rotated_shifted,
                    image_labels=selected_images_labels,
                    image_size=stitched_image_vertical_display_size,
                    justification="left",
                    display_dashed_line=True,
                    enable_selection=False
                )
    
    
    @shiny.render.ui
    @reactive.event(selected_images_original, ignore_init=False)
    def generate_image_transformation_multiple():
        req(len(displayed_images()))
        req(0 <= min(input.select_image()))
        req(max(input.select_image()) < len(displayed_images()))
        n_images_selected = len(selected_images_original())

        if n_images_selected == 1:
            return None
        else:

            dim = len(selected_images_original()[0])
            shift_scale = int(0.9*dim)
            # Create main container
            container = ui.div(
                style="display: flex; flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 0"
            )

            # Add transformation controls for each image
            for i, label in enumerate(selected_images_labels()):
                curr_t_ui_counter = t_ui_counter() + i

                # Add transformation UI group
                container.append(
                    shiny.ui.row(
                        transformation_ui_group(f"t_ui_group_{curr_t_ui_counter}",shift_scale=shift_scale)
                    )
                )

                # Setup reactive bindings
                id_rotation = f"t_ui_group_{curr_t_ui_counter}_pre_rotation"
                id_x_shift = f"t_ui_group_{curr_t_ui_counter}_shift_x"
                id_y_shift = f"t_ui_group_{curr_t_ui_counter}_shift_y"


                @reactive.effect
                @reactive.event(input.select_image, input.image_stitching)
                def update_selecte_images_orignal():
                    selected_images_rotated_shifted.set(
                        [displayed_images()[i] for i in input.select_image()]
                    )
                    transformed_images_x_offsets.set(
                        np.zeros(len(input.select_image()))
                    )

                @reactive.effect
                @reactive.event(input[id_rotation], input[id_y_shift])
                def transform_selected_images(i=i, id_rotation=id_rotation, id_y_shift=id_y_shift):
                    req(len(selected_images_original()))

                    rotated = selected_images_rotated_shifted().copy()
                    if input[id_rotation]() != 0 or input[id_y_shift]() != 0:
                        rotated[i] = helicon.transform_image(
                            image=selected_images_original()[i].copy(),
                            rotation=input[id_rotation](),
                            post_translation=(input[id_y_shift](), 0)
                        )
                    selected_images_rotated_shifted.set(rotated)
                    print(f"rot shift {i} done")

                @reactive.effect
                @reactive.event(selected_images_rotated_shifted, input[id_x_shift])
                def update_transformed_images_displayed(x_shift_i=i, id_x_shift=id_x_shift):
                    req(len(selected_images_rotated_shifted()))

                    

                    images_displayed = []
                    images_displayed_labels = []
                    images_displayed_links = []

                    curr_x_offsets = transformed_images_x_offsets().copy()
                    ny, nx = np.shape(selected_images_rotated_shifted()[0])

                    # Initialize sum and count arrays for averaging
                    total_width = nx * len(selected_images_rotated_shifted())
                    sum_image = np.zeros((ny, total_width), dtype=np.float64)  # Use float for precision
                    count_image = np.zeros((ny, total_width), dtype=np.uint8)   # Track overlaps

                    for img_i, transformed_img in enumerate(selected_images_rotated_shifted()):
                        if img_i == x_shift_i:
                            shift = input[id_x_shift]()  # Get the user-defined shift value
                            start_col = nx * img_i + shift  # Shifted start column
                            curr_x_offsets[x_shift_i] = shift
                        else:
                            start_col = nx * img_i  # Default start column

                        # Calculate the region where the image will be placed
                        end_col = start_col + nx

                        # Clip to canvas boundaries to avoid out-of-bounds errors
                        canvas_start = max(start_col, 0)
                        canvas_end = min(end_col, total_width)

                        # Adjust the image slice if part of it is outside the canvas
                        img_start = max(0, -start_col)  # Offset if shifted left beyond canvas
                        img_end = img_start + (canvas_end - canvas_start)

                        # Extract the valid part of the image to place
                        img_slice = transformed_img[:, img_start:img_end].astype(np.float64)

                        # Add to sum and increment count for averaging
                        sum_image[:, canvas_start:canvas_end] += img_slice
                        count_image[:, canvas_start:canvas_end] += 1

                    # Compute the averaged image (avoid division by zero)
                    image_work = np.divide(
                        sum_image, 
                        count_image, 
                        where=(count_image > 0), 
                        out=np.zeros_like(sum_image)
                    )

                    images_displayed.append(image_work)
                    images_displayed_labels.append("Combined images")
                    images_displayed_links.append("")

                    

                    transformed_images_displayed.set(images_displayed)
                    transformed_images_labels.set(images_displayed_labels)
                    transformed_images_links.set(images_displayed_links)
                    transformed_images_x_offsets.set(curr_x_offsets)

            # Add stitch button
            container.append(
                ui.input_task_button(
                    "perform_stitching",
                    label="Stitch Images",
                    style="width: 100%; margin-top: 10px;"
                )
            )

            # Update counter
            t_ui_counter.set(t_ui_counter() + n_images_selected)

            return container


@shiny.render.ui
@reactive.event(transformed_images_displayed, input.image_stitching, ignore_init=False)
def image_stitching_transformed():
    req(len(displayed_images()))
    req(0 <= min(input.select_image()))
    req(max(input.select_image()) < len(displayed_images()))
    n_images_selected = len(selected_images_original())
    # Return None early if condition isn't met
    if n_images_selected == 1:
        return None
    # Only proceed with accessing other reactive values if we have exactly 1 image
    return helicon.shiny.image_gallery(
                    id="display_transformed_images",
                    label=transformed_images_title,
                    images=transformed_images_displayed,
                    image_labels=transformed_images_labels,
                    image_links=transformed_images_links,
                    image_size=transformed_images_vertical_display_size,
                    justification="left",
                    display_dashed_line=True,
                    enable_selection=False
                )
@shiny.render.ui
@reactive.event(stitched_image_displayed, selected_images_original, input.image_stitching, ignore_init=False)
def display_stitched_image():
    req(len(displayed_images()))
    req(0 <= min(input.select_image()))
    req(max(input.select_image()) < len(displayed_images()))
    n_images_selected = len(selected_images_original())
    if n_images_selected == 1:
        return None
    return helicon.shiny.image_gallery(
            id="display_stitched_image",
            label=stitched_image_title,
            images=stitched_image_displayed,
            image_labels=stitched_image_labels,
            image_links=stitched_image_links,
            image_size=stitched_image_vertical_display_size,
            display_dashed_line=True,
            justification="left",
            enable_selection=False
        )
with ui.div(
    style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0"
):
    @shiny.render.ui
    @reactive.event(selected_images_thresholded_rotated_shifted_cropped, initial_image, ignore_init=False)
    def generate_image_gallery_single():
        req(len(initial_image()))
        req(0 <= min(input.select_image()))
        req(max(input.select_image()) < len(displayed_images()))
        n_images_selected = len(initial_image())
        # To check whether there is image for transformation and display
        if n_images_selected == 1:
            return helicon.shiny.image_gallery(
                id="display_selected_image",
                label=selected_images_title,
                images=selected_images_thresholded_rotated_shifted_cropped,
                image_labels=display_initial_image_value,
                image_size=input.selected_image_display_size,
                justification="left",
                enable_selection=False,
                display_dashed_line=True,
            )
        else:
            return None


    @shiny.render.ui
    @reactive.event(initial_image, ignore_init=False)
    def generate_image_transformation_single():
        req(len(initial_image()))
        req(0 <= min(input.select_image()))
        req(max(input.select_image()) < len(displayed_images()))
        n_images_selected = len(initial_image())
        # To check whether there is image for transformation and display
        if n_images_selected == 1:
            return ui.div(            

                # Transformation controls
                transformation_ui_single(),
                style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0"
            )
        else:
            return None

with ui.div(
    style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0"
):
    with ui.card(style="height: 115px"):
        ui.card_header("Twist (°)")
        with ui.div(
            style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0"
        ):
            ui.input_numeric("twist_min", "min", value=1.2, step=0.1, width="70px")
            ui.input_numeric("twist_max", "max", value=1.2, step=0.1, width="70px")
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

    #output score in windows machine
    #np.save('/mnt/f/script/helicon/test/twist.npy', twists)
    #np.save('/mnt/f/script/helicon/test/score.npy', scores)
    #print('the score is saved')

    import plotly.express as px

    fig = px.line(x=twists, y=scores, color_discrete_sequence=["blue"], markers=True)
    fig.update_layout(xaxis_title="Twist (°)", yaxis_title="Score", showlegend=False)
    fig.update_traces(hovertemplate="Twist: %{x}°<br>Score: %{y}")

    return fig



@render.ui
@reactive.event(reconstructed_projection_images, reconstructed_projection_labels)
def image_label_pairs():

    req(len(reconstructed_projection_images()))
    img_list = reconstructed_projection_images()  # Your reactive images
    label_list = reconstructed_projection_labels()  # Your reactive labels

    from helicon import encode_numpy, encode_PIL_Image
    # Create pairs of labels and images
    pairs = []
    for img, label_value in zip(img_list, label_list):
        height, width = img.shape
        img = encode_numpy(img)
        label_value = str(label_value)
        pairs.extend([
            ui.div(
                {"class": "label-row", "style": "margin: 10px 0;"},
                ui.h4(label_value)
            ),
            # Image row
            ui.div(
                {"class": "image-row", "style":"max-height: 100vh; overflow-y: auto; display: flex; flex-direction: column; align-items: left; margin-bottom: 5px"},
                ui.img({
                    "src": img,
                    #"width": str(width),  # Original width
                    #"height": str(height),
                    "style": "max-width: 100%; height: auto;"
                })
            )
        ])
    
    return ui.div(pairs)

#with ui.div(
#    style="max-height: 100vh; overflow-y: auto; display: flex; flex-direction: column; align-items: left; margin-bottom: 5px"
#):
#    helicon.shiny.image_select(
#        id="display_reconstructed_projections",
#        label=reactive.value("Reconstructed map projections:"),
#        images=reconstructed_projection_images,
#        image_labels=reconstructed_projection_labels,
#        image_size=input.selected_image_display_size,
#        justification="left",
#        enable_selection=False,
#    )


@render.ui
@reactive.event(input.show_download_print_buttons, reconstructed_projection_images)
def generate_ui_print_reeconstructed_images():
    req(input.show_download_print_buttons())
    req(len(reconstructed_projection_images()))
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


@render.download(
    label="Download reconstructed map",
    filename="helicon_denovo3d_reconstructed_map.mrc",
)
@reactive.event(reconstrunction_results)
def download_denovo3D_output_map():
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

    ny, nx = input_data().data[0].shape
    apix = input_data().apix
    #ny, nx = 200,200
    #apix = 5
    print(apix, ny, nx)
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


# below are the code of the reactive function


def transformation_ui_single():
    return shiny.ui.card(shiny.ui.layout_columns(
         ui.input_checkbox("img_transpose", "Transpose", False),
         
         ui.input_checkbox("img_flip", "Flip", False),
         
         ui.input_checkbox("img_negate", "Invert contrast", False),
         
         ui.input_slider(
                "pre_rotation",
                "Rotation (°)",
                min=-180,
                max=180,
                value=0,
                step=0.1,
            ),

            ui.input_slider("threshold", "Threshold", min=0.0, max=1.0, value=0.0, step=0.1),

            ui.input_slider(
                "apix", "Pixel size (Å)", min=0.0, max=10.0, value=1.0, step=0.001
            ),

            ui.input_slider(
                "shift_y",
                "Vertical shift (Å)",
                min=-100,
                max=100,
                value=0,
                step=0.1,
            ),

            ui.input_slider(
                "vertical_crop_size",
                "Vertical crop (pixel)",
                min=32,
                max=256,
                value=32,
                step=2,
            ),

            ui.input_slider(
                "horizontal_crop_size",
                "Horizontal crop (pixel)",
                min=32,
                max=256,
                value=256,
                step=2,
            ),
        col_widths=4),
            ui.input_task_button(
            "auto_transform", label="Auto Transform", style="width: 200px; height: 40px;"
            ),
    id=f"single_card_ui")


def transformation_ui_group(prefix, shift_scale = 100):
    return shiny.ui.card(shiny.ui.layout_columns(
        ui.input_slider(
            prefix+"_pre_rotation",
            "Rotation (°)",
            min=-45,
            max=45,
            value=0,
            step=0.1,
        ),       
        ui.input_slider(
            prefix+"_shift_x",
            "Horizontal shift (pixel)",
            min=-shift_scale,
            max=shift_scale,
            value=0,
            step=1,
        ),
        ui.input_slider(
            prefix+"_shift_y",
            "Vertical shift (pixel)",
            min=-100,
            max=100,
            value=0,
            step=1,
        ),
        # ui.input_slider(
            # prefix+"_vertical_crop_size",
            # "Vertical crop (pixel)",
            # min=32,
            # max=256,
            # value=0,
            # step=2,
        # ),
        col_widths=4),id=f"{prefix}_card")


@reactive.effect
@reactive.event(input.input_mode_images)
def reset_input_data_ui():
    input_data.set(None)
    ui.update_checkbox("is_3d", value=False)
    map_symmetrized.set(None)
    map_xyz_projections.set(None)
    selected_images_thresholded_rotated_shifted_cropped.set(None)


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
        import traceback

        traceback.print_exc()
        print(e)
        data, apix = None, 1
        m = ui.modal(
            f"failed to read the uploaded 2D images from {fileinfo[0]['name']}",
            title="File upload error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
        return

    emdb_id = helicon.get_emdb_id(fileinfo[0]["name"])
    is_3d = emdb_id or helicon.is_3d(data)
    is_amyloid = helicon.is_amyloid(emdb_id)
    d = helicon.DotDict(
        data=data, apix=apix, emdb_id=emdb_id, is_3d=is_3d, is_amyloid=is_amyloid
    )
    input_data.set(d)
    ui.update_checkbox("is_3d", value=is_3d)


@reactive.effect
@reactive.event(input.input_mode_images, input.url_images)
def get_images_from_url():
    req(input.input_mode_images() == "url")
    req(len(input.url_images()) > 0)
    url = input.url_images()
    try:
        data, apix = compute.get_images_from_url(url)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(e)
        data, apix = None, 1
        m = ui.modal(
            f"failed to download 2D images from {input.url_images()}",
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
        return

    emdb_id = helicon.get_emdb_id(url)
    is_3d = emdb_id or helicon.is_3d(data)
    is_amyloid = helicon.is_amyloid(emdb_id)
    d = helicon.DotDict(
        data=data, apix=apix, emdb_id=emdb_id, is_3d=is_3d, is_amyloid=is_amyloid
    )
    input_data.set(d)
    ui.update_checkbox("is_3d", value=is_3d)


@reactive.effect
@reactive.event(input.randomize_emdb_id)
def randomize_emdb_id():
    emdb = helicon.dataset.EMDB()
    ids = emdb.amyloid_atlas_ids()
    import random

    emdb_id = f"EMD-{random.choice(ids)}"
    ui.update_text("emdb_id", value=emdb_id)


@reactive.effect
@reactive.event(input.input_mode_images, input.emdb_id)
def get_images_from_emdb():
    req(input.input_mode_images() == "emdb")
    emdb_id = input.emdb_id()
    req(len(emdb_id) > 0)
    try:
        data, apix = compute.get_images_from_emdb(emdb_id=emdb_id)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(e)
        data, apix = None, 1
        m = ui.modal(
            f"failed to obtain {emdb_id} map from EMDB",
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
        return

    is_amyloid = helicon.is_amyloid(emdb_id)
    d = helicon.DotDict(
        data=data, apix=apix, emdb_id=emdb_id, is_3d=True, is_amyloid=is_amyloid
    )
    input_data.set(d)
    ui.update_checkbox("is_3d", value=True)



@reactive.effect
@reactive.event(input_data)
def update_all_images_from_2d_input_data():
    req(input_data())
    req(len(input_data().data))
    if input_data().is_3d:
        all_images.set(None)
    else:
        d = helicon.DotDict(data=input_data().data, apix=input_data().apix)
        all_images.set(d)


@reactive.effect
@reactive.event(input.is_3d)
def update_input_data_is_3d():
    req(input_data())
    d = input_data()
    d.is_3d = input.is_3d()
    d2 = helicon.DotDict(d)
    input_data.set(d2)


@reactive.effect
@reactive.event(input_data)
def get_xyz_projections():
    req(input_data())
    req(len(input_data().data))
    if input_data().is_3d:
        proj_xyz = compute.generate_xyz_projections(
            input_data().data,
            is_amyloid=input_data().is_amyloid,
            apix=input_data().apix,
        )
        map_xyz_projections.set(proj_xyz)
    else:
        map_xyz_projections.set(None)


@reactive.effect
@reactive.event(input.symmetrization_projection)
def update_all_images_from_3d_input_data():
    req(input_data())
    req(len(input_data().data))
    req(input_data().is_3d)
    m = compute.symmetrize_transform_map(
        data=input_data().data,
        apix=input.input_apix(),
        twist_degree=input.input_twist(),
        rise_angstrom=input.input_rise(),
        csym=input.input_csym(),
        new_size=(input.output_length(), input.output_width(), input.output_width()),
        new_apix=input.output_apix(),
        axial_rotation=input.output_axial_rotation(),
        tilt=input.output_tilt(),
    )

    #import mrcfile
    #with mrcfile.new('/mnt/f/Pili_input.mrc', overwrite=True) as mrc:
    #                        mrc.set_data(m)
    #                        mrc.voxel_size = input.output_apix()

    map_symmetrized.set(m)

    proj = np.transpose(m.sum(axis=-1))[:, ::-1]
    proj = proj[np.newaxis, :, :]
    d = helicon.DotDict(data=proj, apix=input.output_apix())
    all_images.set(d)


@reactive.effect
@reactive.event(all_images, input.ignore_blank)
def get_displayed_images():
    if all_images() is None:
        displayed_images.set([])
        return
    req(len(all_images().data))
    data = all_images().data
    apix = all_images().apix
    n = len(data)
    if n:
        ny, nx = data[0].shape[:2]
        images = [data[i] for i in range(n)]

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
        title = f"{len(images)}/{n} images|{nx}x{ny}|{apix}Å/pixel|length={round(nx*apix):,}Å"
        if input_data().emdb_id:
            emdb = helicon.dataset.EMDB()
            info = emdb.get_info(input_data().emdb_id)
            if not np.isnan(info.pitch):
                title += f"({round(nx*apix/info.pitch,2)}pitch)"

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
@reactive.event(input.select_image, displayed_images)
def update_selecte_images_orignal():
    req(len(displayed_images()))
    req(0 <= min(input.select_image()))
    req(max(input.select_image()) < len(displayed_images()))
    images = [displayed_images()[i] for i in input.select_image()]
    selected_images_original.set(images)
    selected_images_labels.set(
        [displayed_image_labels()[i] for i in input.select_image()]
    )
    reconstrunction_results.set([])

    


@reactive.effect
@reactive.event(selected_images_original)
def set_initial_image():
    req(len(selected_images_original()))
    n_images_selected = len(selected_images_original())
    
    # Return None early if condition isn't met
    if n_images_selected == 1:
        initial_image.set(selected_images_original()) 
    else:

        initial_image.set([]) 

        @reactive.effect
        @reactive.event(stitched_image_displayed, input.threshold)
        def set_stitched_image_initial():
            req(len(stitched_image_displayed()))

            
            
            initial_image.set(stitched_image_displayed()) 

@reactive.effect
@reactive.event(initial_image, input.img_negate)
def update_threshold_scale():
    req(len(initial_image()))
    images = initial_image()
    if input.img_negate():
        images = [-img for img in images]
    min_val = float(np.min([np.min(img) for img in images]))
    max_val = float(np.max([np.max(img) for img in images]))
    step_val = (max_val - min_val) / 100
    ui.update_numeric(
        "threshold",
        value=0,
        min=round(min_val, 3),
        max=round(max_val, 3),
        step=round(step_val, 3),
    )

@reactive.effect
@reactive.event(initial_image, input.threshold, input.img_transpose, input.img_flip)
def threshold_selected_images():
    req(len(initial_image()))

    images = initial_image()

    if input.img_negate():
        tmp = [
            helicon.threshold_data(-img, thresh_value=input.threshold())
            for img in images
        ]
    else:
        tmp = [
            helicon.threshold_data(img, thresh_value=input.threshold())
            for img in images
        ]
    
    if input.img_transpose():
        tmp = [
            np.transpose(img)
            for img in tmp
        ]

    if input.img_flip():
        tmp = [
            np.fliplr(img)
            for img in tmp
        ]
    
    selected_images_thresholded.set(tmp)



def estimate_helix_rotation_center_diameter(
    data, estimate_rotation=True, estimate_center=True, threshold=0
):
    """
    Returns:
        rotation (float): The rotation (degrees) needed to rotate the helix to horizontal direction.
        shift_y (float): The post-rotation vertical shift (pixels) needed to shift the helix to the box center in vertical direction.
        diameter (int): The estimated diameter (pixels) of the helix.
    """
    from skimage.measure import label, regionprops
    from skimage.morphology import closing
    import helicon

    if estimate_rotation:
        bw = closing(data > threshold, mode="ignore")
        label_image = label(bw)
        props = regionprops(label_image=label_image, intensity_image=data)
        props.sort(key=lambda x: x.area, reverse=True)
        angle = (
            np.rad2deg(props[0].orientation) + 90
        )  # relative to +x axis, counter-clockwise
        if abs(angle) > 90:
            angle -= 180
        rotation = helicon.set_to_periodic_range(angle, min=-180, max=180)
        data_rotated = helicon.transform_image(image=data, rotation=rotation)
    else:
        rotation = 0.0
        data_rotated = data
    

    bw = closing(data_rotated > threshold, mode="ignore")
    label_image = label(bw)

    props = regionprops(label_image=label_image, intensity_image=data_rotated)
    props.sort(key=lambda x: x.area, reverse=True)
    minr, minc, maxr, maxc = props[0].bbox
    diameter = maxr - minr + 1

    if estimate_center:
        center = props[0].centroid
    else:
        ny, nx = data.shape
        center = (ny // 2, nx // 2)
    shift_y = data.shape[0] // 2 - center[0]

    return rotation, shift_y, diameter


# change from selected_images_thresholded to auto_transform
@reactive.effect
@reactive.event(input.auto_transform)
def update_selected_image_rotation_shift_diameter():
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
            estimate_helix_rotation_center_diameter(
                img,
                threshold=np.max(img) * 0.2,
                estimate_rotation=estimate_rotation,
                estimate_center=estimate_center,
            )
            for img in images
        ]
    )
    rotation = np.mean(tmp[:, 0])
    shift_y = np.mean(tmp[:, 1]) * input.apix()
    diameter = np.max(tmp[:, 2])

    if input_data().is_3d:
        crop_size = int(diameter * 1.2) // 4 * 4
    else:
        crop_size = int(diameter * 2) // 4 * 4

    apix = round(all_images().apix, 4)
    ui.update_numeric("apix", value=apix, max=apix * 2)
    ui.update_numeric("pre_rotation", value=round(rotation, 1))
    ui.update_numeric(
        "shift_y",
        value=shift_y,
        min=-crop_size * apix // 2,
        max=crop_size * apix // 2,
    )
    ui.update_numeric(
        "vertical_crop_size",
        value=max(32, crop_size),
        min=min(32, int(diameter) // 2 * 2),
        max= ny//2 * 2 ,
    )
    ui.update_numeric("horizontal_crop_size", value=nx, min=32, max= nx//2 * 2)


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
@reactive.event(selected_images_rotated_shifted)
def update_transformed_images_displayed():
    req(len(selected_images_rotated_shifted()))
    
    images_displayed = []
    images_displayed_labels = []
    images_displayed_links = []
    
    ny,nx = np.shape(selected_images_rotated_shifted()[0])
    
    image_work = np.zeros((ny,nx*len(selected_images_rotated_shifted())))
    for i, transformed_img in enumerate(selected_images_rotated_shifted()):
        image_work[:,nx*i:nx*(i+1)]=transformed_img
    
    images_displayed.append(image_work)
    images_displayed_labels.append(f"Selected images:")
    images_displayed_links.append("")

    transformed_images_displayed.set(images_displayed)
    transformed_images_labels.set(images_displayed_labels)
    transformed_images_links.set(images_displayed_links) 

@reactive.effect
@reactive.event(input.perform_stitching)
def update_stitched_image_displayed():
    req(len(selected_images_rotated_shifted()))


    images_displayed = []
    images_displayed_labels = []
    images_displayed_links = []
    ny, nx = np.shape(selected_images_rotated_shifted()[0])

    x_offsets = transformed_images_x_offsets()

    from PIL import Image
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(temp_dir + "/TileConfiguration.txt", "w") as tc:

            tc.write("dim = 2\n\n")
            for i, img in enumerate(selected_images_rotated_shifted()):
                tmp = img
                tmp = np.uint8((tmp - np.min(tmp)) / (np.max(tmp) - np.min(tmp)) * 255)
                tmp_imf = Image.fromarray(tmp, "L")
                tmp_imf.save(temp_dir + "/" + str(i) + ".png")
                tc.write(str(i) + ".png; ; (" + str(i * nx + x_offsets[i]) + ", 0.0)\n")
        

        result = compute.itk_stitch(temp_dir)
    
    result = result.astype(np.float32)
    result = (result-result.mean())/result.std()
    result = result/result.max()
    

    images_displayed.append(result)
    images_displayed_labels.append(f"Stitched image:")
    images_displayed_links.append("")

    stitched_image_displayed.set(images_displayed)
    stitched_image_labels.set(images_displayed_labels)
    stitched_image_links.set(images_displayed_links)

# below are the function doing reconstruction
@reactive.effect
@reactive.event(input.run_denovo3D)
def run_denovo3D_reconstruction():
    data = selected_images_thresholded_rotated_shifted_cropped()
    req(len(data) > 0)

    data = data[0]
    ny, nx = data.shape
    tube_length = nx * input.apix()

    imageFile = selected_images_title().strip(":")
    imageIndex = selected_images_labels()[0]

    logger = helicon.get_logger(
        logfile="helicon.denovo3D.log",
        verbose=1,
    )

    if input.twist_min() < input.twist_max():
        twists = np.arange(
            input.twist_min(),
            input.twist_max() + input.twist_step() / 2,
            input.twist_step(),
        )
    else:
        twists = [input.twist_min()]
    if input.rise_min() < input.rise_max():
        rises = np.arange(
            input.rise_min(),
            input.rise_max() + input.rise_step() / 2,
            input.rise_step(),
        )
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
        target_apix2d = input.target_apix2d()
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

    with ui.Progress(min=0, max=len(tasks)) as p:
        p.set(message="Calculation in progress", detail="This may take a while ...")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=cpu) as executor:
            future_tasks = [
                executor.submit(compute.process_one_task, *task) for task in tasks
            ]
            from time import time

            t0 = time()
            results = []
            for completed_task in as_completed(future_tasks):
                result = completed_task.result()
                results.append(result)
                t1 = time()
                remaining = (len(tasks) - len(results)) / len(results) * (t1 - t0)
                p.set(
                    len(results),
                    message=f"Completed {len(results)}/{len(tasks)}",
                    detail=f"{helicon.timedelta2string(remaining)} remaining",
                )

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
        labels += [f"Input image: {selected_images_labels()[0]}", label_x, f"{ri+1}: Z"]

        rec3d_z_sections = helicon.pad_to_size(rec3d_z_sections, shape=rec3d_x_proj.shape)
        images += [query_image_padded, rec3d_x_proj, rec3d_z_sections]

    reconstructed_projection_labels.set(labels)
    reconstructed_projection_images.set(images)


@render.ui
@reactive.event(reconstrunction_results, input.show_download_print_buttons)
def toggle_input_map_download_button():
    if input.show_download_print_buttons() and map_symmetrized() is not None:
        ret = ui.tags.style(
            "#download_denovo3D_input_map {visibility: visible; width: 270px;}"
        )
    else:
        ret = ui.tags.style("#download_denovo3D_input_map {visibility: hidden;}")
    return ret


@render.ui
@reactive.event(reconstrunction_results)
def toggle_output_map_download_button():
    if len(reconstrunction_results()) == 1:
        ret = ui.tags.style(
            "#download_denovo3D_output_map {visibility: visible; width: 256px;}"
        )
    else:
        ret = ui.tags.style("#download_denovo3D_output_map {visibility: hidden;}")
    return ret
