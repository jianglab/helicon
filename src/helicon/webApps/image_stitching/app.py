from pathlib import Path
import numpy as np
import pandas as pd
import tempfile

from shinywidgets import render_plotly

import shiny
from shiny import reactive, req
from shiny.express import input, ui, render, module

import helicon

from . import compute

tmp_out_dir = tempfile.mkdtemp(dir='./')

prev_t_ui_counter = reactive.value(0)
t_ui_counter = reactive.value(0)

images_all = reactive.value([])
image_size = reactive.value(0)
image_apix = reactive.value(0)

displayed_image_ids = reactive.value([])
displayed_images = reactive.value([])
displayed_image_title = reactive.value("Select an image:")
displayed_image_labels = reactive.value([])

initial_selected_image_indices = reactive.value([0])
selected_images_original = reactive.value([])
selected_images_rotated_shifted = reactive.value([])
selected_image_diameter = reactive.value(0)
selected_images_rotated_shifted_cropped = reactive.value([])
selected_images_title = reactive.value("Selected image:")
selected_images_labels = reactive.value([])

transformed_images_displayed = reactive.value([])
transformed_images_title = reactive.value("Transformed selected images:")
transformed_images_labels = reactive.value([])
transformed_images_links = reactive.value([])
transformed_images_vertical_display_size = reactive.value(128)
transformed_images_x_offsets = reactive.value([])

stitched_image_displayed = reactive.value([])
stitched_image_title = reactive.value("Stitched image:")
stitched_image_labels = reactive.value([])
stitched_image_links = reactive.value([])
stitched_image_vertical_display_size = reactive.value(128)

reconstrunction_results = reactive.value([])
reconstructed_projection_images = reactive.value([])
reconstructed_projection_labels = reactive.value([])
reconstructed_map = reactive.value(None)
 
ui.head_content(ui.tags.title("Image Stitching"))
helicon.shiny.google_analytics(id="G-ELN1JJVYYZ")
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
    )
}
url_key = "empiar-10940_job010"

with ui.sidebar(
    width="33vw", style="display: flex; flex-direction: column; height: 100%;"
):
    with ui.navset_pill(id="tab"):  
        with ui.nav_panel("Input 2D Images"):
            with ui.div(id="input_image_files", style="display: flex; flex-direction: column; align-items: flex-start;"):
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
                    if input.input_mode_images() == 'upload':
                        ret.append(
                            ui.input_file(
                                "upload_images",
                                "Upload the input images in MRC format (.mrcs, .mrc)",
                                accept=[".mrcs", ".mrc"],
                                placeholder="mrcs or mrc file",
                            )                            
                        )
                    elif input.input_mode_images() == 'url':
                        ret.append(
                            ui.input_text(
                                "url_images",
                                "Download URL for a RELION or cryoSPARC image output mrc(s) file",
                                value=urls[url_key][0],
                            )
                        )
                    return ret
            
            with ui.div(id="image-selection", style="max-height: 80vh; overflow-y: auto; display: flex; flex-direction: column; align-items: center;"):
                helicon.shiny.image_select(
                    id="select_image",
                    label=displayed_image_title,
                    images=displayed_images,
                    image_labels=displayed_image_labels,
                    image_size=reactive.value(128),
                    initial_selected_indices=initial_selected_image_indices,
                    allow_multiple_selection=True
                )

                @render.ui
                @reactive.event(input.show_gallery_print_button)
                def generate_ui_print_input_images():
                    req(input.show_gallery_print_button())
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
                            width="200px"
                        )
                        
        with ui.nav_panel("Parameters_stiching"):
            with ui.layout_columns(
                col_widths=6, style="align-items: flex-end;"
            ):
                ui.input_checkbox(
                    "ignore_blank", "Ignore blank input images", value=True
                )
                ui.input_checkbox(
                    "show_gallery_print_button", "Show image gallery print button", value=False
                )
        with ui.nav_panel("Parameters_LR"):
            with ui.layout_columns(col_widths=6, style="align-items: flex-end;"):
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
                    value=4,
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


title = "Helical Image Stitching"
ui.h1(title, style="font-weight: bold;")

with ui.div(style="display: flex; flex-direction: row; align-items: flex-start; gap: 10px; margin-bottom: 0"):
    helicon.shiny.image_select(
        id="display_selected_image",
        label=selected_images_title,
        images=selected_images_rotated_shifted,
        image_labels=selected_images_labels,
        image_size=stitched_image_vertical_display_size,
        justification="left",
        enable_selection=False,
    )
    
    with ui.div(style="display: flex; flex-direction: column; align-items: flex-start; gap: 10px; margin-bottom: 0"):
        @reactive.effect
        @reactive.event(selected_images_original,ignore_init=True)
        def generate_image_transformation_uis():
            req(len(selected_images_labels()))
            print(selected_images_labels())
            labels = selected_images_labels().copy()
            for i,idx in enumerate(labels):
                curr_t_ui_counter=t_ui_counter()
                #ui.remove_ui(selector=f"t_ui_group_{idx}_card", multiple=True)
                #selected_transformation(f"st_{idx}")
                ui.insert_ui(shiny.ui.row(transformation_ui_group(f"t_ui_group_{curr_t_ui_counter}")),
                    selector = "#perform_stitching",
                    where = "beforeBegin")

                id_rotation = "t_ui_group_"+str(curr_t_ui_counter)+"_pre_rotation"
                id_x_shift = "t_ui_group_"+str(curr_t_ui_counter)+"_shift_x"
                id_y_shift = "t_ui_group_"+str(curr_t_ui_counter)+"_shift_y"
        
                @reactive.effect
                @reactive.event(input[id_rotation], input[id_y_shift])
                def transform_selected_images(i=i,id_rotation=id_rotation,id_y_shift=id_y_shift):
                    req(len(selected_images_original()))
                    curr_img_idx=i
                    print(f"listening to {id_rotation}, {id_y_shift}")

                    rotated = selected_images_rotated_shifted().copy()
                    if input[id_rotation]()!=0 or input[id_y_shift]()!=0:
                        rotated[curr_img_idx] = helicon.transform_image(image=selected_images_original()[curr_img_idx].copy(), rotation=input[id_rotation](), post_translation=(input[id_y_shift](), 0))
                    selected_images_rotated_shifted.set(rotated)
                    print("curr_img_idx = " + str(curr_img_idx))
                    print("curr_t_ui_counter = " + str(curr_t_ui_counter))
                    print(f"rot shift {i} done")
                print(f"inserted t_ui_group_{curr_t_ui_counter}")
                curr_t_ui_counter += 1
                t_ui_counter.set(curr_t_ui_counter)
            
                @reactive.effect
                @reactive.event(selected_images_rotated_shifted, input[id_x_shift])
                def update_transformed_images_displayed(x_shift_i=i,id_x_shift=id_x_shift):
                    req(len(selected_images_rotated_shifted()))
    
                    images_displayed = []
                    images_displayed_labels = []
                    images_displayed_links = []
                
                    curr_x_offsets = transformed_images_x_offsets().copy()
                    ny,nx = np.shape(selected_images_rotated_shifted()[0])
    
                    image_work = np.zeros((ny,nx*len(selected_images_rotated_shifted())))
                    for img_i, transformed_img in enumerate(selected_images_rotated_shifted()):
                        if img_i == x_shift_i:
                            image_work[:,nx*img_i+input[id_x_shift]():nx*(img_i+1)+input[id_x_shift]()]=transformed_img
                            curr_x_offsets[x_shift_i] = input[id_x_shift]()
                        else:
                            image_work[:,nx*img_i:nx*(img_i+1)]=transformed_img
    
                    images_displayed.append(image_work)
                    images_displayed_labels.append(f"Selected images:")
                    images_displayed_links.append("")

                    transformed_images_displayed.set(images_displayed)
                    transformed_images_labels.set(images_displayed_labels)
                    transformed_images_links.set(images_displayed_links)
                
                    transformed_images_x_offsets.set(curr_x_offsets)


        
        @render.ui
        @reactive.event(input.select_image)
        def display_action_button():
            req(len(selected_images_rotated_shifted()))
            return ui.input_task_button("perform_stitching", label="Stitch!")

with ui.div(style="max-height: 50vh; overflow-y: auto;"):
    helicon.shiny.image_select(
        id="display_transformed_images",
        label=transformed_images_title,
        images=transformed_images_displayed,
        image_labels=transformed_images_labels,
        image_links=transformed_images_links,
        image_size=transformed_images_vertical_display_size,
        justification="left",
        enable_selection=False
    )
        
with ui.div(style="max-height: 50vh; overflow-y: auto;"):
    helicon.shiny.image_select(
        id="display_stitched_image",
        label=stitched_image_title,
        images=stitched_image_displayed,
        image_labels=stitched_image_labels,
        image_links=stitched_image_links,
        image_size=stitched_image_vertical_display_size,
        justification="left",
        enable_selection=False
    )

with ui.layout_columns(col_widths=2):
    @render.download(label = "Download stitched image")
    @reactive.event(stitched_image_displayed)
    def download_stitched_image():
        req(len(stitched_image_displayed()))
        import tempfile
        import mrcfile
        with mrcfile.new(tmp_out_dir+'/stitched.mrc',overwrite=True) as o_mrc:
            data = np.array(stitched_image_displayed()).astype(np.float32)/255
            o_mrc.set_data(np.array(data,dtype=np.float32))
            o_mrc.voxel_size=image_apix()
            return tmp_out_dir+'/stitched.mrc'

@render.ui
@reactive.event(stitched_image_displayed)
def display_reconstruction_panel():
    req(len(stitched_image_displayed()))

    return ui.div(
        ui.div(
            ui.div(
                ui.h3("Twist (°)", style="margin: 0;"),
                ui.div(
                    ui.input_numeric("twist_min", "min", value=0.1, step=0.1, width="70px"),
                    ui.input_numeric("twist_max", "max", value=2.0, step=0.1, width="70px"),
                    ui.input_numeric("twist_step", "step", value=1, step=0.1, width="70px"),
                    style="display: flex; flex-direction: row; align-items: center; gap: 10px;",
                ),
                style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; width: 33%;",
            ),
            ui.div(
                ui.h3("Rise (Å)", style="margin: 0;"),
                ui.div(
                    ui.input_numeric("rise_min", "min", value=4.75, step=0.1, width="70px"),
                    ui.input_numeric("rise_max", "max", value=4.75, step=0.1, width="70px"),
                    ui.input_numeric("rise_step", "step", value=0.1, step=0.01, width="70px"),
                    style="display: flex; flex-direction: row; align-items: center; gap: 10px;",
                ),
                style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; width: 33%;",
            ),
            ui.div(
                ui.h3("Csym", style="margin: 0;"),
                ui.input_numeric("csym", "n", value=1, min=1, step=1, width="70px"),
                style="padding: 10px; border: 1px solid #ddd; border-radius: 5px; width: 33%;",
            ),
            style="display: flex; flex-direction: row; align-items: flex-start; gap: 15px; width: 100%;",
        ),
        ui.input_task_button(
            "run_denovo3D", label="Reconstruct 3D Map", style="width: 150px; height: 50px; margin-top: 20px;"
        ),
        style="display: flex; flex-direction: column; align-items: flex-start; gap: 20px;",
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
    style="max-height: 100vh; overflow-y: auto; display: flex; flex-direction: column; align-items: left; margin-bottom: 5px"
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
    "<i><p>Developed by the <a href='https://jiang.bio.purdue.edu/HelicalImageStitching' target='_blank'>Jiang Lab</a>. Report issues to <a href='https://github.com/jianglab/HelicalImageStitching/issues' target='_blank'>HelicalImageStitching</a>.</p></i>"
)

#@module
#def selected_transformation(input, output, session):
#	@shiny.render.ui
#	def show_ui_groups():
#		return transformation_ui_group(id=session.ns)

def transformation_ui_group(prefix):
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
            min=-100,
            max=100,
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
    displayed_image_title.set(f"{len(images)}/{n} images | {nx}x{ny} pixels | {image_apix()} Å/pixel")
    displayed_image_labels.set(image_labels)


@reactive.effect
@reactive.event(input.select_image)
def update_selecte_images_orignal():
    selected_images_original.set(
        [displayed_images()[i] for i in input.select_image()]
    )
    selected_images_labels.set(
        [displayed_image_labels()[i] for i in input.select_image()]
    )
    selected_images_rotated_shifted.set(
        [displayed_images()[i] for i in input.select_image()]
    )
    transformed_images_x_offsets.set(
        np.zeros(len(input.select_image()))
    )

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

    images_displayed.append(result)
    images_displayed_labels.append(f"Stitched image:")
    images_displayed_links.append("")

    stitched_image_displayed.set(images_displayed)
    stitched_image_labels.set(images_displayed_labels)
    stitched_image_links.set(images_displayed_links)



@reactive.effect
@reactive.event(input.stitched_image_vertical_display_size)
def update_stitched_image_vertical_display_size():
    stitched_image_vertical_display_size.set(input.stitched_image_vertical_display_size())



@reactive.effect
@reactive.event(input.run_denovo3D)
def run_denovo3D_reconstruction():
    print('start job')

    req(len(stitched_image_displayed()))

    data = stitched_image_displayed()

    data = data[0]
    data = data.astype(np.float32)
    data = (data-data.mean())/data.std()
    ny, nx = data.shape

    apix = image_apix()
    tube_length = nx * apix

    imageFile = selected_images_title().strip(":")
    imageIndex = str(0)

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
        target_apix3d = apix
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

                print(t1)

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


        query_image = stitched_image_displayed()[0]
        query_image_padded = helicon.pad_to_size(query_image, shape=rec3d_x_proj.shape)

        label_x = f"{ri+1}: X|score={score:.4f}|pitch={int(round(rise*360/abs(twist))):,}Å|twist={round(twist,3)}°|rise={round(rise,6)}Å"
        labels += [f"Input image: {selected_images_labels()[0]}", label_x, f"{ri+1}: Z"]
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



 
