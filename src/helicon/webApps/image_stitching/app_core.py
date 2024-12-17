from pathlib import Path
import numpy as np
import pandas as pd

from shinywidgets import render_plotly

import shiny
from shiny import reactive, req
from shiny.express import input, ui, render, module

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


stitched_image_displayed = reactive.value([])
stitched_image_title = reactive.value("Stitched image:")
stitched_image_labels = reactive.value([])
stitched_image_links = reactive.value([])
stitched_image_vertical_display_size = reactive.value(128)
 
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
                        
        with ui.nav_panel("Parameters"):
            with ui.layout_columns(
                col_widths=6, style="align-items: flex-end;"
            ):
                ui.input_checkbox(
                    "ignore_blank", "Ignore blank input images", value=True
                )
                ui.input_checkbox(
                    "show_gallery_print_button", "Show image gallery print button", value=False
                )


title = "Image stitching"
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

    @reactive.effect
    @reactive.event(selected_images_original)
    def generate_image_transformation_uis():
        req(len(selected_images_labels()))
        print(selected_images_labels())
        
        #ui.remove_ui(selector="div:has(> #tuis)",multiple=True)
        for i,idx in enumerate(selected_images_labels()):
            #selected_transformation(f"st_{idx}")
            ui.insert_ui(shiny.ui.row(transformation_ui_group(f"t_ui_group_{idx}")),
                selector = "#perform_stitching",
                where = "beforeBegin")

            print(f"inserted t_ui_group_{idx}")
            
            @reactive.effect
            @reactive.event(input[f"t_ui_group_{idx}_pre_rotation"], input[f"t_ui_group_{idx}_shift_y"], ignore_init=True)
            def transform_selected_images():
                req(len(selected_images_original()))
                curr_img_idx=i
                print(f"listening to t_ui_group_{idx}_pre_rotation")
                
                rotated = selected_images_original().copy()
                if input[f"t_ui_group_{idx}_pre_rotation"]()!=0 or input[f"t_ui_group_{idx}_shift_y"]()!=0:
                    rotated[curr_img_idx] = helicon.transform_image(image=rotated[curr_img_idx].copy(), rotation=input[f"t_ui_group_{idx}_pre_rotation"](), post_translation=(input[f"t_ui_group_{idx}_shift_y"](), 0))

                selected_images_rotated_shifted.set(rotated)
                print(curr_img_idx)
                print(f"rot shift {idx} done")

        
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

with ui.div(style="max-height: 80vh; overflow-y: auto;"):    
    @render.ui
    @reactive.event(input.show_gallery_print_button)
    def generate_ui_print_map_side_projection_images():
        req(input.show_gallery_print_button())
        return ui.input_action_button(
                "print_stitched_images",
                "Print stitched images",
                onclick=""" 
                            var w = window.open();
                            w.document.write(document.head.outerHTML);
                            var printContents = document.getElementById('display_stitched_image-show_image_gallery').innerHTML;
                            w.document.write(printContents);
                            w.document.write('<script type="text/javascript">window.onload = function() { window.print(); w.close();};</script>');
                            w.document.close();
                            w.focus();
                        """
            )

    
ui.HTML(
    "<i><p>Developed by the <a href='https://jiang.bio.purdue.edu/HelicalProjection' target='_blank'>Jiang Lab</a>. Report issues to <a href='https://github.com/jianglab/HelicalProjection/issues' target='_blank'>HelicalProjection@GitHub</a>.</p></i>"
)

#@module
#def selected_transformation(input, output, session):
#	@shiny.render.ui
#	def show_ui_groups():
#		return transformation_ui_group(id=session.ns)

def transformation_ui_group(id):
    return shiny.ui.card(shiny.ui.layout_columns(
        ui.input_slider(
            f"{id}_pre_rotation",
            "Rotation (°)",
            min=-45,
            max=45,
            value=0,
            step=0.1,
        ),       
        ui.input_slider(
            f"{id}_shift_y",
            "Vertical shift (pixel)",
            min=-100,
            max=100,
            value=0,
            step=1,
        ),
        ui.input_slider(
            f"{id}_vertical_crop_size",
            "Vertical crop (pixel)",
            min=32,
            max=256,
            value=0,
            step=2,
        ),
        col_widths=4))


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
    ny,nx = np.shape(selected_images_rotated_shifted()[0])
    
    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        with open(temp_dir+"/TileConfiguration.txt","w") as tc:
            tc.write("dim = 2\n\n")
            for i, img in enumerate(selected_images_rotated_shifted()):
                tmp = img
                tmp = np.uint8((tmp-np.min(tmp))/(np.max(tmp)-np.min(tmp))*255)
                tmp_imf=Image.fromarray(tmp,"L")
                tmp_imf.save(temp_dir+"/"+str(i)+".png")
                tc.write(str(i)+".png; ; ("+ str(i*nx) + ", 0.0)\n")

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




 

