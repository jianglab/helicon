from shiny import render, reactive
from shiny.express import input, output, session, ui, app_opts, module, render
import helicon

import pathlib

app_opts(static_assets=pathlib.Path(__file__).parent/"WWW")

import numpy as np

ny, nx = 200, 200  # You can adjust these dimensions as needed
test_image = helicon.encode_numpy(np.random.rand(ny, nx))

with ui.sidebar():
    ui.h1("Image Stitching"),
    MAX_SIZE = 50000
    ui.input_file("twod_imgs_file", "Choose a file to upload:", multiple=False)
    with ui.panel_conditional("input.twod_imgs_file"):
        selected_xxx = helicon.shiny.image_select(id="XXX", label="TEST", images=[test_image]*3)

@render.code
def get_imgs_from_file():
    file_infos = input.twod_imgs_file()
    if not file_infos:
        return

    out_str = ""
    for file_info in file_infos:
        out_str += (
            "=" * 47
            + "\n"
            + file_info["name"]
        )
        if file_info["size"] > MAX_SIZE:
            out_str += f"\nTruncating at {MAX_SIZE} bytes."

        out_str += "\n" + "=" * 47 + "\n"
        
    return out_str

@render.text
def combined_selection():
    xxx = str(selected_xxx())
    return f"XXX: {', '.join(xxx) if xxx else 'None'}"


