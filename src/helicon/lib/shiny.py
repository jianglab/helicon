from typing import Optional
from pathlib import Path

import shiny
from shiny import reactive
from shiny.express import ui, module, render, expressify


def image_gallery(
    id,
    label=reactive.value(""),
    images=reactive.value([]),
    display_image_labels=True,
    image_labels=reactive.value([]),
    image_links=reactive.value([]),
    image_size=reactive.value(128),
    image_border=2,
    gap=0,
    justification="center",
    enable_selection=False,
    allow_multiple_selection=False,
    initial_selected_indices=reactive.value([]),
):
    if images() is None or len(images()) == 0:
        return None

    import numpy as np
    from PIL import Image
    from helicon import encode_numpy, encode_PIL_Image

    if enable_selection and len(image_links()):
        raise ValueError(
            f"image_gallery(): only allows either enable_selection or image_labels but not both"
        )

    images_final = []
    for i, image in enumerate(images()):
        if isinstance(image, str):
            tmp = image
        elif isinstance(image, Image.Image):
            tmp = encode_PIL_Image(image)
        elif isinstance(image, np.ndarray) and image.ndim == 2:
            tmp = encode_numpy(image)
        else:
            raise ValueError(
                f"Image must be an image file, a PIL Image, or a 2D numpy array. Your have provided {image}"
            )
        images_final.append(tmp)

    assert len(image_labels()) == 0 or len(image_labels()) == len(images_final)

    if len(image_labels()):
        image_labels_final = image_labels()
    else:
        image_labels_final = list(range(1, len(images_final) + 1))

    if len(image_links()):
        image_links_final = image_links()
    else:
        image_links_final = [""] * len(images_final)

    assert image_size() >= 32

    bids = [f"{id}_image_{i+1}" for i in range(len(images_final))]

    def create_image_button(
        i, image, label, link, bid, enable_selection=True, allow_multiple_selection=True
    ):
        img = ui.img(
            src=image,
            alt=f"Image {i+1}",
            title=str(label),
            style=f"object-fit: contain; height: {image_size()}px; border: {image_border}px solid transparent;",
        )
        if link:
            img = ui.a(img, href=link, target="_blank")

        return ui.div(
            (
                ui.div(
                    img,
                    ui.p(
                        label,
                        style="text-align: left; color: white; text-shadow: -1px -1px 0.5px rgba(0,0,0,0.5), 1px -1px 0.5px rgba(0,0,0,0.5), -1px 1px 0.5px rgba(0,0,0,0.5), 1px 1px 0.5px rgba(0,0,0,0.5); position: absolute; top: 2px; left: 5px;",
                    ),
                    style="position: relative;",
                )
                if display_image_labels
                else img
            ),
            id=bid,
            style=f"padding: 0px; border: 0px; margin: 0px; background-color: transparent;",
            onmouseover=(
                f"if (this.querySelector('img').style.border !== '{image_border}px solid red') {{this.querySelector('img').style.border='{image_border}px solid blue'; this.querySelector('p').style.color='blue';}}"
                if enable_selection
                else None
            ),
            onmouseout=(
                f"if (this.querySelector('img').style.border !== '{image_border}px solid red') {{this.querySelector('img').style.border='{image_border}px solid transparent';  this.querySelector('p').style.color='white';}}"
                if enable_selection
                else None
            ),
            onclick=(
                f"""var allow_multiple_selection = {1 if allow_multiple_selection else 0};
                    if (allow_multiple_selection && event.altKey) {{
                        selected = this.getAttribute('selected') === 'true';
                        selected = !selected;
                        var images = this.parentElement.children;
                        for (var i = 0; i < images.length; i++) {{
                            images[i].setAttribute('selected', selected);
                            var img  = images[i].querySelector("img");
                            var text = images[i].querySelector("p");
                            img.style.border = selected ? "{image_border}px solid red" : "{image_border}px solid transparent";
                            if (text) {{
                                text.style.color = selected ? "red" : "white";
                            }}
                        }}                    
                    }}
                    else {{
                        var selected;
                        if (event.shiftKey) {{
                            selected = this.getAttribute('selected') === 'true';
                            selected = !selected;
                        }} else {{
                            selected = true;
                        }}
                        this.setAttribute('selected', selected);
                        var img  = this.querySelector("img");
                        var text = this.querySelector("p");
                        img.style.border = selected ? "{image_border}px solid red" : "{image_border}px solid transparent";
                        if (text) {{
                            text.style.color = selected ? "red" : "white";
                        }}


                        if (!allow_multiple_selection || !event.shiftKey) {{
                            var images = this.parentElement.children;
                            for (var i = 0; i < images.length; i++) {{
                                if (images[i] === this) continue;
                                images[i].setAttribute('selected', false);
                                var img  = images[i].querySelector("img");
                                var text = images[i].querySelector("p");
                                img.style.border = "{image_border}px solid transparent";
                                if (text) {{
                                    text.style.color = "white";
                                }}
                            }}
                        }}                    
                    }}

                    var selected_prev = this.parentElement.getAttribute('selected');
                    if (selected_prev === null) selected_prev = [];
                    else  selected_prev = selected_prev.split(',');
                    for (var i = 0; i < selected_prev.length; i++) {{
                        selected_prev[i] = parseInt(selected_prev[i]);
                    }}
                    var selected_new = [];
                    for (var i = 0; i < selected_prev.length; i++) {{
                        if (this.parentElement.children[selected_prev[i]] && this.parentElement.children[selected_prev[i]].getAttribute('selected') === 'true' && !selected_new.includes(parseInt(selected_prev[i]))) {{
                            selected_new.push(parseInt(selected_prev[i]));
                        }}
                    }}
                    for (var i = 0; i < this.parentElement.children.length; i++) {{
                        if (this.parentElement.children[i].getAttribute('selected') === 'true' && !selected_new.includes(i)) {{
                            selected_new.push(i);
                        }}
                    }}
                    this.parentElement.setAttribute('selected', selected_new);

                    Shiny.setInputValue('{id}', selected_new, {{priority: 'deferred'}});
                """
                if enable_selection
                else None
            ),
        )

    ui_images = ui.div(
        *[
            create_image_button(
                i,
                image,
                image_labels_final[i],
                image_links_final[i],
                bid,
                enable_selection,
                allow_multiple_selection,
            )
            for i, (image, bid) in enumerate(zip(images_final, bids))
        ],
        style=f"display: flex; flex-flow: row wrap; justify-content: {justification}; justify-items: center; align-items: center; gap: {gap}px {gap}px; margin: 0 0 {image_border}px 0",
    )

    if len(label()):
        ui_images = ui.div(
            ui.h6(
                label(),
                style=f"text-align: {justification}; margin: 0;",
                title="Hold the Shift key while clicking to select multiple images; Hold the Alt/Option key while clicking to select/unselect all images",
            ),
            ui_images,
            style=f"display: flex; flex-direction: column; gap: {gap}px; margin: 0",
        )

    if enable_selection and len(initial_selected_indices()) > 0:
        click_scripts = []
        for i in initial_selected_indices():
            click_scripts.append(
                ui.tags.script(
                    f"""
                        var bid = '{bids[i]}';
                        var element = document.getElementById(bid);
                        var event = new MouseEvent('click', {{
                            bubbles: true,
                            cancelable: true,
                            view: window,
                            shiftKey: true
                        }});
                        element.dispatchEvent(event);
                    """
                )
            )
        return (ui_images, click_scripts)
    else:
        return ui_images


@module
def image_select(
    input,
    output,
    session,
    label="Select Image(s):",
    images=reactive.value([]),
    display_image_labels=True,
    image_labels=reactive.value([]),
    image_links=reactive.value([]),
    image_size=reactive.value(128),
    image_border=2,
    gap=0,
    justification="center",
    enable_selection=True,
    allow_multiple_selection=True,
    initial_selected_indices=reactive.value([]),
):
    @shiny.render.ui
    def show_image_gallery():
        return image_gallery(
            id=session.ns,
            label=label,
            images=images,
            display_image_labels=display_image_labels,
            image_labels=image_labels,
            image_links=image_links,
            image_size=image_size,
            initial_selected_indices=initial_selected_indices,
            enable_selection=enable_selection,
            allow_multiple_selection=allow_multiple_selection,
            image_border=image_border,
            gap=gap,
            justification=justification,
        )


# server-side file selection
@shiny.module.ui
def file_selection_ui(label="Select a file", value=None, width="100%"):
    return shiny.ui.div(
        shiny.ui.input_text(
            "selected_file_path", label=label, value=value, width="100%"
        ),
        shiny.ui.accordion(
            shiny.ui.accordion_panel(
                "",
                shiny.ui.input_text(
                    "current_directory",
                    label="Current directory",
                    value=str(Path(value).parent) if value else str(Path.cwd()),
                    width="100%",
                ),
                shiny.ui.layout_column_wrap(
                    shiny.ui.input_select(
                        "file",
                        "Select a file",
                        choices=[Path(value).name] if value else [],
                        selected=Path(value).name if value else None,
                        width="100%",
                    ),
                    shiny.ui.input_select(
                        "sub_directory",
                        "Go to a sub-directory",
                        choices=[],
                        width="100%",
                    ),
                    width=6,
                ),
                style="padding: 10px;",
            ),
            id="accordion_file_selection",
            open=False,
            style="border: 1px solid #ccc;",
        ),
        style=f"margin: 0; margin-bottom: 10px; padding: 0; width: {width};",
    )


@shiny.module.server
def file_selection_server(
    input,
    output,
    session,
    file_types: Optional[str | list[str]] = None,
    ignore_hidden_files=True,
):
    if file_types is None:
        file_types = []
    elif isinstance(file_types, str):
        file_types = [file_types]

    @reactive.effect
    @reactive.event(input.current_directory)
    def update_sub_directories():
        p = Path(input.current_directory())
        shiny.req(p.exists())
        try:
            directories = [d.name for d in sorted(p.iterdir()) if d.is_dir()]
            if ignore_hidden_files:
                directories = [d for d in directories if d[0] != "."]
            directories = [".", ".."] + directories
            ui.update_select("sub_directory", choices=directories)
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            m = ui.modal(
                f"{input.current_directory()}: failed to list sub-directories.",
                title="Folder access error",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.sub_directory)
    def goto_sub_directories():
        shiny.req(len(input.sub_directory()))
        sub_dir = Path(input.current_directory()) / input.sub_directory()
        ui.update_text("current_directory", value=str(sub_dir.resolve()))

    @reactive.effect
    @reactive.event(input.current_directory)
    def update_files():
        p = Path(input.current_directory())
        shiny.req(p.exists())
        try:
            files = [f.name for f in sorted(p.iterdir(), reverse=True) if f.is_file()]
            if ignore_hidden_files:
                files = [f for f in files if f[0] != "."]
            if file_types:
                files_final = []
                for f in files:
                    for ft in file_types:
                        if f.endswith(ft):
                            files_final.append(f)
                            continue
            else:
                files_final = files

            selected = None
            same_folder = Path(input.selected_file_path()).parent.samefile(
                Path(input.current_directory())
            )
            if len(files_final):
                if input.file() and same_folder and input.file() in files_final:
                    selected = input.file()
                else:
                    selected = files_final[0]
            ui.update_select("file", choices=files_final, selected=selected)
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            m = ui.modal(
                f"{str(input.current_directory())}: failed to list files.",
                title="Folder access error",
                easy_close=True,
                footer=None,
            )
            ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.parent_button)
    def go_to_parent_folder():
        parent_directory = Path(input.current_directory()).parent
        if parent_directory.exists():
            ui.update_text("current_directory", value=str(parent_directory))

    @reactive.effect
    @reactive.event(input.file)
    def _():
        ui.update_text(
            "selected_file_path",
            value=str(Path(input.current_directory()) / input.file()),
        )

    return input.selected_file_path


@expressify
def google_analytics(id):
    ui.head_content(
        ui.HTML(
            f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={id}"></script>
            <script>
            window.dataLayer = window.dataLayer || [];
            function gtag(){{dataLayer.push(arguments);}}
            gtag('js', new Date());
            gtag('config', '{id}');
            </script>
            """
        )
    )


def set_input_text_numeric_update_on_change():
    if (
        getattr(set_input_text_numeric_update_on_change, "original_update_text", None)
        is None
    ):
        original_update_text = ui.update_text

        def patched_update_text(input_id, *args):
            original_update_text(input_id, *args)
            ui.insert_ui(
                ui.tags.script(
                    f"""
                    var inputElement = document.getElementById('{input_id}');
                    var event = new Event('change', {{ bubbles: true }});
                    inputElement.dispatchEvent(event);
                """
                ),
                selector="body",
                where="beforeBegin",
            )

        ui.update_text = patched_update_text
        set_input_text_numeric_update_on_change.original_update_text = (
            original_update_text
        )

    if (
        getattr(
            set_input_text_numeric_update_on_change, "original_update_numeric", None
        )
        is None
    ):
        original_update_numeric = ui.update_numeric

        def patched_update_numeric(input_id, *args, **kwargs):
            original_update_numeric(input_id, *args, **kwargs)
            ui.insert_ui(
                ui.tags.script(
                    f"""
                    var inputElement = document.getElementById('{input_id}');
                    var event = new Event('change', {{ bubbles: true }});
                    inputElement.dispatchEvent(event);
                """
                ),
                selector="body",
                where="beforeBegin",
            )

        ui.update_numeric = patched_update_numeric
        set_input_text_numeric_update_on_change.original_update_numeric = (
            original_update_numeric
        )

    ui.insert_ui(
        ui.tags.script(
            """
          document.querySelectorAll('.shiny-input-text, .shiny-input-number').forEach(function(input) {
              input.addEventListener('change', function(event) {
                  Shiny.setInputValue(input.id + '_changed', this.value);
              });
          });
        """
        ),
        selector="body",
        where="beforeBegin",
    )


def get_client_url(input):
    d = input._map
    url = f"{d['.clientdata_url_protocol']()}//{d['.clientdata_url_hostname']()}:{d['.clientdata_url_port']()}{d['.clientdata_url_pathname']()}{d['.clientdata_url_search']()}"
    return url


def get_client_url_query_params(input, keep_list=True):
    d = input._map
    qs = d[".clientdata_url_search"]().strip("?")
    import urllib.parse

    parsed_qs = urllib.parse.parse_qs(qs)
    if not keep_list:
        for k, v in parsed_qs.items():
            if isinstance(v, list) and len(v) == 1:
                parsed_qs[k] = v[0]
    return parsed_qs


def set_client_url_query_params(query_params):
    import urllib.parse

    encoded_query_params = urllib.parse.urlencode(query_params, doseq=True)
    script = ui.tags.script(
        f"""
                var url = new URL(window.location.href);
                url.search = '{encoded_query_params}';
                window.history.pushState(null, '', url.toString());
            """
    )
    return script


@expressify
def setup_ajdustable_sidebar(width="33vw"):
    return [
        ui.div(
            id="handle",
            style=f"position: absolute; top:0; bottom: 0; left: {width}; width: 2px; cursor: ew-resize; background: #ddd;",
        ),
        ui.tags.script(
            """
                const handle = document.getElementById('handle');
                const sidebar = document.querySelector('.bslib-sidebar-layout');

                handle.addEventListener('mousedown', (e) => {
                    const moveHandler = (e) => {
                        var percent = e.clientX * 100 / document.body.clientWidth;
                        percent = Math.min(Math.max(10, percent), 90) + 'vw';
                        handle.style.left = percent;
                        sidebar.style.setProperty('--_sidebar-width', percent);
                    };
                    const upHandler = () => {
                        document.removeEventListener('mousemove', moveHandler);
                        document.removeEventListener('mouseup', upHandler);
                    };
                    document.addEventListener('mousemove', moveHandler);
                    document.addEventListener('mouseup', upHandler);
                });
                
                window.addEventListener('load', () => {
                    const collapse_toggle = sidebar.querySelector('.collapse-toggle');
                    collapse_toggle.addEventListener('click', (e) => {
                        handle.style.display = handle.style.display === 'none' ? 'block' : 'none';
                    });
                });
            """
        ),
    ]
