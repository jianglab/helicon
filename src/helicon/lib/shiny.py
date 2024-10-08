from shiny import reactive, req
from shiny.express import ui, module, render, expressify


@module
def image_select(
    input,
    output,
    session,
    label="Select Image(s):",
    images=reactive.value([]),
    display_image_labels=True,
    image_labels=reactive.value([]),
    image_size=reactive.value(128),
    initial_selected_indices=reactive.value([]),
    enable_selection=True,
    allow_multiple_selection=True,
    image_border=2,
    gap=0,
):
    @render.ui
    def display_images():
        if images() is None or len(images()) == 0:
            return None

        import numpy as np
        from PIL import Image
        from helicon import encode_numpy, encode_PIL_Image

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
                    "image must be an image file, a PIL Image, or a 2D numpy array"
                )
            images_final.append(tmp)

        assert len(image_labels()) == 0 or len(image_labels()) == len(images_final)

        if len(image_labels()):
            image_labels_final = image_labels()
        else:
            image_labels_final = list(range(1, len(images_final) + 1))

        assert image_size() >= 32

        bids = [f"{session.ns}_image_{i+1}" for i in range(len(images_final))]

        def create_image_button(
            i, image, label, bid, enable_selection=True, allow_multiple_selection=True
        ):
            img = ui.img(
                src=image,
                alt=f"Image {i+1}",
                title=str(label),
                style=f"object-fit: contain; max-width: {image_size()-image_border*2}px; max-height: {image_size()-image_border*2}px; border: {image_border}px solid transparent;",
            )

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
                    f"""var selected;
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

                        var allow_multiple_selection = {1 if allow_multiple_selection else 0};

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

                        var selected_prev = this.parentElement.getAttribute('selected');
                        if (selected_prev === null) selected_prev = [];
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
                            if (this.parentElement.children[i] === this && this.getAttribute('selected') === 'true' && !selected_new.includes(i)) {{
                                selected_new.push(i);
                            }}
                        }}
                        this.parentElement.setAttribute('selected', selected_new);

                        Shiny.setInputValue('{session.ns}', selected_new, {{priority: 'deferred'}});
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
                    bid,
                    enable_selection,
                    allow_multiple_selection,
                )
                for i, (image, bid) in enumerate(zip(images_final, bids))
            ],
            style=f"display: flex; flex-flow: row wrap; justify-content: center; justify-items: center; align-items: center; gap: {gap}px {gap}px; margin: 0 0 {image_border}px 0",
        )

        if len(label):
            ui_images = ui.div(
                ui.h6(
                    label,
                    style="text-align: center; margin: 0;",
                    title="Hold the Shift key while clicking to select multiple images",
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
def setup_ajdustable_sidebar():
    return [
        ui.div(
            id="handle",
            style="position: absolute; top:0; bottom: 0; left: 33vw; width: 2px; cursor: ew-resize; background: #ddd;",
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
                """
        ),
    ]
