from shiny import reactive
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
    auto_select_first=False,
    disable_selection=False,
    image_border=2,
    gap=0,
):
    import numpy as np
    from PIL import Image
    from helicon import encode_numpy, encode_PIL_Image

    selection_return = reactive.value(None)
    selection = reactive.value([])
    initial_selection = reactive.value([])
    bids_react = reactive.value([])

    @render.ui
    def display_images():
        if images() is None or len(images()) == 0:
            return None

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

        bids = [f"image_select_{i+1}" for i in range(len(images_final))]

        def create_image_button(i, image, label, bid):
            img = ui.img(
                src=image,
                alt=f"Image {i+1}",
                title=str(label),
                style=f"object-fit: contain; max-width: {image_size()-image_border*2}px; max-height: {image_size()-image_border*2}px; border: {image_border}px solid transparent;",
            )

            return ui.input_action_button(
                id=bid,
                label=(
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
                disabled=disable_selection,
                style=f"padding: 0px; border: 0px; margin: 0px; background-color: transparent;",
                onmouseover=f"if (this.querySelector('img').style.border !== '{image_border}px solid red') {{this.querySelector('img').style.border='{image_border}px solid blue'; this.querySelector('p').style.color='blue';}}",
                onmouseout=f"if (this.querySelector('img').style.border !== '{image_border}px solid red') {{this.querySelector('img').style.border='{image_border}px solid transparent';  this.querySelector('p').style.color='white';}}",
                onclick=f"""var bid = '{session.ns}-{bid}';
                            var count0 = parseInt(this.getAttribute('click_count')) || 0;
                            count = count0 + 1
                            var info = {{
                                count: count,
                                selected: (count)%2,
                                ctrlKey: event.ctrlKey,
                                shiftKey: event.shiftKey,
                                altKey: event.altKey,
                                metaKey: event.metaKey
                            }};

                            var img  = this.querySelector("img");
                            var text = this.querySelector("p");
                            img.style.border = count%2 ? "{image_border}px solid red" : "{image_border}px solid transparent";
                            if (text) {{
                                text.style.color = count%2 ? "red" : "white";
                            }}

                            this.setAttribute('click_count', count);
                            //Shiny.setInputValue(bid, count, {{priority: 'deferred'}});
                            Shiny.setInputValue(bid + '_click', info, {{priority: 'deferred'}});
                            //console.log("click", bid, parseInt(this.getAttribute('click_count')), info);
                        """,
            )

        ui_images = ui.div(
            *[
                create_image_button(i, image, image_labels_final[i], bid)
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

        bids_react.set(bids)
        initial_selection.set([0] * len(bids))

        if not disable_selection and auto_select_first:
            tmp = initial_selection()
            tmp[0] = 1
            initial_selection.set(tmp)

            click_scripts = []
            click_scripts.append(
                ui.tags.script(
                    f"""
                    var bid = '{session.ns}-{bids[0]}';
                    document.getElementById(bid).click(); 
                """
                )
            )
            return (ui_images, click_scripts)
        else:
            return ui_images

    @render.ui
    def ordered_selection():
        status = [
            (input[bid]() + initial_selection()[i]) % 2 == 1
            for i, bid in enumerate(bids_react())
        ]
        current = [i for i, is_selected in enumerate(status) if is_selected]
        previous = getattr(ordered_selection, "previous", [])
        result = [i for i in previous if i in current]
        result += [i for i in current if i not in result]
        scripts = None
        if len(result) > 1:
            shiftKey = input[f"{bids_react()[result[-1]]}_click"]()["shiftKey"]
            if not shiftKey:
                scripts = []
                for i in result[:-1]:
                    bid = f"{session.ns}-{bids_react()[i]}"
                    script = ui.tags.script(
                        f"""document.getElementById('{bid}').click();"""
                    )
                    scripts.append(script)
                result = [result[-1]]

        ordered_selection.previous = result
        selection.set(result)

        return scripts

    @reactive.effect
    @reactive.event(selection)
    def _():
        if selection() != selection_return():
            selection_return.set(selection())

    if disable_selection:
        return None
    else:
        return selection_return


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


@expressify
def set_input_text_numeric_update_on_enter_key():
    js_code = """
        var customInputBinding = new Shiny.InputBinding();
        $.extend(customInputBinding, {
            find: function(scope) {
                return $(scope).find('input[type="text"]:not([readonly]):not([disabled]), input[type="number"]:not([readonly]):not([disabled])');
            },
            getValue: function(el) {
                if ($(el).attr('type') === 'number') {
                    return parseFloat($(el).val());
                } else {
                    return $(el).val();
                }
            },
            setValue: function(el, value) {
                $(el).val(value);
            },
            subscribe: function(el, callback) {
                $(el).on('keydown', function(event) {
                    if (event.key === 'Enter') {
                        callback();
                    }
                });
            },
            unsubscribe: function(el) {
                $(el).off('keydown');
            }
        });

        Shiny.inputBindings.register(customInputBinding);
    """
    ui.head_content(ui.HTML(f"<script>{js_code}</script>"))

def get_client_url(input):
    d = input._map
    url = f"{d['.clientdata_url_protocol']()}//{d['.clientdata_url_hostname']()}:{d['.clientdata_url_port']()}{d['.clientdata_url_pathname']()}{d['.clientdata_url_search']()}"
    return url

def get_client_url_query_params(input):
    d = input._map
    qs = d['.clientdata_url_search']().strip("?")
    import urllib.parse
    parsed_qs = urllib.parse.parse_qs(qs)
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
