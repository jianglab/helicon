from shiny import reactive
from shiny.express import ui, module

@module
def image_select(input, output, session, label="Select Image(s):", images=[], display_image_labels=True, image_labels=[], image_size=128, image_border=2, gap=0):
    import numpy as np
    from PIL import Image
    from helicon import encode_numpy, encode_PIL_Image
    images_final = []
    for i, image in enumerate(images):
        if isinstance(image, str):
            tmp = image
        elif isinstance(image, Image.Image):
            tmp = encode_PIL_Image(image)
        elif isinstance(image, np.ndarray) and image.ndim == 2:            
            tmp = encode_numpy(image)
        else:
            raise ValueError("image must be an image file, a PIL Image, or a 2D numpy array")
        images_final.append(tmp)

    assert len(image_labels)==0 or len(image_labels) == len(images)

    if len(image_labels):
        image_labels_final = image_labels
    else:
        image_labels_final = list(range(1, len(images)+1))

    bids = []
    
    with ui.hold() as ui_imags:
        with ui.div(style=f"display: flex; flex-flow: row wrap; justify-content: center; justify-items: center; align-items: center; gap: {gap}px {gap}px; margin: 0 0 {image_border}px 0"):
            for i, image in enumerate(images_final):
                    bid = f"image_select_{i+1}"
                    bids.append(bid)

                    with ui.hold() as img:
                        ui.img(
                            src=image,
                            alt=f"Image {i+1}",
                            title=str(image_labels_final[i]),
                            style=f"object-fit: contain; max-width: {image_size-image_border*2}px; max-height: {image_size-image_border*2}px; border: {image_border}px solid transparent;"
                        )

                    ui.input_action_button(
                        id=bid,
                        label=
                            ui.div(
                                img,
                                ui.p(image_labels_final[i], style="text-align: left; color: white; text-shadow: -1px -1px 0.5px rgba(0,0,0,0.5), 1px -1px 0.5px rgba(0,0,0,0.5), -1px 1px 0.5px rgba(0,0,0,0.5), 1px 1px 0.5px rgba(0,0,0,0.5); position: absolute; top: 2px; left: 5px;"),
                                style="position: relative;"
                            ) if display_image_labels else
                            img,
                        style=f"padding: 0px; border: 0px; margin: 0px; background-color: transparent;",
                        onmouseover=f"if (this.querySelector('img').style.border !== '{image_border}px solid red') {{this.querySelector('img').style.border='{image_border}px solid blue'; this.querySelector('p').style.color='blue';}}",
                        onmouseout =f"if (this.querySelector('img').style.border !== '{image_border}px solid red') {{this.querySelector('img').style.border='{image_border}px solid transparent';  this.querySelector('p').style.color='white';}}"
                    )

    if len(label):
        ui.div(
            ui.h4(label, style="text-align: center; margin: 0;"),
            ui_imags,
            style=f"display: flex; flex-direction: column; gap: {gap}px; margin: 0"
        )
    else:
        ui_imags
    
    for bid in bids:
        bid_js = f"{session.ns}-{bid}"
        ui.tags.script(f'''
            document.getElementById("{bid_js}").addEventListener("click", function() {{
                var img  = document.getElementById("{bid_js}").querySelector("img");
                var text = document.getElementById("{bid_js}").querySelector("p");
                img.style.border = img.style.border === "{image_border}px solid red" ? "{image_border}px solid transparent" : "{image_border}px solid red";
                if (text) {{
                    text.style.color = img.style.border === "{image_border}px solid red" ? "red" : "white";
                }}
            }});
        ''')

    @reactive.Calc
    def image_buttons_status():
        return [input[bid]() % 2 == 1 for bid in bids]
    
    selection = reactive.value([])

    @reactive.effect
    def ordered_selection():
        status = image_buttons_status()
        current = [i+1 for i, is_selected in enumerate(status) if is_selected]        
        previous = getattr(ordered_selection, 'previous', [])        
        result = [i for i in previous if i in current]
        result += [i for i in current if i not in result]
        ordered_selection.previous = result
        selection.set(result)
        
    return selection
