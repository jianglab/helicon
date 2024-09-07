from shiny import reactive
from shiny.express import ui, module, render

@module
def image_select(input, output, session, images=[], image_labels=[], image_size=128, gap=0, margin_box=0):
    import numpy as np
    from PIL import Image
    images_final = []
    for i, image in enumerate(images):
        if isinstance(image, str):
            tmp = image
        elif isinstance(image, Image.Image):
            tmp = encode_PIL_Image(image)
        elif isinstance(image, np.ndarray) and image.ndim == 2:
            from helicon import encode_numpy, encode_PIL_Image
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

    with ui.layout_column_wrap(gap=f"{gap} px"):
        for i, image in enumerate(images_final):
            bid = f"image_select_{i+1}"
            bids.append(bid)
            ui.input_action_button(
                id=bid,
                label=ui.img(
                    src=image,
                    alt=f"Image {i+1}",
                    title=str(image_labels_final[i]),
                    style=f"width: 100%; height: 100%; object-fit: cover"
                ),
                style=f"width: {image_size}px; height: {image_size}px; padding: 0; margin: {margin_box}px;"
            )

    for bid in bids:
        bid_js = f"{session.ns}-{bid}"
        ui.tags.script(f'''
            document.getElementById("{bid_js}").addEventListener("click", function() {{
                var button = document.getElementById("{bid_js}");
                button.style.border = button.style.border === "2px solid blue" ? "2px solid white" : "2px solid blue";
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
