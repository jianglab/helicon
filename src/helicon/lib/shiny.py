from shiny import reactive
from shiny.express import ui, module, render

@module
def image_select(input, output, session, id="image_selector", label="", images=[], image_labels=[], image_size=128, margin_image=10, margin_box=20):
    # return indices of selected image in the order of selection
    images_final = []
    for i, image in enumerate(images):
        if isinstance(image, str):
            tmp = image
        elif isinstance(image, np.ndarray) and image.ndim == 2:
            tmp = encode_numpy(image)
        else:
            raise ValueError("image must be an image file or a 2D numpy array")
        images_final.append(tmp)

    assert len(image_labels)==0 or len(image_labels) == len(images)

    if len(image_labels):
        image_labels_final = image_labels
    else:
        image_labels_final = list(range(1, len(images)+1))

    ui.div(
        ui.input_checkbox_group(
            id=id,
            label=label,
            choices={
                i+1: ui.img(src=image, alt=f"Image {i+1}", title=str(image_labels_final[i]), style=f"width: {image_size}px; height: {image_size}px; object-fit: cover; margin-bottom: {margin_image}px;")
                for i, image in enumerate(images_final)
            },
            inline=True
        ),
        style=f"display: flex; justify-content: space-around; align-items: flex-start; margin-bottom: {margin_box}px"
    )
    
    ordered_selection = reactive.value([])

    @render.text
    @reactive.event(input[id])
    def selected_indices():
        current = input[id]()
        if current:
            previous = ordered_selection()
            tmp = [i for i in previous if i in current]
            tmp+= [i for i in current if i not in tmp]
            ordered_selection.set(tmp)
        else:
            ordered_selection.set([])
    
    return ordered_selection
