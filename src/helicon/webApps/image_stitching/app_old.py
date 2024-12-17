import numpy as np

from shiny import reactive, req
from shiny.express import input, ui, render
from shinywidgets import render_plotly

import compute

params = reactive.value(None)

data_all = reactive.value(None)
abundance = reactive.value([])
image_size = reactive.value(0)

displayed_class_ids = reactive.value([])
displayed_class_images = reactive.value([])
displayed_class_labels = reactive.value([])

initial_selected_image_indices = reactive.value([0])
selected_image_indices = reactive.value([])
selected_images = reactive.value([])
selected_image_labels = reactive.value([])

stitched = reactive.value([])
stitched_image_label=reactive.value(["stitched image"])
stitched_image_size = reactive.value(0)

selected_helices = reactive.value(([], [], 0))
retained_helices_by_length = reactive.value([])
pair_distances = reactive.value([])


ui.head_content(ui.tags.title("Image Stitching"))
ui.tags.style(
    """
    * { font-size: 10pt; padding:0; border: 0; margin: 0; }
    """
)
urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_data.star",
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
    )
}
url_key = "empiar-10940_job010"

with ui.sidebar(width="33vw"):
    ui.input_radio_buttons(
        "input_mode_params",
        "How to obtain the Class2D parameter file:",
        choices=["upload", "url"],
        selected="url",
        inline=True,
    )
    with ui.panel_conditional("input.input_mode_params === 'upload'"):
        ui.input_file(
            "upload_params",
            "Upload the class2d parameters in a RELION star or cryoSPARC cs file",
            accept=[".star", ".cs"],
            placeholder="star or cs file",
        )

    with ui.panel_conditional("input.input_mode_params === 'url'"):
        ui.input_text(
            "url_params",
            "Download URL for a RELION star or cryoSPARC cs file",
            value=urls[url_key][0],
        )

    ui.input_radio_buttons(
        "input_mode_classes",
        "How to obtain the class average images:",
        choices=["upload", "url"],
        selected="url",
        inline=True,
    )
    with ui.panel_conditional("input.input_mode_classes === 'upload'"):
        ui.input_file(
            "upload_classes",
            "Upload the class averages in MRC format (.mrcs, .mrc)",
            accept=[".mrcs", ".mrc"],
            placeholder="mrcs or mrc file",
        )

    with ui.panel_conditional("input.input_mode_classes === 'url'"):
        ui.input_text(
            "url_classes",
            "Download URL for a RELION or cryoSPARC Class2D output mrc(s) file",
            value=urls[url_key][1],
        )

    ui.input_task_button("run", label="Run")

    with ui.div(style="max-height: 50vh; overflow-y: auto;"):
        selected_image_indices = compute.image_select(
            id="select_classes",
            label="Select classe(s):",
            images=displayed_class_images,
            image_labels=displayed_class_labels,
            image_size=reactive.value(128),
            initial_selected_indices=initial_selected_image_indices,
        )

        @reactive.effect
        @reactive.event(selected_image_indices)
        def update_selected_images():
            selected_images.set(
                [displayed_class_images()[i] for i in selected_image_indices()]
            )
            selected_image_labels.set(
                [displayed_class_labels()[i] for i in selected_image_indices()]
            )
    ui.input_checkbox(
        "ignore_blank", "Ignore blank classes", value=True
    )
    ui.input_checkbox(
        "sort_abundance",
        "Sort the classes by abundance",
        value=True,
    )


title = "Image Stitching"
ui.h1(title)

with ui.card():
    compute.image_select(
        id="display_selected_image",
        label="Selected classe(s):",
        images=selected_images,
        image_labels=selected_image_labels,
        image_size=image_size,
        disable_selection=True,
    )
    ui.input_numeric("trim_top_bottom", label="Trim on top and bottom (Pixels)",min=0,value=20,step=1)
    ui.input_numeric("rot_search_range", label="Rotation search range (Degrees)",min=0,value=15,step=1.0)
    ui.input_task_button("initiate_stitching", label="Stitch selected images")

with ui.card():
    compute.image_select(
        id="display_stitched_image",
        label="Stitched image:",
        images=stitched,
        image_labels=stitched_image_label,
        image_size=stitched_image_size,
        disable_selection=True,
    )   

@reactive.effect
@reactive.event(input.run)
def get_class2d_from_upload():
    req(input.input_mode_classes() == "upload")
    fileinfo = input.upload_classes()
    class_file = fileinfo[0]["datapath"]
    try:
        data, apix = compute.get_class2d_from_file(class_file)
        nx = data.shape[-1]
    except:
        data, apix = None, 0
        nx = 0
        m = ui.modal(
            f"failed to read the uploaded 2D class average images from {fileinfo[0]['name']}",
            title="File upload error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    data_all.set(data)
    image_size.set(nx)


@reactive.effect
@reactive.event(input.run)
def get_class2d_from_url():
    req(input.input_mode_classes() == "url")
    req(len(input.url_classes()) > 0)
    url = input.url_classes()
    try:
        data, apix = compute.get_class2d_from_url(url)
        nx = data.shape[-1]
    except:
        data, apix = None, 0
        nx = 0
        m = ui.modal(
            f"failed to download 2D class average images from {input.url_classes()}",
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
    data_all.set(data)
    image_size.set(nx)


@reactive.effect
@reactive.event(params, data_all, input.ignore_blank, input.sort_abundance)
def get_displayed_class_images():
    req(params() is not None)
    req(data_all() is not None)
    data = data_all()
    n = len(data)
    images = [data[i] for i in range(n)]
    image_size.set(max(images[0].shape))

    try:
        df = params()
        abundance.set(compute.get_class_abundance(df, n))
    except Exception:
        print(Exception)
        m = ui.modal(
            f"Failed to get class abundance from the provided Class2D parameter and  image files. Make sure that the two files are for the same Class2D job",
            title="Information error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)
        return None

    display_seq_all = np.arange(n, dtype=int)
    if input.sort_abundance():
        display_seq_all = np.argsort(abundance())[::-1]

    if input.ignore_blank():
        included = []
        for i in range(n):
            image = images[display_seq_all[i]]
            if np.max(image) > np.min(image):
                included.append(display_seq_all[i])
        images = [images[i] for i in included]
    else:
        included = display_seq_all
    image_labels = [f"{i+1}: {abundance()[i]:,d}" for i in included]

    displayed_class_ids.set(included)
    displayed_class_labels.set(image_labels)
    displayed_class_images.set(images)


@reactive.effect
@reactive.event(input.run)
def get_params_from_upload():
    req(input.input_mode_params() == "upload")
    fileinfo = input.upload_params()
    param_file = fileinfo[0]["datapath"]
    if len(fileinfo) == 2:
        cs_pass_through_file = fileinfo[1]["datapath"]
        assert cs_pass_through_file.endswith(".cs")
    else:
        cs_pass_through_file = None
    try:
        tmp_params = compute.get_class2d_params_from_file(
            param_file, cs_pass_through_file
        )
    except:
        tmp_params = None
    params.set(tmp_params)

    if params() is None:
        m = ui.modal(
            f"failed to parse the upload class2D parameters from {fileinfo[0]['name']}",
            title="File upload error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)


@reactive.effect
@reactive.event(input.run)
def get_params_from_url():
    req(input.input_mode_params() == "url")
    url = input.url_params()
    try:
        tmp_params = compute.get_class2d_params_from_url(url)
    except:
        tmp_params = None
    params.set(tmp_params)

    if params() is None:
        m = ui.modal(
            f"failed to download class2D parameters from {input.url_params()}",
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)

selected_helices_min_len = reactive.value(([[], [], 0], 0))

@reactive.effect
@reactive.event(selected_image_indices, params)
def get_selected_helices():
    req(params() is not None)
    req(image_size())
    req(len(abundance()))
    class_indices = [displayed_class_ids()[i] for i in selected_image_indices()]
    helices = compute.select_classes(params=params(), class_indices=class_indices)
    if len(helices):
        class_indices2 = (
            np.unique(
                np.concatenate([h["rlnClassNumber"] for hi, h in helices])
            ).astype(int)
            - 1
        )
        assert set(class_indices) == set(class_indices2)

    if len(helices):
        filement_lengths = compute.get_filament_length(helices=helices)
        segments_count = np.sum([abundance()[i] for i in class_indices])
    else:
        filement_lengths = []
        segments_count = 0

    selected_helices.set((helices, filement_lengths, segments_count))
    if not input.auto_min_len():
        selected_helices_min_len.set((selected_helices(), input.min_len()))


@reactive.effect
@reactive.event(selected_helices)
def auto_set_filament_min_len():
    req(input.auto_min_len() is True)
    helices, filament_lengths, segments_count = selected_helices()
    _, min_len_tmp = compute.compute_pair_distances(
        helices=helices, lengths=filament_lengths, target_total_count=1000
    )
    min_len_tmp = int(min_len_tmp)
    ui.update_numeric("min_len", value=min_len_tmp)
    selected_helices_min_len.set((selected_helices(), min_len_tmp))


@reactive.effect
@reactive.event(input.min_len)
def update_selected_helices_min_len():
    selected_helices_min_len.set((selected_helices(), input.min_len()))


@reactive.effect
@reactive.event(selected_helices_min_len, input.max_len)
def select_helices_by_length():
    previous = getattr(select_helices_by_length, "previous", ([], 0))
    selected_image_indices_previous, min_len_previous = previous
    (helices, filement_lengths, _), min_len = selected_helices_min_len()
    req(
        set(selected_image_indices_previous) != set(selected_image_indices())
        or min_len_previous != min_len
    )
    if len(helices) == 0:
        retained_helices_by_length.set([])
    elif min_len == 0 and input.max_len() <= 0:
        retained_helices_by_length.set(helices)
    else:
        helices_retained, n_ptcls = compute.select_helices_by_length(
            helices=helices,
            lengths=filement_lengths,
            min_len=min_len,
            max_len=input.max_len(),
        )
        retained_helices_by_length.set(helices_retained)
    select_helices_by_length.previous = (selected_image_indices(), min_len)


@reactive.effect
@reactive.event(retained_helices_by_length)
def get_pair_lengths():
    if len(retained_helices_by_length()):
        dists, _ = compute.compute_pair_distances(helices=retained_helices_by_length())
        pair_distances.set(dists)
    else:
        pair_distances.set([])


float_vars = dict(
    max_len=-1,
    max_pair_dist=-1,
    min_len=0,
    rise=4.75,
)
int_vars = dict(
    auto_min_len=1, bins=100, ignore_blank=1, show_sharable_url=0, sort_abundance=1
)
str_vars = dict(
    input_mode_classes="url",
    input_mode_params="url",
    url_params=urls[url_key][0],
    url_classes=urls[url_key][1],
)
all_input_vars = list(float_vars.keys()) + list(int_vars.keys()) + list(str_vars.keys())
reactive_vars_in = dict(select=(initial_selected_image_indices, int))
reactive_vars_out = dict(selected_image_indices=(selected_image_indices, [0], "select"))

connection_made = reactive.Value(False)


@reactive.effect
@reactive.event(lambda: not connection_made())
def apply_initial_params_from_browser_url():
    d = compute.get_client_url_query_params(input=input, keep_list=True)
    for k, v in d.items():
        if k in float_vars:
            v = list(map(float, v))
            if v[0] != float_vars[k]:
                if k in input:
                    ui.update_numeric(k, value=v[0])
        elif k in int_vars:
            v = list(map(int, v))
            if v[0] != int_vars[k]:
                if k in input:
                    ui.update_numeric(k, value=v[0])
        elif k in str_vars:
            if k in input:
                ui.update_text(k, value=v[0])
        elif k in reactive_vars_in:
            var, val_type = reactive_vars_in[k]
            v = list(map(val_type, v))
            var.set(v)
    if input.input_mode_params() == "url" and input.input_mode_classes() == "url":
        script = ui.tags.script(f"""document.getElementById('run').click();""")
        ui.insert_ui(ui=script, selector="body", where="afterEnd")


@render.ui
@reactive.event(
    *([input[k] for k in all_input_vars] + [v[0] for v in reactive_vars_out.values()])
)
def update_browser_url():
    if input.show_sharable_url():
        d = {}
        d.update(
            {
                k: float(input[k]())
                for k in float_vars
                if float_vars[k] != float(input[k]())
            }
        )
        d.update(
            {k: int(input[k]()) for k in int_vars if int_vars[k] != int(input[k]())}
        )
        d.update({k: input[k]() for k in str_vars if str_vars[k] != input[k]()})
        d.update(
            {
                var_url: var()
                for k, (var, val, var_url) in reactive_vars_out.items()
                if val != var()
            }
        )
        d = {k: d[k] for k in sorted(d.keys())}
    else:
        d = {}
    script = compute.set_client_url_query_params(query_params=d)
    return script
    

row = 0
col = 1
horizontal_trim = 0
vertical_trim = 20
    
@reactive.effect
@reactive.event(input.initiate_stitching)
def merge_multiple_images():
    if selected_images() is not None:
        images = selected_images()
        if len(images) > 0:
            image_iter = iter(images)
            #res=auto_trim_image(next(image_iter).copy(), col)
            res=trim_and_shift_image(next(image_iter).copy(), vertical_trim, row)
            for image in image_iter:
                res, shift, cc_max, error = merge_images(res, image.copy())
            stitched.set([res])
            stitched_image_size.set(max(np.shape(res))*100)
    else:
        stitched.set([])

def merge_images(image1, image2):
    from scipy.fft import fftn, ifftn, fftfreq
    
    max_degrees = 15
    image2 = auto_trim_image(image2, col)
    shape_difference = tuple(np.subtract(image1.shape,image2.shape))
    [shift, cc_max, error], image2 = get_best_rotation(image1, image2, max_degrees, shape_difference)
    
    return stitch_images(image1, image2, shift), shift, cc_max, error

def get_best_rotation(image1, image2, max_rotation, shape_difference):
    from skimage.transform import rotate
    from numpy import flip

    best_image = trim_and_shift_image(image2, vertical_trim, row)
    best_cc_data = ((0,0), 0, 0)
    best_degree = 0
    # image1 = trim_and_shift_image(image1, vertical_trim, row)

    for i in range(-max_rotation, max_rotation+1):
        check_image = trim_and_shift_image(rotate(image2, i), vertical_trim, 0)
        cc_data = xcross_ski(image1, check_image, shape_difference)
        for i in range(0,2):
            for j in range(0,2):
                check_flipped_image = check_image
                if i == 1:
                    check_flipped_image = flip(check_flipped_image, row)
                if j == 1:
                    check_flipped_image = flip(check_flipped_image, col)

                cc_flipped_data = xcross_ski(image1, check_flipped_image, shape_difference)
            if cc_flipped_data[1] > cc_data[1]:
                cc_data = cc_flipped_data
                check_image = check_flipped_image
        if cc_data[1] > best_cc_data[1]:
            best_cc_data = cc_data
            best_image = check_image
    return best_cc_data, best_image

def xcross_ski(image, other_image, shape_difference):
    from skimage.registration._phase_cross_correlation import _compute_error
    if shape_difference[col] > 0:
        other_image = shift_image(other_image, -shape_difference[row], row)
        other_image = shift_image(other_image, -shape_difference[col], col)
    elif shape_difference[col] < 0:
        image = shift_image(image, shape_difference[row], row)
        image = shift_image(image, shape_difference[col], col)
        
    # Compute the Fourier transforms of the images
    image_fft = np.fft.fft2(image)
    other_image_fft = np.fft.fft2(other_image)

    # Compute the cross-power spectrum (complex conjugate)
    cross_power_spectrum = image_fft * np.conj(other_image_fft)

    # Compute the inverse Fourier transform of the cross-power spectrum
    cross_correlation = np.sqrt(np.fft.ifft2(cross_power_spectrum)).real

    shift, maxima = get_shift(cross_correlation,image.shape)
    maximum = cross_correlation[maxima]

    src_amp = np.sum(np.real(image_fft * image_fft.conj()))
    src_amp /= image_fft.size
    target_amp = np.sum(np.real(other_image_fft * other_image_fft.conj()))
    target_amp /= other_image_fft.size
    error = _compute_error(maximum, src_amp,target_amp)

    return (int(shift[row]), int(shift[col])), maximum, error

def get_shift(cc_image, shape):

    maxima = np.unravel_index(np.argmax(np.abs(cc_image)), shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])
    float_dtype = cc_image.real.dtype
    shifts = np.stack(maxima).astype(float_dtype, copy=False)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    return shifts, maxima

#@st.cache_data(show_spinner=False)
def stitch_images(image1, image2, shift):
    merge_start = (0, 0)
    merge_end = (0, 0)

    merge_start, merge_end = find_overlap(image1, image2, (0,shift[col]))

    image1, image2 = frame_images(image1, image2, shift)
    image2 = balance_brightness(image1, image2, merge_start, merge_end)
    merged_image = image1 + image2
    for i in range(merge_start[row], merge_end[row]):
        for j in range(merge_start[col], merge_end[col]):
            merged_image[i][j] = merged_image[i][j] / 2
    return merged_image

def find_overlap(image1, image2, shift):
    merge_start = (0, abs(shift[col]))
    if image2.shape[col] + shift[col] > image1.shape[col]:
        merge_end = (image1.shape[row], image1.shape[col])
    else:
        merge_end = (image1.shape[row], image2.shape[col]+max(0, shift[col]))

    return merge_start, merge_end

def balance_brightness(template_image, image, overlap_start, overlap_end):
    from scipy.linalg import lstsq
    A = to_coordinates(image, overlap_start, overlap_end, 2)
    b = to_coordinates(template_image, overlap_start, overlap_end, 1)
    if A == [] or b == []:
        return image
    else:
        x = lstsq(np.array(A), np.array(b))[0]
        matrix = to_coordinates(image, (0,0), image.shape, 2) @ x
        new_image = to_image(matrix, image.shape)
        return new_image

def to_coordinates(image, start, end, width):
    matrix = []
    for i in range(start[row], end[row]):
        for j in range(min(start[col], end[col]), max(start[col], end[col])):
                if width == 2:
                    matrix.append(np.array([1, image[i][j]]))
                else:
                    matrix.append(image[i][j])

    return matrix

def to_image(matrix, shape):
    image = [ [0]*shape[col] for i in range(shape[row])]
    if matrix.ndim == 1:
        matrix_iter = iter(matrix)
    else:
        matrix_iter = iter(matrix[:, 1])
    for i in range(0, shape[row]):
        for j in range(0, shape[col]):
                image[i][j] = next(matrix_iter)
    return image


def frame_images(image1, image2, shift):
    end_extend = shift[col] + image2.shape[col] - image1.shape[col]
    front_extend = shift[col]
    image1 = shift_image(image1, max(0, -front_extend), col)
    image2 = shift_image(image2, max(0, front_extend), col)
    image1 = shift_image(image1, min(0, -end_extend), col)
    image2 = shift_image(image2, min(0, end_extend), col)
    image2 = cut_and_shift_image(image2, shift[row], row)
    return image1, image2

# def norm_image(image):

#     norm = np.linalg.norm(image, 1)
    
#     # normalized matrix
#     image = image/norm  

#     return image

def shift_image(image, shift, axis):

    if axis == 1:
        add = np.zeros((len(image) , abs(shift)))
    else:
        add = np.zeros((abs(shift) , len(image[0])))

    if shift > 0:
        image = np.concatenate((add , image), axis)
    elif shift < 0:
        image = np.concatenate((image , add), axis)

    return image

def cut_image(image, cut_length, axis):

    if cut_length > 0:
        if axis == 1:
            image = np.delete(image, range(len(image[0])-cut_length , len(image[0])), axis)
        else:
            image = np.delete(image, range(len(image)-cut_length , len(image)), axis)
    elif cut_length < 0:
        image = np.delete(image, range(0 , abs(cut_length)), axis)
    
    return image

def cut_and_shift_image(image, shift, axis):
    image = cut_image(shift_image(image, shift, axis), shift, axis)
    
    return image

def auto_trim_image(image, axis):
    i = 40
    image = cut_image(image, -i, axis)
    image = cut_image(image, i, axis)

    return image
    
def trim_image(image, cut_length, axis):
    
    image = cut_image(image, cut_length, axis)
    image = cut_image(image, -cut_length, axis)
    
    return image

def trim_and_shift_image(image, cut_length, axis):
    
    image = trim_image(image, cut_length, axis)
    image = shift_image(image, cut_length, axis)
    image = shift_image(image, -cut_length, axis)
    
    return image
