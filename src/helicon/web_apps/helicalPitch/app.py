import numpy as np
import helicon
import compute

from shiny import reactive, req
from shiny.express import input, ui, render
from shinywidgets import render_plotly

params_orig = reactive.value(None)
params_work = reactive.value(None)
apix_micrograph_auto = reactive.value(0)
apix_micrograph_auto_source = reactive.value(None)
apix_micrograph = reactive.value(0)
apix_particle = reactive.value(0)

data_all = reactive.value(None)
abundance = reactive.value([])
apix_class = reactive.value(0)
image_size = reactive.value(0)

displayed_class_ids = reactive.value([])
displayed_class_images = reactive.value([])
displayed_class_labels = reactive.value([])

min_len = reactive.value(0)
selected_image_indices = reactive.value([])
selected_images = reactive.value([])
selected_image_labels = reactive.value([])

selected_helices = reactive.value(([], [], 0))
retained_helices_by_length = reactive.value([])
pair_distances = reactive.value([]) 


ui.head_content(ui.tags.title("HelicalPitch"))
helicon.shiny.google_analytics(id="G-998MGRETTF")
ui.tags.style(
    """
    * { font-size: 10pt; margin: 0; padding: 0; }
    input[type="text"].form-control, input[type="number"].form-control, .form-control, .selectize-input, .shiny-input-container, .form-group, .irs, .radio, .radio-inline, .selectize-control, label, .radio label, .radio-inline label, .accordion-button, .accordion-body, .shiny-text-output, .shiny-html-output, .shiny-output-error {font-size: 10pt; margin: 0; padding: 0; }
    """
)
urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_data.star",
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
    )
}
url_key = "empiar-10940_job010"

with ui.sidebar(width=456):
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
        # https://stackoverflow.com/questions/31415301/shiny-responds-to-enter

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

    with ui.div(style="max-height: 40vh; overflow-y: auto;"):
        selected_image_indices = helicon.shiny.image_select(
            id="select_classes",
            label="Select classe(s):",
            images=displayed_class_images,
            image_labels=displayed_class_labels,
            image_size=reactive.value(128),
            auto_select_first=True,
        )

        @reactive.effect
        def _():
            selected_images.set(
                [displayed_class_images()[i] for i in selected_image_indices()]
            )
            selected_image_labels.set(
                [displayed_class_labels()[i] for i in selected_image_indices()]
            )

    with ui.layout_columns(col_widths=[6, 6], style="align-items: flex-end;"):
        ui.input_numeric(
            "apix_particle",
            "Pixel size of particles (Å/pixel)",
            min=0.1,
            max=100.0,
            value=None,
            step=0.001,
        )
        ui.input_numeric(
            "apix_micrograph",
            "Pixel size of micrographs for particle extraction (Å/pixel)",
            min=0.1,
            max=100.0,
            value=None,
            step=0.001,
        )

    @render.ui
    def display_warning_apix_micrograph():
        msg = "Please carefully verify the pixel size of the micrographs used for particle picking/extraction."
        return ui.markdown(f"<span style='color: red;'>{msg}</span>")


title = "HelicalPitch: determine helical pitch/twist using 2D Classification info"
ui.h1(title)

with ui.layout_columns(col_widths=(5, 7, 12)):
    with ui.card():
        helicon.shiny.image_select(
            id="display_selected_image",
            label="Selected classe(s):",
            images=selected_images,
            image_labels=selected_image_labels,
            image_size=image_size,
            disable_selection=True,
        )

        with ui.layout_columns(col_widths=[12, 12], style="align-items: flex-end;"):

            @render_plotly
            @reactive.event(selected_helices, input.bins)
            def lengths_histogram_display():
                req(input.bins() is not None and input.bins() > 0)
                helices, lengths, count = selected_helices()
                data = lengths
                class_indices = [
                    str(displayed_class_ids()[i] + 1) for i in selected_image_indices()
                ]
                title = f"Filament Lengths: Class {' '.join(class_indices)}<br><i>{len(helices):,} filaments | {count:,} segments</i>"
                xlabel = "Filament Legnth (Å)"
                ylabel = "# of Filaments"
                log_y = True
                nbins = input.bins()
                fig = compute.plot_histogram(
                    data=data,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    bins=nbins,
                    log_y=log_y,
                )

                return fig

            with ui.layout_columns(
                col_widths=[6, 6, 12], style="align-items: flex-end;"
            ):
                ui.input_numeric(
                    "min_len", "Minimal length (Å)", min=0.0, value=0, step=1.0
                )
                ui.input_numeric(
                    "rise",
                    "Helical rise (Å)",
                    min=0.01,
                    max=1000.0,
                    value=4.75,
                    step=0.01,
                )
                with ui.accordion(id="additional_parameters", open=False):
                    with ui.accordion_panel(title="Additional parameters:"):
                        with ui.layout_columns(
                            col_widths=6, style="align-items: flex-end;"
                        ):
                            ui.input_checkbox(
                                "ignore_blank", "Ignore blank classes", value=True
                            )
                            ui.input_checkbox(
                                "sort_abundance",
                                "Sort the classes by abundance",
                                value=True,
                            )
                            ui.input_checkbox(
                                "auto_min_len",
                                "Auto-set minimal filament length",
                                value=True,
                            )
                            ui.input_numeric(
                                "max_len",
                                "Maximal length (Å)",
                                min=-1,
                                value=-1,
                                step=1.0,
                            )
                            ui.input_numeric(
                                "max_pair_dist",
                                "Maximal pair distance (Å) to plot",
                                min=-1,
                                value=-1,
                                step=1.0,
                            )
                            ui.input_numeric(
                                "bins",
                                "Number of histogram bins",
                                min=1,
                                value=100,
                                step=1,
                            )

                @reactive.effect
                @reactive.event(input.min_len)
                def _():
                    ui.update_numeric("max_len", min=input.min_len())
                    if 0 < input.max_len() < input.min_len():
                        ui.update_numeric("max_len", value=-1)

                @reactive.effect
                @reactive.event(input.max_len)
                def _():
                    if input.max_len() > 0:
                        ui.update_numeric("min_len", max=input.max_len())
                    if input.min_len() >= input.max_len():
                        ui.update_numeric("min_len", value=0)

    with ui.card(max_height="90vh"):

        @render_plotly
        @reactive.event(pair_distances, input.bins, input.max_pair_dist, input.rise)
        def pair_distances_histogram_display():
            req(input.bins() is not None and input.bins() > 0)
            req(input.max_pair_dist() is not None)
            req(input.rise() is not None and input.rise() > 0)
            data = pair_distances()
            segment_count = np.sum([len(h) for hi, h in retained_helices_by_length()])
            class_indices = [
                str(displayed_class_ids()[i] + 1) for i in selected_image_indices()
            ]
            rise = input.rise()
            log_y = True
            title = f"Pair Distances: Class {' '.join(class_indices)}<br><i>{len(retained_helices_by_length())} filaments | {segment_count:,} segments | {len(pair_distances()):,} segment pairs"
            xlabel = "Pair Distance (Å)"
            ylabel = "# of Pairs"
            nbins = input.bins()
            max_pair_dist = input.max_pair_dist()

            fig = compute.plot_histogram(
                data=data,
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
                max_pair_dist=max_pair_dist,
                bins=nbins,
                log_y=log_y,
                show_pitch_twist=dict(rise=rise, csyms=(1, 2, 3, 4)),
                multi_crosshair=True,
            )

            return fig

        ui.markdown(
            "**How to interpretate the histogram:** an informative histogram should have clear peaks with equal spacing. If so, hover your mouse pointer to the first major peak off the origin to align the vertial lines well with the peaks. Once you have decided on the line postion, read the hover text which shows the twist values assuming the pair-distance is the helical pitch (adjusted for the cyclic symmetries around the helical axis). You need to decide which cyclic symmetry and the corresponding twist should be used.  \n&nbsp;&nbsp;&nbsp;&nbsp;If the histogram does not show clear peaks, it indicates that the Class2D quality is bad. You might consider changing the 'Minimal length (Å)' from 0 to a larger value (for example, 1000 Å) to improve the peaks in the histogram. If that does not help, you might consider redoing the Class2D task with longer extracted segments (>0.5x helical pitch) from longer filaments (> 1x pitch)."
        )

        @render.ui
        def _():
            if len(pair_distances()) > 0:
                download_ui = render.download(
                    label="Download selected helices", filename="helices.star"
                )

                @download_ui
                def download_retained_helices():
                    req(retained_helices_by_length())
                    indices = np.concatenate(
                        [h.index for hi, h in retained_helices_by_length()]
                    )
                    params_to_save = params_orig().iloc[indices]
                    import io

                    output = io.StringIO()
                    helicon.dataframe2star(params_to_save, output)
                    output.seek(0)
                    yield output.read()

                return download_ui
            else:
                return None

    ui.markdown(
        "*Developed by the [Jiang Lab@Purdue University](https://jiang.bio.purdue.edu/HelicalPitch). Report problems to [HelicalPitch@GitHub](https://github.com/jianglab/HelicalPitch/issues)*"
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
    apix_class.set(apix)
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
    apix_class.set(apix)
    image_size.set(nx)


@reactive.effect
@reactive.event(params_work, data_all, input.ignore_blank, input.sort_abundance)
def get_displayed_class_images():
    req(params_work() is not None)
    req(data_all() is not None)
    data = data_all()
    n = len(data)
    images = [data[i] for i in range(n)]
    image_size.set(max(images[0].shape))

    try:
        df = params_work()
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
    params_orig.set(tmp_params)

    if params_orig() is not None:
        apix = compute.get_pixel_size(
            params_orig(), attrs=["blob/psize_A", "rlnImagePixelSize"]
        )
        if apix:
            ui.update_numeric("apix_particle", value=apix)
    else:
        ui.update_numeric("apix_particle", value=0)
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
    params_orig.set(tmp_params)

    if params_orig() is not None:
        apix = compute.get_pixel_size(
            params_orig(), attrs=["blob/psize_A", "rlnImagePixelSize"]
        )
        if apix:
            ui.update_numeric("apix_particle", value=apix)
    else:
        ui.update_numeric("apix_particle", value=0)
        m = ui.modal(
            f"failed to download class2D parameters from {input.url_params()}",
            title="File download error",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)


@reactive.effect
@reactive.event(params_orig)
def get_apix_micrograph_auto():
    req(params_orig() is not None)
    apix = compute.get_pixel_size(params_orig(), return_source=True)
    if apix is None:
        apix = input.apix_particle()
        source = "particle pixel size"
    else:
        apix, source = apix
    apix_micrograph_auto.set(apix)
    apix_micrograph_auto_source.set(source)
    ui.update_numeric("apix_micrograph", value=apix)


@reactive.effect
@reactive.event(params_orig, input.apix_micrograph)
def update_particle_locations():
    req(params_orig() is not None)
    req(input.apix_micrograph())
    cols_to_keep = "rlnImageName rlnMicrographName rlnImagePixelSize rlnCoordinateX rlnCoordinateY rlnHelicalTrackLengthAngst rlnHelicalTubeID rlnClassNumber rlnOriginX rlnOriginY rlnAnglePsi".split()
    cols = [col for col in cols_to_keep if col in params_orig()]
    tmp = params_orig().loc[:, cols]
    distseg = compute.estimate_inter_segment_distance(tmp)
    tmp["segmentid"] = compute.assign_segment_id(tmp, distseg)
    tmp = compute.update_particle_locations(
        params=tmp, apix_micrograph=input.apix_micrograph()
    )
    params_work.set(tmp)


@reactive.effect
@reactive.event(selected_image_indices, params_work)
def get_selected_helices():
    req(params_work() is not None)
    req(input.apix_particle())
    req(image_size())
    req(len(abundance()))
    class_indices = [displayed_class_ids()[i] for i in selected_image_indices()]
    helices = compute.select_classes(params=params_work(), class_indices=class_indices)

    if len(helices):
        filement_lengths = compute.get_filament_length(
            helices=helices,
            particle_box_length=input.apix_particle() * image_size(),
        )
        segments_count = np.sum([abundance()[i] for i in class_indices])
    else:
        filement_lengths = []
        segments_count = 0

    selected_helices.set((helices, filement_lengths, segments_count))


@reactive.effect
@reactive.event(input.min_len)
def get_filament_min_len():
    min_len.set(input.min_len())


@reactive.effect
@reactive.event(selected_helices, input.auto_min_len)
def auto_set_filament_min_len():
    req(input.auto_min_len() is True)
    helices, filament_lengths, segments_count = selected_helices()
    _, min_len_tmp = compute.compute_pair_distances(
        helices=helices, lengths=filament_lengths, target_total_count=1000
    )
    ui.update_numeric("min_len", value=np.floor(min_len_tmp))


def __select_helices_by_length():
    helices, filement_lengths, _ = selected_helices()
    if len(helices) == 0:
        retained_helices_by_length.set([])
    elif min_len() == 0 and input.max_len() <= 0:
        retained_helices_by_length.set(helices)
    else:
        helices_retained, n_ptcls = compute.select_helices_by_length(
            helices=helices,
            lengths=filement_lengths,
            min_len=min_len(),
            max_len=input.max_len(),
        )
        retained_helices_by_length.set(helices_retained)


@reactive.effect
@reactive.event(min_len, input.max_len)
def select_helices_by_length_with_auto_min_len():
    req(input.auto_min_len())
    __select_helices_by_length()


@reactive.effect
@reactive.event(selected_helices, min_len, input.max_len)
def select_helices_by_length_without_auto_min_len():
    req(not input.auto_min_len())
    __select_helices_by_length()


@reactive.effect
@reactive.event(retained_helices_by_length)
def get_pair_lengths():
    if len(retained_helices_by_length()):
        dists, _ = compute.compute_pair_distances(helices=retained_helices_by_length())
        pair_distances.set(dists)
    else:
        pair_distances.set([])
