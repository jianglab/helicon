"""AbInitio3D tab — an ab initio 3D map of a helical filament from its 2D classes.

Each class average of one helical type is a view of the filament at an
azimuthal angle; the segments of a filament run through the classes in the
order of those angles, one repeat per turn of the ring. The tab finds the
repeat and the class azimuths from the segment pairs, assigns every segment
its Euler angles and origin, and builds a 3D map from the class averages or,
with relion_reconstruct, from the segments.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

import helicon
from shiny import reactive, ui, module, req, render
from shiny.module import resolve_id

from .. import bookmark, class2d_files, deployment, map_viewers

from ..lib import helical_pitch_compute as compute
from ..lib import helical_pitch_context as context
from ..lib import helical_pitch_lattice as lattice
from ..lib import helical_pitch_map as maps
from ..lib import helical_pitch_mixture as mixture
from ..lib import helical_pitch_phase as phase
from ..lib import helical_pitch_pixel as pixel
from ..lib import helical_pitch_types as types
from ..lib import helical_pitch_relion as relion

from .abinitio3d_ui import (  # noqa: E402,F401 (moved to abinitio3d_ui; also what callers and tests take from here)
    _MAP_METHODS,
    _MAP_PLOT_WIDTH,
    _MAP_STRIP_SHARE,
    _PITCH_ROW_CSS,
    _PLOT_BOX,
    _RESIZE_WITH_BOX,
    _RESULTS_BOX,
    _RESULTS_ORDER_CSS,
    _SELECT_ALL_JS,
    _SET_DISABLED_JS,
    _SUGGESTIONS_BOX,
    _TYPES_METHODS,
    _fig_to_html,
    _filament_joint_figure,
    _info,
    _map_figure,
    _mrc_bytes,
    _tip,
    _url_key,
    _urls,
    abinitio3d_tab_ui,
)


logger = logging.getLogger(__name__)


# The suggestions whose pixel agreement is predicted: the best-fitting ones, as each
# is registered against every class of the ring.
MAX_AGREEMENT_CANDIDATES = 40

BOOKMARK_DEFAULTS = {
    "mode_params": ("input_mode_params", "url"),
    "url_params": ("url_params", _urls[_url_key][0]),
    "mode_classes": ("input_mode_classes", "url"),
    "url_classes": ("url_classes", _urls[_url_key][1]),
    "server_params": ("server_params", ""),
    "server_classes": ("server_classes", ""),
    "ignore_blank": ("ignore_blank", True),
    "sort_abundance": ("sort_abundance", True),
    "rise": ("rise", 4.75),
    "min_twist": ("min_twist", 0.3),
    "merge_counterparts": ("merge_counterparts", True),
    "types_method": ("types_method", "markers"),
    # starts at the number of clear splits of the filaments, from the classes
    "types_n": ("types_n", 1, bookmark.DERIVED),
    "agree_cut": ("agree_threshold", 0.0),
    "csym": ("rot_fold", 1),
    "hand": ("map_hand", "left"),
    "map_method": ("map_method", "joint"),
    "n_boot": ("phase_n_boot", 20),
    "map_fit": ("repeat_map_fit", False),
    "split": ("split_axis_distance", 50),
    "two_rounds": ("map_refine", False),
    # suggested from the class fits of each result
    "fit_cut": ("fit_threshold", 0.1, bookmark.DERIVED),
}


def _class_positions(class_ids, ids):
    """Where in ``ids`` each of the 1-based ``class_ids`` is (``ids`` are 0-based)."""
    position = {int(c): i for i, c in enumerate(ids)}
    return [position[int(c) - 1] for c in class_ids]


# How the reasons for taking a class out are shown in a gallery label.
_SHORT_REASON = {
    phase.TOO_FEW: "too few pairs",
    phase.AT_CHANCE: "off the ring",
    phase.BELOW_THRESHOLD: "below the threshold",
    phase.PIXELS_DISAGREE: "pixels disagree",
}


def _box_text(shape):
    """The size of a map: ``N\u00b3`` for a cube, else ``nx \u00d7 ny \u00d7 nz``."""
    nz, ny, nx = (int(n) for n in shape)
    return f"{nx}\u00b3" if nx == ny == nz else f"{nx} \u00d7 {ny} \u00d7 {nz}"


def _counterpart_key(class_numbers, ids, images):
    """What a counterpart-pair computation is cached on.

    The pairs come from the images, not from the class numbers alone: a second
    dataset can show the same class numbers, and a cache keyed on those alone
    hands back the first dataset's pairs. The image sum is the cheap
    fingerprint of the data the pairs depend on.
    """
    return (
        tuple(sorted(class_numbers)),
        tuple(int(c) for c in ids),
        float(sum(np.sum(image) for image in images)),
    )


@module.server
def abinitio3d_tab_server(input, output, session):
    # the Class2D parameters as read, and as used: with tube ids that hold
    # several filaments split (helicon.split_distinct_filaments)
    params_raw = reactive.value(None)
    params = reactive.value(None)
    data_all = reactive.value(None)

    # Run is pressed for the user, when the tab starts, if both inputs are URLs;
    # an upload needs a file the user has yet to choose, so nothing runs for it.
    auto_run = reactive.value(0)

    abundance = reactive.value([])
    image_size = reactive.value(0)

    displayed_class_ids = reactive.value([])
    displayed_class_images = reactive.value([])
    displayed_class_title = reactive.value("Select class(es):")
    displayed_class_labels = reactive.value([])

    initial_selected_image_indices = reactive.value([0])
    # bumped whenever the code, not the user, changes the selection, so that the
    # gallery is redrawn even when the new selection equals the previous one
    gallery_version = reactive.value(0)
    selected_images = reactive.value([])
    selected_image_labels = reactive.value([])

    # ── Data loading ──

    @reactive.effect
    @reactive.event(input.run, auto_run, ignore_none=False, ignore_init=True)
    def get_class2d_from_upload():
        req(input.input_mode_classes() == "upload")
        fileinfo = input.upload_classes()
        req(fileinfo)
        class_file = fileinfo[0]["datapath"]
        try:
            data, apix = compute.get_class2d_from_file(class_file)
            nx = data.shape[-1]
        except Exception as e:
            logger.error("Failed to read uploaded class images: %s", e)
            data, apix, nx = None, 0, 0
            ui.modal_show(
                ui.modal(
                    f"failed to read the uploaded 2D class average images from {fileinfo[0]['name']}",
                    title="File upload error",
                    easy_close=True,
                    footer=None,
                )
            )
        data_all.set((data, apix))
        image_size.set(nx)

    @reactive.effect
    @reactive.event(input.run, auto_run, ignore_none=False, ignore_init=True)
    def get_class2d_from_url():
        req(input.input_mode_classes() in ("url", "server"))
        url = _source(
            input.input_mode_classes(), input.url_classes, input.server_classes
        )
        req(len(url) > 0)
        if input.input_mode_classes() == "server" and deployment.refuse_server_mode():
            return
        if deployment.refuse_local_path(url):
            return
        try:
            data, apix = compute.get_class2d_from_url(url)
            nx = data.shape[-1]
        except Exception as e:
            logger.error("Failed to download class images: %s", e)
            data, apix, nx = None, 0, 0
            ui.modal_show(
                ui.modal(
                    f"failed to read 2D class average images from {url}",
                    title="File download error",
                    easy_close=True,
                    footer=None,
                )
            )
        data_all.set((data, apix))
        image_size.set(nx)

    @reactive.effect
    @reactive.event(input.run, auto_run, ignore_none=False, ignore_init=True)
    def get_params_from_upload():
        req(input.input_mode_params() == "upload")
        fileinfo = input.upload_params()
        req(fileinfo)
        param_file = fileinfo[0]["datapath"]
        msg = None
        try:
            tmp_params = compute.get_class2d_helix_params_from_file(param_file)
        except Exception as e:
            msg = str(e).replace(param_file, fileinfo[0]["name"])
            tmp_params = None
        params_raw.set(tmp_params)
        if tmp_params is None:
            if msg is None:
                msg = f"failed to parse the upload class2D parameters from {fileinfo[0]['name']}"
            msg_ui = ui.markdown(
                msg.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br><br>")
            )
            ui.modal_show(
                ui.modal(
                    msg_ui, title="File upload error", easy_close=True, footer=None
                )
            )

    @reactive.effect
    @reactive.event(input.run, auto_run, ignore_none=False, ignore_init=True)
    def get_params_from_url():
        req(input.input_mode_params() in ("url", "server"))
        url = _source(input.input_mode_params(), input.url_params, input.server_params)
        req(len(url) > 0)
        if input.input_mode_params() == "server" and deployment.refuse_server_mode():
            return
        if deployment.refuse_local_path(url):
            return
        msg = None
        try:
            tmp_params = compute.get_class2d_helix_params_from_url(url)
        except Exception as e:
            msg = str(e)
            tmp_params = None
        params_raw.set(tmp_params)
        if tmp_params is None:
            if msg is None:
                msg = f"failed to read class2D parameters from {url}"
            msg_ui = ui.markdown(
                msg.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br><br>")
            )
            ui.modal_show(
                ui.modal(
                    msg_ui, title="File download error", easy_close=True, footer=None
                )
            )

    @reactive.effect
    @reactive.event(params_raw, input.split_axis_distance)
    def _split_distinct_filaments():
        raw = params_raw()
        if raw is None:
            params.set(None)
            return
        distance = input.split_axis_distance() or 0
        try:
            fixed, n0, n1 = compute.split_distinct_filaments(raw, distance)
        except (KeyError, ValueError) as e:
            logger.warning("Tube ids were not checked for distinct filaments: %s", e)
            fixed, n0, n1 = raw, 0, 0
        if n1 > n0:
            ui.notification_show(
                f"{n1 - n0:,} tube ids held more than one filament (segments over "
                f"{distance:g} \u00c5 from each other's axis) and were split: "
                f"{n1:,} filaments. The exported star file has the new ids.",
                duration=10,
            )
        params.set(fixed)

    def _source(mode, url_input, server_input):
        """The path or URL the mode reads from."""
        return (server_input() if mode == "server" else url_input()).strip()

    def _class2d_file_names():
        """What the parameter file and the class averages are called: a path, a
        URL, or the name of an upload."""

        def name(mode, url_input, server_input, upload_input):
            if mode == "upload":
                info = upload_input()
                return info[0]["name"] if info else ""
            return _source(mode, url_input, server_input)

        return (
            name(
                input.input_mode_params(),
                input.url_params,
                input.server_params,
                input.upload_params,
            ),
            name(
                input.input_mode_classes(),
                input.url_classes,
                input.server_classes,
                input.upload_classes,
            ),
        )

    def _class_counts(df, n_images):
        """``(classes in the parameters, classes in the images, highest class
        number)``, the numbers the failure message gives."""
        try:
            numbers = df["rlnClassNumber"].astype(int)
            return int(numbers.nunique()), int(n_images), int(numbers.max())
        except Exception:
            return None, int(n_images), None

    # Browse... in the server mode: pick the files on this computer
    if deployment.has_server_files():
        helicon.shiny.file_picker_fill(
            "params_browse",
            "server_params",
            input,
            ("*.star", "*.cs"),
            "Select the Class2D parameter file (RELION star or cryoSPARC cs)",
        )
        helicon.shiny.file_picker_fill(
            "classes_browse",
            "server_classes",
            input,
            ("*.mrcs", "*.mrc"),
            "Select the Class2D class averages (mrcs or mrc)",
        )

    # A server file chosen or typed in one of the two server fields brings the
    # other file of the same 2D classification with it, when it is beside it.
    def _fill_companion(given, other_id, other_value):
        if not deployment.has_server_files():  # no looking at the server's files
            return
        found = class2d_files.companion(given)
        if (
            found
            and found != other_value
            and class2d_files.needs_filling(other_value, given)
        ):
            ui.update_text(other_id, value=found)

    @reactive.effect
    @reactive.event(input.server_params)
    def _fill_classes_from_params():
        if input.input_mode_classes() == "server":
            _fill_companion(
                input.server_params(), "server_classes", input.server_classes()
            )

    @reactive.effect
    @reactive.event(input.server_classes)
    def _fill_params_from_classes():
        if input.input_mode_params() == "server":
            _fill_companion(
                input.server_classes(), "server_params", input.server_params()
            )

    # ── Build class gallery ──

    # created after the Run handlers, so that they see it change rather than
    # its first value
    auto_started = []

    @reactive.effect
    def _auto_run_at_start():
        # the reads wait for the inputs to reach the server, and the effect
        # goes again until they have; once is all it is for
        if auto_started:
            return
        modes = (input.input_mode_params(), input.input_mode_classes())
        both_urls = (
            all(m in ("url", "server") for m in modes)
            and bool(_source(modes[0], input.url_params, input.server_params))
            and bool(_source(modes[1], input.url_classes, input.server_classes))
        )
        auto_started.append(True)
        if both_urls:
            with reactive.isolate():
                auto_run.set(1)

    @reactive.effect
    @reactive.event(params, data_all, input.ignore_blank, input.sort_abundance)
    def get_displayed_class_images():
        req(params() is not None)
        req(data_all() is not None)
        data, apix = data_all()
        n = len(data)
        images = [data[i] for i in range(n)]
        image_size.set(max(images[0].shape))
        try:
            df = params()
            abundance.set(compute.get_class_abundance(df, n))
        except Exception as e:
            logger.error("Failed to get class abundance: %s", e)
            ui.modal_show(
                ui.modal(
                    class2d_files.abundance_error_message(
                        *_class2d_file_names(),
                        *_class_counts(df, n),
                    ),
                    title="Information error",
                    easy_close=True,
                    footer=None,
                )
            )
            return
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

        if not images:
            ui.modal_show(
                ui.modal(
                    "Every class average is blank, or the file has none: "
                    "there is nothing to show with the blank ones ignored.",
                    title="No classes to show",
                    easy_close=True,
                    footer=None,
                )
            )
            return

        image_labels = [f"{i+1}: {abundance()[i]:,d}" for i in included]
        displayed_class_ids.set(included)
        displayed_class_images.set(images)
        displayed_class_title.set(
            f"{len(included)}/{n} classes | {_counts_text(int(i) + 1 for i in included)}"
            f" | {images[0].shape[1]}x{images[0].shape[0]} pixels | {apix} \u00c5/pixel"
        )
        displayed_class_labels.set(image_labels)

    # the filaments the analysis is restricted to (Use group of the Types card):
    # {"label": ..., "keys": set of (micrograph, tube)}, or None for all
    type_filter = reactive.value(None)
    # the group Use group last chose (with what Add group added): the checkbox
    # Only the filaments of the type group in use switches to it and back
    type_group_used = reactive.value(None)

    @reactive.calc
    def analysis_params():
        """The Class2D parameters the analysis uses: all of them, or the segments
        of the filaments of the type group in use."""
        p = params()
        chosen = type_filter()
        if p is None or chosen is None:
            return p
        keys = list(zip(p["rlnMicrographName"].astype(str), p["rlnHelicalTubeID"]))
        keep = np.array([(m, int(t)) in chosen["keys"] for m, t in keys], dtype=bool)
        out = p.loc[keep].reset_index(drop=True)
        out.attrs = dict(p.attrs)
        return out

    # ── Sidebar class selection ──

    @reactive.calc
    def _segment_filament():
        """Filament number of every row of the Class2D parameters."""
        p = params()
        req(p is not None)
        return (
            p.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False)
            .ngroup()
            .values
        )

    @reactive.calc
    def _used_segment_filament():
        """Filament number of every row of :func:`analysis_params`."""
        p = analysis_params()
        req(p is not None)
        return (
            p.groupby(["rlnMicrographName", "rlnHelicalTubeID"], sort=False)
            .ngroup()
            .values
        )

    def _counts_text(class_numbers, in_use=False):
        """'N filaments | M segments' for the segments of these classes: of all
        the filaments, or (``in_use``) of those the analysis uses."""
        p = analysis_params() if in_use else params()
        if p is None:
            return ""
        mask = p["rlnClassNumber"].astype(int).isin(list(class_numbers)).values
        fil = _used_segment_filament() if in_use else _segment_filament()
        n_fil = len(np.unique(fil[mask]))
        return f"{n_fil:,} filaments | {int(mask.sum()):,} segments"

    def _gallery(gallery_id, images, labels, label="", selection=False, initial=None):
        """An image gallery, with the value of its input cleared on each draw."""
        parts = helicon.shiny.image_gallery(
            id=session.ns(gallery_id),
            label=reactive.value(label),
            images=reactive.value(images),
            image_labels=reactive.value(labels),
            image_size=reactive.value(128),
            initial_selected_indices=reactive.value(initial or []),
            enable_selection=selection,
            allow_multiple_selection=selection,
        )
        if parts is None:
            parts = ui.div()
        elif isinstance(parts, tuple):
            parts = ui.TagList(parts[0], *parts[1])
        if selection and not initial:
            # a redrawn gallery has nothing picked; the browser still holds the
            # old value of its input until told otherwise
            parts = ui.TagList(
                parts,
                ui.tags.script(
                    f"Shiny.setInputValue('{session.ns(gallery_id)}', [], "
                    "{priority: 'event'});"
                ),
            )
        return parts

    # Drawn even while hidden: Remove picked and the suggestions' Add buttons
    # change the selection by redrawing this gallery, and with the Parameters
    # tab in front of it a hidden output waited -- so the buttons did nothing
    # until the Inputs tab was shown again.
    @output(suspend_when_hidden=False)
    @render.ui
    def select_classes_gallery():
        gallery_version()
        return _gallery(
            "select_classes_inner",
            displayed_class_images(),
            displayed_class_labels(),
            label=displayed_class_title(),
            selection=True,
            initial=initial_selected_image_indices(),
        )

    def _set_selection(positions):
        """Select these classes (positions in the class gallery) in the sidebar."""
        initial_selected_image_indices.set(sorted(set(int(i) for i in positions)))
        with reactive.isolate():
            gallery_version.set(gallery_version() + 1)

    @reactive.effect
    @reactive.event(input.select_classes_inner)
    def update_selected_images():
        sel = input.select_classes_inner()
        if sel is None or len(sel) == 0:
            selected_images.set([])
            selected_image_labels.set([])
            return
        selected_images.set([displayed_class_images()[i] for i in sel])
        selected_image_labels.set([displayed_class_labels()[i] for i in sel])

    # ── pitch from the phases of the selected classes ──

    MAX_PHASE_CLASSES = 200
    phase_result = reactive.value(None)

    def _selected_class_numbers():
        sel = input.select_classes_inner() or []
        return [int(displayed_class_ids()[i]) + 1 for i in sel]

    @reactive.effect
    @reactive.event(analysis_params, input.select_classes_inner)
    def _clear_phase_result():
        # a result belongs to the selection it was computed from
        phase_result.set(None)

    counterpart_cache = {}

    def _counterpart_pairs(class_numbers, ids, images):
        """Pairs of the selected classes that are each other's 180-degree rotation.

        ``ids`` and ``images`` are the displayed class ids and images, read by
        the caller: this runs in a worker thread, which cannot read them.
        """
        key = _counterpart_key(class_numbers, ids, images)
        if key not in counterpart_cache:
            position = {int(c): i for i, c in enumerate(ids)}
            at = [position[int(c) - 1] for c in class_numbers if int(c) - 1 in position]
            found = phase.pair_counterparts(images, at)
            counterpart_cache[key] = [
                dict(
                    a=int(ids[d["a"]]) + 1,
                    b=int(ids[d["b"]]) + 1,
                    corr=d["corr"],
                    dx=d["dx"],
                )
                for d in found
            ]
        return counterpart_cache[key]

    # The pitch estimate, the suggestions and the two reconstructions take
    # from seconds to minutes: they run in worker threads (background_task),
    # so the event loop every session shares stays free. Their inputs are read
    # in the effect that starts them, and a result is applied only if it still
    # belongs to the data and selection it was computed from.

    def _phase_work(job, progress):
        counterparts = None
        if job["merge"]:
            progress.set(
                0, message="pairing classes that are turned copies of each other"
            )
            counterparts = _counterpart_pairs(
                job["class_ids"], job["ids"], job["images"]
            )
            progress.inc(1, message=f"{len(counterparts)} pair(s) of classes")
        result = phase.analyze(
            job["params"],
            class_ids=job["class_ids"],
            n_boot=job["n_boot"],
            progress=lambda msg: progress.inc(1, message=msg),
            counterparts=counterparts,
            image_apix=job["apix"],
            # a class average that does not repeat at half a candidate repeat rules
            # that candidate out (the segments alone cannot, when views are missing)
            class_images={int(c) + 1: im for c, im in zip(job["ids"], job["images"])},
            max_sep=job["max_repeat"],
            map_fit=job["map_fit"],
        )
        result.pixel_check = _pixel_check(job, result, progress)
        result.lattice = _lattice(job, result)
        return result

    def _lattice(job, result):
        """The subunit lattice the ring's class averages show, or None (as for an amyloid)."""
        try:
            at = _class_positions(result.class_ids, job["ids"])
            straight = maps.straighten_classes(
                [job["images"][i] for i in at],
                result.class_zdir,
                target_apix=job["apix"],
                apix=job["apix"],
            )
            return lattice.detect(straight, job["apix"])
        except Exception as e:  # a diagnostic: never in the way of the result
            logger.warning("subunit lattice of the class averages failed: %s", e)
            return None

    def _pixel_check(job, result, progress):
        """The ring against the registered class averages; None if it cannot be made."""
        try:
            progress.inc(1, message="comparing with the registered class averages")
            at = _class_positions(result.class_ids, job["ids"])
            return pixel.class_pixel_check(
                [job["images"][i] for i in at],
                result.class_zdir,
                result.phases,
                result.period,
                job["apix"],
            )
        except Exception as e:  # a diagnostic: never in the way of the result
            logger.warning("class-average check of the ring failed: %s", e)
            return None

    def _phase_apply(job, result):
        # a result belongs to the selection it was computed from
        if (
            analysis_params() is not job["params"]
            or _selected_class_numbers() != job["class_ids"]
        ):
            return
        phase_result.set(result)
        spans = result.filaments["span"]
        if len(spans):
            top = int(np.ceil(float(spans.max()) / 10.0) * 10)
            ui.update_slider("length_range", min=0, max=top, value=(0, top))
        fil = result.filaments
        pitches = fil["pitch"][fil["fitted"]]
        if len(pitches):
            lo_p = int(np.floor(float(pitches.min())))
            hi_p = int(np.ceil(float(pitches.max())))
            # centred on the pooled repeat (the green line), wide enough for
            # the 40% of the filaments closest to it
            half = float(np.percentile(np.abs(pitches - result.period), 40))
            low, high = result.period - half, result.period + half
            ui.update_slider(
                "pitch_band",
                min=lo_p,
                max=max(hi_p, lo_p + 1),
                value=(round(float(low)), round(float(high))),
            )

    def _phase_error(job, e):
        ui.modal_show(
            ui.modal(
                f"The class-azimuth pitch estimate failed: {e}",
                title="Pitch estimate error",
                easy_close=True,
                footer=None,
            )
        )

    phase_task = helicon.shiny.background_task(
        "phase_run",
        _phase_work,
        _phase_apply,
        _phase_error,
        # 4 more stages with the class-average maps
        progress_max=13,
        session=session,
        label="Class-azimuth pitch estimate",
    )

    def _max_repeat():
        """The longest repeat the pitch estimate searches, in A.

        One repeat turns the helix by 360 degrees at the C1 twist, the largest
        of the twists a repeat allows: 360 * rise / |smallest twist|.
        """
        rise = helicon.shiny.clamp_number(input.rise(), 4.75, 0.01, 1000.0, float)
        twist = helicon.shiny.clamp_number(
            abs(input.min_twist() or 0.3), 0.3, 0.05, 30.0, float
        )
        # the scan starts at 100 A
        return max(200.0, 360.0 * rise / twist)

    @reactive.effect
    @reactive.event(input.phase_run)
    def run_phase_pitch():
        req(params() is not None)
        class_ids = _selected_class_numbers()
        if len(class_ids) < MIN_PHASE_CLASSES:
            ui.modal_show(
                ui.modal(
                    "Select at least three classes of one helical type: the method "
                    "places the selected classes around a ring, and fewer than "
                    "three do not define one.",
                    title="Too few classes selected",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        if len(class_ids) > MAX_PHASE_CLASSES:
            ui.modal_show(
                ui.modal(
                    f"{len(class_ids)} classes are selected; the method takes at "
                    f"most {MAX_PHASE_CLASSES}, since its cost grows with the "
                    "square of the number of classes. Remove the junk classes and "
                    "those of other helical types from the computation.",
                    title="Too many classes selected",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        merge = bool(input.merge_counterparts())
        phase_task.invoke(
            dict(
                params=analysis_params(),
                class_ids=class_ids,
                # the UI's bounds, enforced here too: the browser's are not
                # binding
                n_boot=helicon.shiny.clamp_number(input.phase_n_boot(), 20, 2, 200),
                merge=merge,
                max_repeat=_max_repeat(),
                map_fit=(
                    dict(
                        rise=helicon.shiny.clamp_number(
                            input.rise(), 4.75, 0.01, 1000.0, float
                        )
                    )
                    if input.repeat_map_fit()
                    else None
                ),
                # also read for the check of the ring against the registered averages
                ids=list(displayed_class_ids()),
                images=list(displayed_class_images()),
                apix=float(data_all()[1]),
            )
        )

    # ── the classes in the computation, and suggestions for more ──

    @render.ui
    def accepted_gallery():
        numbers = _selected_class_numbers()
        r = phase_result()
        # the classes to take out come up picked, ready to remove, each
        # labelled with the reason
        poor = _poorly_fitting(r) if r is not None else {}
        flags = _pixel_flags(r) if r is not None else {}
        labels = [
            (
                f"{lab} \u2717 {_SHORT_REASON.get(poor[c], poor[c])}"
                if c in poor
                else lab
            )
            + (" \u25c6" if flags.get(c, (0.0, False))[1] else "")
            for lab, c in zip(selected_image_labels(), numbers)
        ]
        return _gallery(
            "accepted_pick",
            selected_images(),
            labels,
            label=f"{len(selected_images())} class(es) | "
            + _counts_text(numbers, in_use=True)
            + (f" | {len(poor)} picked to remove" if poor else ""),
            selection=True,
            initial=[i for i, c in enumerate(numbers) if c in poor],
        )

    @reactive.effect
    @reactive.event(input.accepted_remove)
    def remove_picked_classes():
        sel = list(input.select_classes_inner() or [])
        picked = {sel[i] for i in (input.accepted_pick() or []) if i < len(sel)}
        if not picked:
            ui.notification_show("Pick the classes to remove first.", duration=4)
            return
        _set_selection([i for i in sel if i not in picked])

    # at most this many unselected classes, the most populated, are compared by
    # shape: the cost is one registration per (selected, candidate) pair
    MAX_COUNTERPART_CANDIDATES = 60

    # suggestions: {"seeds": display positions they were computed from, "items":
    # [{"pos": display position, "text": label}], "n_candidates": int}
    suggestions = reactive.value(None)

    @reactive.effect
    @reactive.event(params, input.select_classes_inner)
    def _update_suggestions():
        r = suggestions()
        if r is None:
            return
        sel = set(input.select_classes_inner() or [])
        if not r["seeds"] <= sel:
            # a class the suggestions rest on was removed
            suggestions.set(None)
        elif any(d["pos"] in sel for d in r["items"]):
            suggestions.set(
                dict(r, items=[d for d in r["items"] if d["pos"] not in sel])
            )

    def _suggest_work(job, progress):
        pos_of = {c: i for i, c in enumerate(job["ids"])}
        found = {}  # display position -> label parts
        progress.set(0.05, message="finding classes used by the same filaments")
        for d in phase.suggest_expansion(
            job["params"],
            [c + 1 for c in job["selected"]],
            [c + 1 for c in job["others"]],
        ):
            c = d["candidate"] - 1
            found.setdefault(pos_of[c], {}).update(
                share=d["share"], baseline=d["baseline"], filaments=d["filaments"]
            )
        if job["data"] is not None:
            ab = job["abundance"]
            by_size = sorted(job["others"], key=lambda c: -ab[c] if c < len(ab) else 0)
            progress.set(0.1, message="matching class averages turned by 180\u00b0")
            for d in phase.suggest_counterparts(
                job["data"],
                job["selected"],
                by_size[:MAX_COUNTERPART_CANDIDATES],
                progress=lambda k, n: progress.set(0.1 + 0.9 * k / max(n, 1)),
            ):
                found.setdefault(pos_of[d["candidate"]], {}).update(
                    corr=d["corr"], of=d["selected"] + 1
                )
        r = job["phase_result"]
        if r is not None and found:
            # the fit each would get on the ring of the current estimate, on
            # the same scale as the ranked fits
            progress.set(0.95, message="placing the suggestions on the ring")
            fits, placed = phase.candidate_fit(
                job["params"],
                r,
                [job["ids"][pos] + 1 for pos in found],
                placement=True,
            )
            for pos, d in found.items():
                d["fit"] = fits.get(job["ids"][pos] + 1, float("nan"))
            _suggest_agreement(job, r, found, placed)
        return found

    def _suggest_agreement(job, r, found, placed):
        """Add the pixel agreement each suggestion would have to ``found``."""
        check = getattr(r, "pixel_check", None)
        if check is None or job.get("images") is None:
            return
        try:
            ring = _class_positions(r.class_ids, job["ids"])
            # the best-fitting candidates first, as many as are worth the time
            order = sorted(
                (pos for pos in found if job["ids"][pos] + 1 in placed),
                key=lambda pos: -np.nan_to_num(found[pos]["fit"], nan=-1.0),
            )[:MAX_AGREEMENT_CANDIDATES]
            if not order:
                return
            cls = [job["ids"][pos] + 1 for pos in order]
            agree = pixel.candidate_agreement(
                [job["images"][i] for i in ring],
                r.class_zdir,
                r.phases,
                r.period,
                check.sign,
                job["apix"],
                [job["images"][pos] for pos in order],
                [placed[c][1] for c in cls],
                [placed[c][0] for c in cls],
            )
            for pos, a in zip(order, agree):
                found[pos]["agree"] = float(a)
        except Exception as e:  # a check: the suggestions stand without it
            logger.warning("pixel agreement of the suggestions failed: %s", e)

    def _suggest_apply(job, found):
        # suggestions for a selection or data set no longer shown are dropped
        if analysis_params() is not job["params"] or list(
            input.select_classes_inner() or []
        ) != list(job["sel_disp"]):
            return
        labels = displayed_class_labels()
        items = []

        def best_fit_first(item):
            fit = item[1].get("fit", float("nan"))
            return (0, -fit) if np.isfinite(fit) else (1, item[0])

        for pos, d in sorted(found.items(), key=best_fit_first):
            why = []
            if np.isfinite(d.get("fit", float("nan"))):
                why.append(f"fit {d['fit']:.2f}")
            if np.isfinite(d.get("agree", float("nan"))):
                why.append(f"pixels {d['agree']:.2f}")
            if "corr" in d:
                why.append(f"\u21bb{d['corr']:.2f}")
            if "share" in d:
                why.append(f"\u25cf{100 * d['share']:.0f}%")
            items.append(
                dict(
                    pos=pos,
                    text=f"{labels[pos]}  {' '.join(why)}",
                    fit=d.get("fit", float("nan")),
                    agree=d.get("agree", float("nan")),
                )
            )
        suggestions.set(
            dict(
                seeds=set(job["sel_disp"]),
                items=items,
                n_candidates=len(job["others"]),
            )
        )

    def _suggest_error(job, e):
        ui.modal_show(
            ui.modal(
                f"The suggestions could not be computed: {e}",
                title="Suggestion error",
                easy_close=True,
                footer=None,
            )
        )

    suggest_task = helicon.shiny.background_task(
        "suggest_run",
        _suggest_work,
        _suggest_apply,
        _suggest_error,
        session=session,
        label="Class suggestions",
    )

    @reactive.effect
    @reactive.event(input.suggest_run)
    def run_suggestions():
        req(params() is not None)
        sel_disp = list(input.select_classes_inner() or [])
        if not sel_disp:
            ui.modal_show(
                ui.modal(
                    "Select the classes of one type first; other classes of that "
                    "type are then looked for among the rest.",
                    title="No classes selected",
                    easy_close=True,
                    footer=None,
                )
            )
            return
        ids = [int(c) for c in displayed_class_ids()]
        selected = [ids[i] for i in sel_disp]
        has_data = data_all() is not None and data_all()[0] is not None
        suggest_task.invoke(
            dict(
                params=analysis_params(),
                sel_disp=sel_disp,
                ids=ids,
                selected=selected,
                others=[c for c in ids if c not in set(selected)],
                data=data_all()[0] if has_data else None,
                abundance=abundance() if has_data else None,
                # the current estimate, to place the suggestions on its ring
                phase_result=phase_result(),
                images=list(displayed_class_images()),
                apix=float(data_all()[1]) if has_data else None,
            )
        )

    @render.ui
    def suggested_gallery():
        r = suggestions()
        if r is None:
            return None
        items, n_below = _shown_suggestions()
        below = (
            f" ({n_below} more would be picked to remove by a threshold)"
            if n_below
            else ""
        )
        if not items:
            return ui.tags.small(
                f"None of the {r['n_candidates']} other classes stands out" + below,
                class_="text-muted",
            )
        images = displayed_class_images()
        return _gallery(
            "suggested_pick",
            [images[d["pos"]] for d in items],
            [d["text"] for d in items],
            label=f"{len(items)} suggested | "
            + _counts_text(
                (int(displayed_class_ids()[d["pos"]]) + 1 for d in items), in_use=True
            )
            + below,
            selection=True,
        )

    @reactive.calc
    def _shown_suggestions():
        """The suggestions to offer, and how many are held back.

        A class whose predicted fit is below the Fit threshold, or whose
        predicted pixel agreement is below the Pixel agreement threshold, would
        be picked to remove as soon as it was added -- it would lower the ring's
        agreement rather than add to it -- so it is not offered; moving a
        threshold brings it back. One without a predicted fit (no pairs with
        the classes of the fit) is offered, last, and one without a pixel
        agreement is not judged by that threshold.
        """
        r = suggestions()
        if r is None:
            return [], 0
        threshold = input.fit_threshold()
        agreement = float(input.agree_threshold() or 0.0)
        if threshold is None and not agreement:
            return list(r["items"]), 0
        threshold = -np.inf if threshold is None else float(threshold)
        # a candidate without a predicted fit, or without a pixel agreement
        # (not checked, or no registered pair), is not judged by that one
        shown = [
            d
            for d in r["items"]
            if (not np.isfinite(d.get("fit", np.nan)) or d["fit"] >= threshold)
            and (not np.isfinite(d.get("agree", np.nan)) or d["agree"] >= agreement)
        ]
        return shown, len(r["items"]) - len(shown)

    @render.ui
    def suggested_buttons():
        if not _shown_suggestions()[0]:
            return None
        return ui.layout_columns(
            ui.input_action_button(
                "suggested_add_all",
                "Add all",
                class_="btn-sm btn-primary",
            ),
            ui.input_action_button(
                "suggested_add_picked",
                "Add picked",
                class_="btn-sm btn-primary",
            ),
            col_widths=(6, 6),
            fill=False,
        )

    def _add_suggested(indices):
        items = _shown_suggestions()[0]  # as the gallery shows them
        req(items)
        add = [items[i]["pos"] for i in indices if i < len(items)]
        if not add:
            ui.notification_show("Pick the suggested classes to add first.", duration=4)
            return
        _set_selection(list(input.select_classes_inner() or []) + add)

    @reactive.effect
    @reactive.event(input.suggested_add_all)
    def add_all_suggested():
        _add_suggested(range(len(_shown_suggestions()[0])))

    @reactive.effect
    @reactive.event(input.suggested_add_picked)
    def add_picked_suggested():
        _add_suggested(input.suggested_pick() or [])

    # ── types: the filaments grouped by the classes they use ──

    # {"params": the table it was computed for, "classes": class numbers,
    #  "similarity": (K, K) registered correlations, "counts": (filaments, K)
    #  segments, "keys": (micrograph, tube) of each filament, "splits": the
    #  marker splits best first, "default_n": groups up to the last clear split}
    types_state = reactive.value(None)
    # the groups' test for two repeats: {"state": the types_state it was made
    # for, "n": the number of groups, "cut": held-out gain needed, "tests":
    # {group: two_pitch_test result or None}}
    types_pitch = reactive.value(None)

    @reactive.effect
    @reactive.event(params)
    def _forget_types():
        types_state.set(None)
        types_pitch.set(None)
        type_group_used.set(None)
        ui.update_checkbox("types_restrict", value=False)

    def _types_work(job, progress):
        position = {int(c): i for i, c in enumerate(job["ids"])}
        counts = job["abundance"]
        floor = max(100, int(0.002 * int(np.sum(counts))))
        classes = [
            int(c) + 1
            for c in job["ids"]
            if counts[int(c)] >= floor
            and job["images"][position[int(c)]].max()
            > job["images"][position[int(c)]].min()
        ]
        if len(classes) < 3:
            raise ValueError(
                f"only {len(classes)} class(es) have {floor:,} segments or more; "
                "at least three are needed"
            )
        progress.set(0, message=f"registering the averages of {len(classes)} classes")
        similarity = types.class_similarity(
            [job["images"][position[c - 1]] for c in classes],
            list(range(len(classes))),
            job["apix"],
            progress=lambda k, n: progress.set(
                int(90 * k / max(n, 1)), message=f"registered {k:,} of {n:,} pairs"
            ),
        )
        full_similarity = similarity
        # a class that looks like no other (junk, or a view of its own) is no
        # evidence of a type: it is left out of the split and shown apart
        grouped = np.flatnonzero(types.pixel_groups(similarity) >= 0)
        if len(grouped) < 3:
            grouped = np.arange(len(classes))
        loners = [classes[i] for i in range(len(classes)) if i not in set(grouped)]
        modelled = list(classes)
        classes = [classes[i] for i in grouped]
        similarity = similarity[np.ix_(grouped, grouped)]
        if job.get("method") == "context":
            return _types_context_work(
                job, modelled, classes, loners, floor, progress, full_similarity
            )
        progress.set(90, message="splitting the filaments by the classes they use")
        usage, keys = types.usage_matrix(job["params"], classes)
        splits = types.marker_groups(usage, similarity)
        return dict(
            classes=classes,
            loners=loners,
            similarity=similarity,
            floor=floor,
            counts=usage,
            keys=[(str(m), int(t)) for m, t in keys],
            splits=splits,
            default_n=types.default_n_marker_groups(splits),
        )

    def _particle_diameter(p):
        """The box of the segments in A (the mask diameter RELION takes by default), or None."""
        try:
            optics = p.attrs["optics"]
            return float(
                optics["rlnImageSize"].iloc[0] * optics["rlnImagePixelSize"].iloc[0]
            )
        except (KeyError, IndexError, TypeError, AttributeError, ValueError):
            return None

    def _types_context_work(
        job, modelled, classes, loners, floor, progress, similarity=None
    ):
        """The filaments grouped by the sequence of classes along them, for 2 to 8
        types, in the shape of the marker splits (see helical_pitch_context).

        The model has every class with segments enough: a class that looks like no
        other (junk, or a view of its own) is evidence of a junk type, and without
        it the junk filaments fall into the others. What is listed is as in the
        marker method: the classes that look like others, the rest as Not grouped."""
        counts, keys = types.usage_matrix(job["params"], classes)
        # Filaments are picked in either direction, and the views of a helix follow
        # the molecule: a filament picked backwards uses the 180-degree counterparts
        # of the classes of a forward one, and its class sequence runs the other
        # way, which a sequence model reads as a type of its own (on the default
        # data, half of the two "types" were the two directions). The counterparts
        # are merged, and such a filament is read the other way round.
        counterparts = None
        if job.get("merge", True):
            progress.set(0, message="pairing the classes that are 180° counterparts")
            counterparts = _counterpart_pairs(modelled, job["ids"], job["images"])
        progress.set(
            0, message=f"fitting the filament context of {len(modelled)} classes"
        )
        splits, info = context.context_splits(
            job["params"],
            modelled,
            keys,
            counterparts=counterparts,
            detail=True,
            diameter=_particle_diameter(job["params"]),
            similarity=similarity,
            similarity_classes=modelled,
            compare_classes=classes,
            progress=lambda i, n, message: progress.set(
                int(100 * i / max(n, 1)), message=f"{message} ({i} of {n})"
            ),
        )
        return dict(
            classes=classes,
            loners=loners,
            floor=floor,
            counts=counts,
            keys=[(str(m), int(t)) for m, t in keys],
            splits=splits,
            default_n=info.get("default_n", context.CONTEXT_DEFAULT_TYPES),
            method="context",
            context=info,
        )

    def _types_apply(job, result):
        if params() is not job["params"]:
            return  # other data have been loaded meanwhile
        types_pitch.set(None)
        types_state.set(dict(result, params=job["params"]))

    def _types_error(job, e):
        ui.modal_show(
            ui.modal(
                f"The filaments could not be grouped: {e}",
                title="Grouping error",
                easy_close=True,
                footer=None,
            )
        )

    types_task = helicon.shiny.background_task(
        "types_run",
        _types_work,
        _types_apply,
        _types_error,
        progress_max=100,
        session=session,
        label="Filament types",
    )

    @reactive.effect
    @reactive.event(input.types_run)
    def run_types():
        req(params() is not None)
        req(data_all() is not None and data_all()[0] is not None)
        types_task.invoke(
            dict(
                params=params(),
                ids=[int(c) for c in displayed_class_ids()],
                images=list(displayed_class_images()),
                abundance=np.asarray(abundance()),
                apix=float(data_all()[1]),
                method=input.types_method(),
                merge=bool(input.merge_counterparts()),
            )
        )

    types_first = [True]

    @reactive.effect
    @reactive.event(types_state)
    def _default_types_n():
        # each grouping starts at the number of its clear splits; a bookmarked
        # number is kept for the first one (the card is static, so only a bookmark
        # can have moved the slider before there is a grouping)
        state = types_state()
        if state is None:
            return
        if types_first[0]:
            types_first[0] = False
            if input.types_n() not in (None, BOOKMARK_DEFAULTS["types_n"][1]):
                return
        ui.update_slider("types_n", value=int(state["default_n"]))

    def _types_number():
        return int(
            helicon.shiny.clamp_number(
                input.types_n(), 1, 1, types.MAX_MARKER_GROUPS, float
            )
        )

    @reactive.calc
    def _marker_groups():
        """The groups at the current number, before any split by repeat: a list of
        dicts, largest first; None before the filaments have been grouped."""
        state = types_state()
        if state is None:
            return None
        labels, described = types.describe_marker_groups(
            state["splits"], _types_number()
        )
        counts = np.asarray(state["counts"])
        if labels is None:
            labels = np.where(counts.sum(1) > 0, 0, -1)
        return _group_entries(state, labels, described)

    def _context_facts(state, g):
        """The period of the class sequence and the mean section of type ``g`` (the
        filament context method), or None."""
        if state.get("method") != "context":
            return None
        d = state["context"].get("detail", {}).get(_types_number())
        if not d or g >= len(d["types"]):
            return None
        row = d["types"].iloc[g]
        n = _types_number()
        like = state["context"].get("likeness", {}).get(n)
        # the other group whose class averages look most like this one's
        alike = []
        if like is not None and g < len(like) and len(like) > 1:
            h = max((h for h in range(len(like)) if h != g), key=lambda h: like[g, h])
            alike = [(h, float(like[g, h]))]
        period = float(row["PeriodAngst"])
        return dict(
            # at the longest period tried, the fit found none
            period=0.0 if period >= 0.99 * context.MAX_PERIOD_ANGST else period,
            score=float(row["PeriodScore"]),
            section=float(row["MeanSectionAngst"]),
            sections=int(row["NrSections"]),
            alike=alike,
        )

    def _group_entries(state, labels, described, mixed=None, fits=None, numbers=None):
        """One dict per group of ``labels``, largest first, numbered from 1; the
        group ``mixed`` (the groups whose classes fit no ring, merged) last.
        ``fits``: group -> ring fit, from the check of the groups; ``numbers``:
        group -> the number to show (by default, by size)."""
        fits = fits or {}
        classes = np.asarray(state["classes"])
        counts = np.asarray(state["counts"])
        keys = state["keys"]
        # the share of all segments, those of the classes in no group included
        total = max(float(np.sum(abundance())), 1.0)
        found = []
        for g in range(int(labels.max()) + 1 if labels.size else 0):
            rows = np.flatnonzero(labels == g)
            if not len(rows):
                continue
            seg = counts[rows].sum(0)
            n_seg = max(float(seg.sum()), 1.0)
            # the classes the group's filaments use: 1% of their segments or more
            used = [
                i for i in np.argsort(-seg, kind="stable") if seg[i] >= 0.01 * n_seg
            ]
            markers = set(int(m) for m in described[g]["markers"])
            if g == mixed:
                markers = set()
            found.append(
                dict(
                    id=g,
                    classes=[int(classes[i]) for i in used],
                    class_share=[float(seg[i] / n_seg) for i in used],
                    markers={int(classes[i]) for i in markers},
                    keys={keys[i] for i in rows},
                    filaments=len(rows),
                    segments=int(seg.sum()),
                    share=float(seg.sum() / total),
                    lacks=[] if g == mixed else list(described[g]["lacks"]),
                    repeat=None,
                    fit=fits.get(g),
                    mixed=g == mixed,
                    context=_context_facts(state, g),
                )
            )
        found.sort(key=lambda x: (x["mixed"], -x["filaments"]))
        number = numbers or {
            x["id"]: i + 1 for i, x in enumerate(found) if not x["mixed"]
        }
        # only the groups still there can be named (not those merged as Mixed)
        present = {x["id"] for x in found}
        number = {i: k for i, k in number.items() if i != mixed and i in present}
        found.sort(key=lambda x: (x["mixed"], number.get(x["id"], 0)))
        for x in found:
            x["label"] = "Mixed" if x["mixed"] else f"Group {number[x['id']]}"
            # a group merged into the mixed one is no longer there to name
            x["lacks"] = sorted(number[g] for g in x["lacks"] if g in number)
            if x.get("context"):
                x["context"]["alike"] = sorted(
                    (number[h], v) for h, v in x["context"]["alike"] if h in number
                )
        return found

    @reactive.calc
    def _type_groups():
        """The groups shown: those of :func:`_marker_groups`; once each group has
        been fitted with a ring, the groups whose classes fit none merged as Mixed
        and the groups with two repeats split in two."""
        groups = _marker_groups()
        if groups is None:
            return None
        tests = _pitch_tests()
        if not tests:
            return groups + _not_grouped()
        poor = sorted(
            i for i, t in tests.items() if not t.get("split") and t.get("poor")
        )
        state = types_state()
        labels, described = types.describe_marker_groups(
            state["splits"], _types_number()
        )
        if labels is None:
            labels = np.where(np.asarray(state["counts"]).sum(1) > 0, 0, -1)
        labels = labels.copy()
        if poor:
            # the groups whose classes fit no single ring: one group of mixtures
            labels[np.isin(labels, poor)] = poor[0]
        fits = {i: t.get("fit") for i, t in tests.items()}
        if poor:
            fits[poor[0]] = None
        # the numbers of the groups as first shown, which the note refers to
        numbers = {g["id"]: int(g["label"].split()[-1]) for g in groups}
        groups = _group_entries(
            state,
            labels,
            described,
            mixed=poor[0] if poor else None,
            fits=fits,
            numbers=numbers,
        )
        for g in groups:
            if g["mixed"]:
                g["merged"] = [tests[i]["fit"] for i in poor]
        out = []
        for g in groups:
            t = tests.get(g["id"])
            if t is None or g["mixed"] or not t.get("split"):
                out.append(g)
                continue
            # the filaments without a ring of their own go with the larger one
            larger = int(np.argmax(t["counts"]))
            ring = {k: t["rings"].get(k, larger) for k in g["keys"]}
            for r, letter in ((0, "a"), (1, "b")):
                keys = {k for k, v in ring.items() if v == r}
                out.append(
                    dict(
                        g,
                        label=f"{g['label']}{letter}",
                        keys=keys,
                        filaments=len(keys),
                        segments=None,
                        repeat=t["periods"][r],
                    )
                )
        return out + _not_grouped()

    def _not_grouped():
        """The classes that look like no other, as an entry of their own."""
        loners = list(types_state().get("loners", []))
        if not loners:
            return []
        counts = np.asarray(abundance(), dtype=float)
        n = np.array([counts[c - 1] for c in loners])
        order = np.argsort(-n, kind="stable")
        return [
            dict(
                id=None,
                label="Not grouped",
                classes=[loners[i] for i in order],
                class_share=[float(n[i] / max(n.sum(), 1.0)) for i in order],
                markers=set(),
                keys=set(),
                filaments=None,
                segments=int(n.sum()),
                share=float(n.sum() / max(float(counts.sum()), 1.0)),
                lacks=[],
                repeat=None,
            )
        ]

    def _group_text(g):
        seg = "" if g.get("segments") is None else f", {g['segments']:,} segments"
        share = "" if g.get("segments") is None else f" ({100 * g['share']:.0f}%)"
        if g["filaments"] is None:
            return (
                f"{g['label']}: {len(g['classes'])} classes that look like no "
                f"other{seg}{share}"
            )
        if g.get("mixed"):
            fits = g.get("merged") or []
            return (
                f"{g['label']}: {g['filaments']:,} filaments{seg}{share} from "
                f"{len(fits)} group(s) whose classes fit no single ring (ring fit "
                f"{min(fits):.2f} to {max(fits):.2f}): probably several types"
            )
        text = f"{g['label']}: {g['filaments']:,} filaments{seg}{share}"
        if g.get("repeat"):
            text += f", repeat ≈ {g['repeat']:.0f} Å"
        if g.get("fit") is not None and np.isfinite(g["fit"]):
            text += f", ring fit {g['fit']:.2f}"
        c = g.get("context")
        if c:
            if c["period"] > 0:
                text += f", class period {c['period']:.0f} Å"
            if c["section"] > 0:
                text += f", sections of {c['section']:.0f} Å on average"
            if c.get("alike"):
                k, v = c["alike"][0]
                text += f", class averages {v:.2f} like Group {k}'s"
        if g["markers"]:
            text += f", {len(g['markers'])} marker classes"
        if g["lacks"]:
            names = ", ".join(f"Group {x}" for x in g["lacks"])
            text += f" — may also hold filaments of {names} that lack its markers"
        return text

    @reactive.effect
    def _types_choices():
        groups = _type_groups()
        if groups is None:
            return
        ui.update_select(
            "types_group",
            choices={str(i): _group_text(g) for i, g in enumerate(groups)},
            selected="0",
        )

    def _cycle_note(state):
        """Whether the filaments are long enough to tell types from phases of one
        structure's cycle of views."""
        if state.get("method") != "context":
            return ""
        info = state["context"]
        text = ""
        cov = info.get("coverage") or {}
        if cov.get("n_classes") and cov["median_classes"] < 0.5 * cov["n_classes"]:
            text += (
                f". Filaments here are short (a median unbroken stretch of "
                f"{cov['median_span']:.0f} Å, showing {cov['median_classes']:.0f} of "
                f"{cov['n_classes']} classes), so types of one structure can differ "
                "only by which part of its cycle of views they show"
            )
        return text

    def _prediction_note(state):
        """How well the filament alone predicts the class of a segment (filament context,
        at the number of types it starts at; the number does not move it much)."""
        if state.get("method") != "context":
            return ""
        d = state["context"].get("detail", {}).get(state["default_n"])
        if not d or "context_hit" not in d:
            return ""
        return (
            f". The filament alone predicts the class of {100 * d['context_hit']:.0f}% "
            f"of the segments, from those at least {d['context_angst']:.0f} Å away"
        )

    @render.ui
    def types_controls():
        # drawn once for each grouping, not at each move of the number; the
        # groups are then followed by update_select
        state = types_state()
        if state is None:
            return None
        with reactive.isolate():
            groups = _type_groups()
        n_fil = len(state["keys"])
        loners = len(state.get("loners", []))
        if state.get("method") == "context":
            apart = (
                f"; {loners} that look like no other are modelled, and listed apart"
                if loners
                else ""
            )
        else:
            apart = f"; {loners} that look like no other are left out" if loners else ""
        by = (
            "by the sequence of their segments' classes along them, over the"
            if state.get("method") == "context"
            else "by their segments in the"
        )
        likeness = "" if state.get("method") == "context" else " that look like others"
        apart += _prediction_note(state) + _cycle_note(state)
        return ui.TagList(
            ui.p(
                f"{n_fil:,} filaments, {by} "
                f"{len(state['classes'])} classes with {state['floor']:,} segments "
                f"or more{likeness}{apart}",
                class_="small text-muted",
                style="margin-bottom: 4px;",
            ),
            ui.input_select(
                "types_group",
                None,
                {str(i): _group_text(g) for i, g in enumerate(groups)},
                width="100%",
            ),
        )

    def _chosen_group():
        groups = _type_groups()
        if groups is None:
            return None
        try:
            return groups[int(input.types_group() or 0)]
        except (ValueError, IndexError):
            return groups[0] if groups else None

    @render.ui
    def types_gallery():
        g = _chosen_group()
        if g is None:
            return None
        position = {int(c): i for i, c in enumerate(displayed_class_ids())}
        images = displayed_class_images()
        shown = [
            (c, s) for c, s in zip(g["classes"], g["class_share"]) if c - 1 in position
        ]
        return _gallery(
            "types_pick",
            [images[position[c - 1]] for c, _ in shown],
            [
                f"{c}: {100 * s:.0f}%" + (" ★" if c in g["markers"] else "")
                for c, s in shown
            ],
            label=(
                "Classes that look like no other (junk, or views of their own), "
                "with the share of their segments"
                if g["id"] is None
                else "The classes the group's filaments use, with the share of "
                "their segments (★: the marker classes that set the group apart)"
            ),
            selection=True,
        )

    def _typed_star():
        """The segments with the labels of the filament context at the number of types
        shown: ``rlnHelicalType``, ``rlnHelicalTypeProbability``,
        ``rlnHelicalSectionID`` and ``rlnHelicalContextProbability``; None if there
        are none."""
        state = types_state()
        if state is None or state.get("method") != "context":
            return None
        d = state["context"].get("detail", {}).get(_types_number())
        p = state.get("params")
        if not d or p is None or len(p) != len(d["labels"]):
            return None
        out = p.reset_index(drop=True).copy()
        out.attrs = dict(p.attrs)
        for c in d["labels"].columns:
            out[c] = d["labels"][c].to_numpy()
        return out

    @render.ui
    def types_buttons():
        if _chosen_group() is None:
            return None
        with reactive.isolate():
            typed = _typed_star() is not None
        export = None
        if typed:
            n = _types_number()
            btn = render.download(
                label=f"Download the {n} types of every segment (.star)",
                filename=f"filament_context_{n}_types_data.star",
            )

            @btn
            def download_types():
                out = _typed_star()
                req(out is not None)
                yield from _write_star(out)

            export = ui.tooltip(
                map_viewers.download_link(
                    session,
                    "download_types",
                    f"Download the {n} types of every segment (.star)",
                ),
                "The segments with the type of each (rlnHelicalType and its "
                "probability), the section along its filament it is in "
                "(rlnHelicalSectionID) and the probability of its class from the "
                "segments far along the filament (rlnHelicalContextProbability), "
                "as RELION's helix_filament_context writes them",
            )
        return ui.TagList(
            _types_button_row(),
            export,
        )

    def _types_button_row():
        return ui.layout_columns(
            _tip(
                ui.input_action_button(
                    "types_use",
                    "Use group",
                    class_="btn-sm btn-primary",
                ),
                "Select the classes of this group (picked ones if some are "
                "picked) and no others, and restrict the analysis to the "
                "group's filaments, ready for Estimate pitch",
            ),
            _tip(
                ui.input_action_button(
                    "types_add",
                    "Add group",
                    class_="btn-sm btn-primary",
                ),
                "Add all the classes of this group to those selected; when the "
                "analysis is restricted to a group, add this group's filaments "
                "to it",
            ),
            _tip(
                ui.input_action_button(
                    "types_add_picked",
                    "Add selected",
                    class_="btn-sm btn-primary",
                ),
                "Add the classes picked in this group's gallery to those "
                "selected; when the analysis is restricted to a group, add this "
                "group's filaments to it",
            ),
            col_widths=(4, 4, 4),
            fill=False,
        )

    def _type_positions(picked_only=None):
        """Sidebar positions of the chosen group's classes: the picked ones if
        some are picked (``picked_only=None``), only those (True) or all (False)."""
        g = _chosen_group()
        req(g is not None)
        position = {int(c): i for i, c in enumerate(displayed_class_ids())}
        classes = [c for c in g["classes"] if c - 1 in position]
        picked = input.types_pick() or []
        if picked_only or (picked_only is None and picked):
            classes = [classes[i] for i in picked if i < len(classes)]
        return [position[c - 1] for c in classes]

    @reactive.effect
    @reactive.event(input.types_use)
    def use_type_group():
        positions = _type_positions()
        req(positions)
        g = _chosen_group()
        # the classes in no group have no filaments of their own
        if g["id"] is None:
            ui.update_checkbox("types_restrict", value=False)
        else:
            type_group_used.set(dict(label=g["label"], keys=set(g["keys"])))
            ui.update_checkbox("types_restrict", value=True)
        _set_selection(positions)

    @reactive.effect
    @reactive.event(input.types_add)
    def add_type_group():
        _add_type_classes(_type_positions(picked_only=False))

    @reactive.effect
    @reactive.event(input.types_add_picked)
    def add_picked_type_classes():
        positions = _type_positions(picked_only=True)
        if not positions:
            ui.notification_show(
                "Pick classes in the group's gallery first.", duration=4
            )
            return
        _add_type_classes(positions)

    def _add_type_classes(positions):
        """Add these classes of the chosen group to the selection, and the
        group's filaments to the restriction if there is one."""
        req(positions)
        g = _chosen_group()
        current = type_filter()
        if current is not None and g["id"] is not None:
            type_group_used.set(
                dict(
                    label=f"{current['label']} + {g['label']}",
                    keys=set(current["keys"]) | set(g["keys"]),
                )
            )
        _set_selection(list(input.select_classes_inner() or []) + positions)

    @reactive.effect
    @reactive.event(input.types_restrict)
    def _restrict_without_group():
        # ticked by the user with no group to restrict to
        if input.types_restrict() and type_group_used() is None:
            ui.update_checkbox("types_restrict", value=False)
            ui.notification_show(
                "No group is in use: choose one in the Types card and press Use "
                "group.",
                duration=6,
            )

    @reactive.effect
    def _restrict_to_group():
        """The restriction follows the checkbox, to the group last used."""
        group = type_group_used()
        on = bool(input.types_restrict())
        with reactive.isolate():
            before = type_filter()
            had_result = phase_result() is not None
        after = group if on and group is not None else None
        if after is before:
            return
        type_filter.set(after)
        if before is not None and after is None and group is not None:
            # unticked: say what changed
            ui.notification_show(
                f"All filaments are used again, not only those of {before['label']}."
                + (
                    f" The pitch estimate of {before['label']} was cleared: run "
                    "Estimate pitch for the selected classes on all filaments."
                    if had_result
                    else ""
                ),
                duration=8,
            )

    @render.ui
    def types_restrict_note():
        group = type_group_used()
        if group is None:
            text = "No group in use yet (Types card: Use group)."
        elif type_filter() is None:
            text = f"Off: all filaments. On: the {len(group['keys']):,} of {group['label']}."
        else:
            text = f"On: the {len(group['keys']):,} filaments of {group['label']}."
        return ui.p(text, class_="small text-muted", style="margin: -8px 0 8px 0;")

    @render.ui
    def type_filter_status():
        chosen = type_filter()
        if chosen is None:
            return None
        return ui.p(
            f"Only the {len(chosen['keys']):,} filaments of {chosen['label']} are "
            "used (Parameters: untick Only the filaments of the type group in "
            "use, for all).",
            class_="small text-muted",
            style="margin: 2px 0;",
        )

    # ── types: does a group hold two repeats ──

    # a sample of 400 filaments gave the same ring fit and held-out gain as 800
    # (and as the whole group) in half the time; two starts of each two-ring fit
    # instead of four kept the same splits at half the time again
    MAX_TEST_FILAMENTS = 400
    TEST_RESTARTS = 2

    def _types_pitch_work(job, progress):
        groups = [
            g for g in job["groups"] if g["filaments"] >= 100 and len(g["classes"]) >= 3
        ]
        if not groups:
            raise ValueError(
                "no group has 100 filaments and three classes, the least a ring "
                "fit needs"
            )
        p = job["params"]
        keys = [
            (m, int(t))
            for m, t in zip(p["rlnMicrographName"].astype(str), p["rlnHelicalTubeID"])
        ]

        def rows_of(g):
            out = p.loc[np.array([k in g["keys"] for k in keys], dtype=bool)]
            out.attrs = dict(p.attrs)
            return out

        jobs = [(rows_of(g), g["classes"][:MAX_PHASE_CLASSES]) for g in groups]
        progress.set(0, message=f"one ring and two, for {len(groups)} groups")
        found = mixture.two_pitch_tests(
            jobs,
            max_filaments=MAX_TEST_FILAMENTS,
            restarts=TEST_RESTARTS,
            progress=lambda k, n: progress.set(
                int(90 * k / max(n, 1)), message=f"{k} of {n} fits"
            ),
        )
        tests = {}
        for g, t in zip(groups, found):
            if isinstance(t, Exception):
                logger.warning("ring fits of %s failed: %s", g["label"], t)
            elif t is not None:
                tests[g["id"]] = t
        gains = [t["heldout"] for t in tests.values() if np.isfinite(t["heldout"])]
        cut = mixture.split_cut(gains)
        again = []
        for g in groups:
            t = tests.get(g["id"])
            if t is None:
                continue
            if t.get("periods") is None:
                t["split"] = False
            else:
                lo, hi = t["periods"]
                t["split"] = t["heldout"] > cut and hi / lo - 1 >= mixture.MIN_APART
            # classes that fit no single ring: several types (or junk) together
            t["poor"] = not t["split"] and t["fit"] < mixture.MIN_RING_FIT
            if t["split"] and g["filaments"] > MAX_TEST_FILAMENTS:
                again.append((g, rows_of(g)))
        if again:
            # the rings of every filament of a group split in two, not a sample's
            progress.set(90, message="the two repeats of every filament")
            full = mixture.two_pitch_tests(
                [(rows, g["classes"][:MAX_PHASE_CLASSES]) for g, rows in again],
                restarts=TEST_RESTARTS,
            )
            for (g, _), t in zip(again, full):
                if isinstance(t, dict) and t.get("periods") is not None:
                    tests[g["id"]] = dict(
                        t, split=True, poor=False, fit=tests[g["id"]]["fit"]
                    )
        return dict(tests=tests, cut=cut)

    def _types_pitch_apply(job, result):
        if types_state() is not job["state"] or _types_number() != job["n"]:
            return  # regrouped meanwhile
        types_pitch.set(dict(result, state=job["state"], n=job["n"]))

    def _types_pitch_error(job, e):
        ui.modal_show(
            ui.modal(
                f"The groups could not be evaluated: {e}",
                title="Ring fit error",
                easy_close=True,
                footer=None,
            )
        )

    types_pitch_task = helicon.shiny.background_task(
        "types_pitch_run",
        _types_pitch_work,
        _types_pitch_apply,
        _types_pitch_error,
        progress_max=100,
        session=session,
        label="One vs two rings",
    )

    def _pitch_tests():
        """The groups evaluated so far, for the groups shown now: {id: test}."""
        test = types_pitch()
        if (
            test is None
            or test["state"] is not types_state()
            or test["n"] != _types_number()
        ):
            return {}
        return test["tests"]

    @reactive.effect
    @reactive.event(input.types_pitch_run)
    def run_types_pitch():
        groups = _marker_groups()
        req(groups)
        types_pitch_task.invoke(
            dict(
                params=params(),
                state=types_state(),
                n=_types_number(),
                groups=groups,
            )
        )

    @render.ui
    def types_pitch_ui():
        if types_state() is None:
            return None
        tests = _pitch_tests()
        note = None
        if tests:
            label = {g["id"]: g["label"] for g in _marker_groups()}
            parts = []
            for i, t in tests.items():
                if i not in label:
                    continue
                if t.get("split"):
                    what = (
                        f"two repeats, {t['periods'][0]:.0f} and "
                        f"{t['periods'][1]:.0f} Å: split in two"
                    )
                elif t.get("poor"):
                    what = "no single ring: merged as Mixed"
                else:
                    what = "one ring"
                gain = (
                    f", second ring {t['heldout']:+.2f}"
                    if np.isfinite(t.get("heldout", np.nan))
                    else ""
                )
                parts.append(f"{label[i]} (fit {t['fit']:.2f}{gain}): {what}")
            cut = types_pitch()["cut"]
            untested = [g["label"] for g in _marker_groups() if g["id"] not in tests]
            note = ui.p(
                "; ".join(parts)
                + f". A split needs a held-out gain above {cut:.2f}"
                + (
                    " (the default: fewer than four groups to compare)"
                    if len(tests) < 4
                    else " (from the groups' gains)"
                )
                + f"; a fit below {mixture.MIN_RING_FIT:.1f} means several types."
                + (
                    " Not evaluated (fewer than 100 filaments or three classes): "
                    + ", ".join(untested)
                    + "."
                    if untested
                    else ""
                ),
                class_="small text-muted",
                style="margin: 4px 0 0 0;",
            )
        return ui.TagList(
            _tip(
                ui.input_task_button(
                    "types_pitch_run",
                    label="Evaluate one vs two ring fits for each group",
                    class_="btn-sm btn-primary",
                ),
                "For each group, fits one ring of its classes and two. A group "
                "whose classes fit no single ring (ring fit below 0.3) holds "
                "several types or junk: such groups are merged as Mixed. Types "
                "that share their classes can differ in repeat: a group is split "
                "in two where the second ring clearly helps on filaments it was "
                "not fitted to (far more than in the other groups) and the two "
                "repeats differ by 8% or more. A good fit does not prove one type: "
                "two types of nearly the same repeat (a doublet and a triplet, "
                "say) fit one ring well. The groups are fitted in parallel, on "
                "the processors available: about a minute.",
            ),
            note,
        )

    def _poorly_fitting(r):
        """Selected classes to take out, with the reason for each."""
        threshold = input.fit_threshold()
        agreement = input.agree_threshold()
        return phase.diagnose_classes(
            r,
            threshold=None if threshold is None else float(threshold),
            agreement_threshold=None if not agreement else float(agreement),
        )

    def _pixel_flags(r):
        """Class number -> (offset residual in A, disagrees); {} without a check."""
        check = getattr(r, "pixel_check", None)
        if check is None:
            return {}
        gone = check.disagree()
        return {
            int(c): (float(v), bool(g))
            for c, v, g in zip(r.class_ids, check.residual, gone)
            if np.isfinite(v)
        }

    first_result = [True]

    @reactive.effect
    @reactive.event(phase_result)
    def _suggest_fit_threshold():
        # each new result starts from where its own ranked fits break; a
        # bookmarked threshold is kept for the first result (the slider is
        # hidden until there is one, so only a bookmark can have moved it)
        r = phase_result()
        if r is None:
            return
        if first_result[0]:
            first_result[0] = False
            default = BOOKMARK_DEFAULTS["fit_cut"][1]
            if input.fit_threshold() not in (None, default):
                return
        value = phase.fit_threshold(r.class_fit)
        if value is not None:
            ui.update_slider("fit_threshold", value=round(value, 2))

    MIN_PHASE_CLASSES = 3

    # whether a ring has been drawn for the data loaded now
    ring_drawn = reactive.value(False)

    @reactive.effect
    @reactive.event(params)
    def _forget_ring():
        ring_drawn.set(False)

    @reactive.effect
    def _remember_ring():
        if phase_result() is not None:
            ring_drawn.set(True)

    @render.ui
    def results_visibility():
        # Both start hidden in the page itself (_START_HIDDEN_CSS): hiding them
        # from here instead left them on screen while the tab started, before
        # this had run. This only shows them, once there is something to show.
        shown = []
        if phase_result() is not None:
            shown.append(f"#{_RESULTS_BOX} {{ display: block; }}")
        if ring_drawn():
            # a card's id is namespaced like an input's
            shown.append(f"#{session.ns(_SUGGESTIONS_BOX)} {{ display: flex; }}")
        if not shown:
            return None
        return ui.tags.style("\n".join(shown))

    @reactive.effect
    def _enable_phase_run():
        n = len(input.select_classes_inner() or [])
        if phase_task.status() == "running":
            return  # the task button shows it busy
        message = {"id": session.ns("phase_run"), "disabled": n < MIN_PHASE_CLASSES}

        # sent after the flush: the task button's own "ready" update, which
        # re-enables the button when a run ends, goes out with the flush
        async def _send():
            await session.send_custom_message("helicon-set-disabled", message)

        session.on_flushed(_send, once=True)

    @render.ui
    def phase_summary():
        r = phase_result()
        if r is None:
            n = len(input.select_classes_inner() or [])
            need = (
                ""
                if n >= MIN_PHASE_CLASSES
                else f"; select at least {MIN_PHASE_CLASSES} to estimate the pitch"
            )
            return _tip(
                ui.tags.small(f"{n} class(es) selected{need}", class_="text-muted"),
                "Select at least three classes of one helical type (shift-click to "
                "add) or add suggested classes, then press Estimate pitch. The result "
                "is cleared whenever the selection changes.",
            )
        rise = input.rise()
        het = r.heterogeneity
        n_sel = len(r.class_ids) + len(r.merged_classes)
        merged = (
            f" ({len(r.merged_classes)} merged -> {len(r.class_ids)} unique classes)"
            if r.merged_classes
            else ""
        )
        lines = [
            ui.p(
                ui.tags.b(f"Repeat {r.period:.1f} \u00b1 {r.period_sd:.1f} \u00c5"),
                f" \u00b7 {n_sel} classes{merged} \u00b7 {r.n_filaments:,} filaments",
                _info(
                    f"Raw peak {r.period_raw:.1f} \u00c5, corrected by "
                    f"{-100 * r.calibration['bias']:+.2f}% for this dataset's segment "
                    f"geometry; {r.n_pairs:,} segment pairs; positions "
                    + (
                        "along the refined filament path."
                        if r.refined_path
                        else "from the track length (no coordinates/origins)."
                    )
                    + (
                        f" {len(r.merged_classes)} classes were merged into their "
                        "180\u00b0-rotated counterparts."
                        if r.merged_classes
                        else ""
                    )
                ),
                style="margin-bottom: 2px;",
            )
        ]
        mf = r.map_fit
        if mf is not None:
            fits = ", ".join(f"C{c} {v:.3f}" for c, v in mf["fit_by_csym"].items())
            lines.append(
                ui.p(
                    (
                        "Repeat refined from the segments; the class-average maps "
                        f"put it near {mf['basin']:.0f} Å"
                        if mf["source"] == "metadata"
                        else "Repeat from the class-average maps (the segment pairs "
                        "do not reach half of it)"
                    ),
                    f" · the averages fit C{mf['csym']} best",
                    _info(
                        "Maps from two thirds of the averages at each candidate "
                        "repeat, scored by how well the others fit them, combined "
                        "with the segments' score (Parameters). Best mean fit of "
                        f"the held-out averages: {fits}; {mf['workers']} worker "
                        "processes."
                    ),
                    style="margin-bottom: 2px;",
                )
            )
            if int(input.rot_fold() or 1) != mf["csym"]:
                lines.append(
                    ui.p(
                        f"The class averages fit C{mf['csym']} better than the "
                        f"C{int(input.rot_fold() or 1)} set in Parameters.",
                        class_="text-warning",
                        style="margin-bottom: 2px;",
                    )
                )
        lat = getattr(r, "lattice", None)
        if lat is not None and mf is not None:
            lines.append(
                ui.p(
                    "The class-average maps assume one turn per repeat, which this "
                    "lattice is not: read the repeat from the segments.",
                    class_="text-warning",
                    style="margin-bottom: 2px;",
                )
            )
        if lat is not None:
            peaks = ", ".join(f"{d:.1f}" for d, _, _ in lat.peaks)
            lines.append(
                ui.p(
                    f"Lattice: rise {lat.rise:.2f} \u00c5 (from the class averages)",
                    _info(
                        "The class averages match themselves shifted along the axis "
                        f"by whole subunits ({peaks} \u00c5), best by {lat.k_s}: so "
                        f"C \u00d7 twist is close to 360\u00b0 \u00d7 {lat.m}/{lat.k_s}, "
                        "and the helix turns many times 360\u00b0/C over one repeat. "
                        "The twists below take that into account; which C it is, "
                        "the class averages cannot tell, and it is set in Parameters."
                    ),
                    style="margin-bottom: 2px;",
                )
            )
            if rise and abs(rise - lat.rise) > 0.03 * lat.rise:
                lines.append(
                    ui.p(
                        f"The rise in Parameters ({rise} \u00c5) is not the one the "
                        "class averages show. ",
                        ui.input_action_link(
                            "use_lattice_rise", f"Use {lat.rise:.2f} \u00c5"
                        ),
                        class_="text-warning",
                        style="margin-bottom: 2px;",
                    )
                )
        if rise is not None and rise > 0:
            j, n = lattice.periods_per_repeat(r.period, rise, lat)
            choices = (1, 2, 3, 4, 5, 6, 8, 12) if lat is not None else (1, 2, 3, 4)
            chosen = int(input.rot_fold() or 1)
            parts = []
            for c in choices:
                text = f"C{c} {lattice.twist(r.period, rise, c, lat):.3f}\u00b0"
                parts.append(ui.tags.b(text) if c == chosen else text)
                parts.append(" \u00b7 ")
            lines.append(
                ui.p(
                    "Twist: ",
                    *parts[:-1],
                    _info(
                        (
                            f"{n:.0f} subunits and {j} turns of 360\u00b0/C per "
                            f"repeat, at rise {rise} \u00c5: twist = {j} \u00d7 "
                            f"360\u00b0 / (C \u00d7 {n:.0f}). The hand cannot be "
                            "seen in projections; the other hand has "
                            f"{int(n) - j} turns."
                        )
                        if j > 1
                        else (
                            f"If the repeat is the pitch divided by C, at rise {rise} "
                            f"\u00c5 ({r.period / rise:.1f} subunits per repeat)"
                        )
                    ),
                    style="margin-bottom: 2px;",
                )
            )
        if het["n"] >= 10:
            lo, hi = het["sd_between_ci"]
            lines.append(
                ui.p(
                    f"Filament spread {het['sd_between']:.0f} \u00c5 "
                    f"(95% CI {lo:.0f}\u2013{hi:.0f})",
                    _info(
                        f"Split-half test on {het['n']} filaments \u2265 1.2 repeats "
                        f"long; correlation between halves {het['corr']:.2f}"
                    ),
                    style="margin-bottom: 2px;",
                )
            )
        warnings = []
        if r.double_ratio >= phase.DOUBLE_RATIO_WARN:
            warnings.append(
                (
                    f"2\u00d7 ({2 * r.period:.0f} \u00c5) fits almost as well",
                    "The selected classes may cover only half of the views, which "
                    "places them around a ring half the true size. Add the missing "
                    "views -- often the 180\u00b0-rotated versions of the selected "
                    "classes -- or read the repeat as twice this.",
                )
            )
        if r.support < phase.MIN_SUPPORT:
            warnings.append(
                (
                    "Too few pairs beyond one repeat",
                    "Almost no segment pairs are further apart than this distance, so "
                    "it cannot be told apart from half the true value: the pitch may "
                    "be twice this.",
                )
            )
        if r.phase_gap > 120:
            warnings.append(
                (
                    f"{r.phase_gap:.0f}\u00b0 of the ring empty",
                    "Some views of this type are not selected, and the estimate rests "
                    "on the ones that are.",
                )
            )
        poor = _poorly_fitting(r)
        if poor:
            by_reason = {}
            for c, why in poor.items():
                by_reason.setdefault(why, []).append(c)
            warnings.append(
                (
                    f"{len(poor)} class(es) to remove: "
                    + "; ".join(
                        f"{', '.join(map(str, cs))} ({why})"
                        for why, cs in by_reason.items()
                    ),
                    "Judged from the segments alone, not the look of the averages "
                    "(numbers as before the colon in the labels). 'No better than "
                    "chance': set against segments a quarter repeat or more away, "
                    "the class's segments do not sit at one azimuth on the ring "
                    "(dim on the ring plot) -- junk, or another type. 'Too few "
                    "informative segment pairs': fewer than 5% of a typical "
                    "class's pairs a quarter repeat or more apart, too little to "
                    "tell either way. 'Below the threshold': a fit under the Fit "
                    "threshold set below the ranked fits. A 180\u00b0 copy goes "
                    "with the class it was merged into. They are picked in the "
                    "Selected classes gallery; press Remove picked to drop them.",
                )
            )
        apolar = phase.apolar_warning(r.params_used)
        if apolar:
            warnings.append(apolar)
        if r.vetoed_period is not None:
            warnings.append(
                (
                    f"{r.vetoed_period:.0f} \u00c5 ruled out as the repeat",
                    "The segments alone fit a repeat of "
                    f"{r.vetoed_period:.0f} \u00c5 best, but the class averages "
                    "do not repeat at it (an average and itself shifted by that "
                    "distance do not match), so it is not the repeat. This "
                    "happens when views are missing from the selection: the "
                    "classes that are left can be placed on a ring of half the "
                    "size. The estimate is the best repeat the averages allow.",
                )
            )
        flags = _pixel_flags(r)
        if flags:
            ids = [int(c) for c in r.class_ids]
            differ = [c for c in ids if flags.get(c, (0.0, False))[1]]
            if differ:
                warnings.append(
                    (
                        f"{len(differ)} class(es) \u25c6 differ from their registered "
                        "averages: " + ", ".join(map(str, differ)),
                        "Independent of the segments' labels and positions: the "
                        "class averages, registered against each other along the "
                        "filament, put these classes more than "
                        f"{100 * pixel.DISAGREE_FRACTION:.0f}% of a repeat from "
                        "where their segments put them. A mixture of views, "
                        "another helical type, or segments assigned to the wrong "
                        "class. Marked \u25c6 in the gallery and on the ranked "
                        "fits; nothing is removed because of it.",
                    )
                )
            # only a class the segments judge by their positions (not one with too
            # few of them to judge, nor a turned copy that went with another)
            kept = [
                c
                for c, why in poor.items()
                if why in (phase.AT_CHANCE, phase.BELOW_THRESHOLD)
                and flags.get(int(c), (0.0, True))[1] is False
            ]
            if kept:
                warnings.append(
                    (
                        f"{len(kept)} flagged class(es) look right in the pixels: "
                        + ", ".join(map(str, kept)),
                        "Their segments do not fit the ring, but their averages sit "
                        "where the ring puts them, so the class may be sound with "
                        "its segments poorly assigned. Look at them before taking "
                        "them out.",
                    )
                )
        if len(r.groups) > 1:
            parts = "; ".join(
                f"{len(g)} classes \u2192 {p:.1f} \u00c5"
                for g, p in zip(r.groups, r.group_periods)
            )
            warnings.append(
                (
                    f"{len(r.groups)} groups of classes",
                    "The selected classes fall into groups that rarely share "
                    f"filaments -- probably different helical types: {parts}. The "
                    "repeat above mixes them.",
                )
            )
        if het["n"] < 10:
            warnings.append(
                (
                    "Too few long filaments for the spread",
                    f"Only {het['n']} filaments are at least 1.2 repeats long.",
                )
            )
        for short, long in warnings:
            lines.append(
                ui.p(
                    "\u26a0\ufe0f " + short,
                    _info(long),
                    style="margin-bottom: 2px; color: #b35c00;",
                )
            )
        return ui.div(*lines)

    def _add_marker(fig, x, y0, y1):
        """A vertical line drawn as a trace, not a shape: the page script that
        gives the HelicalPitch histograms their crosshairs takes over the shapes of
        every plot on the page, and would move a shape with the mouse."""
        import plotly.graph_objects as go

        fig.add_trace(
            go.Scatter(
                x=[x, x],
                y=[y0, y1],
                mode="lines",
                line=dict(color="green", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    @render.ui
    def phase_scan_plot():
        r = phase_result()
        req(r is not None)
        import plotly.graph_objects as go

        rise = input.rise()
        periods = np.asarray(r.scan_periods, dtype=float)
        hover = [
            f"Repeat: {p:.0f} \u00c5<br>Score: {v:.3f}"
            for p, v in zip(periods, r.scan_scores)
        ]
        if rise is not None and rise > 0:
            # the same read-out as the HelicalPitch histogram's hover text
            hover = [
                h
                + "".join(
                    f"<br>Twist for C{c}: {360.0 / (p * c / rise):.2f}\u00b0"
                    for c in (1, 2, 3, 4)
                )
                for h, p in zip(hover, periods)
            ]
        fig = go.Figure(
            go.Scatter(
                x=periods,
                y=r.scan_scores,
                mode="lines",
                name="score",
                text=hover,
                hoverinfo="text",
                showlegend=False,
            )
        )
        top = float(np.max(r.scan_scores))
        ungated = getattr(r, "scan_scores_ungated", None)
        if ungated is not None:
            # the gate by the class averages' self-similarity zeroes the repeats
            # they rule out; the scan without it stays in view, so that the
            # short repeats the gate removed are not missing from the plot
            fig.data[0].name = "with the class-average check"
            fig.add_trace(
                go.Scatter(
                    x=periods,
                    y=ungated,
                    mode="lines",
                    name="segments only",
                    line=dict(color="#aaaaaa", width=1.5, dash="dot"),
                    hoverinfo="skip",
                )
            )
            fig.update_traces(showlegend=True, selector=dict(mode="lines"))
            top = max(top, float(np.max(ungated)))
        _add_marker(fig, r.period_raw, 0.0, top)
        fig.update_layout(
            legend=dict(x=1.0, xanchor="right", y=1.0, font=dict(size=10)),
            template="plotly_white",
            title_text="Repeat scan",
            title_x=0.5,
            title_font=dict(size=12),
            xaxis_title="Repeat distance (\u00c5)",
            yaxis_title="Top eigenvalue",
            hovermode="closest",
            hoverlabel=dict(bgcolor="white", font_size=12),
            margin=dict(t=40, b=50, l=50, r=20),
        )
        return _fig_to_html(fig)

    @render.ui
    def dihedral_plot():
        r = phase_result()
        req(r is not None)
        if not input.dihedral():
            return None
        import plotly.graph_objects as go
        from ..lib import helical_pitch_dihedral as dihedral

        apix = float(data_all()[1])
        images = displayed_class_images()
        at = _class_positions(r.class_ids, displayed_class_ids())
        csym = helicon.shiny.clamp_number(input.rot_fold(), 1, 1, 12)
        prepared = maps.straighten_classes(
            [images[i] for i in at],
            r.class_zdir,
            target_apix=max(5.0, apix),
            apix=apix,
        )
        # the ring turns 360 degrees per repeat, the azimuth 360 / csym
        res = dihedral.dihedral_scan(
            prepared, np.rad2deg(r.phases) / csym, period=360.0 / csym
        )
        fig = go.Figure(
            go.Scatter(x=res["alpha"], y=res["score"], mode="lines", showlegend=False)
        )
        if res["best_alpha"] is None:
            title = "Dihedral test: too few classes to pair"
        else:
            title = (
                f"Dihedral test: best {res['best_score']:.2f} at "
                f"{res['best_alpha']:.1f}\u00b0, peak above the median "
                f"{res['contrast']:.2f}"
            )
        fig.update_layout(
            template="plotly_white",
            title_text=title,
            title_x=0.5,
            title_font=dict(size=12),
            xaxis_title="Two-fold axis azimuth (\u00b0)",
            yaxis_title="Mirror correlation",
            yaxis_range=[0, 1],
            margin=dict(t=40, b=50, l=50, r=20),
        )
        return ui.div(_fig_to_html(fig), style=_PLOT_BOX, class_="ab-plot-box ab-wait")

    @render.ui
    def phase_circle_plot():
        r = phase_result()
        req(r is not None)
        import plotly.graph_objects as go

        w = np.sqrt(np.maximum(r.class_weight, 0))
        size = 6 + 24 * w / max(w.max(), 1e-12)
        fit = np.nan_to_num(np.asarray(r.class_fit, dtype=float), nan=0.0)
        fig = go.Figure(
            go.Scatterpolar(
                theta=np.rad2deg(r.phases) % 360.0,
                r=np.ones(len(r.phases)),
                mode="markers+text" if input.ring_labels() else "markers",
                text=[str(int(c)) for c in r.class_ids],
                textposition="top center",
                customdata=fit,
                marker=dict(
                    size=size,
                    color=fit,
                    colorscale="Viridis",
                    # the full 0-1 range, so the colours mean the same on
                    # every data set
                    cmin=0.0,
                    cmax=1.0,
                    colorbar=dict(title="fit", thickness=10, len=0.6),
                ),
                hovertemplate=(
                    "class %{text}<br>azimuth %{theta:.0f}\u00b0"
                    "<br>fit %{customdata:.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            template="plotly_white",
            title_text="Class azimuths (one repeat)",
            title_x=0.5,
            title_font=dict(size=12),
            polar=dict(radialaxis=dict(visible=False, range=[0, 1.3])),
            showlegend=False,
            margin=dict(t=40, b=20, l=20, r=20),
        )
        return _fig_to_html(fig)

    def _rank_figure(r):
        """The class fits of the ring, ranked: how they are spread.

        The ring shows where each class sits; this shows the fits themselves,
        best first, on the same fixed scale and colours, each marker's area
        following the class's number of segments, the classes to take out
        outlined in red (a class without a fit as a cross at 0) and the line
        below which a fit is taken as chance.
        """
        import plotly.graph_objects as go

        ids = [int(c) for c in r.class_ids]
        fit = np.asarray(r.class_fit, dtype=float)
        members = np.asarray(
            r.class_count if r.class_count is not None else r.class_weight,
            dtype=float,
        )
        order = np.argsort(-np.nan_to_num(fit, nan=-1.0), kind="stable")
        f = np.nan_to_num(fit[order], nan=0.0)
        m = members[order]
        # area by the number of segments
        size = 6 + 24 * np.sqrt(m) / max(float(np.sqrt(m).max()), 1e-12)
        poor = _poorly_fitting(r)
        flags = _pixel_flags(r)
        reasons = [poor.get(ids[i], "") for i in order]
        text = [str(ids[i]) for i in order]
        no_fit = ~np.isfinite(fit[order])
        disagree = [flags.get(ids[i], (0.0, False))[1] for i in order]

        def _pixel_line(c):
            if int(c) not in flags:
                return ""
            res, bad = flags[int(c)]
            return f"<br>registered averages {res:.0f} \u00c5 from the ring" + (
                " \u25c6 disagree" if bad else ""
            )

        hover = [
            f"class {c}<br>"
            + ("no fit" if missing else f"fit {v:.2f}")
            + f"<br>{int(n):,} segments"
            + (f"<br>\u2717 {_SHORT_REASON.get(why, why)}" if why else "")
            + _pixel_line(c)
            for c, v, n, why, missing in zip(text, f, m, reasons, no_fit)
        ]
        rank = np.arange(1, len(order) + 1)
        fig = go.Figure(
            go.Scatter(
                x=rank,
                y=f,
                mode="markers+text" if input.ring_labels() else "markers",
                text=text,
                textposition="top center",
                hovertext=hover,
                hoverinfo="text",
                marker=dict(
                    size=size,
                    color=f,
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    # a class without a fit is drawn at 0 as a cross
                    symbol=[
                        "x" if missing else "diamond" if bad else "circle"
                        for missing, bad in zip(no_fit, disagree)
                    ],
                    line=dict(
                        color=[
                            "#d62728" if why else "rgba(0,0,0,0.3)" for why in reasons
                        ],
                        width=[2.5 if why else 0.5 for why in reasons],
                    ),
                ),
                showlegend=False,
            )
        )
        if any(disagree):
            fig.add_annotation(
                text="\u25c6 pixels disagree",
                xref="paper",
                yref="paper",
                x=1.0,
                y=0.97,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                font=dict(size=10, color="#555"),
            )
        cut = phase.poorly_fitting_cut(fit)
        if cut is not None:
            # the chance level: a class below it is taken out. Drawn over the
            # markers and the grid, so it is not lost among them
            fig.add_hline(
                y=cut,
                line=dict(color="#b2182b", width=2, dash="dash"),
                layer="above",
                annotation_text=f"\u2248 junk ({cut:.2f})",
                # at the best-fitting end, clear of the poor classes
                annotation_position="top left",
                annotation_font=dict(size=10, color="#b2182b"),
            )
        threshold = input.fit_threshold()
        if threshold is not None:
            fig.add_hline(
                y=float(threshold),
                line=dict(color="#ff7f0e", width=2),
                layer="above",
                annotation_text=f"threshold ({float(threshold):.2f})",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#d95f02"),
            )
        fig.update_layout(
            template="plotly_white",
            title_text="Class fits, ranked",
            title_x=0.5,
            title_font=dict(size=12),
            xaxis=dict(title="Rank", range=[0.3, len(order) + 0.7]),
            # the same fixed scale as the ring's colours
            # fixed at the top, as the ring's colours; down to the worst fit,
            # which can be below 0 (a class against the ring)
            yaxis=dict(title="Fit", range=[min(-0.05, float(f.min()) - 0.07), 1.05]),
            margin=dict(t=40, b=50, l=50, r=20),
            hovermode="closest",
        )
        return fig

    def _agreement_figure(r, agreement):
        """Class fit against the agreement of the registered class averages.

        The good classes are at the top right: they fit the ring by their segments
        (up) and the registered averages put them where the ring does (right).
        Bad classes, other types and junk are elsewhere: left (the pixels disagree),
        down (the segments do), or both. Each marker's area follows the class's
        number of segments, the classes to take out are outlined in red, and the
        fit threshold and the chance level are the same horizontal lines as before.
        """
        import plotly.graph_objects as go

        ids = [int(c) for c in r.class_ids]
        fit = np.asarray(r.class_fit, dtype=float)
        agree = np.asarray(agreement, dtype=float)
        members = np.asarray(
            r.class_count if r.class_count is not None else r.class_weight,
            dtype=float,
        )
        size = 6 + 24 * np.sqrt(members) / max(float(np.sqrt(members).max()), 1e-12)
        poor = _poorly_fitting(r)
        reasons = [poor.get(c, "") for c in ids]
        no_fit = ~np.isfinite(fit)
        no_pixels = ~np.isfinite(agree)
        check = getattr(r, "pixel_check", None)
        pairs = (
            np.zeros(len(ids), dtype=int)
            if check is None or check.pairs is None
            else np.asarray(check.pairs)
        )
        y = np.nan_to_num(fit, nan=0.0)
        x = np.nan_to_num(agree, nan=0.0)
        hover = [
            f"class {c}<br>"
            + ("no fit" if nf else f"fit {f:.2f}")
            + "<br>"
            + (
                "no registered pair: not judged"
                if npx
                else f"pixel agreement {a:.2f} ({int(k)} registered pairs)"
            )
            + f"<br>{int(n):,} segments"
            + (f"<br>\u2717 {_SHORT_REASON.get(why, why)}" if why else "")
            for c, f, a, n, k, why, nf, npx in zip(
                ids, y, x, members, pairs, reasons, no_fit, no_pixels
            )
        ]
        fig = go.Figure(
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text" if input.ring_labels() else "markers",
                text=[str(c) for c in ids],
                textposition="top center",
                hovertext=hover,
                hoverinfo="text",
                marker=dict(
                    size=size,
                    color=y,
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=1.0,
                    symbol=[
                        "x" if nf or npx else "circle"
                        for nf, npx in zip(no_fit, no_pixels)
                    ],
                    line=dict(
                        color=[
                            "#d62728" if why else "rgba(0,0,0,0.3)" for why in reasons
                        ],
                        width=[2.5 if why else 0.5 for why in reasons],
                    ),
                ),
                showlegend=False,
            )
        )
        cut = phase.poorly_fitting_cut(fit)
        if cut is not None:
            fig.add_hline(
                y=cut,
                line=dict(color="#b2182b", width=2, dash="dash"),
                layer="above",
                annotation_text=f"\u2248 junk ({cut:.2f})",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#b2182b"),
            )
        threshold = input.fit_threshold()
        if threshold is not None:
            fig.add_hline(
                y=float(threshold),
                line=dict(color="#ff7f0e", width=2),
                layer="above",
                annotation_text=f"threshold ({float(threshold):.2f})",
                annotation_position="top left",
                annotation_font=dict(size=10, color="#d95f02"),
            )
        cut_x = input.agree_threshold()
        if cut_x:
            fig.add_vline(
                x=float(cut_x),
                line=dict(color="#7b3294", width=2),
                layer="above",
                annotation_text=f"agreement ({float(cut_x):.2f})",
                annotation_position="bottom right",
                annotation_font=dict(size=10, color="#7b3294"),
            )
        fig.add_annotation(
            text="good classes",
            xref="paper",
            yref="paper",
            x=1.0,
            y=1.0,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=10, color="#555"),
        )
        fig.update_layout(
            template="plotly_white",
            title_text="Class fit and pixel agreement",
            title_x=0.5,
            title_font=dict(size=12),
            # both on their fixed 0 to 1 scales (the fit down to the worst class,
            # which can be below 0)
            xaxis=dict(
                title="Pixel agreement",
                range=[-0.04, 1.06],
            ),
            yaxis=dict(
                title="Fit",
                range=[min(-0.05, float(y.min()) - 0.07), 1.05],
            ),
            margin=dict(t=40, b=50, l=50, r=20),
            hovermode="closest",
        )
        return fig

    @render.ui
    def class_fit_rank_plot():
        """The class fit against the agreement of the registered class averages.

        With the class-average check of the ring (helical_pitch_pixel) the classes
        are plotted by it and their fit: the good ones at the top right. Without it
        (too few registered pairs, or the check failed) the fits ranked, best first.
        """
        r = phase_result()
        req(r is not None)
        check = getattr(r, "pixel_check", None)
        agreement = None if check is None else check.agreement
        if agreement is not None and np.any(np.isfinite(agreement)):
            return _fig_to_html(_agreement_figure(r, agreement))
        return _fig_to_html(_rank_figure(r))

    def _length_range():
        lo, hi = input.length_range() or (None, None)
        return lo, hi

    def _pitch_band():
        lo, hi = input.pitch_band() or (None, None)
        return lo, hi

    @render.ui
    def filament_pitch_plot():
        r = phase_result()
        req(r is not None)
        table = r.filaments
        n_unfitted = int((~table["fitted"]).sum())
        table = table[table["fitted"]]
        req(len(table) > 0)
        return _fig_to_html(
            _filament_joint_figure(
                table, r.period, _length_range(), _pitch_band(), n_unfitted
            )
        )

    def _write_star(out):
        import starfile

        yield starfile.to_string(
            dict(optics=out.attrs["optics"], particles=out)
            if "optics" in out.attrs
            else dict(particles=out)
        )

    @reactive.calc
    def _angles():
        r = phase_result()
        req(r is not None)
        fold = helicon.shiny.clamp_number(input.rot_fold(), 1, 1, 12)
        return phase.segment_angles(analysis_params(), r, fold=fold)

    def _download_label(subset, prefix="Download"):
        n_fil = subset.groupby(["rlnMicrographName", "rlnHelicalTubeID"]).ngroups
        return f"{prefix} {n_fil:,} filaments / {len(subset):,} segments (.star)"

    @render.ui
    def pitch_band_download():
        r = phase_result()
        req(r is not None)
        low, high = _pitch_band()
        lo, hi = _length_range()
        req(low is not None and high is not None and high >= low)
        subset = phase.select_segments(
            analysis_params(),
            r.filaments,
            low,
            high,
            min_length=lo,
            max_length=hi,
            class_numbers=_selected_class_numbers(),
        )
        star_name = f"helices_pitch_{low:.0f}-{high:.0f}.star"
        csym = _imposed_csym()
        band_ui = render.download(
            label=_download_label(subset, "Download"),
            filename=star_name,
        )

        @band_ui
        def download_pitch_band():
            low, high = _pitch_band()
            lo, hi = _length_range()
            yield from _write_star(
                phase.select_segments(
                    analysis_params(),
                    r.filaments,
                    low,
                    high,
                    min_length=lo,
                    max_length=hi,
                    angles=_angles(),
                    class_numbers=_selected_class_numbers(),
                )
            )

        return ui.tooltip(
            map_viewers.download_link(
                session, "download_pitch_band", _download_label(subset, "Download")
            ),
            ui.HTML(
                "A RELION star file of the segments in the selected classes, with "
                'their estimated <span style="font-family: monospace; color: #ffd479;">rlnAngleRot</span>, <span style="font-family: monospace; color: #ffd479;">rlnAngleTilt</span> '
                '(90), <span style="font-family: monospace; color: #ffd479;">rlnAnglePsi</span> and origins, and each filament\'s '
                "pitch, length and direction score. It can be used directly to "
                "reconstruct a 3D map, without helical symmetry, for example (the "
                "rlnAngleRot follow the C symmetry set below; --sym imposes it when "
                "Impose C symmetry is ticked in Parameters):<br>"
                f'<span style="font-family: monospace; color: #ffd479;">relion_reconstruct --i {star_name} '
                f"--o map.mrc --ctf --sym c{csym}</span><br>"
                "and the map serves as a starting reference for refinement."
            ),
            placement="top",
        )

    # ── a 3D map from the segments, by relion_reconstruct ──

    def _imposed_csym():
        """The C symmetry to impose on a map: the tab's, when Impose C symmetry is
        ticked."""
        return (
            helicon.shiny.clamp_number(input.rot_fold(), 1, 1, 12)
            if input.map_csym()
            else 1
        )

    relion_map = reactive.value(None)

    @reactive.effect
    @reactive.event(phase_result)
    def _clear_relion_map():
        relion_map.set(None)

    def _relion_project_guess():
        """The RELION project folder of the parameter file, if it can be found."""
        if deployment.is_cloud():  # no looking at the server's files
            return ""
        p = params()
        if p is None:
            return ""
        # the parameter file on this computer: chosen in the server mode, or a
        # path (or file:// URL) typed in the url mode
        mode = input.input_mode_params()
        source = _source(mode, input.url_params, input.server_params)
        star = class2d_files.local_path(source)
        if star is not None:
            # the project root, two folders above the star file (../..), which has
            # default_pipeline.star; the paths in the star file (stacks, or
            # micrographs) start from there
            root = class2d_files.relion_project_root(star)
            if root:
                return root
        if "rlnImageName" not in p:
            return ""
        image = str(p["rlnImageName"].iloc[0]).split("@")[-1]
        if Path(image).is_absolute():
            return str(Path(image).parent) if Path(image).exists() else ""
        found = class2d_files.project_folder(source, image)
        if found:
            return found
        for folder in [Path.cwd(), *Path.cwd().parents]:
            if (folder / image).exists():
                return str(folder)
        return ""

    # the pixel size the previous time the size boxes were set, to keep the physical
    # extent when the pixel size is changed
    relion_prev_apix = reactive.value(None)

    def _relion_default_geometry():
        """Pixel size and box of the stored segment images (those of the class
        averages), or None."""
        found = data_all()
        images = displayed_class_images()
        if found is None or found[0] is None or not len(images):
            return None
        return float(found[1]), int(images[0].shape[0])

    @reactive.effect
    @reactive.event(input.relion_apix)
    def _rescale_relion_boxes():
        # the boxes are in pixels: a new pixel size keeps what they cover
        new, old = input.relion_apix(), relion_prev_apix()
        if new is None or not old or new <= 0 or abs(new - old) < 1e-9:
            return
        relion_prev_apix.set(float(new))
        for name in ("relion_diameter", "relion_length"):
            value = input[name]()
            if value:
                size = int(round(value * old / new / 2.0)) * 2
                ui.update_numeric(name, value=max(2, size))

    @render.ui
    def relion_ui():
        r = phase_result()
        req(r is not None)
        # nothing is shown, and no message, where RELION cannot be used: on a hosted
        # copy (no external programs run on visitors' data, and the reconstruction
        # would need a RELION project folder on the server and minutes of its CPUs
        # per visitor) and where relion_reconstruct is not on this server's PATH
        if deployment.is_cloud():
            return None
        exe = relion.find_relion_reconstruct()
        if exe is None:
            return None
        return ui.TagList(_relion_project_row(exe), ui.output_ui("relion_fused_ui"))

    def _relion_project_row(exe):
        return ui.layout_columns(
            ui.input_text(
                "relion_project_dir",
                ui.span(
                    "RELION project",
                    _info(
                        "The directory the paths in the star file start from: the "
                        "particle stacks (to reconstruct from the stored segment "
                        "images) or the micrographs (to cut the segments from them "
                        "at another pixel size or box); only the ones used need to "
                        "be there"
                    ),
                ),
                value=_relion_project_guess(),
                width="100%",
            ),
            _tip(
                ui.input_task_button(
                    "relion_run",
                    "relion_reconstruct",
                    style="width: 100%;",
                ),
                "Reconstruct the segments in the selected ranges with their angles "
                "and origins: no helical symmetry, no C symmetry unless Impose C "
                "symmetry is ticked in Parameters (as for the map from the class "
                "averages), and "
                "CTF correction when the star file has it. "
                + (
                    "Runs relion_reconstruct_mpi, with as many MPI processes as "
                    "the free CPUs and memory allow."
                    if relion.find_relion_reconstruct_mpi()
                    else "Runs relion_reconstruct with threads (no MPI version found)."
                ),
            ),
            col_widths=(7, 5),
            style="align-items: flex-end; margin-top: 8px;",
            fill=False,
        )

    @reactive.calc
    def _fused_state():
        """Whether the segments can be cut from the micrographs: ``"yes"``, ``"no
        micrographs"`` (this RELION can, but they are not under the project folder)
        or ``"no"``. A calc, so what depends on it is redrawn when it changes, not
        at each edit of the project folder."""
        exe = relion.find_relion_reconstruct()
        if exe is None or deployment.is_cloud():
            return "no"
        if not relion.fused_extract_available(exe):
            return "no"
        if _relion_default_geometry() is None:
            return "no"
        found = relion.micrographs_found(params(), input.relion_project_dir())
        return "yes" if found else "no micrographs"

    @render.ui
    def relion_fused_ui():
        state = _fused_state()
        if state == "no":
            return None
        if state == "no micrographs":
            return ui.p(
                "To reconstruct at another pixel size and box, the micrographs of "
                "the segments (and their coordinates) have to be under the RELION "
                "project folder.",
                class_="small text-muted",
                style="margin: 2px 0 0 0;",
            )
        geometry = _relion_default_geometry()
        relion_prev_apix.set(geometry[0])
        return ui.TagList(
            ui.layout_columns(
                ui.input_numeric(
                    "relion_apix",
                    ui.span(
                        "Pixel size (\u00c5)",
                        _info(
                            "The pixel size of the reconstruction. The segments "
                            "are cut from the micrographs at this pixel size, so "
                            "it can be finer than the class averages' "
                            f"({geometry[0]:.3f} \u00c5). Changing it keeps the "
                            "area the two sizes cover."
                        ),
                    ),
                    value=round(geometry[0], 4),
                    min=0.1,
                    step=0.01,
                    update_on="blur",
                    width="100%",
                ),
                ui.input_numeric(
                    "relion_diameter",
                    ui.span(
                        "Diameter (px)",
                        _info(
                            "The width of the reconstruction across the filament, "
                            "in pixels of this pixel size: the map is this wide "
                            "in x and y."
                        ),
                    ),
                    value=geometry[1],
                    min=2,
                    step=2,
                    update_on="blur",
                    width="100%",
                ),
                ui.input_numeric(
                    "relion_length",
                    ui.span(
                        "Length (px)",
                        _info(
                            "The length of each segment along the filament, in "
                            "pixels of this pixel size; the map is this long "
                            "(z)."
                        ),
                    ),
                    value=geometry[1],
                    min=2,
                    step=2,
                    update_on="blur",
                    width="100%",
                ),
                col_widths=(4, 4, 4),
                fill=False,
            ),
            ui.output_ui("relion_fused_note"),
        )

    def _relion_fused_request():
        """The ``fused`` argument of relion.reconstruct for the boxes set now, or
        None when they are those of the stored segment images (reconstruct from
        them) or there are no boxes."""
        if _fused_state() != "yes":
            return None
        geometry = _relion_default_geometry()
        apix, diameter, length = (
            input.relion_apix(),
            input.relion_diameter(),
            input.relion_length(),
        )
        if geometry is None or not (apix and diameter and length):
            return None
        apix = float(helicon.shiny.clamp_number(apix, geometry[0], 0.1, 100.0, float))
        diameter = int(helicon.shiny.clamp_number(diameter, geometry[1], 2, 4096))
        length = int(helicon.shiny.clamp_number(length, geometry[1], 2, 4096))
        if (
            abs(apix - geometry[0]) < 1e-6
            and diameter == geometry[1]
            and length == geometry[1]
        ):
            return None
        return dict(angpix=apix, diameter=diameter, length=length)

    @render.ui
    def relion_fused_note():
        fused = _relion_fused_request()
        if fused is None:
            text = (
                "From the stored segment images, at the class averages' pixel size "
                "and box."
            )
        else:
            a, d, n = fused["angpix"], fused["diameter"], fused["length"]
            text = (
                f"Cut from the micrographs at {a:g} \u00c5/pixel: {d * a:.0f} \u00c5 "
                f"wide, {n * a:.0f} \u00c5 long; the map is {d} \u00d7 {d} \u00d7 {n} "
                "pixels."
            )
        return ui.p(text, class_="small text-muted", style="margin: 2px 0 0 0;")

    def _relion_work(job, progress):
        progress.set(
            0,
            message=f"relion_reconstruct: {len(job['subset']):,} segments",
            detail="this can take a few minutes",
        )
        return relion.reconstruct(
            job["subset"],
            job["project_dir"],
            cpu=max(1, int(helicon.available_cpu())),
            csym=job["csym"],
            fused=job["fused"],
        )

    def _relion_apply(job, result):
        # a map from segments of an older pitch result is not shown
        if phase_result() is not job["phase_result"]:
            return
        relion_map.set(result)

    def _relion_error(job, e):
        ui.modal_show(
            ui.modal(
                ui.tags.pre(str(e)[-3000:]),
                title="relion_reconstruct failed",
                easy_close=True,
                footer=None,
            )
        )

    relion_task = helicon.shiny.background_task(
        "relion_run",
        _relion_work,
        _relion_apply,
        _relion_error,
        session=session,
        label="relion_reconstruct",
    )

    @reactive.effect
    @reactive.event(input.relion_run)
    def run_relion():
        r = phase_result()
        req(r is not None)
        # the RELION project folder is a free-text path on the server
        if deployment.refuse_server_mode():
            return
        low, high = _pitch_band()
        lo, hi = _length_range()
        subset = phase.select_segments(
            analysis_params(),
            r.filaments,
            low,
            high,
            min_length=lo,
            max_length=hi,
            angles=_angles(),
            class_numbers=_selected_class_numbers(),
        )
        if not len(subset):
            ui.notification_show("No segment is in the selected ranges.", duration=5)
            return
        fused = _relion_fused_request()
        if fused is not None and "rlnCoordinateX" not in subset:
            ui.notification_show(
                "Cutting the segments from the micrographs needs rlnCoordinateX "
                "and rlnCoordinateY in the star file.",
                duration=6,
            )
            return
        relion_task.invoke(
            dict(
                phase_result=r,
                subset=subset,
                project_dir=input.relion_project_dir(),
                csym=_imposed_csym(),
                fused=fused,
            )
        )

    map_viewers.register(
        input,
        session,
        "relion_view",
        lambda: (
            None
            if relion_map() is None
            else (relion_map()["volume"], relion_map()["apix"])
        ),
        "abinitio3d_relion_map",
    )

    @render.ui
    def relion_display():
        m = relion_map()
        req(m is not None)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        volume = m["volume"]
        nz = volume.shape[0]
        side = volume.sum(axis=1)  # z rows, x columns
        z_view = volume[nz // 2]
        fig = make_subplots(
            rows=1,
            cols=2,
            column_widths=[0.6, 0.4],
            horizontal_spacing=0.04,
            subplot_titles=("Side projection", "Central z section"),
        )
        for image, col in ((side.T, 1), (z_view, 2)):
            fig.add_trace(
                go.Heatmap(
                    z=image, colorscale="gray", showscale=False, hoverinfo="skip"
                ),
                row=1,
                col=col,
            )
            fig.update_yaxes(
                autorange="reversed",
                scaleanchor=f"x{'' if col == 1 else col}",
                showticklabels=False,
                row=1,
                col=col,
            )
            fig.update_xaxes(showticklabels=False, row=1, col=col)
        fig.update_layout(
            template="plotly_white",
            title_text=(
                "relion_reconstruct"
                + (f" (MPI \u00d7{m['mpi']})" if m.get("mpi") else "")
                + f" \u00b7 {m['n_segments']:,} segments \u00b7 "
                f"{_box_text(volume.shape)} at {m['apix']:.3f} \u00c5 \u00b7 C{m['csym']}"
            ),
            title_x=0.5,
            title_font=dict(size=12),
            margin=dict(t=60, b=10, l=10, r=10),
            height=380,
        )
        btn = render.download(
            label="Download map (.mrc)",
            filename="abinitio3d_relion_map.mrc",
        )

        @btn
        def download_relion_map():
            # the run's own folder is gone (it was temporary): write the map
            # from memory
            yield _mrc_bytes(m["volume"], m["apix"])

        return ui.div(
            _fig_to_html(fig),
            map_viewers.button_row(
                map_viewers.download_link(
                    session, "download_relion_map", "Download map (.mrc)"
                ),
                session,
                "relion_view",
                button_class="btn-primary",
            ),
        )

    # ── a 3D map from the class averages ──

    map_result = reactive.value(None)

    @reactive.effect
    @reactive.event(phase_result)
    def _clear_map_result():
        map_result.set(None)

    def _map_work(job, progress):
        r = job["phase_result"]
        apix_orig = job["apix"]
        progress.set(0, message="straightening the class averages")
        prepared = maps.straighten_classes(
            job["images"],
            r.class_zdir,
            target_apix=max(5.0, apix_orig),
            apix=apix_orig,
        )
        progress.set(
            1,
            message=f"reconstructing from {len(prepared)} "
            + ("unique classes" if r.merged_classes else "classes"),
        )
        out = maps.reconstruct_map(
            prepared,
            np.rad2deg(r.phases),
            r.period,
            job["csym"],
            job["rise"],
            left_handed=job["left_handed"],
            apix=max(5.0, apix_orig),
            helical_sym_order=job["hsym"],
            impose_csym=job["impose_csym"],
            method=(
                "projection"
                if job["turns"] > 1
                else "backprojection" if job["method"] == "backprojection" else "joint"
            ),
            algorithm=dict(model="gauss") if job["method"] == "gauss" else None,
            turns=job["turns"],
            rounds=job["rounds"],
            output_box=int(job["images"][0].shape[0]),
            output_apix=apix_orig,
            cpu=max(1, int(helicon.available_cpu())),
        )
        out["turns"] = job["turns"]
        out["method_asked"] = job["method"]
        return out

    def _map_apply(job, result):
        # a map from an older pitch result is not shown
        if phase_result() is not job["phase_result"]:
            return
        map_result.set(result)

    def _map_error(job, e):
        ui.modal_show(
            ui.modal(
                f"The 3D map could not be reconstructed: {e}",
                title="3D map error",
                easy_close=True,
                footer=None,
            )
        )

    map_task = helicon.shiny.background_task(
        "map_run",
        _map_work,
        _map_apply,
        _map_error,
        progress_max=2,
        session=session,
        label="3D map from the class averages",
    )

    @reactive.effect
    @reactive.event(input.map_run)
    def run_map():
        r = phase_result()
        req(r is not None)
        images = displayed_class_images()
        at = _class_positions(r.class_ids, displayed_class_ids())
        rise = helicon.shiny.clamp_number(input.rise(), 4.75, 0.01, 1000.0, float)
        map_task.invoke(
            dict(
                phase_result=r,
                images=[images[i] for i in at],
                apix=float(data_all()[1]),
                csym=helicon.shiny.clamp_number(input.rot_fold(), 1, 1, 12),
                rise=rise,
                left_handed=input.map_hand() == "left",
                hsym=helicon.shiny.clamp_number(input.map_hsym(), 1, 1, 1000),
                impose_csym=bool(input.map_csym()),
                method=input.map_method(),
                # turns of 360/C per repeat; with more than one the ring angles
                # are not precise enough to place the averages, which are then
                # placed by projection matching
                turns=lattice.periods_per_repeat(
                    r.period, rise, getattr(r, "lattice", None)
                )[0],
                rounds=2 if input.map_refine() else 1,
            )
        )

    @reactive.effect
    @reactive.event(input.use_lattice_rise)
    def _use_lattice_rise():
        r = phase_result()
        lat = getattr(r, "lattice", None) if r is not None else None
        req(lat is not None)
        ui.update_numeric("rise", value=round(float(lat.rise), 2))

    def _map_classes_text(n_used):
        """The selected classes, and how many the map used after merging."""
        r = phase_result()
        n_merged = len(r.merged_classes) if r is not None else 0
        if not n_merged:
            return f"{n_used} classes"
        return (
            f"{n_used + n_merged} classes ({n_merged} merged -> {n_used} unique "
            "classes)"
        )

    @render.ui
    def map_display():
        m = map_result()
        if m is None:
            return None
        title = (
            f"{_map_classes_text(len(m['phis']))} \u00b7 twist {m['twist']:.3f}\u00b0 \u00b7 "
            f"fit {m['score']:.2f}"
            + (
                f" \u00b7 {m['turns']} turns per repeat: placed by projection "
                "matching"
                + (
                    # the ring angles cannot place the averages, which is all
                    # back-projection has to go on: it is not used
                    f" ({_MAP_METHODS['backprojection']} is not used then)"
                    if m.get("method_asked") == "backprojection"
                    else ""
                )
                if m.get("turns", 1) > 1
                else ""
            )
        )
        return _fig_to_html(_map_figure(m, title))

    @render.ui
    def map_download():
        m = map_result()
        req(m is not None)
        btn = render.download(
            label="Download map (.mrc)",
            filename="abinitio3d_class_map.mrc",
        )

        @btn
        def download_map():
            import tempfile

            import mrcfile

            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "map.mrc")
                with mrcfile.new(path) as mrc:
                    mrc.set_data(m["volume_out"].astype(np.float32))
                    mrc.voxel_size = m["apix_out"]
                yield Path(path).read_bytes()

        return map_viewers.button_row(
            map_viewers.download_link(session, "download_map", "Download map (.mrc)"),
            session,
            "map_view",
            button_class="btn-primary",
        )

    map_viewers.register(
        input,
        session,
        "map_view",
        lambda: (
            None
            if map_result() is None
            else (map_result()["volume_out"], map_result()["apix_out"])
        ),
        "abinitio3d_class_map",
    )
