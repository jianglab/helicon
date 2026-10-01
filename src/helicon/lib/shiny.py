from typing import Optional
from pathlib import Path

import shiny
from shiny import reactive
from shiny.express import ui, module, render, expressify
import logging

logger = logging.getLogger(__name__)


__all__ = [
    "file_selection_server",
    "file_selection_ui",
    "get_client_url",
    "get_client_url_query_params",
    "google_analytics",
    "image_gallery",
    "image_select",
    "launch_shiny_app",
    "range_slider",
    "set_client_url_query_params",
    "slider",
]


# The script every editable slider carries (it does its work once per page):
#
# * Double-click a number of a slider -- for a range slider the lower or the
#   upper one, and for either kind the smallest or the largest value at its ends
#   -- to type it in place; Enter applies it, Escape (or clicking away) leaves
#   the slider as it was.
# * A slider sends its value to the server only when the drag ends, not all along
#   the way, unless its wrapper says ``data-emit-while-sliding``. Shiny's own
#   slider sends on every move (a quarter of a second apart at most), which is a
#   lot of reactive work for an app that redraws on each. The numbers shown on the
#   slider still follow the drag. Clicks on the bar, the keyboard, typed values
#   and updates from the server are not drags and send at once.
_SLIDER_EDIT_JS = """
(function () {
  if (window.__heliconRangeEdit) { return; }
  window.__heliconRangeEdit = true;

  var LABELS = '.helicon-editable-slider .irs-from, .helicon-editable-slider .irs-to, ' +
               '.helicon-editable-slider .irs-single, .helicon-editable-slider .irs-min, ' +
               '.helicon-editable-slider .irs-max';

  $(document).on('dblclick', LABELS, function (event) {
    var irs = $(this).closest('.irs');
    var input = irs.closest('.shiny-input-container').find('input.js-range-slider').first();
    var slider = input.data('ionRangeSlider');
    if (!slider) { return; }
    var rect = this.getBoundingClientRect();
    var single = slider.options.type === 'single';
    var bound = $(this).hasClass('irs-min') ? 'min' : ($(this).hasClass('irs-max') ? 'max' : null);
    // close together, or equal, the two numbers of a range are shown as one
    // label: its left half is the lower number and its right half the upper
    var merged = !single && (slider.result.from === slider.result.to ||
      $(this).hasClass('irs-single'));
    var isFrom = single || (merged ? event.clientX < rect.left + rect.width / 2
                                   : $(this).hasClass('irs-from'));
    var current = bound ? slider.options[bound] : (isFrom ? slider.result.from : slider.result.to);
    var box = $('<input type="text" inputmode="decimal">')
      .val(current)
      .css({position: 'fixed', left: rect.left - 8, top: rect.top - 3,
            width: Math.max(rect.width + 16, 64), height: rect.height + 6,
            zIndex: 10000, fontSize: '12px', textAlign: 'center',
            border: '1px solid #0d6efd', borderRadius: '3px', padding: '0 2px'});
    $('body').append(box);
    box.focus().select();
    var done = function () { if (box[0].isConnected) { box.remove(); } };
    box.on('keydown', function (ev) {
      if (ev.key === 'Enter') {
        var v = parseFloat(String(box.val()).replace(/[ ,]/g, ''));
        done();
        if (isNaN(v)) { return; }
        if (bound === 'min') {
          if (v >= slider.options.max) { return; }
          slider.update({min: v});
        } else if (bound === 'max') {
          if (v <= slider.options.min) { return; }
          slider.update({max: v});
        } else if (single) {
          slider.update({from: v});
        } else {
          var from = slider.result.from, to = slider.result.to;
          if (isFrom) { from = v; to = Math.max(to, v); } else { to = v; from = Math.min(from, v); }
          slider.update({from: from, to: to});
        }
        input.trigger('change');
      } else if (ev.key === 'Escape') {
        done();
      }
    });
    box.on('blur', done);
  });

  // send on release: hold back the change events of a drag and send one when it ends
  function gate(wrapper) {
    if (wrapper.hasAttribute('data-emit-while-sliding')) { return true; }
    var input = $(wrapper).find('input.js-range-slider').first();
    if (!input.length) { return false; }
    var slider = input.data('ionRangeSlider');
    var events = $._data(input[0], 'events');
    if (!slider || !events || !events.change) { return false; }
    if (input.data('heliconGated')) { return true; }
    input.data('heliconGated', true);
    events.change.forEach(function (h) {
      var original = h.handler;
      h.handler = function () {
        if (slider.dragging && !input.data('heliconFlush')) { return; }
        return original.apply(this, arguments);
      };
    });
    var onFinish = slider.options.onFinish;
    slider.update({
      onFinish: function (data) {
        input.data('heliconFlush', true);
        input.trigger('change');
        input.data('heliconFlush', false);
        if (onFinish) { onFinish(data); }
      }
    });
    return true;
  }

  function scan() {
    document.querySelectorAll('.helicon-editable-slider').forEach(function (w) {
      if (!w.__heliconGateTried || w.__heliconGateTried < 40) {
        w.__heliconGateTried = (w.__heliconGateTried || 0) + 1;
        if (gate(w)) { w.__heliconGateTried = 1000; }
      }
    });
  }
  new MutationObserver(scan).observe(document.documentElement, {childList: true, subtree: true});
  $(document).on('shiny:connected shiny:value shiny:idle', scan);
  setInterval(scan, 500);
  scan();
})();
"""


def range_slider(
    id,
    label,
    min,
    max,
    value,
    step=None,
    width="100%",
    emit_while_sliding=False,
    **kwargs,
):
    """A two-handle slider whose numbers can also be typed in place.

    Everything a Shiny range slider does is unchanged: drag a handle, or click
    the bar, and ``input.<id>()`` is the ``(low, high)`` pair, and
    ``ui.update_slider`` works on it. In addition, double-clicking the lower or
    the upper number opens an editor on that number; Enter applies what was
    typed and Escape, or clicking away, leaves the slider as it was. Where the
    two numbers are close enough to be shown as one label, its left half is the
    lower number and its right half the upper one (likewise when the two are equal). A lower number typed above
    the upper one carries the upper one with it, and the other way round; both
    are limited to the slider's ends. Double-clicking the smallest or the largest
    value at the ends of the slider edits the end itself.

    Parameters
    ----------
    id : str
        Input id, as for ``shiny.ui.input_slider``.
    label : str
        The label shown above the slider.
    min, max : float
        The ends of the slider.
    value : tuple of float
        The initial ``(low, high)``.
    step : float, optional
        The step of the slider.
    width : str, optional
        CSS width. Defaults to ``"100%"``.
    emit_while_sliding : bool, optional
        By default the value is sent to the server when a drag ends, not all
        along the way, so an app that redraws on each change is not asked to
        while the handle is still moving; the numbers on the slider follow the
        drag either way. True sends as Shiny's own slider does. Clicking the
        bar, the keyboard, typed values and ``ui.update_slider`` always send at
        once. Defaults to False.
    **kwargs
        Passed to ``shiny.ui.input_slider``.

    Returns
    -------
    shiny.ui.Tag
        A ``div`` holding the slider and the editor's script (which does its
        work once per page however many sliders there are).
    """
    return _editable_slider(
        id, label, min, max, value, step, width, emit_while_sliding, kwargs
    )


def slider(
    id,
    label,
    min,
    max,
    value,
    step=None,
    width="100%",
    emit_while_sliding=False,
    **kwargs,
):
    """A single-value slider whose number can also be typed in place.

    Like ``shiny.ui.input_slider`` with one handle: drag it, or click the bar,
    and ``input.<id>()`` is the value; ``ui.update_slider`` works on it. In
    addition, double-clicking the number opens an editor on it; Enter applies
    what was typed and Escape, or clicking away, leaves the slider as it was. A
    number beyond either end is limited to it, and double-clicking the smallest
    or the largest value at the ends of the slider edits the end itself.

    Parameters
    ----------
    id : str
        Input id, as for ``shiny.ui.input_slider``.
    label : str
        The label shown above the slider.
    min, max : float
        The ends of the slider.
    value : float
        The initial value.
    step : float, optional
        The step of the slider.
    width : str, optional
        CSS width. Defaults to ``"100%"``.
    emit_while_sliding : bool, optional
        By default the value is sent to the server when a drag ends, not all
        along the way, so an app that redraws on each change is not asked to
        while the handle is still moving; the numbers on the slider follow the
        drag either way. True sends as Shiny's own slider does. Clicking the
        bar, the keyboard, typed values and ``ui.update_slider`` always send at
        once. Defaults to False.
    **kwargs
        Passed to ``shiny.ui.input_slider``.

    Returns
    -------
    shiny.ui.Tag
        A ``div`` holding the slider and the editor's script (which does its
        work once per page however many sliders there are).
    """
    return _editable_slider(
        id, label, min, max, value, step, width, emit_while_sliding, kwargs
    )


def _editable_slider(
    id, label, min, max, value, step, width, emit_while_sliding, kwargs
):
    from shiny import ui as core_ui

    # one element, the script inside it, so that a layout that arranges its
    # children (a grid of columns, say) sees one child per slider
    return core_ui.div(
        core_ui.input_slider(
            id,
            label,
            min=min,
            max=max,
            value=value,
            step=step,
            width=width,
            **kwargs,
        ),
        core_ui.tags.script(_SLIDER_EDIT_JS),
        class_="helicon-editable-slider",
        **({"data-emit-while-sliding": "true"} if emit_while_sliding else {}),
    )


def image_gallery(
    id,
    label=reactive.value(""),
    images=reactive.value([]),
    display_image_labels=True,
    display_dashed_line=False,
    image_labels=reactive.value([]),
    image_links=reactive.value([]),
    image_size=reactive.value(128),
    image_border=2,
    gap=0,
    justification="center",
    enable_selection=False,
    allow_multiple_selection=False,
    initial_selected_indices=reactive.value([]),
    style="",
):
    """Render a gallery of images as a flexbox grid.

    Supports selection, labels, links, and custom styling via Shiny UI.

    Parameters
    ----------
    id : str
        Unique element ID.
    label : reactive.value, optional
        Gallery heading label.
    images : reactive.value, optional
        List of images (file paths, PIL Images, or 2D numpy arrays).
    display_image_labels : bool, optional
        Whether to show image labels. Defaults to True.
    display_dashed_line : bool, optional
        Whether to show a dashed midline. Defaults to False.
    image_labels : reactive.value, optional
        Labels for each image.
    image_links : reactive.value, optional
        Links for each image.
    image_size : reactive.value, optional
        Image display size in pixels. Defaults to 128.
    image_border : int, optional
        Border width in pixels. Defaults to 2.
    gap : int, optional
        Gap between images in pixels. Defaults to 0.
    justification : str, optional
        Flexbox justify-content value. Defaults to ``"center"``.
    enable_selection : bool, optional
        If True, adds click-to-select behavior. Defaults to False.
    allow_multiple_selection : bool, optional
        If True, allows selecting multiple images. Defaults to False.
    initial_selected_indices : reactive.value, optional
        Indices of pre-selected images.
    style : str, optional
        Additional CSS inline styles.

    Returns
    -------
    ui.Tag or tuple
        UI element(s) for the gallery.
    """
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

        if display_image_labels or display_dashed_line:
            elements = [img]
            if display_image_labels:
                elements.append(
                    ui.p(
                        label,
                        style="text-align: left; color: white; text-shadow: -1px -1px 0.5px rgba(0,0,0,0.5), 1px -1px 0.5px rgba(0,0,0,0.5), -1px 1px 0.5px rgba(0,0,0,0.5), 1px 1px 0.5px rgba(0,0,0,0.5); position: absolute; top: 2px; left: 5px;",
                    )
                )
            if display_dashed_line:
                elements.append(
                    ui.div(
                        style=f"border-top: 1px dashed white; position: absolute; top: {image_size() // 2}px; left: 0; right: 0;"
                    ),
                )
            ui_img = ui.div(
                *elements,
                style="position: relative;",
            )
        else:
            ui_img = img

        return ui.div(
            ui_img,
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
        style=f"display: flex; flex-flow: row wrap; justify-content: {justification}; justify-items: center; align-items: center; gap: {gap}px {gap}px; margin: 0 0 {image_border}px 0;",
    )

    if len(label()):
        ui_images = ui.div(
            ui.h6(
                label(),
                style=f"text-align: {justification}; margin: 0;",
                title="Hold the Shift key while clicking to select multiple images; Hold the Alt/Option key while clicking to select/unselect all images",
            ),
            ui_images,
            style=f"display: flex; flex-direction: column; gap: {gap}px; margin: 0;",
        )

    if len(style):
        ui_images.add_style(style)

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
    display_dashed_line=False,
    image_labels=reactive.value([]),
    image_links=reactive.value([]),
    image_size=reactive.value(128),
    image_border=2,
    gap=0,
    justification="center",
    enable_selection=True,
    allow_multiple_selection=True,
    initial_selected_indices=reactive.value([]),
    style="",
):
    @shiny.render.ui
    def show_image_gallery():
        return image_gallery(
            id=session.ns,
            label=label,
            images=images,
            display_image_labels=display_image_labels,
            display_dashed_line=display_dashed_line,
            image_labels=image_labels,
            image_links=image_links,
            image_size=image_size,
            initial_selected_indices=initial_selected_indices,
            enable_selection=enable_selection,
            allow_multiple_selection=allow_multiple_selection,
            image_border=image_border,
            gap=gap,
            justification=justification,
            style=style,
        )


# server-side file selection
@shiny.module.ui
def file_selection_ui(label="Select a file", value=None, width="100%"):
    """Shiny UI component for selecting a file via a browse popover.

    Parameters
    ----------
    label : str, optional
        Label for the file selector. Defaults to ``"Select a file"``.
    value : str, optional
        Initial file path. Defaults to None.
    width : str, optional
        CSS width. Defaults to ``"100%"``.

    Returns
    -------
    ui.Tag
        The file selection UI element.
    """
    return shiny.ui.div(
        shiny.ui.popover(
            shiny.ui.input_action_button(
                "browse", label="Browse", style="height: 30px; --bs-btn-padding-y: 0"
            ),
            shiny.ui.input_text(
                "current_directory",
                label="Current directory",
                value=str(Path(value).parent) if value else str(Path.cwd()),
                width="100%",
            ),
            shiny.ui.layout_column_wrap(
                shiny.ui.input_select(
                    "sub_directory",
                    "Go to a sub-directory",
                    choices=[],
                    width="100%",
                ),
                shiny.ui.input_select(
                    "file",
                    "Select a file",
                    choices=[Path(value).name] if value else [],
                    selected=Path(value).name if value else None,
                    width="100%",
                ),
                title="Select a file",
                width="100%",
            ),
            width="100%",
        ),
        shiny.ui.input_text(
            "selected_file_path", label=None, value=value, width="100%"
        ),
        style=f"display: flex; flex-flow: row; align-items: stretch; gap: 2px; margin: 0; padding: 0; width: {width};",
    )


@shiny.module.server
def file_selection_server(
    input,
    output,
    session,
    file_types: Optional[str | list[str]] = None,
    ignore_hidden_files=True,
):
    """Shiny server module for file selection with directory browsing.

    Parameters
    ----------
    input : shiny.Inputs
        Module input.
    output : shiny.Outputs
        Module output.
    session : shiny.Session
        Module session.
    file_types : str or list of str, optional
        Allowed file extensions. If None, all files shown.
    ignore_hidden_files : bool, optional
        If True, hide filenames starting with ``.``. Defaults to True.

    Returns
    -------
    reactive.value
        Reactive value with the selected file path.
    """
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
        except Exception:
            logger.error(
                "Failed to list sub-directories in %s",
                input.current_directory(),
                exc_info=True,
            )
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
        except Exception:
            logger.error(
                "Failed to list files in %s",
                str(input.current_directory()),
                exc_info=True,
            )
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
    if id is None or not len(id):
        return
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


def get_client_url(input):
    """Reconstruct the full client URL from Shiny input data.

    Parameters
    ----------
    input : shiny.Inputs
        Shiny input object.

    Returns
    -------
    str
        The full client URL.
    """
    d = input._map
    url = f"{d['.clientdata_url_protocol']()}//{d['.clientdata_url_hostname']()}:{d['.clientdata_url_port']()}{d['.clientdata_url_pathname']()}{d['.clientdata_url_search']()}"
    return url


def get_client_url_query_params(input, keep_list=True):
    """Parse query parameters from the client URL.

    Parameters
    ----------
    input : shiny.Inputs
        Shiny input object.
    keep_list : bool, optional
        If True, keep single-value parameters as lists.
        Defaults to True.

    Returns
    -------
    dict
        Parsed query parameters.
    """
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
    """Update the client URL query parameters without reloading the page.

    Parameters
    ----------
    query_params : dict
        Query parameters to set.

    Returns
    -------
    ui.Tag
        A script tag that updates the browser URL.
    """
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


def encode_query_params(query_params):
    """Encode a query_params dict into a URL query string (no leading ``?``).

    Keys with an empty string value are emitted bare; everything else is
    ``key=urlencoded_value``, with commas, slashes and colons left as they are
    (lists and file paths stay readable, the URL shorter). It is shared by the
    browser URL builder here and the ``/helicon/navigate`` endpoint of the web
    app so that a display re-click reproduces the exact launch URL, and it
    matches how the page writes its bookmark URL (``helicon.webApps.bookmark``).

    Parameters
    ----------
    query_params : dict
        Mapping of query key to value.

    Returns
    -------
    str
        The encoded query string, e.g. ``tab=HILL&url=/data/classes.mrcs``.
    """
    import urllib.parse

    parts = []
    for k, v in query_params.items():
        if v == "":
            parts.append(k)
        else:
            parts.append(f"{k}={urllib.parse.quote(str(v), safe=',/:')}")
    return "&".join(parts)


def _make_pdeathsig_preexec():
    """Return a ``preexec_fn`` that ties child lifetime to the parent (Linux).

    Uses ``prctl(PR_SET_PDEATHSIG, SIGTERM)`` so when the parent process
    exits for any reason (including SIGKILL), the kernel delivers SIGTERM
    to the child.  Returns ``None`` on non-Linux or if setup is unavailable.
    """
    import os
    import sys

    if sys.platform != "linux":
        return None

    def _preexec():
        try:
            import ctypes
            import signal as _signal

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_PDEATHSIG = 1
            libc.prctl(PR_SET_PDEATHSIG, int(_signal.SIGTERM))
            if os.getppid() == 1:
                os._exit(1)
        except Exception:
            pass

    return _preexec


def launch_shiny_app(
    app_file, env=None, block=True, query_params=None, reload=False, url_callback=None
):
    """Launch a Shiny app with automatic browser opening.

    Handles WSL2 where Python's webbrowser module fails to open the Windows
    browser. Captures the random port from server output and opens the browser
    manually.

    Parameters
    ----------
    app_file : str or Path
        Path to the Shiny app file or a module path like "package.module:app".
    env : dict, optional
        Environment variables for the subprocess. Defaults to None (inherits
        current environment).
    block : bool, optional
        If True (default), wait for the process to finish. If False, return
        the Popen object immediately (used by the file browser to avoid
        blocking the Qt event loop).
    query_params : dict, optional
        URL query parameters to append when opening the browser.
    reload : bool, optional
        If True, run in dev mode with auto-reload on the app's directory.
        Defaults to False.
    url_callback : callable, optional
        If given, called with the final app URL (including query params)
        instead of opening the browser.  Used by the file browser to record
        the launched URL for tab reuse.
    """
    import importlib
    import re
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "-m",
        "shiny",
        "run",
        "--no-dev-mode" if not reload else "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "0",
    ]

    if isinstance(app_file, str) and ":" in app_file and "/" not in app_file:
        import importlib.util

        module_name = app_file.split(":")[0]
        app_dir = None
        parts = module_name.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            try:
                spec = importlib.util.find_spec(candidate)
            except (ImportError, ValueError):
                continue
            if spec is not None and hasattr(spec, "origin") and spec.origin is not None:
                app_dir = str(Path(spec.origin).parent)
                break
        if app_dir is not None:
            cmd += ["--app-dir", app_dir]
    elif reload:
        app_dir = str(Path(str(app_file)).parent)
        cmd += ["--app-dir", app_dir]

    cmd.append(str(app_file))

    # On Linux, ask the kernel to SIGTERM this child when the parent dies
    # (including SIGKILL of the parent).  Harmless no-op elsewhere.
    preexec = _make_pdeathsig_preexec()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        preexec_fn=preexec,
    )

    if not block:
        import threading

        def _reader():
            found_url = False
            for line in proc.stdout:
                if not found_url:
                    m = re.search(r"Uvicorn running on http://[\d.]+:(\d+)", line)
                    if m:
                        base = f"http://localhost:{m.group(1)}/"
                        if query_params:
                            base += "?" + encode_query_params(query_params)
                        if url_callback is not None:
                            url_callback(base)
                        else:
                            _open_browser(base)
                        found_url = True

        threading.Thread(target=_reader, daemon=True).start()
        return proc

    url = None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = re.search(r"Uvicorn running on http://[\d.]+:(\d+)", line)
        if m:
            url = f"http://localhost:{m.group(1)}/"
            break

    if url:
        if query_params:
            url += "?" + encode_query_params(query_params)
        if url_callback is not None:
            url_callback(url)
        else:
            _open_browser(url)

    proc.wait()


def _is_wsl():
    """Detect if running inside WSL."""
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def _open_browser(url):
    """Open a URL in the system browser exactly once.

    Avoids ``webbrowser.open()``: on Linux it tries every registered browser
    in order (e.g. Opera then ``xdg-open``).  Opera often exits 0 after
    opening a tab, so Python treats that as failure and falls through to
    ``xdg-open`` — same URL, two tabs.  One explicit launcher prevents that.
    """
    import shutil
    import subprocess

    print(f"Opening browser at {url}...")
    if _is_wsl() and shutil.which("wslview"):
        subprocess.Popen(
            ["wslview", url],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return
    for cmd in ("xdg-open", "gio", "open"):
        exe = shutil.which(cmd)
        if exe is None:
            continue
        args = [exe, "open", url] if cmd == "gio" else [exe, url]
        subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return
    import webbrowser

    webbrowser.open(url, new=2)
