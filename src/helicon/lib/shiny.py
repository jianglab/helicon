from pathlib import Path

import shiny
from shiny import reactive
from shiny.express import ui, module, render
import logging

logger = logging.getLogger(__name__)

from .shiny_file_picker import (  # noqa: E402
    SERVER,
    source_modes,
    file_picker_button,
    file_picker_field,
    file_picker_fill,
    file_picker_server,
)

__all__ = [
    "SERVER",
    "source_modes",
    "file_picker_button",
    "file_picker_field",
    "file_picker_fill",
    "file_picker_server",
    "google_analytics",
    "image_gallery",
    "image_select",
    "launch_shiny_app",
    "range_slider",
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

  // show numbers to the slider's own precision: a value worked out on the
  // server (-0.012799999999970169) or a range end (300.0064) otherwise shows
  // every digit. Only the display: the value sent is as it was.
  function decimals(step) {
    var t = String(step);
    if (t.indexOf('e-') >= 0) { return Math.min(6, parseInt(t.split('e-')[1], 10)); }
    var dot = t.indexOf('.');
    return dot < 0 ? 0 : Math.min(6, t.length - dot - 1);
  }
  function tidy(wrapper) {
    var input = $(wrapper).find('input.js-range-slider').first();
    var slider = input.length ? input.data('ionRangeSlider') : null;
    if (!slider) { return false; }
    var type = input.attr('data-data-type');
    if (type === 'date' || type === 'datetime') { return true; }
    var current = slider.options.prettify;
    // Shiny puts its own formatter back on every update from the server, so
    // this is checked again on each scan rather than done once
    if (current && current.__heliconTidy) { return true; }
    var enabled = slider.options.prettify_enabled;
    var tidied = function (n) {
      var d = decimals(slider.options.step);
      var v = Number(Number(n).toFixed(d));
      if (Object.is(v, -0)) { v = 0; }
      // Shiny's own formatter (separators, ...) reads its settings from this
      return enabled && current ? current.call(this, v) : String(v);
    };
    tidied.__heliconTidy = true;
    slider.update({prettify_enabled: true, prettify: tidied});
    return true;
  }

  function scan() {
    document.querySelectorAll('.helicon-editable-slider').forEach(function (w) {
      tidy(w);
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


# ionRangeSlider shows and hides its number labels itself, by setting
# ``visibility`` on each: a label it shows (``visibility: visible``) stays on
# screen inside a container hidden with ``visibility: hidden``, which is how a
# page keeps the place of something not ready yet -- the numbers of a hidden
# slider showed alone. A shown label takes its container's visibility instead;
# a label the slider hides itself (the two ends merged into one) stays hidden.
_SLIDER_CSS = """
.helicon-editable-slider .irs [style*="visibility: visible"] {
  visibility: inherit !important;
}
"""


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
        core_ui.tags.style(_SLIDER_CSS),
        class_="helicon-editable-slider",
        **({"data-emit-while-sliding": "true"} if emit_while_sliding else {}),
    )


def clamp_number(value, default, lo, hi, kind=int):
    """A numeric input's value, kept within the bounds the UI gives it.

    The ``min``/``max`` of a numeric input are enforced only by the browser,
    so a server function that lets the value drive its cost reads it through
    this.

    Parameters
    ----------
    value : number, str or None
        The input's value.
    default : number
        Used when the value is missing or not a number.
    lo, hi : number
        The bounds (inclusive).
    kind : type, optional
        ``int`` (default) or ``float``.

    Returns
    -------
    number
        ``kind(value)`` clipped to ``[lo, hi]``.
    """
    try:
        v = kind(value)
    except (TypeError, ValueError, OverflowError):
        v = kind(default)
    if v != v:  # NaN
        v = kind(default)
    return kind(min(max(v, lo), hi))


class ThreadProgress:
    """A progress bar that a worker thread can update.

    ``shiny.ui.Progress`` sends its messages from the event loop's thread
    only. This one is created on the event loop, and its ``set``/``inc``
    hand the update back to the loop, so work running in a thread (see
    :func:`background_task`) can report its progress without blocking the
    other sessions.

    Parameters
    ----------
    min, max : int, optional
        The range of the bar.
    session : shiny.Session, optional
        The session to show it in. Defaults to the current one.
    """

    def __init__(self, min=0, max=1, session=None):
        import asyncio

        from shiny import ui as core_ui

        self._loop = asyncio.get_running_loop()
        self._progress = core_ui.Progress(min=min, max=max, session=session)

    def _call(self, fn, *args, **kwargs):
        self._loop.call_soon_threadsafe(lambda: fn(*args, **kwargs))

    def set(self, value=None, message=None, detail=None):
        """Set the bar, as ``shiny.ui.Progress.set``. Safe from any thread."""
        self._call(self._progress.set, value, message=message, detail=detail)

    def inc(self, amount=0.1, message=None, detail=None):
        """Advance the bar, as ``shiny.ui.Progress.inc``. Safe from any thread."""
        self._call(self._progress.inc, amount, message=message, detail=detail)

    def close(self):
        """Remove the bar. Safe from any thread."""
        self._call(self._progress.close)


def background_task(
    button_id, work, apply, on_error, progress_max=1, session=None, label=""
):
    """Run a long computation in a thread, so other sessions are not blocked.

    Shiny runs every session's reactive code on one event loop: a computation
    that takes minutes inside an effect stalls every visitor for that long.
    The work given here runs in a worker thread instead, with the task button
    showing it is busy, and its result is handed back to the session.

    Call it in a server function. Read the inputs in an effect and pass them
    to ``.invoke(job)`` on the returned task; ``work`` must not read reactive
    values (they are refused in the thread).

    Parameters
    ----------
    button_id : str or None
        The ``input_task_button`` that starts the task; it is disabled while
        the task runs. None for a task started some other way (a dialog's
        button, say).
    work : callable
        ``work(job, progress)``, run in a thread; returns the result.
        ``progress`` is a :class:`ThreadProgress`.
    apply : callable
        ``apply(job, result)``, run in the session when the work succeeds.
    on_error : callable
        ``on_error(job, exception)``, run in the session when the work fails.
    progress_max : int, optional
        The ``max`` of the progress bar.
    session : shiny.Session, optional
        The session. Defaults to the current one.
    label : str, optional
        A name for log messages.

    Returns
    -------
    shiny.reactive.ExtendedTask
        The task; ``.invoke(job)`` starts it.
    """
    import asyncio

    from shiny import ui as core_ui

    if session is None:
        from shiny.session import require_active_session

        session = require_active_session(None)

    async def run(job):
        progress = ThreadProgress(min=0, max=progress_max, session=session)
        try:
            return job, await asyncio.to_thread(work, job, progress)
        except Exception as e:
            # keep the job with the error, for on_error
            e.helicon_job = job
            raise
        finally:
            progress.close()

    task = reactive.extended_task(run)
    if button_id is not None:
        task = core_ui.bind_task_button(button_id=button_id)(task)

    @reactive.effect
    @reactive.event(task.status)
    def _finished():
        status = task.status()
        if status == "success":
            job, result = task.value.get()
            apply(job, result)
        elif status == "error":
            e = task.error.get()
            logger.error("%s failed: %s", label or button_id, e)
            on_error(getattr(e, "helicon_job", None), e)

    return task


def _getter(value, default):
    """A zero-argument callable giving ``value``.

    The gallery parameters take a reactive value (or any callable), a plain
    value, or None for ``default``. A fresh default is made on each call, so
    no reactive value is created at import time or shared between sessions.

    Parameters
    ----------
    value : callable, object or None
        What the caller passed.
    default : object
        What None stands for.

    Returns
    -------
    callable
    """
    if value is None:
        return lambda: default
    if callable(value):
        return value
    return lambda: value


def image_gallery(
    id,
    label=None,
    images=None,
    display_image_labels=True,
    display_dashed_line=False,
    image_labels=None,
    image_links=None,
    image_size=None,
    image_border=2,
    gap=0,
    justification="center",
    enable_selection=False,
    allow_multiple_selection=False,
    initial_selected_indices=None,
    style="",
):
    """Render a gallery of images as a flexbox grid.

    Supports selection, labels, links, and custom styling via Shiny UI.

    Parameters
    ----------
    id : str
        Unique element ID.
    label : reactive.value, callable or value, optional
        Gallery heading label.
    images : reactive.value, callable or value, optional
        List of images (file paths, PIL Images, or 2D numpy arrays).
    display_image_labels : bool, optional
        Whether to show image labels. Defaults to True.
    display_dashed_line : bool, optional
        Whether to show a dashed midline. Defaults to False.
    image_labels : reactive.value, callable or value, optional
        Labels for each image.
    image_links : reactive.value, callable or value, optional
        Links for each image.
    image_size : reactive.value, callable or value, optional
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
    initial_selected_indices : reactive.value, callable or value, optional
        Indices of pre-selected images.
    style : str, optional
        Additional CSS inline styles.

    Returns
    -------
    ui.Tag or tuple
        UI element(s) for the gallery.
    """
    label = _getter(label, "")
    images = _getter(images, [])
    image_labels = _getter(image_labels, [])
    image_links = _getter(image_links, [])
    image_size = _getter(image_size, 128)
    initial_selected_indices = _getter(initial_selected_indices, [])
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
    images=None,
    display_image_labels=True,
    display_dashed_line=False,
    image_labels=None,
    image_links=None,
    image_size=None,
    image_border=2,
    gap=0,
    justification="center",
    enable_selection=True,
    allow_multiple_selection=True,
    initial_selected_indices=None,
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
def google_analytics(id, tab_input="helicon_tab"):
    """The Google tag (gtag.js) for a page, reporting the page without its query.

    A Helicon page's address carries its inputs -- data URLs and, run locally,
    file paths on the server -- so only the page itself and the tab shown
    (``?tab=<name>``) are reported, both on load and on each change of tab.

    Parameters
    ----------
    id : str
        The tag ID, e.g. ``"GT-579RJLLW"`` or ``"G-XXXXXXX"``. Nothing is
        returned for an empty one.
    tab_input : str, optional
        The input holding the tab shown, for a page view on each change.

    Returns
    -------
    htmltools.Tag or None
        The head content to put in the page (core) -- or, in Shiny Express,
        a call whose value is the content shown.
    """
    if not id:
        return None
    import json

    tag = json.dumps(str(id))
    tab = json.dumps(str(tab_input))
    return shiny.ui.head_content(
        shiny.ui.tags.script(
            src=f"https://www.googletagmanager.com/gtag/js?id={id}", async_=True
        ),
        shiny.ui.tags.script(
            shiny.ui.HTML(
                f"""
window.dataLayer = window.dataLayer || [];
function gtag(){{dataLayer.push(arguments);}}
(function () {{
  var ID = {tag}, TAB = {tab};
  function where(t) {{
    if (t === undefined) t = new URLSearchParams(location.search).get('tab');
    return location.origin + location.pathname + (t ? '?tab=' + encodeURIComponent(t) : '');
  }}
  gtag('js', new Date());
  gtag('set', {{page_location: where()}});
  gtag('config', ID, {{page_location: where()}});
  // the tab the page opened on, counted by the config above
  var last = new URLSearchParams(location.search).get('tab') || 'Home';
  if (window.jQuery) jQuery(document).on('shiny:inputchanged', function (e) {{
    if (e.name !== TAB || e.value === last) return;
    last = e.value;
    var loc = where(e.value === 'Home' ? '' : e.value);
    gtag('set', {{page_location: loc}});
    gtag('event', 'page_view', {{page_location: loc, page_title: 'Helicon: ' + e.value}});
  }});
}})();
"""
            )
        ),
    )


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
