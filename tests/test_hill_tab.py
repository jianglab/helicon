"""Regression coverage for HILL's reactive images and Bokeh controls."""

import ast
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from bokeh.models import ColumnDataSource, Slider, Spinner
from bokeh.plotting import figure
from shiny import reactive

from helicon.webApps.tabs import hill_tab


def test_resolution_limits_follow_pixel_size():
    assert hill_tab._resolution_limits_from_apix(1.25) == (3.75, 2.5)
    assert hill_tab._resolution_limits_from_apix(2.3438) == (7.0314, 4.6876)


def test_loaded_pixel_size_updates_resolution_inputs(monkeypatch):
    updates = []
    monkeypatch.setattr(
        hill_tab.ui,
        "update_numeric",
        lambda input_id, **kwargs: updates.append((input_id, kwargs)),
    )

    hill_tab._update_with_apix_from_file(1.25)

    assert updates == [
        ("hill_apix", {"value": 1.25}),
        ("hill_cutoff_res_x", {"value": 3.75, "min": 2.5}),
        ("hill_cutoff_res_y", {"value": 2.5, "min": 2.5}),
    ]


def test_invalid_pixel_size_does_not_update_inputs(monkeypatch):
    updates = []
    monkeypatch.setattr(
        hill_tab.ui,
        "update_numeric",
        lambda input_id, **kwargs: updates.append((input_id, kwargs)),
    )

    hill_tab._update_with_apix_from_file(0)

    assert updates == []


def _controls(prefix="other-hill"):
    sliders = [Slider(start=1, end=100, value=v) for v in (30, 240, 20)]
    state = ColumnDataSource(
        dict(
            twist=[30.0],
            pitch=[240.0],
            rise=[20.0],
            keep=["Twist"],
            tilt=[0.0],
            revision=[0],
        )
    )
    ids = {
        k: f"{prefix}_{k}"
        for k in ("twist", "pitch", "rise", "use_twist_pitch", "helix_revision")
    }
    callbacks = hill_tab._setup_helix_control_js(*sliders, [], ids, state)
    return sliders, state, callbacks


class _Effects:
    """Run production nested effects with real Shiny scheduling, without a browser.

    Compile their original decorators and bodies; substitute only session/UI and
    expensive numerical work. This tests dependency wiring as well as results.
    """

    def __init__(self, mode="PS", **overrides):
        values = dict(
            hill_select_image=[0],
            hill_active_image=[0],
            hill_input_type=mode,
            hill_is_3d=False,
            hill_inhibit_update=False,
            hill_apix=1.0,
            hill_angle=0.0,
            hill_dx=0.0,
            hill_dy=0.0,
            hill_mask_radius=0.0,
            hill_mask_len=90.0,
            hill_negate=False,
            hill_auto_method="threshold",
            hill_cutoff_res_x=3.0,
            hill_cutoff_res_y=2.0,
            hill_pnx=8,
            hill_pny=8,
            hill_log_amp=False,
            hill_ps_scaling="log",
            hill_lp_fraction=0.0,
            hill_hp_fraction=0.0,
            hill_m_max=1,
            hill_diameter=100.0,
            hill_csym=1,
            hill_out_of_plane_tilt=0.0,
            hill_fft_top_only=False,
            hill_ll_colors="lime cyan",
            hill_LL=True,
            hill_LLText=True,
            hill_twist=30.0,
            hill_rise=20.0,
            hill_ms=["0"],
            hill_use_twist_pitch="Twist",
            hill_helix_revision=0,
        )
        values.update(overrides)
        self.input = SimpleNamespace(
            **{k: reactive.Value(v) for k, v in values.items()}
        )
        self.images = [
            np.arange(64.0).reshape(8, 8),
            np.arange(64.0).reshape(8, 8)[::-1].copy(),
        ]
        self.transforms = []
        self.computations = []
        self.updates = []
        self.acf = []

        def transform(data, angle, dx, dy, negate, apix):
            self.transforms.append((angle, dx, dy, negate, apix))
            return (-data if negate else data) + angle + dx + dy

        def spectrum(data, **kwargs):
            self.computations.append(data.copy())
            return data.copy()

        # Use real layerline calculations and figures: their coordinates matter.
        compute = SimpleNamespace(
            transform_2d_image=transform,
            mask_2d_filament=lambda data, *args: data * 0.5,
            update_image_figure=lambda *args, **kwargs: None,
            compute_power_spectra=lambda data, **kw: (spectrum(data), data * 0),
            compute_phase_difference_across_meridian=lambda phase: phase,
            resize_rescale_power_spectra=spectrum,
            compute_layer_line_positions=hill_tab.hill.compute_layer_line_positions,
            bessel_n_image=lambda ny, nx, *args: np.zeros((ny, nx)),
            twist2pitch=hill_tab.hill.twist2pitch,
            auto_upright_transform=hill_tab.hill.auto_upright_transform,
        )
        self.sliders, controls, callbacks = _controls()
        figs = [figure(), figure(), figure()]
        self.ns = dict(vars(hill_tab))

        def output(*args, **kwargs):
            """The module's output registry, as a decorator, bare or with options."""
            if args and callable(args[0]) and not kwargs:
                return args[0]
            return lambda fn: fn

        self.ns.update(
            output=output,
            input=self.input,
            hill=compute,
            ui=SimpleNamespace(
                update_numeric=lambda *a, **kw: self.updates.append((a, kw)),
                update_slider=lambda *a, **kw: self.updates.append((a, kw)),
                update_checkbox=lambda *a, **kw: self.updates.append((a, kw)),
            ),
            selected_images=reactive.Value([]),
            has_images=reactive.Value(False),
            _layerline_frame_key=[None],
            selected_image_labels=reactive.Value([]),
            data_all_2d=reactive.Value(self.images),
            data_all_2d_labels=reactive.Value(["1", "2"]),
            prev_data_from=reactive.Value("main"),
            data_2d_transformed=reactive.Value(None),
            ps_data=reactive.Value(None),
            pd_data=reactive.Value(None),
            phase_data=reactive.Value(None),
            _last_ps_pd_selection=[None],
            _last_transform_key=[None],
            _active_orig=[None],
            csym_controls=ColumnDataSource(data=dict(value=[6], revision=[0])),
            spinner_csym=Spinner(low=1, mode="int", value=6),
            slider_twist=Slider(start=-180, end=180, value=29.4, step=0.01),
            slider_pitch=Slider(
                start=134.2,
                end=536.8,
                value=hill_tab.hill.twist2pitch(29.4, 21.92),
                step=0.5,
            ),
            slider_rise=Slider(start=10.96, end=43.84, value=21.92, step=0.05),
            revision_of_csym=[0],
            _set_acf=lambda img: self.acf.append(img),
            avg_ps_pd_main_in_df=reactive.Value({}),
            source_transformed_img=None,
            ny_curr_img=reactive.Value(8),
            nx_curr_img=reactive.Value(8),
            fig_transformed_img=None,
            fig_ellipses=[],
            figs=figs,
            figs_image=[figs[0], figs[2]],
            data_source_ps=ColumnDataSource(dict(image=[self.images[0]])),
            data_source_pd=ColumnDataSource(dict(image=[self.images[0]])),
            slider_callbacks=callbacks,
            helix_controls=controls,
        )
        self.effects = []
        tree = ast.parse(Path(hill_tab.__file__).read_text(encoding="utf-8"))
        self.server = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "hill_tab_server"
        )
        # the helpers that the effects call
        self.add(
            "_active_position",
            "_current_params",
            "_stash_active_params",
            "_selected_indices",
            "_csym",
            "_model_follows_inputs",
            "_transformed",
            "_ps_scaling",
        )
        self.add("_track_has_images", "_redraw_layerline_figures")

    def add(self, *names):
        for name in names:
            node = next(
                n
                for n in self.server.body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]), hill_tab.__file__, "exec"
                ),
                self.ns,
            )
            if hasattr(self.ns[name], "destroy"):  # an effect, not a helper
                self.effects.append(self.ns[name])

    def get(self, name):
        with reactive.isolate():
            return self.ns[name]()

    def close(self):
        for effect in self.effects:
            effect.destroy()


@pytest.mark.parametrize("mode", ["PS", "PD"])
def test_spectrum_selection_updates_without_control_changes(mode):
    async def run():
        h = _Effects(mode)
        h.add("_update_selected_images", "_apply_2d_transform_ps_pd", "_compute_ps_pd")
        try:
            await reactive.flush()
            np.testing.assert_array_equal(h.get("data_2d_transformed"), h.images[0])
            h.input.hill_select_image.set([1])
            await reactive.flush()
            np.testing.assert_array_equal(h.get("data_2d_transformed"), h.images[1])
            np.testing.assert_array_equal(
                h.get("ps_data" if mode == "PS" else "pd_data"), h.images[1]
            )
            assert len(h.computations) == 2
            # Multiple selection still waits for explicit averaging/use-current.
            h.input.hill_select_image.set([0, 1])
            await reactive.flush()
            assert len(h.computations) == 2
        finally:
            h.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["PS", "PD"])
@pytest.mark.parametrize("inhibit", [False, True])
def test_spectrum_transforms_reset_locally_or_preserve_when_inhibited(mode, inhibit):
    async def run():
        h = _Effects(
            mode,
            hill_inhibit_update=inhibit,
            hill_angle=10.0,
            hill_dx=2.0,
            hill_dy=3.0,
            hill_mask_radius=4.0,
            hill_negate=True,
        )
        h.add("_update_selected_images", "_apply_2d_transform_ps_pd", "_compute_ps_pd")
        try:
            await reactive.flush()
            assert h.transforms[-1] == (
                (10.0, 2.0, 3.0, False, 1.0) if inhibit else (0.0, 0.0, 0.0, False, 1.0)
            )
            # No UI echo is simulated: reset must already apply on the server.
            np.testing.assert_array_equal(
                h.get("data_2d_transformed"), h.images[0] + (15 if inhibit else 0)
            )
            h.input.hill_angle.set(20.0)
            await reactive.flush()
            assert h.transforms[-1] == (20.0, 2.0, 3.0, False, 1.0)
            h.input.hill_select_image.set([1])
            await reactive.flush()
            assert h.transforms[-1] == (
                (20.0, 2.0, 3.0, False, 1.0) if inhibit else (0.0, 0.0, 0.0, False, 1.0)
            )
        finally:
            h.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["PS", "PD"])
def test_mode_switch_recomputes_before_spectrum(mode):
    async def run():
        h = _Effects("Image", hill_negate=True, hill_mask_radius=4.0)
        h.ns["selected_images"].set([h.images[0]])
        h.add("_apply_2d_transform", "_apply_2d_transform_ps_pd", "_compute_ps_pd")
        try:
            await reactive.flush()
            np.testing.assert_array_equal(h.computations[-1], -h.images[0] * 0.5)
            h.input.hill_input_type.set(mode)
            await reactive.flush()
            np.testing.assert_array_equal(h.computations[-1], h.images[0])
            h.input.hill_input_type.set("Image")
            await reactive.flush()
            np.testing.assert_array_equal(h.computations[-1], -h.images[0] * 0.5)
            assert len(h.computations) == 3
        finally:
            h.close()

    asyncio.run(run())


def test_image_transform_cache_distinguishes_source_images():
    async def run():
        h = _Effects("Image")
        h.ns["selected_images"].set([h.images[0]])
        h.add("_apply_2d_transform")
        try:
            await reactive.flush()
            h.ns["selected_images"].set([h.images[1]])
            # A round trip invalidates the effect but leaves numerical settings
            # unchanged, as can happen when dynamic image controls are rebound.
            h.input.hill_angle.set(1.0)
            h.input.hill_angle.set(0.0)
            await reactive.flush()
            np.testing.assert_array_equal(h.get("data_2d_transformed"), h.images[1])
        finally:
            h.close()

    asyncio.run(run())


def test_url_mode_keeps_restored_parameters_until_leaving_emdb():
    async def run():
        h = _Effects(hill_input_mode_params="2", hill_twist=60.0, hill_rise=30.0)
        h.ns["_previous_input_mode"] = [None]
        h.add("_reset_helix_after_emdb", "_sync_helix_controls")
        try:
            await reactive.flush()
            assert not h.updates
            assert h.ns["helix_controls"].data["twist"] == [60.0]
            assert h.ns["helix_controls"].data["pitch"] == [180.0]
            h.input.hill_input_mode_params.set("3")
            await reactive.flush()
            assert not h.updates
            h.input.hill_input_mode_params.set("2")
            await reactive.flush()
            assert [args[0] for args, _ in h.updates] == [
                "hill_twist",
                "hill_rise",
                "hill_csym",
            ]
        finally:
            h.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["Image", "PS", "PD"])
def test_layerlines_follow_selection_and_current_parameters(mode):
    async def run():
        h = _Effects(mode)
        h.add("_sync_helix_controls", "_update_layerline_figures")
        try:
            await reactive.flush()
            assert not h.ns["fig_ellipses"]
            h.ns["selected_images"].set([h.images[0]])
            await reactive.flush()
            assert h.ns["fig_ellipses"]
            h.input.hill_twist.set(40.0)
            h.input.hill_rise.set(25.0)
            await reactive.flush()
            assert h.ns["helix_controls"].data["pitch"] == [225.0]
            for i in range(4):
                old = list(h.ns["fig_ellipses"])
                h.ns["selected_images"].set([h.images[i % 2]])
                h.input.hill_diameter.set(110.0 + i)
                h.input.hill_LLText.set(i % 2 == 0)
                await reactive.flush()
                for renderer in h.ns["fig_ellipses"]:
                    m, orders = renderer.tags
                    np.testing.assert_allclose(
                        renderer.data_source.data["y"],
                        m / 25.0 + np.asarray(orders) / 225.0,
                    )
                for callback in h.ns["slider_callbacks"]:
                    assert callback.args["fig_ellipses"] == h.ns["fig_ellipses"]
                    assert not any(r in old for r in callback.args["fig_ellipses"])
                assert all(
                    len(s.js_property_callbacks["change:value"]) == 1 for s in h.sliders
                )
                assert all(
                    len(s.js_property_callbacks["change:value_throttled"]) == 1
                    for s in h.sliders
                )
            h.ns["selected_images"].set([])
            await reactive.flush()
            assert not h.ns["fig_ellipses"]
            assert all(not c.args["fig_ellipses"] for c in h.ns["slider_callbacks"])
        finally:
            h.close()

    asyncio.run(run())


@pytest.mark.parametrize("delayed", [False, True])
def test_browser_control_callbacks_execute_without_feedback(delayed):
    """Preview locally, then commit once despite nested or delayed callbacks."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the Bokeh JavaScript callbacks")
    sliders, state, callbacks = _controls()
    payload = dict(
        preview=sliders[0].js_property_callbacks["change:value"][0].code,
        commit=sliders[0].js_property_callbacks["change:value_throttled"][0].code,
        restore=state.js_property_callbacks["change:data"][0].code,
        ids=callbacks[0].args["input_ids"],
        delayed=delayed,
    )
    script = r"""
const assert = require('node:assert/strict');
const p = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
const sent = [];
const pending = [];
function drain() {
    let count = 0;
    while (pending.length) {
        assert.ok(++count < 100, 'callback feedback loop');
        pending.shift()();
    }
}
const controls = {data: {twist:[30], pitch:[240], rise:[20],
                         keep:['Twist'], tilt:[0], revision:[0]}};
let keep = 'Twist';
const host = {Shiny:{setInputValue:(id,v)=>sent.push([id,v])},
    document:{getElementsByName:id=>{assert.equal(id,p.ids.use_twist_pitch);return [{checked:true,value:keep}];}}};
const window = {parent:host};
function model(value) {
    return { _value:value, _value_throttled:value, value_cb:null, throttled_cb:null,
        start:1, end:500, step:1, low:null, high:null,
        get value(){return this._value;},
        set value(v){if(this._value !== v){this._value=v; schedule(this.value_cb);}},
        get value_throttled(){return this._value_throttled;},
        set value_throttled(v){if(this._value_throttled !== v){
            this._value_throttled=v; schedule(this.throttled_cb);
        }}
    };
}
function schedule(cb) {
    if (!cb) return;
    if (p.delayed) pending.push(cb); else cb();
}
const slider_twist=model(30), slider_pitch=model(240), slider_rise=model(20);
const el={tags:[1,[0,1]],data_source:{data:{y:[0,0]},change:{emit(){}}}};
const args={slider_twist,slider_pitch,slider_rise,
    controls,window,input_ids:p.ids,fig_ellipses:[el]};
function run(code, cb_obj, extra={}) {
    const a={...args,...extra,cb_obj};
    new Function(...Object.keys(a),code)(...Object.values(a));
}
const sliders = [slider_twist,slider_pitch,slider_rise];
for (let origin=0; origin<sliders.length; origin++) {
    const control = sliders[origin];
    control.value_cb=()=>run(p.preview,control,{origin,control});
    control.throttled_cb=()=>run(p.commit,control,{origin,control});
}
// a number typed on a slider is applied as the end of a drag: value, then
// value_throttled (Bokeh's setv sets both, value first)
function type_in(control, v) {
    control.value=v;
    control.value_throttled=v;
    drain();
}
function release(control) {
    control.value_throttled=control.value;
    drain();
}
function expectBatch(values, revision) {
    assert.deepEqual(sent.splice(0), [
        [p.ids.twist,values[0]], [p.ids.pitch,values[1]],
        [p.ids.rise,values[2]], [p.ids.helix_revision,revision],
    ]);
}

// Rapid pitch previews must settle on the newest value without publishing.
slider_pitch.value=200;
if (p.delayed) pending.shift()(); // queue dependent callbacks from the first value
slider_pitch.value=180;
drain();
assert.equal(slider_pitch.value,180);
assert.equal(slider_twist.value,40);
assert.equal(sent.length,0);
assert.equal(el.data_source.data.y[1],1/20+1/180);
release(slider_pitch);
expectBatch([40,180,20],1);

// Current server acknowledgement is accepted but never echoed.
controls.data={twist:[40],pitch:[180],rise:[20],keep:['Twist'],tilt:[0],revision:[1]};
run(p.restore,controls);
drain();
assert.equal(sent.length,0);

// Twist slider previews continuously and commits on release.
slider_twist.value=45;
drain();
assert.equal(slider_pitch.value,160);
assert.equal(sent.length,0);
release(slider_twist);
expectBatch([45,160,20],2);

// Rise slider in Keep Twist mode derives pitch. An old server response during
// the drag cannot overwrite the active preview.
slider_rise.value=25;
drain();
assert.equal(slider_pitch.value,200);
controls.data={twist:[45],pitch:[160],rise:[20],keep:['Twist'],tilt:[0],revision:[1]};
run(p.restore,controls);
drain();
assert.equal(slider_rise.value,25);
assert.equal(slider_pitch.value,200);
assert.equal(sent.length,0);
release(slider_rise);
expectBatch([45,200,25],3);

// Rise slider in Keep Pitch mode derives twist.
keep='Pitch';
slider_rise.value=30;
drain();
assert.equal(slider_pitch.value,200);
assert.equal(slider_twist.value,54);
assert.equal(sent.length,0);
release(slider_rise);
expectBatch([54,200,30],4);

// Numbers typed on each slider follow the same preview/commit contract.
keep='Twist';
type_in(slider_twist,-60);
assert.equal(slider_pitch.value,180);
expectBatch([-60,180,30],5);

type_in(slider_pitch,150);
assert.equal(slider_twist.value,-72);
expectBatch([-72,150,30],6);

type_in(slider_rise,35);
assert.equal(slider_twist.value,-72);
assert.equal(slider_pitch.value,175);
expectBatch([-72,175,35],7);

// A number beyond the slider's range moves the range with it.
type_in(slider_pitch,900);
assert.ok(slider_pitch.start <= 900 && 900 <= slider_pitch.end);
expectBatch([-72*175/900,900,35],8);

// Invalid edits restore the last valid preview and are not committed.
type_in(slider_pitch,0);
assert.equal(slider_pitch.value,900);
assert.equal(sent.length,0);
"""
    result = subprocess.run(
        [node, "-e", script], input=json.dumps(payload), text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


class TestEditableSliders:
    """Twist, pitch and rise are one slider each, their numbers typed in place."""

    def test_each_slider_names_its_model_for_the_page_script(self):
        s = Slider(start=1, end=10, value=5)
        classes = hill_tab._editable_slider_classes(s)
        assert hill_tab._EDITABLE_SLIDER in classes
        assert f"{hill_tab._MODEL_CLASS}{s.id}" in classes
        js = hill_tab._BOKEH_SLIDER_EDIT_JS
        assert hill_tab._EDITABLE_SLIDER in js and hill_tab._MODEL_CLASS in js
        # applied as the end of a drag, so the slider's own callbacks commit it
        assert "slider.setv({value: v, value_throttled: v})" in js

    def test_no_separate_number_boxes(self):
        src = Path(hill_tab.__file__).read_text(encoding="utf-8")
        # none for the helix; the one number box is the csym, an integer that no
        # slider shows
        assert src.count("Spinner(") == 1 and "spinner_csym = Spinner(" in src
        for name in ("twist", "pitch", "rise"):
            assert f"spinner_{name}" not in src
        assert "ui.tags.script(_BOKEH_SLIDER_EDIT_JS)" in src
        assert "slider.css_classes = _editable_slider_classes(slider)" in src


def _saved(angle, dx=0.0, dy=0.0, mask_radius=0.0, negate=False):
    return {
        "apix": 1.0,
        "negate": negate,
        "angle": angle,
        "dx": dx,
        "dy": dy,
        "mask_radius": mask_radius,
        "mask_len": 90.0,
    }


class TestActiveImageGallery:
    """The transform controls act on the image clicked in the gallery of the
    selected images (there is no "Transformation for" list)."""

    def _harness(self):
        h = _Effects("Image", hill_select_image=[0, 1], hill_active_image=[0])
        h.ns["selected_images"].set(h.images)
        h.ns["selected_image_labels"].set(["1", "2"])
        h.ns["_active_orig"][0] = 0  # what the selection effect leaves
        h.ns["avg_ps_pd_main_in_df"].set({0: _saved(10.0), 1: _saved(-20.0, dx=5.0)})
        self.shown = []
        h.ns["hill"].update_image_figure = lambda *a, **kw: self.shown.append(a[2])
        h.add("_update_current_image_from_multiselect", "_show_transformed")
        return h

    def _saved_of(self, h, i):
        with reactive.isolate():
            return h.ns["avg_ps_pd_main_in_df"]()[i]

    def _loaded(self, h):
        return {a[0]: kw["value"] for a, kw in h.updates}

    def test_clicking_an_image_loads_its_own_parameters(self):
        async def run():
            h = self._harness()
            try:
                await reactive.flush()
                assert h.updates == []  # the first image is already showing
                h.input.hill_active_image.set([1])
                await reactive.flush()
                loaded = self._loaded(h)
                assert loaded["hill_angle"] == -20.0
                assert loaded["hill_dx"] == 5.0
                assert h.ns["_active_orig"][0] == 1
                # the transformed image is shown at once, not when the controls
                # echo their new values (they do not when the values are equal)
                np.testing.assert_array_equal(self.shown[-1], h.images[1] - 15.0)
            finally:
                h.close()

        asyncio.run(run())

    def test_what_was_set_for_an_image_is_kept_when_leaving_it(self):
        async def run():
            h = self._harness()
            try:
                await reactive.flush()
                h.input.hill_angle.set(33.0)  # the user edits the first image
                h.input.hill_active_image.set([1])
                await reactive.flush()
                assert self._saved_of(h, 0)["angle"] == 33.0
                assert self._saved_of(h, 1)["angle"] == -20.0  # untouched
                # the controls echo the second image's values, then come back
                h.input.hill_angle.set(-20.0)
                h.input.hill_dx.set(5.0)
                h.updates.clear()
                h.input.hill_active_image.set([0])
                await reactive.flush()
                assert self._loaded(h)["hill_angle"] == 33.0
                assert self._saved_of(h, 1)["dx"] == 5.0
            finally:
                h.close()

        asyncio.run(run())

    def test_the_auto_correlation_follows_the_clicked_image(self):
        async def run():
            h = self._harness()
            h.add("_update_transformed_from_multiselect")
            try:
                await reactive.flush()
                h.acf.clear()  # what the first image drew when it was selected
                h.input.hill_active_image.set([1])
                await reactive.flush()
                np.testing.assert_array_equal(h.acf[-1], h.images[1] - 15.0)
                # ... and when its controls are then changed
                h.input.hill_angle.set(3.0)
                await reactive.flush()
                np.testing.assert_array_equal(h.acf[-1], h.images[1] + 3.0)
            finally:
                h.close()

        asyncio.run(run())

    def test_no_auto_correlation_for_spectra(self):
        async def run():
            h = _Effects("PS", hill_select_image=[0, 1], hill_active_image=[0])
            h.ns["selected_images"].set(h.images)
            h.ns["hill"].update_image_figure = lambda *a, **kw: None
            h.add("_show_transformed")
            with reactive.isolate():
                h.ns["_show_transformed"](h.images[0], _saved(0.0))
            assert h.acf == []

        asyncio.run(run())

    def test_the_radial_profile_follows_the_clicked_image(self):
        async def run():
            h = self._harness()
            line = lambda: SimpleNamespace(data_source=SimpleNamespace(data=None))
            lines = {
                k: line()
                for k in (
                    "fig_radial_profile_line_max",
                    "fig_radial_profile_line_max_flipped",
                    "fig_radial_profile_line_mean",
                    "fig_radial_profile_line_mean_flipped",
                )
            }
            h.ns.update(
                lines,
                _rp_rmin_span=SimpleNamespace(location=0),
                _rp_rmax_span=SimpleNamespace(location=0),
            )
            h.add("_update_radial_profile_data")
            try:
                await reactive.flush()
                top = lines["fig_radial_profile_line_max"].data_source
                np.testing.assert_array_equal(top.data["y"], h.images[0].max(axis=0))
                h.input.hill_active_image.set([1])
                await reactive.flush()
                np.testing.assert_array_equal(top.data["y"], h.images[1].max(axis=0))
            finally:
                h.close()

        asyncio.run(run())

    def test_clicking_the_image_already_showing_changes_nothing(self):
        async def run():
            h = self._harness()
            try:
                await reactive.flush()
                h.input.hill_angle.set(33.0)
                h.input.hill_active_image.set([0])
                await reactive.flush()
                assert h.updates == []
                assert self._saved_of(h, 0)["angle"] == 10.0  # not stashed yet
            finally:
                h.close()

        asyncio.run(run())

    def test_a_click_past_the_selection_means_the_first_image(self):
        async def run():
            h = self._harness()
            h.add("_active_position")
            try:
                await reactive.flush()
                h.input.hill_active_image.set([7])  # left over from a longer one
                with reactive.isolate():
                    assert h.ns["_active_position"]() == 0
                h.input.hill_active_image.set([])  # the click deselected it
                with reactive.isolate():
                    assert h.ns["_active_position"]() == 0
            finally:
                h.close()

        asyncio.run(run())

    def test_averaging_uses_the_edits_of_the_image_being_worked_on(self):
        async def run():
            h = self._harness()
            h.input.hill_run_get_average_power_spectra = reactive.Value(0)
            h.ns["avg_ps_pd_df"] = reactive.Value(None)
            h.add("_average_ps_pd_from_multiselect")
            try:
                await reactive.flush()
                h.input.hill_angle.set(33.0)
                h.input.hill_run_get_average_power_spectra.set(1)
                await reactive.flush()
                assert self._saved_of(h, 0)["angle"] == 33.0
                assert (33.0, 0.0, 0.0, False, 1.0) in h.transforms
            finally:
                h.close()

        asyncio.run(run())

    def test_the_symmetry_search_uses_the_edits_of_the_image_being_worked_on(self):
        async def run():
            h = self._harness()
            h.input.hill_suggest_symmetry = reactive.Value(None)
            h.input.hill_diameter = reactive.Value(100.0)
            search = lambda *a, **kw: None
            search.status = lambda: "idle"
            h.ns.update(
                symmetry_message=reactive.Value(""),
                symmetry_candidates=reactive.Value([]),
                _symmetry_search=search,
            )
            records = []

            def images_for_indexing(images, recs, apix, method):
                records.extend(recs)
                return images

            h.ns["hill"].images_for_indexing = images_for_indexing
            h.add("_start_symmetry_search")
            try:
                await reactive.flush()
                h.input.hill_angle.set(33.0)  # edited, the image not yet left
                h.input.hill_suggest_symmetry.set(1)
                await reactive.flush()
                assert [r["angle"] for r in records] == [33.0, -20.0]
            finally:
                h.close()

        asyncio.run(run())

    def test_edits_are_not_kept_for_spectra(self):
        async def run():
            h = _Effects("PS", hill_select_image=[0, 1])
            h.ns["_active_orig"][0] = 0
            h.input.hill_angle.set(33.0)
            with reactive.isolate():
                h.ns["_stash_active_params"]()
                assert h.ns["avg_ps_pd_main_in_df"]() == {}

        asyncio.run(run())


class TestSelectedImagesUI:
    def test_the_gallery_replaces_the_picture_and_the_dropdown(self):
        html = str(hill_tab.hill_tab_ui("hill"))
        assert "hill_active_image_gallery" in html
        assert "hill_display_selected_image" not in html
        assert "hill_curr_img_in_selected_list" not in html
        assert "Transformation for" not in html


class TestImageParametersTitle:
    def _title(self, labels, active):
        h = _Effects("Image", hill_active_image=active)
        h.ns["selected_images"].set([h.images[0]] * len(labels))
        h.ns["selected_image_labels"].set(labels)
        h.add("hill_image_params_title")

        async def run():
            with reactive.isolate():
                return await h.ns["hill_image_params_title"].fn()

        return asyncio.run(run())

    def test_it_names_the_image_the_controls_act_on(self):
        assert self._title(["3", "7", "9"], [1]) == "Image Parameters (image 7)"

    def test_it_is_plain_without_images(self):
        assert self._title([], [0]) == "Image Parameters"

    def test_a_named_image_is_not_numbered(self):
        assert self._title(["Projection"], [0]) == "Image Parameters (Projection)"


class TestAutoTransformButton:
    def _run(self, mode="Image", active=1, estimate=None):
        async def run():
            h = _Effects(mode, hill_active_image=[active], hill_auto_transform=None)
            h.ns["selected_images"].set(h.images)
            measured = []

            def auto(image, apix, method):
                measured.append((image, apix, method))
                if estimate is not None:
                    raise estimate
                return {"angle": 12.5, "dx": -3.0, "dy": 0.0, "mask_radius": 40.0}

            h.ns["hill"].auto_upright_transform = auto
            shown = []
            h.ns["ui"].notification_show = lambda *a, **kw: shown.append(a[0])
            h.add("_auto_transform_current_image")
            try:
                await reactive.flush()
                assert not measured  # nothing happens until it is clicked
                h.input.hill_auto_transform.set(1)
                await reactive.flush()
                return h, measured, shown
            finally:
                h.close()

        return asyncio.run(run())

    def test_the_image_being_worked_on_is_measured_and_the_controls_set(self):
        h, measured, _ = self._run(active=1)
        assert len(measured) == 1
        np.testing.assert_array_equal(measured[0][0], h.images[1])
        assert measured[0][1:] == (1.0, "threshold")
        assert {a[0]: kw["value"] for a, kw in h.updates} == {
            "hill_angle": 12.5,
            "hill_dx": -3.0,
            "hill_dy": 0.0,
            "hill_mask_radius": 40.0,
            "hill_mask_len": 90.0,
        }

    def test_a_failed_estimate_leaves_the_controls_and_says_so(self):
        h, measured, shown = self._run(estimate=ValueError("no filament"))
        assert h.updates == []
        assert shown and "no filament" in shown[0]

    def test_there_is_nothing_to_estimate_for_a_spectrum(self):
        h, measured, _ = self._run(mode="PS")
        assert not measured and h.updates == []

    def test_the_row_holds_input_is_then_the_contrast_then_the_button(self):
        html = str(hill_tab.hill_tab_ui("hill"))
        row = html[html.index('class="hill-img-row"') :]
        row = row[: row.index("hill_img_2d_uis")]
        # the radio is drawn here; the other two by the output beside it
        assert row.index("hill_input_type") < row.index("hill_img_flags_ui")

        h = _Effects("Image")
        h.ns["ui"] = hill_tab.ui  # the real builders, not the update stubs
        h.add("hill_img_flags_ui")

        async def run():
            with reactive.isolate():
                return str(await h.ns["hill_img_flags_ui"].fn())

        flags = asyncio.run(run())
        assert flags.index("hill_negate") < flags.index("hill_auto_transform")

    def test_a_spectrum_has_neither(self):
        h = _Effects("PS")
        h.add("hill_img_flags_ui")

        async def run():
            with reactive.isolate():
                return await h.ns["hill_img_flags_ui"].fn()

        assert asyncio.run(run()) is None


class TestLayerlineRedrawIsOnlyAsBigAsTheChange:
    """Redrawing the layer lines is slow (Bokeh serialises every change to the
    browser), so a change redraws what it must and no more."""

    def _harness(self):
        h = _Effects("Image")
        h.ns["selected_images"].set([h.images[0]])
        h.add("_update_layerline_figures")
        return h

    def test_a_new_list_of_images_does_not_redraw(self):
        async def run():
            h = self._harness()
            try:
                await reactive.flush()
                drawn = list(h.ns["fig_ellipses"])
                assert drawn
                h.ns["selected_images"].set([h.images[1]])  # another image
                h.ns["selected_images"].set([h.images[0], h.images[1]])
                await reactive.flush()
                assert h.ns["fig_ellipses"] == drawn  # the very same renderers
            finally:
                h.close()

        asyncio.run(run())

    def test_twist_and_rise_do_not_resend_the_bessel_image(self):
        async def run():
            h = self._harness()
            try:
                await reactive.flush()
                bessel = h.ns["data_source_ps"].data["bessel"][0]
                h.input.hill_twist.set(40.0)
                await reactive.flush()
                h.input.hill_rise.set(25.0)
                await reactive.flush()
                assert h.ns["data_source_ps"].data["bessel"][0] is bessel
                assert h.ns["data_source_pd"].data["bessel"][0] is bessel
            finally:
                h.close()

        asyncio.run(run())

    @pytest.mark.parametrize(
        "name,value",
        [
            ("hill_diameter", 130.0),
            ("hill_out_of_plane_tilt", 5.0),
            ("hill_cutoff_res_x", 4.0),
            ("hill_cutoff_res_y", 3.0),
            ("hill_fft_top_only", True),
        ],
    )
    def test_what_the_bessel_image_depends_on_does_resend_it(self, name, value):
        async def run():
            h = self._harness()
            try:
                await reactive.flush()
                bessel = h.ns["data_source_ps"].data["bessel"][0]
                getattr(h.input, name).set(value)
                await reactive.flush()
                assert h.ns["data_source_ps"].data["bessel"][0] is not bessel
            finally:
                h.close()

        asyncio.run(run())


def assert_binary_encodable(*arrays):
    """Bokeh sends float32/float64 and integer arrays as a buffer; any other
    dtype (float16 was used) is encoded a number at a time, in Python, which for
    a 1024x512 image takes about a second."""
    from bokeh.util.serialization import array_encoding_disabled

    for array in arrays:
        assert not array_encoding_disabled(array), array.dtype


def test_the_figures_of_the_layer_lines_hold_only_fast_to_send_arrays():
    from helicon.webApps.lib import hill_compute as hill

    n = 16
    fig, source = hill.create_layerline_image_figure(
        data=np.random.rand(n, n),
        cutoff_res_x=7.0,
        cutoff_res_y=4.7,
        helical_radius=77.6,
        tilt=0.0,
        phase=np.random.rand(n, n),
        fft_top_only=False,
        pseudo_color=False,
        const_image_color="",
        title="PS",
        yaxis_visible=True,
        tooltips=[("Amp", "@image")],
    )
    for name in ("image", "phase", "bessel"):
        assert_binary_encodable(*source.data[name])


class TestFigureUpdatesSendOnlyWhatChanged:
    """A power spectrum or phase difference is about 4 MB; the Bessel image and
    the coordinates beside it must not travel with it every time."""

    def _watch(self, source):
        from bokeh.document import Document

        doc = Document()
        doc.add_root(source)
        events = []
        doc.on_change(events.append)
        return events

    def _sent(self, events):
        # a ColumnDataChangedEvent names the columns it carries; an assigned
        # `data` dict has no list and sends them all
        assert events and all(e.cols is not None for e in events)
        return sorted({c for e in events for c in e.cols})

    def _prepared(self, h, source_name):
        source = h.ns[source_name]
        source.data.update(
            bessel=[np.zeros((8, 8), dtype=np.int16)],
            x=[0.0],
            phase=[np.zeros((8, 8), dtype=np.float16)],
        )
        return source, self._watch(source)

    def test_the_phase_difference_sends_its_image_and_phase_only(self):
        async def run():
            h = _Effects("Image")
            source, events = self._prepared(h, "data_source_pd")
            h.add("_update_figure_pd")
            try:
                h.ns["phase_data"].set(np.ones((8, 8)))
                h.ns["pd_data"].set(np.ones((8, 8)))
                await reactive.flush()
                assert self._sent(events) == ["image", "phase"]
                assert source.data["image"][0].dtype == np.float32
                assert_binary_encodable(
                    source.data["image"][0], source.data["phase"][0]
                )
            finally:
                h.close()

        asyncio.run(run())

    def test_the_power_spectrum_sends_its_image_and_phase_only(self):
        async def run():
            h = _Effects("Image")
            source, events = self._prepared(h, "data_source_ps")
            empty = lambda *cols: ColumnDataSource({c: [] for c in cols})
            h.ns.update(
                data_source_yp=empty("yprofile", "y", "resy"),
                data_source_yp_peaks=empty("x", "y", "label"),
                data_source_yp_fits=empty("x", "y"),
                fig_yp=figure(),
            )
            h.input.hill_yp_peak_detect = reactive.Value(False)
            h.input.hill_yp_peak_prominence = reactive.Value(0.1)
            h.input.hill_yp_peak_fit_hw = reactive.Value(2)
            h.add("_update_figure_pwr_yp")
            try:
                h.ns["phase_data"].set(np.ones((8, 8)))
                h.ns["ps_data"].set(np.ones((8, 8)))
                await reactive.flush()
                assert self._sent(events) == ["image", "phase"]
                assert source.data["image"][0].dtype == np.float32
                assert_binary_encodable(
                    source.data["image"][0], source.data["phase"][0]
                )
            finally:
                h.close()

        asyncio.run(run())


class TestHeldDocumentChanges:
    def _renderers_added(self, hold):
        from bokeh.document import Document

        fig = figure()
        doc = Document()
        doc.add_root(fig)
        events = []
        doc.on_change(events.append)
        with hill_tab._held(doc if hold else None):
            for i in range(20):
                fig.circle([i], [i], radius=1)
        return events, fig

    def test_twenty_renderers_are_one_message_when_held(self):
        held, fig = self._renderers_added(hold=True)
        loose, _ = self._renderers_added(hold=False)
        assert len(loose) >= 20
        assert len(held) <= 3
        assert len(fig.renderers) == 20  # and nothing is lost

    def test_it_lets_go_when_the_body_fails(self):
        from bokeh.document import Document

        fig = figure()
        doc = Document()
        doc.add_root(fig)
        events = []
        doc.on_change(events.append)
        with pytest.raises(RuntimeError):
            with hill_tab._held(doc):
                raise RuntimeError("redraw failed")
        fig.title.text = "later"  # not held back by the failed block
        assert len(events) == 1

    def test_figures_not_on_a_page_yet_need_no_document(self):
        with hill_tab._held(None):
            pass

    def test_without_the_internal_freeze_the_changes_are_still_combined(self):
        from types import SimpleNamespace

        from bokeh.document import Document

        fig = figure()
        doc = Document()
        doc.add_root(fig)
        events = []
        doc.on_change(events.append)

        class NoFreeze:
            """A Document whose ``models`` has no ``freeze`` (another Bokeh)."""

            models = SimpleNamespace()

            def __getattr__(self, name):
                return getattr(doc, name)

        with hill_tab._held(NoFreeze()):
            for i in range(20):
                fig.circle([i], [i], radius=1)
        assert len(fig.renderers) == 20 and len(events) <= 3


class TestParametersTabIsGroupedByFunction:
    """One panel per function, and every input of the tab in exactly one."""

    GROUPS = {
        "hill_params_symmetry": [
            "hill_diameter",
            "hill_out_of_plane_tilt",
            "hill_use_twist_pitch",
        ],
        "hill_params_alignment": ["hill_auto_method", "hill_inhibit_update"],
        "hill_params_fft": [
            "hill_cutoff_res_x",
            "hill_cutoff_res_y",
            "hill_pnx",
            "hill_pny",
            "hill_ps_scaling",
            "hill_hp_fraction",
            "hill_lp_fraction",
            "hill_log_amp",
        ],
        "hill_params_yprofile": [
            "hill_yp_peak_detect",
            "hill_yp_peak_prominence",
            "hill_yp_peak_fit_hw",
        ],
        "hill_params_layerlines": [
            "hill_LL",
            "hill_LLText",
            "hill_m_max",
            "hill_ms",
            "hill_ll_colors",
        ],
        "hill_params_display": [
            "hill_PS",
            "hill_YP",
            "hill_PD",
            "hill_Phase",
            "hill_Color",
            "hill_fft_top_only",
            "hill_const_image_color",
        ],
    }

    def _panels(self):
        import re

        html = str(hill_tab.hill_tab_ui("hill"))
        start = html.index('id="hill-hill_params_accordion"')
        # the hidden helix inputs follow the accordion: not part of its panels
        body = html[start : html.index('id="hill-hill_twist"')]
        # each panel runs to the next one's marker
        marks = [
            (m.start(), m.group(1))
            for m in re.finditer(r'data-value="(hill_params_\w+)"', body)
        ]
        panels = {}
        for (a, name), (b, _) in zip(marks, marks[1:] + [(len(body), None)]):
            panels[name] = re.findall(r'id="hill-(hill_\w+)"', body[a:b])
        return html, panels

    def test_each_function_has_its_own_panel_in_order(self):
        _, panels = self._panels()
        assert list(panels) == list(self.GROUPS)

    def test_every_input_is_in_its_functions_panel(self):
        _, panels = self._panels()
        for name, ids in self.GROUPS.items():
            for input_id in ids:
                assert input_id in panels[name], (input_id, name)

    def test_no_input_is_in_two_panels(self):
        _, panels = self._panels()
        seen = [i for ids in panels.values() for i in ids]
        assert len(seen) == len(set(seen))

    def test_the_hidden_helix_inputs_follow_the_panels(self):
        html, panels = self._panels()
        last_panel = html.index('data-value="hill_params_display"')
        for hidden in (
            "hill_twist",
            "hill_pitch",
            "hill_rise",
            "hill_helix_revision",
            "hill_csym",
        ):
            assert html.index(f'id="hill-{hidden}"') > last_panel


class TestMOrdersInOneRow:
    def test_the_m_checkboxes_are_drawn_inline(self):
        html = str(hill_tab.hill_tab_ui("hill"))
        group = html[html.index('id="hill-hill_ms"') :]
        assert "shiny-input-container-inline" in group[:400]

    def test_changing_max_m_keeps_them_inline(self):
        async def run():
            h = _Effects("Image", hill_ms=["-1", "0", "1"])
            calls = []
            h.ns["ui"].update_checkbox_group = lambda *a, **kw: calls.append((a, kw))
            h.add("_update_m_choices")
            try:
                await reactive.flush()  # ignore_init: nothing yet
                h.input.hill_m_max.set(2)
                await reactive.flush()
                ((args, kwargs),) = calls
                assert kwargs["choices"] == ["2", "1", "0", "-1", "-2"]
                assert kwargs["inline"] is True
            finally:
                h.close()

        asyncio.run(run())


class TestSpectrumScalingChoice:
    def test_it_defaults_to_the_log_and_the_fourier_filter(self):
        from helicon.webApps.tabs.hill_bookmark import BOOKMARK_DEFAULTS

        assert BOOKMARK_DEFAULTS["ps_scaling"] == ("hill_ps_scaling", "log")
        html = str(hill_tab.hill_tab_ui("hill"))
        radio = html[html.index('id="hill-hill_ps_scaling"') :][:900]
        assert 'value="log" checked="checked"' in radio
        assert 'value="shell" checked' not in radio

    def test_the_two_options_are_in_one_row(self):
        html = str(hill_tab.hill_tab_ui("hill"))
        radio = html[html.index('id="hill-hill_ps_scaling"') :][:900]
        assert radio.count('class="radio-inline"') == 2

    def test_the_filter_and_log_controls_show_only_for_the_log_mode(self):
        html = str(hill_tab.hill_tab_ui("hill"))
        gate = html.index("hill_ps_scaling == ")
        for shown_only_for_log in (
            "hill_hp_fraction",
            "hill_lp_fraction",
            "hill_log_amp",
        ):
            assert html.index(f'id="hill-{shown_only_for_log}"') > gate
        # what both modes use stays outside the condition
        assert html.index('id="hill-hill_pnx"') < gate

    @pytest.mark.parametrize("choice,shell", [("log", False), ("shell", True)])
    def test_the_keywords_for_the_spectrum_functions(self, choice, shell):
        h = _Effects("Image", hill_ps_scaling=choice, hill_hp_fraction=0.4)
        h.input.hill_lp_fraction = reactive.Value(2.0)
        with reactive.isolate():
            kw = h.ns["_ps_scaling"]()
        assert kw == dict(
            shell_median=shell,
            log=False,
            low_pass_fraction=0.02,
            high_pass_fraction=0.004,
        )


class TestSuggestSymmetryIsOnTheInputsTab:
    """The search is a main step of the workflow, so it sits on the first tab,
    after the images and above the credits, not in the parameters."""

    def _html(self):
        return str(hill_tab.hill_tab_ui("hill"))

    def _panes(self, html):
        # each tab's value is on its link in the nav bar, then on its pane
        return (
            html.rindex('data-value="hill_input_sidebar"'),
            html.rindex('data-value="hill_params_sidebar"'),
        )

    def test_it_is_above_the_credits_on_the_inputs_tab(self):
        html = self._html()
        inputs_pane, params_pane = self._panes(html)
        button = html.index('id="hill-hill_suggest_symmetry"')
        credits = html.index("Developed by")
        assert inputs_pane < button < credits < params_pane

    def test_its_message_and_results_follow_the_button(self):
        html = self._html()
        button = html.index('id="hill-hill_suggest_symmetry"')
        message = html.index('id="hill-hill_symmetry_message"')
        table = html.index('id="hill-hill_symmetry_candidates"')
        assert button < message < table < html.index("Developed by")

    def test_the_parameters_tab_no_longer_holds_it(self):
        html = self._html()
        _, params_pane = self._panes(html)
        assert html.count('id="hill-hill_suggest_symmetry"') == 1
        assert html.find('id="hill-hill_suggest_symmetry"', params_pane) == -1

    def test_the_button_is_centred(self):
        html = self._html()
        wrapper = html.rindex(
            'class="hill-suggest-button"',
            0,
            html.index('id="hill-hill_suggest_symmetry"'),
        )
        assert "text-align: center" in html[wrapper : wrapper + 200]


class TestSymmetrySearchProgressBar:
    def _html(self, done, total):
        return str(hill_tab._progress_bar(done, total))

    def test_it_shows_the_share_done(self):
        html = self._html(35, 70)
        assert "width: 50%" in html and ">50%<" in html
        assert 'aria-valuenow="50"' in html and 'role="progressbar"' in html
        assert "progress-bar-animated" not in html

    def test_it_runs_from_empty_to_full(self):
        assert "width: 0%" in self._html(0, 70)
        assert "width: 100%" in self._html(70, 70)

    def test_it_never_leaves_the_bar(self):
        assert "width: 100%" in self._html(75, 70)
        assert "width: 0%" in self._html(-3, 70)

    def test_it_is_animated_while_the_number_of_steps_is_not_known(self):
        # the first report comes a few seconds in, while the search times a
        # sample to decide how many processes to use
        html = self._html(0, 0)
        assert "progress-bar-striped" in html and "progress-bar-animated" in html
        assert "width: 100%" in html

    def test_the_bar_sits_between_the_message_and_the_results(self):
        html = str(hill_tab.hill_tab_ui("hill"))
        message = html.index('id="hill-hill_symmetry_message"')
        bar = html.index('id="hill-hill_symmetry_progress"')
        table = html.index('id="hill-hill_symmetry_candidates"')
        assert message < bar < table

    def test_the_search_passes_a_progress_callback_to_the_index(self):
        # the source: the task cannot be run without a session
        source = Path(hill_tab.__file__).read_text(encoding="utf-8")
        task = source[source.index("async def _symmetry_search") :]
        task = task[: task.index("def _poll_symmetry_progress")]
        assert "progress=progress" in task
        assert "_symmetry_steps.update(done=done, total=total)" in task

    def test_the_fill_is_no_taller_than_the_track_that_clips_it(self):
        # the fill's line was 20 px in a track of 11: the bottom was cut off
        html = self._html(35, 70)
        assert 'style="height: 1.3rem;' in html
        assert "height: 100%" in html  # the ipywidgets css makes it 28 px otherwise
        assert "line-height: 1.3rem" in html


class TestImageParametersStartsCollapsed:
    def _item(self):
        html = str(hill_tab.hill_tab_ui("hill"))
        start = html.rindex('data-value="hill_image_params"')
        # the item's button, which says whether it is collapsed
        return html[start : start + 900]

    def test_the_panel_is_collapsed(self):
        item = self._item()
        assert "accordion-button collapsed" in item
        assert 'aria-expanded="false"' in item
        assert "accordion-collapse collapse show" not in item

    def test_open_is_not_left_on_the_panel(self):
        # it is a parameter of the accordion; on a panel it is a stray attribute
        assert 'open="False"' not in self._item()

    def test_what_the_pipeline_needs_is_drawn_although_the_panel_is_collapsed(self):
        # Shiny does not run an output that is hidden: with the panel collapsed
        # hill_apix() had no value, and the transform, the spectra and the
        # symmetry search silently did nothing
        source = Path(hill_tab.__file__).read_text(encoding="utf-8")
        for name in ("hill_img_2d_uis", "hill_img_flags_ui"):
            head = source[: source.index(f"def {name}(")]
            decorators = head[head.rindex("\n\n") :]
            assert "@output(suspend_when_hidden=False)" in decorators, name


class TestCsymBesideTheSliders:
    """csym is part of indexing, so it is set in a narrow box at the right of the
    rise slider; the input in the Parameters tab is hidden, kept for what reads it."""

    def test_the_spinner_is_the_first_of_the_slider_row(self):
        src = Path(hill_tab.__file__).read_text(encoding="utf-8")
        row = src[src.index("            row(\n                spinner_csym,") :][:300]
        assert (
            row.index("spinner_csym")
            < row.index("slider_twist")
            < row.index("slider_pitch")
            < row.index("slider_rise")
        )

    def test_its_label_is_capitalised_like_the_sliders(self):
        src = Path(hill_tab.__file__).read_text(encoding="utf-8")
        assert 'title="Csym"' in src[src.index("spinner_csym = Spinner(") :][:120]

    def test_it_is_narrow_whole_and_starts_at_the_bookmark_default(self):
        src = Path(hill_tab.__file__).read_text(encoding="utf-8")
        box = src[src.index("spinner_csym = Spinner(") :][:260]
        assert 'mode="int"' in box and "low=1" in box and "width=64" in box
        assert 'BOOKMARK_DEFAULTS["csym"][1]' in box

    def test_the_input_in_the_parameters_tab_is_hidden(self):
        html = str(hill_tab.hill_tab_ui("hill"))
        assert html.count('id="hill-hill_csym"') == 1
        # in the hidden div that holds the other hidden helix inputs
        wrapper = html.rindex('<div hidden="">', 0, html.index('id="hill-hill_csym"'))
        assert wrapper < html.index('id="hill-hill_twist"')
        assert 'id="hill-hill_helix_revision"' in html[wrapper:]
        assert "accordion" not in html[wrapper : html.index('id="hill-hill_csym"')]

    def test_a_csym_set_elsewhere_reaches_the_box_every_time(self):
        async def run():
            h = _Effects("Image", hill_csym=6)
            h.add("_sync_csym_spinner")
            try:
                await reactive.flush()
                first = h.ns["csym_controls"].data["revision"][0]
                h.input.hill_csym.set(4)
                await reactive.flush()
                data = h.ns["csym_controls"].data
                assert data["value"] == [4] and data["revision"][0] > first
                # a box still being drawn starts from the model's own value
                assert h.ns["spinner_csym"].value == 4
                # the same value again (the box was edited meanwhile, and a
                # candidate row says 4 again) must still be a change
                h.input.hill_csym.set(5)
                await reactive.flush()
                h.input.hill_csym.set(4)
                await reactive.flush()
                assert h.ns["csym_controls"].data["value"] == [4]
                assert (
                    h.ns["csym_controls"].data["revision"][0] > data["revision"][0] + 1
                )
            finally:
                h.close()

        asyncio.run(run())

    def test_a_csym_below_1_is_made_1(self):
        async def run():
            h = _Effects("Image", hill_csym=0)
            updates = []
            h.ns["ui"].update_numeric = lambda *a, **kw: updates.append((a, kw))
            h.add("_sync_csym_spinner")
            try:
                await reactive.flush()
                assert updates == [(("hill_csym",), {"value": 1})]
                assert h.ns["csym_controls"].data["value"] == [1]
            finally:
                h.close()

        asyncio.run(run())


@pytest.mark.parametrize(
    "typed,sent,shown",
    [
        (4, [4], 4),
        (12, [12], 12),
        (1, [1], 1),
        (0, [], 6),  # not a csym: the box goes back to the last one, 6
        (-3, [], 6),
        (2.5, [], 6),  # half-typed
        (float("nan"), [], 6),  # emptied
    ],
)
def test_the_csym_box_sends_only_whole_numbers_from_1(typed, sent, shown):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the Bokeh JavaScript callbacks")
    from bokeh.models import ColumnDataSource, Spinner

    spinner = Spinner(low=1, mode="int", value=6)
    controls = ColumnDataSource(data=dict(value=[6], revision=[0]))
    commit, _ = hill_tab._setup_csym_control_js(spinner, controls, "hill-hill_csym")
    script = f"""
const sent = [];
const window = {{Shiny: {{setInputValue: (id, v) => sent.push([id, v])}}}};
const spinner = {{value: {json.dumps(typed) if typed == typed else "NaN"}}};
const controls = {{data: {{value: [6]}}}};
new Function('spinner', 'controls', 'input_id', 'window', {json.dumps(commit.code)})(
    spinner, controls, 'hill-hill_csym', window);
process.stdout.write(JSON.stringify([sent, spinner.value]));
"""
    out = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [[["hill-hill_csym", v] for v in sent], shown]


def test_the_csym_box_takes_the_value_the_server_sends():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the Bokeh JavaScript callbacks")
    from bokeh.models import ColumnDataSource, Spinner

    spinner = Spinner(low=1, mode="int", value=6)
    controls = ColumnDataSource(data=dict(value=[6], revision=[0]))
    _, restore = hill_tab._setup_csym_control_js(spinner, controls, "hill-hill_csym")
    script = f"""
const run = (shown, sent) => {{
    const spinner = {{value: shown}};
    const controls = {{data: {{value: [sent]}}}};
    new Function('spinner', 'controls', {json.dumps(restore.code)})(spinner, controls);
    return spinner.value;
}};
process.stdout.write(JSON.stringify([run(6, 3), run(3, 3), run(6, 0), run(6, 2.5)]));
"""
    out = subprocess.run([node, "-e", script], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout) == [3, 3, 6, 6]  # nonsense is not shown


class TestStaleGallerySelection:
    """The gallery's selection can outlive the stack it was made on."""

    def _harness(self):
        # three images selected of a stack that now has two
        h = _Effects("Image", hill_select_image=[0, 1, 11], hill_active_image=[0])
        h.ns["selected_images"].set(h.images)
        h.ns["avg_ps_pd_main_in_df"].set({0: _saved(10.0), 1: _saved(-20.0)})
        h.ns["_active_orig"][0] = 0
        shown = []
        h.ns["hill"].update_image_figure = lambda *a, **kw: shown.append(a[2])
        h.add("_show_transformed")
        return h, shown

    @staticmethod
    def _errors(caught):
        # an error in an effect is only a warning, which a test would not notice
        return [str(w.message) for w in caught if "Error in Effect" in str(w.message)]

    def test_indices_past_the_stack_are_left_out(self):
        h, _ = self._harness()
        with reactive.isolate():
            assert h.ns["_selected_indices"]() == [0, 1]

    def test_a_click_on_such_a_gallery_does_not_raise(self):
        import warnings

        async def run():
            h, shown = self._harness()
            h.add("_update_current_image_from_multiselect")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    await reactive.flush()
                    h.input.hill_active_image.set([1])
                    await reactive.flush()
                assert self._errors(caught) == []
                assert len(shown) == 1  # the second image, shown
            finally:
                h.close()

        asyncio.run(run())

    def test_a_control_change_on_such_a_gallery_does_not_raise(self):
        import warnings

        async def run():
            h, shown = self._harness()
            h.input.hill_active_image.set([2])  # past both
            h.add("_update_transformed_from_multiselect")
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    await reactive.flush()
                    h.input.hill_angle.set(5.0)
                    await reactive.flush()
                assert self._errors(caught) == []
                assert len(shown) >= 1  # the first image stands in
            finally:
                h.close()

        asyncio.run(run())

    def test_the_old_unguarded_index_would_have_raised(self):
        # the failure these guard against, shown on the same stack
        h, _ = self._harness()
        with reactive.isolate():
            with pytest.raises(IndexError):
                h.ns["data_all_2d"]()[list(h.input.hill_select_image())[2]]


class TestCsymIsAlwaysAnInteger:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (6, 6),
            (6.0, 6),
            ("4", 4),
            (2.4, 2),
            (2.6, 3),
            (2.5, 2),  # round half to even, as Python's round does
            (0, 1),
            (-3, 1),
            (0.2, 1),
        ],
    )
    def test_a_value_becomes_the_nearest_whole_number_from_1(self, value, expected):
        got = hill_tab._as_csym(value)
        assert got == expected and type(got) is int

    def test_a_non_integer_input_is_made_whole_at_the_source(self):
        async def run():
            h = _Effects("Image", hill_csym=2.6)
            updates = []
            h.ns["ui"].update_numeric = lambda *a, **kw: updates.append((a, kw))
            h.add("_sync_csym_spinner")
            try:
                await reactive.flush()
                assert updates == [(("hill_csym",), {"value": 3})]
                # the box and its channel take the whole number
                assert h.ns["spinner_csym"].value == 3
                assert h.ns["csym_controls"].data["value"] == [3]
                assert type(h.ns["csym_controls"].data["value"][0]) is int
            finally:
                h.close()

        asyncio.run(run())

    def test_a_whole_input_is_left_alone(self):
        async def run():
            h = _Effects("Image", hill_csym=4)
            updates = []
            h.ns["ui"].update_numeric = lambda *a, **kw: updates.append((a, kw))
            h.add("_sync_csym_spinner")
            try:
                await reactive.flush()
                assert updates == []
            finally:
                h.close()

        asyncio.run(run())

    @pytest.mark.parametrize("empty", [None, float("nan")])
    def test_an_emptied_input_is_ignored(self, empty):
        async def run():
            h = _Effects("Image", hill_csym=empty)
            h.add("_sync_csym_spinner")
            try:
                before = dict(h.ns["csym_controls"].data)
                await reactive.flush()
                assert h.ns["csym_controls"].data == before
            finally:
                h.close()

        asyncio.run(run())

    def test_the_layer_lines_are_computed_with_an_int(self):
        async def run():
            h = _Effects("Image", hill_csym=3.0)
            seen = []
            real = h.ns["hill"].compute_layer_line_positions

            def record(**kwargs):
                seen.append(kwargs["csym"])
                return real(**kwargs)

            h.ns["hill"].compute_layer_line_positions = record
            h.ns["selected_images"].set([h.images[0]])
            h.add("_update_layerline_figures")
            try:
                await reactive.flush()
                assert seen and all(type(c) is int and c == 3 for c in seen)
            finally:
                h.close()

        asyncio.run(run())


class TestTheModelFollowsTheInputsBeforeDisplay:
    """A bookmark's twist, rise and csym can arrive after the model was built and
    before the browser has the widget; a patch sent then is lost."""

    def _sliders(self):
        return (
            Slider(start=-180, end=180, value=29.4, step=0.01),
            Slider(start=134.2, end=536.8, value=268.4, step=0.5),
            Slider(start=10.96, end=43.84, value=21.92, step=0.05),
        )

    def test_the_sliders_take_the_values_and_ranges_the_browser_would_give_them(self):
        twist_s, pitch_s, rise_s = self._sliders()
        assert hill_tab._set_helix_sliders(twist_s, pitch_s, rise_s, 40.0, 324.0, 50.0)
        assert (twist_s.value, pitch_s.value, rise_s.value) == (40.0, 324.0, 50.0)
        # twist's range always holds +-180 and the value; the others widen only
        # when the value is outside them, to half and double it
        assert (twist_s.start, twist_s.end) == (-180.0, 180.0)
        assert (pitch_s.start, pitch_s.end) == (134.2, 536.8)
        assert (rise_s.start, rise_s.end) == (25.0, 100.0)
        assert rise_s.step == pytest.approx(100.0 * 0.001)

    def test_a_twist_outside_180_widens_its_range(self):
        twist_s, pitch_s, rise_s = self._sliders()
        hill_tab._set_helix_sliders(twist_s, pitch_s, rise_s, 200.0, 268.4, 21.92)
        assert (twist_s.start, twist_s.end) == (-180.0, 200.0)

    def test_values_already_shown_change_nothing(self):
        twist_s, pitch_s, rise_s = self._sliders()
        assert not hill_tab._set_helix_sliders(
            twist_s, pitch_s, rise_s, 29.4, 268.4, 21.92
        )

    def test_the_widget_is_rebuilt_when_the_model_had_to_change(self):
        h = _Effects("Image", hill_twist=40.0, hill_rise=25.0, hill_csym=3)
        with reactive.isolate():
            assert h.ns["_model_follows_inputs"]() is True
            assert h.ns["slider_twist"].value == 40.0
            assert h.ns["slider_rise"].value == 25.0
            assert h.ns["spinner_csym"].value == 3
            # a second time there is nothing to bring up to date
            assert h.ns["_model_follows_inputs"]() is False

    def test_a_model_that_already_agrees_is_not_rebuilt(self):
        h = _Effects("Image", hill_twist=29.4, hill_rise=21.92, hill_csym=6)
        with reactive.isolate():
            assert h.ns["_model_follows_inputs"]() is False

    def test_inputs_that_are_not_there_yet_are_not_followed(self):
        h = _Effects("Image")
        h.ns["input"].hill_twist = reactive.Value(None)  # unset: reading it is silent

        class Silent:
            def __call__(self):
                from shiny.types import SilentException

                raise SilentException()

        h.ns["input"].hill_twist = Silent()
        with reactive.isolate():
            assert h.ns["_model_follows_inputs"]() is False

    def test_the_render_uses_it_before_returning_the_widget(self):
        src = Path(hill_tab.__file__).read_text(encoding="utf-8")
        render = src[src.index("    def hill_main_plots():") :][:1100]
        assert "_model_follows_inputs()" in render
        assert render.index("update_from_model") < render.index(
            "return main_plot_widget"
        )


class TestCollapsiblePanelsAreSafe:
    """Shiny does not run an output that is hidden, so what the pipeline needs
    must not depend on an output inside a panel that starts collapsed."""

    def test_no_panel_carries_open_as_a_stray_attribute(self):
        # on a panel `open` is only an HTML attribute and says nothing: the first
        # panel of an accordion is open whatever it says. It belongs to the accordion.
        tree = ast.parse(Path(hill_tab.__file__).read_text(encoding="utf-8"))
        stray = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "accordion_panel"
            and any(kw.arg == "open" for kw in node.keywords)
        ]
        assert stray == []

    def test_every_accordion_says_whether_it_starts_open(self):
        tree = ast.parse(Path(hill_tab.__file__).read_text(encoding="utf-8"))
        silent = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "attr", None) == "accordion"
            and not any(kw.arg == "open" for kw in node.keywords)
        ]
        assert silent == []

    def test_the_outputs_in_the_collapsed_image_parameters_are_not_suspended(self):
        import re

        html = str(hill_tab.hill_tab_ui("hill"))
        item = html.rindex('data-value="hill_image_params"')
        body = html[html.index('class="accordion-collapse', item) :]
        body = body[: body.index('id="hill-hill_display_transformed"')]
        outputs = re.findall(
            r'class="[^"]*shiny-\w+-output[^"]*" id="hill-(hill_\w+)"', body
        )
        assert "hill_img_2d_uis" in outputs and "hill_img_flags_ui" in outputs
        source = Path(hill_tab.__file__).read_text(encoding="utf-8")
        for name in outputs:
            head = source[: source.index(f"def {name}(")]
            decorators = head[head.rindex("\n\n") :]
            assert "@output(suspend_when_hidden=False)" in decorators, name


class TestTransformedImage:
    """One place applies the inversion, rotation, shift and mask to an image."""

    def _h(self):
        h = _Effects("Image")
        h.ns["hill"].mask_2d_filament = lambda data, radius, apix, length: (
            data * 0.5 + radius + length
        )
        return h

    def test_nothing_is_done_for_what_is_not_asked(self):
        h = self._h()
        with reactive.isolate():
            out = h.ns["_transformed"](h.images[0], {"apix": 1.0})
        np.testing.assert_array_equal(out, h.images[0])
        assert h.transforms == []

    def test_the_transform_is_applied_with_the_parameters_in_order(self):
        h = self._h()
        params = dict(apix=2.0, angle=10.0, dx=2.0, dy=3.0, negate=True)
        with reactive.isolate():
            out = h.ns["_transformed"](h.images[0], params)
        assert h.transforms == [(10.0, 2.0, 3.0, True, 2.0)]
        np.testing.assert_array_equal(out, -h.images[0] + 15.0)

    @pytest.mark.parametrize(
        "radius,length,masked",
        [(4.0, 90.0, True), (0.0, 90.0, False), (4.0, 0.0, False), (4.0, 120.0, False)],
    )
    def test_the_mask_needs_a_radius_and_a_length_up_to_100_percent(
        self, radius, length, masked
    ):
        h = self._h()
        params = dict(apix=1.0, mask_radius=radius, mask_len=length)
        with reactive.isolate():
            out = h.ns["_transformed"](h.images[0], params)
        expected = (
            h.images[0] * 0.5 + radius + length / 100.0 if masked else h.images[0]
        )
        np.testing.assert_array_equal(out, expected)

    def test_a_missing_key_means_nothing(self):
        h = self._h()
        with reactive.isolate():
            out = h.ns["_transformed"](h.images[1], {"apix": 1.0, "angle": 0.0})
        np.testing.assert_array_equal(out, h.images[1])
