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
    spinners = [Spinner(value=v) for v in (30, 240, 20)]
    state = ColumnDataSource(dict(twist=[30.], pitch=[240.], rise=[20.],
                                 keep=["Twist"], tilt=[0.], revision=[0]))
    ids = {k: f"{prefix}_{k}" for k in (
        "twist", "pitch", "rise", "use_twist_pitch", "helix_revision")}
    callbacks = hill_tab._setup_helix_control_js(*sliders, *spinners, [], ids, state)
    return sliders, spinners, state, callbacks


class _Effects:
    """Run production nested effects with real Shiny scheduling, without a browser.

    Compile their original decorators and bodies; substitute only session/UI and
    expensive numerical work. This tests dependency wiring as well as results.
    """

    def __init__(self, mode="PS", **overrides):
        values = dict(
            hill_select_image=[0], hill_input_type=mode, hill_is_3d=False,
            hill_inhibit_update=False, hill_apix=1., hill_angle=0., hill_dx=0.,
            hill_dy=0., hill_mask_radius=0., hill_mask_len=90., hill_negate=False,
            hill_cutoff_res_x=3., hill_cutoff_res_y=2., hill_pnx=8, hill_pny=8,
            hill_log_amp=False, hill_lp_fraction=0., hill_hp_fraction=0.,
            hill_m_max=1, hill_diameter=100., hill_csym=1, hill_out_of_plane_tilt=0.,
            hill_fft_top_only=False, hill_ll_colors="lime cyan", hill_LL=True,
            hill_LLText=True, hill_twist=30., hill_rise=20., hill_ms=["0"],
            hill_use_twist_pitch="Twist",
            hill_helix_revision=0,
        )
        values.update(overrides)
        self.input = SimpleNamespace(**{k: reactive.Value(v) for k, v in values.items()})
        self.images = [np.arange(64.).reshape(8, 8), np.arange(64.).reshape(8, 8)[::-1].copy()]
        self.transforms = []
        self.computations = []
        self.updates = []

        def transform(data, angle, dx, dy, negate, apix):
            self.transforms.append((angle, dx, dy, negate, apix))
            return (-data if negate else data) + angle + dx + dy

        def spectrum(data, **kwargs):
            self.computations.append(data.copy())
            return data.copy()

        # Use real layerline calculations and figures: their coordinates matter.
        compute = SimpleNamespace(
            transform_2d_image=transform,
            mask_2d_filament=lambda data, *args: data * .5,
            update_image_figure=lambda *args, **kwargs: None,
            compute_power_spectra=lambda data, **kw: (spectrum(data), data * 0),
            compute_phase_difference_across_meridian=lambda phase: phase,
            resize_rescale_power_spectra=spectrum,
            compute_layer_line_positions=hill_tab.hill.compute_layer_line_positions,
            bessel_n_image=lambda ny, nx, *args: np.zeros((ny, nx)),
            twist2pitch=hill_tab.hill.twist2pitch,
        )
        self.sliders, self.spinners, controls, callbacks = _controls()
        figs = [figure(), figure(), figure()]
        self.ns = dict(vars(hill_tab))
        self.ns.update(
            input=self.input, hill=compute,
            ui=SimpleNamespace(update_numeric=lambda *a, **kw: self.updates.append((a, kw))),
            selected_images=reactive.Value([]), selected_image_labels=reactive.Value([]),
            data_all_2d=reactive.Value(self.images), data_all_2d_labels=reactive.Value(["1", "2"]),
            prev_data_from=reactive.Value("main"), data_2d_transformed=reactive.Value(None),
            ps_data=reactive.Value(None), pd_data=reactive.Value(None), phase_data=reactive.Value(None),
            _last_ps_pd_selection=[None], _last_transform_key=[None],
            ny_curr_img=reactive.Value(8), nx_curr_img=reactive.Value(8),
            fig_transformed_img=None, source_transformed_img=None,
            fig_ellipses=[], figs=figs, figs_image=[figs[0], figs[2]],
            data_source_ps=ColumnDataSource(dict(image=[self.images[0]])),
            data_source_pd=ColumnDataSource(dict(image=[self.images[0]])),
            slider_callbacks=callbacks, helix_controls=controls,
        )
        self.effects = []
        tree = ast.parse(Path(hill_tab.__file__).read_text(encoding="utf-8"))
        self.server = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                           and n.name == "hill_tab_server")

    def add(self, *names):
        for name in names:
            node = next(n for n in self.server.body if isinstance(n, ast.FunctionDef)
                        and n.name == name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), hill_tab.__file__, "exec"), self.ns)
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
            np.testing.assert_array_equal(h.get("ps_data" if mode == "PS" else "pd_data"), h.images[1])
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
        h = _Effects(mode, hill_inhibit_update=inhibit, hill_angle=10., hill_dx=2., hill_dy=3.,
                     hill_mask_radius=4., hill_negate=True)
        h.add("_update_selected_images", "_apply_2d_transform_ps_pd", "_compute_ps_pd")
        try:
            await reactive.flush()
            assert h.transforms[-1] == ((10., 2., 3., False, 1.) if inhibit else (0., 0., 0., False, 1.))
            # No UI echo is simulated: reset must already apply on the server.
            np.testing.assert_array_equal(h.get("data_2d_transformed"), h.images[0] + (15 if inhibit else 0))
            h.input.hill_angle.set(20.)
            await reactive.flush()
            assert h.transforms[-1] == (20., 2., 3., False, 1.)
            h.input.hill_select_image.set([1])
            await reactive.flush()
            assert h.transforms[-1] == ((20., 2., 3., False, 1.) if inhibit else (0., 0., 0., False, 1.))
        finally:
            h.close()
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["PS", "PD"])
def test_mode_switch_recomputes_before_spectrum(mode):
    async def run():
        h = _Effects("Image", hill_negate=True, hill_mask_radius=4.)
        h.ns["selected_images"].set([h.images[0]])
        h.add("_apply_2d_transform", "_apply_2d_transform_ps_pd", "_compute_ps_pd")
        try:
            await reactive.flush()
            np.testing.assert_array_equal(h.computations[-1], -h.images[0] * .5)
            h.input.hill_input_type.set(mode)
            await reactive.flush()
            np.testing.assert_array_equal(h.computations[-1], h.images[0])
            h.input.hill_input_type.set("Image")
            await reactive.flush()
            np.testing.assert_array_equal(h.computations[-1], -h.images[0] * .5)
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
            h.input.hill_angle.set(1.)
            h.input.hill_angle.set(0.)
            await reactive.flush()
            np.testing.assert_array_equal(h.get("data_2d_transformed"), h.images[1])
        finally:
            h.close()
    asyncio.run(run())


def test_url_mode_keeps_restored_parameters_until_leaving_emdb():
    async def run():
        h = _Effects(hill_input_mode_params="2", hill_twist=60., hill_rise=30.)
        h.ns["_previous_input_mode"] = [None]
        h.add("_reset_helix_after_emdb", "_sync_helix_controls")
        try:
            await reactive.flush()
            assert not h.updates
            assert h.ns["helix_controls"].data["twist"] == [60.]
            assert h.ns["helix_controls"].data["pitch"] == [180.]
            h.input.hill_input_mode_params.set("3")
            await reactive.flush()
            assert not h.updates
            h.input.hill_input_mode_params.set("2")
            await reactive.flush()
            assert [args[0] for args, _ in h.updates] == ["hill_twist", "hill_rise", "hill_csym"]
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
            h.input.hill_twist.set(40.)
            h.input.hill_rise.set(25.)
            await reactive.flush()
            assert h.ns["helix_controls"].data["pitch"] == [225.]
            for i in range(4):
                old = list(h.ns["fig_ellipses"])
                h.ns["selected_images"].set([h.images[i % 2]])
                h.input.hill_diameter.set(110. + i)
                h.input.hill_LLText.set(i % 2 == 0)
                await reactive.flush()
                for renderer in h.ns["fig_ellipses"]:
                    m, orders = renderer.tags
                    np.testing.assert_allclose(renderer.data_source.data["y"],
                                               m / 25. + np.asarray(orders) / 225.)
                for callback in h.ns["slider_callbacks"]:
                    assert callback.args["fig_ellipses"] == h.ns["fig_ellipses"]
                    assert not any(r in old for r in callback.args["fig_ellipses"])
                assert all(len(s.js_property_callbacks["change:value"]) == 1 for s in h.sliders)
                assert all(len(s.js_property_callbacks["change:value_throttled"]) == 1
                           for s in h.sliders)
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
    sliders, spinners, state, callbacks = _controls()
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
const spinner_twist=model(30), spinner_pitch=model(240), spinner_rise=model(20);
const el={tags:[1,[0,1]],data_source:{data:{y:[0,0]},change:{emit(){}}}};
const args={slider_twist,slider_pitch,slider_rise,spinner_twist,spinner_pitch,spinner_rise,
    controls,window,input_ids:p.ids,fig_ellipses:[el]};
function run(code, cb_obj, extra={}) {
    const a={...args,...extra,cb_obj};
    new Function(...Object.keys(a),code)(...Object.values(a));
}
const pairs = [[slider_twist,spinner_twist],[slider_pitch,spinner_pitch],[slider_rise,spinner_rise]];
for (let origin=0; origin<pairs.length; origin++) {
    for (const control of pairs[origin]) {
        control.value_cb=()=>run(p.preview,control,{origin,control});
        control.throttled_cb=()=>run(p.commit,control,{origin,control});
    }
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

// All three spinner paths follow the same preview/commit contract.
keep='Twist';
spinner_twist.value=-60;
drain();
assert.equal(slider_twist.value,-60);
assert.equal(spinner_pitch.value,180);
assert.equal(sent.length,0);
release(spinner_twist);
expectBatch([-60,180,30],5);

spinner_pitch.value=150;
drain();
assert.equal(slider_twist.value,-72);
assert.equal(sent.length,0);
release(spinner_pitch);
expectBatch([-72,150,30],6);

spinner_rise.value=35;
drain();
assert.equal(slider_twist.value,-72);
assert.equal(slider_pitch.value,175);
assert.equal(sent.length,0);
release(spinner_rise);
expectBatch([-72,175,35],7);

// Invalid edits restore the last valid preview and are not committed.
spinner_pitch.value=0;
drain();
assert.equal(spinner_pitch.value,175);
assert.equal(sent.length,0);
run(p.commit,spinner_pitch,{origin:1,control:spinner_pitch});
drain();
assert.equal(sent.length,0);
"""
    result = subprocess.run([node, "-e", script], input=json.dumps(payload),
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
