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
                                 keep=["Twist"], tilt=[0.]))
    ids = {k: f"{prefix}_{k}" for k in ("twist", "pitch", "rise", "use_twist_pitch")}
    hill_tab._setup_spinner_js(*spinners, *sliders, state)
    callbacks = hill_tab._setup_slider_js(*sliders, *spinners, [], ids, state)
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


@pytest.mark.parametrize("mode", ["PS", "PD"])
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
            h.ns["selected_images"].set([])
            await reactive.flush()
            assert not h.ns["fig_ellipses"]
            assert all(not c.args["fig_ellipses"] for c in h.ns["slider_callbacks"])
        finally:
            h.close()
    asyncio.run(run())


@pytest.mark.parametrize("delayed", [False, True])
def test_browser_control_callbacks_execute_without_feedback(delayed):
    """Execute production JS, including nested change events, with a tiny host."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the Bokeh JavaScript callbacks")
    sliders, spinners, state, callbacks = _controls()
    payload = dict(
        slider=callbacks[0].code,
        spinner=spinners[0].js_property_callbacks["change:value"][0].code,
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
const controls = {data: {keep:['Twist'], tilt:[0]}};
let keep = 'Twist';
const host = {Shiny:{setInputValue:(id,v)=>sent.push([id,v])},
    document:{getElementsByName:id=>{assert.equal(id,p.ids.use_twist_pitch);return [{checked:true,value:keep}];}}};
const window = {parent:host};
function model(value) {
    return { _value:value, cb:null, get value(){return this._value;},
        set value(v){if(this._value !== v){this._value=v; if(this.cb){
            if(p.delayed) pending.push(this.cb); else this.cb();
        }}}};
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
for (const s of [slider_twist,slider_pitch,slider_rise]) s.cb=()=>run(p.slider,s);
for (const [s,slider] of [[spinner_twist,slider_twist],[spinner_pitch,slider_pitch],[spinner_rise,slider_rise]])
    s.cb=()=>run(p.spinner,s,{slider});
slider_twist.value=40;
drain();
assert.equal(slider_pitch.value,180);
assert.deepEqual(sent.splice(0),[[p.ids.twist,40],[p.ids.pitch,180],[p.ids.rise,20]]);
assert.equal(el.data_source.data.y[1],1/20+1/180);
spinner_rise.value=25;
drain();
assert.equal(slider_pitch.value,225);
assert.equal(sent.splice(0).length,3);
keep='Pitch';
slider_rise.value=30;
drain();
assert.equal(slider_pitch.value,225);
assert.equal(slider_twist.value,48);
assert.equal(sent.splice(0).length,3);
controls.data={twist:[-60],pitch:[180],rise:[30],keep:['Twist'],tilt:[0]};
run(p.restore,controls);
drain();
assert.equal(slider_twist.value,-60);
assert.equal(spinner_pitch.value,180);
assert.equal(sent.length,0);
// This tuple does not round-trip exactly through 360 * rise / pitch.
controls.data={twist:[29.4],pitch:[360*21.92/29.4],rise:[21.92],keep:['Twist'],tilt:[0]};
run(p.restore,controls);
drain();
assert.equal(sent.length,0);
assert.equal(slider_twist.value,29.4);
keep='Twist';
slider_twist.value=0;
drain();
assert.equal(slider_pitch.value,21.92);
assert.equal(sent.splice(0).length,3);
"""
    result = subprocess.run([node, "-e", script], input=json.dumps(payload),
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
