"""Tests for the shared proc3d engine (GUI + CLI dispatch loop)."""

from __future__ import annotations

import numpy as np
import pytest

from helicon.lib.proc3d_engine import (
    apply_options,
    gui_operation_specs,
    operation_specs,
    stack_to_namespace,
)


def _volume(nz=4, ny=5, nx=6):
    """Return a deterministic float32 volume with shape (nz, ny, nx)."""
    return np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx)


class TestOperationSpecs:
    def test_specs_cover_all_registered_plugins(self):
        from helicon.plugins import proc3d as plugins

        specs = operation_specs()
        assert set(specs) == set(plugins._plugins)
        for name in ("apix", "clip", "flip_hand", "z_moving_average"):
            assert name in specs

    def test_spec_shape_matches_engine_contract(self):
        specs = operation_specs()
        expected_keys = {
            "dest",
            "option_string",
            "metavar",
            "type",
            "nargs",
            "choices",
            "default",
            "help",
            "append",
        }
        for name, spec in specs.items():
            assert set(spec) == expected_keys, name
            assert spec["dest"] == name
            assert spec["option_string"] == f"--{name}"
            assert spec["type"] is str
            assert spec["nargs"] in (None, "?")
            assert spec["append"] is False

    def test_gui_specs_equal_full_specs(self):
        # Every proc3d operation transforms the volume in memory, so the GUI
        # panel exposes the whole option set (unlike images2star, which hides
        # file-writing operations).
        assert gui_operation_specs() == operation_specs()


class TestStackToNamespace:
    def test_seeds_plugin_defaults_and_cli_defaults(self):
        specs = operation_specs()
        stack = [("apix", "1.35")]
        args = stack_to_namespace(stack, specs)
        for name in specs:
            if name in {n for n, _ in stack}:
                continue
            assert args.__dict__[name] == specs[name]["default"]
        assert args.verbose == 0
        assert args.cpu == 1
        assert args.force == 0
        assert args.apix == "1.35"

    def test_duplicate_non_append_option_rejected(self):
        specs = operation_specs()
        with pytest.raises(ValueError, match="can only be applied once"):
            stack_to_namespace([("flip_hand", "x"), ("flip_hand", "y")], specs)

    def test_unknown_option_rejected(self):
        specs = operation_specs()
        with pytest.raises(ValueError, match="unknown operation"):
            stack_to_namespace([("not_an_option", "x")], specs)


class TestApplyOptions:
    def _apply(self, data, stack, apix=1.0):
        specs = operation_specs()
        args = stack_to_namespace(stack, specs)
        return apply_options(
            data, apix, [name for name, _ in stack], args
        )

    def test_apix_updates_pixel_size_only(self):
        data = _volume()
        out, apix = self._apply(data, [("apix", "1.35")])
        assert np.array_equal(out, data)
        assert apix == 1.35

    def test_flip_hand_x(self):
        data = _volume()
        out, _ = self._apply(data, [("flip_hand", "x")])
        assert np.array_equal(out, data[:, :, ::-1])

    def test_flip_hand_y_and_z(self):
        data = _volume()
        out, _ = self._apply(data, [("flip_hand", "y")])
        assert np.array_equal(out, data[:, ::-1, :])
        out, _ = self._apply(data, [("flip_hand", "z")])
        assert np.array_equal(out, data[::-1, :, :])

    def test_clip_changes_dimensions(self):
        data = _volume()
        out, apix = self._apply(data, [("clip", "new_nx=3:new_ny=2:new_nz=1")])
        assert out.shape == (1, 2, 3)
        assert apix == 1.0

    def test_z_moving_average_with_length(self):
        data = _volume()
        out, _ = self._apply(data, [("z_moving_average", "length=2")])
        assert out.shape == data.shape
        # The window average leaves the boundary slices untouched.
        assert np.array_equal(out[0], data[0])
        assert np.array_equal(out[-1], data[-1])

    def test_ordered_dispatch_clip_sees_flipped_geometry(self):
        data = _volume()
        out, _ = self._apply(
            data,
            [
                ("flip_hand", "x"),
                ("clip", "new_nx=3:new_ny=3:new_nz=3"),
            ],
        )
        expected = data[:, :, ::-1][1:4, 1:4, 1:4]
        assert out.shape == (3, 3, 3)
        # clip centers on the flipped geometry: center_x=3, center_y=2,
        # center_z=2, so the 3x3x3 window starts at [1, 1, 2].
        expected = data[:, :, ::-1][1:4, 1:4, 2:5]
        assert np.array_equal(out, expected)

    def test_source_volume_never_mutated(self):
        data = _volume()
        snapshot = data.copy()
        self._apply(
            data,
            [
                ("flip_hand", "x"),
                ("clip", "new_nx=3:new_ny=3:new_nz=3"),
                ("z_moving_average", "length=2"),
                ("apix", "2.0"),
            ],
        )
        assert np.array_equal(data, snapshot)

    def test_unknown_option_raises(self):
        specs = operation_specs()
        args = stack_to_namespace([], specs)
        # Mirror the CLI/engine contract: an option missing from the
        # namespace cannot be dispatched (argparse would never produce it).
        with pytest.raises(KeyError):
            apply_options(_volume(), 1.0, ["missing"], args)

    def test_fft_resample_changes_geometry_and_apix(self):
        data = _volume(4, 6, 8)
        out, apix = self._apply(data, [("fft_resample", "new_nx=4:new_ny=4:new_nz=4")])
        assert out.shape == (4, 4, 4)
        assert apix == pytest.approx(round(1.0 * 8 / 4, 4))

    def test_helical_sym_preserves_shape(self):
        data = _volume()
        out, _ = self._apply(data, [("helical_sym", "twist=10:rise=2:center_len=5")])
        assert out.shape == data.shape
