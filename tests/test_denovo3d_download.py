"""Tests for denovo3D reconstructed-map download sizing."""

import numpy as np

from helicon.webApps.tabs import denovo3d_tab


def _result(compact_map):
    reconstruction = np.ones((4, 4, 4), dtype=np.float32)
    return (
        0.9,
        (None, None, None, (reconstruction, None, None), 0, 0, 0, 0, compact_map),
        (None, "input.mrcs", 1, 4.0, 2.0, -1.5, 4.75, 1, 0, 0, 0),
    )


def test_prepare_download_map_matches_input_box_and_apix(monkeypatch):
    calls = []

    def fake_apply_helical_symmetry(**kwargs):
        calls.append(kwargs)
        return np.zeros(kwargs["new_size"], dtype=np.float64)

    monkeypatch.setattr(
        denovo3d_tab.helicon,
        "apply_helical_symmetry",
        fake_apply_helical_symmetry,
    )

    output, apix = denovo3d_tab._prepare_download_map(
        _result(np.ones((24, 8, 8))),
        match_input_box=True,
        input_image_shape=(48, 64),
        input_apix=1.25,
        compact_apix=2.5,
        cpu=3,
    )

    assert output.shape == (64, 64, 64)
    assert output.dtype == np.float32
    assert apix == 1.25
    assert calls[0]["new_size"] == (64, 64, 64)
    assert calls[0]["new_apix"] == 1.25


def test_match_input_box_is_enabled_by_default():
    assert denovo3d_tab.BOOKMARK_DEFAULTS["match_input_box"] == (
        "dn_match_input_box",
        True,
    )


def test_prepare_download_map_keeps_compact_map_when_disabled(monkeypatch):
    compact_map = np.ones((24, 8, 8), dtype=np.float64)

    def unexpected_apply_helical_symmetry(**_kwargs):
        raise AssertionError("compact export must not resize the reconstruction")

    monkeypatch.setattr(
        denovo3d_tab.helicon,
        "apply_helical_symmetry",
        unexpected_apply_helical_symmetry,
    )

    output, apix = denovo3d_tab._prepare_download_map(
        _result(compact_map),
        match_input_box=False,
        input_image_shape=(64, 64),
        input_apix=1.25,
        compact_apix=2.5,
        cpu=3,
    )

    assert output.shape == (24, 8, 8)
    assert output.dtype == np.float32
    assert apix == 2.5
