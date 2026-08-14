"""Focused tests for HILL tab parameter updates."""

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
