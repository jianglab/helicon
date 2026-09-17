"""Every input the HI3D tab declares should be read somewhere.

A Shiny input that nothing reads is inert: it renders, it accepts typing, and
it changes nothing. That is how the radial range broke -- rmin/rmax were
declared, seeded from the map, and drawn as markers, but ``input.hi3d_rmin``
appeared nowhere, so editing them moved neither the markers nor the cylindrical
projection computed from them.

Nothing here can prove an input is wired *correctly*; that needs the running
app. It does prove an input is not wired at all, which is a cheap check for a
failure mode that looks, from the outside, exactly like nothing happening.
"""

import pathlib
import re

import pytest

TAB = pathlib.Path(__file__).resolve().parents[1] / (
    "src/helicon/webApps/tabs/hi3d_tab.py"
)

# Inputs known to be declared and never read. Each is a real gap, not an
# exemption on principle: adding one here should mean writing down why.
KNOWN_INERT = {
    # "Center & verticalize" under "Transform the map". The checkbox exists and
    # nothing consumes it, so ticking it does nothing at all.
    "hi3d_do_transform",
}


def _declared_and_used():
    src = TAB.read_text()
    declared = set(re.findall(r'ui\.input_\w+\(\s*["\'](hi3d_\w+)["\']', src))
    used = set(re.findall(r"input\.(hi3d_\w+)", src))
    return declared, used


def test_the_tab_declares_inputs_at_all():
    declared, _ = _declared_and_used()
    assert len(declared) > 20, "the scan found almost nothing; has the UI moved?"


def test_no_new_input_is_left_unread():
    declared, used = _declared_and_used()
    inert = declared - used - KNOWN_INERT
    assert (
        not inert
    ), f"declared but never read, so editing them does nothing: {sorted(inert)}"


def test_the_radial_range_reaches_the_computation():
    """The specific regression: rmin/rmax must be read, and must also be able to
    wake the computation that uses them, which is gated by reactive.event."""
    src = TAB.read_text()
    for name in ("hi3d_rmin", "hi3d_rmax"):
        assert f"input.{name}" in src, f"{name} is declared but never read"
    run = src[
        src.index("def _run_computation") - 600 : src.index("def _run_computation")
    ]
    for name in ("hi3d_rmin", "hi3d_rmax"):
        assert name in run, (
            f"{name} is missing from _run_computation's reactive.event list, so a"
            " change to it cannot re-trigger the cylindrical projection"
        )


def test_the_known_inert_list_stays_honest():
    """If one of these gets wired up, drop it from the list rather than leaving
    a stale exemption that would hide a later regression."""
    declared, used = _declared_and_used()
    stale = {n for n in KNOWN_INERT if n in used}
    assert not stale, f"now wired up, remove from KNOWN_INERT: {sorted(stale)}"

    missing = {n for n in KNOWN_INERT if n not in declared}
    assert (
        not missing
    ), f"no longer declared, remove from KNOWN_INERT: {sorted(missing)}"
