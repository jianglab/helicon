"""Shared reactive project state for Helicon Lab.

This module defines a single ``ProjectState`` whose reactive values persist
across all tabs.  Any tab can read or write these values, enabling
cross-tab data flow:

    HelicalPitch → (twist estimate) → HILL / denovo3D / HelicalProjection
    denovo3D     → (reconstructed 3D map) → HelicalProjection
    whereIsMyClass → (selected class images) → HelicalPitch / denovo3D

The reactive values are intentionally lightweight wrappers; heavy data
(numpy arrays, DataFrames) live in ``_data`` and are accessed by key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shiny import reactive


@dataclass
class ProjectState:
    """Centralised reactive state shared across all Helicon Lab tabs.

    Attributes
    ----------
    twist : reactive.value
        Estimated helical twist (degrees).  Populated by HelicalPitch,
        HILL, or HI3D; consumed by denovo3D, HelicalProjection.
    rise : reactive.value
        Estimated helical rise (Angstrom).  Same producer/consumer pattern.
    csym : reactive.value
        Cyclic symmetry order (integer >= 1).
    apix : reactive.value
        Pixel size (Angstrom/pixel) of the currently loaded image set.
    diameter : reactive.value
        Estimated helical diameter (Angstrom).

    input_images : reactive.value
        Currently selected 2D images (list of numpy arrays), shared
        between whereIsMyClass / HelicalPitch / denovo3D.
    input_params : reactive.value
        DataFrame of particle/star-file parameters, shared between
        HelicalPitch and whereIsMyClass.
    input_map : reactive.value
        3D map (numpy array) produced by denovo3D or loaded for
        HelicalProjection / HI3D.
    input_map_apix : reactive.value
        Voxel size of ``input_map``.
    active_tab : reactive.value
        Name of the currently visible tab, so side-effects can be gated.
    """

    # ── Helical parameters (shared estimates) ──────────────────────
    twist: reactive.Value[float] = field(default_factory=lambda: reactive.value(0.0))
    rise: reactive.Value[float] = field(default_factory=lambda: reactive.value(0.0))
    csym: reactive.Value[int] = field(default_factory=lambda: reactive.value(1))
    apix: reactive.Value[float] = field(default_factory=lambda: reactive.value(0.0))
    diameter: reactive.Value[float] = field(default_factory=lambda: reactive.value(0.0))

    # ── Shared input data ─────────────────────────────────────────
    input_images: reactive.Value[Any] = field(
        default_factory=lambda: reactive.value(None)
    )
    input_params: reactive.Value[Any] = field(
        default_factory=lambda: reactive.value(None)
    )
    input_map: reactive.Value[Any] = field(default_factory=lambda: reactive.value(None))
    input_map_apix: reactive.Value[float] = field(
        default_factory=lambda: reactive.value(0.0)
    )

    # ── Navigation ────────────────────────────────────────────────
    active_tab: reactive.Value[str] = field(
        default_factory=lambda: reactive.value("helical_lattice")
    )

    # ── Scratch namespace for tab-private-but-readable values ──────
    _data: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """Store an ad-hoc value that other tabs may read by key."""
        if key in (
            "twist",
            "rise",
            "csym",
            "apix",
            "diameter",
            "input_images",
            "input_params",
            "input_map",
            "input_map_apix",
            "active_tab",
        ):
            getattr(self, key).set(value)
        else:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value by key (first-class reactive or scratch)."""
        if key in (
            "twist",
            "rise",
            "csym",
            "apix",
            "diameter",
            "input_images",
            "input_params",
            "input_map",
            "input_map_apix",
            "active_tab",
        ):
            return getattr(self, key)()
        return self._data.get(key, default)


# Singleton instance imported by all tabs
project = ProjectState()
