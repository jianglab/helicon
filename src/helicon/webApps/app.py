"""Helicon Lab — unified Shiny web app for helical structure analysis.

Integrates seven tools into a single tabbed interface with shared
project state for cross-tab data flow:

    HelicalLattice   — 2D lattice ⇔ helical lattice interconversion
    HelicalPitch     — derive twist from 2D class pair-distance histograms
    HILL             — helical indexing via Fourier layer lines
    HI3D             — helical indexing via cylindrical projection of 3D map
    denovo3D         — de novo 3D reconstruction from a single 2D image
    HelicalProjection— compare 2D images with helical structure projections
    whereIsMyClass   — map 2D classes to helical tube/filament images

Architecture: uses Shiny's classic ``App()`` API with modules so that
each tool is an independent ``@module.ui`` + ``@module.server`` pair
that can be composed into the parent navset.  A shared
``ProjectState`` singleton (defined in ``shared_state.py``) holds
reactive values that enable cross-tab data flow.

This is the only Shiny web app in helicon; the individual apps
(denovo3D, whereIsMyClass) were consolidated into this app.
"""

from __future__ import annotations

import logging
import os
import secrets
import signal
import threading
import time
from pathlib import Path

import ipywidgets.widgets.widget as _ipyw_mod

# ipyfilechooser creates sub-widgets (Output, etc.) whose comms aren't opened
# until shinywidgets processes them.  On browser refresh, shinywidgets calls
# get_state() on every widget → _widget_to_json() → model_id → crashes because
# sub-widget.comm is None.
#
# Two patches are needed:
# 1) Widget.model_id — returns None instead of crashing when comm is None.
# 2) Widget.get_state — skips individual traits that fail serialization, so a
#    single broken sub-widget doesn't prevent the entire parent from serializing.
_orig_model_id = _ipyw_mod.Widget.model_id.fget
_orig_get_state = _ipyw_mod.Widget.get_state


def _safe_model_id(self):
    if self.comm is None:
        return None
    return _orig_model_id(self)


def _safe_get_state(self, key=None, drop_defaults=False):
    try:
        return _orig_get_state(self, key=key, drop_defaults=drop_defaults)
    except (AttributeError, TypeError):
        from collections.abc import Iterable

        if key is None:
            keys = self.keys
        elif isinstance(key, str):
            keys = [key]
        elif isinstance(key, Iterable):
            keys = key
        else:
            return {}
        state = {}
        traits = self.traits()
        for k in keys:
            try:
                to_json = self.trait_metadata(k, "to_json", self._trait_to_json)
                value = to_json(getattr(self, k), self)
                if not drop_defaults or not self._compare(
                    value, traits[k].default_value
                ):
                    state[k] = value
            except (AttributeError, TypeError):
                continue
        return state


_ipyw_mod.Widget.model_id = property(_safe_model_id)
_ipyw_mod.Widget.get_state = _safe_get_state

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from shiny import App, reactive, ui

from ..lib.shiny import encode_query_params
from .lib.shared_state import project

logger = logging.getLogger(__name__)

_WEB_THEMES = {"Dark", "Light", "System"}


def _web_theme(request: Request) -> str:
    """Return the requested web-app theme, defaulting safely to Dark."""
    theme = str(request.query_params.get("helicon_theme", "Dark"))
    return theme if theme in _WEB_THEMES else "Dark"


# ── Tab module imports ────────────────────────────────────────────
# Each tab module provides:
#   - <name>_tab_ui(id)   → ui components for the tab
#   - <name>_tab_server(input, output, session, project)   → reactive logic

from .tabs.helical_lattice_tab import helical_lattice_tab_ui, helical_lattice_tab_server
from .tabs.helical_pitch_tab import helical_pitch_tab_ui, helical_pitch_tab_server
from .tabs.hill_tab import hill_tab_ui, hill_tab_server
from .tabs.hi3d_tab import hi3d_tab_ui, hi3d_tab_server
from .tabs.denovo3d_tab import denovo3d_tab_ui, denovo3d_tab_server
from .tabs.helical_projection_tab import (
    helical_projection_tab_ui,
    helical_projection_tab_server,
)
from .tabs.where_is_my_class_tab import (
    where_is_my_class_tab_ui,
    where_is_my_class_tab_server,
)

from .tabs import hill_tab, hi3d_tab, denovo3d_tab, helical_projection_tab
from .tabs import helical_pitch_tab, where_is_my_class_tab, helical_lattice_tab

# ── Bookmark module map ─────────────────────────────────────────
# Maps tab names to (module_prefix, tab_module) for constructing full
# input IDs from short keys in BOOKMARK_DEFAULTS.

_TAB_MODULE_MAP: dict[str, tuple[str, object]] = {
    "HILL": ("hill", hill_tab),
    "HI3D": ("hi3d", hi3d_tab),
    "Denovo3D": ("denovo3d", denovo3d_tab),
    "HelicalProjection": ("helical_projection", helical_projection_tab),
    "HelicalPitch": ("helical_pitch", helical_pitch_tab),
    "WhereIsMyClass": ("where_is_my_class", where_is_my_class_tab),
    "HelicalLattice": ("helical_lattice", helical_lattice_tab),
}


# ── Display → tab navigation control ─────────────────────────────
# The file browser (helicon display) launches this app with a per-launch
# helicon_token in the URL.  A control endpoint pair lets the browser
# navigate an already-open tab instead of spawning a second server+tab:
#
#   POST /helicon/navigate?token=...  {query_params}  → store pending nav
#   GET  /helicon/pending?token=...                  → consume pending nav
#
# The page polls /helicon/pending while open; a pending navigation makes
# it reload itself with the new query string (Shiny's URL bookmark store
# restores tab + inputs on load).  The token, learned from session URLs,
# scopes navigation to the display instance that launched this server.

_YOUNG_SERVER_SECS = 20.0  # a just-launched server is presumed alive
_STALE_POLL_SECS = 150.0  # hidden tabs poll ~1/min; a longer gap means dead tab
_IDLE_TIMEOUT_SECS = 600.0  # no poll + no session for 10 min → self-reap


class _AppControl:
    """Per-server state for display-driven tab navigation."""

    def __init__(self):
        self.seen_tokens: set[str] = set()
        self.pending: dict | None = None
        self.active_sessions = 0
        self.start_ts = time.monotonic()
        self.last_poll_ts = time.monotonic()
        self._lock = threading.Lock()
        self._watchdog_started = False

    def start_session(self) -> None:
        self.active_sessions += 1
        self.last_poll_ts = time.monotonic()

    def end_session(self) -> None:
        self.active_sessions = max(0, self.active_sessions - 1)
        self.last_poll_ts = time.monotonic()

    def register_token(self, url_search: str) -> None:
        from urllib.parse import parse_qs

        tokens = parse_qs(url_search.lstrip("?")).get("helicon_token")
        if tokens:
            self.seen_tokens.add(tokens[0])

    def is_alive(self) -> bool:
        now = time.monotonic()
        return (
            self.active_sessions > 0
            or now - self.start_ts < _YOUNG_SERVER_SECS
            or now - self.last_poll_ts < _STALE_POLL_SECS
        )

    def navigate(self, token: str, query_params) -> dict:
        if self.seen_tokens and token not in self.seen_tokens:
            return {"ok": False, "error": "token mismatch"}
        if not isinstance(query_params, dict) or not query_params:
            return {"ok": False, "error": "query_params dict required"}
        query_string = encode_query_params(query_params)
        if token:
            import urllib.parse

            query_string += "&helicon_token=" + urllib.parse.quote(token, safe="")
        with self._lock:
            self.pending = {"query_string": query_string}
        return {"ok": True, "alive": self.is_alive()}

    def poll(self, token: str) -> dict:
        self.last_poll_ts = time.monotonic()
        if token not in self.seen_tokens:
            return {"pending": False}
        with self._lock:
            pending = self.pending
            self.pending = None
        if pending is None:
            return {"pending": False}
        return {"pending": True, "query_string": pending["query_string"]}

    def start_watchdog(self) -> None:
        if self._watchdog_started:
            return
        self._watchdog_started = True
        threading.Thread(target=self._watchdog_loop, daemon=True).start()

    def _watchdog_loop(self) -> None:
        while True:
            time.sleep(30)
            if (
                self.active_sessions == 0
                and time.monotonic() - self.last_poll_ts > _IDLE_TIMEOUT_SECS
            ):
                os.kill(os.getpid(), signal.SIGTERM)
                return


_control = _AppControl()


async def _helicon_pending(request: Request):
    return JSONResponse(_control.poll(request.query_params.get("token", "")))


async def _helicon_navigate(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    return JSONResponse(
        _control.navigate(
            request.query_params.get("token", ""), body.get("query_params")
        )
    )


# ── Main app UI ────────────────────────────────────────────────────


def app_ui(request: Request):
    theme = _web_theme(request)
    initial_theme = "dark" if theme in {"Dark", "System"} else "light"
    return ui.page_fillable(
        ui.head_content(
            ui.tags.title("Helicon"),
            ui.tags.link(rel="icon", type="image/png", href="icon.png"),
            ui.tags.script(
                f"""
                (function() {{
                    var requested = {theme!r};
                    function applyTheme() {{
                        var dark = requested === 'Dark' ||
                            (requested === 'System' && window.matchMedia &&
                             window.matchMedia('(prefers-color-scheme: dark)').matches);
                        document.documentElement.dataset.heliconTheme = dark ? 'dark' : 'light';
                    }}
                    applyTheme();
                    if (requested === 'System' && window.matchMedia) {{
                        window.matchMedia('(prefers-color-scheme: dark)')
                            .addEventListener('change', applyTheme);
                    }}
                }})();
                """
            ),
            ui.tags.script(
                """
                var _BOOKMARK_TABS = {
                    "HILL": {
                        "input_mode": "hill-hill_input_mode",
                        "twist": "hill-hill_twist",
                        "rise": "hill-hill_rise",
                        "csym": "hill-hill_csym",
                        "apix": "hill-hill_apix",
                        "diameter": "hill-hill_diameter",
                        "cutoff_res_x": "hill-hill_cutoff_res_x",
                        "cutoff_res_y": "hill-hill_cutoff_res_y",
                        "log_amp": "hill-hill_log_amp",
                        "pnx": "hill-hill_pnx",
                        "pny": "hill-hill_pny",
                        "hp_fraction": "hill-hill_hp_fraction",
                        "lp_fraction": "hill-hill_lp_fraction",
                        "m_max": "hill-hill_m_max",
                        "const_image_color": "hill-hill_const_image_color",
                        "ll_colors": "hill-hill_ll_colors",
                        "fft_top_only": "hill-hill_fft_top_only",
                        "use_twist_pitch": "hill-hill_use_twist_pitch",
                        "out_of_plane_tilt": "hill-hill_out_of_plane_tilt",
                        "input_type": "hill-hill_input_type",
                        "angle": "hill-hill_angle",
                        "dx": "hill-hill_dx",
                        "dy": "hill-hill_dy"
                    },
                    "HI3D": {
                        "input_mode": "hi3d-hi3d_input_mode",
                        "apix": "hi3d-hi3d_apix",
                        "rmin": "hi3d-hi3d_rmin",
                        "rmax": "hi3d-hi3d_rmax",
                        "axial_step": "hi3d-hi3d_axial_step",
                        "npeaks": "hi3d-hi3d_npeaks",
                        "peak_width": "hi3d-hi3d_peak_width",
                        "peak_height": "hi3d-hi3d_peak_height"
                    },
                    "Denovo3D": {
                        "input_mode_images": "denovo3d-dn_input_mode_images",
                        "url_images": "denovo3d-dn_url_images",
                        "show_emdb": "denovo3d-dn_show_emdb_input_mode",
                        "is_3d": "denovo3d-dn_is_3d",
                        "ignore_blank": "denovo3d-dn_ignore_blank",
                        "plot_scores": "denovo3d-dn_plot_scores",
                        "show_download": "denovo3d-dn_show_download_buttons",
                        "display_size": "denovo3d-dn_selected_image_display_size",
                        "rec_length": "denovo3d-dn_reconstruct_length_rise",
                        "target_apix2d": "denovo3d-dn_target_apix2d",
                        "target_apix3d": "denovo3d-dn_target_apix3d",
                        "sym_oversample": "denovo3d-dn_sym_oversample",
                        "lr_alpha": "denovo3d-dn_lr_alpha",
                        "lr_l1_ratio": "denovo3d-dn_lr_l1_ratio",
                        "top_n": "denovo3d-dn_top_n_results",
                        "lr_algorithm": "denovo3d-dn_lr_algorithm",
                        "positive": "denovo3d-dn_positive_constraint",
                        "interpolation": "denovo3d-dn_interpolation",
                        "score_metric": "denovo3d-dn_score_metric",
                        "input_ui_type": "denovo3d-dn_input_ui_type"
                    },
                    "HelicalProjection": {
                        "mode_images": "helical_projection-input_mode_images",
                        "url_images": "helical_projection-url_images",
                        "mode_maps": "helical_projection-input_mode_maps",
                        "ignore_blank": "helical_projection-ignore_blank",
                        "show_pdb": "helical_projection-show_pdb",
                        "use_curated": "helical_projection-use_curated_helical_parameters",
                        "show_twist_star": "helical_projection-show_twist_star",
                        "proj_xyz": "helical_projection-map_projection_xyz_choices",
                        "xyz_size": "helical_projection-map_xyz_projection_display_size",
                        "side_size": "helical_projection-map_side_projection_vertical_display_size",
                        "length_z": "helical_projection-length_z",
                        "length_xy": "helical_projection-length_xy",
                        "scale_range": "helical_projection-scale_range",
                        "rescale_apix": "helical_projection-rescale_apix",
                        "match_sf": "helical_projection-match_sf",
                        "plot_scores": "helical_projection-plot_scores",
                        "hide_query": "helical_projection-hide_query_image"
                    },
                    "HelicalPitch": {
                        "mode_params": "helical_pitch-input_mode_params",
                        "url_params": "helical_pitch-url_params",
                        "mode_classes": "helical_pitch-input_mode_classes",
                        "url_classes": "helical_pitch-url_classes",
                        "ignore_blank": "helical_pitch-ignore_blank",
                        "sort_abundance": "helical_pitch-sort_abundance",
                        "auto_min_len": "helical_pitch-auto_min_len",
                        "max_len": "helical_pitch-max_len",
                        "max_pair_dist": "helical_pitch-max_pair_dist",
                        "bins": "helical_pitch-bins",
                        "min_len": "helical_pitch-min_len",
                        "rise": "helical_pitch-rise"
                    },
                    "WhereIsMyClass": {
                        "input_mode": "where_is_my_class-wimc_input_mode",
                        "url_star": "where_is_my_class-wimc_url_star",
                        "ignore_blank": "where_is_my_class-wimc_ignore_blank",
                        "sort_abundance": "where_is_my_class-wimc_sort_abundance",
                        "show_sharable": "where_is_my_class-wimc_show_sharable_url",
                        "rise": "where_is_my_class-wimc_rise",
                        "target_apix": "where_is_my_class-wimc_target_apix",
                        "low_pass": "where_is_my_class-wimc_low_pass_angstrom",
                        "high_pass": "where_is_my_class-wimc_high_pass_angstrom",
                        "max_len": "where_is_my_class-wimc_max_len",
                        "max_pair_dist": "where_is_my_class-wimc_max_pair_dist",
                        "bins": "where_is_my_class-wimc_bins",
                        "plot_height": "where_is_my_class-wimc_plot_height"
                    },
                    "HelicalLattice": {
                        "mode": "helical_lattice-radio",
                        "twist": "helical_lattice-twist",
                        "rise": "helical_lattice-rise",
                        "csym": "helical_lattice-csym",
                        "diameter": "helical_lattice-diameter",
                        "length": "helical_lattice-length",
                        "primitive_unitcell": "helical_lattice-primitive_unitcell",
                        "horizontal": "helical_lattice-horizontal",
                        "lattice_size_factor": "helical_lattice-lattice_size_factor",
                        "marker_size": "helical_lattice-marker_size",
                        "figure_height": "helical_lattice-figure_height",
                        "ax": "helical_lattice-ax",
                        "ay": "helical_lattice-ay",
                        "bx": "helical_lattice-bx",
                        "by": "helical_lattice-by",
                        "na": "helical_lattice-na",
                        "nb": "helical_lattice-nb"
                    }
                };

                var _initialValues = {};
                var _bookmarkTimer = null;
                var _initialCaptureDone = false;
                var _DEBUG_BOOKMARK = true;

                // Shiny appends a type suffix like :shiny.number to certain input
                // keys in $inputValues.  Look up both bare and typed versions.
                function _getVal(vals, fullId) {
                    if (fullId in vals) return vals[fullId];
                    // Try with common Shiny input type suffixes
                    var types = ['shiny.number', 'shiny.text', 'shiny.integer',
                                 'shiny.password', 'shiny.select', 'shiny.textarea'];
                    for (var i = 0; i < types.length; i++) {
                        var key = fullId + ':' + types[i];
                        if (key in vals) return vals[key];
                    }
                    return undefined;
                }

                function _captureInitialValues() {
                    var vals = Shiny.shinyapp.$inputValues;
                    if (!vals) {
                        if (_DEBUG_BOOKMARK) console.log('[bookmark] _captureInitialValues: $inputValues is null');
                        return;
                    }
                    for (var tabName in _BOOKMARK_TABS) {
                        var inputs = _BOOKMARK_TABS[tabName];
                        if (!_initialValues[tabName]) _initialValues[tabName] = {};
                        for (var shortKey in inputs) {
                            if (shortKey in _initialValues[tabName]) continue;
                            var val = _getVal(vals, inputs[shortKey]);
                            if (val !== undefined) {
                                _initialValues[tabName][shortKey] = val;
                            }
                        }
                    }
                    if (_DEBUG_BOOKMARK) console.log('[bookmark] _captureInitialValues: captured', JSON.stringify(_initialValues));
                }

                // Capture initial values after Shiny has fully initialized
                // and all dynamic (render.ui) inputs are in the DOM.
                $(document).on('shiny:connected', function() {
                    if (_DEBUG_BOOKMARK) console.log('[bookmark] shiny:connected');
                    _captureInitialValues();
                });
                setTimeout(function() {
                    if (_DEBUG_BOOKMARK) console.log('[bookmark] 2000ms timeout — capturing initial values');
                    _captureInitialValues();
                    _initialCaptureDone = true;
                    if (_DEBUG_BOOKMARK) console.log('[bookmark] _initialCaptureDone = true. initialValues:', JSON.stringify(_initialValues));
                }, 2000);

                $(document).on('shiny:inputchanged', function(event) {
                    if (_DEBUG_BOOKMARK) console.log('[bookmark] shiny:inputchanged:', event.name, '=', JSON.stringify(event.value));
                    _captureInitialValues();

                    clearTimeout(_bookmarkTimer);
                    _bookmarkTimer = setTimeout(function() {
                        if (_DEBUG_BOOKMARK) console.log('[bookmark] debounce fired, _initialCaptureDone=', _initialCaptureDone);
                        if (_initialCaptureDone) _buildBookmarkUrl();
                    }, 800);
                });

                function _buildBookmarkUrl() {
                    var vals = Shiny.shinyapp.$inputValues;
                    if (!vals) {
                        if (_DEBUG_BOOKMARK) console.log('[bookmark] _buildBookmarkUrl: $inputValues is null');
                        return;
                    }
                    var tab = vals['helicon_tab'];
                    if (!tab) {
                        if (_DEBUG_BOOKMARK) console.log('[bookmark] _buildBookmarkUrl: helicon_tab is undefined. Known keys:', Object.keys(vals).filter(function(k){ return k.indexOf('helicon') >= 0 || k.indexOf('tab') >= 0; }).join(','));
                        return;
                    }
                    var inputs = _BOOKMARK_TABS[tab];
                    if (!inputs) {
                        if (_DEBUG_BOOKMARK) console.log('[bookmark] _buildBookmarkUrl: tab', tab, 'not in _BOOKMARK_TABS');
                        return;
                    }

                    var parts = ['_inputs_',
                        'helicon_tab=' + encodeURIComponent(JSON.stringify(tab))];
                    var params = {};
                    var defaults = _initialValues[tab] || {};
                    var debug_excluded = [];
                    var debug_included = [];
                    for (var shortKey in inputs) {
                        var fullId = inputs[shortKey];
                        var val = _getVal(vals, fullId);
                        if (val === undefined || val === null) continue;
                        if (shortKey in defaults && JSON.stringify(val) === JSON.stringify(defaults[shortKey])) {
                            debug_excluded.push(shortKey);
                            continue;
                        }
                        params[shortKey] = val;
                        debug_included.push(shortKey + '=' + JSON.stringify(val) + ' (default=' + JSON.stringify(defaults[shortKey]) + ')');
                    }
                    if (_DEBUG_BOOKMARK) {
                        console.log('[bookmark] _buildBookmarkUrl: tab=', tab);
                        console.log('[bookmark]   excluded (matches default):', debug_excluded.join(', '));
                        console.log('[bookmark]   included:', debug_included.join(', '));
                    }
                    if (Object.keys(params).length > 0) {
                        parts.push('_values_');
                        parts.push('p=' + encodeURIComponent(JSON.stringify(params)));
                    }
                    var tok = new URLSearchParams(window.location.search).get('helicon_token');
                    if (tok) {
                        if (Object.keys(params).length === 0) parts.push('_values_');
                        parts.push('helicon_token=' + encodeURIComponent(tok));
                    }
                    var currentTheme = new URLSearchParams(window.location.search).get('helicon_theme');
                    if (currentTheme) {
                        parts.push('helicon_theme=' + encodeURIComponent(currentTheme));
                    }
                    var url = '?' + parts.join('&');
                    if (_DEBUG_BOOKMARK) console.log('[bookmark] final URL:', url);
                    window.history.replaceState(null, '', url);
                }

                Shiny.addCustomMessageHandler('triggerUrlSync', function(msg) {
                    if (_initialCaptureDone) _buildBookmarkUrl();
                });
            """
            ),
            ui.tags.script(
                """
                var _heliconToken = new URLSearchParams(window.location.search).get('helicon_token');
                if (_heliconToken) {
                    setInterval(function() {
                        fetch('/helicon/pending?token=' + encodeURIComponent(_heliconToken))
                            .then(function(r) { return r.json(); })
                            .then(function(data) {
                                if (data && data.pending) {
                                    var url = new URL(window.location.href);
                                    url.search = '?' + data.query_string;
                                    window.location.href = url.toString();
                                }
                            })
                            .catch(function() {});
                    }, 2000);
                }
                """
            ),
        ),
        ui.tags.style(
            f"""
            :root {{
                --helicon-page-bg: #2d2d2d;
                --helicon-text: #cccccc;
                --helicon-navbar-bg: #1f2937;
            }}
            :root[data-helicon-theme="light"] {{
                --helicon-page-bg: #f4f4f4;
                --helicon-text: #202020;
                --helicon-navbar-bg: #d9e2f0;
            }}
            html, body, .bslib-page-fill {{
                background-color: var(--helicon-page-bg) !important;
                color: var(--helicon-text);
            }}
            .navbar, .navbar-default, .navbar-inverse {{
                background-color: var(--helicon-navbar-bg) !important;
            }}
            * { font-size: 10pt; padding: 0; border: 0; margin: 0; }
           aside { --_padding-icon: 10px; }
            html, body { height: 100%; margin: 0; padding: 0; overflow-x: hidden; }
            .nav { padding: 0 8px; }
            .layout-sidebar { gap: 4px !important; }
            .sidebar { padding-right: 4px !important; }
             .main { padding-left: 4px !important; }
           /* bslib-page-fill puts 1rem padding + 1rem gap on <body>, which is a
              column flex container. When a child of <body> also becomes a flex
              column container (the case for div.container-fluid.html-fill-container
              wrapping navset_bar content), the parent's gap+padding interact with
              flex-basis:auto in a way that keeps flex-grow:1 from consuming the
              leftover vertical space. The result is that the
              div.container-fluid.html-fill-item.html-fill-container under <body>
              stops ~100px short of the body's bottom, leaving a visible white
              stripe after dynamic content (Bokeh plots) is injected.

              Removing body padding+gap gives the column flex container + its
              grow:1 child a clean main-axis to fill, so the tab content spans
              the entire height of the viewport. The navbar already hovers at
              the top so the 1rem gap was never visually meaningful anyway. */
           body.bslib-page-fill { padding: 0 !important; gap: 0 !important; }
        """
        ),
        ui.navset_bar(
            ui.nav_panel(
                "WhereIsMyClass", where_is_my_class_tab_ui("where_is_my_class")
            ),
            ui.nav_panel(
                "HelicalProjection", helical_projection_tab_ui("helical_projection")
            ),
            ui.nav_panel("HILL", hill_tab_ui("hill")),
            ui.nav_panel("HelicalPitch", helical_pitch_tab_ui("helical_pitch")),
            ui.nav_panel("Denovo3D", denovo3d_tab_ui("denovo3d")),
            ui.nav_panel("HelicalLattice", helical_lattice_tab_ui("helical_lattice")),
            ui.nav_panel("HI3D", hi3d_tab_ui("hi3d")),
            title=ui.tags.a(
                "Helicon",
                href="https://jianglab.science.psu.edu/helicon/",
                target="_blank",
                style="color: inherit; text-decoration: none; font-weight: bold; font-size: 12pt;",
            ),
            navbar_options=ui.navbar_options(
                underline=False, bg="#1f2937", theme=initial_theme
            ),
            fillable=True,
            gap=0,
            padding=0,
            id="helicon_tab",
        ),
    )


def server(input, output, session):
    """Top-level server: wires shared state and delegates to tab modules."""

    _control.start_session()
    _control.start_watchdog()

    @session.on_ended
    async def _on_webapp_session_ended():
        _control.end_session()

    # Learns the launch token from the browser URL (reactive-only API).
    @reactive.effect(priority=1000)
    def _register_launch_token():
        _control.register_token(session.clientdata.url_search())

    # ── Global unhandled-exception handler ────────────────────
    # Shiny catches exceptions inside reactive effects / render
    # functions and calls session._unhandled_error(e).  The default
    # only logs to stderr and closes the session without telling the
    # user.  We override it to also show a popup in the browser.
    import traceback as _tb

    _orig_unhandled = session._unhandled_error

    async def _show_error_modal(e: Exception) -> None:
        _tb_str = "".join(_tb.format_exception(type(e), e, e.__traceback__)).strip()
        ui.modal_show(
            ui.modal(
                ui.pre(
                    _tb_str,
                    style="white-space: pre-wrap; word-break: break-word;"
                    " font-size: 9pt; max-height: 60vh; overflow-y: auto;",
                ),
                title="Unhandled Error",
                easy_close=True,
                footer=None,
            )
        )
        await _orig_unhandled(e)

    type(session)._unhandled_error = lambda self, e: _show_error_modal(e)

    helical_lattice_tab_server("helical_lattice", project)
    helical_pitch_tab_server("helical_pitch", project)
    hill_tab_server("hill", project)
    hi3d_tab_server("hi3d", project)
    denovo3d_tab_server("denovo3d", project)
    helical_projection_tab_server("helical_projection", project)

    # Create FileChooser at top-level session context (ipywidgets comms
    # fail when created inside a @module.server nested session).
    from ipyfilechooser import FileChooser

    wimc_filechooser = FileChooser(
        path=".",
        select_desc="Select",
        show_hidden=False,
        filter_pattern=["*_data.star", "*.cs"],
        title="Select a RELION star or cryoSPARC cs file on the server",
    )
    where_is_my_class_tab_server(
        "where_is_my_class", project, wimc_filechooser=wimc_filechooser
    )

    @reactive.effect
    def _track_active_tab():
        tab = input.helicon_tab()
        if tab:
            project.active_tab.set(tab)

    @session.bookmark.on_bookmark
    async def _(state):
        tab = input.helicon_tab()
        if not tab or tab not in _TAB_MODULE_MAP:
            return
        prefix, tab_mod = _TAB_MODULE_MAP[tab]
        params = {}
        for short_key, (local_id, _default) in tab_mod.BOOKMARK_DEFAULTS.items():
            full_id = f"{prefix}-{local_id}"
            try:
                val = input[full_id]()
                if val is not None:
                    params[short_key] = val
            except Exception:
                pass
        if params:
            state.values["p"] = params

    @session.bookmark.on_bookmarked
    async def _(url: str):
        if "?" not in url:
            await session.bookmark.update_query_string(url)
            return
        base, query = url.split("?", 1)

        segments = query.split("&")
        kept = []
        in_inputs = False
        in_values = False
        for seg in segments:
            if seg == "_inputs_":
                in_inputs = True
                in_values = False
                kept.append(seg)
            elif seg == "_values_":
                in_values = True
                in_inputs = False
                kept.append(seg)
            elif in_inputs:
                if seg.startswith("helicon_tab="):
                    kept.append(seg)
            elif in_values:
                kept.append(seg)
        await session.bookmark.update_query_string(f"{base}?{'&'.join(kept)}")

    @session.bookmark.on_restore
    def _(state):
        params = state.values.get("p", {})
        if not params:
            return
        tab = input.helicon_tab()
        if not tab or tab not in _TAB_MODULE_MAP:
            return
        prefix, tab_mod = _TAB_MODULE_MAP[tab]
        for short_key, val in params.items():
            if short_key in tab_mod.BOOKMARK_DEFAULTS:
                local_id, _ = tab_mod.BOOKMARK_DEFAULTS[short_key]
                session.send_input_message(f"{prefix}-{local_id}", {"value": val})


async def sync_bookmark_url(session) -> None:
    """Trigger bookmark URL sync from a tab module."""
    await session.send_custom_message("triggerUrlSync", {})


# ── App object ────────────────────────────────────────────────────

app = App(
    app_ui,
    server,
    bookmark_store="url",
    static_assets=Path(__file__).parent / "www",
)

# Insert at the front: Starlette matches routes in order and
# ``init_starlette_app`` ends its list with a catch-all Mount("/"),
# which would shadow any routes appended after it.
app.starlette_app.routes.insert(
    0, Route("/helicon/pending", _helicon_pending, methods=["GET"])
)
app.starlette_app.routes.insert(
    0, Route("/helicon/navigate", _helicon_navigate, methods=["POST"])
)
