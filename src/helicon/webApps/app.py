"""Helicon Lab — unified Shiny web app for helical structure analysis.

Integrates eight tools into a single tabbed interface with shared
project state for cross-tab data flow, behind a Home tab that places each
tool on a helical data processing workflow diagram (alongside launchers for
related apps that run outside this one, such as ``helicon procart``):

    HelicalLattice   — 2D lattice ⇔ helical lattice interconversion
    HelicalPitch     — derive twist from 2D class pair-distance histograms
    HILL             — helical indexing via Fourier layer lines
    HI3D             — helical indexing via cylindrical projection of 3D map
    denovo3D         — de novo 3D reconstruction from a single 2D image
    abinitio3D       — ab initio 3D map from the 2D classes of one helical type
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

from helicon.lib.shiny import encode_query_params
from helicon.webApps.lib.shared_state import project

logger = logging.getLogger(__name__)

_WEB_THEMES = {"Dark", "Light", "System"}


def _web_theme(request: Request) -> str:
    """Return the requested web-app theme, defaulting to Light."""
    theme = str(request.query_params.get("helicon_theme", "Light"))
    return theme if theme in _WEB_THEMES else "Light"


# ── Tab module imports ────────────────────────────────────────────
# Each tab module provides:
#   - <name>_tab_ui(id)   → ui components for the tab
#   - <name>_tab_server(input, output, session, project)   → reactive logic

from helicon.webApps.tabs.helical_lattice_tab import (
    helical_lattice_tab_ui,
    helical_lattice_tab_server,
)
from helicon.webApps.tabs.helical_pitch_tab import (
    helical_pitch_tab_ui,
    helical_pitch_tab_server,
)
from helicon.webApps.tabs.hill_tab import hill_tab_ui, hill_tab_server
from helicon.webApps.tabs.hi3d_tab import hi3d_tab_ui, hi3d_tab_server
from helicon.webApps.tabs.denovo3d_tab import denovo3d_tab_ui, denovo3d_tab_server
from helicon.webApps.tabs.abinitio3d_tab import (
    abinitio3d_tab_ui,
    abinitio3d_tab_server,
)
from helicon.webApps.tabs.helical_projection_tab import (
    helical_projection_tab_ui,
    helical_projection_tab_server,
)
from helicon.webApps.tabs.where_is_my_class_tab import (
    where_is_my_class_tab_ui,
    where_is_my_class_tab_server,
)
from helicon.webApps import bookmark
from helicon.webApps.tabs.home_tab import (
    HOME_APPS,
    HOME_TAB,
    home_tab_server,
    home_tab_ui,
)

from helicon.webApps.tabs import (
    hill_tab,
    hi3d_tab,
    denovo3d_tab,
    abinitio3d_tab,
    helical_projection_tab,
)
from helicon.webApps.tabs import (
    helical_pitch_tab,
    where_is_my_class_tab,
    helical_lattice_tab,
)

# ── Bookmark module map ─────────────────────────────────────────
# Maps tab names to (module_prefix, tab_module) for constructing full
# input IDs from short keys in BOOKMARK_DEFAULTS.

_TAB_MODULE_MAP: dict[str, tuple[str, object]] = {
    "HILL": ("hill", hill_tab),
    "HI3D": ("hi3d", hi3d_tab),
    "Denovo3D": ("denovo3d", denovo3d_tab),
    "AbInitio3D": ("abinitio3d", abinitio3d_tab),
    "HelicalProjection": ("helical_projection", helical_projection_tab),
    "HelicalPitch": ("helical_pitch", helical_pitch_tab),
    "WhereIsMyClass": ("where_is_my_class", where_is_my_class_tab),
    "HelicalLattice": ("helical_lattice", helical_lattice_tab),
}


assert {a.name for a in HOME_APPS if a.is_tab} == set(_TAB_MODULE_MAP), (
    "HOME_APPS must name exactly the tabs in _TAB_MODULE_MAP; a renamed or "
    "added tab would otherwise be missing from, or unreachable from, Home"
)

# ── Lazy tab loading ────────────────────────────────────────────


def _resolve_active_tab(request: Request) -> str:
    """Which tab to open: the one the URL names, otherwise Home.

    Only the resolved tab's server function runs at session start. Every tab's
    UI is still built (that measured 0.01 s for all seven), so navigation works
    normally and Shiny keeps hidden outputs suspended; what lazy loading avoids
    is each tab module's eager reactive effects, which is where session-start
    time actually went.
    """
    tab, _ = bookmark.parse(request.url.query, _TAB_MODULE_MAP)
    return tab or HOME_TAB


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


# ── Bookmark URL (page side) ────────────────────────────────────
# Keeps the address bar a bookmark of the current tab: only the parameters that
# differ from their defaults (see webApps/bookmark.py, which also restores
# them). _BOOKMARK_TABS, generated from the tabs' BOOKMARK_DEFAULTS, maps
# tab -> short key -> [full input id, default, derived].
_BOOKMARK_JS = r"""
(function() {
    var HOME = "Home";
    var KEEP = ["helicon_token", "helicon_theme"];

    // The form values take in the URL; the same as bookmark.encode_value.
    function encode(v) {
        if (v === true) return "1";
        if (v === false) return "0";
        if (Array.isArray(v)) return v.map(encode).join(",");
        return String(v);
    }
    // Commas, slashes and colons are left as they are: lists and file paths
    // stay readable, and the URL shorter.
    function q(text) {
        return encodeURIComponent(text)
            .replace(/%2C/g, ",").replace(/%2F/g, "/").replace(/%3A/g, ":");
    }

    // The values the bookmark being viewed set, by short key: kept in the URL,
    // until changed, even while the input is still to be built.
    var start = new URLSearchParams(window.location.search);
    var restoredTab = start.get("tab");
    var restored = {};
    if (start.has("helicon_tab")) {  // the earlier form, with a JSON blob
        try { restoredTab = JSON.parse(start.get("helicon_tab")); } catch (e) {}
        try {
            var p = JSON.parse(start.get("p") || "{}");
            for (var k in p) restored[k] = encode(p[k]);
        } catch (e) {}
    } else {
        start.forEach(function(v, k) {
            if (k !== "tab" && KEEP.indexOf(k) < 0) restored[k] = v;
        });
    }

    // For derived parameters: the value the app itself last gave each input
    // (when built, or updated by the server), and whether the user has
    // changed it since.
    var baseline = {}, setByApp = {}, touched = {};
    function bare(name) { return String(name).split(":")[0]; }
    function idOf(el) { return el && el.id ? el.id : null; }
    $(document).on("shiny:bound shiny:updateinput", function(e) {
        var id = idOf(e.target);
        if (id) setByApp[id] = Date.now();
    });

    // The latest value of each input, as Shiny reports the change: its own
    // store is updated only after the change event, so reading it there gives
    // the value before the change.
    var latest = {};

    function value(vals, id) {
        if (id in latest) return latest[id];
        if (id in vals) return vals[id];
        for (var k in vals) if (bare(k) === id) return vals[k];
        return undefined;
    }

    function build() {
        var vals = window.Shiny && Shiny.shinyapp && Shiny.shinyapp.$inputValues;
        if (!vals) return;
        var tab = value(vals, "helicon_tab");
        if (!tab) return;
        var parts = [];
        if (tab !== HOME) parts.push("tab=" + q(tab));
        var entries = _BOOKMARK_TABS[tab] || {};  // Home has none
        for (var key in entries) {
            var id = entries[key][0], def = entries[key][1], derived = entries[key][2];
            var was = tab === restoredTab ? restored[key] : undefined;
            var v = value(vals, id);
            var text;
            if (v === undefined || v === null) {
                text = was;
            } else {
                text = encode(v);
                if (derived) {
                    if (!touched[id] && text !== was) text = undefined;
                } else if (text === encode(def)) {
                    text = undefined;
                }
            }
            if (text !== undefined) parts.push(key + "=" + q(text));
        }
        var now = new URLSearchParams(window.location.search);
        KEEP.forEach(function(k) {
            if (now.has(k)) parts.push(k + "=" + q(now.get(k)));
        });
        var url = parts.length ? "?" + parts.join("&") : window.location.pathname;
        if (url !== window.location.search) window.history.replaceState(null, "", url);
    }

    var timer = null;
    $(document).on("shiny:inputchanged", function(e) {
        var id = bare(e.name), text = encode(e.value);
        latest[id] = e.value;
        if (!(id in baseline) || Date.now() - (setByApp[id] || 0) < 1500) {
            baseline[id] = text;
        } else if (text !== baseline[id]) {
            touched[id] = true;
        }
        clearTimeout(timer);
        timer = setTimeout(build, 300);
    });
})();
"""


# ── Main app UI ────────────────────────────────────────────────────


def app_ui(request: Request):
    theme = _web_theme(request)
    initial_theme = "dark" if theme in {"Dark", "System"} else "light"
    active_tab = _resolve_active_tab(request)
    _, restored = bookmark.parse(request.url.query, _TAB_MODULE_MAP)
    with bookmark.ui_restore_context(restored):
        return _page(theme, initial_theme, active_tab)


def _page(theme: str, initial_theme: str, active_tab: str):
    """The page, with every tab's UI; see app_ui."""
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
                        var themeName = dark ? 'dark' : 'light';
                        document.documentElement.dataset.heliconTheme = themeName;
                        document.documentElement.setAttribute('data-bs-theme', themeName);
                        if (document.body) {{
                            document.body.setAttribute('data-bs-theme', themeName);
                        }}
                    }}
                    if (document.readyState === 'loading') {{
                        document.addEventListener('DOMContentLoaded', applyTheme);
                    }} else {{
                        applyTheme();
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
                "var _BOOKMARK_TABS = "
                + bookmark.js_table(_TAB_MODULE_MAP)
                + ";\n"
                + _BOOKMARK_JS
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
            :root, [data-bs-theme="dark"], :root[data-helicon-theme="dark"] {{
                --helicon-page-bg: #1e1e1e;
                --helicon-text: #e0e0e0;
                --helicon-navbar-bg: #1a202c;
            }}
            [data-bs-theme="light"], :root[data-helicon-theme="light"] {{
                --helicon-page-bg: #f8f9fa;
                --helicon-text: #212529;
                --helicon-navbar-bg: #1a202c;
            }}
            html, body, .bslib-page-fill {{
                background-color: var(--helicon-page-bg) !important;
                color: var(--helicon-text);
            }}
            .navbar, .navbar-default, .navbar-inverse {{
                background-color: var(--helicon-navbar-bg) !important;
            }}
            .navbar .nav-link, .navbar-brand {{
                color: #ffffff !important;
            }}
            [data-bs-theme="dark"] .card {{
                background-color: #262626 !important;
                color: #e0e0e0 !important;
                border-color: #3d3d3d !important;
            }}
            [data-bs-theme="dark"] .card-header {{
                background-color: #1a1a1a !important;
                color: #ffffff !important;
                border-bottom: 1px solid #3d3d3d !important;
            }}
            [data-bs-theme="dark"] .form-control,
            [data-bs-theme="dark"] .form-select,
            [data-bs-theme="dark"] select,
            [data-bs-theme="dark"] input[type="text"],
            [data-bs-theme="dark"] input[type="number"],
            [data-bs-theme="dark"] textarea {{
                background-color: #2b2b2b !important;
                color: #ffffff !important;
                border-color: #4a4a4a !important;
            }}
            [data-bs-theme="dark"] label,
            [data-bs-theme="dark"] .form-label,
            [data-bs-theme="dark"] .form-check-label {{
                color: #e0e0e0 !important;
            }}
            [data-bs-theme="light"] .card {{
                background-color: #ffffff !important;
                color: #212529 !important;
                border-color: #dee2e6 !important;
            }}
            [data-bs-theme="light"] .card-header {{
                background-color: #f1f3f5 !important;
                color: #212529 !important;
                font-weight: 600;
            }}
            [data-bs-theme="light"] .form-control,
            [data-bs-theme="light"] .form-select,
            [data-bs-theme="light"] select,
            [data-bs-theme="light"] input[type="text"],
            [data-bs-theme="light"] input[type="number"],
            [data-bs-theme="light"] textarea {{
                background-color: #ffffff !important;
                color: #212529 !important;
                border-color: #ced4da !important;
            }}
            [data-bs-theme="light"] label,
            [data-bs-theme="light"] .form-label,
            [data-bs-theme="light"] .form-check-label {{
                color: #212529 !important;
            }}
            * {{ font-size: 10pt; padding: 0; border: 0; margin: 0; }}
            aside {{ --_padding-icon: 10px; }}
            html, body {{ height: 100%; margin: 0; padding: 0; overflow-x: hidden; }}
            .nav {{ padding: 0 8px; }}
            .layout-sidebar {{ gap: 4px !important; }}
            .sidebar {{ padding-right: 4px !important; }}
            .main {{ padding-left: 4px !important; }}
            body.bslib-page-fill {{ padding: 0 !important; gap: 0 !important; }}
        """
        ),
        ui.navset_bar(
            ui.nav_panel(HOME_TAB, home_tab_ui()),
            ui.nav_panel(
                "WhereIsMyClass", where_is_my_class_tab_ui("where_is_my_class")
            ),
            ui.nav_panel(
                "HelicalProjection", helical_projection_tab_ui("helical_projection")
            ),
            ui.nav_panel("HILL", hill_tab_ui("hill")),
            ui.nav_panel("HelicalPitch", helical_pitch_tab_ui("helical_pitch")),
            ui.nav_panel("Denovo3D", denovo3d_tab_ui("denovo3d")),
            ui.nav_panel("AbInitio3D", abinitio3d_tab_ui("abinitio3d")),
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
            # Open directly on the target tab. Without this the navbar starts on
            # the first tab and the bookmark script switches afterwards, so a
            # deep link paid for two tabs' worth of rendering before showing the
            # one asked for.
            selected=active_tab,
        ),
    )


def server(input, output, session):
    """Top-level server: wires shared state and delegates to tab modules."""

    _control.start_session()
    _control.start_watchdog()

    # Inputs the tabs build later (when their data arrives, or when a tab is
    # first opened) start from the bookmark's values too.
    with reactive.isolate():
        _, restored = bookmark.parse(
            session.clientdata.url_search() or "", _TAB_MODULE_MAP
        )
    bookmark.restore_in_session(session, restored)

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

    # Home has no reactive work besides starting non-integrated apps, and it
    # is the default tab, so it is wired up eagerly rather than lazily.
    home_tab_server(input, session)

    # ── Lazy tab initialisation ───────────────────────────────
    # Each tab's server function is started the first time that tab is
    # actually shown, not at session start. Seven tabs' worth of eager
    # reactive effects (EMDB lookups, default-map downloads, plot setup)
    # used to run on every page load even though only one tab is visible.

    def _start_where_is_my_class():
        # FileChooser must be built in the top-level session context;
        # ipywidgets comms fail when it is created inside a @module.server.
        from ipyfilechooser import FileChooser

        where_is_my_class_tab_server(
            "where_is_my_class",
            project,
            wimc_filechooser=FileChooser(
                path=".",
                select_desc="Select",
                show_hidden=False,
                filter_pattern=["*_data.star", "*.cs"],
                title="Select a RELION star or cryoSPARC cs file on the server",
            ),
        )

    _TAB_STARTERS: dict[str, object] = {
        "WhereIsMyClass": _start_where_is_my_class,
        "HelicalProjection": lambda: helical_projection_tab_server(
            "helical_projection", project
        ),
        "HILL": lambda: hill_tab_server("hill", project),
        "HelicalPitch": lambda: helical_pitch_tab_server("helical_pitch", project),
        "Denovo3D": lambda: denovo3d_tab_server("denovo3d", project),
        "AbInitio3D": lambda: abinitio3d_tab_server("abinitio3d", project),
        "HelicalLattice": lambda: helical_lattice_tab_server(
            "helical_lattice", project
        ),
        "HI3D": lambda: hi3d_tab_server("hi3d", project),
    }
    _started: set[str] = set()

    def _start_tab(tab: str) -> None:
        """Run a tab's server function once, ever, for this session."""
        if tab in _started or tab not in _TAB_STARTERS:
            return
        # Marked before the call, so a failure is not retried on every tab
        # switch; the tab's UI still renders, it just has no reactivity.
        _started.add(tab)
        try:
            _TAB_STARTERS[tab]()
        except Exception:
            logger.error("failed to initialise tab %s", tab, exc_info=True)

    @reactive.effect
    def _track_active_tab():
        tab = input.helicon_tab()
        if tab:
            _start_tab(tab)
            project.active_tab.set(tab)


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
