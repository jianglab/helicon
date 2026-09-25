"""Home tab — helical data processing workflow with a launcher for each app.

Draws the processing steps (micrographs → particle picking → 2D classes →
3D models) as an SVG diagram, with every app wired between the data it reads
and the result it produces (pixel size, twist, rise, an initial 3D model, ...).
Hovering an app shows its name and a brief introduction and highlights its
arrows. Clicking it opens the app in one of three ways:

- an app integrated in this web app switches the navbar to its tab, by
  clicking the matching navbar link so the normal tab-change path (lazy tab
  start, bookmark URL) runs unchanged;
- an app with a helicon subcommand (``helicon procart``) is started by the
  server as a detached process and opens in a new browser window;
- an app with only a hosted site opens that site in a new browser tab.

A button above the diagram starts the helicon file browser (``helicon
display``) the same detached way; it is shown only when the server is on the
user's machine, since its window opens on the server's desktop.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from html import escape

from shiny import reactive, ui

logger = logging.getLogger(__name__)

HOME_TAB = "Home"


@dataclass(frozen=True)
class HomeApp:
    name: str  # label; for an integrated app, also the navbar value of its tab
    abbr: str  # placeholder icon text, used while ``icon`` is None
    color: str  # placeholder icon background and hover accent
    description: str
    cx: int  # chip centre in diagram coordinates
    cy: int
    icon: str | None = None  # image path under www/, e.g. "icons/hill.png"
    # Not integrated in this web app: ``command`` is a helicon subcommand to
    # start in a new window; ``url`` is its hosted site, opened instead when
    # there is no command or the server is not on the user's machine.
    command: str | None = None
    url: str | None = None

    @property
    def is_tab(self) -> bool:
        return self.command is None and self.url is None


# Diagram layout: a centre column of data at x=555, apps in columns at x=280
# and x=830, and results at the outer edges (x=80 and x=1030). Rows are
# y = 50, 160, 280, 400, 510 and 610; the Learning group fills the bottom left.
HOME_APPS: tuple[HomeApp, ...] = (
    HomeApp(
        "WebCalEM",
        "WE",
        "#64748b",
        "Calibrate the magnification (pixel size) of the microscope from the "
        "Fourier transform of micrographs of a calibration sample.",
        830,
        50,
        url="https://jianglab.github.io/WebCalEM/",
    ),
    HomeApp(
        "HelicalProjection",
        "HP",
        "#0891b2",
        "Compare 2D class averages with projections of helical 3D maps or "
        "models to find structures that match.",
        280,
        160,
    ),
    HomeApp(
        "HILL",
        "HL",
        "#2563eb",
        "Helical indexing of a 2D class average using the layer lines of "
        "its Fourier power spectrum.",
        280,
        280,
    ),
    HomeApp(
        "Denovo3D",
        "3D",
        "#7c3aed",
        "Build a de novo 3D helical reconstruction from a single 2D class "
        "average, giving an initial 3D model and its twist and rise.",
        280,
        400,
    ),
    HomeApp(
        "WhereIsMyClass",
        "WC",
        "#db2777",
        "Map 2D classes back onto the helical tubes/filaments in the "
        "micrographs, to see where each class came from.",
        830,
        160,
    ),
    HomeApp(
        "HelicalPitch",
        "Pi",
        "#ea580c",
        "Estimate the helical pitch/twist from the distances between "
        "segments of the same 2D class along each filament.",
        830,
        280,
    ),
    HomeApp(
        "HI3D",
        "H3",
        "#16a34a",
        "Helical indexing of a 3D map using its cylindrical projection, to "
        "check or refine the twist and rise.",
        830,
        400,
    ),
    HomeApp(
        "Map2seq",
        "MS",
        "#dc2626",
        "Identify the protein sequence that best explains a 3D density map, "
        "to validate the map and its atomic model.",
        830,
        510,
        # ``helicon map2seq`` only opens this hosted site, so the browser
        # opens it directly.
        url="https://map2seq.streamlit.app/",
    ),
    HomeApp(
        "ProCart",
        "PC",
        "#0d9488",
        "Plot cartoon illustrations of the residue properties of amyloid "
        "atomic models.",
        830,
        610,
        command="procart",
        url="https://jianglab.science.psu.edu/procart",
    ),
    HomeApp(
        "HelicalLattice",
        "La",
        "#ca8a04",
        "Interconvert 2D lattices and helical lattices to explore helical "
        "symmetry.",
        280,
        510,
    ),
    HomeApp(
        "CtfSimulation",
        "CT",
        "#65a30d",
        "Simulate the 1D/2D contrast transfer function (CTF) of a TEM to see "
        "how the imaging parameters shape it.",
        280,
        610,
        command="ctfSimulation",
        url="https://jianglab.science.psu.edu/ctfsimulation",
    ),
)

_CHIP_W, _CHIP_H = 176, 48
_ICON = 34

# Data: (x, y, width, height, lines of text)
_STEPS = (
    (470, 25, 170, 50, ("Micrographs",)),
    (470, 135, 170, 50, ("Particle picking",)),
    (405, 240, 150, 80, ("2D class", "average images")),
    (555, 240, 150, 80, ("2D class", "metadata")),
    (470, 485, 170, 50, ("Atomic model",)),
)

# Results: (centre x, centre y, width, text)
_RESULTS = (
    (1030, 50, 120, "Pixel size"),
    (80, 340, 120, "Twist & Rise"),
    (1030, 280, 120, "Twist"),
    (555, 400, 170, "Initial 3D model"),
    (1030, 400, 120, "Twist & Rise"),
    (1030, 510, 120, "Validation"),
    (1030, 610, 120, "Visualization"),
)
_RESULT_H = 46

# Arrows as polylines (drawn with rounded corners), each tagged with the app
# it belongs to, if any, so hovering that app can highlight it. A third
# element of True draws an arrowhead at both ends.
_ARROWS: tuple[tuple, ...] = (
    (None, ((555, 75), (555, 133))),  # micrographs → particle picking
    (None, ((555, 185), (555, 238))),  # particle picking → 2D classes
    ("WebCalEM", ((642, 50), (740, 50))),
    ("WebCalEM", ((918, 50), (968, 50))),
    ("HelicalProjection", ((445, 240), (445, 212), (280, 212), (280, 186))),
    ("HILL", ((405, 280), (370, 280))),
    ("HILL", ((192, 280), (80, 280), (80, 315))),
    ("Denovo3D", ((445, 320), (445, 348), (280, 348), (280, 374))),
    ("Denovo3D", ((368, 400), (468, 400))),
    ("Denovo3D", ((192, 400), (80, 400), (80, 365))),
    ("WhereIsMyClass", ((640, 160), (740, 160))),
    ("WhereIsMyClass", ((665, 240), (665, 212), (830, 212), (830, 186))),
    ("WhereIsMyClass", ((800, 136), (800, 105), (610, 105), (610, 77))),
    ("HelicalPitch", ((705, 280), (740, 280))),
    ("HelicalPitch", ((918, 280), (968, 280))),
    ("HI3D", ((640, 400), (740, 400))),
    ("HI3D", ((918, 400), (968, 400))),
    ("Map2seq", ((620, 423), (620, 455), (800, 455), (800, 484))),
    ("Map2seq", ((640, 510), (740, 510))),
    ("Map2seq", ((918, 510), (968, 510))),
    ("ProCart", ((600, 535), (600, 610), (740, 610))),
    ("ProCart", ((918, 610), (968, 610))),
)


def _rounded_path(points, r: float = 10) -> str:
    """SVG path through ``points`` with each corner rounded by radius ``r``."""
    (x0, y0), *rest = points
    d = [f"M{x0},{y0}"]
    for (xa, ya), (xb, yb), (xc, yc) in zip(points, points[1:], points[2:]):
        # Stop short of the corner, then curve onto the next segment.
        ra = min(r, (abs(xb - xa) + abs(yb - ya)) / 2)
        rc = min(r, (abs(xc - xb) + abs(yc - yb)) / 2)
        sx = (xb > xa) - (xb < xa)
        sy = (yb > ya) - (yb < ya)
        tx = (xc > xb) - (xc < xb)
        ty = (yc > yb) - (yc < yb)
        d.append(f"L{xb - sx * ra},{yb - sy * ra}")
        d.append(f"Q{xb},{yb} {xb + tx * rc},{yb + ty * rc}")
    xn, yn = rest[-1]
    d.append(f"L{xn},{yn}")
    return " ".join(d)


def _text_lines(cx, cy, lines, cls) -> str:
    line_h = 18
    y0 = cy - line_h * (len(lines) - 1) / 2
    tspans = "".join(
        f'<tspan x="{cx}" y="{y0 + i * line_h}">{escape(t)}</tspan>'
        for i, t in enumerate(lines)
    )
    return f'<text class="{cls}" dominant-baseline="central">{tspans}</text>'


def _step_svg(x, y, w, h, lines) -> str:
    return (
        f'<rect class="hh-step" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>'
        + _text_lines(x + w / 2, y + h / 2, lines, "hh-step-text")
    )


def _result_svg(cx, cy, w, text) -> str:
    return (
        f'<rect class="hh-result" x="{cx - w / 2}" y="{cy - _RESULT_H / 2}" '
        f'width="{w}" height="{_RESULT_H}" rx="{_RESULT_H / 2}"/>'
        + _text_lines(cx, cy, (text,), "hh-result-text")
    )


def _app_svg(app: HomeApp) -> str:
    x, y = app.cx - _CHIP_W / 2, app.cy - _CHIP_H / 2
    ix, iy = x + 7, app.cy - _ICON / 2
    if app.icon:
        face = (
            f'<image href="{escape(app.icon)}" x="{ix}" y="{iy}" '
            f'width="{_ICON}" height="{_ICON}"/>'
        )
    else:
        face = (
            f'<rect x="{ix}" y="{iy}" width="{_ICON}" height="{_ICON}" rx="8" '
            f'style="fill: {app.color}"/>'
            f'<text class="hh-abbr" x="{ix + _ICON / 2}" y="{app.cy}" '
            f'dominant-baseline="central">{escape(app.abbr)}</text>'
        )
    if app.is_tab:
        hint, data, external = "Click to open", f' data-tab="{escape(app.name)}"', ""
    else:
        if app.command:
            hint = f"Click to open in a new window (helicon {app.command})"
            data = f' data-command="{escape(app.command)}"'
        else:
            hint, data = "Click to open the hosted site in a new tab", ""
        if app.url:
            data += f' data-url="{escape(app.url)}"'
        external = (
            f'<text class="hh-external" x="{x + _CHIP_W - 12}" y="{app.cy}" '
            f'dominant-baseline="central">↗</text>'
        )
    return (
        f'<g class="hh-app" tabindex="0" role="button" '
        f'style="--hh-app: {app.color}" '
        f'aria-label="Open {escape(app.name)}: {escape(app.description)}" '
        f'data-name="{escape(app.name)}" data-desc="{escape(app.description)}" '
        f'data-hint="{escape(hint)}"{data}>'
        f'<rect class="hh-chip" x="{x}" y="{y}" width="{_CHIP_W}" '
        f'height="{_CHIP_H}" rx="10"/>'
        f"{face}"
        f'<text class="hh-app-name" x="{ix + _ICON + 9}" y="{app.cy}" '
        f'dominant-baseline="central">{escape(app.name)}</text>'
        f"{external}"
        f"</g>"
    )


def _legend_svg() -> str:
    y = 50
    return (
        f'<rect class="hh-step" x="20" y="{y - 8}" width="22" height="16" rx="3"/>'
        f'<text class="hh-legend" x="50" y="{y}" dominant-baseline="central">'
        "Data</text>"
        f'<rect class="hh-chip" x="100" y="{y - 8}" width="22" height="16" rx="4"/>'
        f'<text class="hh-legend" x="130" y="{y}" dominant-baseline="central">'
        "App (click to open)</text>"
        f'<rect class="hh-result" x="262" y="{y - 8}" width="22" height="16" '
        f'rx="8"/><text class="hh-legend" x="292" y="{y}" '
        f'dominant-baseline="central">Result</text>'
    )


def _learning_group_svg() -> str:
    return (
        '<rect class="hh-group" x="170" y="450" width="220" height="200" rx="12"/>'
        '<text class="hh-group-title" x="186" y="469" dominant-baseline="central">'
        "Learning</text>"
    )


def _workflow_svg() -> str:
    parts = [
        '<svg class="hh-diagram" viewBox="0 0 1110 670" '
        'xmlns="http://www.w3.org/2000/svg" role="group" '
        'aria-label="Helical data processing workflow">',
        "<defs>",
        *(
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path class="{cls}" d="M0,0 L10,5 L0,10 z"/></marker>'
            for mid, cls in (
                ("hh-arrowhead", "hh-arrowhead"),
                ("hh-arrowhead-hot", "hh-arrowhead hh-hot"),
            )
        ),
        "</defs>",
        _learning_group_svg(),
    ]
    for app, points, *both in _ARROWS:
        cls = "hh-link hh-both" if both and both[0] else "hh-link"
        data = f' data-app="{escape(app)}"' if app else ""
        parts.append(f'<path class="{cls}"{data} d="{_rounded_path(points)}"/>')
    parts += [_step_svg(*s) for s in _STEPS]
    parts += [_result_svg(*r) for r in _RESULTS]
    parts += [_app_svg(a) for a in HOME_APPS]
    parts.append(_legend_svg())
    parts.append("</svg>")
    return "".join(parts)


_CSS = """
.helicon-home {
    --hh-text: #212529;
    --hh-muted: #6c757d;
    --hh-step-bg: #f1f3f5;
    --hh-step-border: #adb5bd;
    --hh-chip-bg: #ffffff;
    --hh-chip-border: #ced4da;
    --hh-result-bg: #ecfdf5;
    --hh-result-border: #10b981;
    --hh-result-text: #065f46;
    --hh-group-border: #ced4da;
    --hh-line: #868e96;
    --hh-tip-bg: #ffffff;
    --hh-tip-border: #ced4da;
    position: relative;
    height: 100%;
    overflow: auto;
    padding: 16px !important;
    color: var(--hh-text);
}
[data-bs-theme="dark"] .helicon-home {
    --hh-text: #e0e0e0;
    --hh-muted: #a0a0a0;
    --hh-step-bg: #262626;
    --hh-step-border: #5a5a5a;
    --hh-chip-bg: #2b2b2b;
    --hh-chip-border: #4a4a4a;
    --hh-result-bg: #10302a;
    --hh-result-border: #34d399;
    --hh-result-text: #a7f3d0;
    --hh-group-border: #4a4a4a;
    --hh-line: #7a7a7a;
    --hh-tip-bg: #2b2b2b;
    --hh-tip-border: #4a4a4a;
}
.helicon-home .hh-title {
    font-size: 16pt; font-weight: 600; text-align: center;
}
.helicon-home .hh-subtitle {
    color: var(--hh-muted); text-align: center; margin: 4px 0 8px !important;
}
.helicon-home .hh-toolbar { text-align: center; margin-bottom: 8px !important; }
.helicon-home .hh-toolbar[hidden] { display: none; }
.helicon-home .hh-diagram {
    display: block; width: 100%; max-width: 1110px; margin: 0 auto !important;
}
.helicon-home .hh-step {
    fill: var(--hh-step-bg); stroke: var(--hh-step-border); stroke-width: 1.5;
}
.helicon-home .hh-step-text {
    fill: var(--hh-text); font-size: 15px; text-anchor: middle;
}
.helicon-home .hh-result {
    fill: var(--hh-result-bg); stroke: var(--hh-result-border); stroke-width: 1.5;
}
.helicon-home .hh-result-text {
    fill: var(--hh-result-text); font-size: 14px; font-weight: 600;
    text-anchor: middle;
}
.helicon-home .hh-group {
    fill: none; stroke: var(--hh-group-border); stroke-width: 1.5;
    stroke-dasharray: 6 4;
}
.helicon-home .hh-group-title {
    fill: var(--hh-muted); font-size: 13px; font-weight: 600;
}
.helicon-home .hh-link {
    fill: none; stroke: var(--hh-line); stroke-width: 1.5;
    marker-end: url(#hh-arrowhead);
    transition: stroke 0.15s;
}
.helicon-home .hh-link.hh-both { marker-start: url(#hh-arrowhead); }
.helicon-home .hh-arrowhead { fill: var(--hh-line); }
.helicon-home .hh-link.hh-hot {
    stroke: var(--hh-hot); stroke-width: 2.5; marker-end: url(#hh-arrowhead-hot);
}
.helicon-home .hh-link.hh-both.hh-hot { marker-start: url(#hh-arrowhead-hot); }
.helicon-home .hh-arrowhead.hh-hot { fill: var(--hh-hot); }
.helicon-home .hh-legend { fill: var(--hh-muted); font-size: 12px; }
.helicon-home .hh-app { cursor: pointer; outline: none; }
.helicon-home .hh-chip {
    fill: var(--hh-chip-bg); stroke: var(--hh-chip-border); stroke-width: 1.5;
    transition: stroke 0.15s;
}
.helicon-home .hh-app:hover .hh-chip,
.helicon-home .hh-app:focus-visible .hh-chip {
    stroke: var(--hh-app); stroke-width: 2.5;
}
.helicon-home .hh-abbr {
    fill: #ffffff; font-size: 13px; font-weight: 700; text-anchor: middle;
    pointer-events: none;
}
.helicon-home .hh-app-name {
    fill: var(--hh-text); font-size: 13px; font-weight: 600;
}
.helicon-home .hh-external {
    fill: var(--hh-muted); font-size: 13px; text-anchor: middle;
}
.helicon-home .hh-tooltip {
    position: absolute; z-index: 10; max-width: 260px; pointer-events: none;
    padding: 8px 10px !important; border-radius: 6px;
    background: var(--hh-tip-bg); border: 1px solid var(--hh-tip-border);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}
.helicon-home .hh-tooltip[hidden] { display: none; }
.helicon-home .hh-tooltip strong { display: block; margin-bottom: 2px !important; }
.helicon-home .hh-tooltip .hh-tip-hint {
    color: var(--hh-muted); font-size: 9pt; margin-top: 4px !important;
}
"""

# Hostnames that mean the server runs on the user's own machine, so a
# subcommand it starts opens its window in front of that user.
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]")

# Started by the "Open file browser" button rather than by a diagram app.
_DISPLAY_COMMAND = "display"

# Delegated on document so it does not depend on when the tab is rendered.
_JS = (
    "var _HH_LOCAL_HOSTS = %s;\n" % list(_LOCAL_HOSTS)
    + """
(function() {
    var local = _HH_LOCAL_HOSTS.indexOf(location.hostname) >= 0;
    // The file browser opens on the server's desktop, so only offer it there.
    document.querySelectorAll('.helicon-home .hh-toolbar')
        .forEach(function(bar) { bar.hidden = !local; });
    function appOf(el) { return el && el.closest ? el.closest('.helicon-home .hh-app') : null; }
    function launch(app) {
        var d = app.dataset;
        if (d.tab) {
            var link = document.querySelector(
                '.navbar a.nav-link[data-value="' + d.tab + '"]');
            if (link) link.click();
        } else if (d.command && local) {
            Shiny.setInputValue('home_launch', d.command, {priority: 'event'});
        } else if (d.url) {
            // Opened here, inside the click, so popup blockers allow it.
            window.open(d.url, '_blank', 'noopener');
        }
    }
    function setHot(app, on) {
        var home = app.closest('.helicon-home');
        home.style.setProperty('--hh-hot', app.style.getPropertyValue('--hh-app'));
        home.querySelectorAll('.hh-link[data-app="' + app.dataset.name + '"]')
            .forEach(function(p) { p.classList.toggle('hh-hot', on); });
    }
    function showTip(app) {
        var home = app.closest('.helicon-home');
        var tip = home.querySelector('.hh-tooltip');
        tip.querySelector('strong').textContent = app.dataset.name;
        tip.querySelector('.hh-tip-desc').textContent = app.dataset.desc;
        tip.querySelector('.hh-tip-hint').textContent = app.dataset.hint;
        tip.hidden = false;
        setHot(app, true);
        var box = home.getBoundingClientRect();
        var r = app.querySelector('.hh-chip').getBoundingClientRect();
        var left = r.left - box.left + home.scrollLeft + r.width / 2 - tip.offsetWidth / 2;
        left = Math.max(4, Math.min(left, home.clientWidth - tip.offsetWidth - 4));
        // Open away from the diagram's middle, where most arrows run.
        var svg = home.querySelector('.hh-diagram').getBoundingClientRect();
        var below = r.top + r.height / 2 > svg.top + svg.height / 2;
        var top = r.top - box.top + home.scrollTop - tip.offsetHeight - 8;
        if (below || top < home.scrollTop) top = r.bottom - box.top + home.scrollTop + 8;
        tip.style.left = left + 'px';
        tip.style.top = top + 'px';
    }
    function hideTip(app) {
        app.closest('.helicon-home').querySelector('.hh-tooltip').hidden = true;
        setHot(app, false);
    }
    document.addEventListener('mouseover', function(e) {
        var app = appOf(e.target);
        if (app && !app.contains(e.relatedTarget)) showTip(app);
    });
    document.addEventListener('mouseout', function(e) {
        var app = appOf(e.target);
        if (app && !app.contains(e.relatedTarget)) hideTip(app);
    });
    document.addEventListener('focusin', function(e) {
        var app = appOf(e.target);
        if (app) showTip(app);
    });
    document.addEventListener('focusout', function(e) {
        var app = appOf(e.target);
        if (app) hideTip(app);
    });
    document.addEventListener('click', function(e) {
        var display = e.target.closest && e.target.closest('.helicon-home .hh-display');
        if (display && local) {
            Shiny.setInputValue('home_launch', display.dataset.command, {priority: 'event'});
            return;
        }
        var app = appOf(e.target);
        if (app) { hideTip(app); launch(app); }
    });
    document.addEventListener('keydown', function(e) {
        var app = appOf(e.target);
        if (app && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            hideTip(app);
            launch(app);
        }
    });
})();
"""
)


def home_tab_ui():
    return ui.div(
        ui.tags.style(_CSS),
        ui.div("Helical data processing workflow", class_="hh-title"),
        ui.div(
            "Hover over an app to see what it does; click it to open the app.",
            class_="hh-subtitle",
        ),
        ui.div(
            ui.tags.button(
                "Open file browser",
                type="button",
                class_="btn btn-sm btn-outline-secondary hh-display",
                title="Browse a folder and view its images, maps, STAR files and "
                f"more in a new window (helicon {_DISPLAY_COMMAND})",
                data_command=_DISPLAY_COMMAND,
            ),
            class_="hh-toolbar",
            # Shown by the script once it knows the server is on this machine.
            hidden=True,
        ),
        ui.HTML(_workflow_svg()),
        ui.div(
            ui.tags.strong(),
            ui.div(class_="hh-tip-desc"),
            ui.div(class_="hh-tip-hint"),
            class_="hh-tooltip",
            hidden=True,
            role="tooltip",
        ),
        ui.tags.script(_JS),
        class_="helicon-home",
    )


_COMMAND_APPS = {a.command: a for a in HOME_APPS if a.command}
# Every command Home may start, by the name its notifications use.
_COMMAND_NAMES = {c: a.name for c, a in _COMMAND_APPS.items()}
_COMMAND_NAMES[_DISPLAY_COMMAND] = "the file browser"
_RELAUNCH_SECS = 5.0  # ignore a repeat click (e.g. a double click) this soon


def home_tab_server(input, session) -> None:
    """Start the subcommand of a Home app that is not integrated here, or the
    file browser."""
    last_launch: dict[str, float] = {}

    @reactive.effect
    @reactive.event(input.home_launch)
    def _launch_command():
        command = input.home_launch()
        # Only commands listed on Home, never an arbitrary client string.
        name = _COMMAND_NAMES.get(command)
        if name is None:
            return
        # The browser only asks when it is on this machine, but it is the
        # server that must not start processes for a remote visitor.
        if session.clientdata.url_hostname() not in _LOCAL_HOSTS:
            return
        now = time.monotonic()
        if now - last_launch.get(command, float("-inf")) < _RELAUNCH_SECS:
            return
        last_launch[command] = now

        import helicon
        from helicon.helicon import streamlit_commands
        from helicon.lib.terminal import _spawn_detached

        # helicon hides a subcommand whose GUI stack is missing, so the spawned
        # process would exit at once without any window.
        if command == _DISPLAY_COMMAND and not helicon.has_napari():
            ui.notification_show(
                "The file browser needs napari and PySide6 "
                '(pip install "helicon[gui]").',
                type="warning",
                duration=15,
            )
            return
        if command in streamlit_commands and not helicon.has_streamlit():
            app = _COMMAND_APPS[command]
            ui.notification_show(
                ui.span(
                    f"{app.name} needs Streamlit (pip install streamlit). "
                    "Meanwhile, use the ",
                    ui.tags.a("hosted version", href=app.url, target="_blank"),
                    ".",
                ),
                type="warning",
                duration=15,
            )
            return

        # Same as running ``helicon <command>`` in a terminal.
        started = _spawn_detached(
            [
                sys.executable,
                "-c",
                "import sys; from helicon.helicon import main; sys.exit(main())",
                command,
            ]
        )
        if started:
            where = "window" if command == _DISPLAY_COMMAND else "browser window"
            ui.notification_show(
                f"Starting {name} (helicon {command}); it will open in a "
                f"new {where}.",
                duration=8,
            )
        else:
            logger.error("failed to launch helicon %s", command)
            ui.notification_show(
                f"Failed to start {name}. Try running `helicon {command}` "
                "in a terminal.",
                type="error",
                duration=10,
            )
