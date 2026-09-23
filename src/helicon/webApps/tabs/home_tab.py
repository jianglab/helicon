"""Home tab — helical data processing workflow with a launcher for each app.

Draws the processing steps (micrographs → particle picking → 2D classes →
initial model) as an SVG diagram, with every app wired between the data it
reads and the result it produces (twist, rise, an initial model, ...).
Hovering an app shows its name and a brief introduction and highlights its
arrows; clicking it switches the navbar to that app's tab.

The tab is static: it has no server function, and switching tabs happens in
the browser by clicking the matching navbar link, so the normal tab-change
path (lazy tab start, bookmark URL) runs unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

from shiny import ui

HOME_TAB = "Home"


@dataclass(frozen=True)
class HomeApp:
    tab: str  # navbar value of the app's tab
    abbr: str  # placeholder icon text, used while ``icon`` is None
    color: str  # placeholder icon background and hover accent
    description: str
    cx: int  # chip centre in diagram coordinates
    cy: int
    icon: str | None = None  # image path under www/, e.g. "icons/hill.png"


# Diagram layout: a centre column of data steps at x=555, apps in columns at
# x=280 and x=830, and results at the outer edges. Rows are y = 50, 170,
# 300 and 430; "Others" sits below at y=560.
HOME_APPS: tuple[HomeApp, ...] = (
    HomeApp(
        "HelicalProjection",
        "HP",
        "#0891b2",
        "Compare 2D class averages with projections of helical 3D maps or "
        "models to find structures that match.",
        280,
        170,
    ),
    HomeApp(
        "HILL",
        "HL",
        "#2563eb",
        "Helical indexing of a 2D class average using the layer lines of "
        "its Fourier power spectrum.",
        280,
        300,
    ),
    HomeApp(
        "Denovo3D",
        "3D",
        "#7c3aed",
        "Build a de novo 3D helical reconstruction from a single 2D class "
        "average, giving an initial model and its twist and rise.",
        280,
        430,
    ),
    HomeApp(
        "WhereIsMyClass",
        "WC",
        "#db2777",
        "Map 2D classes back onto the helical tubes/filaments in the "
        "micrographs, to see where each class came from.",
        830,
        170,
    ),
    HomeApp(
        "HelicalPitch",
        "Pi",
        "#ea580c",
        "Estimate the helical pitch/twist from the distances between "
        "segments of the same 2D class along each filament.",
        830,
        300,
    ),
    HomeApp(
        "HI3D",
        "H3",
        "#16a34a",
        "Helical indexing of a 3D map using its cylindrical projection, to "
        "check or refine the twist and rise.",
        830,
        430,
    ),
    HomeApp(
        "HelicalLattice",
        "La",
        "#ca8a04",
        "Interconvert 2D lattices and helical lattices to explore helical "
        "symmetry.",
        280,
        560,
    ),
)

_CHIP_W, _CHIP_H = 176, 48
_ICON = 34

# Data steps: (x, y, width, height, lines of text)
_STEPS = (
    (470, 25, 170, 50, ("Micrographs",)),
    (470, 145, 170, 50, ("Particle picking",)),
    (405, 260, 150, 80, ("2D class", "average images")),
    (555, 260, 150, 80, ("2D class", "metadata")),
    (470, 405, 170, 50, ("Initial model",)),
)

# Results: (centre x, centre y, text)
_RESULTS = (
    (80, 365, "Twist & Rise"),
    (1030, 300, "Twist"),
    (1030, 430, "Twist & Rise"),
)
_RESULT_W, _RESULT_H = 120, 46

# Arrows as polylines (drawn with rounded corners), each tagged with the app
# it belongs to, if any, so hovering that app can highlight it.
_ARROWS: tuple[tuple[str | None, tuple[tuple[int, int], ...]], ...] = (
    (None, ((555, 75), (555, 143))),  # micrographs → particle picking
    (None, ((555, 195), (555, 258))),  # particle picking → 2D classes
    (None, ((555, 340), (555, 403))),  # 2D classes → initial model
    ("HelicalProjection", ((445, 260), (445, 222), (280, 222), (280, 196))),
    ("HILL", ((405, 300), (370, 300))),
    ("HILL", ((192, 300), (80, 300), (80, 340))),
    ("Denovo3D", ((445, 340), (445, 378), (280, 378), (280, 404))),
    ("Denovo3D", ((368, 430), (468, 430))),
    ("Denovo3D", ((192, 430), (80, 430), (80, 390))),
    ("WhereIsMyClass", ((640, 170), (740, 170))),
    ("WhereIsMyClass", ((665, 260), (665, 222), (830, 222), (830, 196))),
    ("WhereIsMyClass", ((830, 146), (830, 50), (642, 50))),
    ("HelicalPitch", ((705, 300), (740, 300))),
    ("HelicalPitch", ((918, 300), (968, 300))),
    ("HI3D", ((640, 430), (740, 430))),
    ("HI3D", ((918, 430), (968, 430))),
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


def _result_svg(cx, cy, text) -> str:
    return (
        f'<rect class="hh-result" x="{cx - _RESULT_W / 2}" '
        f'y="{cy - _RESULT_H / 2}" width="{_RESULT_W}" height="{_RESULT_H}" '
        f'rx="{_RESULT_H / 2}"/>' + _text_lines(cx, cy, (text,), "hh-result-text")
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
    return (
        f'<g class="hh-app" tabindex="0" role="button" '
        f'style="--hh-app: {app.color}" '
        f'aria-label="Open {escape(app.tab)}: {escape(app.description)}" '
        f'data-tab="{escape(app.tab)}" data-desc="{escape(app.description)}">'
        f'<rect class="hh-chip" x="{x}" y="{y}" width="{_CHIP_W}" '
        f'height="{_CHIP_H}" rx="10"/>'
        f"{face}"
        f'<text class="hh-app-name" x="{ix + _ICON + 9}" y="{app.cy}" '
        f'dominant-baseline="central">{escape(app.tab)}</text>'
        f"</g>"
    )


def _legend_svg() -> str:
    y = 560
    return (
        f'<rect class="hh-step" x="660" y="{y - 8}" width="22" height="16" rx="3"/>'
        f'<text class="hh-legend" x="690" y="{y}" dominant-baseline="central">'
        "Data</text>"
        f'<rect class="hh-chip" x="750" y="{y - 8}" width="22" height="16" rx="4"/>'
        f'<text class="hh-legend" x="780" y="{y}" dominant-baseline="central">'
        "App (click to open)</text>"
        f'<rect class="hh-result" x="920" y="{y - 8}" width="22" height="16" '
        f'rx="8"/><text class="hh-legend" x="950" y="{y}" '
        f'dominant-baseline="central">Result</text>'
    )


def _workflow_svg() -> str:
    parts = [
        '<svg class="hh-diagram" viewBox="0 0 1110 600" '
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
    ]
    for app, points in _ARROWS:
        data = f' data-app="{escape(app)}"' if app else ""
        parts.append(f'<path class="hh-link"{data} d="{_rounded_path(points)}"/>')
    parts += [_step_svg(*s) for s in _STEPS]
    parts += [_result_svg(*r) for r in _RESULTS]
    parts.append(
        '<text class="hh-others" x="180" y="560" dominant-baseline="central">'
        "Others:</text>"
    )
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
.helicon-home .hh-link {
    fill: none; stroke: var(--hh-line); stroke-width: 1.5;
    marker-end: url(#hh-arrowhead);
    transition: stroke 0.15s;
}
.helicon-home .hh-arrowhead { fill: var(--hh-line); }
.helicon-home .hh-link.hh-hot {
    stroke: var(--hh-hot); stroke-width: 2.5; marker-end: url(#hh-arrowhead-hot);
}
.helicon-home .hh-arrowhead.hh-hot { fill: var(--hh-hot); }
.helicon-home .hh-others { fill: var(--hh-text); font-size: 15px; text-anchor: end; }
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

# Delegated on document so it does not depend on when the tab is rendered.
_JS = """
(function() {
    function appOf(el) { return el && el.closest ? el.closest('.helicon-home .hh-app') : null; }
    function openTab(name) {
        var link = document.querySelector(
            '.navbar a.nav-link[data-value="' + name + '"]');
        if (link) link.click();
    }
    function setHot(app, on) {
        var home = app.closest('.helicon-home');
        home.style.setProperty('--hh-hot', app.style.getPropertyValue('--hh-app'));
        home.querySelectorAll('.hh-link[data-app="' + app.dataset.tab + '"]')
            .forEach(function(p) { p.classList.toggle('hh-hot', on); });
    }
    function showTip(app) {
        var home = app.closest('.helicon-home');
        var tip = home.querySelector('.hh-tooltip');
        tip.querySelector('strong').textContent = app.dataset.tab;
        tip.querySelector('.hh-tip-desc').textContent = app.dataset.desc;
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
        var app = appOf(e.target);
        if (app) { hideTip(app); openTab(app.dataset.tab); }
    });
    document.addEventListener('keydown', function(e) {
        var app = appOf(e.target);
        if (app && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            hideTip(app);
            openTab(app.dataset.tab);
        }
    });
})();
"""


def home_tab_ui():
    return ui.div(
        ui.tags.style(_CSS),
        ui.div("Helical data processing workflow", class_="hh-title"),
        ui.div(
            "Hover over an app to see what it does; click it to open the app.",
            class_="hh-subtitle",
        ),
        ui.HTML(_workflow_svg()),
        ui.div(
            ui.tags.strong(),
            ui.div(class_="hh-tip-desc"),
            ui.div("Click to open", class_="hh-tip-hint"),
            class_="hh-tooltip",
            hidden=True,
            role="tooltip",
        ),
        ui.tags.script(_JS),
        class_="helicon-home",
    )
