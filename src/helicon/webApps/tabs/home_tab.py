"""Home tab — helical data processing workflow with a launcher for each app.

Draws the processing steps (micrographs → particle picking → 2D classes →
initial model) as an SVG diagram and places every app next to the step it
works on. Hovering an app shows its name and a brief introduction; clicking
it switches the navbar to that app's tab.

The tab is static: it has no server function, and switching tabs happens in
the browser by clicking the matching navbar link, so the normal tab-change
path (lazy tab start, last-tab cookie, bookmark URL) runs unchanged.
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
    color: str  # placeholder icon background
    description: str
    cx: int  # icon centre in diagram coordinates
    cy: int
    icon: str | None = None  # image path under www/, e.g. "icons/hill.png"


HOME_APPS: tuple[HomeApp, ...] = (
    HomeApp(
        "HILL",
        "HILL",
        "#2563eb",
        "Helical indexing of a 2D class average using the layer lines of "
        "its Fourier power spectrum.",
        130,
        170,
    ),
    HomeApp(
        "HelicalProjection",
        "Proj",
        "#0891b2",
        "Compare 2D class averages with projections of helical 3D maps or "
        "models to find structures that match.",
        130,
        290,
    ),
    HomeApp(
        "Denovo3D",
        "3D",
        "#7c3aed",
        "Build a de novo 3D helical reconstruction from a single 2D class "
        "average, as an initial model.",
        130,
        420,
    ),
    HomeApp(
        "WhereIsMyClass",
        "WIMC",
        "#db2777",
        "Map 2D classes back onto the helical tubes/filaments in the "
        "micrographs, to see where each class came from.",
        830,
        170,
    ),
    HomeApp(
        "HelicalPitch",
        "Pitch",
        "#ea580c",
        "Estimate the helical pitch/twist from the distances between "
        "segments of the same 2D class along each filament.",
        830,
        290,
    ),
    HomeApp(
        "HI3D",
        "HI3D",
        "#16a34a",
        "Helical indexing of a 3D map using its cylindrical projection, to "
        "check or refine the twist and rise.",
        830,
        450,
    ),
    HomeApp(
        "HelicalLattice",
        "Lat",
        "#ca8a04",
        "Interconvert 2D lattices and helical lattices to explore helical "
        "symmetry.",
        220,
        560,
    ),
)

_TILE = 56  # app icon size in diagram units

# Workflow steps: (x, y, width, height, lines of text)
_STEPS = (
    (390, 20, 180, 60, ("Micrographs",)),
    (390, 130, 180, 60, ("Particle picking",)),
    (300, 250, 180, 80, ("2D class", "average images")),
    (480, 250, 180, 80, ("2D class", "metadata")),
    (390, 420, 180, 60, ("Initial model",)),
)

# Connectors between steps and apps. Arrows show data flowing into a step;
# plain lines attach an app to the data it works on.
_ARROWS = (
    "M480,80 L480,128",  # micrographs → particle picking
    "M480,190 L480,248",  # particle picking → 2D classes
    "M480,330 L480,418",  # 2D classes → initial model
    "M162,440 C230,475 300,462 388,452",  # Denovo3D → initial model
    "M830,138 C830,70 700,45 572,50",  # WhereIsMyClass → micrographs
)
_LINES = (
    "M162,178 L300,285",  # HILL
    "M162,290 L300,290",  # HelicalProjection
    "M162,412 L300,296",  # Denovo3D
    "M660,285 L798,178",  # WhereIsMyClass
    "M660,290 L798,290",  # HelicalPitch
    "M570,450 L798,450",  # HI3D
)


def _step_svg(x, y, w, h, lines) -> str:
    line_h = 18
    y0 = y + h / 2 - line_h * (len(lines) - 1) / 2
    tspans = "".join(
        f'<tspan x="{x + w / 2}" y="{y0 + i * line_h}">{escape(t)}</tspan>'
        for i, t in enumerate(lines)
    )
    return (
        f'<rect class="hh-step" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>'
        f'<text class="hh-step-text" dominant-baseline="central">{tspans}</text>'
    )


def _app_svg(app: HomeApp) -> str:
    x, y = app.cx - _TILE / 2, app.cy - _TILE / 2
    if app.icon:
        face = (
            f'<image href="{escape(app.icon)}" x="{x}" y="{y}" '
            f'width="{_TILE}" height="{_TILE}"/>'
        )
    else:
        face = (
            f'<text class="hh-abbr" x="{app.cx}" y="{app.cy}" '
            f'dominant-baseline="central">{escape(app.abbr)}</text>'
        )
    return (
        f'<g class="hh-app" tabindex="0" role="button" '
        f'aria-label="Open {escape(app.tab)}: {escape(app.description)}" '
        f'data-tab="{escape(app.tab)}" data-desc="{escape(app.description)}">'
        f'<rect class="hh-tile" x="{x}" y="{y}" width="{_TILE}" '
        f'height="{_TILE}" rx="12" style="fill: {app.color}"/>'
        f"{face}"
        f'<text class="hh-app-name" x="{app.cx}" y="{app.cy + _TILE / 2 + 16}">'
        f"{escape(app.tab)}</text>"
        f"</g>"
    )


def _workflow_svg() -> str:
    parts = [
        '<svg class="hh-diagram" viewBox="0 0 960 620" '
        'xmlns="http://www.w3.org/2000/svg" role="group" '
        'aria-label="Helical data processing workflow">',
        "<defs>"
        '<marker id="hh-arrowhead" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="hh-arrowhead" d="M0,0 L10,5 L0,10 z"/></marker>'
        "</defs>",
    ]
    parts += [f'<path class="hh-link" d="{d}"/>' for d in _LINES]
    parts += [
        f'<path class="hh-link" d="{d}" marker-end="url(#hh-arrowhead)"/>'
        for d in _ARROWS
    ]
    parts += [_step_svg(*s) for s in _STEPS]
    parts.append(
        '<text class="hh-others" x="60" y="560" dominant-baseline="central">'
        "Others:</text>"
    )
    parts += [_app_svg(a) for a in HOME_APPS]
    parts.append("</svg>")
    return "".join(parts)


_CSS = """
.helicon-home {
    --hh-text: #212529;
    --hh-muted: #6c757d;
    --hh-step-bg: #ffffff;
    --hh-step-border: #adb5bd;
    --hh-line: #6c757d;
    --hh-focus: #1f2937;
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
    --hh-line: #8a8a8a;
    --hh-focus: #ffffff;
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
    display: block; width: 100%; max-width: 960px; margin: 0 auto !important;
}
.helicon-home .hh-step {
    fill: var(--hh-step-bg); stroke: var(--hh-step-border); stroke-width: 1.5;
}
.helicon-home .hh-step-text {
    fill: var(--hh-text); font-size: 15px; text-anchor: middle;
}
.helicon-home .hh-link {
    fill: none; stroke: var(--hh-line); stroke-width: 1.5;
}
.helicon-home .hh-arrowhead { fill: var(--hh-line); }
.helicon-home .hh-others { fill: var(--hh-text); font-size: 16px; }
.helicon-home .hh-app { cursor: pointer; outline: none; }
.helicon-home .hh-tile { stroke: transparent; stroke-width: 3; }
.helicon-home .hh-app:hover .hh-tile,
.helicon-home .hh-app:focus-visible .hh-tile { stroke: var(--hh-focus); }
.helicon-home .hh-abbr {
    fill: #ffffff; font-size: 14px; font-weight: 700; text-anchor: middle;
    pointer-events: none;
}
.helicon-home .hh-app-name {
    fill: var(--hh-text); font-size: 13px; text-anchor: middle;
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
    function showTip(app) {
        var home = app.closest('.helicon-home');
        var tip = home.querySelector('.hh-tooltip');
        tip.querySelector('strong').textContent = app.dataset.tab;
        tip.querySelector('.hh-tip-desc').textContent = app.dataset.desc;
        tip.hidden = false;
        var box = home.getBoundingClientRect();
        var r = app.querySelector('.hh-tile').getBoundingClientRect();
        var left = r.left - box.left + home.scrollLeft + r.width / 2 - tip.offsetWidth / 2;
        left = Math.max(4, Math.min(left, home.clientWidth - tip.offsetWidth - 4));
        var top = r.top - box.top + home.scrollTop - tip.offsetHeight - 8;
        if (top < home.scrollTop) top = r.bottom - box.top + home.scrollTop + 8;
        tip.style.left = left + 'px';
        tip.style.top = top + 'px';
    }
    function hideTip(app) {
        app.closest('.helicon-home').querySelector('.hh-tooltip').hidden = true;
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
