"""HelicalLattice tab — 2D lattice <=> helical lattice interconversion.

Adapted from HelicalLattice.git (Streamlit reference) and HelicalLattice_shiny.git.
Wrapped as a Shiny module using render.ui + pio.to_html().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from shiny import reactive, render, ui
from shiny import module

from ..lib.shared_state import ProjectState

BOOKMARK_DEFAULTS = {
    "mode": ("radio", "Helical\u21d22D"),
    "twist": ("twist", -81.1),
    "rise": ("rise", 19.4),
    "csym": ("csym", 1),
    "diameter": ("diameter", 290.0),
    "length": ("length", 1000.0),
    "primitive_unitcell": ("primitive_unitcell", False),
    "horizontal": ("horizontal", True),
    "lattice_size_factor": ("lattice_size_factor", 1.25),
    "marker_size": ("marker_size", 5.0),
    "figure_height": ("figure_height", 800),
    "ax": ("ax", 34.65),
    "ay": ("ay", 0.0),
    "bx": ("bx", 10.63),
    "by": ("by", -23.01),
    "na": ("na", 16),
    "nb": ("nb", 1),
}


def _fig_to_html(fig):
    """Convert a plotly figure to responsive HTML for rendering via render.ui."""
    html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={"responsive": True, "displayModeBar": False},
    )
    return ui.HTML(html)


# ── Conversion functions (verbatim from HelicalLattice_shiny) ───


def convert_2d_lattice_to_helical_lattice(a=(1, 0), b=(0, 1), endpoint=(10, 0)):
    def set_to_periodic_range(v, min=-180, max=180):
        from math import fmod

        tmp = fmod(v - min, max - min)
        if tmp >= 0:
            tmp += min
        else:
            tmp += max
        return tmp

    def length(v):
        return np.linalg.norm(v)

    def transform_vector(v, vref=(1, 0)):
        ang = np.arctan2(vref[1], vref[0])
        cos = np.cos(ang)
        sin = np.sin(ang)
        m = [[cos, sin], [-sin, cos]]
        v2 = np.dot(m, v.T)
        return v2

    def on_equator(v, epsilon=0.5):
        if abs(v[1]) > epsilon:
            return 0
        return 1

    a, b, endpoint = map(np.array, (a, b, endpoint))
    na, nb = endpoint
    v_equator = na * a + nb * b
    circumference = length(v_equator)
    va = transform_vector(a, v_equator)
    vb = transform_vector(b, v_equator)
    minLength = max(1.0, min(np.linalg.norm(va), np.linalg.norm(vb)) * 0.9)
    vs_on_equator = []
    vs_off_equator = []
    epsilon = 0.5
    maxI = 10
    for i in range(-maxI, maxI + 1):
        for j in range(-maxI, maxI + 1):
            if i or j:
                v = i * va + j * vb
                v[0] = set_to_periodic_range(v[0], min=0, max=circumference)
                if np.linalg.norm(v) > minLength:
                    if v[1] < 0:
                        v *= -1
                    if on_equator(v, epsilon=epsilon):
                        vs_on_equator.append(v)
                    else:
                        vs_off_equator.append(v)
    twist, rise, csym = 0, 0, 1
    if vs_on_equator:
        vs_on_equator.sort(key=lambda v: abs(v[0]))
        best_spacing = abs(vs_on_equator[0][0])
        csym_f = circumference / best_spacing
        expected_spacing = circumference / round(csym_f)
        if abs(best_spacing - expected_spacing) / expected_spacing < 0.05:
            csym = int(round(csym_f))
    if vs_off_equator:
        vs_off_equator.sort(key=lambda v: (abs(round(v[1] / epsilon)), abs(v[0])))
        twist, rise = vs_off_equator[0]
        twist *= 360 / circumference
        twist = set_to_periodic_range(
            twist, min=-360 / (2 * csym), max=360 / (2 * csym)
        )
    diameter = circumference / np.pi
    return twist, rise, csym, diameter


def convert_helical_lattice_to_2d_lattice(
    twist=30, rise=20, csym=1, diameter=100, primitive_unitcell=False, horizontal=True
):
    def angle90(v1, v2):
        p = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        p = np.clip(abs(p), 0, 1)
        ret = np.rad2deg(np.arccos(p))
        return ret

    def transform_vector(v, vref=(1, 0)):
        ang = np.arctan2(vref[1], vref[0])
        cos = np.cos(ang)
        sin = np.sin(ang)
        m = [[cos, sin], [-sin, cos]]
        v2 = np.dot(m, v.T)
        return v2

    imax = int(5 * 360 / abs(twist))
    n = np.tile(np.arange(-imax, imax), reps=(2, 1)).T
    v = np.array([twist, rise], dtype=float) * n
    if csym > 1:
        vs = []
        for ci in range(csym):
            tmp = v * 1.0
            tmp[:, 0] += ci / csym * 360
            vs.append(tmp)
        v = np.vstack(vs)
    v[:, 0] = np.fmod(v[:, 0], 360)
    v[v[:, 0] < 0, 0] += 360
    v[:, 0] *= np.pi * diameter / 360
    dist = np.linalg.norm(v, axis=1)
    dist_indices = np.argsort(dist)
    v = v[dist_indices]
    err = 1.0
    vb = v[1]
    for i in range(1, len(v)):
        if angle90(vb, v[i]) > err:
            va = v[i]
            break
    ve = np.array([np.pi * diameter, 0])
    m = np.vstack((va, vb)).T
    na, nb = np.linalg.solve(m, ve)
    endpoint = (round(na), round(nb))

    if not primitive_unitcell:
        vabs = []
        for ia in range(-1, 2):
            for ib in range(-1, 2):
                vabs.append(ia * va + ib * vb)
        vabs_good = []
        area = np.linalg.norm(np.cross(va, vb))
        for vai, vatmp in enumerate(vabs):
            for vbi in range(vai + 1, len(vabs)):
                vbtmp = vabs[vbi]
                areatmp = np.linalg.norm(np.cross(vatmp, vbtmp))
                if abs(areatmp - area) > err:
                    continue
                vabs_good.append((vatmp, vbtmp))
        dist = []
        for vi, (vatmp, vbtmp) in enumerate(vabs_good):
            m = np.vstack((vatmp, vbtmp)).T
            na, nb = np.linalg.solve(m, ve)
            if abs(na - round(na)) > 1e-3:
                continue
            if abs(nb - round(nb)) > 1e-3:
                continue
            dist.append(
                (
                    abs(na) + abs(nb),
                    -round(na),
                    -round(nb),
                    round(na),
                    round(nb),
                    vatmp,
                    vbtmp,
                )
            )
        if len(dist):
            dist.sort(key=lambda x: x[:3])
            na, nb, va, vb = dist[0][3:]
            if np.linalg.norm(vb) > np.linalg.norm(va):
                va, vb = vb, va
                na, nb = nb, na
            endpoint = (na, nb)

    if va[0] < 0:
        va *= -1
        vb *= -1
        na *= -1
        nb *= -1

    if horizontal:
        vb = transform_vector(vb, vref=va)
        va = np.array([np.linalg.norm(va), 0.0])

    return va, vb, endpoint


# ── Plotly figure builders (verbatim from HelicalLattice_shiny) ─


def plot_2d_lattice(
    a=(1, 0),
    b=(0, 1),
    endpoint=(10, 0),
    length=10,
    lattice_size_factor=1.25,
    marker_size=10,
    figure_height=500,
):
    a = np.array(a)
    b = np.array(b)
    na, nb = endpoint
    v0 = na * a + nb * b
    circumference = np.linalg.norm(v0)
    v1 = np.array([-v0[1], v0[0]])
    v1 = length * v1 / np.linalg.norm(v1)
    corner_points = [np.array([0, 0]), v0, v0 + v1, v1]
    x, y = zip(*(corner_points + [na * a]))
    x0, x1 = min(x), max(x)
    y0, y1 = min(y), max(y)
    pad = min(x1 - x0, y1 - y0) * (lattice_size_factor - 1)
    xmin = x0 - pad
    xmax = x1 + pad
    ymin = y0 - pad
    ymax = y1 + pad

    nas = []
    nbs = []
    m = np.vstack((a, b)).T
    for v in [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]:
        tmp_a, tmp_b = np.linalg.solve(m, v)
        nas.append(tmp_a)
        nbs.append(tmp_b)
    na_min = np.floor(sorted(nas)[0]) - 2
    na_max = np.floor(sorted(nas)[-1]) + 2
    nb_min = np.floor(sorted(nbs)[0]) - 2
    nb_max = np.floor(sorted(nbs)[-1]) + 2

    ia = np.arange(na_min, na_max)
    ib = np.arange(nb_min, nb_max)
    x = []
    y = []
    for j in ib:
        for i in ia:
            v = i * a + j * b
            if xmin <= v[0] <= xmax and ymin <= v[1] <= ymax:
                x.append(v[0])
                y.append(v[1])

    df = pd.DataFrame({"x": x, "y": y})
    fig = px.scatter(df, x="x", y="y", color_discrete_sequence=["#636EFA"])

    x, y = zip(*corner_points)
    x = [*x, 0]
    y = [*y, 0]
    rectangle = go.Scatter(
        x=x,
        y=y,
        fill="toself",
        mode="lines",
        line=dict(color="green", width=marker_size / 5, dash="dash"),
    )
    fig.add_trace(rectangle)

    fig.data = (fig.data[1], fig.data[0])

    arrow_start = [0, 0]
    arrow_end = na * a
    fig.add_annotation(
        x=arrow_end[0],
        y=arrow_end[1],
        ax=arrow_start[0],
        ay=arrow_start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=3,
        arrowcolor="grey",
        opacity=1.0,
    )

    arrow_start = na * a
    arrow_end = v0
    fig.add_annotation(
        x=arrow_end[0],
        y=arrow_end[1],
        ax=arrow_start[0],
        ay=arrow_start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=3,
        arrowcolor="grey",
        opacity=1.0,
    )

    arrow_start = [0, 0]
    arrow_end = v0
    fig.add_annotation(
        x=arrow_end[0],
        y=arrow_end[1],
        ax=arrow_start[0],
        ay=arrow_start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=3,
        arrowcolor="red",
        opacity=1.0,
    )

    fig.update_traces(marker_size=marker_size, showlegend=False)

    fig.update_layout(
        xaxis=dict(title="X (\u00c5)", range=[xmin, xmax], constrain="domain"),
        yaxis=dict(title="Y (\u00c5)", range=[ymin, ymax], constrain="domain"),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    title = f"a=({a[0]:.2f}, {a[1]:.2f})\u00c5\tb=({b[0]:.2f}, {b[1]:.2f})\u00c5<br>equator=(0,0)\u2192{na}*a{'+' if nb>=0 else ''}{nb}*b\tcircumference={circumference:.2f}"
    fig.update_layout(title_text=title, title_x=0.5, title_xanchor="center")
    fig.update_layout(height=figure_height, margin=dict(l=50, r=10, t=80, b=50))
    fig.update_layout(paper_bgcolor="rgba(0, 0, 0, 0)", plot_bgcolor="rgba(0, 0, 0, 0)")

    return fig


def plot_helical_lattice_unrolled(
    diameter, length, twist, rise, csym, marker_size=10, figure_height=800
):
    circumference = np.pi * diameter
    if rise > 0:
        n = min(int(length / 2 / rise) + 2, 1000)
        i = np.arange(-n, n + 1)
        xs = []
        ys = []
        syms = []
        for si in range(csym):
            x = np.fmod(twist * i + si / csym * 360, 360)
            x[x > 360] -= 360
            x[x < 0] += 360
            y = rise * i
            xs.append(x)
            ys.append(y)
            syms.append(np.array([si] * len(x)))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    sym = np.concatenate(syms)

    df = pd.DataFrame({"x": x, "y": y, "csym": sym})
    df["csym"] = df["csym"].astype(str)

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="csym" if csym > 1 else None,
        color_discrete_sequence=["#636EFA"],
    )

    if twist >= 0:
        arrow_start = [0, 0]
        arrow_end = [twist, rise]
    else:
        arrow_start = [360, 0]
        arrow_end = [360 + twist, rise]
    fig.add_annotation(
        x=arrow_end[0],
        y=arrow_end[1],
        ax=arrow_start[0],
        ay=arrow_start[1],
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        opacity=1.0,
    )

    i = np.arange(-n, n + 1, 0.01)
    for si in range(csym):
        x = np.fmod(twist * i + si / csym * 360, 360)
        x[x > 360] -= 360
        x[x < 0] += 360
        y = rise * i
        color = fig.data[si].marker.color
        line = go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=color, width=marker_size / 10, dash="dot"),
            opacity=1,
            showlegend=False,
        )
        fig.add_trace(line)
    equator = go.Scatter(
        x=[0, 360],
        y=[0, 0],
        xaxis="x",
        line=dict(color="grey", width=marker_size / 3, dash="dash"),
    )
    fig.add_trace(equator)
    fig.update_traces(marker_size=marker_size, showlegend=False)

    fig.update_yaxes(scaleanchor="x", scaleratio=360 / circumference)
    fig.update_layout(
        xaxis=dict(
            title="twist (\u00b0)",
            range=[0, 360],
            tickvals=np.linspace(0, 360, 13),
            constrain="domain",
        ),
        yaxis=dict(
            title="rise (\u00c5)", range=[-length / 2, length / 2], constrain="domain"
        ),
    )

    title = f"pitch={rise*abs(360/twist):.2f}\u00c5\ttwist={twist:.2f}\u00b0 rise={rise:.2f}\u00c5 sym=c{csym}<br>diameter={diameter:.2f}\u00c5 circumference={circumference:.2f}\u00c5"
    fig.update_layout(title_text=title, title_x=0.5, title_xanchor="center")
    fig.update_layout(height=figure_height, margin=dict(l=50, r=10, t=80, b=50))

    return fig


def plot_helical_lattice(
    diameter,
    length,
    twist,
    rise,
    csym,
    rot=0,
    tilt=10,
    marker_size=10,
    figure_height=500,
    draw_equator_circle=True,
    draw_equator_disk=True,
    draw_cylinder=False,
    draw_axis=True,
):
    if rise > 0:
        n = min(int(length / 2 / rise) + 2, 1000)
        i = np.arange(-n, n + 1)
        xs = []
        ys = []
        zs = []
        syms = []
        for si in range(csym):
            x = diameter / 2 * np.cos(np.deg2rad(twist) * i + si / csym * 2 * np.pi)
            y = diameter / 2 * np.sin(np.deg2rad(twist) * i + si / csym * 2 * np.pi)
            z = i * rise
            xs.append(x)
            ys.append(y)
            zs.append(z)
            syms.append(np.array([si] * len(z)))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    sym = np.concatenate(syms)

    df = pd.DataFrame({"x": x, "y": y, "z": z, "csym": sym})
    df["csym"] = df["csym"].astype(str)

    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        labels={"x": "X (\u00c5)", "y": "Y (\u00c5)", "z": "Z (\u00c5)"},
        color="csym" if csym > 1 else None,
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_traces(marker_size=marker_size)

    i = np.arange(-n, n + 1, 5.0 / (abs(twist)))
    for si in range(csym):
        x = diameter / 2 * np.cos(np.deg2rad(twist) * i + si / csym * 2 * np.pi)
        y = diameter / 2 * np.sin(np.deg2rad(twist) * i + si / csym * 2 * np.pi)
        z = i * rise
        color = fig.data[si].marker.color
        spiral = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color=color, width=marker_size / 2),
            opacity=1,
            showlegend=False,
        )
        fig.add_trace(spiral)

    def cylinder(r, h, z0=0, n_points=100, nv=50):
        theta = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(z0, z0 + h, nv)
        theta, v = np.meshgrid(theta, v)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = v
        return x, y, z

    def equator_circle(r, z, n_points=36):
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z0 = z * np.ones(x.shape)
        return x, y, z0

    def equator_disk(r, z, n_points=36):
        rad = np.linspace(0, r, n_points)
        theta = np.linspace(0, 2 * np.pi, n_points)
        rad, theta = np.meshgrid(rad, theta)
        x = rad * np.cos(theta)
        y = rad * np.sin(theta)
        z0 = z * np.ones(x.shape)
        return x, y, z0

    if draw_axis:
        axis = go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[-length / 2, length / 2],
            mode="lines",
            line=dict(color="grey", width=marker_size / 2, dash="dash"),
            opacity=1,
            showlegend=False,
        )
        fig.add_trace(axis)

    if draw_cylinder:
        x, y, z = cylinder(r=diameter / 2 - marker_size / 2, h=length, z0=-length / 2)
        colorscale = [[0, "white"], [1, "white"]]
        cyl = go.Surface(
            x=x, y=y, z=z, colorscale=colorscale, showscale=False, opacity=0.2
        )
        fig.add_trace(cyl)

    if draw_equator_circle:
        x, y, z = equator_circle(r=diameter / 2, z=0)
        equator = go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="grey", width=marker_size / 2, dash="dash"),
            opacity=1,
            showlegend=False,
        )
        fig.add_trace(equator)

    if draw_equator_disk:
        x, y, z = equator_disk(r=diameter / 2, z=0)
        colorscale = [[0, "grey"], [1, "grey"]]
        equator = go.Surface(
            x=x, y=y, z=z, colorscale=colorscale, showscale=False, opacity=0.2
        )
        fig.add_trace(equator)

    title = f"pitch={rise*abs(360/twist):.2f}\u00c5\ttwist={twist:.2f}\u00b0 rise={rise:.2f}\u00c5 sym=c{csym}<br>diameter={diameter:.2f}\u00c5 circumference={np.pi*diameter:.2f}\u00c5"
    fig.update_layout(title_text=title, title_x=0.5, title_xanchor="center")

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(
            x=2.5 * np.cos(np.deg2rad(tilt)) * np.cos(np.deg2rad(rot)),
            y=2.5 * (-np.sin(np.deg2rad(rot))),
            z=2.5 * np.sin(np.deg2rad(tilt)),
        ),
    )
    fig.update_layout(scene_camera=camera)

    fig.update_scenes(
        xaxis=dict(range=[-diameter * 1.2, diameter * 1.2]),
        yaxis=dict(range=[-diameter * 1.2, diameter * 1.2]),
        zaxis=dict(range=[-length / 2 - marker_size, length / 2 + marker_size]),
    )

    fig.update_scenes(
        xaxis=dict(
            visible=False,
            showspikes=False,
            spikesides=False,
            spikethickness=0,
            spikecolor="rgba(0,0,0,0)",
            hoverformat="",
        ),
        yaxis=dict(
            visible=False,
            showspikes=False,
            spikesides=False,
            spikethickness=0,
            spikecolor="rgba(0,0,0,0)",
            hoverformat="",
        ),
        zaxis=dict(
            visible=False,
            showspikes=False,
            spikesides=False,
            spikethickness=0,
            spikecolor="rgba(0,0,0,0)",
            hoverformat="",
        ),
        camera_projection_type="orthographic",
        aspectmode="data",
    )
    fig.update_layout(height=figure_height, margin=dict(l=0, r=0, t=80, b=0))
    fig.update_layout(paper_bgcolor="rgba(0, 0, 0, 0)")

    return fig


# ── Shiny module ────────────────────────────────────────────────


@module.ui
def helical_lattice_tab_ui():
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_radio_buttons("radio", "", ["Helical\u21d22D", "2D\u21d2Helical"]),
            ui.output_ui("conditional_inputs"),
            ui.HTML(
                "<i><p>Developed by the <a href='https://jianglab.science.psu.edu/helicon' target='_blank'>Jiang Lab</a>. "
                "Report issues to <a href='https://github.com/jianglab/helicon/issues' target='_blank'>helicon@GitHub</a>.</p></i>"
            ),
            width="200px",
        ),
        ui.h1(
            "HelicalLattice: 2D Lattice \u21c4 Helical Lattice",
            style="font-weight: bold;",
        ),
        ui.output_ui("dynamic_plot"),
    )


@module.server
def helical_lattice_tab_server(input, output, session, project: ProjectState):
    @output
    @render.ui
    def conditional_inputs():
        if input.radio() == "Helical\u21d22D":
            return ui.TagList(
                ui.input_numeric(
                    "twist",
                    "Twist (\u00b0)",
                    value=-81.1,
                    min=-180.0,
                    max=180.0,
                    step=1.0,
                ),
                ui.input_numeric(
                    "rise", "Rise (\u00c5)", value=19.4, min=0.001, step=1.0
                ),
                ui.input_numeric("csym", "Axial symmetry", value=1, min=1, step=1),
                ui.input_numeric(
                    "diameter",
                    "Helical diameter (\u00c5)",
                    value=290.0,
                    min=0.1,
                    step=1.0,
                ),
                ui.input_numeric(
                    "length", "Helical length (\u00c5)", value=1000.0, min=0.1, step=1.0
                ),
                ui.input_checkbox(
                    "primitive_unitcell", "Use primitive unit cell", value=False
                ),
                ui.input_checkbox(
                    "horizontal", "Set unit cell vector a along x-axis", value=True
                ),
                ui.input_numeric(
                    "lattice_size_factor",
                    "2D lattice size factor",
                    value=1.25,
                    min=1.0,
                    step=0.1,
                ),
                ui.input_numeric(
                    "marker_size", "Marker size (\u00c5)", value=5.0, min=0.1, step=1.0
                ),
                ui.input_numeric(
                    "figure_height", "Plot height (pixels)", value=800, min=1, step=10
                ),
            )
        elif input.radio() == "2D\u21d2Helical":
            return ui.TagList(
                ui.input_numeric(
                    "ax", "Unit cell vector a.x (\u00c5)", value=34.65, step=1.0
                ),
                ui.input_numeric(
                    "ay", "Unit cell vector a.y (\u00c5)", value=0.0, step=1.0
                ),
                ui.input_numeric(
                    "bx", "Unit cell vector b.x (\u00c5)", value=10.63, step=1.0
                ),
                ui.input_numeric(
                    "by", "Unit cell vector b.y (\u00c5)", value=-23.01, step=1.0
                ),
                ui.input_numeric(
                    "na", "# units along unit cell vector a", value=16, step=1
                ),
                ui.input_numeric(
                    "nb", "# units along unit cell vector b", value=1, step=1
                ),
                ui.input_numeric(
                    "length", "Helical length (\u00c5)", value=1000.0, min=0.1, step=1.0
                ),
                ui.input_numeric(
                    "lattice_size_factor",
                    "2D lattice size factor",
                    value=1.25,
                    min=1.0,
                    step=0.1,
                ),
                ui.input_numeric(
                    "marker_size", "Marker size (\u00c5)", value=5.0, min=0.1, step=1.0
                ),
                ui.input_numeric(
                    "figure_height", "Plot height (pixels)", value=800, min=1, step=10
                ),
            )

    @output
    @render.ui
    def dynamic_plot():
        if input.radio() == "2D⇒Helical":
            na, nb = input.na(), input.nb()
            a = (input.ax(), input.ay())
            b = (input.bx(), input.by())
            length = input.length()
            lsf = input.lattice_size_factor()
            ms = input.marker_size()
            fh = input.figure_height()
            twist2, rise2, csym2, dia2 = convert_2d_lattice_to_helical_lattice(
                a=a, b=b, endpoint=(na, nb)
            )

            fig1 = plot_2d_lattice(
                a,
                b,
                endpoint=(na, nb),
                length=length,
                lattice_size_factor=lsf,
                marker_size=ms,
                figure_height=fh,
            )
            fig2 = plot_helical_lattice_unrolled(
                dia2, length, twist2, rise2, csym2, marker_size=ms, figure_height=fh
            )
            fig3 = plot_helical_lattice(
                dia2,
                length,
                twist2,
                rise2,
                csym2,
                marker_size=ms * 0.6,
                figure_height=fh,
            )

            return ui.row(
                ui.column(
                    5,
                    ui.h3(
                        "2D Lattice: from which a block of area is selected to be rolled into a helix",
                        style="text-align:center;",
                    ),
                    _fig_to_html(fig1),
                ),
                ui.column(
                    4,
                    ui.h3(
                        "2D Lattice: selected area is ready to be rolled into a helix around the vertical axis",
                        style="text-align:center;",
                    ),
                    _fig_to_html(fig2),
                ),
                ui.column(
                    3,
                    ui.h3(
                        "Helical Lattice: rolled up from the starting 2D lattice",
                        style="text-align:center;",
                    ),
                    _fig_to_html(fig3),
                ),
            )
        else:
            diameter = input.diameter()
            length = input.length()
            twist = input.twist()
            rise = input.rise()
            csym = input.csym()
            ms = input.marker_size()
            fh = input.figure_height()

            a, b, endpoint = convert_helical_lattice_to_2d_lattice(
                twist=twist,
                rise=rise,
                csym=csym,
                diameter=diameter,
                primitive_unitcell=input.primitive_unitcell(),
                horizontal=input.horizontal(),
            )

            fig1 = plot_helical_lattice(
                diameter,
                length,
                twist,
                rise,
                csym,
                marker_size=ms * 0.6,
                figure_height=fh,
            )
            fig2 = plot_helical_lattice_unrolled(
                diameter, length, twist, rise, csym, marker_size=ms, figure_height=fh
            )
            fig3 = plot_2d_lattice(
                a,
                b,
                endpoint,
                length=length,
                lattice_size_factor=input.lattice_size_factor(),
                marker_size=ms,
                figure_height=fh,
            )

            return ui.row(
                ui.column(
                    3,
                    ui.h3("Helical Lattice", style="text-align:center;"),
                    _fig_to_html(fig1),
                ),
                ui.column(
                    4,
                    ui.h3(
                        "Helical Lattice: unrolled into a 2D lattice",
                        style="text-align:center;",
                    ),
                    _fig_to_html(fig2),
                ),
                ui.column(
                    5,
                    ui.h3(
                        "2D Lattice: from which the helix is built",
                        style="text-align:center;",
                    ),
                    _fig_to_html(fig3),
                ),
            )

    # ── Shared state propagation ──────────────────────────────────

    @reactive.effect
    def _propagate_to_shared_state():
        """Push lattice-derived helical params to shared project state."""
        twist, rise, csym, diameter = _get_params()
        project.twist.set(float(twist))
        project.rise.set(float(rise))
        project.csym.set(int(csym))
        project.diameter.set(float(diameter))

    def _get_params():
        if input.radio() == "2D\u21d2Helical":
            return convert_2d_lattice_to_helical_lattice(
                a=(input.ax(), input.ay()),
                b=(input.bx(), input.by()),
                endpoint=(input.na(), input.nb()),
            )
        else:
            return (input.twist(), input.rise(), input.csym(), input.diameter())
