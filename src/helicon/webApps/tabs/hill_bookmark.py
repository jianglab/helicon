"""The bookmarked parameters of the HILL tab.

Kept apart from the tab, which is imported only when it is first opened
(``app.py``): the page needs this table at start-up for the bookmark script, and
the tab's modules (the widget bridge, SciPy) take seconds to import.
"""

from shiny import ui

from .. import bookmark

# Worked out from the loaded image (its pixel size, the filament's diameter,
# the rotation and shift that straighten and centre it) rather than chosen:
# in a bookmark only once edited.
_DERIVED = bookmark.DERIVED

BOOKMARK_DEFAULTS = {
    "input_mode": ("hill_input_mode_params", "2"),
    "url": ("hill_img_file_url", "https://tinyurl.com/y5tq9fqa"),
    "server_image": ("hill_img_file_server", ""),
    "is_3d": ("hill_is_3d", False),
    "twist": ("hill_twist", 29.40),
    "rise": ("hill_rise", 21.92),
    "csym": ("hill_csym", 6),
    "diameter": ("hill_diameter", 155.2, _DERIVED),
    "apix": ("hill_apix", 2.3438, _DERIVED),
    "cutoff_res_x": ("hill_cutoff_res_x", 7.03, _DERIVED),
    "cutoff_res_y": ("hill_cutoff_res_y", 4.69, _DERIVED),
    "log_amp": ("hill_log_amp", True),
    "pnx": ("hill_pnx", 512),
    "pny": ("hill_pny", 1024),
    "hp_fraction": ("hill_hp_fraction", 0.40),
    "lp_fraction": ("hill_lp_fraction", 0.00),
    "m_max": ("hill_m_max", 3),
    "const_image_color": ("hill_const_image_color", ""),
    "ll_colors": ("hill_ll_colors", "lime cyan violet salmon silver"),
    "fft_top_only": ("hill_fft_top_only", False),
    "use_twist_pitch": ("hill_use_twist_pitch", "Twist"),
    "out_of_plane_tilt": ("hill_out_of_plane_tilt", 0.0),
    "input_type": ("hill_input_type", "Image"),
    "angle": ("hill_angle", 0.0, _DERIVED),
    "dx": ("hill_dx", 0.0, _DERIVED),
    "dy": ("hill_dy", 0.0, _DERIVED),
}


def bokeh_dependency():
    """Bokeh's scripts, from the CDN, for the page head.

    The page loads them at the start, for the HILL tab to use when it is first
    opened: scripts that arrive with the tab's own page come too late for the
    widget that shows the figures ("Bokeh is not defined").
    """
    from bokeh.resources import Resources

    resources = Resources(
        mode="cdn", components=["bokeh", "bokeh-widgets", "bokeh-tables"]
    )
    return ui.head_content(ui.HTML(resources.render()))
