"""The bookmarked parameters of the WhereIsMyClass tab.

Kept apart from the tab, which is imported only when it is first opened
(``app.py``): the page needs this table at start-up for the bookmark script, and
the tab's modules (the widget bridge) take seconds to import.
"""

BOOKMARK_DEFAULTS = {
    "input_mode": ("wimc_input_mode", "server"),
    "url_star": ("wimc_url_star", ""),
    "server_star": ("wimc_server_star", ""),
    "ignore_blank": ("wimc_ignore_blank", True),
    "sort_abundance": ("wimc_sort_abundance", True),
    "download_micrographs": ("wimc_download_micrographs", True),
    "rise": ("wimc_rise", 4.75),
    "target_apix": ("wimc_target_apix", 5),
    "low_pass": ("wimc_low_pass_angstrom", 20),
    "high_pass": ("wimc_high_pass_angstrom", 0),
    "max_len": ("wimc_max_len", -1),
    "max_pair_dist": ("wimc_max_pair_dist", -1),
    "bins": ("wimc_bins", 100),
    "plot_height": ("wimc_plot_height", 640),
}
