"""The bookmarked parameters of the Denovo3D tab, and the example it starts from.

Kept apart from the tab, which is imported only when it is first opened
(``app.py``): the page needs this table at start-up for the bookmark script.
"""

_urls = {
    "empiar-10940_job010": (
        "https://ftp.ebi.ac.uk/empiar/world_availability/10940/data/EMPIAR/Class2D/job010/run_it020_classes.mrcs",
        "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-14046/map/emd_14046.map.gz",
    ),
}
_url_key = "empiar-10940_job010"

BOOKMARK_DEFAULTS = {
    "input_mode_images": ("dn_input_mode_images", "url"),
    "url_images": ("dn_url_images", _urls[_url_key][0]),
    "server_images": ("dn_server_images", ""),
    "selected": ("dn_select_image_ids", [1]),
    "is_3d": ("dn_is_3d", False),
    "ignore_blank": ("dn_ignore_blank", True),
    "auto_method": ("dn_auto_method", "threshold"),
    "plot_scores": ("dn_plot_scores", True),
    "center_line": ("dn_show_center_line", True),
    "stitch_extend": ("dn_stitch_extend", True),
    "show_download": ("dn_show_download_buttons", False),
    "match_input_box": ("dn_match_input_box", True),
    "display_size": ("dn_selected_image_display_size", 128),
    "rec_length": ("dn_reconstruct_length_rise", 3),
    "target_apix2d": ("dn_target_apix2d", 5),
    "target_apix3d": ("dn_target_apix3d", 5),
    "sym_oversample": ("dn_sym_oversample", -1),
    "lr_alpha": ("dn_lr_alpha", -1),
    "lr_l1_ratio": ("dn_lr_l1_ratio", 0.5),
    "top_n": ("dn_top_n_results", 10),
    "lr_algorithm": ("dn_lr_algorithm", "gauss"),
    "rec_algorithm": ("dn_rec_algorithm", "elasticnet"),
    "positive": ("dn_positive_constraint", -1),
    "interpolation": ("dn_interpolation", "linear"),
    "score_metric": ("dn_score_metric", "cosine"),
    "input_ui_type": ("dn_input_ui_type", "Slider"),
}
