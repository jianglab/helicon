"""Qt widgets for napari integration."""

from __future__ import annotations

import os
import sys
import time
from functools import partial
from pathlib import Path

from helicon.lib.terminal import _open_terminal, _spawn_detached
from helicon.lib.gui.browser_files import (  # noqa: F401 (also what callers and tests take from here)
    _find_chimerax,
    _folder_is_helical,
    _folder_is_helical_cached,
    _open_url,
    _star_has_helix_col,
    _DIGIT_RUN,
    _IMAGE_EXTENSIONS,
    _KNOWN_EXTENSIONS,
    _KNOWN_TEXT_EXTENSIONS,
    _METADATA_STAR_SUFFIXES,
    _MRC_EXTENSIONS,
    _MRC_VOLUME_EXTENSIONS,
    _RASTER_IMAGE_EXTENSIONS,
    _count_dir_items,
    _count_folder_entries,
    _format_item_count,
    _format_modified,
    _format_size,
    _get_file_info,
    _get_file_type_label,
    _is_text_file,
    _name_matches_filter,
    _natural_sort_key,
)
from helicon.lib.gui.browser_model import (  # noqa: F401
    COL_IMAGES,
    COL_INFO,
    COL_MODIFIED,
    COL_NAME,
    COL_PIXELSIZE,
    COL_SIZE,
    COL_TYPE,
    INSERT_SLICE_S,
    NUM_COLUMNS,
    ROLE_KIND,
    ROLE_SORT,
    SYNC_ROWS,
    DirCountWorker,
    FileBrowserModel,
    InfoWorker,
    _icon_kind,
    _themed_icon,
)
from helicon.lib.gui.browser_settings import (  # noqa: F401
    _BookmarksDialog,
    _add_bookmark,
    _bookmark_text,
    _get_bookmark_names,
    _rename_bookmark,
    _add_recent_folder,
    _bookmark_key,
    _clear_bookmarks,
    _confirm_clear_bookmarks,
    _get_bookmarks,
    _get_recent_folders,
    _is_bookmarked,
    _move_bookmark,
    _remove_bookmark,
    _save_bookmarks,
)
from helicon.lib.gui.browser_style import (  # noqa: F401
    _DEFAULT_THEME,
    _THEME_COLORS,
    _THEMES,
    _browser_stylesheet,
    _disabled_button_stylesheet,
    _resolved_theme,
    _save_theme,
    _saved_theme,
    _widgets_qss,
)
from helicon.lib.gui.browser_parts import (
    FlowLayout,
    PathBar,
    PlacesSidebar,
    PreviewPane,
)

from PySide6.QtWidgets import (
    QWidget,
    QMainWindow,
    QTreeView,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QHeaderView,
    QMenu,
    QApplication,
    QPushButton,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QSplitter,
)
from PySide6.QtGui import (
    QKeySequence,
    QShortcut,
    QAction,
    QActionGroup,
)
from PySide6.QtCore import (
    QModelIndex,
    Signal,
    Qt,
    QThread,
    QSettings,
    QSize,
    QTimer,
)

__all__ = ["FolderBrowserWidget"]


# Standalone tools exposed through the Apps menu, in display order.
# Tuple: (menu label, shiny tab name or None, streamlit subcommand module or
# None, picker kind or None). A shiny tab is opened in the consolidated
# Helicon Lab web app without an input file; a streamlit module is launched via
# its subcommand ``main`` (WebCalEM is no streamlit app: its command only opens
# the hosted page in the browser). The picker kind selects a file picker that
# lets the user choose an input before launching the tool: "map" requires one .mrc,
# "two_maps" requires two half-map .mrc files, and "star" opens the tool's
# own in-panel file selector with no pre-set input.
_APP_LAUNCH_TABLE = [
    ("WebCalEM", None, "helicon.commands.webCalEM", None),
    ("Images2Star", None, None, "star"),
    ("Proc3D", None, None, "map"),
    ("WhereIsMyClass", "WhereIsMyClass", None, None),
    ("HelicalProjection", "HelicalProjection", None, None),
    ("HILL", "HILL", None, None),
    ("HelicalPitch", "HelicalPitch", None, None),
    ("Denovo3D", "Denovo3D", None, None),
    ("AbInitio3D", "AbInitio3D", None, None),
    ("HelicalLattice", "HelicalLattice", None, None),
    ("HI3D", "HI3D", None, None),
    ("TrueFSC", None, None, "two_maps"),
    ("Map2Seq", None, "helicon.commands.map2seq", None),
    ("ProCart", None, "helicon.commands.procart", None),
    ("CTF Simulation", None, "helicon.commands.ctfSimulation", None),
]


def _has_class_averages(params_path: str) -> bool:
    """Whether the class averages that go with a Class2D parameter file are beside it."""
    from helicon.webApps import class2d_files

    return class2d_files.companion(params_path) is not None


class _ModesWorker(QThread):
    """Works out, off the main thread, what can be done with a file: the display modes
    (which read headers and datasets), the label of the slice button, and whether
    ChimeraX is there. ``ready(serial, info)`` at the end."""

    ready = Signal(int, dict)

    def __init__(self, serial: int, path: str, browser, parent=None):
        super().__init__(parent)
        self._serial = serial
        self._path = path
        self._browser = browser

    def run(self) -> None:
        path = self._path
        info = dict(path=path, modes=[], slice_text="Image Slice", chimerax=True)
        try:
            modes = self._browser._display_modes_for(path)
            info["modes"] = modes
            if modes:
                suffix = Path(path).suffix.lower()
                if suffix == ".pdf":
                    info["slice_text"] = "PDF"
                elif suffix == ".eps":
                    info["slice_text"] = "EPS"
                elif "volume" in modes:
                    info["slice_text"] = "2D Slice"
                elif self._browser._is_image_stack(path):
                    info["slice_text"] = "2D Image"
                if "chimerax" in modes:
                    info["chimerax"] = _find_chimerax() is not None
        except Exception:
            info["modes"] = []  # a file that cannot be read gets no buttons
        self.ready.emit(self._serial, info)


# the icon (Font Awesome, drawn by qtawesome) and the category of each action button.
# "viewer" shows the file in a window of the browser; "app" opens a web app or a panel
# of tools, and is outlined in the accent colour.
_ACTION_STYLE = {
    "slice": ("fa5s.image", "viewer"),
    "volume": ("fa5s.cube", "viewer"),
    "3dplot": ("fa5s.project-diagram", "viewer"),
    "chimerax": ("fa5s.atom", "viewer"),
    "stats": ("fa5s.chart-bar", "viewer"),
    "text": ("fa5s.file-alt", "viewer"),
    "gallery": ("fa5s.th", "viewer"),
    "optimiser": ("fa5s.layer-group", "viewer"),
    "2dclasses": ("fa5s.th-large", "viewer"),
    "orthogonal": ("fa5s.cubes", "viewer"),
    "fsc": ("fa5s.chart-line", "viewer"),
    "html": ("fa5s.globe", "viewer"),
    "denovo3D": ("fa5s.dna", "app"),
    "abInitio3D": ("fa5s.shapes", "app"),
    "whereIsMyClass": ("fa5s.map-marker-alt", "app"),
    "helicalProjection": ("fa5s.film", "app"),
    "helicalPitch": ("fa5s.ruler-horizontal", "app"),
    "hill": ("fa5s.wave-square", "app"),
    "hi3d": ("fa5s.circle-notch", "app"),
    "proc3d": ("fa5s.sliders-h", "app"),
    "trueFSC": ("fa5s.chart-line", "app"),
    "images2star": ("fa5s.exchange-alt", "app"),
}


class FolderBrowserWidget(QMainWindow):
    """A folder browser widget for selecting image and volume files.

    Displays a tree view of files with columns for name, size, type,
    info, images, and modification date. Double-click a file to emit
    the ``file_selected`` signal.

    Parameters
    ----------
    start_dir : str or Path, optional
        Initial directory to display. Defaults to the current working directory.

    Signals
    -------
    file_selected(str)
        Emitted when a file is double-clicked. The value is the full file path.
    file_selected_new_window(str)
        Emitted when a file is Shift+double-clicked. The value is the full file path.
    """

    file_selected = Signal(str)
    file_selected_new_window = Signal(str)
    # (path, mode, new_window) where mode is a display-mode string from _DISPLAY_BUTTONS.
    display_requested = Signal(str, str, bool)

    # Maps QPushButton attribute name → display-mode string. Drives button
    # construction, click wiring, per-mode visibility, and alphabetical
    # reordering in ``_reorder_buttons_alphabetically``.
    _DISPLAY_BUTTONS: list[tuple[str, str]] = [
        ("_btn_slice", "slice"),
        ("_btn_volume", "volume"),
        ("_btn_3dplot", "3dplot"),
        ("_btn_chimerax", "chimerax"),
        ("_btn_stats", "stats"),
        ("_btn_text", "text"),
        ("_btn_gallery", "gallery"),
        ("_btn_optimiser", "optimiser"),
        ("_btn_2dclasses", "2dclasses"),
        ("_btn_orthogonal", "orthogonal"),
        ("_btn_proc3d", "proc3d"),
        ("_btn_fsc", "fsc"),
        ("_btn_html", "html"),
        ("_btn_denovo3d", "denovo3D"),
        ("_btn_abinitio3d", "abInitio3D"),
        ("_btn_whereismyclass", "whereIsMyClass"),
        ("_btn_helicalprojection", "helicalProjection"),
        ("_btn_helicalpitch", "helicalPitch"),
        ("_btn_hill", "hill"),
        ("_btn_hi3d", "hi3d"),
        ("_btn_truefsc", "trueFSC"),
        ("_btn_images2star", "images2star"),
    ]

    def __init__(self, start_dir: str | Path | None = None, parent=None):
        super().__init__(parent)

        if start_dir is None:
            start_dir = os.getcwd()
        start_dir = str(Path(start_dir).resolve())

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self._main_layout = layout
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._menu_bar = self.menuBar()
        self._file_menu = self._menu_bar.addMenu("File")
        self._bookmarks_menu = self._menu_bar.addMenu("Bookmarks")
        self._apps_menu = self._menu_bar.addMenu("Apps")

        self._open_folder_action = QAction("Open Folder…", self)
        self._open_folder_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_folder_action.triggered.connect(self._open_folder)
        self._file_menu.addAction(self._open_folder_action)

        # Terminal is the first app, kept on the Apps menu where every other
        # standalone tool lives rather than buried in the File menu.
        self._open_terminal_action = QAction("Terminal", self)
        self._open_terminal_action.setShortcut(QKeySequence("Ctrl+Shift+T"))
        self._open_terminal_action.triggered.connect(self._open_terminal)
        self._apps_menu.addAction(self._open_terminal_action)

        self._configure_relion_action = QAction("Configure RELION…", self)
        # On macOS, Qt moves an action whose text contains "config" (or
        # "settings", "options", ...) into the application menu as
        # "Preferences…", so it would vanish from this menu.
        self._configure_relion_action.setMenuRole(QAction.MenuRole.NoRole)
        self._configure_relion_action.triggered.connect(self._configure_relion)
        self._apps_menu.addAction(self._configure_relion_action)
        self._relion_processes = {}
        self._relion_timer = QTimer(self)
        self._relion_timer.setInterval(1000)
        self._relion_timer.timeout.connect(self._poll_relion)

        self._app_actions = {}
        for label, key, streamlit_mod, picker in _APP_LAUNCH_TABLE:
            action = QAction(label, self)
            action.setData((key, streamlit_mod, picker))
            action.triggered.connect(self._on_launch_app)
            self._apps_menu.addAction(action)
            self._app_actions[label] = action

        self._bookmark_toggle_action = QAction("Bookmark This Folder", self)
        self._bookmark_toggle_action.setShortcut(QKeySequence("Ctrl+D"))
        self._bookmark_toggle_action.triggered.connect(self._toggle_bookmark)
        # filled each time the menu opens (the model does not exist yet); the
        # window owns the action, so Ctrl+D works with the menu closed
        self._bookmarks_menu.aboutToShow.connect(self._refresh_bookmarks_menu)
        self.addAction(self._bookmark_toggle_action)

        self._recent_menu = self._file_menu.addMenu("Recent Folders")
        self._recent_menu.aboutToShow.connect(self._refresh_recent_menu)
        self._refresh_recent_menu()

        self._file_menu.addSeparator()
        self._quit_action = QAction("Quit", self)
        self._quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self._quit_action.triggered.connect(self._quit)
        self._file_menu.addAction(self._quit_action)

        self._view_menu = self._menu_bar.addMenu("View")

        self._theme_menu = self._view_menu.addMenu("Theme")
        self._theme_actions = {}
        self._theme_action_group = QActionGroup(self)
        self._theme_action_group.setExclusive(True)
        for theme in _THEMES:
            action = QAction(theme, self)
            action.setCheckable(True)
            action.setData(theme)
            action.triggered.connect(self._on_theme_action_triggered)
            self._theme_action_group.addAction(action)
            self._theme_menu.addAction(action)
            self._theme_actions[theme] = action

        self._view_menu.addSeparator()
        self._sidebar_action = QAction("Sidebar", self)
        self._sidebar_action.setCheckable(True)
        self._sidebar_action.setShortcut(QKeySequence("F9"))
        self._sidebar_action.toggled.connect(self._on_sidebar_toggled)
        self._view_menu.addAction(self._sidebar_action)
        self._preview_action = QAction("Preview Pane", self)
        self._preview_action.setCheckable(True)
        self._preview_action.setShortcut(QKeySequence("F3"))
        self._preview_action.toggled.connect(self._on_preview_toggled)
        self._view_menu.addAction(self._preview_action)
        self._location_action = QAction("Edit Location", self)
        self._location_action.setShortcut(QKeySequence("Ctrl+L"))
        self._location_action.triggered.connect(lambda: self._path_edit.edit())
        self._view_menu.addAction(self._location_action)

        self._help_menu = self._menu_bar.addMenu("Help")
        self._home_page_action = QAction("Home Page", self)
        self._home_page_action.triggered.connect(self._open_home_page)
        self._help_menu.addAction(self._home_page_action)
        self._docs_action = QAction("Documentation", self)
        self._docs_action.triggered.connect(self._open_documentation)
        self._help_menu.addAction(self._docs_action)
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(4, 4, 4, 4)
        nav_layout.setSpacing(4)

        self._back_btn = QToolButton()
        self._back_btn.setText("←")
        self._back_btn.setToolTip("Go back")
        self._back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self._back_btn)

        self._up_btn = QToolButton()
        self._up_btn.setText("↑")
        self._up_btn.setToolTip("Go up")
        self._up_btn.clicked.connect(self._go_up)
        nav_layout.addWidget(self._up_btn)

        self._refresh_btn = QToolButton()
        self._refresh_btn.setText("↻")
        self._refresh_btn.setToolTip("Refresh")
        self._refresh_btn.clicked.connect(self._refresh)
        nav_layout.addWidget(self._refresh_btn)

        self._path_edit = PathBar(start_dir)
        self._path_edit.returnPressed.connect(self._go_to_path)
        self._path_edit.pathClicked.connect(self._navigate_to)
        # Stretch factor 1 makes the path box expand to fill all space up to
        # the filter controls at the right end of the row.
        nav_layout.addWidget(self._path_edit, 1)

        self._relion_btn = QPushButton("Open RELION")
        self._relion_btn.setToolTip(
            "Open this project in RELION; setup: Apps → Configure RELION…"
        )
        self._relion_btn.clicked.connect(self._open_relion)
        nav_layout.addWidget(self._relion_btn)

        self._recent_combo = QComboBox()
        self._recent_combo.setPlaceholderText("Recent")
        self._recent_combo.setToolTip("Recent starting folders")
        self._recent_combo.setFixedWidth(80)
        self._recent_combo.setStyleSheet(
            "QComboBox { color: #cccccc; background: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 3px; }"
            "QComboBox QAbstractItemView { color: #cccccc; background: #2d2d2d; }"
        )
        _add_recent_folder(start_dir)
        self._refresh_recent_combo()
        self._recent_combo.activated.connect(self._on_recent_selected)
        self._recent_combo.hide()

        self._theme_combo = QComboBox()
        self._theme_combo.addItems(_THEMES)
        self._theme_combo.setCurrentText(_saved_theme())
        self._theme_combo.setToolTip("Choose the file browser and launch-button theme")
        self._theme_combo.setFixedWidth(90)
        self._theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self._theme_combo.hide()

        # Filter controls at the right end of the navigation row, next to the
        # current-folder input box.

        filter_label = QLabel("Filter:")
        nav_layout.addWidget(filter_label)

        self._filter_edit = QLineEdit("*")
        self._filter_edit.setPlaceholderText("e.g. *.mrc, *.tif")
        self._filter_edit.returnPressed.connect(self._apply_filter)
        nav_layout.addWidget(self._filter_edit)

        self._regex_cb = QCheckBox("Regex")
        self._regex_cb.stateChanged.connect(self._apply_filter)
        nav_layout.addWidget(self._regex_cb)

        layout.addLayout(nav_layout)

        self._model = FileBrowserModel(start_dir)

        # Active background info-population threads. Cleared (and quit) whenever
        # the directory is reloaded so superseded workers don't write to the
        # new model.
        self._info_threads: list[QThread] = []

        self._history = [start_dir]
        self._history_index = 0

        self._model.load_finished.connect(self._populate_file_info_async)
        self._tree = QTreeView()
        self._tree.setModel(self._model)
        # all rows are as tall as the first: the view need not measure a row to place it,
        # which is what made a folder of tens of thousands of files slow to scroll
        self._tree.setUniformRowHeights(True)
        self._tree.setRootIndex(QModelIndex())

        self._tree.setSortingEnabled(True)
        self._tree.sortByColumn(COL_NAME, Qt.SortOrder.AscendingOrder)

        # Restore saved sort state.
        self._restore_sort_state()

        self._tree.setSelectionBehavior(QTreeView.SelectionBehavior.SelectItems)
        self._tree.setSelectionMode(QTreeView.SelectionMode.ExtendedSelection)

        copy_sc = QShortcut(QKeySequence.StandardKey.Copy, self._tree)
        copy_sc.activated.connect(self._copy_selection)

        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)

        header = self._tree.header()
        header.setStretchLastSection(True)
        header.setSectionsMovable(False)
        for col in range(NUM_COLUMNS):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(COL_NAME, 280)
        header.resizeSection(COL_SIZE, 80)
        header.resizeSection(COL_TYPE, 130)
        header.resizeSection(COL_IMAGES, 70)
        header.resizeSection(COL_INFO, 100)
        header.resizeSection(COL_PIXELSIZE, 90)
        header.resizeSection(COL_MODIFIED, 140)

        # Restore previously saved column widths (persisted between launches).
        # Done before connecting sectionResized so the initial programmatic
        # resizes above don't trigger a redundant save.
        self._restore_col_widths()

        # Persist column widths as the user drags dividers (debounced).
        self._col_save_timer = QTimer(self)
        self._col_save_timer.setSingleShot(True)
        self._col_save_timer.timeout.connect(self._save_col_widths)
        header.sectionResized.connect(lambda *a: self._col_save_timer.start(300))
        header.sortIndicatorChanged.connect(self._save_sort_state)

        self._tree.setAnimated(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setIconSize(QSize(18, 18))
        self._tree.setIndentation(20)
        self._tree.setExpandsOnDoubleClick(False)

        self._tree.setStyleSheet(
            """
            QTreeView {
                font-size: 12px;
                background-color: #2d2d2d;
                color: #cccccc;
                border: none;
            }
            QTreeView::item:selected {
                background-color: #4a6fa5;
            }
            QTreeView::item:hover {
                background-color: #3a3a3a;
            }
            QHeaderView::section {
                padding: 4px;
                font-weight: bold;
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #2d2d2d;
            }
            """
        )

        self.setStyleSheet(
            """
            QWidget {
                background-color: #2d2d2d;
                color: #cccccc;
            }
            QLineEdit {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px;
            }
            QToolButton {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QToolButton:hover {
                background-color: #4a6fa5;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 3px 8px;
            }
            QPushButton:hover {
                background-color: #4a6fa5;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #5a82c4;
                border: 1px solid #6f9bd6;
            }
            QCheckBox {
                color: #e8e8e8;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #888888;
                border-radius: 3px;
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:unchecked {
                background-color: #3c3c3c;
            }
            QCheckBox::indicator:checked {
                background-color: #4a6fa5;
                border: 1px solid #6f9bd6;
                image: none;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #5a82c4;
            }
            """
        )

        self._sidebar = PlacesSidebar()
        self._sidebar.navigate.connect(self._navigate_to)
        self._sidebar.bookmark_removed.connect(self._on_bookmark_removed)
        self._sidebar.bookmark_renamed.connect(self._on_bookmark_renamed)
        self._sidebar.alias_requested.connect(self._on_alias_requested)
        self._preview = PreviewPane()
        self._preview.path_requested.connect(self._navigate_to)
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.addWidget(self._sidebar)
        self._splitter.addWidget(self._tree)
        self._splitter.addWidget(self._preview)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setStretchFactor(2, 0)
        self._splitter.setSizes([180, 760, 300])
        layout.addWidget(self._splitter, 1)
        self._restore_panes()

        # Contextual display-mode buttons + "new window" toggle, in the preview pane
        # below the facts (at the bottom of the window when the pane is hidden).
        # Hidden unless exactly one file (not a directory) is selected.
        self._action_bar = QWidget()
        self._action_bar.setObjectName("actionBar")
        # the buttons wrap to a second row when the window is narrow
        action_layout = FlowLayout(self._action_bar, margin=8, h_spacing=6, v_spacing=6)

        self._btn_slice = QPushButton("Image Slice")
        self._btn_volume = QPushButton("3D Volume")
        self._btn_3dplot = QPushButton("3D Plot")
        self._btn_3dplot.setToolTip("Display 3D coordinates from a .bild file")
        self._btn_chimerax = QPushButton("ChimeraX")
        self._btn_chimerax.setToolTip("Open this file in ChimeraX")
        self._btn_stats = QPushButton("Stats")
        self._btn_text = QPushButton("Text")
        self._btn_text.setToolTip("Open this file as text")
        self._btn_gallery = QPushButton("Gallery")
        self._btn_gallery.setToolTip("Show a lazy thumbnail grid of the stack")
        self._btn_optimiser = QPushButton("XYZ Slice")
        self._btn_optimiser.setToolTip(
            "Show center slices (Z, Y, X) of referenced MRC maps"
        )
        self._btn_2dclasses = QPushButton("2D Classes")
        self._btn_2dclasses.setToolTip("Show 2D class averages sorted by abundance")
        self._btn_orthogonal = QPushButton("Ortho Slice")
        self._btn_orthogonal.setToolTip("Interactive XYZ slice viewer for 3D volumes")
        self._btn_proc3d = QPushButton("Proc3D")
        self._btn_proc3d.setToolTip(
            "Open the Proc3D tools panel to preview and transform "
            "this 3D map (flip hand, clip, resample, ...), comparing the "
            "result side by side with the original, then save a new MRC file"
        )
        self._btn_fsc = QPushButton("FSC")
        self._btn_fsc.setToolTip(
            "Display the FSC curve, or the SSNR-derived W_MAP curve for Class3D, "
            "from model.star. Pick which refinement iterations to overlay when "
            "the job directory holds more than one model.star file."
        )
        self._btn_html = QPushButton("Browser")
        self._btn_html.setToolTip("Open this file in your web browser")
        self._btn_denovo3d = QPushButton("denovo3D")
        self._btn_denovo3d.setToolTip(
            "Open this file in the denovo3D web app for de novo helical indexing"
        )
        self._btn_abinitio3d = QPushButton("AbInitio3D")
        self._btn_abinitio3d.setToolTip(
            "Open this Class2D job in the AbInitio3D web app for an ab initio "
            "3D map from the 2D classes"
        )
        self._btn_whereismyclass = QPushButton("WhereIsMyClass")
        self._btn_whereismyclass.setToolTip(
            "Open this file in the WhereIsMyClass web app for mapping 2D classes"
        )
        self._btn_helicalprojection = QPushButton("HelicalProjection")
        self._btn_helicalprojection.setToolTip(
            "Open this file in the HelicalProjection web app"
        )
        self._btn_helicalpitch = QPushButton("HelicalPitch")
        self._btn_helicalpitch.setToolTip(
            "Open this file in the HelicalPitch web app for determining helical pitch/twist"
        )
        self._btn_hill = QPushButton("HILL")
        self._btn_hill.setToolTip(
            "Open this file in the HILL web app for helical indexing via Fourier layer lines"
        )
        self._btn_hi3d = QPushButton("HI3D")
        self._btn_hi3d.setToolTip(
            "Open this file in the HI3D web app for helical indexing via cylindrical projection"
        )
        self._btn_truefsc = QPushButton("TrueFSC")
        self._btn_truefsc.setToolTip("Compute True FSC curve from the two half-maps")
        self._btn_images2star = QPushButton("Images2Star")
        self._btn_images2star.setToolTip(
            "Open the Images2Star tools panel to preview and transform "
            "this dataset, then save a new RELION STAR/CryoSPARC file"
        )
        self._new_window_cb = QCheckBox("New window")
        self._new_window_cb.setToolTip(
            "<div style='white-space: normal; width: 200px;'>Create a new display window instead of reusing the current one. Most recently clicked/focused display window will be used as the display target if there are two or more display windows</div>"
        )
        for attr, mode in self._DISPLAY_BUTTONS:
            btn = getattr(self, attr)
            btn.setFixedHeight(30)
            btn.setIconSize(QSize(16, 16))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            # the web apps and the panels that open apart are marked from the viewers
            btn.setProperty("category", _ACTION_STYLE.get(mode, ("", "viewer"))[1])
            action_layout.addWidget(btn)
        action_layout.addWidget(self._new_window_cb)
        self._action_bar.hide()

        self._apply_theme(self._theme_combo.currentText())

        for attr, mode in self._DISPLAY_BUTTONS:
            btn = getattr(self, attr)
            btn.clicked.connect(partial(self._emit_display, mode))

        self._place_action_bar()

        # the buttons of the selected file are worked out in a thread (_ModesWorker)
        self._modes_worker: _ModesWorker | None = None
        self._modes_pending: str | None = None
        self._modes_serial = 0
        self._modes_applied = 0
        sel_model = self._tree.selectionModel()
        sel_model.selectionChanged.connect(self._on_selection_changed)

        self._tree.doubleClicked.connect(self._on_double_clicked)
        self._tree.clicked.connect(self._on_clicked)

        # Disabled widgets do not receive hover events, so Qt will not show
        # the ChimeraX "not found" tooltip on its own. This filter surfaces it
        # manually when the cursor is over the (disabled) button.
        self.installEventFilter(self)

        # a minimum size of our own: otherwise Qt sets it from what the preview shows, a new one
        # at each click, and over X11 (ssh -X) each is sent to the window manager, which
        # lays a maximized window out again: the window jitters
        self.setMinimumSize(640, 400)

        self._refresh_sidebar()
        self._update_preview()

        # Fill Dimension / Images / Pixel Size for the initial directory in
        # the background so the browser opens instantly (these columns start
        # empty and are populated by worker threads).
        self._populate_file_info_async()

    def _restore_panes(self) -> None:
        """Restore which side panes are shown, and how wide, from the last launch."""
        settings = QSettings("helicon", "display")
        for action, key in (
            (self._sidebar_action, "browser_show_sidebar"),
            (self._preview_action, "browser_show_preview"),
        ):
            action.setChecked(settings.value(key, True, type=bool))
        sizes = settings.value("browser_pane_sizes", [], type=list)
        try:
            sizes = [int(x) for x in sizes]
        except (TypeError, ValueError):
            sizes = []
        if len(sizes) == 3 and sizes[1] > 0:
            self._splitter.setSizes(sizes)

    def _save_panes(self) -> None:
        settings = QSettings("helicon", "display")
        settings.setValue("browser_show_sidebar", self._sidebar_action.isChecked())
        settings.setValue("browser_show_preview", self._preview_action.isChecked())
        sizes = self._splitter.sizes()
        # a hidden pane has no width; keep the last one it had
        if all(sizes[i] > 0 for i in (0, 2)):
            settings.setValue("browser_pane_sizes", sizes)

    def _on_sidebar_toggled(self, shown: bool) -> None:
        self._sidebar.setVisible(shown)

    def _on_preview_toggled(self, shown: bool) -> None:
        self._preview.setVisible(shown)
        self._place_action_bar()
        if shown:
            self._update_preview()

    def _place_action_bar(self) -> None:
        """Put the buttons of the selected file in the preview pane, below the facts of
        the file, or at the bottom of the window when the pane is hidden."""
        bar = getattr(self, "_action_bar", None)
        if bar is None:
            return  # the panes are restored before the buttons are made
        hidden = bar.isHidden()
        in_preview = self._preview_action.isChecked()
        if in_preview:
            self._main_layout.removeWidget(bar)
            self._preview.set_actions(bar)
            bar.layout().setContentsMargins(0, 2, 0, 2)
        else:
            self._preview.set_actions(None)
            self._main_layout.addWidget(bar)
            bar.layout().setContentsMargins(8, 8, 8, 8)
        bar.setProperty("place", "preview" if in_preview else "bottom")
        bar.style().unpolish(bar)
        bar.style().polish(bar)
        bar.setHidden(hidden)

    def _refresh_sidebar(self) -> None:
        self._sidebar.refresh(
            self._model._root_path,
            bookmarks=_get_bookmarks(),
            recent=_get_recent_folders(),
            bookmark_names=_get_bookmark_names(),
        )

    def _update_preview(self) -> None:
        """Preview the selected file or folder, or else the folder shown."""
        # the graph of the jobs (in the preview and in its own window) follows the folder shown
        self._preview.set_current(self._model._root_path)
        if not self._preview_action.isChecked():
            return
        indexes = self._tree.selectionModel().selectedIndexes()
        rows = sorted({idx.row() for idx in indexes})
        path = None
        if len(rows) == 1:
            path = self._model.file_path(self._model.index(rows[0], 0))
        self._preview.show_path(str(path) if path else self._model._root_path)

    def _set_action_icons(self, colors: dict[str, str]) -> None:
        """Icons on the action buttons, in the colour of the text (white when hovered)
        and of the accent for the apps. None where qtawesome is not installed."""
        try:
            import qtawesome
        except ImportError:
            return
        for attr, mode in self._DISPLAY_BUTTONS:
            name, category = _ACTION_STYLE.get(mode, ("fa5s.file", "viewer"))
            color = colors["app_text"] if category == "app" else colors["text"]
            getattr(self, attr).setIcon(
                qtawesome.icon(
                    name,
                    color=color,
                    color_active="#ffffff",
                    color_disabled=colors["disabled"],
                )
            )

    def _set_nav_icons(self, colors: dict[str, str]) -> None:
        """Icons for back, up and refresh in the colour of the theme (the arrows of the
        text stay where qtawesome is not installed)."""
        try:
            import qtawesome
        except ImportError:
            return
        for button, name in (
            (self._back_btn, "fa5s.arrow-left"),
            (self._up_btn, "fa5s.arrow-up"),
            (self._refresh_btn, "fa5s.sync-alt"),
        ):
            button.setIcon(qtawesome.icon(name, color=colors["text"]))
            button.setText("")

    def _on_theme_changed(self, theme: str) -> None:
        """Persist and immediately apply a newly selected browser theme."""
        _save_theme(theme)
        self._apply_theme(theme)
        try:
            from helicon.commands.display import _refresh_display_theme_windows

            _refresh_display_theme_windows()
        except Exception:
            # Keep the browser usable when auxiliary display dependencies are
            # unavailable or display.py is imported without the full Qt stack.
            pass

    def _on_theme_action_triggered(self, checked: bool = False) -> None:
        """Apply the theme selected from the View menu."""
        action = self.sender()
        if isinstance(action, QAction):
            theme = str(action.data())
            self._theme_combo.setCurrentText(theme)

    def _apply_theme(self, theme: str) -> None:
        """Apply a theme to the file browser and its launch buttons."""
        colors = _THEME_COLORS[_resolved_theme(theme)]
        stylesheet = _browser_stylesheet(colors)
        self.setStyleSheet(stylesheet)
        self._tree.setStyleSheet(stylesheet)
        self._recent_combo.setStyleSheet(stylesheet)
        self._theme_combo.setStyleSheet(stylesheet)
        self._model.set_theme_colors(colors)
        self._sidebar.set_theme(colors)
        self._preview.set_theme(colors)
        self._set_nav_icons(colors)
        self._set_action_icons(colors)
        for name, action in self._theme_actions.items():
            action.setChecked(name == theme)
        for attr, _mode in self._DISPLAY_BUTTONS:
            getattr(self, attr).setStyleSheet("")

    def _on_double_clicked(self, index: QModelIndex) -> None:
        path = self._model.file_path(index)
        if path and not self._model.is_dir(index):
            if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.file_selected_new_window.emit(path)
            else:
                self.file_selected.emit(path)

    def _on_clicked(self, index: QModelIndex) -> None:
        if self._model.is_dir(index):
            path = self._model.file_path(index)
            if path:
                self._navigate_to(path)

    def _display_modes_for(self, path: str) -> list[str]:
        """Return the applicable display modes for a single file.

        Detection order mirrors _get_file_type_label:
        1. Known file types by suffix
        2. Pure-text fallback via UTF-8 decode -> ["general"]
        3. Unknown binary -> []
        """
        from pathlib import Path

        try:
            if Path(path).stat().st_size == 0:
                return []
        except OSError:
            return []

        ext = Path(path).suffix.lower()
        if ext == ".star":
            name = Path(path).name.lower()
            if name.endswith("optimiser.star") or name.endswith("model.star"):
                _is_class2d = any(p.startswith("Class2D") for p in Path(path).parts)
                if _is_class2d:
                    return ["text", "2dclasses"]
                modes = ["text", "optimiser"]
                if name.endswith("model.star"):
                    modes.append("fsc")
                    _is_refine3d = any(
                        p.startswith("Refine3D") for p in Path(path).parts
                    )
                    if _is_refine3d:
                        modes.append("trueFSC")
                return modes
            if name.endswith("data.star"):
                _is_class2d = any(p.startswith("Class2D") for p in Path(path).parts)
                if _is_class2d:
                    # WhereIsMyClass, HelicalPitch and AbInitio3D are helical
                    # web apps and only make sense for helical 2D class averages.
                    modes = ["slice", "gallery", "text", "images2star"]
                    if _folder_is_helical(str(Path(path).parent)):
                        modes.extend(["whereIsMyClass", "helicalPitch", "abInitio3D"])
                    return modes
                _is_class3d_or_refine3d = any(
                    p.startswith("Class3D") or p.startswith("Refine3D")
                    for p in Path(path).parts
                )
                if _is_class3d_or_refine3d:
                    return ["slice", "gallery", "stats", "text", "images2star"]
            if any(name.endswith(s) for s in _METADATA_STAR_SUFFIXES):
                return ["text"]
            return ["slice", "gallery", "text", "images2star"]
        if ext == ".cs":
            # A CryoSPARC dataset that names images is an image stack, the way
            # a RELION data.star is: ``blob/path`` and ``blob/idx`` say where
            # each one lives. Those get the stack actions; a passthrough
            # dataset carries inherited parameters and no images, and a
            # volume's dataset points at a map, so both keep to images2star.
            from helicon.lib import cryosparc_project

            modes = ["images2star"]
            if cryosparc_project.has_image_refs(path):
                modes = ["slice", "gallery"] + modes
                if cryosparc_project.is_helical(path):
                    modes.append("denovo3D")
            # the particles of a helical 2D classification, with the class averages
            # beside them, are what HelicalPitch and AbInitio3D read (as a RELION
            # Class2D data.star is)
            if cryosparc_project.is_class2d_particles(path) and (
                cryosparc_project.is_helical(path) and _has_class_averages(path)
            ):
                modes.extend(["helicalPitch", "abInitio3D"])
            return modes
        if ext == ".mrcs":
            modes = ["slice", "gallery"]
            _is_class2d = any(p.startswith("Class2D") for p in Path(path).parts)
            _helical = _folder_is_helical(str(Path(path).parent))
            if _is_class2d and _helical:
                modes.extend(["helicalProjection", "hill"])
            if _helical:
                modes.append("denovo3D")
            return modes
        if ext in _MRC_VOLUME_EXTENSIONS:
            _has_volume = self._mrc_is_volume(path)
            if _has_volume:
                modes = [
                    "slice",
                    "volume",
                    "gallery",
                    "chimerax",
                    "orthogonal",
                    "proc3d",
                ]
            else:
                # Single-slice (nz == 1) or unreadable .mrc/.map files are
                # treated as 2D image stacks, not 3D volumes -- as are the
                # many-image stacks CryoSPARC writes with an .mrc suffix.
                modes = ["slice", "gallery"]
                from helicon.lib import cryosparc_project

                if cryosparc_project.is_class_averages(path):
                    # Sorting by abundance needs the particles dataset that
                    # records each particle's class; offered only when the
                    # companion files are actually there.
                    if cryosparc_project.has_class_abundance(path):
                        modes.append("2dclasses")
                    if cryosparc_project.is_helical(path):
                        modes.extend(["helicalProjection", "hill", "denovo3D"])
                        # it needs the particles dataset as well as the averages
                        if cryosparc_project.has_class_abundance(path):
                            modes.append("abInitio3D")
            if _has_volume:
                _is_class3d_or_refine3d = any(
                    p.startswith("Class3D") or p.startswith("Refine3D")
                    for p in Path(path).parts
                )
                if _is_class3d_or_refine3d and _folder_is_helical(
                    str(Path(path).parent)
                ):
                    modes.append("hi3d")
                else:
                    from helicon.lib import cryosparc_project

                    if cryosparc_project.is_helical(path):
                        modes.append("hi3d")
            return modes
        if ext == ".bild":
            return ["text", "3dplot", "chimerax"]
        if ext == ".pdf":
            return ["slice"]
        if ext == ".eps":
            return ["slice"]
        if ext in _RASTER_IMAGE_EXTENSIONS:
            return ["slice"]
        if ext in (".html", ".htm"):
            return ["html"]
        if ext in _KNOWN_EXTENSIONS:
            return []

        if _is_text_file(path):
            return ["text"]

        return []

    def _mrc_is_volume(self, path: str) -> bool:
        """Whether an MRC-format file holds a 3D volume rather than a stack.

        The header alone cannot say. CryoSPARC writes everything as ``.mrc``,
        so a stack of 100 class averages has nz=100 and looks exactly like a
        map -- one such stack measured 200 x 200 x 200, which no rule about
        dimensions could ever call correctly. Where the file sits in a
        CryoSPARC project, its job type and CryoSPARC's own naming say which
        it is; everywhere else the Z extent is still the best signal there is.
        """
        from helicon.lib import cryosparc_project

        kind = cryosparc_project.mrc_content(path)
        if kind == "volume":
            return True
        if kind == "stack":
            return False
        return self._volume_has_nz_gt1(path)

    def _volume_has_nz_gt1(self, path: str) -> bool:
        """Return True if an MRC-format file has more than one Z slice."""
        try:
            import mrcfile

            # Read only the header: the data block can be gigabytes.
            with mrcfile.open(path, permissive=True, header_only=True) as mrc:
                header = mrc.header
                nz = int(header.nz)
                mz = int(header.mz)
                ispg = int(header.ispg)
            # A volume stack (space group 401-630 with several volumes) is
            # 4D, not a single volume.
            if 401 <= ispg <= 630 and 0 < mz < nz:
                return False
            return nz > 1
        except Exception:
            return False

    def _is_image_stack(self, path: str) -> bool:
        """Return True for files that are stacks of 2D images.

        This covers ``.mrcs`` particle stacks, data ``.star`` files (which
        reference many individual 2D images), and single-slice ``.mrc`` /
        ``.map`` files (nz == 1) that carry one 2D image, plus common raster
        image formats (PNG, JPEG, TIFF, ...). Genuine volumes (nz > 1) are
        not stacks and keep the "2D Slice" label instead.
        """
        from pathlib import Path

        ext = Path(path).suffix.lower()
        if ext == ".mrcs":
            return True
        if ext == ".star":
            name = Path(path).name.lower()
            if any(name.endswith(s) for s in _METADATA_STAR_SUFFIXES):
                return False
            return True
        if ext in _MRC_VOLUME_EXTENSIONS:
            return not self._mrc_is_volume(path)
        if ext in _RASTER_IMAGE_EXTENSIONS:
            return True
        return False

    def _is_volume(self, path: str) -> bool:
        """Return True for genuine 3D volume files (MRC-format files with more
        than one Z slice).

        These expose both a 2D-slice mode and a 3D-volume mode; the slice
        button is labelled "2D Slice" to distinguish it from a 2D image.
        """
        from pathlib import Path

        return Path(path).suffix.lower() in _MRC_VOLUME_EXTENSIONS and (
            self._mrc_is_volume(path)
        )

    def _on_selection_changed(self, selected, deselected) -> None:
        self._update_preview()
        indexes = self._tree.selectionModel().selectedIndexes()
        # selectedIndexes() returns one index per column; collapse to rows.
        rows = sorted({idx.row() for idx in indexes})
        self._current_path = None
        # what is being worked out for an earlier file is dropped
        self._modes_serial += 1
        if len(rows) != 1:
            self._action_bar.hide()
            self._modes_applied = self._modes_serial
            return
        index = self._model.index(rows[0], 0)
        path = self._model.file_path(index)
        if not path or self._model.is_dir(index):
            self._action_bar.hide()
            self._modes_applied = self._modes_serial
            return
        # the buttons depend on what is in the file (and its folder), which can mean
        # reading headers and datasets, on a network disk too: not on the main thread
        self._action_bar.hide()
        self._modes_pending = str(path)
        if self._modes_worker is None or not self._modes_worker.isRunning():
            self._start_modes()

    def _start_modes(self) -> None:
        path, self._modes_pending = self._modes_pending, None
        if path is None:
            return
        worker = _ModesWorker(self._modes_serial, path, self)
        worker.ready.connect(self._on_modes_ready)
        worker.finished.connect(self._on_modes_finished)
        self._modes_worker = worker
        worker.start()

    def _on_modes_finished(self) -> None:
        if self._modes_pending is not None:
            self._start_modes()

    def _on_modes_ready(self, serial: int, info: dict) -> None:
        if serial != self._modes_serial:
            return  # another file was selected meanwhile
        self._apply_modes(info)

    def wait_for_selection(self, timeout_ms: int = 10000) -> None:
        """Wait until the buttons of the selected file are shown (for tests and scripts)."""
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            worker = self._modes_worker
            if worker is not None and worker.isRunning():
                worker.wait(50)
            QApplication.processEvents()
            if (
                self._modes_pending is None
                and (worker is None or not worker.isRunning())
                and self._modes_serial == self._modes_applied
            ):
                return

    def _apply_modes(self, info: dict) -> None:
        """Show the buttons for what the worker found: ``path``, ``modes``, ``slice_text``
        and whether ChimeraX is installed."""
        path, modes = info["path"], info["modes"]
        self._modes_applied = self._modes_serial
        if not modes:
            self._action_bar.hide()
            return
        for attr, mode in self._DISPLAY_BUTTONS:
            getattr(self, attr).setVisible(mode in modes)
        if "chimerax" in modes:
            if not info["chimerax"]:
                self._btn_chimerax.setEnabled(False)
                # Explicit disabled styling so the button is unambiguously
                # greyed-out regardless of the active Qt theme.
                colors = _THEME_COLORS[_resolved_theme(_saved_theme())]
                self._btn_chimerax.setStyleSheet(_disabled_button_stylesheet(colors))
                self._btn_chimerax.setToolTip(
                    "ChimeraX not found. Install it from "
                    "https://www.cgl.ucsf.edu/chimerax/ or add it to your PATH."
                )
            else:
                self._btn_chimerax.setEnabled(True)
                self._btn_chimerax.setStyleSheet("")
                self._btn_chimerax.setToolTip("Open this file in ChimeraX")
        if "stats" in modes:
            self._btn_stats.setEnabled(True)
            self._btn_stats.setStyleSheet("")
            self._btn_stats.setToolTip(
                "Estimate and display per-filament tilt, psi, and rot-angle variance"
            )
        self._btn_slice.setText(info["slice_text"])
        self._current_path = str(path)
        self._reorder_buttons_alphabetically(modes)
        self._action_bar.show()

    def _emit_display(self, mode: str) -> None:
        if not getattr(self, "_current_path", None):
            return
        self.display_requested.emit(
            self._current_path, mode, self._new_window_cb.isChecked()
        )

    def _reorder_buttons_alphabetically(self, modes: list[str]) -> None:
        """Re-order visible display-mode buttons alphabetically by label.

        All display buttons are removed from the layout and only the non-hidden
        ones are re-inserted, sorted by their current text label.  Visibility
        is controlled by ``_on_selection_changed`` (which calls ``setVisible``
        per mode) before this method runs.  Because the button set is driven by
        ``_DISPLAY_BUTTONS``, any new button added there is automatically
        included — no separate mapping to maintain.
        """
        layout = self._action_bar.layout()
        buttons = [getattr(self, attr) for attr, _ in self._DISPLAY_BUTTONS]
        for btn in buttons:
            layout.removeWidget(btn)
        visible = [btn for btn in buttons if not btn.isHidden()]
        visible.sort(key=lambda b: b.text().lower())
        for btn in reversed(visible):
            layout.insertWidget(0, btn)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QToolTip

        # Disabled buttons do not emit hover events, so Qt will not show
        # their tooltips on its own. Surface them manually on mouse-move.
        if event.type() == QEvent.MouseMove:
            for btn in (self._btn_chimerax, self._btn_stats):
                if not btn.isEnabled() and btn.isVisible():
                    local = btn.mapFromGlobal(QCursor.pos())
                    if btn.rect().contains(local):
                        QToolTip.showText(QCursor.pos(), btn.toolTip(), btn)
                        break
            else:
                QToolTip.hideText()
        return False

    def _navigate_to(self, path: str) -> None:
        path = str(Path(path).resolve())
        if path != self._history[self._history_index]:
            # (going to the folder one is in again is a refresh, not a new step back)
            if self._history_index < len(self._history) - 1:
                self._history = self._history[: self._history_index + 1]
            self._history.append(path)
            self._history_index = len(self._history) - 1
        self._model.set_root_path(path)
        sort_col = self._tree.header().sortIndicatorSection()
        sort_order = self._tree.header().sortIndicatorOrder()
        self._model.sort(sort_col, sort_order)
        self._path_edit.setText(path)
        self._refresh_recent_combo()
        self._refresh_recent_menu()
        self._refresh_sidebar()
        self._update_preview()
        # (the columns that need reading a file are filled when the model says that
        # the folder is all there: load_finished)

    def _toggle_bookmark(self) -> None:
        """Bookmark the folder shown, or remove its bookmark if it has one."""
        path = self._model._root_path
        if _is_bookmarked(path):
            _remove_bookmark(path)
        else:
            _add_bookmark(path)
        self._refresh_bookmarks_menu()

    def _refresh_bookmarks_menu(self) -> None:
        """Fill the Bookmarks menu: the toggle, then one entry per bookmark."""
        menu = self._bookmarks_menu
        menu.clear()
        if hasattr(self, "_sidebar"):
            self._refresh_sidebar()
        here = self._model._root_path
        self._bookmark_toggle_action.setText(
            "Remove Bookmark of This Folder"
            if _is_bookmarked(here)
            else "Bookmark This Folder"
        )
        menu.addAction(self._bookmark_toggle_action)
        menu.addSeparator()
        marks = _get_bookmarks()
        if not marks:
            menu.addAction("No bookmarks yet").setEnabled(False)
            return
        for path in marks:
            action = menu.addAction(_bookmark_text(path))
            if Path(path).is_dir():
                action.triggered.connect(partial(self._on_bookmark_selected, path))
            else:
                action.setText(f"{_bookmark_text(path)}  (not available)")
                action.setEnabled(False)
        menu.addSeparator()
        remove_menu = menu.addMenu("Remove Bookmark")
        for path in marks:
            action = remove_menu.addAction(_bookmark_text(path))
            action.triggered.connect(partial(self._on_bookmark_removed, path))
        menu.addAction("Manage Bookmarks…").triggered.connect(self._manage_bookmarks)
        menu.addAction("Clear All Bookmarks").triggered.connect(
            self._clear_all_bookmarks
        )

    def _on_bookmark_selected(self, path: str) -> None:
        """Go to a bookmarked folder."""
        if Path(path).is_dir():
            self._navigate_to(path)

    def _manage_bookmarks(self) -> None:
        """Open the dialog to reorder or remove the bookmarks."""
        _BookmarksDialog(self).exec()
        self._refresh_bookmarks_menu()

    def _clear_all_bookmarks(self) -> None:
        if _confirm_clear_bookmarks(self):
            _clear_bookmarks()
            self._refresh_bookmarks_menu()

    def _on_bookmark_removed(self, path: str) -> None:
        _remove_bookmark(path)
        self._refresh_bookmarks_menu()

    def _on_bookmark_renamed(self, path: str, name: str) -> None:
        _rename_bookmark(path, name)
        self._refresh_bookmarks_menu()

    def _on_alias_requested(self, folder: str, alias: str) -> None:
        """Give a RELION job an alias, or take it away (``alias`` empty), and show the change."""
        from helicon.lib.gui.job_alias import set_alias

        try:
            set_alias(folder, alias or None)
        except (ValueError, OSError) as e:
            QMessageBox.warning(self, "Alias of the job", str(e))
            return
        self._sidebar.invalidate()
        self._navigate_to(
            self._model._root_path
        )  # the list shows the job with its alias

    def _refresh_recent_combo(self) -> None:
        self._recent_combo.blockSignals(True)
        self._recent_combo.clear()
        self._recent_combo.addItems(_get_recent_folders())
        self._recent_combo.setCurrentIndex(-1)
        self._recent_combo.blockSignals(False)

    def _refresh_recent_menu(self) -> None:
        """Populate the File menu's recent-folder submenu."""
        self._recent_menu.clear()
        recent = _get_recent_folders()
        if not recent:
            action = self._recent_menu.addAction("No recent folders")
            action.setEnabled(False)
            return
        for path in recent:
            action = self._recent_menu.addAction(path)
            action.triggered.connect(partial(self._on_recent_path_selected, path))

    def _on_recent_path_selected(self, path: str) -> None:
        """Navigate to a folder selected from the File menu."""
        if Path(path).is_dir():
            self._navigate_to(path)

    def _on_recent_selected(self, index: int) -> None:
        path = self._recent_combo.itemText(index)
        if path and Path(path).is_dir():
            self._navigate_to(path)

    def _open_folder(self) -> None:
        """Show a folder chooser and navigate to the selected directory."""
        path = QFileDialog.getExistingDirectory(
            self,
            "Open Folder",
            self._model._root_path,
        )
        if path:
            self._navigate_to(path)

    def _open_terminal(self) -> None:
        """Open the host OS terminal in the currently displayed folder."""
        _open_terminal(self._model._root_path)

    def _configure_relion(self) -> None:
        settings = QSettings("helicon", "display")
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure RELION")
        layout = QVBoxLayout(dialog)
        help_text = QLabel(
            "Select a Bash launcher script that loads your RELION module and/or\n"
            "activates its conda environment, then runs exec relion.\n"
            "Clean environment: initialize all required software in the script.\n"
            "User Bash setup: inherit the environment and read login profiles,\n"
            "then source the script in that shell. Switch conda environments as needed.\n\n"
            "With no script, use relion from the current environment (Clean)\n"
            "or from the initialized login shell (User Bash setup).\n"
            "HELICON_RELION_LAUNCHER, when set, overrides this saved choice.\n"
            "Queued jobs also need environment setup in their submission template."
        )
        layout.addWidget(help_text)
        startup_combo = QComboBox()
        startup_combo.addItem("Clean environment (default)", "clean")
        startup_combo.addItem("User Bash setup (interactive login shell)", "user-shell")
        startup_combo.setCurrentIndex(
            max(
                0,
                startup_combo.findData(str(settings.value("relion/startup", "clean"))),
            )
        )
        layout.addWidget(startup_combo)
        row = QHBoxLayout()
        path_edit = QLineEdit(str(settings.value("relion/launcher", "") or ""))
        path_edit.setPlaceholderText("Absolute path to launcher script (optional)")
        row.addWidget(path_edit)
        browse = QPushButton("Browse…")

        def choose_script():
            path, _ = QFileDialog.getOpenFileName(dialog, "RELION launcher script")
            if path:
                path_edit.setText(path)

        browse.clicked.connect(choose_script)
        row.addWidget(browse)
        layout.addLayout(row)
        override = os.environ.get("HELICON_RELION_LAUNCHER", "")
        if override:
            label = QLabel(f"Environment override: {override}")
            label.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )

        def save():
            value = path_edit.text().strip()
            path = Path(value).expanduser()
            if value and (not path.is_absolute() or not path.is_file()):
                QMessageBox.warning(
                    dialog,
                    "RELION launcher",
                    "Select an existing script using an absolute path.",
                )
                return
            settings.setValue("relion/launcher", str(path) if value else "")
            settings.setValue("relion/startup", startup_combo.currentData())
            dialog.accept()

        buttons.accepted.connect(save)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _open_relion(self) -> None:
        from helicon.lib.relion_launcher import launch_relion

        project = str(Path(self._model._root_path).resolve())
        existing = self._relion_processes.get(project)
        if existing and existing[0].poll() is None:
            QMessageBox.information(
                self, "RELION", "RELION is already running for this project."
            )
            return
        launcher = os.environ.get("HELICON_RELION_LAUNCHER") or str(
            QSettings("helicon", "display").value("relion/launcher", "") or ""
        )
        startup = str(QSettings("helicon", "display").value("relion/startup", "clean"))
        try:
            process, log = launch_relion(project, launcher, startup=startup)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Cannot open RELION", str(exc))
            return
        self._relion_processes[project] = (process, log)
        self.statusBar().showMessage(f"RELION starting. Launch log: {log}")
        self._relion_timer.start()

    def _poll_relion(self) -> None:
        for project, (process, log) in list(self._relion_processes.items()):
            code = process.poll()
            if code is None:
                continue
            del self._relion_processes[project]
            if code:
                QMessageBox.warning(
                    self,
                    "RELION exited",
                    f"RELION exited with status {code}.\nProject: {project}\n"
                    f"See the launch log: {log}\nCheck Apps → Configure RELION…",
                )
        if not self._relion_processes:
            self._relion_timer.stop()

    def _on_launch_app(self, checked: bool = False) -> None:
        """Launch the standalone tool listed in the Apps menu.

        Shiny web-app tabs reuse the same launch pipeline as the display
        buttons (``_launch_or_reuse_web_app``) but with no input file, so the
        app opens with its own file-picker ready. Streamlit subcommands are
        launched through their command module's ``main``.
        The Proc3D, Images2Star and trueFSC entries pick their inputs first.
        """
        action = self.sender()
        if not isinstance(action, QAction):
            return
        key, streamlit_mod, picker = action.data()
        if picker == "map":
            self._on_pick_proc3d_map()
            return
        if picker == "star":
            self._on_pick_images2star_file()
            return
        if picker == "two_maps":
            self._on_pick_truefsc_maps()
            return
        if streamlit_mod is not None:
            self._launch_streamlit_app(streamlit_mod)
            return
        self._launch_web_app_tab(key)

    def _on_pick_proc3d_map(self) -> None:
        """Open the Proc3D tools panel with its in-panel file selector."""
        try:
            from helicon.commands import display as _display

            _display._open_proc3d_tools(
                None,
                parent=self,
                reuse_window=(
                    None
                    if self._new_window_cb.isChecked()
                    else _display._proc3d.active()
                ),
                tracker=_display._proc3d,
            )
        except Exception:
            import logging

            logging.getLogger("helicon").exception("failed to launch Proc3D")

    def _on_pick_images2star_file(self) -> None:
        """Open the Images2Star tools panel with its in-panel file selector."""
        try:
            from helicon.commands import display as _display

            _display._open_images2star_tools(
                None,
                parent=self,
                reuse_window=(
                    None
                    if self._new_window_cb.isChecked()
                    else _display._images2star.active()
                ),
                tracker=_display._images2star,
            )
        except Exception:
            import logging

            logging.getLogger("helicon").exception("failed to launch Images2Star")

    def _on_pick_truefsc_maps(self) -> None:
        """Open the trueFSC panel with empty input map selectors."""
        try:
            from helicon.commands import display as _display

            _display._launch_truefsc_maps(parent=self)
        except Exception:
            import logging

            logging.getLogger("helicon").exception("failed to launch trueFSC")

    def _launch_web_app_tab(self, tab: str) -> None:
        """Open *tab* in Helicon Lab with no pre-set input file."""
        try:
            from helicon.commands import display as _display

            _display._launch_or_reuse_web_app(
                tab,
                _display._make_bookmark_query(tab, {}),
                new_window=False,
            )
        except Exception:
            import logging

            logging.getLogger("helicon").exception(
                "failed to launch web app tab %r", tab
            )

    @staticmethod
    def _launch_streamlit_app(module_name: str) -> None:
        """Run a streamlit subcommand (e.g. ctfSimulation) in a new process.

        The subcommand ``main`` blocks until the server exits (``streamlit
        run``), so it must run detached from the GUI event loop.
        """
        # Import the command module and invoke ``main`` with a freshly parsed
        # (empty) argument namespace, rather than running its ``__main__``
        # block.  ``runpy.run_module(..., alter_sys=True)`` leaves the module
        # name in ``sys.argv`` and the ``__main__`` block's ``parse_args()``
        # then rejects that unrecognized argument, so the app never starts.
        cmd = [
            sys.executable,
            "-c",
            (
                "import importlib,sys;"
                "m=importlib.import_module(sys.argv[1]);"
                "import argparse;"
                "p=argparse.ArgumentParser();"
                "(m.add_args(p) if hasattr(m,'add_args') else None);"
                "m.main(p.parse_args([]))"
            ),
            module_name,
        ]
        try:
            _spawn_detached(cmd)
        except Exception:
            import logging

            logging.getLogger("helicon").exception(
                "failed to launch streamlit app %r", module_name
            )

    def _quit(self) -> None:
        """Exit the application (File → Quit / Ctrl+Q)."""
        app = QApplication.instance()
        if app is not None:
            app.quit()
        else:
            self.close()

    def _open_home_page(self) -> None:
        """Open the Helicon project home page in the system browser."""
        _open_url("https://jianglab.science.psu.edu/helicon")

    def _open_documentation(self) -> None:
        """Open the Helicon user docs in the system browser."""
        _open_url("https://helicon.readthedocs.io")

    def _go_up(self) -> None:
        current = self._model._root_path
        parent = str(Path(current).parent)
        if parent != current:
            self._navigate_to(parent)

    def _go_back(self) -> None:
        if self._history_index > 0:
            self._history_index -= 1
            path = self._history[self._history_index]
            self._model.set_root_path(path)
            sort_col = self._tree.header().sortIndicatorSection()
            sort_order = self._tree.header().sortIndicatorOrder()
            self._model.sort(sort_col, sort_order)
            self._path_edit.setText(path)
            self._refresh_sidebar()
            self._update_preview()

    def _go_to_path(self) -> None:
        path = self._path_edit.text().strip()
        if path and Path(path).is_dir():
            self._navigate_to(path)

    def _refresh(self) -> None:
        self._sidebar.invalidate()
        self._refresh_sidebar()
        self._model.refresh()

    def _populate_file_info_async(self) -> None:
        """Fill Dimension / Images / Pixel Size and folder counts in the background.

        The model rows are created synchronously with those columns empty so
        the folder opens instantly; this method then resolves the per-file
        metadata and per-folder entry counts off the GUI thread using small
        pools of worker threads. Old workers from a previous directory load
        are dropped (their results are discarded via the epoch guard and they
        self-delete on finish), and every result carries the model epoch so
        stale results are ignored.
        """
        from helicon.lib.relion_launcher import is_relion_project

        self._relion_btn.setVisible(is_relion_project(self._model._root_path))
        # Stop tracking any in-flight workers from a previous load.
        self._info_threads.clear()

        files, dirs = self._model.listed_paths()
        file_tasks = [(-1, path) for path in files]  # (the workers need no row)
        dir_tasks = [(-1, path) for path in dirs]
        if not file_tasks and not dir_tasks:
            return

        epoch = self._model.current_epoch()
        sort_col = self._tree.header().sortIndicatorSection()
        re_sort_needed = sort_col in (COL_SIZE, COL_IMAGES, COL_INFO, COL_PIXELSIZE)
        pending = 0

        def on_info_ready(batch, result_epoch):
            if result_epoch != self._model.current_epoch():
                return  # directory was reloaded; ignore stale result
            self._model.apply_file_infos(batch)

        def on_count_ready(batch, result_epoch):
            if result_epoch != self._model.current_epoch():
                return  # directory was reloaded; ignore stale result
            self._model.apply_dir_counts(batch)

        def on_worker_finished():
            nonlocal pending
            pending -= 1
            if pending == 0 and re_sort_needed:
                self._model.sort(sort_col, self._tree.header().sortIndicatorOrder())

        if file_tasks:
            num_workers = min(4, len(file_tasks))
            chunks = [file_tasks[i::num_workers] for i in range(num_workers)]
            pending += len(chunks)
            for chunk in chunks:
                worker = InfoWorker(chunk, epoch)
                worker.info_ready.connect(on_info_ready)
                # Defer deletion until after any still-queued info_ready events
                # from this worker have been delivered to the GUI thread.
                worker.finished.connect(
                    lambda w=worker: QTimer.singleShot(0, w.deleteLater)
                )
                worker.finished.connect(on_worker_finished)
                self._info_threads.append(worker)
                worker.start()

        if dir_tasks:
            num_workers = min(4, len(dir_tasks))
            chunks = [dir_tasks[i::num_workers] for i in range(num_workers)]
            pending += len(chunks)
            filter_pattern = self._model._filter_pattern
            use_regex = self._model._use_regex
            for chunk in chunks:
                worker = DirCountWorker(chunk, epoch, filter_pattern, use_regex)
                worker.count_ready.connect(on_count_ready)
                worker.finished.connect(
                    lambda w=worker: QTimer.singleShot(0, w.deleteLater)
                )
                worker.finished.connect(on_worker_finished)
                self._info_threads.append(worker)
                worker.start()

    def _apply_filter(self) -> None:
        pattern = self._filter_edit.text().strip()
        if not pattern:
            pattern = "*"
        use_regex = self._regex_cb.isChecked()
        self._model.set_filter(pattern, use_regex)

    def _copy_selection(self) -> None:
        indexes = self._tree.selectionModel().selectedIndexes()
        if not indexes:
            return

        rows = sorted(set(idx.row() for idx in indexes))
        cols = sorted(set(idx.column() for idx in indexes))

        lines = []
        for row in rows:
            cells = []
            for col in cols:
                idx = self._model.index(row, col)
                text = self._model.data(idx, Qt.ItemDataRole.DisplayRole) or ""
                cells.append(text)
            lines.append("\t".join(cells))

        QApplication.clipboard().setText("\n".join(lines))

    def _show_context_menu(self, pos):
        index = self._tree.indexAt(pos)
        if not index.isValid():
            return

        file_path = self._model.file_path(index)
        if not file_path:
            return

        menu = QMenu(self._tree)

        copy_path_action = QAction("Copy Path", self._tree)
        copy_path_action.triggered.connect(
            lambda: QApplication.clipboard().setText(file_path)
        )
        menu.addAction(copy_path_action)

        copy_name_action = QAction("Copy Name", self._tree)
        copy_name_action.triggered.connect(
            lambda: QApplication.clipboard().setText(Path(file_path).name)
        )
        menu.addAction(copy_name_action)

        if Path(file_path).is_dir():
            menu.addSeparator()
            marked = _is_bookmarked(file_path)
            bookmark_action = QAction(
                "Remove Bookmark" if marked else "Bookmark This Folder", self._tree
            )
            bookmark_action.triggered.connect(
                lambda: (_remove_bookmark if marked else _add_bookmark)(file_path)
            )
            menu.addAction(bookmark_action)

        menu.exec_(self._tree.viewport().mapToGlobal(pos))

    def set_root_path(self, path: str | Path) -> None:
        self._navigate_to(str(path))

    def _restore_col_widths(self) -> None:
        """Restore column widths saved from a previous launch.

        Stored as a comma-separated string under the same QSettings group as
        the window geometry (``helicon`` / ``display``). Only columns that
        still exist are applied, so adding/removing columns later is safe.
        """
        try:
            settings = QSettings("helicon", "display")
            raw = settings.value("browser_colwidths")
            if not raw:
                return
            widths = [int(x) for x in str(raw).split(",") if x.strip()]
            header = self._tree.header()
            for col, w in enumerate(widths):
                if col < NUM_COLUMNS and w > 0:
                    header.resizeSection(col, w)
        except (TypeError, ValueError, RuntimeError):
            pass

    def _save_col_widths(self) -> None:
        """Persist current column widths to QSettings."""
        try:
            header = self._tree.header()
            widths = [header.sectionSize(col) for col in range(NUM_COLUMNS)]
            settings = QSettings("helicon", "display")
            settings.setValue("browser_colwidths", ",".join(str(w) for w in widths))
        except RuntimeError:
            pass

    def _restore_sort_state(self) -> None:
        """Restore sort column and order saved from a previous launch."""
        try:
            settings = QSettings("helicon", "display")
            col = settings.value("browser_sort_col")
            order = settings.value("browser_sort_order")
            if col is None or order is None:
                return
            col = int(col)
            order = Qt.SortOrder(int(order))
            if 0 <= col < NUM_COLUMNS:
                self._tree.sortByColumn(col, order)
        except (TypeError, ValueError):
            pass

    def _save_sort_state(self) -> None:
        """Persist current sort column and order to QSettings."""
        try:
            header = self._tree.header()
            settings = QSettings("helicon", "display")
            settings.setValue("browser_sort_col", header.sortIndicatorSection())
            settings.setValue("browser_sort_order", header.sortIndicatorOrder().value)
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        try:
            self._save_col_widths()
            self._save_sort_state()
            self._save_panes()
            self._preview.wait()
        except Exception:
            pass
        super().closeEvent(event)
