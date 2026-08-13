"""Non-napari file openers used by the display command.

Text (editable standalone window with find/save), HTML, PDF (rasterized
pages), EPS (Ghostscript/Quick-Look rasterization), and BILD (mesh
surfaces in napari) plus the mesh builders they share.
"""

from __future__ import annotations

import os
from pathlib import Path

from .theme import _display_theme_palette, _display_theme_stylesheet
from .trackers import _install_window_shortcuts, _text
from .viewer import (
    _SliceDirectionWidget,
    _auto_contrast,
    _enable_continuous_auto_contrast,
    _hide_layer_panels,
    _reset_view,
)


def _is_text_file(path: str) -> bool:
    """Check if a file is a text file by attempting UTF-8 decode.

    Called only after known-type suffix checks have failed, so this is the
    fallback that distinguishes pure-text files from unknown binary.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
        if b"\x00" in chunk:
            return False
        chunk.decode("utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False


def _open_text_window(path, reuse_window=None):
    """Open a text file in a standalone text window."""
    from pathlib import Path

    try:
        from PySide6.QtWidgets import QMainWindow, QTextEdit
        from PySide6.QtGui import QFont, QShortcut, QKeySequence
    except ImportError:
        return

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return

    if reuse_window is not None:
        try:
            if reuse_window.isVisible():
                if not reuse_window._maybe_save_before_replace(Path(path).name):
                    return reuse_window
                reuse_window._source_path = path
                reuse_window._text_edit.setPlainText(content)
                reuse_window._text_edit.document().setModified(False)
                reuse_window.setWindowTitle(f"Helicon - {Path(path).name}")
                reuse_window.show()
                reuse_window.raise_()
                return reuse_window
        except Exception:
            pass

    class _TextWindow(QMainWindow):
        def __init__(self, parent=None):
            super().__init__(parent)
            from PySide6.QtWidgets import (
                QWidget,
                QVBoxLayout,
                QHBoxLayout,
                QLineEdit,
                QToolButton,
            )
            from PySide6.QtCore import Qt

            self._source_path = path
            self.setProperty("helicon_theme_window", True)
            self.setStyleSheet(_display_theme_stylesheet())
            self.setPalette(_display_theme_palette())
            self.setWindowTitle(f"Helicon - {Path(path).name}")
            self.resize(700, 500)

            central = QWidget()
            layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            te = QTextEdit(self)
            # Editable: the buffer can be edited and saved to a *new* file via
            # Save As…; the original file on disk is never overwritten.
            te.setReadOnly(False)
            te.setFont(QFont("Courier New", 12))
            te.setLineWrapMode(QTextEdit.WidgetWidth)
            te.setStyleSheet(_display_theme_stylesheet())
            self._text_edit = te
            layout.addWidget(te, 1)

            action_bar = QWidget()
            action_bar.setStyleSheet(_display_theme_stylesheet())
            action_layout = QHBoxLayout(action_bar)
            action_layout.setContentsMargins(6, 4, 6, 4)
            action_layout.setSpacing(6)

            save_btn = QToolButton()
            save_btn.setText("Save As…")
            save_btn.setToolTip("Save the edited text to a new file (Ctrl+Shift+S)")
            save_btn.setStyleSheet(_display_theme_stylesheet())
            save_btn.clicked.connect(self._save_as)
            action_layout.addWidget(save_btn)
            action_layout.addStretch(1)
            layout.addWidget(action_bar)

            find_bar = QWidget()
            find_bar.setStyleSheet(_display_theme_stylesheet())
            find_layout = QHBoxLayout(find_bar)
            find_layout.setContentsMargins(6, 4, 6, 4)
            find_layout.setSpacing(6)

            find_input = QLineEdit()
            find_input.setPlaceholderText("Find…")
            find_input.setStyleSheet(_display_theme_stylesheet())
            find_input.returnPressed.connect(self._find_next)
            self._find_input = find_input

            close_btn = QToolButton()
            close_btn.setText("✕")
            close_btn.setToolTip("Close find bar")
            close_btn.clicked.connect(lambda: self._toggle_find_bar(False))
            close_btn.setStyleSheet(_display_theme_stylesheet())

            find_layout.addWidget(find_input, 1)
            find_layout.addWidget(close_btn)
            self._find_bar = find_bar
            self._find_bar_visible = False
            find_bar.hide()
            layout.addWidget(find_bar)

            self.setCentralWidget(central)

            find_sc = QShortcut(QKeySequence.StandardKey.Find, self)
            find_sc.activated.connect(lambda: self._toggle_find_bar(True))

            esc_sc = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
            esc_sc.activated.connect(self._close_find_bar)

            wrap_sc = QShortcut(QKeySequence("Ctrl+Shift+W"), self)
            wrap_sc.activated.connect(self._toggle_wrap)

            save_as_sc = QShortcut(QKeySequence("Ctrl+Shift+S"), self)
            save_as_sc.activated.connect(self._save_as)

            _install_window_shortcuts(self)

            self._text_edit.document().modificationChanged.connect(self._on_modified)

        def _on_modified(self, modified: bool) -> None:
            title = f"Helicon - {Path(self._source_path).name}"
            if modified:
                title += " *"
            self.setWindowTitle(title)

        def _save_as(self) -> None:
            from PySide6.QtWidgets import QFileDialog, QMessageBox

            source = Path(self._source_path)
            default = str(source.with_name(f"{source.stem}_edited{source.suffix}"))
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save As…",
                default,
                "All Files (*)",
            )
            if not filename:
                return
            try:
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(self._text_edit.toPlainText())
            except OSError as exc:
                QMessageBox.warning(
                    self, "Save failed", f"Could not write {filename}:\n{exc}"
                )
                return
            self._source_path = filename
            self._text_edit.document().setModified(False)
            self.statusBar().showMessage(f"Saved to {filename}", 4000)

        def _maybe_save_before_replace(self, new_name: str) -> bool:
            """Return True if it is safe to replace the buffer with *new_name*."""
            if not self._text_edit.document().isModified():
                return True
            from PySide6.QtWidgets import QMessageBox

            return self._confirm_unsaved(
                f"Save changes to {Path(self._source_path).name} "
                f"before opening {new_name}?"
            )

        def _maybe_save_before_close(self) -> bool:
            """Return True if it is safe to close the window."""
            if not self._text_edit.document().isModified():
                return True
            return self._confirm_unsaved(
                f"Save changes to {Path(self._source_path).name} before closing?"
            )

        def _confirm_unsaved(self, message: str) -> bool:
            """Ask Save / Discard / Cancel for unsaved changes.

            Returns True when the buffer may be discarded (saved or discarded
            explicitly); False when the user cancelled.
            """
            from PySide6.QtWidgets import QMessageBox

            buttons = (
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel
            )
            ret = QMessageBox.question(
                self,
                "Unsaved changes",
                message,
                buttons,
                QMessageBox.StandardButton.Save,
            )
            if ret == QMessageBox.StandardButton.Save:
                self._save_as()
                return not self._text_edit.document().isModified()
            return ret == QMessageBox.StandardButton.Discard

        def _toggle_find_bar(self, show: bool) -> None:
            if show:
                self._find_bar.show()
                self._find_input.setFocus()
                self._find_input.selectAll()
                self._find_bar_visible = True
            else:
                self._find_bar.hide()
                self._find_bar_visible = False
                self._text_edit.setFocus()

        def _close_find_bar(self) -> None:
            if self._find_bar_visible:
                self._toggle_find_bar(False)

        def _find_next(self) -> None:
            text = self._find_input.text()
            if not text:
                return
            from PySide6.QtGui import QTextCursor, QTextDocument

            found = self._text_edit.find(
                text, QTextDocument.FindFlag.FindCaseSensitively
            )
            if not found:
                cursor = self._text_edit.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.Start)
                self._text_edit.setTextCursor(cursor)
                self._text_edit.find(text, QTextDocument.FindFlag.FindCaseSensitively)

        def _toggle_wrap(self) -> None:
            current = self._text_edit.lineWrapMode()
            if current == QTextEdit.NoWrap:
                self._text_edit.setLineWrapMode(QTextEdit.WidgetWidth)
                self.statusBar().showMessage("Word wrap: ON", 2000)
            else:
                self._text_edit.setLineWrapMode(QTextEdit.NoWrap)
                self.statusBar().showMessage("Word wrap: OFF", 2000)

        def closeEvent(self, event):
            if self._text_edit.document().isModified():
                if not self._maybe_save_before_close():
                    event.ignore()
                    return
            _text.on_close(self)
            super().closeEvent(event)

        def changeEvent(self, event):
            from PySide6.QtCore import QEvent

            if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
                _text.on_activate(self)
            super().changeEvent(event)

    win = _TextWindow()
    win._text_edit.setPlainText(content)
    _text.register(win)
    win.show()
    return win


def _open_html(viewer, path: str) -> None:
    import webbrowser

    webbrowser.open(f"file://{path}")


def _open_pdf(viewer, path: str) -> None:
    """Open a PDF file, rendering pages as images in napari."""
    from pathlib import Path

    try:
        from PySide6.QtPdf import QPdfDocument
        from PySide6.QtCore import QSize
        from PySide6.QtGui import QImage
    except ImportError:
        return
    import numpy as np

    doc = QPdfDocument()
    doc.load(path)
    n_pages = doc.pageCount()
    if n_pages == 0:
        return

    dpi = 150
    pages = []
    max_w, max_h = 0, 0
    for i in range(n_pages):
        pt_size = doc.pagePointSize(i)
        w_px = int(pt_size.width() * dpi / 72)
        h_px = int(pt_size.height() * dpi / 72)
        max_w = max(max_w, w_px)
        max_h = max(max_h, h_px)

    for i in range(n_pages):
        pt_size = doc.pagePointSize(i)
        w_px = int(pt_size.width() * dpi / 72)
        h_px = int(pt_size.height() * dpi / 72)
        scale = min(max_w / w_px, max_h / h_px)
        rw = int(w_px * scale)
        rh = int(h_px * scale)
        img = doc.render(i, QSize(rw, rh))
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        ptr = img.bits()
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(
            img.height(), img.width(), 4
        )
        rgb = arr[:, :, 2::-1].astype(np.float32)
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        composite = alpha * rgb + (1.0 - alpha) * 255.0
        canvas = np.full((max_h, max_w, 3), 255.0, dtype=np.float32)
        y0 = (max_h - rh) // 2
        x0 = (max_w - rw) // 2
        canvas[y0 : y0 + rh, x0 : x0 + rw] = composite
        pages.append(canvas)

    data = np.stack(pages) if len(pages) > 1 else pages[0]
    name = Path(path).name
    contrast = _auto_contrast(data)
    # Multi-page PDF is a stack of 2D pages, not a 3D volume: hide the
    # axis selector so it does not appear for the page-frame axis.
    _SliceDirectionWidget.set_stack_mode(len(pages) > 1)
    layer = viewer.add_image(
        data,
        name=name,
        contrast_limits=contrast,
        interpolation2d="linear",
        interpolation3d="linear",
    )
    _enable_continuous_auto_contrast(layer, viewer)
    layer.contrast_limits_range = (float(data.min()), float(data.max()))
    _reset_view(viewer)
    _hide_layer_panels(viewer)
    if len(pages) > 1:
        step = list(viewer.dims.current_step)
        step[0] = 0
        viewer.dims.current_step = step


def _open_eps(viewer, path: str) -> None:
    """Open an EPS (PostScript) file by rasterizing it to a PNG.

    Qt has no PostScript interpreter, so EPS cannot be read by QPdfDocument
    or QImageReader. We rasterize with Ghostscript (``gs``) when available,
    falling back to Quick Look (``qlmanage``) on macOS, then display the
    image reusing the same white-background compositing and contrast logic
    as the PDF viewer.
    """
    import numpy as np
    from PySide6.QtGui import QImage

    out_path = _rasterize_eps(path)
    if out_path is None:
        print(
            "[helicon] Ghostscript (gs) is required to display EPS files but "
            "was not found on your PATH. Install it with "
            "'conda install -c conda-forge ghostscript' or "
            "'brew install ghostscript'."
        )
        return

    try:
        img = QImage(out_path)
        if img.isNull():
            print(f"[helicon] failed to render EPS: {path}")
            return
        img = img.convertToFormat(QImage.Format.Format_ARGB32)
        ptr = img.bits()
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape(
            img.height(), img.width(), 4
        )
        rgb = arr[:, :, 2::-1].astype(np.float32)  # BGRA -> RGB float
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        # Composite onto white background
        composite = alpha * rgb + (1.0 - alpha) * 255.0

        name = Path(path).name
        contrast = _auto_contrast(composite)
        layer = viewer.add_image(
            composite,
            name=name,
            contrast_limits=contrast,
            interpolation2d="linear",
            interpolation3d="linear",
        )
        _enable_continuous_auto_contrast(layer, viewer)
        layer.contrast_limits_range = (
            float(composite.min()),
            float(composite.max()),
        )
        _reset_view(viewer)
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass
        # qlmanage writes into its own temp dir; drop the empty dir as well.
        try:
            os.rmdir(Path(out_path).parent)
        except OSError:
            pass


def _rasterize_eps(path: str) -> str | None:
    """Rasterize an EPS file to a PNG and return its path, or None on failure.

    Tries Ghostscript first (all platforms), then Quick Look (``qlmanage``)
    on macOS. The caller owns the returned temporary file and is expected
    to remove it.
    """
    import shutil
    import subprocess
    import sys
    import tempfile

    gs = shutil.which("gs") or shutil.which("ghostscript")
    if gs is not None:
        tmp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_png.close()
        try:
            subprocess.run(
                [
                    gs,
                    "-dQUIET",
                    "-dNOPAUSE",
                    "-dBATCH",
                    "-dSAFER",
                    "-dEPSCrop",
                    "-sDEVICE=png16m",
                    "-r150",
                    f"-sOutputFile={tmp_png.name}",
                    path,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            return tmp_png.name
        except (subprocess.CalledProcessError, OSError):
            try:
                os.remove(tmp_png.name)
            except OSError:
                pass

    if sys.platform == "darwin":
        qlmanage = shutil.which("qlmanage")
        if qlmanage is not None:
            out_dir = tempfile.mkdtemp(prefix="helicon_eps_")
            try:
                result = subprocess.run(
                    [qlmanage, "-t", "-s", "2048", "-o", out_dir, path],
                    capture_output=True,
                    timeout=60,
                )
            except (subprocess.SubprocessError, OSError):
                result = None
            candidate = Path(out_dir) / f"{Path(path).name}.png"
            if result is not None and result.returncode == 0 and candidate.is_file():
                return str(candidate)
            try:
                os.rmdir(out_dir)
            except OSError:
                pass

    return None


def _capped_cylinder_mesh(p1, p2, radius, segments=24):
    """Vertices + triangle indices for a capped cylinder between ``p1`` and
    ``p2`` (ChimeraX BILD semantics: capped by default).

    Ported from ChimeraX ``shape.cylinder_geometry``: the side is a tube and
    each flat end is closed with a triangle fan so the rod looks solid.
    """
    import numpy as np

    p1 = np.asarray(p1, float)
    p2 = np.asarray(p2, float)
    axis = p2 - p1
    length = float(np.linalg.norm(axis))
    if length == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    a = axis / length
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(a, helper)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)

    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = radius * (np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v)
    bot = p1 + ring
    top = p2 + ring
    verts = np.vstack([bot, top, p1, p2])
    n_bot, n_top = 2 * segments, segments
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, n_top + j])
        faces.append([i, n_top + j, n_top + i])
        faces.append([n_bot, j, i])
        faces.append([n_bot + 1, n_top + i, n_top + j])
    return verts, np.array(faces, int)


def _sphere_mesh(center, radius, rings=16, segments=24):
    """Vertices + triangle indices for a UV sphere (always closed)."""
    import numpy as np

    center = np.asarray(center, float)
    verts = [center]
    for i in range(1, rings):
        phi = np.pi * i / rings
        r = radius * np.sin(phi)
        y = radius * np.cos(phi)
        theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
        for t in theta:
            verts.append(center + np.array([r * np.cos(t), y, r * np.sin(t)]))
    verts.append(center + np.array([0.0, radius, 0.0]))
    verts = np.array(verts)
    n = len(verts)
    pole_s, pole_n = 0, n - 1
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([pole_s, j + 1, i + 1])
    for k in range(rings - 2):
        base = 1 + k * segments
        nxt = base + segments
        for i in range(segments):
            j = (i + 1) % segments
            faces.append([base + i, base + j, nxt + j])
            faces.append([base + i, nxt + j, nxt + i])
    last_base = 1 + (rings - 2) * segments
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([pole_n, last_base + j, last_base + i])
    return verts, np.array(faces, int)


def _open_bild(viewer, path: str) -> None:
    from pathlib import Path
    import numpy as np

    # Each primitive becomes a capped 3D surface (ChimeraX semantics: cylinders
    # are solid/capped unless the ``open`` keyword is given). We render every
    # primitive as its own surface layer so per-object colors are preserved.
    meshes = []  # list of (vertices, faces, color)
    current_color = [1.0, 1.0, 1.0]

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(".color"):
                parts = line.split()
                current_color = [float(parts[1]), float(parts[2]), float(parts[3])]
            elif line.startswith(".cylinder"):
                parts = line.split()
                x1, y1, z1 = float(parts[1]), float(parts[2]), float(parts[3])
                x2, y2, z2 = float(parts[4]), float(parts[5]), float(parts[6])
                r = float(parts[7])
                # ``open`` keyword (8th field) leaves the cylinder uncapped.
                capped = not (len(parts) > 8 and parts[8] == "open")
                verts, faces = _capped_cylinder_mesh((x1, y1, z1), (x2, y2, z2), r)
                if not capped:
                    # Drop the two end-cap fans (last 2 * segments triangles);
                    # approximate by rebuilding an uncapped tube.
                    n = len(verts)
                    segs = (n - 4) // 2
                    side_faces = faces[: 2 * segs]
                    verts = verts[: 2 * segs]
                    faces = side_faces
                meshes.append((verts, faces, current_color))
            elif line.startswith(".sphere"):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r = float(parts[4])
                verts, faces = _sphere_mesh((x, y, z), r)
                meshes.append((verts, faces, current_color))

    if not meshes:
        return

    # Merge every primitive into a single surface layer. A BILD file can hold
    # thousands of objects (e.g. an angular-distribution plot); one layer per
    # object would create thousands of layers and stall napari's 3D view.
    all_verts = []
    all_faces = []
    all_colors = []
    offset = 0
    for verts, faces, color in meshes:
        if len(verts) == 0 or len(faces) == 0:
            continue
        all_verts.append(verts)
        all_faces.append(faces + offset)
        all_colors.append(np.tile(np.asarray(color, float), (len(verts), 1)))
        offset += len(verts)

    if not all_verts:
        return

    vertices = np.vstack(all_verts)
    faces = np.vstack(all_faces)
    vertex_colors = np.vstack(all_colors)
    name = Path(path).name
    viewer.add_surface(
        (vertices, faces),
        name=name,
        vertex_colors=vertex_colors,
        shading="smooth",
    )
    viewer.dims.ndisplay = 3
    _reset_view(viewer)
