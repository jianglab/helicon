"""Gallery classes for displaying image stacks with type-specific controls.

Each gallery type handles its own data parsing, read function, labels, and
panel UI configuration. The base class manages window creation, reuse, and
the common brightness/contrast/gamma panel.

Gallery Types
-------------
- ``StackGallery``: Plain .mrcs stacks, data.star files, or generic read_fn.
- ``Class3dGallery``: Class3D optimiser/model.star → XYZ slices + abundance.
- ``Refine3dGallery``: Refine3D optimiser/model.star → XYZ slices.
- ``Class2dGallery``: Class2D optimiser/model.star → class averages + sort.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


class BaseGallery:
    """Base class for all gallery types.

    Manages window creation, reuse, and common panel infrastructure.
    Subclasses must implement ``_parse()``, ``_read_image()``, and
    ``_setup_panel()``.

    Parameters
    ----------
    star_path : str
        Path to the star file (or mrcs for StackGallery).
    """

    def __init__(self, star_path: str) -> None:
        self.star_path = star_path
        self._parsed = False
        self._n = 0
        self._img_w = 0
        self._img_h = 0
        self._apix = 1.0
        self._labels: list[str] = []
        self._read_fn: Callable[[int], np.ndarray] | None = None

    def open(self, reuse_window: Any | None = None) -> Any | None:
        """Open the gallery window, optionally reusing an existing one.

        Parameters
        ----------
        reuse_window : QMainWindow, optional
            Window to reuse instead of creating a new one.

        Returns
        -------
        QMainWindow or None
            The gallery window, or None if parsing failed.
        """
        from helicon.commands.display import (
            _on_gallery_closing,
            _galleries,
            _active_gallery,
        )

        if not self._parsed:
            self._parse()
        if self._n == 0:
            return None

        from PySide6.QtWidgets import QMainWindow
        from helicon.lib.gallery_widget import ImageGalleryWidget

        read_fn = self._read_fn if self._read_fn is not None else self._read_image
        widget = ImageGalleryWidget()
        widget.set_data(
            read_fn,
            self._n,
            self._img_w,
            self._img_h,
            None,
            labels=self._labels,
            source_name=Path(self.star_path).name,
        )

        from helicon.commands.display import _wrap_gallery_with_panel

        container = _wrap_gallery_with_panel(widget)
        self._setup_panel(container, widget)

        if reuse_window is not None:
            reuse_window.setWindowTitle(f"helicon - {Path(self.star_path).name}")
            reuse_window.setCentralWidget(container)
            reuse_window.show()
            reuse_window.raise_()
            return reuse_window

        class _GalleryWindow(QMainWindow):
            def closeEvent(self, event):
                _on_gallery_closing(self)
                super().closeEvent(event)

        window = _GalleryWindow()
        window.setWindowTitle(f"helicon - {Path(self.star_path).name}")
        window.setCentralWidget(container)
        tile = 128 + widget._panel.min_sep
        window.resize(5 * tile + widget._sb_width, 5 * tile)
        _galleries.append(window)
        _active_gallery[0] = window
        window.show()
        return window

    def _parse(self) -> None:
        """Parse the input file and populate _n, _img_w, _img_h, _apix, _labels, _read_fn."""
        raise NotImplementedError

    def _setup_panel(self, container: Any, widget: Any) -> None:
        """Configure type-specific panel controls. Override in subclasses."""
        pass


class StackGallery(BaseGallery):
    """Gallery for plain .mrcs stacks, data.star files, or generic read_fn.

    Parameters
    ----------
    star_path : str
        Path to .mrcs file, .star file, or any image source.
    read_fn : callable, optional
        Custom ``read_fn(i) -> ndarray``. If None, auto-detect from file extension.
    n : int, optional
        Number of images. Required if read_fn is provided.
    img_w : int, optional
        Image width. Required if read_fn is provided.
    img_h : int, optional
        Image height. Required if read_fn is provided.
    apix : float, optional
        Pixel size in Angstroms. Defaults to 1.0.
    labels : list of str, optional
        Custom labels for each image.
    """

    def __init__(
        self,
        star_path: str,
        read_fn: Callable[[int], np.ndarray] | None = None,
        n: int = 0,
        img_w: int = 0,
        img_h: int = 0,
        apix: float = 1.0,
        labels: list[str] | None = None,
    ) -> None:
        super().__init__(star_path)
        self._external_read_fn = read_fn
        self._external_n = n
        self._external_img_w = img_w
        self._external_img_h = img_h
        self._external_apix = apix
        self._external_labels = labels

    def _parse(self) -> None:
        if self._external_read_fn is not None:
            self._read_fn = self._external_read_fn
            self._n = self._external_n
            self._img_w = self._external_img_w
            self._img_h = self._external_img_h
            self._apix = self._external_apix
            self._labels = self._external_labels or []
            self._parsed = True
            return

        ext = Path(self.star_path).suffix.lower()
        if ext == ".mrcs":
            self._parse_mrcs()
        elif ext == ".star":
            self._parse_star()
        self._parsed = True

    def _parse_mrcs(self) -> None:
        import mrcfile

        with mrcfile.open(self.star_path, permissive=True) as mrc:
            data = mrc.data
            self._apix = float(mrc.voxel_size.x) if mrc.voxel_size.x > 0 else 1.0
            if data.ndim == 2:
                self._n = 1
                self._img_h, self._img_w = data.shape
            else:
                self._n = data.shape[0]
                self._img_h, self._img_w = data.shape[1], data.shape[2]

        def _read(i: int) -> np.ndarray:
            with mrcfile.open(self.star_path, permissive=True) as mrc:
                d = np.asarray(mrc.data)
                if d.ndim == 2:
                    return d.astype(np.float32)
                return d[i].astype(np.float32)

        self._read_fn = _read
        self._labels = [str(i + 1) for i in range(self._n)]

    def _parse_star(self) -> None:
        from helicon.commands.display import (
            _parse_star_image_refs,
            _LazyStarStack,
            _auto_contrast,
        )

        result = _parse_star_image_refs(self.star_path)
        if result is None:
            return
        entries, first_shape, first_apix = result
        self._n = len(entries)
        self._img_w, self._img_h = first_shape[0], first_shape[1]
        self._apix = first_apix

        stack_shape = (self._n,) + first_shape
        lazy = _LazyStarStack(entries, stack_shape, None)
        self._read_fn = lazy.__getitem__
        self._labels = [str(i + 1) for i in range(self._n)]


class Class3dGallery(BaseGallery):
    """Gallery for Class3D optimiser/model.star files.

    Displays Z/Y/X center slices of referenced 3D MRC maps with
    abundance labels (_rlnClassDistribution). Z Thickness control
    allows averaging over a specified thickness in Angstroms.
    """

    def __init__(self, star_path: str) -> None:
        super().__init__(star_path)
        self._mrc_paths: list[str] = []
        self._dists: list[float] | None = None
        self._nz_first = 0
        self._z_thickness_a = [0.0]

    def _parse(self) -> None:
        from helicon.commands.display import (
            _parse_optimiser_star,
            _parse_model_star,
            _parse_class2d_model_star,
        )
        import mrcfile

        name = Path(self.star_path).name
        if name.endswith("model.star"):
            self._mrc_paths = _parse_model_star(self.star_path) or []
        else:
            self._mrc_paths = _parse_optimiser_star(self.star_path) or []

        if not self._mrc_paths:
            return

        # Distributions live in the model.star, not the optimiser.star.
        dist_source = self.star_path
        if not Path(self.star_path).name.endswith("model.star"):
            from helicon.commands.display import _find_model_star_from_optimiser

            resolved = _find_model_star_from_optimiser(self.star_path)
            if resolved is not None:
                dist_source = resolved
        parsed = _parse_class2d_model_star(dist_source)
        if parsed is not None:
            _, self._dists = parsed

        with mrcfile.open(self._mrc_paths[0], permissive=True) as mrc:
            self._apix = float(mrc.voxel_size.x) if mrc.voxel_size.x > 0 else 1.0
            self._nz_first = mrc.data.shape[0]

        slices_per_mrc = 3
        self._n = len(self._mrc_paths) * slices_per_mrc

        sample = self._read_image(0)
        self._img_h, self._img_w = sample.shape

        self._labels = self._make_labels(0.0)
        self._parsed = True

    def _read_image(self, i: int) -> np.ndarray:
        import mrcfile

        mrc_idx = i // 3
        slice_idx = i % 3

        with mrcfile.open(self._mrc_paths[mrc_idx], permissive=True) as mrc:
            data = mrc.data
            nz, ny, nx = data.shape
            z_center = nz // 2
            y_center = ny // 2
            x_center = nx // 2

            if slice_idx == 0:
                t_a = self._z_thickness_a[0]
                if t_a > 0.0 and self._apix > 0.0:
                    n_sl = max(1, round(t_a / self._apix))
                    lo = max(0, z_center - n_sl // 2)
                    hi = min(nz, lo + n_sl)
                    sl = np.mean(data[lo:hi, :, :], axis=0).astype(np.float32)
                else:
                    sl = data[z_center, :, :].astype(np.float32)
            elif slice_idx == 1:
                sl = data[:, y_center, :].astype(np.float32)
            else:
                sl = data[:, :, x_center].astype(np.float32)

            return sl

    def _make_labels(self, thickness_a: float) -> list[str]:
        suffix = f" ({thickness_a:.1f} Å)" if thickness_a > 0 else ""
        axis_labels = ["Z", "Y", "X"]
        labels = []
        for mrc_idx in range(len(self._mrc_paths)):
            dist_str = ""
            if self._dists is not None and mrc_idx < len(self._dists):
                dist_str = f" {self._dists[mrc_idx]:.1%}"
            for si in range(3):
                labels.append(
                    f"{mrc_idx + 1}{dist_str}-{axis_labels[si]}"
                    f"{suffix if si == 0 else ''}"
                )
        return labels

    def _setup_panel(self, container: Any, widget: Any) -> None:
        from helicon.lib.gallery_widget import _ControlPanel

        panel = container.findChild(_ControlPanel)
        if panel is not None and self._nz_first > 1:
            max_thickness = self._nz_first * self._apix
            panel.show_z_thickness(True, max_thickness)

            def _on_z_thickness(val):
                self._z_thickness_a[0] = val
                widget._labels = self._make_labels(val)
                widget._thumb_cache.clear()
                widget.update()

            panel.z_thickness_changed.connect(_on_z_thickness)


class Refine3dGallery(BaseGallery):
    """Gallery for Refine3D optimiser/model.star files.

    Displays Z/Y/X center slices of referenced 3D MRC maps.
    Z Thickness control allows averaging over a specified thickness.
    Does NOT show abundance labels.
    """

    def __init__(self, star_path: str) -> None:
        super().__init__(star_path)
        self._mrc_paths: list[str] = []
        self._nz_first = 0
        self._z_thickness_a = [0.0]

    def _parse(self) -> None:
        from helicon.commands.display import _parse_optimiser_star, _parse_model_star
        import mrcfile

        name = Path(self.star_path).name
        if name.endswith("model.star"):
            self._mrc_paths = _parse_model_star(self.star_path) or []
        else:
            self._mrc_paths = _parse_optimiser_star(self.star_path) or []

        if not self._mrc_paths:
            return

        with mrcfile.open(self._mrc_paths[0], permissive=True) as mrc:
            self._apix = float(mrc.voxel_size.x) if mrc.voxel_size.x > 0 else 1.0
            self._nz_first = mrc.data.shape[0]

        slices_per_mrc = 3
        self._n = len(self._mrc_paths) * slices_per_mrc

        sample = self._read_image(0)
        self._img_h, self._img_w = sample.shape

        self._labels = self._make_labels(0.0)
        self._parsed = True

    def _read_image(self, i: int) -> np.ndarray:
        import mrcfile

        mrc_idx = i // 3
        slice_idx = i % 3

        with mrcfile.open(self._mrc_paths[mrc_idx], permissive=True) as mrc:
            data = mrc.data
            nz, ny, nx = data.shape
            z_center = nz // 2
            y_center = ny // 2
            x_center = nx // 2

            if slice_idx == 0:
                t_a = self._z_thickness_a[0]
                if t_a > 0.0 and self._apix > 0.0:
                    n_sl = max(1, round(t_a / self._apix))
                    lo = max(0, z_center - n_sl // 2)
                    hi = min(nz, lo + n_sl)
                    sl = np.mean(data[lo:hi, :, :], axis=0).astype(np.float32)
                else:
                    sl = data[z_center, :, :].astype(np.float32)
            elif slice_idx == 1:
                sl = data[:, y_center, :].astype(np.float32)
            else:
                sl = data[:, :, x_center].astype(np.float32)

            return sl

    def _make_labels(self, thickness_a: float) -> list[str]:
        suffix = f" ({thickness_a:.1f} Å)" if thickness_a > 0 else ""
        axis_labels = ["Z", "Y", "X"]
        labels = []
        for mrc_idx in range(len(self._mrc_paths)):
            for si in range(3):
                labels.append(
                    f"{mrc_idx + 1}-{axis_labels[si]}" f"{suffix if si == 0 else ''}"
                )
        return labels

    def _setup_panel(self, container: Any, widget: Any) -> None:
        from helicon.lib.gallery_widget import _ControlPanel

        panel = container.findChild(_ControlPanel)
        if panel is not None and self._nz_first > 1:
            max_thickness = self._nz_first * self._apix
            panel.show_z_thickness(True, max_thickness)

            def _on_z_thickness(val):
                self._z_thickness_a[0] = val
                widget._labels = self._make_labels(val)
                widget._thumb_cache.clear()
                widget.update()

            panel.z_thickness_changed.connect(_on_z_thickness)


class Class2dGallery(BaseGallery):
    """Gallery for Class2D optimiser/model.star files.

    Displays 2D class averages sorted by abundance with sort controls.
    """

    def __init__(self, star_path: str) -> None:
        super().__init__(star_path)
        self._entries: list[tuple[str, int]] = []
        self._dists: list[float] = []
        self._order: list[int] = []
        self._sort_column = "Abundance"
        self._sort_reverse = True
        self._sort_columns = ["Abundance", "Name"]

    def _parse(self) -> None:
        from helicon.commands.display import (
            _parse_class2d_model_star,
            _find_model_star_from_optimiser,
        )
        import mrcfile

        settings = self._get_settings()
        self._sort_column = settings.value("sort_column", "Abundance", type=str)
        self._sort_reverse = settings.value("sort_reverse", True, type=bool)

        name = Path(self.star_path).name
        if name.endswith("model.star"):
            result = _parse_class2d_model_star(self.star_path)
        else:
            model_path = _find_model_star_from_optimiser(self.star_path)
            if model_path is None:
                return
            result = _parse_class2d_model_star(model_path)

        if result is None:
            return
        self._entries, self._dists = result
        self._n = len(self._entries)
        if self._n == 0:
            return

        first_mrc, first_frame = self._entries[0]
        with mrcfile.open(first_mrc, permissive=True) as mrc:
            self._apix = float(mrc.voxel_size.x) if mrc.voxel_size.x > 0 else 1.0
            sample = np.asarray(mrc.data)
            if sample.ndim >= 3:
                sample = sample[first_frame]
        self._img_h, self._img_w = sample.shape

        self._apply_sort()
        self._labels = self._make_labels()
        self._parsed = True

    @staticmethod
    def _get_settings():
        from PySide6.QtCore import QSettings

        return QSettings("helicon", "display")

    def _apply_sort(self) -> None:
        if self._sort_column == "Abundance":
            indexed = list(enumerate(self._dists))
            indexed.sort(key=lambda x: x[1], reverse=self._sort_reverse)
            self._order = [i for i, _ in indexed]
        else:
            indexed = list(enumerate(self._entries))
            indexed.sort(key=lambda x: x[1], reverse=self._sort_reverse)
            self._order = [i for i, _ in indexed]

    def _read_image(self, i: int) -> np.ndarray:
        import mrcfile

        orig_idx = self._order[i]
        mrc_path, frame_idx = self._entries[orig_idx]
        with mrcfile.open(mrc_path, permissive=True) as mrc:
            d = np.asarray(mrc.data)
            if d.ndim >= 3:
                d = d[frame_idx]
            return d.astype(np.float32)

    def _make_labels(self) -> list[str]:
        labels = []
        for pos in range(self._n):
            orig_idx = self._order[pos]
            dist = self._dists[orig_idx]
            labels.append(f"Class {orig_idx + 1} ({dist:.1%})")
        return labels

    def _setup_panel(self, container: Any, widget: Any) -> None:
        from helicon.lib.gallery_widget import _ControlPanel

        panel = container.findChild(_ControlPanel)
        if panel is not None:
            panel.show_sort_ui(True, columns=self._sort_columns)
            idx = panel._sort_column_combo.findText(self._sort_column)
            if idx >= 0:
                panel._sort_column_combo.setCurrentIndex(idx)
            panel._sort_reverse_chk.setChecked(self._sort_reverse)

            def _on_sort_changed():
                self._sort_column = panel._sort_column_combo.currentText()
                self._sort_reverse = panel._sort_reverse_chk.isChecked()
                settings = self._get_settings()
                settings.setValue("sort_column", self._sort_column)
                settings.setValue("sort_reverse", self._sort_reverse)
                self._apply_sort()
                widget._labels = self._make_labels()
                widget._thumb_cache.clear()
                widget.update()

            panel.sort_column_changed.connect(lambda _: _on_sort_changed())
            panel.sort_reverse_changed.connect(lambda _: _on_sort_changed())


def gallery_for_star(star_path: str) -> BaseGallery | None:
    """Create the appropriate gallery class for a star file.

    Parameters
    ----------
    star_path : str
        Path to the star file.

    Returns
    -------
    BaseGallery or None
        The appropriate gallery instance, or None if the file type is not recognized.
    """
    from pathlib import Path

    name = Path(star_path).name
    parts = Path(star_path).parts

    is_class2d = any(p.startswith("Class2D") for p in parts)
    is_class3d = any(p.startswith("Class3D") for p in parts)
    is_refine3d = any(p.startswith("Refine3D") for p in parts)

    if is_class2d and (name.endswith("optimiser.star") or name.endswith("model.star")):
        return Class2dGallery(star_path)
    elif is_class3d and (
        name.endswith("optimiser.star") or name.endswith("model.star")
    ):
        return Class3dGallery(star_path)
    elif is_refine3d and (
        name.endswith("optimiser.star") or name.endswith("model.star")
    ):
        return Refine3dGallery(star_path)
    elif name.endswith("optimiser.star") or name.endswith("model.star"):
        return Class3dGallery(star_path)
    else:
        return None
