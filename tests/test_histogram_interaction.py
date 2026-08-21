import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from helicon.lib.gui.gallery_widget import _HistogramWidget


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


@pytest.fixture
def histogram(qapp):
    widget = _HistogramWidget()
    widget.resize(180, widget.HIST_HEIGHT)
    widget._bins = np.ones(256, dtype=np.float64)
    widget.show()
    qapp.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()


def test_black_point_drag_updates_brightness_and_contrast(histogram, qapp):
    changes = []
    histogram.bcg_changed.connect(lambda *values: changes.append(values))
    black = histogram._endpoint_screen_position("black")
    assert black is not None

    QTest.mousePress(
        histogram,
        Qt.LeftButton,
        Qt.NoModifier,
        QPoint(round(black.x()), round(black.y())),
    )
    QTest.mouseMove(histogram, QPoint(round(histogram.width() * 0.2), round(black.y())))
    QTest.mouseRelease(
        histogram,
        Qt.LeftButton,
        Qt.NoModifier,
        QPoint(round(histogram.width() * 0.2), round(black.y())),
    )
    qapp.processEvents()

    assert changes
    brightness, contrast, gamma = changes[-1]
    assert brightness < 0.0
    assert contrast > 1.0
    assert gamma == pytest.approx(1.0)


def test_curve_vertical_drag_updates_only_gamma(histogram, qapp):
    changes = []
    histogram.bcg_changed.connect(lambda *values: changes.append(values))
    rect = histogram._plot_rect()
    center = QPoint(round(rect.center().x()), round(rect.center().y()))

    QTest.mousePress(histogram, Qt.LeftButton, Qt.NoModifier, center)
    dragged = QPoint(center.x(), round(rect.top() + rect.height() * 0.3))
    QTest.mouseMove(histogram, dragged)
    QTest.mouseRelease(
        histogram,
        Qt.LeftButton,
        Qt.NoModifier,
        dragged,
    )
    qapp.processEvents()

    assert changes
    brightness, contrast, gamma = changes[-1]
    assert brightness == pytest.approx(0.0)
    assert contrast == pytest.approx(1.0)
    assert gamma > 1.0


def test_curve_drag_values_drive_orthogonal_viewer_panel(qapp):
    from helicon.commands.display import _wrap_gallery_with_panel
    from helicon.lib.gui.gallery_widget import OrthogonalViewerWidget, _ControlPanel

    viewer = OrthogonalViewerWidget(np.arange(64, dtype=np.float32).reshape(4, 4, 4))
    container = _wrap_gallery_with_panel(viewer)
    panel = container.findChild(_ControlPanel)
    assert panel is not None

    panel._histogram_widget.bcg_changed.emit(-0.2, 1.5, 2.0)
    qapp.processEvents()

    assert viewer._brightness == pytest.approx(-0.2)
    assert viewer._contrast == pytest.approx(1.5)
    assert viewer._gamma == pytest.approx(2.0)
    assert panel._brightness_slider.value() == -20
    assert panel._contrast_slider.value() == 150
    assert panel._gamma_slider.value() == 200
    assert panel._brightness_val.text() == "-0.20"
    assert panel._contrast_val.text() == "1.50"
    assert panel._gamma_val.text() == "2.00"
    for view in (viewer._xy_view, viewer._xz_view, viewer._yz_view):
        assert view._brightness == pytest.approx(-0.2)
        assert view._contrast == pytest.approx(1.5)
        assert view._gamma == pytest.approx(2.0)

    container.close()
    container.deleteLater()
