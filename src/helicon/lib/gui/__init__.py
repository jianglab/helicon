"""Qt GUI sub-package of ``helicon.lib``.

Houses all Qt widgets and display-command implementation. The
``helicon.commands.display`` command is a thin coordinator: the actual
implementation lives in these modules, which the coordinator re-exports so
that ``from helicon.commands.display import X`` keeps working for every
existing caller (file browser, gallery backends/widgets, proc3d/images2star
tools, and the tests).
"""

from . import (  # noqa: F401
    file_browser,
    file_openers,
    fsc,
    gallery_backends,
    gallery_widget,
    images2star_widget,
    macos,
    proc3d_widget,
    star_parsers,
    theme,
    trackers,
    viewer,
    webapps,
)
