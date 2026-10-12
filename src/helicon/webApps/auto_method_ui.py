"""The wording of the control that chooses how a filament is straightened and centred.

HILL, DeNovo3D and HelicalProjection offer the same choice in their Parameters
tab, each declaring the select box itself (so that its bookmark entry can be
checked against it) with the label, choices and explanation kept here.
"""

from .lib import helix_transform

LABEL = "Auto-transform method:"
CHOICES = helix_transform.METHODS
DEFAULT = helix_transform.DEFAULT_METHOD
HELP = (
    "How the angle and the centre of the filament are worked out. "
    "Threshold and moments: the image is thresholded at a fifth of its maximum "
    "and the filament's angle and centre are those of what is left; it needs a "
    "bright filament on a darker background. "
    "Projection sharpness and phases across the meridian: the angle is the one "
    "that makes the image summed along the filament sharpest, and the centre is "
    "where the two sides of the layer lines agree in phase; it takes no "
    "threshold, does not care whether the filament is bright or dark, and "
    "holds up better in noisy images."
)
