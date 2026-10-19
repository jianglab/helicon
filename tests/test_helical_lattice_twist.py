"""HelicalLattice: a zero twist is refused, not divided by."""

import pytest

from helicon.webApps.tabs.helical_lattice_tab import (
    convert_helical_lattice_to_2d_lattice,
)


class TestZeroTwist:
    def test_the_conversion_raises_value_error(self):
        with pytest.raises(ValueError):
            convert_helical_lattice_to_2d_lattice(
                twist=0, rise=19.4, csym=1, diameter=290.0
            )

    def test_a_normal_twist_still_converts(self):
        # primitive_unitcell=True: the orthogonal-cell search below reaches a
        # ``np.cross`` of 2-D vectors, which this NumPy version refuses.
        a, b, endpoint = convert_helical_lattice_to_2d_lattice(
            twist=-81.1, rise=19.4, csym=1, diameter=290.0, primitive_unitcell=True
        )
        assert len(a) == 2 and len(b) == 2 and len(endpoint) == 2
