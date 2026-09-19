"""Joint search: several class averages scored against one map together.

The score of a map is the mean over the selected images, and the picture that
comes back is all of them placed into the same container -- the map's own side
projection -- so it can be seen at a glance whether they sit where the
structure says they should.
"""

import numpy as np
import pytest

import helicon
from helicon.webApps.lib import helical_projection_compute as compute


def _helical_map(nz=64, ny=48, nx=48, apix=2.0, radius=9.0, sigma=3.5):
    """A filament with a real twist, so the projection has structure to match."""
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx].astype(np.float32)
    z = (z - nz // 2) * apix
    y = (y - ny // 2) * apix
    x = (x - nx // 2) * apix
    vol = np.zeros((nz, ny, nx), dtype=np.float32)
    rise, twist = 6.0, 25.0
    for k in range(-nz, nz):
        cz = k * rise
        if abs(cz) > nz * apix / 2 + 3 * sigma:
            continue
        ang = np.deg2rad(twist * k)
        vol += np.exp(
            -(
                (x - radius * np.cos(ang)) ** 2
                + (y - radius * np.sin(ang)) ** 2
                + (z - cz) ** 2
            )
            / (2 * sigma**2)
        )
    return vol, apix, twist, rise


@pytest.fixture(scope="module")
def setup():
    vol, apix, twist, rise = _helical_map()
    map_info = compute.MapInfo(
        data=vol, label="synthetic", apix=apix, twist=twist, rise=rise, csym=1
    )
    sym = helicon.apply_helical_symmetry(
        data=vol,
        apix=apix,
        twist_degree=twist,
        rise_angstrom=rise,
        csym=1,
        fraction=5 * rise / (vol.shape[0] * apix),
        new_size=(128, 48, 48),
        new_apix=apix,
        cpu=1,
    )
    proj = sym.sum(axis=2).T.astype(np.float32)
    rng = np.random.default_rng(0)
    w = 32
    queries = []
    for centre in (40, 70, 96):
        crop = np.ascontiguousarray(proj[:, centre - w // 2 : centre + w // 2])
        queries.append(crop + rng.normal(0, 0.3 * crop.std(), crop.shape).astype("f4"))
    return map_info, queries, apix


def _run(map_info, query, label, apix):
    _, result = compute.symmetrize_project_align_one_map(
        map_info, query, label, apix, True, 1.2, False, 0.0, 0.1
    )
    return result


class TestJointSearch:
    def test_the_score_is_the_mean_over_the_selected_images(self, setup):
        map_info, queries, apix = setup
        singles = [_run(map_info, q, f"q{i}", apix) for i, q in enumerate(queries)]
        joint = _run(map_info, queries, [f"q{i}" for i in range(len(queries))], apix)
        assert joint is not None and all(s is not None for s in singles)
        assert joint[4] == pytest.approx(np.mean([s[4] for s in singles]), rel=1e-6)

    def test_every_image_is_placed_in_the_one_container(self, setup):
        map_info, queries, apix = setup
        singles = [_run(map_info, q, f"q{i}", apix) for i, q in enumerate(queries)]
        joint = _run(map_info, queries, [f"q{i}" for i in range(len(queries))], apix)
        composite, projection = joint[5], joint[7]
        assert composite.shape == projection.shape
        # combining by maximum used to erase a placement wherever one image was
        # negative and another contributed the zero of its own padding
        for s in singles:
            covered = (composite != 0) | (s[5] == 0)
            assert covered.all()

    def test_the_label_says_how_many_were_searched(self, setup):
        map_info, queries, apix = setup
        joint = _run(map_info, queries, [f"q{i}" for i in range(len(queries))], apix)
        assert joint[6] == "%d images" % len(queries)

    def test_one_image_behaves_exactly_as_before(self, setup):
        map_info, queries, apix = setup
        bare = _run(map_info, queries[0], "q0", apix)
        listed = _run(map_info, [queries[0]], ["q0"], apix)
        assert bare[4] == listed[4]
        assert bare[6] == listed[6] == "q0"
        assert np.array_equal(bare[5], listed[5])
