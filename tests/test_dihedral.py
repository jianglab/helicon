"""A helix with a two-fold axis perpendicular to its own (dihedral, Dn) in the helical methods.

Such a filament has no polarity: the view of a segment turned by 180 degrees in plane is the plain
view at the mirrored azimuth. The tests measure what that does to the placement of a view along a
projection of the map (Denovo3D) and to the repeat read from the classes of the segments
(HelicalPitch, AbInitio3D), on maps and tables made with a known two-fold.
"""

import numpy as np
import pytest

from helical_sim import DihedralHelix, Helix, Screw21Helix, simulate
from helicon.webApps.lib import denovo3d_align as A
from helicon.webApps.lib import denovo3d_joint as J
from helicon.webApps.lib import helical_pitch_map as maps
from helicon.webApps.lib import helical_pitch_phase as ph
from test_denovo3D_align import APIX, CSYM, RISE, TWIST, _helix_volume


def _dyad(volume):
    """The map rotated by 180 degrees about the axis that is perpendicular to the helical axis
    (index 0) and passes through the box centre."""
    flipped = volume[::-1, ::-1, :]
    return np.roll(np.roll(flipped, 1, axis=0), 1, axis=1)


def _model(volume):
    length = A.model_length_pixel(TWIST, RISE, APIX, CSYM, 128)
    return A.long_side_projection(volume, APIX, TWIST, RISE, CSYM, APIX, 40, length)


def _place(model, dx, turned):
    m = model.shape[1]
    lo = int(round(m // 2 + dx - 64))
    window = model[:, lo : lo + 128]
    image = window[::-1, ::-1] if turned else window
    return A.align_to_model(image, model, TWIST, RISE, APIX)


@pytest.fixture(scope="module")
def plain():
    return _model(_helix_volume())


@pytest.fixture(scope="module")
def dihedral():
    volume = _helix_volume()
    both = volume + _dyad(volume)
    assert np.abs(_dyad(both) - both).max() == 0  # it has the two-fold
    return _model(both)


class TestPlacementAlongTheModel:
    @pytest.mark.parametrize("dx", [-100.0, 37.0, 120.0])
    def test_a_plain_helix_turned_by_180_is_reversed_and_stays_where_it_is(
        self, plain, dx
    ):
        res = _place(plain, dx, turned=True)
        assert res["reversed"]
        assert res["dx"] == pytest.approx(dx, abs=3.0)

    @pytest.mark.parametrize("dx", [-100.0, 37.0, 120.0])
    def test_with_a_two_fold_a_turned_view_is_an_upright_view_at_the_mirrored_place(
        self, dihedral, dx
    ):
        res = _place(dihedral, dx, turned=True)
        assert not res["reversed"]  # the polarity is in the placement, not in psi
        assert res["dx"] == pytest.approx(-dx, abs=3.0)
        assert res["corr"] > 0.95

    def test_a_two_fold_makes_the_rival_placement_stronger(self, plain, dihedral):
        """The second hump of the correlation, which the module judges a placement by, is the
        mirrored placement of the same view: it is much closer to the first one."""
        rivals = {}
        for name, model in (("plain", plain), ("dihedral", dihedral)):
            rivals[name] = np.mean(
                [
                    _place(model, dx, turned=False)["rival"]
                    for dx in (-100.0, 37.0, 120.0)
                ]
            )
        assert rivals["dihedral"] > rivals["plain"] + 0.05


class TestRepeatFromTheClasses:
    @pytest.mark.parametrize("fold", [1, 2])
    def test_the_repeat_is_the_same_with_half_the_classes_and_no_turned_segments(
        self, fold
    ):
        helix = Helix(pitch=700.0 * fold, fold=fold)
        df, repeat, _ = simulate(
            helix, n_fil=60, n_seg=40, misclass=0.1, seed=3, dyad=True
        )
        d = (df.rlnAnglePsi - df.rlnAnglePsiPrior + 180) % 360 - 180
        assert (np.abs(d) < 90).all()  # no polarity to find
        assert df.rlnClassNumber.nunique() == 12  # the plain views only
        result = ph.analyze(df, n_boot=3)
        assert result.period == pytest.approx(repeat, rel=0.02)


def _correlation(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def _class_averages(helix, seed, n=10):
    """Views at random azimuths, half of them from the other end of the filament (turned)."""
    rng = np.random.default_rng(seed)
    per = 360.0 / helix.fold
    azimuth = (np.arange(n) + rng.uniform(0.0, 1.0)) * per / n
    turned = rng.random(n) < 0.5
    return [
        np.asarray(
            helix.render(0.0, a, 180.0 if t else 0.0, size=48, pixel=5.0),
            dtype=np.float32,
        )
        for a, t in zip(azimuth, turned)
    ]


def _refine(helix, seed, two_fold):
    images = _class_averages(helix, seed)
    apix2d, apix3d, geometry = maps._joint_geometry(images, 5.0, helix.rise, 3)
    return J.joint_refine(
        images,
        None,
        -helix.twist,
        helix.rise,
        apix2d,
        apix3d,
        helix.fold,
        geometry,
        iterations=8,
        tolerance=0.5,
        two_fold=two_fold,
        algorithm=dict(model="gauss"),
    )


class TestTheSimulatedStructures:
    @pytest.mark.parametrize("rot", [20.0, 55.0, 100.0])
    def test_a_turned_view_is_the_plain_view_at_the_mirrored_azimuth_only_with_a_two_fold(
        self, rot
    ):
        for cls, same in ((DihedralHelix, True), (Helix, False)):
            helix = cls(pitch=1400.0, fold=2)
            turned = helix.render(0.0, rot, 180.0, size=48, pixel=5.0)
            mirrored = helix.render(0.0, 180.0 - rot, 0.0, size=48, pixel=5.0)
            assert (_correlation(turned, mirrored) > 0.99) is same


class TestHalfPeriodSimilarity:
    def test_a_projection_that_repeats_at_half_a_period_scores_one(self):
        x = np.arange(400)
        half = np.sin(2 * np.pi * x / 100.0) + 0.3 * np.sin(2 * np.pi * x / 33.0)
        model = np.tile(half, (8, 1))
        assert A.half_period_similarity(model, period=200.0) > 0.99

    def test_one_that_does_not_scores_low(self):
        rng = np.random.default_rng(0)
        assert abs(A.half_period_similarity(rng.normal(size=(8, 400)), 200.0)) < 0.2

    def test_a_flat_or_too_short_projection_scores_zero(self):
        assert A.half_period_similarity(np.ones((8, 400)), 200.0) == 0.0
        assert A.half_period_similarity(np.ones((8, 50)), 200.0) == 0.0


class TestJointRefinement:
    def test_a_dihedral_map_is_not_taken_for_a_two_fold_about_the_axis(self):
        """With C2 it was, in 3 of 8 runs: both placements of a view are as good, the flips
        looked like a half turn, and the search was halved (fit 0.86 against 0.91)."""
        auto = _refine(DihedralHelix(pitch=1400.0, fold=2), seed=2, two_fold="auto")
        assert auto["two_fold"] is False
        assert auto["history"][-1]["corr"] > 0.89

    def test_the_model_of_a_2_1_screw_repeats_at_half_a_period_and_a_dihedral_one_does_not(
        self,
    ):
        similarity = {}
        for name, helix in (
            ("screw", Screw21Helix(pitch=700.0, fold=1)),
            ("dihedral", DihedralHelix(pitch=700.0, fold=1)),
        ):
            res = _refine(helix, seed=0, two_fold=False)
            period = A.period_pixel(-helix.twist, helix.rise, 5.0, 1)
            similarity[name] = A.half_period_similarity(res["model_projection"], period)
        assert similarity["screw"] > J.TWO_FOLD_MIN_SIMILARITY
        assert similarity["dihedral"] < 0.4


class TestApolarHint:
    def _table(self, dyad):
        helix = Helix(pitch=1400.0, fold=2)
        df, _, _ = simulate(
            helix, n_fil=60, n_seg=20, flip_rate=0.3, misclass=0.1, seed=1, dyad=dyad
        )
        return df

    def test_a_dihedral_table_has_no_turned_segment(self):
        assert ph.looks_apolar(self._table(dyad=True)) is True

    def test_a_polar_table_picked_from_both_ends_has_turned_segments(self):
        assert ph.looks_apolar(self._table(dyad=False)) is False

    def test_too_little_data_or_missing_angles_say_nothing(self):
        df = self._table(dyad=True)
        assert ph.looks_apolar(df.head(100)) is False
        assert ph.looks_apolar(df.drop(columns=["rlnAnglePsiPrior"])) is False
        assert ph.looks_apolar(None) is False

    def test_the_user_is_told_for_a_dihedral_table_and_not_for_a_polar_one(self):
        title, text = ph.apolar_warning(self._table(dyad=True))
        assert "turned" in title and "dihedral" in text
        assert ph.apolar_warning(self._table(dyad=False)) is None


class TestDihedralScan:
    """The pairwise mirror test on class averages that are not centred on the dyad."""

    @staticmethod
    def _classes(helix, n=24):
        period = 360.0 / helix.fold
        az = (np.arange(n) + 0.37) * period / n
        imgs = [helix.render(0.0, a, 0.0, size=48, pixel=5.0) for a in az]
        return imgs, az, period

    def test_a_dihedral_helix_has_a_sharp_peak(self):
        from helicon.webApps.lib import helical_pitch_dihedral as dih

        r = dih.dihedral_scan(*self._classes(DihedralHelix(pitch=700.0, fold=1)))
        assert r["best_score"] > 0.9
        assert r["contrast"] > 0.4

    def test_a_plain_helix_has_none(self):
        from helicon.webApps.lib import helical_pitch_dihedral as dih

        r = dih.dihedral_scan(*self._classes(Helix(pitch=700.0, fold=1)))
        assert r["best_score"] < 0.75
        assert r["contrast"] < 0.3
