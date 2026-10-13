"""The spectral way of straightening and centring a filament, next to the threshold one.

Projection sharpness gives the angle and the phases across the meridian give
the centre; neither takes a threshold or minds whether the filament is bright
or dark. The checks use a simulated helix, whose true placement is known.
"""

import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

import helicon
from helicon.webApps import auto_method_ui
from helicon.webApps.lib import helix_transform as ht
from helicon.webApps.lib import hill_compute


def _helix(n=128):
    """A simulated helix with its axis horizontal and centred, non-negative."""
    img = hill_compute.simulate_helix(29.4, 21.92, 6, 77.6, 8.0, n, n, 2.3438, az0=17.0)
    img = img.T
    return img - img.min()


def _placed(img, rotation, down):
    return ndi.shift(
        ndi.rotate(img, rotation, reshape=False, order=3), (down, 0), order=3
    )


def _left_over(clean, rotation, shift):
    """Tilt (deg) and off-centre (px) of the clean image after a transform."""
    out = helicon.transform_image(image=clean, rotation=rotation)
    if shift:
        out = helicon.transform_image(image=out, post_translation=(shift, 0))
    tilt = ht._sharpness_rotation(out)
    straight = helicon.transform_image(image=out, rotation=tilt) if tilt else out
    return abs(tilt), abs(ht.meridian_offset(straight))


class TestSpectralEstimate:
    @pytest.mark.parametrize(
        "rotation, down", [(-35.0, 9.0), (4.0, -12.0), (52.0, 3.0)]
    )
    def test_it_straightens_and_centres_a_helix(self, rotation, down):
        clean = _placed(_helix(), rotation, down)
        r, s, _ = ht.spectral_rotation_center_diameter(clean)
        tilt, off = _left_over(clean, r, s)
        assert tilt < 0.3
        assert off < 0.5

    def test_the_sign_of_the_filament_makes_no_difference(self):
        clean = _placed(_helix(), 17.0, -6.0)
        bright = ht.spectral_rotation_center_diameter(clean)
        dark = ht.spectral_rotation_center_diameter(-clean)
        assert dark[0] == pytest.approx(bright[0], abs=0.05)
        assert dark[1] == pytest.approx(bright[1], abs=0.05)
        assert dark[2] == bright[2]

    def test_it_holds_up_in_noise(self):
        # three times the noise of the picture's spread: where a threshold fails
        base = _helix()
        clean = _placed(base, -22.0, 7.0)
        noisy = clean + np.random.default_rng(0).normal(0, 3 * base.std(), clean.shape)
        r, s, _ = ht.spectral_rotation_center_diameter(noisy)
        tilt, off = _left_over(clean, r, s)
        assert tilt < 0.5
        assert off < 0.8

    def test_what_is_not_asked_for_is_not_estimated(self):
        clean = _placed(_helix(), 25.0, 8.0)
        r, s, _ = ht.spectral_rotation_center_diameter(clean, estimate_rotation=False)
        assert r == 0.0
        r, s, _ = ht.spectral_rotation_center_diameter(
            _placed(_helix(), 0.0, 8.0), estimate_center=False
        )
        assert s == 0.0

    def test_the_width_is_that_of_the_filament_and_not_of_its_halo(self):
        y = np.arange(128.0) - 64
        profile = np.exp(-(y**2) / (2 * 6.0**2))
        plain = np.tile(profile[:, None], (1, 128))
        # class averages have a halo of the opposite sign around the filament,
        # weaker than the filament itself
        halo = plain - 0.3 * np.tile(np.exp(-(y**2) / (2 * 24.0**2))[:, None], (1, 128))
        expected = (
            2 * 6.0 * np.sqrt(2 * np.log(5.0))
        )  # where a Gaussian falls to a fifth
        assert ht._profile_width(plain) == pytest.approx(expected, abs=3)
        assert ht._profile_width(halo) < 2 * expected
        assert ht._profile_width(-halo) == ht._profile_width(halo)


class TestMethods:
    def test_the_default_is_the_threshold_one(self):
        assert ht.DEFAULT_METHOD == "threshold"
        assert set(ht.METHODS) == {"threshold", "spectral", "pd"}

    def test_the_threshold_method_is_what_it_was(self):
        img = _placed(_helix(), 12.0, 4.0)
        threshold = img.max() * 0.2
        assert ht.estimate_rotation_center_diameter(
            img, method="threshold", threshold=threshold
        ) == ht.refine_helix_rotation_center(img, threshold=threshold)
        assert ht.estimate_rotation_center_diameter(
            img, method="threshold", refine=False
        ) == helicon.estimate_helix_rotation_center_diameter(img)

    def test_the_spectral_method_is_reached_by_its_name(self):
        img = _placed(_helix(), 12.0, 4.0)
        assert ht.estimate_rotation_center_diameter(
            img, method="spectral"
        ) == ht.spectral_rotation_center_diameter(img)

    def test_an_unknown_method_is_refused(self):
        with pytest.raises(ValueError, match="unknown method"):
            ht.estimate_rotation_center_diameter(_helix(64), method="magic")

    def test_auto_transform_keeps_the_methods_apart_in_its_cache(self):
        img = _placed(_helix(), -14.0, 5.0)
        cache = {}
        a = ht.auto_transform([img], cache=cache, method="threshold")
        b = ht.auto_transform([img], cache=cache, method="spectral")
        assert len(cache) == 2
        assert {k[-1] for k in cache} == {"threshold", "spectral"}
        # and a second ask of either is the cached answer
        assert (
            ht.auto_transform([img], cache=cache, method="spectral").per_image
            == b.per_image
        )
        assert (
            ht.auto_transform([img], cache=cache, method="threshold").per_image
            == a.per_image
        )

    def test_auto_transform_with_the_spectral_method_straightens(self):
        clean = _placed(_helix(), -14.0, 5.0)
        auto = ht.auto_transform([clean], method="spectral")
        r, s = auto.per_image[0]
        tilt, off = _left_over(clean, r, s)
        assert tilt < 0.3 and off < 0.5


class TestHillStandsItUpright:
    """HILL keeps the axis along y, where the others keep it along x."""

    @staticmethod
    def _upright(clean, apix, method):
        auto = hill_compute.auto_upright_transform(clean, apix, method)
        assert set(auto) == {"angle", "dx", "dy", "mask_radius"}
        assert auto["dy"] == 0.0  # the shift is across the axis, which is x
        out = hill_compute.transform_2d_image(
            clean, auto["angle"], auto["dx"], auto["dy"], False, apix
        )
        # upright: laid on its side it is horizontal, and then it is centred
        return _left_over(out.T, 0.0, 0.0)

    # the threshold method is HILL's old single estimate, good to a degree or two
    @pytest.mark.parametrize(
        "method, tilt_limit", [("threshold", 2.0), ("spectral", 0.5)]
    )
    def test_the_transform_uprights_and_centres(self, method, tilt_limit):
        apix = 2.3438
        clean = ndi.rotate(
            _placed(_helix(), -30.0, 8.0).T, 17.0, reshape=False, order=3
        )
        # class averages have a mean of about 0, which the threshold method needs
        tilt, off = self._upright(clean - clean.mean(), apix, method)
        assert tilt < tilt_limit
        assert off < 2 * tilt_limit

    @pytest.mark.parametrize("method", ["threshold", "spectral"])
    def test_an_upright_filament_off_centre_is_moved_across_and_not_along_itself(
        self, method
    ):
        # the shift used to be passed as dy, which for an upright filament moves
        # it along itself and leaves it as far off centre as it was
        apix = 2.3438
        clean = ndi.shift(_helix().T, (0, 9.0), order=3)  # 9 px off in x
        tilt, off = self._upright(clean - clean.mean(), apix, method)
        assert tilt < 1.0
        assert off < 1.0


class TestTheControl:
    """The three apps offer the same choice with the same default."""

    def test_its_choices_are_the_methods(self):
        assert auto_method_ui.CHOICES == ht.METHODS
        assert auto_method_ui.DEFAULT == ht.DEFAULT_METHOD
        assert "threshold" in auto_method_ui.HELP.lower()

    @pytest.mark.parametrize(
        "module, key",
        [
            ("hill_bookmark", "auto_method"),
            ("denovo3d_bookmark", "auto_method"),
        ],
    )
    def test_the_bookmark_default_is_the_methods_default(self, module, key):
        import importlib

        table = importlib.import_module(
            f"helicon.webApps.tabs.{module}"
        ).BOOKMARK_DEFAULTS
        assert table[key][1] == ht.DEFAULT_METHOD

    def test_every_app_declares_it_and_passes_it_on(self):
        root = Path(inspect.getsourcefile(auto_method_ui)).parent / "tabs"
        for tab, input_id in (
            ("hill_tab.py", "hill_auto_method"),
            ("denovo3d_tab.py", "dn_auto_method"),
            ("helical_projection_tab.py", "auto_method"),
        ):
            src = (root / tab).read_text()
            assert re.search(rf'ui\.input_select\(\s*"{input_id}"', src), tab
            assert f"input.{input_id}()" in src, tab
            assert "auto_method_ui.CHOICES" in src, tab
        projection = (root / "helical_projection_tab.py").read_text()
        assert re.search(r'"auto_method": \("auto_method", "threshold"\)', projection)


class TestTheControlSaysWhatItCovers:
    def test_the_help_names_the_scope_and_what_is_outside_it(self):
        from helicon.webApps import auto_method_ui

        assert "straightens and centres filaments automatically" in auto_method_ui.HELP
        assert "tube diameter" in auto_method_ui.HELP  # a separate estimate


class TestPhaseDifferenceMethod:
    """Rotation and centre together, by the weighted sum of cos(2 PD)."""

    def _placed_helix(self, tilt, down, noise=0.5, seed=0):
        img = hill_compute.simulate_helix(
            29.4, 21.92, 6, 77.6, 8.0, 128, 128, 4.6876, az0=17.0
        ).T
        img = helicon.transform_image(
            image=img - img.mean(), rotation=tilt, post_translation=(down, 0)
        )
        return img + np.random.default_rng(seed).normal(0, noise * img.std(), img.shape)

    @pytest.mark.parametrize("tilt,down", [(0.4, 2.0), (-0.7, -3.0), (0.6, -1.0)])
    def test_it_undoes_a_known_tilt_and_shift(self, tilt, down):
        rotation, shift, _ = ht.estimate_rotation_center_diameter(
            self._placed_helix(tilt, down), method="pd"
        )
        assert rotation == pytest.approx(-tilt, abs=0.2)
        assert shift == pytest.approx(-down, abs=0.2)

    def test_the_phase_of_a_clean_helix_is_zero_or_a_half_turn(self):
        img = hill_compute.simulate_helix(
            29.4, 21.92, 6, 77.6, 8.0, 128, 128, 4.6876, az0=17.0
        ).T
        phase, weight, freq = ht._pd_terms(img - img.mean())
        offset, score = ht._pd_best_offset(phase, weight, freq, 0.0)
        assert abs(offset) < 0.1
        assert 0.8 < score <= 1.0 + 1e-9
        # a noise-only image has random phases: nothing to agree on
        noise = np.random.default_rng(1).normal(size=(128, 128))
        phase, weight, freq = ht._pd_terms(noise)
        assert abs(ht._pd_best_offset(phase, weight, freq, 0.0)[1]) < 0.3

    def test_a_shift_does_not_change_the_score_once_it_is_solved_for(self):
        img = hill_compute.simulate_helix(
            29.4, 21.92, 6, 77.6, 8.0, 128, 128, 4.6876, az0=17.0
        ).T
        img = img - img.mean()
        scores = []
        for down in (0.0, 2.5):
            moved = helicon.transform_image(image=img, post_translation=(down, 0))
            phase, weight, freq = ht._pd_terms(moved)
            scores.append(
                ht._pd_best_offset(phase, weight, freq, ht.meridian_offset(moved))[1]
            )
        assert scores[0] == pytest.approx(scores[1], abs=0.05)

    def test_the_default_does_not_change(self):
        assert ht.DEFAULT_METHOD == "threshold"
