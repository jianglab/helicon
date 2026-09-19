"""The transform machinery shared by denovo3D and helicalProjection.

Both tabs ask the same question of a class average -- how far from horizontal,
how far off centre, how wide -- so the estimators, the auto-transform decision
and the per-image card switching live in one module. These cover the shared
behaviour and the two ways sharing the script went wrong on a page that has
both tabs on it.
"""

import json

import numpy as np
import pytest

from helicon.webApps.lib import helix_transform


def _filament(ny=64, nx=128, rotation=0.0, shift=0.0, width=6.0):
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float32)
    y = y - ny // 2 - shift
    x = x - nx // 2
    ang = np.deg2rad(rotation)
    across = -x * np.sin(ang) + y * np.cos(ang)
    along = x * np.cos(ang) + y * np.sin(ang)
    img = np.exp(-(across**2) / (2 * width**2))
    img[np.abs(along) > nx * 0.4] = 0
    return img.astype(np.float32)


class TestAutoTransform:
    def test_it_measures_each_image_separately(self):
        images = [_filament(rotation=r) for r in (-8.0, 0.0, 11.0)]
        auto = helix_transform.auto_transform(images)
        assert len(auto.per_image) == 3
        found = [r for r, _ in auto.per_image]
        # the point of per-image values: one shared number cannot straighten
        # images that sit at different angles
        assert max(found) - min(found) > 10.0
        for measured, applied in zip(found, (-8.0, 0.0, 11.0)):
            assert measured == pytest.approx(-applied, abs=3.0)

    def test_it_reports_a_crop_that_fits_every_image(self):
        images = [_filament(width=w) for w in (4.0, 10.0)]
        auto = helix_transform.auto_transform(images)
        assert auto.crop_size > 0
        assert auto.crop_size % 4 == 0
        assert auto.ny == 64 and auto.nx == 128

    def test_it_refuses_an_empty_set(self):
        with pytest.raises(ValueError):
            helix_transform.auto_transform([])


class TestCardSwitchingScript:
    def _js(self, galleries):
        return str(helix_transform.card_switching_script(galleries))

    def test_each_tab_gets_its_own_configuration(self):
        # as a plain global `var`, the second tab to load overwrote the first
        # and that tab's cards stopped switching entirely
        js = self._js([{"input": "a", "cards": ".a-card", "key": "pi"}])
        assert "(function ()" in js
        assert "var _CARD_GALLERIES" not in js

    def test_the_gallery_id_is_matched_exactly(self):
        # a substring match also caught Shiny's own clientdata inputs --
        # `.clientdata_output_hill-hill_display_selected_image_bg`, whose value
        # is a CSS colour -- and that colour became the card index
        js = self._js([{"input": "display_selected_image", "cards": ".c", "key": "pi"}])
        assert "e.name.indexOf(g.input) < 0" not in js
        assert "e.name !== g.input" in js
        assert "parseInt(idx, 10)" in js

    def test_the_configuration_survives_as_json(self):
        galleries = [
            {"input": "dn_active_image", "cards": ".dn-pi-card", "key": "pi"},
            {"input": "hp_active", "cards": ".hp-pi-card", "key": "pi"},
        ]
        js = self._js(galleries)
        assert json.dumps(galleries) in js


class TestAutoTransformCache:
    """Measuring is the whole cost, so an image is measured at most once.

    The helicalProjection tab re-runs the transform on every change to the
    selection. Without a cache, adding a fourth image re-measures the three
    that were already there, and the delay grows with the size of the
    selection rather than with what changed.
    """

    def _images(self):
        return [_filament(rotation=r) for r in (-8.0, 0.0, 11.0)]

    def test_a_second_call_measures_only_what_is_new(self, monkeypatch):
        calls = []
        real = helix_transform.refine_helix_rotation_center

        def counted(image, **kwargs):
            calls.append(1)
            return real(image, **kwargs)

        monkeypatch.setattr(helix_transform, "refine_helix_rotation_center", counted)

        images = self._images()
        cache = {}
        helix_transform.auto_transform(images[:2], cache=cache)
        assert len(calls) == 2
        helix_transform.auto_transform(images, cache=cache)
        assert len(calls) == 3, "the two already measured should not be redone"

    def test_the_cached_result_is_the_measured_one(self):
        images = self._images()
        cache = {}
        first = helix_transform.auto_transform(images, cache=cache)
        second = helix_transform.auto_transform(images, cache=cache)
        assert first.per_image == second.per_image
        assert first.crop_size == second.crop_size
        assert first.diameter == second.diameter

    def test_without_a_cache_nothing_is_kept(self, monkeypatch):
        calls = []
        real = helix_transform.refine_helix_rotation_center

        def counted(image, **kwargs):
            calls.append(1)
            return real(image, **kwargs)

        monkeypatch.setattr(helix_transform, "refine_helix_rotation_center", counted)
        images = self._images()
        helix_transform.auto_transform(images)
        helix_transform.auto_transform(images)
        assert len(calls) == 6

    def test_images_are_told_apart_by_content(self):
        a = _filament(rotation=-8.0)
        b = _filament(rotation=11.0)
        assert helix_transform.image_key(a) != helix_transform.image_key(b)
        assert helix_transform.image_key(a) == helix_transform.image_key(a.copy())


class TestApplyTransform:
    """Thresholding, straightening and cropping, as both tabs apply them."""

    def test_the_smallest_offered_crop_is_still_applied(self):
        # the bug: `32 < crop_size` skipped the crop at exactly 32, which is
        # both the minimum offered and what the auto transform clamps thin
        # filaments to, so 10 of 42 EMPIAR-10940 classes came out rotated but
        # full height and looked as though nothing had run
        image = _filament(ny=128, nx=128)
        out = helix_transform.apply_transform(image, crop_size=32)
        assert out.shape == (32, 128)

    @pytest.mark.parametrize("crop", [16, 32, 33, 64, 127])
    def test_every_crop_smaller_than_the_image_applies(self, crop):
        image = _filament(ny=128, nx=128)
        assert helix_transform.apply_transform(image, crop_size=crop).shape[0] == crop

    @pytest.mark.parametrize("crop", [None, 128, 256, 0])
    def test_a_crop_that_is_not_smaller_leaves_the_image_alone(self, crop):
        image = _filament(ny=128, nx=128)
        assert helix_transform.apply_transform(image, crop_size=crop).shape == (
            128,
            128,
        )

    def test_rotation_and_crop_compose(self):
        image = _filament(ny=128, nx=128, rotation=10.0)
        out = helix_transform.apply_transform(image, rotation=-10.0, crop_size=32)
        assert out.shape == (32, 128)
        # straightened: the brightest row of the crop is near its middle
        assert abs(int(np.argmax(out.sum(axis=1))) - 16) <= 3

    def test_no_arguments_returns_an_equivalent_image(self):
        image = _filament(ny=64, nx=128)
        assert np.array_equal(helix_transform.apply_transform(image), image)
