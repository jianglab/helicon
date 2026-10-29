"""The repeat from the segment pairs and the class-average maps together."""

from types import SimpleNamespace

import numpy as np

import synthetic_pitch as syn

from helicon.webApps.lib import helical_pitch_mapfit as mf
from helicon.webApps.lib import helical_pitch_phase as ph

PERIODS = np.exp(np.arange(np.log(300.0), np.log(1500.0), 0.04))


def _curves(meta_peak, map_peak, meta_width=0.15, map_width=0.2, seed=0):
    """A ring score and per-class map fits peaking where asked, with noise."""
    rng = np.random.default_rng(seed)
    lp = np.log(PERIODS)
    meta = 0.8 * np.exp(-0.5 * ((lp - np.log(meta_peak)) / meta_width) ** 2)
    boot = meta + 0.002 * rng.standard_normal((30, len(PERIODS)))
    fit = 0.7 + 0.05 * np.exp(-0.5 * ((lp - np.log(map_peak)) / map_width) ** 2)
    fits = fit[:, None] + 0.01 * rng.standard_normal((len(PERIODS), 20))
    return meta, boot, fits


class TestCombine:
    def test_the_ring_refines_the_repeat_where_its_pairs_reach(self):
        meta, boot, fits = _curves(meta_peak=700.0, map_peak=760.0)
        out = mf.combine(PERIODS, meta, boot, fits, np.full(len(PERIODS), 0.3))
        assert out["source"] == "metadata"
        assert abs(out["basin"] / 700.0 - 1) < 0.05

    def test_the_maps_decide_where_the_pairs_fall_short(self):
        # filaments of 350 A: the ring is precise and wrong, and has no pairs
        # beyond half of the true repeat
        meta, boot, fits = _curves(meta_peak=520.0, map_peak=700.0)
        half = np.where(PERIODS / 2 < 350.0, 0.02, 0.0)
        out = mf.combine(PERIODS, meta, boot, fits, half)
        assert out["source"] == "map"
        assert abs(out["period"] / 700.0 - 1) < 0.05

    def test_a_vetoed_repeat_cannot_win(self):
        meta, boot, fits = _curves(meta_peak=350.0, map_peak=350.0)
        gate = (np.abs(PERIODS - 350.0) > 40).astype(float)
        out = mf.combine(PERIODS, meta, boot, fits, np.full(len(PERIODS), 0.3), gate)
        assert abs(out["basin"] - 350.0) > 40


class TestHalfSupport:
    def test_the_share_of_pair_weight_beyond_half_the_repeat(self):
        pairs = SimpleNamespace(D=np.array([100.0, -300.0, 500.0]), W=np.ones(3))
        share = mf._half_support(pairs, np.array([100.0, 400.0, 2000.0]))
        assert np.allclose(share, [1.0, 2 / 3, 0.0])


class TestEstimate:
    def test_filaments_half_a_repeat_long(self):
        # the ring alone puts this repeat about 40% short (validation run of
        # 2026-10-28); with the maps it is within a few percent
        period, n_classes, apix = 700.0, 20, 4.0
        nx = 2 * int(round(0.96 * period / apix / 2))
        images, _ = syn.class_averages(
            period, n_classes, nx=nx, apix=apix, contamination=0.4, seed=12,
            structure_seed=2,
        )  # fmt: skip
        table = syn.class2d_table(
            period, n_classes, n_fil=200, length=0.5 * period, mislabel=0.4, seed=12
        )
        classes = list(range(1, n_classes + 1))
        r = ph.analyze(
            table,
            class_ids=classes,
            n_boot=2,
            image_apix=apix,
            class_images={c: images[c - 1] for c in classes},
            map_fit=dict(rise=4.75),
        )
        assert r.map_fit is not None and r.map_fit["csym"] == 1
        assert abs(r.period / period - 1) < 0.05
