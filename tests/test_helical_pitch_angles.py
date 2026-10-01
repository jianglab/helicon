"""Euler angles of the AbInitio3D star files, validated on simulated helices."""

import numpy as np
import pandas as pd
import pytest

from helical_sim import Helix, simulate
from helicon.webApps.lib import helical_pitch_phase as ph


def _wrap(a, period=360.0):
    return (a + period / 2) % period - period / 2


def _errors(fold, **kw):
    """Rot and psi errors after removing the free global gauge.

    The gauge is the origin of rot, the hand (rot -> -rot), and 180 degrees of
    psi; a single one is fitted for the whole data set, so filament-to-filament
    coherence is what is measured.
    """
    hx = Helix(pitch=700.0 * fold, fold=fold)
    df, _, _ = simulate(hx, **kw)
    r = ph.analyze(df, n_boot=3)
    out = ph.select_segments(
        df, r.filaments, angles=ph.segment_angles(df, r, fold=fold)
    )
    truth = df.loc[out.index]
    # the angle is evaluated at the refined centre, which the shift error moved
    rot_t = truth.true_rot_est.values
    per = 360.0 / fold
    best = None
    for mu in (1, -1):
        for dpsi in (0.0, 180.0):
            d = mu * out.rlnAngleRot.values - rot_t
            g = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(d * fold)).mean())) / fold
            e_rot = np.abs(_wrap(d - g, per))
            e_psi = np.abs(_wrap(out.rlnAnglePsi.values + dpsi - truth.true_psi.values))
            score = np.median(e_rot) + np.median(e_psi)
            if best is None or score < best[0]:
                best = (score, e_rot, e_psi)
    return best[1], best[2], out, truth


class TestSimulatedHelix:
    @pytest.mark.parametrize("fold", [1, 2])
    def test_rot_and_psi_are_coherent_across_filaments(self, fold):
        e_rot, e_psi, out, _ = _errors(
            fold, n_fil=100, n_seg=40, flip_rate=0.3, seed=fold
        )
        assert len(out) == 100 * 40
        assert np.median(e_rot) < 4.0 and np.percentile(e_rot, 90) < 12.0
        assert np.median(e_psi) < 4.0 and np.percentile(e_psi, 90) < 12.0

    def test_segments_put_in_the_wrong_class_keep_a_good_angle(self):
        e_rot, e_psi, _, truth = _errors(
            2, n_fil=100, n_seg=40, flip_rate=0.3, misclass=0.4, seed=5
        )
        assert np.percentile(e_rot, 95) < 20.0
        assert np.mean(e_psi > 60.0) < 0.01

    def test_the_output_is_complete_and_uses_the_priors(self):
        hx = Helix()
        df, _, _ = simulate(hx, n_fil=30, n_seg=30, flip_rate=0.3, seed=3)
        r = ph.analyze(df, n_boot=3)
        out = ph.select_segments(df, r.filaments, angles=ph.segment_angles(df, r, 2))
        for col in ("rlnAngleRot", "rlnAnglePsi", "rlnAnglePsiPrior"):
            assert out[col].notna().all()
        assert (out["rlnAngleTilt"] == 90.0).all()
        assert (out["rlnAngleTiltPrior"] == 90.0).all()
        # the origins stay near the Class2D ones, moved across the path
        for col in ("rlnOriginXAngst", "rlnOriginYAngst"):
            assert out[col].notna().all()
            assert (out[col] - df.loc[out.index, col]).abs().max() < 60.0
        # one direction per filament: prior and psi differ by the same 0/180
        # for all of its segments
        turn = np.round(
            np.abs(
                _wrap(out["rlnAnglePsiPrior"] - df.loc[out.index, "rlnAnglePsiPrior"])
            )
            / 180.0
        )
        key = out["rlnMicrographName"] + out["rlnHelicalTubeID"].astype(str)
        assert (turn.groupby(key).nunique() == 1).all()


def _perpendicular_error(df, centres):
    """Error of centres across the filament (RELION's y runs down), in A."""
    a = np.deg2rad(df["true_psi"].values)
    across = np.stack([np.sin(a), np.cos(a)], 1)
    true = df[["true_center_x", "true_center_y"]].values
    return ((centres - true) * across).sum(1)


class TestCentresAndDirection:
    def test_centres_are_moved_onto_the_path_and_psi_is_smoothed(self):
        hx = Helix(pitch=1400.0, fold=2)
        df, _, _ = simulate(
            hx,
            n_fil=60,
            n_seg=40,
            flip_rate=0.3,
            bend=40.0,
            class_dy_sd=6.0,
            ydir=-1.0,
            seed=4,
        )
        r = ph.analyze(df, n_boot=3)
        ang = ph.segment_angles(df, r, fold=2)
        coord = df[["rlnCoordinateX", "rlnCoordinateY"]].values
        before = coord - df[["rlnOriginXAngst", "rlnOriginYAngst"]].values
        after = coord - ang[["rlnOriginXAngst", "rlnOriginYAngst"]].values
        e_before = _perpendicular_error(df, before)
        e_after = _perpendicular_error(df, after)
        assert np.sqrt(np.mean(e_after**2)) < 0.6 * np.sqrt(np.mean(e_before**2))
        raw = np.abs(_wrap(df["rlnAnglePsi"].values - df["true_psi"].values, 180.0))
        smooth = np.abs(_wrap(ang["rlnAnglePsi"].values - df["true_psi"].values, 180.0))
        assert np.median(smooth) < 0.5 * np.median(raw)

    def test_the_micrograph_y_direction_is_read_from_the_priors(self):
        hx = Helix()
        for ydir in (1.0, -1.0):
            df, _, _ = simulate(hx, n_fil=20, n_seg=30, ydir=ydir, seed=1)
            assert ph._direction_convention(df) == ydir

    def test_classes_kept_apart_by_direction_are_tied_by_their_pixels(self):
        hx = Helix(pitch=1400.0, fold=2)
        df, _, info = simulate(
            hx,
            n_fil=100,
            n_seg=40,
            n_views=12,
            flip_rate=0.0,
            misclass=0.0,
            turned_offset_sd=25.0,
            seed=0,
        )
        images = [
            hx.render(0.0, az, 180.0 if turned else 0.0)
            for az, turned in zip(info["azimuth"], info["turned"])
        ]
        pairs = ph.pair_counterparts(images, range(len(images)), min_corr=0.8)
        assert len(pairs) >= 8
        pairs = [dict(a=d["a"] + 1, b=d["b"] + 1, dx=d["dx"]) for d in pairs]

        def rot_error(counterparts):
            r = ph.analyze(df, n_boot=3, counterparts=counterparts, image_apix=5.0)
            out = ph.select_segments(
                df, r.filaments, angles=ph.segment_angles(df, r, fold=2)
            )
            truth = df.loc[out.index]
            best = None
            for mu in (1, -1):
                d = mu * out["rlnAngleRot"].values - truth["true_rot_est"].values
                g = np.rad2deg(np.angle(np.exp(2j * np.deg2rad(d)).mean())) / 2.0
                e = np.abs(_wrap(d - g, 180.0))
                if best is None or np.median(e) < np.median(best):
                    best = e
            return np.percentile(best, 90)

        # one set of classes has no fixed azimuth relative to the other
        assert rot_error(None) > 30.0
        assert rot_error(pairs) < 15.0


class TestDirectionScore:
    def test_short_filaments_score_lower_than_long_ones_and_the_column_is_written(self):
        hx = Helix(pitch=1400.0, fold=2)
        long_, _, _ = simulate(hx, n_fil=40, n_seg=40, flip_rate=0.3, seed=1)
        short, _, _ = simulate(
            hx, n_fil=200, n_seg=5, flip_rate=0.3, seed=2, misclass=0.3
        )
        short["rlnMicrographName"] = "s_" + short["rlnMicrographName"]
        both = pd.concat([long_, short], ignore_index=True)
        r = ph.analyze(both, n_boot=3)
        table = r.filaments
        is_short = table["rlnMicrographName"].str.startswith("s_")
        assert table["direction_score"].between(0.0, 1.0).all()
        long_score = table.loc[~is_short, "direction_score"]
        short_score = table.loc[is_short, "direction_score"]
        assert long_score.median() > short_score.median()
        assert short_score.quantile(0.25) < 0.7 < long_score.quantile(0.25)
        out = ph.select_segments(both, table)
        assert out["heliconFilamentDirectionScore"].between(0.0, 1.0).all()
