"""Numbers of the main methods on synthetic data whose truth is known.

Each scenario runs a method on data made with fixed seeds and returns a few numbers that say
how well it did (an error against the truth, an agreement with the true groups). The numbers
are kept in ``quality_baselines.json``; ``test_quality_baselines.py`` fails when one gets worse
by more than its tolerance, so that a change which makes a method less accurate shows up even
though every unit test still passes. After a change that is meant to move a number, refresh
the file with ``python tests/update_quality_baselines.py`` and read the diff.

``SPECS`` says, for each kind of number, which way is better and how much it may move: the
numbers come from BLAS, thread counts and random starts, which differ a little between machines.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# the way each kind of number may move: (direction, tolerance); "max": lower is better,
# "min": higher is better. Chosen by what the number ends in.
SPECS = {
    "period_error_pct": ("max", 0.8),
    "phase_rms": ("max", 0.01),
    "found_ratio_error": ("max", 0.03),
    "gain": ("max", 0.5),
    "accuracy": ("min", 0.05),
    "ari": ("min", 0.05),
    "ll_per_segment": ("min", 0.05),
    "res_bias": ("max", 0.1),
    "res_true": ("max", 0.1),
    "signal_kept": ("min", 0.005),
    "rmse_ratio": ("max", 0.01),
}


def spec_for(name: str) -> tuple[str, float]:
    """The direction and the tolerance of a number, from the end of its name."""
    kind = name.rsplit(".", 1)[-1]
    return SPECS[kind]


def _synthetic_module(name: str = ""):
    """The synthetic data makers of the pitch tests (``tests/synthetic_pitch.py``)."""
    import synthetic_pitch

    return synthetic_pitch


def ring_period() -> dict[str, float]:
    """The repeat and the class phases from the segments' classes, pixels, or both."""
    from helicon.webApps.lib import helical_pitch_phase as ph
    from helicon.webApps.lib import helical_pitch_pixel as px

    syn = _synthetic_module()
    period, n_classes, apix = 400.0, 12, 4.0
    out = {}
    for label, mislabel in (("clean", 0.0), ("mislabelled", 0.4)):
        errors = {"meta": [], "pixel": [], "joint": []}
        phases = {"meta": [], "pixel": [], "joint": []}
        for seed in (0, 1):
            nx = 2 * int(round(0.96 * period / apix / 2))
            images, phase_true = syn.class_averages(
                period,
                n_classes,
                nx=nx,
                apix=apix,
                contamination=mislabel,
                seed=seed,
                structure_seed=0,
            )
            ids = list(range(n_classes))
            offsets = px.class_pixel_offsets(images, ids, apix)
            params = syn.class2d_table(
                period,
                n_classes,
                n_fil=60,
                length=400.0,
                mislabel=mislabel,
                position_sd=0.0,
                seed=seed + 1000,
            )
            pairs = ph.prepare_pairs(params, use_refined_path=False)
            for method, alpha in (("meta", 0.0), ("pixel", 1.0), ("joint", 0.5)):
                est = px.estimate_period_joint(pairs, offsets, alpha=alpha)
                errors[method].append(abs(100.0 * (est["period"] / period - 1.0)))
                phases[method].append(
                    syn.phase_error(est["phases"], phase_true, period)
                )
        for method in errors:
            out[f"ring.{method}.{label}.period_error_pct"] = float(
                np.mean(errors[method])
            )
            out[f"ring.{method}.{label}.phase_rms"] = float(np.mean(phases[method]))
    return out


def ring_map_period() -> dict[str, float]:
    """The repeat from the segments alone and with the class-average maps, on
    filaments half a repeat long (where the segments alone are biased)."""
    from helicon.webApps.lib import helical_pitch_phase as ph

    syn = _synthetic_module()
    period, n_classes, apix = 700.0, 20, 4.0
    nx = 2 * int(round(0.96 * period / apix / 2))
    images, _ = syn.class_averages(
        period,
        n_classes,
        nx=nx,
        apix=apix,
        contamination=0.4,
        seed=12,
        structure_seed=2,
    )
    table = syn.class2d_table(
        period, n_classes, n_fil=200, length=0.5 * period, mislabel=0.4, seed=12
    )
    classes = list(range(1, n_classes + 1))
    out = {}
    for label, fit in (("meta", None), ("maps", dict(rise=4.75))):
        r = ph.analyze(
            table,
            class_ids=classes,
            n_boot=2,
            image_apix=apix,
            class_images={c: images[c - 1] for c in classes},
            map_fit=fit,
        )
        out[f"ring_map.{label}.short.period_error_pct"] = abs(
            100.0 * (r.period / period - 1.0)
        )
    return out


def two_types() -> dict[str, float]:
    """Two helical types sharing their classes are found; a single type is not split."""
    from helicon.webApps.lib import helical_pitch_mixture as mx
    from helicon.webApps.lib import helical_pitch_phase as ph

    syn = _synthetic_module()

    def one(ratio, seed=0, length=800.0, share=0.5):
        periods = (400.0, 400.0 * ratio)
        params, types = syn.class2d_table_mixture(
            periods, (1 - share, share), 12, 200, length, mislabel=0.0, seed=seed
        )
        pairs = ph.prepare_pairs(params, use_refined_path=False, max_sep=2000.0)
        est = ph.estimate_period(pairs)
        est, _ = ph.sync_filament_directions(pairs, est)
        single = mx.one_ring_objective(pairs, est["period"], est["phases"])
        truth = np.array(
            [types[int(k[0][3:6]) * 10 + k[1] - 1] for k in pairs.filament_keys]
        )
        rng = np.random.default_rng(seed + 7)
        best = None
        for _ in range(6):
            labels = rng.integers(0, 2, pairs.n_filaments)
            two = mx.fit_two_rings(
                pairs, labels, (est["period"] * 0.93, est["period"] * 1.07)
            )
            if two is not None and (
                best is None or two["objective"] > best["objective"]
            ):
                best = two
        lo, hi = np.sort(best["periods"])
        accuracy = max(
            np.mean(best["labels"] == truth), np.mean(best["labels"] != truth)
        )
        return hi / lo, float(accuracy), float(best["objective"] - single)

    ratio, accuracy, _ = one(1.2)
    _, _, null_gain = one(1.0)
    return {
        "types.two_types.found_ratio_error": float(abs(ratio - 1.2)),
        "types.two_types.accuracy": accuracy,
        "types.one_type.gain": float(null_gain),
    }


def _walks(n_fil, noise, seed):
    """Filaments of three types, each walking its own six classes round and round."""
    rng = np.random.default_rng(seed)
    classes = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]]
    seqs, truth = [], []
    for _ in range(n_fil):
        t = int(rng.integers(3))
        n = int(rng.integers(30, 100))
        phase = int(rng.integers(6))
        seq = [classes[t][(phase + s) % 6] for s in range(n)]
        seq = [int(rng.integers(18)) if rng.random() < noise else c for c in seq]
        seqs.append([c + 1 for c in seq])
        truth.append(t)
    return seqs, np.array(truth)


def context_typing() -> dict[str, float]:
    """The types of filaments from the order of their classes (the filament-context HMM)."""
    from sklearn.metrics import adjusted_rand_score

    from helicon.webApps.lib import helical_pitch_context as hc

    sys.path.insert(0, str(ROOT / "tests"))
    from test_helical_pitch_context import _chains

    out = {}
    for label, noise, prob in (("clean", 0.2, 0.8), ("noisy", 0.8, 0.5)):
        seqs, truth = _walks(300, noise, seed=0)
        chains = _chains(seqs, prob=prob, n_classes=18)
        result = hc.fit(chains, 3, n_iter=4, cpu=2)
        types, _ = hc.filament_types(chains, result["tpost"])
        out[f"context.{label}.ari"] = float(adjusted_rand_score(truth, types))
        if label == "noisy":
            n_segments = sum(len(s) for s in seqs)
            out["context.noisy.ll_per_segment"] = float(result["ll_train"] / n_segments)
    return out


def true_fsc() -> dict[str, float]:
    """trueFSC's automatic mask on half maps of known signal and noise: how far the
    reported resolution is from the real one of the mask chosen, that real one
    (both as 1/area under the curve, which a small numerical difference does not
    move by a whole shell as it does the 0.143 crossing), and how much of the
    signal the mask keeps."""
    import tempfile

    import _truefsc_synthetic as syn

    from helicon.commands import trueFSC

    out = {}
    search = getattr(trueFSC._optimal_mask, "__wrapped__", trueFSC._optimal_mask)
    real, trueFSC._optimal_mask = trueFSC._optimal_mask, search  # no cached answer
    try:
        for label, maps, options in (
            ("particle", syn.particle(), {}),
            ("helix_central", syn.helix(), {"mask_z_fraction": 0.5}),
        ):
            with tempfile.TemporaryDirectory() as tmp:
                r = syn.run(Path(tmp), maps, **options)
            key = f"truefsc.{label}"
            out[f"{key}.res_bias"] = round(abs(r["measured"] - r["truth"]), 4)
            out[f"{key}.res_true"] = round(r["truth"], 4)
            out[f"{key}.signal_kept"] = round(r["signal_kept"], 4)
    finally:
        trueFSC._optimal_mask = real
    return out


def denoising() -> dict[str, float]:
    """Denoising of filaments in white noise (micrographs) and of a map in coloured
    noise: the error left against the clean image, as a fraction of the noisy one's.
    """
    from test_denoise import filaments, rmse, rods_map

    from helicon.lib.denoise import denoise_map, denoise_micrograph

    out = {}
    for label, noise_sd in (("snr_high", 1.0), ("snr_low", 2.5)):
        clean, noisy = filaments(shape=(256, 256), noise_sd=noise_sd, seed=4)
        for strength in (1.0, 2.0):
            denoised, _ = denoise_micrograph(noisy, strength=strength, workers=1)
            out[f"denoise.{label}.strength{strength:g}.rmse_ratio"] = round(
                rmse(denoised, clean) / rmse(noisy, clean), 4
            )
    clean, noisy = rods_map(n=64, noise_sd=1.0, seed=4)
    denoised, _ = denoise_map(noisy, strength=2.0, workers=1)
    out["denoise.map.strength2.rmse_ratio"] = round(
        rmse(denoised, clean) / rmse(noisy, clean), 4
    )
    return out


SCENARIOS = {
    "ring_period": ring_period,
    "ring_map_period": ring_map_period,
    "two_types": two_types,
    "context_typing": context_typing,
    "true_fsc": true_fsc,
    "denoising": denoising,
}


def run_all() -> dict[str, float]:
    """Every number of every scenario."""
    numbers = {}
    for scenario in SCENARIOS.values():
        numbers.update(scenario())
    return numbers
