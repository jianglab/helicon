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
}


def spec_for(name: str) -> tuple[str, float]:
    """The direction and the tolerance of a number, from the end of its name."""
    kind = name.rsplit(".", 1)[-1]
    return SPECS[kind]


def _synthetic_module(name: str):
    """A module of ``benchmarks/<name>`` (the synthetic data makers live there)."""
    import importlib

    folder = str(ROOT / "benchmarks" / name)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    return importlib.import_module("synthetic")


def ring_period() -> dict[str, float]:
    """The repeat and the class phases from the segments' classes, pixels, or both."""
    from helicon.webApps.lib import helical_pitch_phase as ph
    from helicon.webApps.lib import helical_pitch_pixel as px

    syn = _synthetic_module("joint_pixel")
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


def two_types() -> dict[str, float]:
    """Two helical types sharing their classes are found; a single type is not split."""
    from helicon.webApps.lib import helical_pitch_mixture as mx
    from helicon.webApps.lib import helical_pitch_phase as ph

    syn = _synthetic_module("joint_pixel")

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


SCENARIOS = {
    "ring_period": ring_period,
    "two_types": two_types,
    "context_typing": context_typing,
}


def run_all() -> dict[str, float]:
    """Every number of every scenario."""
    numbers = {}
    for scenario in SCENARIOS.values():
        numbers.update(scenario())
    return numbers
