"""Filaments told apart by the alignment of their class sequences and by their own repeat.

``filament_alignment.py`` showed that an ungapped alignment of the class sequences of
two filaments separates types that differ in their classes, but not types that share
their classes and differ in pitch: a pitch difference only makes the two walks around
the ring drift apart slowly, and a local alignment takes the best run over all shifts.
What shows a pitch is a rate fitted over the whole filament. Here each filament gets
its own repeat from its class sequence alone (no ring, which for a mixture is a blend):
the smallest lag at which the sequence matches itself nearly as well as at its best.
The distance of a pair is then the alignment score per letter, minus a penalty for
the relative difference of the two repeats when both filaments have one.

    python filament_combined.py --reference reference.parquet \
        --mrcs run_it025_classes.mrcs --similarity similarity.npy
"""

from __future__ import annotations

import argparse

import numba as nb
import numpy as np

from filament_alignment import add_arguments, auc, prepare


@nb.njit(parallel=True, cache=True)
def _self_repeat(seq, off, M, dmin, dmax, min_overlap, near_best):
    """Per filament: the lag (segments) at which its class sequence returns, and how
    strongly.

    Neighbouring segments mostly share a class, so the match of a filament with
    itself is strongest at short lags and falls as the walk leaves a class. The repeat
    is where it comes back: after the match has first fallen to zero (or below), the
    smallest lag scoring at least ``near_best`` of the best of the later lags. The
    match at a lag is the mean substitution score of the letters that far apart,
    times the square root of their number (a z-like score). 0 where the filament is
    too short for a lag beyond the first fall.
    """
    n = len(off) - 1
    lag = np.zeros(n)
    strength = np.zeros(n)
    for f in nb.prange(n):
        s = seq[off[f] : off[f + 1]]
        L = len(s)
        top = min(dmax, L - min_overlap)
        if top <= dmin:
            continue
        mean = np.zeros(top + 1)
        z = np.zeros(top + 1)
        for d in range(dmin, top + 1):
            tot, cnt = 0.0, 0
            for i in range(L - d):
                x, y = s[i], s[i + d]
                if x >= 0 and y >= 0:
                    tot += M[x, y]
                    cnt += 1
            if cnt >= min_overlap:
                mean[d] = tot / cnt
                z[d] = mean[d] * np.sqrt(cnt)
        fall = -1
        for d in range(dmin, top + 1):
            if mean[d] <= 0.0:
                fall = d
                break
        if fall < 0 or fall >= top:
            continue
        zmax = z[fall + 1 : top + 1].max()
        if zmax <= 0:
            continue
        for d in range(fall + 1, top + 1):
            if z[d] >= near_best * zmax:
                lag[f] = d
                strength[f] = zmax
                break
    return lag, strength


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    p.add_argument("--min-lag", type=int, default=1, help="segments")
    p.add_argument("--max-lag", type=int, default=200, help="segments")
    p.add_argument("--min-overlap", type=int, default=10)
    p.add_argument("--near-best", type=float, default=0.85)
    p.add_argument(
        "--min-strength",
        type=float,
        default=2.0,
        help="z of the self-match for a repeat to count",
    )
    a = p.parse_args()

    d = prepare(a)
    sets, seqs, truth, length, A, M, step = (
        d["sets"],
        d["seqs"],
        d["truth"],
        d["length"],
        d["A"],
        d["M"],
        d["step"],
    )
    n = len(seqs)
    off = np.r_[0, np.cumsum([len(x) for x in seqs])].astype(np.int64)
    seq = np.concatenate(seqs).astype(np.int32)
    lag, strength = _self_repeat(
        seq, off, M, a.min_lag, a.max_lag, a.min_overlap, a.near_best
    )
    repeat = lag * step
    has = (lag > 0) & (strength >= a.min_strength)
    print(
        f"\nrepeat of a filament from its own class sequence: {int(has.sum())} of {n} filaments have one (step {step:.2f} A)"
    )
    for i, name in enumerate(sets):
        m = has & (truth == i)
        if m.sum() >= 5:
            q = np.percentile(repeat[m], [25, 50, 75])
            print(
                f"  {name}: {int(m.sum())} filaments, repeat {q[1]:.0f} A (middle half {q[0]:.0f} to {q[2]:.0f})"
            )

    iu = np.triu_indices(n, 1)
    ti, tj = truth[iu[0]], truth[iu[1]]
    same = ti == tj
    minlen = np.minimum(length[iu[0]], length[iu[1]])
    per_letter = A[iu] / minlen
    both = has[iu[0]] & has[iu[1]]
    rp = repeat[iu[0]], repeat[iu[1]]
    reldiff = np.where(
        both, np.abs(rp[0] - rp[1]) / np.maximum((rp[0] + rp[1]) / 2, 1.0), 0.0
    )

    lams = (0.0, 1.0, 2.0, 4.0, 8.0)
    print(
        "\nAUC of telling pairs of one set from pairs of two sets, by the weight of the repeat difference"
    )
    print(
        "  (the score is alignment per letter minus weight x relative repeat difference, where both have a repeat)"
    )
    print(
        f"{'pair of sets':44s}"
        + "".join(f"{('w=%g' % w):>8s}" for w in lams)
        + f"{'repeat only':>13s}"
    )

    def row(label, within, diff, lo):
        m = minlen >= lo
        cells = []
        for w in lams:
            sc = per_letter - w * reldiff
            cells.append(auc(sc[m & within], sc[m & diff]))
        both_m = m & both
        only = auc(-reldiff[both_m & within], -reldiff[both_m & diff])
        print(f"{label:44s}" + "".join(f"{c:8.3f}" for c in cells) + f"{only:13.3f}")

    for lo in (12, 50):
        print(f"-- both filaments of at least {lo} segments")
        row("all pairs", same, ~same, lo)
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                diff = ((ti == i) & (tj == j)) | ((ti == j) & (tj == i))
                within = same & ((ti == i) | (ti == j))
                row(f"{sets[i][:18]} vs {sets[j][:18]}", within, diff, lo)


if __name__ == "__main__":
    main()
