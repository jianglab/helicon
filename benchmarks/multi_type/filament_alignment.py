"""Filaments as sequences of class labels: does an ungapped alignment tell the types apart?

Each filament is read as the sequence of the classes of its segments, in order along
its length. Two filaments of one type walk the same cyclic path (class after class as
the azimuth advances), begun at different points, so they match at equal distances
along their lengths, in one direction or the other. Filaments of different pitch drift
apart and match badly. The alignment therefore has no gaps (a gap would stretch one
filament to fit another and hide a pitch difference): the score of a pair is the best
ungapped run over all shifts and both directions (BLAST's ungapped extension), with a
substitution matrix made from the registered correlation of the class averages.

It is scored against the reference sets (filaments that belong to exactly one): the
area under the ROC curve of telling pairs of one set from pairs of two, for every
pair of sets, in bins of filament length, next to two baselines -- the cosine
similarity of the filaments' class usage, and of their class bigrams (which keep the
order but not the alignment).

    python filament_alignment.py --reference reference.parquet \
        --mrcs run_it025_classes.mrcs --similarity similarity.npy
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mrcfile
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp

OUT = Path(os.environ.get("TYPES_OUT", "types_out"))
STEP_DEFAULT = 14.25


@nb.njit(cache=True)
def _best_ungapped(a, b, M):
    """Best ungapped local score of sequences ``a`` and ``b`` (letters < 0 score 0),
    over every shift, reading ``b`` forwards and backwards."""
    la, lb = len(a), len(b)
    best = 0.0
    for rev in range(2):
        for d in range(-(lb - 1), la):
            run = 0.0
            i0 = max(0, d)
            i1 = min(la, lb + d)
            for i in range(i0, i1):
                j = i - d
                if rev:
                    j = lb - 1 - j
                x, y = a[i], b[j]
                s = 0.0 if (x < 0 or y < 0) else M[x, y]
                run = run + s
                if run < 0.0:
                    run = 0.0
                elif run > best:
                    best = run
    return best


@nb.njit(parallel=True, cache=True)
def _all_pairs(seq, off, M):
    n = len(off) - 1
    out = np.zeros((n, n), dtype=np.float32)
    for i in nb.prange(n):
        a = seq[off[i] : off[i + 1]]
        for j in range(i + 1, n):
            b = seq[off[j] : off[j + 1]]
            out[i, j] = _best_ungapped(a, b, M)
    return out


def auc(pos, neg, cap=400_000, seed=0):
    """Area under the ROC curve of ``pos`` against ``neg`` scores (rank-sum)."""
    rng = np.random.default_rng(seed)
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) < 20 or len(neg) < 20:
        return float("nan")
    if len(pos) > cap:
        pos = rng.choice(pos, cap, replace=False)
    if len(neg) > cap:
        neg = rng.choice(neg, cap, replace=False)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    xs = allv[order]
    edges = np.flatnonzero(np.diff(xs) != 0) + 1
    for lo, hi in zip(np.r_[0, edges], np.r_[edges, len(allv)]):
        if hi - lo > 1:
            ranks[order[lo:hi]] = 0.5 * (lo + 1 + hi)
    n1, n0 = len(pos), len(neg)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def add_arguments(p):
    p.add_argument("--reference", required=True)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--similarity", required=True)
    p.add_argument(
        "--per-set", type=int, default=300, help="filaments sampled from each set"
    )
    p.add_argument("--min-segments", type=int, default=12)
    p.add_argument(
        "--margin",
        type=float,
        default=0.03,
        help="added to the mean similarity of unrelated letters",
    )


def prepare(a):
    """Sample filaments of the reference sets and align them all.

    Returns
    -------
    dict
        ``sets``, ``seqs`` (the class letters of each sampled filament along it),
        ``truth`` (its set), ``length`` (segments), ``M`` (the substitution matrix),
        ``A`` (the symmetric matrix of alignment scores) and ``step``.
    """
    ref = pd.read_parquet(a.reference)
    sets = [
        c
        for c in ref.columns
        if c
        not in (
            "rlnMicrographName",
            "rlnHelicalTubeID",
            "rlnHelicalTrackLengthAngst",
            "rlnClassNumber",
        )
    ]
    cls = ref["rlnClassNumber"].to_numpy().astype(int) - 1
    n_classes = int(cls.max()) + 1
    with mrcfile.open(a.mrcs, permissive=True) as f:
        has_signal = [x.max() > x.min() for x in f.data]
    total = np.bincount(cls, minlength=n_classes).astype(float)
    floor = max(100, int(0.002 * total.sum()))
    classes = [c for c in range(n_classes) if total[c] >= floor and has_signal[c]]
    S = np.load(a.similarity)
    assert S.shape == (
        len(classes),
        len(classes),
    ), "the similarity does not match the classes"
    letter = np.full(n_classes, -1, dtype=np.int32)
    letter[classes] = np.arange(len(classes))

    # filaments in order along their length, each with the one set it belongs to
    key = (
        ref["rlnMicrographName"].astype(str) + "|" + ref["rlnHelicalTubeID"].astype(str)
    ).astype("category")
    code = key.cat.codes.to_numpy()
    order = np.lexsort((ref["rlnHelicalTrackLengthAngst"].to_numpy(), code))
    keys_all = key.cat.categories
    code_o, let_o = code[order], letter[cls[order]]
    starts = np.r_[0, np.flatnonzero(np.diff(code_o)) + 1]
    lens = np.diff(np.r_[starts, len(code_o)])
    track_o = ref["rlnHelicalTrackLengthAngst"].to_numpy()[order]
    gaps = np.diff(track_o)[np.diff(code_o) == 0]
    step = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else STEP_DEFAULT
    flags = ref[sets].to_numpy()[order]
    in_set = np.add.reduceat(flags.astype(np.int32), starts, axis=0) > 0
    single = (in_set.sum(1) == 1) & (lens >= a.min_segments)
    which = np.where(single, in_set.argmax(1), -1)
    print(
        "filaments of exactly one set, with at least",
        a.min_segments,
        "segments:",
        {s: int((which == i).sum()) for i, s in enumerate(sets)},
    )

    rng = np.random.default_rng(0)
    pick = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(which == i),
                min(a.per_set, int((which == i).sum())),
                replace=False,
            )
            for i in range(len(sets))
            if (which == i).any()
        ]
    )
    pick.sort()
    seqs = [let_o[starts[i] : starts[i] + lens[i]] for i in pick]
    truth = which[pick]
    length = lens[pick]
    n = len(pick)
    print(f"{n} filaments sampled; median length {int(np.median(length))} segments")

    # substitution matrix: unrelated letters score below zero on average
    flat = np.concatenate(seqs)
    freq = np.bincount(flat[flat >= 0], minlength=len(classes)).astype(float)
    freq /= freq.sum()
    t = float(freq @ S @ freq) + a.margin
    M = ((S - t) / (1.0 - t)).astype(np.float32)
    print(
        f"substitution matrix: threshold {t:.3f}; identical letters +1, range {M.min():.2f} to {M.max():.2f}"
    )

    off = np.r_[0, np.cumsum([len(s) for s in seqs])].astype(np.int64)
    seq = np.concatenate(seqs).astype(np.int32)
    t0 = time.time()
    A = _all_pairs(seq, off, M)
    print(f"{n * (n - 1) // 2:,} pairs aligned in {time.time() - t0:.0f} s")
    A = A + A.T

    return dict(
        sets=sets,
        seqs=seqs,
        truth=truth,
        length=length,
        M=M,
        A=A,
        n_letters=len(classes),
        step=step,
        keys=[str(keys_all[code_o[starts[i]]]) for i in pick],
        similarity=S,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_arguments(p)
    a = p.parse_args()
    d = prepare(a)
    sets, seqs, truth, length, A = d["sets"], d["seqs"], d["truth"], d["length"], d["A"]
    n = len(seqs)
    # baselines: class usage and class bigrams, as cosine similarities
    K = d["n_letters"]
    use = sp.lil_matrix((n, K))
    big = sp.lil_matrix((n, K * K))
    for i, s in enumerate(seqs):
        ok = s >= 0
        for x in s[ok]:
            use[i, x] += 1
        for x, y in zip(s[:-1], s[1:]):
            if x >= 0 and y >= 0 and x != y:
                big[i, x * K + y] += 1
                big[i, y * K + x] += 1  # either direction

    def cosine(X):
        X = sp.csr_matrix(X)
        nrm = np.sqrt(X.multiply(X).sum(1)).A1
        nrm[nrm == 0] = 1
        return (sp.diags(1 / nrm) @ X @ X.T @ sp.diags(1 / nrm)).toarray()

    U, B = cosine(use), cosine(big)

    iu = np.triu_indices(n, 1)
    same = truth[iu[0]] == truth[iu[1]]
    minlen = np.minimum(length[iu[0]], length[iu[1]])
    methods = {
        "ungapped alignment": A[iu],
        "class usage": U[iu],
        "class bigrams": B[iu],
    }

    print(
        "\nAUC of telling filament pairs of one set from pairs of two sets (0.5 = chance)"
    )
    print(f"{'':28s}" + "".join(f"{m:>22s}" for m in methods))
    print(
        f"{'all pairs':28s}"
        + "".join(f"{auc(v[same], v[~same]):22.3f}" for v in methods.values())
    )
    for lo, hi in ((12, 25), (25, 50), (50, 100), (100, 10**6)):
        m = (minlen >= lo) & (minlen < hi)
        print(
            f"{'both >= %d segments' % lo + ('' if hi > 10**5 else ' (< %d)' % hi):28s}"
            + "".join(
                f"{auc(v[m & same], v[m & ~same]):22.3f}" for v in methods.values()
            )
        )
    # the score grows with the length of the filaments: the same, per letter of the shorter
    norm = A[iu] / np.minimum(length[iu[0]], length[iu[1]])
    print("\nthe alignment score divided by the shorter length:")
    print("  all pairs: %.3f" % auc(norm[same], norm[~same]))

    print(
        "\nby pair of sets, for pairs of filaments of at least 12 / 50 / 100 segments"
    )
    print("  (alignment | alignment per letter | class usage)")
    ti, tj = truth[iu[0]], truth[iu[1]]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            diff = ((ti == i) & (tj == j)) | ((ti == j) & (tj == i))
            within = same & ((ti == i) | (ti == j))
            cells = []
            for lo in (12, 50, 100):
                m = minlen >= lo
                cells.append(
                    "%d: %.2f | %.2f | %.2f (n=%d,%d)"
                    % (
                        lo,
                        auc(A[iu][m & within], A[iu][m & diff]),
                        auc(norm[m & within], norm[m & diff]),
                        auc(U[iu][m & within], U[iu][m & diff]),
                        int((m & within).sum()),
                        int((m & diff).sum()),
                    )
                )
            print(f"  {sets[i]} vs {sets[j]}:\n      " + "\n      ".join(cells))


if __name__ == "__main__":
    main()
