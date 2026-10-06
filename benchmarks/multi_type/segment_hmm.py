"""Types along the filaments: classes that look alike at some views, told apart in context.

Most classes of two types look different, but at some views (a crossover seen
edge-on) the averages of the two are alike, and no class-level grouping can tell
them apart. A filament, however, carries one type over a stretch, so the wide
views next to a narrow one say what it is. Here each filament is a chain of
segments; its type is a hidden state that stays the same from one segment to the
next with probability ``1 - switch``; each type has a probability of showing each
class. Starting from the classes' groups by their averages, the class
probabilities and the switch rate are re-estimated from the data (EM,
forward-backward), with no labels. The result is scored against a reference.

    python segment_hmm.py --reference reference.parquet --mrcs run_it025_classes.mrcs \
        --set-a typeA --set-b typeB
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from helicon.webApps.lib import helical_pitch_types as ty

OUT = Path(os.environ.get("TYPES_OUT", "types_out"))


def chains(mic, tube, track, cls, n_classes):
    """Segments as padded chains: ``(n_filaments, L)`` class numbers (-1: padding)."""
    key = pd.Series(mic).astype(str) + "|" + pd.Series(tube).astype(str)
    code = key.astype("category").cat.codes.to_numpy()
    order = np.lexsort((track, code))
    code, cls_o = code[order], cls[order]
    starts = np.r_[0, np.flatnonzero(np.diff(code)) + 1]
    lens = np.diff(np.r_[starts, len(code)])
    L = int(lens.max())
    X = np.full((len(starts), L), -1, dtype=np.int32)
    pos = np.arange(len(code)) - np.repeat(starts, lens)
    X[np.repeat(np.arange(len(starts)), lens), pos] = cls_o
    return X, order, np.repeat(np.arange(len(starts)), lens), pos


def forward_backward(X, B, switch, prior):
    """Posterior type of every segment of every chain.

    Parameters
    ----------
    X : numpy.ndarray
        ``(n, L)`` classes (index into the rows of ``B``), -1 for padding.
    B : numpy.ndarray
        ``(K, T)`` probability of each class given the type.
    switch : float
        Probability that the type changes from one segment to the next.
    prior : numpy.ndarray
        ``(T,)`` prior of the types at the start of a chain.

    Returns
    -------
    gamma : numpy.ndarray
        ``(n, L, T)`` posterior; ``xi_sum`` the expected number of switches and
        ``steps`` the number of transitions, for re-estimating ``switch``.
    """
    n, L = X.shape
    T = B.shape[1]
    emit = np.where(X[..., None] >= 0, B[np.maximum(X, 0)], 1.0)  # (n, L, T)
    stay = 1.0 - switch
    move = switch / max(T - 1, 1)
    A = np.full((T, T), move) + np.eye(T) * (stay - move)
    alpha = np.zeros((n, L, T))
    scale = np.ones((n, L))
    a = prior[None, :] * emit[:, 0]
    scale[:, 0] = a.sum(1)
    alpha[:, 0] = a / scale[:, 0, None]
    for t in range(1, L):
        live = (X[:, t] >= 0)[:, None]
        a = (alpha[:, t - 1] @ A) * emit[:, t]
        s = a.sum(1)
        s = np.where(s > 0, s, 1.0)
        a = a / s[:, None]
        alpha[:, t] = np.where(live, a, alpha[:, t - 1])
        scale[:, t] = np.where(live[:, 0], s, 1.0)
    beta = np.ones((n, L, T))
    for t in range(L - 2, -1, -1):
        live = (X[:, t + 1] >= 0)[:, None]
        b = (beta[:, t + 1] * emit[:, t + 1]) @ A.T / scale[:, t + 1, None]
        beta[:, t] = np.where(live, b, 1.0)
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(2, keepdims=True), 1e-300)
    # expected number of switches
    xi = 0.0
    steps = 0.0
    for t in range(1, L):
        live = X[:, t] >= 0
        if not live.any():
            continue
        num = (
            alpha[live, t - 1][:, :, None]
            * A[None]
            * (emit[live, t] * beta[live, t])[:, None, :]
        )
        num /= np.maximum(num.sum((1, 2), keepdims=True), 1e-300)
        xi += float(num.sum() - np.einsum("nii->", num))
        steps += float(live.sum())
    return gamma, xi, steps


def fit(X, B0, switch=0.05, n_iter=30, tol=1e-4, freeze_switch=False, share=None):
    """EM for the class probabilities of each type and the switch rate.

    ``share`` ``(K, T)``: for the classes in which the types are known from outside
    (rows that sum to 1; NaN for a free class), the share of the class's segments
    in each type is held at this, and only the free classes learn their type from
    the chains.
    """
    B = B0.copy()
    prior = np.full(B.shape[1], 1.0 / B.shape[1])
    for it in range(n_iter):
        gamma, xi, steps = forward_backward(X, B, switch, prior)
        live = X >= 0
        counts = np.zeros_like(B)
        for t_ in range(B.shape[1]):
            counts[:, t_] = np.bincount(
                X[live], weights=gamma[..., t_][live], minlength=B.shape[0]
            )
        if share is not None:
            held = np.isfinite(share[:, 0])
            counts[held] = counts[held].sum(1, keepdims=True) * share[held]
        new = (counts + 1e-3) / (counts + 1e-3).sum(0, keepdims=True)
        new_switch = (
            switch if freeze_switch else float(np.clip(xi / max(steps, 1.0), 1e-4, 0.5))
        )
        moved = np.abs(new - B).max()
        B, switch = new, new_switch
        prior = gamma[:, 0].mean(0)
        if moved < tol:
            break
    return B, switch, gamma, it + 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--set-a", required=True)
    p.add_argument("--set-b", required=True)
    p.add_argument("--cut", type=float, default=0.15)
    p.add_argument("--init", default="spectral", choices=["spectral", "tree"])
    p.add_argument("--switch", type=float, default=0.05)
    p.add_argument(
        "--clamp",
        type=float,
        default=0.7,
        help="share of the classes held at their pixel type",
    )
    p.add_argument("--freeze-switch", action="store_true")
    a = p.parse_args()

    ref = pd.read_parquet(a.reference)
    cls = ref["rlnClassNumber"].to_numpy().astype(int) - 1
    n = int(cls.max()) + 1
    with mrcfile.open(a.mrcs, permissive=True) as f:
        ok_img = [x.max() > x.min() for x in f.data]
    total = np.bincount(cls, minlength=n).astype(float)
    floor = max(100, int(0.002 * total.sum()))
    classes = [c for c in range(n) if total[c] >= floor and ok_img[c]]
    S = np.load(OUT / "similarity.npy")
    groups = ty.pixel_groups(S, a.cut)
    cnt = {
        s: np.bincount(cls, weights=ref[s].to_numpy(float), minlength=n)[classes]
        for s in (a.set_a, a.set_b)
    }
    g = int(
        np.argmax(
            [
                (cnt[a.set_a] + cnt[a.set_b])[groups == k].sum()
                for k in range(groups.max() + 1)
            ]
        )
    )
    members = [classes[i] for i in np.flatnonzero(groups == g)]
    print(f"group {g} at cut {a.cut}: {len(members)} classes")

    # two starting types from the tree of this group's classes (no labels)
    idx = np.flatnonzero(groups == g)
    if a.init == "spectral":
        two = np.asarray(ty.spectral_groups(S[np.ix_(idx, idx)], 2))
    else:
        D = 1 - S[np.ix_(idx, idx)]
        np.fill_diagonal(D, 0)
        link = linkage(squareform(D, checks=False), method="average")
        two = fcluster(link, 2, criterion="maxclust") - 1
    K = len(members)
    row = {c: i for i, c in enumerate(members)}
    B0 = np.where(np.eye(2)[two] > 0, 0.85, 0.15)
    B0 = B0 / B0.sum(0, keepdims=True)
    print("starting split of the classes:", np.bincount(two).tolist())

    inside = np.isin(cls, members)
    sub = ref[inside]
    X, order, fil_of, pos = chains(
        sub["rlnMicrographName"].to_numpy(),
        sub["rlnHelicalTubeID"].to_numpy(),
        sub["rlnHelicalTrackLengthAngst"].to_numpy(),
        np.array([row[c] for c in cls[inside]]),
        K,
    )
    print(f"{len(sub):,} segments on {X.shape[0]:,} filaments (longest {X.shape[1]})")
    # the classes the pixels make clear: silhouette of each class in its group
    Sg = S[np.ix_(idx, idx)]
    own = np.array([Sg[i, two == two[i]].sum() - 1.0 for i in range(K)]) / np.maximum(
        np.bincount(two)[two] - 1, 1
    )
    other = np.array([Sg[i, two != two[i]].mean() for i in range(K)])
    sil = (own - other) / np.maximum(np.maximum(own, other), 1e-12)
    share = None
    if a.clamp > 0:
        keep = sil >= np.quantile(sil, 1.0 - a.clamp)
        share = np.full((K, 2), np.nan)
        share[keep] = np.where(np.eye(2)[two[keep]] > 0, 0.97, 0.03)
        print(
            f"{int(keep.sum())} of {K} classes held at their pixel type (silhouette >= {np.quantile(sil, 1.0 - a.clamp):.3f})"
        )
    B, switch, gamma, iters = fit(
        X, B0, a.switch, freeze_switch=a.freeze_switch, share=share
    )
    print(f"EM: {iters} rounds, switch probability {switch:.3f}")

    y_a = sub[a.set_a].to_numpy()[order]
    y_b = sub[a.set_b].to_numpy()[order]
    labelled = y_a ^ y_b
    truth = np.where(y_b, 1, 0)[labelled]
    post = gamma[fil_of, pos]  # (n_segments_ordered, T)
    # the best any rule on the class alone could do: each class to the set most of its
    # labelled segments are in (uses the labels, an upper bound for the class-only rows)
    cl_seg = X[fil_of, pos]
    ca = np.bincount(cl_seg[labelled], weights=(1 - truth), minlength=K)
    cb = np.bincount(cl_seg[labelled], weights=truth, minlength=K)
    oracle = np.maximum(ca, cb).sum() / max(ca.sum() + cb.sum(), 1)
    print(f"{'oracle class-only (labels)':30s}: {oracle:.3f}")
    for name, pred in (
        ("class only (starting split)", two[X[fil_of, pos]]),
        ("class only (re-estimated)", np.argmax(B[X[fil_of, pos]], axis=1)),
        ("along the filament", np.argmax(post, axis=1)),
    ):
        pred = pred[labelled]
        acc = max(np.mean(pred == truth), np.mean(pred != truth))
        sa = (
            np.mean((pred == truth)[truth == 0])
            if np.mean(pred == truth) >= 0.5
            else np.mean((pred != truth)[truth == 0])
        )
        print(
            f"{name:30s}: {acc:.3f} of {int(labelled.sum()):,} labelled segments ({a.set_a}/{a.set_b})"
        )
    kind = np.argmax(post, axis=1)
    flip = np.mean(kind[labelled] == truth) < 0.5
    ptype = B / B.sum(1, keepdims=True)  # P(type | class) with equal type priors
    ptype = ptype[:, 0] if flip else ptype[:, 1]
    # where the remaining disagreement with the reference is, class by class
    t_full = np.where(y_b, 1, 0)
    miss = labelled & ((kind != t_full) ^ flip)
    byc = np.bincount(cl_seg[miss], minlength=K)
    print(
        "  classes with most segments of the wrong type (class, wrong, labelled A, labelled B, P(B-like)):"
    )
    for k in np.argsort(-byc)[:12]:
        m = labelled & (cl_seg == k)
        print(
            f"    {members[k] + 1:4d} {int(byc[k]):6d}  {int((m & ~y_b).sum()):6d} {int((m & y_b).sum()):6d}  {ptype[k]:.2f}"
        )


if __name__ == "__main__":
    main()
