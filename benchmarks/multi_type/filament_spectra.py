"""Fingerprints of each filament from its own segment images.

Class labels lose what tells two types apart when Class2D puts both into one class.
The segment images keep it. Each segment is centred and turned by its Class2D shift
and in-plane angle so that the filament lies along x, and only a strip about the
filament's width is kept (``--width``, by default the tube diameter of the Class2D
job). Summed over the segments of a filament, which see it from many angles around
its axis, every fingerprint is an average over those angles:

* the density profile across the filament (the segment projected along its axis),
  its mean and its spread over the segments, with the CTF corrected (Wiener filter
  of each segment's own defocus) and folded about the axis;
* the power spectrum of the strip, which needs no centring along the filament (a
  shift changes only the phases). Each ring is divided by its own mean, which
  removes the CTF and the noise level (both nearly the same all round a ring), and
  the four quadrants are folded together (a 2D view cannot show the hand). The
  equator (the columns next to kx = 0) and the whole quadrant are scored apart.

Filaments are compared by the correlation of their fingerprints, scored against the
reference sets (AUC of telling pairs of one set from pairs of two, for each pair of
sets), and clustered, next to the Types card's first stage on the same filaments. A
classifier trained on the sets shows how much each fingerprint carries at best. The
segment images are read from the stacks of the RELION project the star file is in.

    python filament_spectra.py --star run_it025_data.star --reference reference.parquet \
        --mrcs run_it025_classes.mrcs --similarity similarity.npy
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd
import starfile
from scipy import ndimage as ndi
from scipy.cluster.hierarchy import fcluster, linkage

from compare_stage1 import balanced
from filament_alignment import auc
from two_stage import ari

SKIP = (
    "rlnMicrographName",
    "rlnHelicalTubeID",
    "rlnHelicalTrackLengthAngst",
    "rlnClassNumber",
)
OUT = Path(os.environ.get("TYPES_OUT", "types_out"))


def strip_mask(n, width, length, edge):
    """A strip along x, ``width`` pixels across and ``length`` long, with cosine edges."""
    c = n / 2
    y, x = np.mgrid[:n, :n] - c

    def soft(d, half):
        m = np.clip((half + edge - np.abs(d)) / edge, 0, 1)
        return 0.5 - 0.5 * np.cos(np.pi * m)

    return soft(y, width / 2) * soft(x, length / 2)


def straighten(img, angle, shift):
    """``img`` turned by ``angle`` degrees about the point ``shift`` (row, column
    pixels) away from its centre, which lands on the centre."""
    c = img.shape[0] / 2
    t = np.deg2rad(angle)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return ndi.affine_transform(
        img, rot, offset=np.array([c, c]) + shift - rot @ [c, c], order=1
    )


def power(a):
    """The power spectrum, zero frequency at (n//2, n//2)."""
    return np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2


def ctf_1d(n, apix, defocus, volt_kv, cs_mm, amp):
    """The CTF along one axis of an ``n``-pixel image (numpy FFT order)."""
    v = volt_kv * 1e3
    lam = 12.2643247 / np.sqrt(v * (1 + 0.978466e-6 * v))
    k2 = np.fft.fftfreq(n, apix) ** 2
    chi = np.pi * lam * defocus * k2 - 0.5 * np.pi * cs_mm * 1e7 * lam**3 * k2**2
    return -(np.sqrt(1 - amp**2) * np.sin(chi) + amp * np.cos(chi))


def ring_folded(ps):
    """The spectrum divided by its ring means, folded into one quadrant (row: the
    frequency across the filament)."""
    n = ps.shape[0]
    c = n // 2
    y, x = np.mgrid[:n, :n] - c
    r = np.rint(np.hypot(x, y)).astype(int)
    ring = np.bincount(r.ravel(), weights=ps.ravel()) / np.maximum(
        np.bincount(r.ravel()), 1
    )
    rel = ps / np.maximum(ring[r], 1e-30) - 1.0
    h = n - c
    return (
        rel[c:, c:]
        + rel[c:0:-1, c:][:h, :h]
        + rel[c:, c:0:-1][:h, :h]
        + rel[c:0:-1, c:0:-1][:h, :h]
    ) / 4.0


def fold(profile, half):
    """A profile across the filament folded about its axis, out to ``half`` pixels."""
    c = len(profile) // 2
    return 0.5 * (profile[c : c + half + 1] + profile[c - np.arange(half + 1)])


def trained(F, truth, i, j, folds=5, dims=20, seed=0):
    """Balanced accuracy of a nearest-centroid classifier (on the first principal
    components) telling set i from set j, cross-validated. Not a method, since it
    learns from the sets: it shows how much the fingerprint carries."""
    keep = np.flatnonzero((truth == i) | (truth == j))
    if min((truth == i).sum(), (truth == j).sum()) < 2 * folds:
        return float("nan")
    y = truth[keep] == j
    fold_of = np.random.default_rng(seed).permutation(len(keep)) % folds
    right = np.zeros(2)
    for f in range(folds):
        train, test = keep[fold_of != f], keep[fold_of == f]
        mean = F[train].mean(0)
        _, _, vt = np.linalg.svd(F[train] - mean, full_matrices=False)
        P = vt[:dims].T
        a = (F[train][~y[fold_of != f]] - mean) @ P
        b = (F[train][y[fold_of != f]] - mean) @ P
        x = (F[test] - mean) @ P
        sd = np.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-12
        da = (((x - a.mean(0)) / sd) ** 2).sum(1)
        db = (((x - b.mean(0)) / sd) ** 2).sum(1)
        yt = y[fold_of == f]
        right += [(da < db)[~yt].sum(), (db <= da)[yt].sum()]
    return float(0.5 * (right[0] / (~y).sum() + right[1] / y.sum()))


def segment_images(rows, names, root):
    """Yield (row, image) for the given rows of the star file, reading each stack once."""
    by_stack = pd.Series(rows).groupby([names[k].split("@")[1] for k in rows])
    for path, ks in by_stack:
        with mrcfile.mmap(str(root / path), permissive=True) as m:
            data = m.data if m.data.ndim == 3 else m.data[None]
            for k in ks.to_numpy():
                img = np.asarray(data[int(names[k].split("@")[0]) - 1], np.float32)
                yield k, img - img.mean()


def tube_diameter(star_path):
    """The helical tube diameter (Å) of the job that wrote ``star_path``, if recorded."""
    job = Path(star_path).parent / "job.star"
    if not job.exists():
        return None
    blocks = starfile.read(job)
    options = (
        blocks if isinstance(blocks, pd.DataFrame) else blocks.get("joboptions_values")
    )
    if options is None:
        return None
    hit = options[options["rlnJobOptionVariable"] == "helical_tube_outer_diameter"]
    return float(hit["rlnJobOptionValue"].iloc[0]) if len(hit) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--star", required=True, help="the Class2D _data.star")
    p.add_argument("--reference", required=True)
    p.add_argument("--mrcs", required=True)
    p.add_argument("--similarity", required=True)
    p.add_argument("--per-set", type=int, default=400)
    p.add_argument("--min-segments", type=int, default=20)
    p.add_argument("--width", type=float, help="Å, the strip kept across the filament")
    p.add_argument("--low", type=float, default=200.0, help="Å, lowest resolution used")
    p.add_argument(
        "--high", type=float, default=20.0, help="Å, highest resolution used"
    )
    a = p.parse_args()

    from helicon.lib.io import get_relion_project_folder
    from helicon.webApps.lib import helical_pitch_types as ty

    star = starfile.read(a.star)
    optics, star = star["optics"], star["particles"]
    apix = float(optics["rlnImagePixelSize"].iloc[0])
    n = int(optics["rlnImageSize"].iloc[0])
    volt = float(optics["rlnVoltage"].iloc[0])
    cs = float(optics["rlnSphericalAberration"].iloc[0])
    amp = float(optics["rlnAmplitudeContrast"].iloc[0])
    root = Path(get_relion_project_folder(a.star))
    width = a.width or tube_diameter(a.star) or 250.0
    ref = pd.read_parquet(a.reference)
    assert (
        len(ref) == len(star)
        and (
            ref["rlnClassNumber"].to_numpy() == star["rlnClassNumber"].to_numpy()
        ).all()
    ), "the reference is not in the order of the star file"
    sets = [c for c in ref.columns if c not in SKIP]

    key = (
        ref["rlnMicrographName"].astype(str) + "|" + ref["rlnHelicalTubeID"].astype(str)
    ).astype("category")
    fil = key.cat.codes.to_numpy()
    n_fil = len(key.cat.categories)
    per_fil = np.zeros((n_fil, len(sets)), dtype=np.int64)
    np.add.at(per_fil, fil, ref[sets].to_numpy().astype(np.int64))
    n_seg = np.bincount(fil, minlength=n_fil)
    single = ((per_fil > 0).sum(1) == 1) & (n_seg >= a.min_segments)
    which = np.where(single, per_fil.argmax(1), -1)
    rng = np.random.default_rng(0)
    pick = np.sort(
        np.concatenate(
            [
                rng.choice(
                    np.flatnonzero(which == i),
                    min(a.per_set, int((which == i).sum())),
                    replace=False,
                )
                for i in range(len(sets))
            ]
        )
    )
    slot_of = np.full(n_fil, -1)
    slot_of[pick] = np.arange(len(pick))
    rows = np.flatnonzero(slot_of[fil] >= 0)
    print(
        f"{len(pick)} filaments of one set with {a.min_segments}+ segments "
        f"(up to {a.per_set} a set), {len(rows):,} segments; {n} px of {apix} Å; "
        f"a strip {width:.0f} Å wide"
    )
    names = star["rlnImageName"].to_numpy()
    psi = star["rlnAnglePsi"].to_numpy(float)
    origin = star[["rlnOriginYAngst", "rlnOriginXAngst"]].to_numpy(float) / apix
    defocus = 0.5 * (star["rlnDefocusU"] + star["rlnDefocusV"]).to_numpy(float)
    w = width / apix
    mask = strip_mask(n, w, 0.7 * n, 0.06 * n)
    along = strip_mask(n, 4 * n, 0.7 * n, 0.06 * n)  # the length window only
    c, half = n // 2, int(0.75 * w)

    # the sense of the turn and of the shift: the one that gathers the density of
    # the first filaments into the strip
    first = rows[np.isin(fil[rows], pick[:40])]
    trials = list(itertools.product((1.0, -1.0), repeat=3))
    profiles = np.zeros((len(trials), 40, n))
    for k, img in segment_images(first, names, root):
        for t, (sr, sy, sx) in enumerate(trials):
            s = straighten(img, sr * psi[k], origin[k] * [sy, sx])
            profiles[t, slot_of[fil[k]]] += (s * along).sum(1)
    inside = np.abs(np.arange(n) - c) <= w / 2
    gathered = [(x[:, inside] ** 2).sum() / (x**2).sum() for x in profiles]
    sr, sy, sx = trials[int(np.argmax(gathered))]
    print(
        "share of the profile within the strip: "
        + ", ".join(
            f"{'+-'[t[0] < 0]}psi {'+-'[t[1] < 0]}y {'+-'[t[2] < 0]}x {g:.3f}"
            for t, g in zip(trials, gathered)
        )
    )

    t0 = time.time()
    spectra = np.zeros((len(pick), n, n))
    mean_profile = np.zeros((len(pick), half + 1))
    profile_sq = np.zeros((len(pick), half + 1))
    for i, (k, img) in enumerate(segment_images(rows, names, root)):
        f = slot_of[fil[k]]
        s = straighten(img, sr * psi[k], origin[k] * [sy, sx])
        spectra[f] += power(s * mask)
        ctf = ctf_1d(n, apix, defocus[k], volt, cs, amp)
        prof = np.fft.ifft(np.fft.fft((s * along).sum(1)) * ctf / (ctf**2 + 0.1)).real
        prof = fold(prof, half)
        mean_profile[f] += prof
        profile_sq[f] += prof**2
        if i % 20000 == 0:
            print(f"  {i:,} segments, {time.time() - t0:.0f} s", flush=True)
    print(f"read and transformed in {time.time() - t0:.0f} s")

    truth, length = which[pick], n_seg[pick]
    mean_profile /= length[:, None]
    spread = np.sqrt(np.maximum(profile_sq / length[:, None] - mean_profile**2, 0))
    quads = np.array([ring_folded(s) for s in spectra])
    h = n - c
    rr = np.hypot(*np.mgrid[:h, :h])
    band = (rr >= n * apix / a.low) & (rr <= n * apix / a.high)
    equator = band & (np.arange(h)[None, :] < 3)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "filament_spectra.npz",
        quads=quads.astype(np.float32),
        mean_profile=mean_profile,
        spread=spread,
        truth=truth,
        length=length,
        sets=np.array(sets),
        apix=apix,
        n=n,
    )

    def unit(F):
        F = F - F.mean(0)
        return F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)

    fingerprints = {
        "profile": unit(mean_profile),
        "profile and spread": np.hstack([unit(mean_profile), unit(spread)]) / 2**0.5,
        "equator (3 columns)": unit(quads[:, equator]),
        "whole quadrant": unit(quads[:, band]),
    }
    pairs = [(i, j) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    iu = np.triu_indices(len(pick), 1)
    ti, tj = truth[iu[0]], truth[iu[1]]
    same = ti == tj
    shorter = np.minimum(length[iu[0]], length[iu[1]])
    for name, F in fingerprints.items():
        cc = (F @ F.T)[iu]
        print(f"\n{name}: AUC of telling pairs of one set from pairs of two")
        for least in (20, 50, 100):
            m = shorter >= least
            print(
                f"  both of {least}+ segments: {auc(cc[m & same], cc[m & ~same]):.3f}"
            )
        print("  for each pair of sets (both of 20+ / 50+ segments), and how well a")
        print("  classifier trained on the sets tells them apart (cross-validated):")
        for i, j in pairs:
            two = ((ti == i) & (tj == j)) | ((ti == j) & (tj == i))
            one = same & ((ti == i) | (ti == j))
            v = [
                auc(cc[(shorter >= least) & one], cc[(shorter >= least) & two])
                for least in (20, 50)
            ]
            print(
                f"  {sets[i][:22]:>22s} vs {sets[j][:22]:22s} {v[0]:.3f} / {v[1]:.3f}"
                f"   {trained(F, truth, i, j):.3f}"
            )

    # clusters of the fingerprints, and the Types card's first stage
    cls = ref["rlnClassNumber"].to_numpy().astype(int) - 1
    n_classes = int(cls.max()) + 1
    with mrcfile.open(a.mrcs, permissive=True) as f:
        has_signal = [x.max() > x.min() for x in f.data]
    total = np.bincount(cls, minlength=n_classes).astype(float)
    floor = max(100, int(0.002 * total.sum()))
    classes = [k for k in range(n_classes) if total[k] >= floor and has_signal[k]]
    S = np.load(a.similarity)
    assert S.shape == (len(classes),) * 2, "the similarity does not match the classes"
    print(
        "\nadjusted Rand index with the sets; balanced accuracy of telling each "
        "pair of sets apart (in the order above)"
    )
    for k in (5, 6, 8):
        cg, _ = ty.pixel_groups_n(S, k)
        group_of = np.full(n_classes, -1)
        group_of[classes] = cg
        g = group_of[cls]
        votes = np.zeros((n_fil, int(cg.max()) + 1))
        np.add.at(votes, (fil[g >= 0], g[g >= 0]), 1)
        found = {"card": votes[pick].argmax(1)}
        for name, F in fingerprints.items():
            found[name] = fcluster(linkage(F, "ward"), k, "maxclust") - 1
        for name, groups in found.items():
            acc = " ".join(f"{balanced(groups, truth, i, j):.2f}" for i, j in pairs)
            print(f"  {k} groups, {name:20s} ARI {ari(groups, truth):.3f}   {acc}")


if __name__ == "__main__":
    main()
