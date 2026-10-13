"""Helical types from the context of a segment: the classes along its filament.

The segments of one filament are not independent. Neighbours show the same
helical type, and within one type the class (the view of the helix, that is the
position within its crossover) changes in a regular way with the distance along
the filament. Where the Types card groups the classes by their averages and gives
each filament the group most of its segments are in, this reads the *sequence* of
classes along each filament.

The model is a hidden Markov chain over a regular grid of positions along each
filament (one step is the spacing of the segments), with a two-level, "sticky"
hidden state, the helical type ``t`` and the class ``c``:

* the type stays with probability ``1 - rho`` from one step to the next, otherwise
  it switches to another type;
* within type ``t`` the class follows ``A_t[c][c']``; after a switch to type ``t'``
  the class is drawn from ``q_t'``;
* the emission of a segment is the likelihood of the class it was assigned to by
  the 2D classification, from the probability of that class (a segment of a
  position where there is none carries no evidence).

``rho``, ``A_t`` and ``q_t`` are learnt by Baum-Welch (forward-backward). The types
start from a topic model (non-negative matrix factorisation) of the classes each
filament uses, and their number can be chosen by the likelihood of filaments held
out of the fit. A chain of the classes alone, with no type above it, cannot do
this: noisy and junk classes connect the types, and after a few hundred A the
class no longer tells the type.

This is a port of ``helix_filament_context`` of RELION (``src/helix_context.cpp``).
Here the forward-backward of all the filaments of a chunk is run together, one
matrix product per type and position over every filament that is long enough to
have that position (so that BLAS does the work, over all CPUs), in single
precision, and the chunks are sized from the free memory.

Segments overlap a lot (the box is much longer than the step), so neighbours share
most of their pixels and their evidence is not independent. ``temper`` multiplies
the log-likelihoods of the emissions to account for that if it is wanted.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

import numpy as np

from helicon.lib.numba_compat import lazy_jit

logger = logging.getLogger(__name__)

# the builtin range, when numba is not there (lazy_jit swaps in numba.prange)
prange = range

# the log-likelihood of a class other than the assigned one is at least this
# below that of the assigned class
LL_FLOOR = -6.9
# the probability of the assigned class is at most this
MAX_PROB = 0.99
# bounds of the type switch probability per step
RHO_MIN, RHO_MAX = 1e-6, 0.05
# a gap of more than this many steps breaks a filament into chains
MAX_GAP_STEPS = 3.0
# runs of fewer segments than this between two runs of one type are taken into them
MIN_RUN = 3


@dataclass
class Chains:
    """The segments laid out on regular grids along their filaments.

    Attributes
    ----------
    step : float
        The spacing of the segments along a filament, A (the median).
    n_classes : int
        Number of classes.
    filament : numpy.ndarray
        ``(N,)`` the filament (micrograph and tube) of each segment, from 0.
    filament_key : numpy.ndarray
        ``(F,)`` ``"micrograph|tube"`` of each filament.
    chain : numpy.ndarray
        ``(N,)`` the chain of each segment: a run of one filament without gaps of
        more than :data:`MAX_GAP_STEPS` steps. Numbered from 0 by length, the
        longest first.
    pos : numpy.ndarray
        ``(N,)`` its position on the grid of its chain, in steps.
    chain_len : numpy.ndarray
        ``(C,)`` grid length of each chain, steps (not increasing).
    chain_filament : numpy.ndarray
        ``(C,)`` the filament of each chain.
    cls : numpy.ndarray
        ``(N,)`` the class of each segment, 0-based (-1: none).
    eps : numpy.ndarray
        ``(N,)`` the likelihood of every other class against the assigned one
        (1: no evidence), tempered.
    """

    step: float
    n_classes: int
    filament: np.ndarray
    filament_key: np.ndarray
    chain: np.ndarray
    pos: np.ndarray
    chain_len: np.ndarray
    chain_filament: np.ndarray
    cls: np.ndarray
    eps: np.ndarray
    repeat: np.ndarray
    _chunks: dict = field(default_factory=dict, repr=False)

    @property
    def n_segments(self):
        return len(self.cls)

    @property
    def n_chains(self):
        return len(self.chain_len)

    @property
    def n_filaments(self):
        return len(self.filament_key)


def emission_floor(prob, n_classes, temper=1.0, max_prob=MAX_PROB):
    """The likelihood of each class but the assigned one, against the assigned one.

    A segment of class ``c`` with probability ``p`` has log-likelihood ``log p`` for
    ``c`` and ``log((1 - p) / (K - 1))`` for each other class; the difference is
    taken to the floor :data:`LL_FLOOR` and multiplied by ``temper``.

    Parameters
    ----------
    prob : numpy.ndarray
        The probability of the assigned class of each segment (from
        ``rlnMaxValueProbDistribution``).
    n_classes : int
    temper : float, optional
        Defaults to 1.
    max_prob : float, optional
        Probabilities are limited to this. Defaults to :data:`MAX_PROB`.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` in (0, 1].
    """
    p = np.clip(np.asarray(prob, dtype=float), 1.0 / n_classes, max_prob)
    rel = np.log((1.0 - p) / ((n_classes - 1) * p))
    return np.exp(temper * np.maximum(LL_FLOOR, rel)).astype(np.float32)


def build_chains(
    micrograph,
    tube,
    track,
    cls,
    prob=None,
    n_classes=None,
    temper=1.0,
    max_gap_steps=MAX_GAP_STEPS,
    backward=None,
):
    """Lay the segments of a 2D classification on grids along their filaments.

    Parameters
    ----------
    micrograph, tube, track : array_like
        Per segment: the micrograph name, the helical tube id and the position
        along the tube, A (``rlnMicrographName``, ``rlnHelicalTubeID``,
        ``rlnHelicalTrackLengthAngst``).
    cls : array_like of int
        The class of each segment, 1-based as RELION numbers them (0 or less: none).
    prob : array_like, optional
        The probability of that class (``rlnMaxValueProbDistribution``). Defaults
        to 0.9.
    n_classes : int, optional
        Number of classes. Defaults to the largest class number.
    temper : float, optional
        Factor on the log-likelihoods of the emissions. Defaults to 1.
    max_gap_steps : float, optional
        A gap of more than this many steps breaks a filament into chains.
    backward : array_like of bool, optional
        Per segment: whether it faces the other way along the filament than the
        direction it was picked in. The views of a helix follow the molecule, not
        the picking direction, so the class sequence of a filament picked backwards
        runs the other way, which a sequence model reads as a type of its own. A
        filament most of whose segments face backwards is laid out in the opposite
        order. Defaults to none.

    Returns
    -------
    Chains

    Raises
    ------
    ValueError
        When no filament has two segments with different positions.
    """
    import pandas as pd

    cls = np.asarray(cls).astype(np.int64) - 1
    n = len(cls)
    if n_classes is None:
        n_classes = int(cls.max()) + 1
    prob = np.full(n, 0.9) if prob is None else np.asarray(prob, dtype=float)
    key = (
        pd.Series(np.asarray(micrograph).astype(str))
        + "|"
        + pd.Series(np.asarray(tube).astype(str))
    )
    fil, uniq = pd.factorize(key)
    track = np.asarray(track, dtype=float)
    order = np.lexsort((track, fil))
    tr, fl = track[order], fil[order]
    d = np.diff(tr)
    same = np.diff(fl) == 0
    positive = d[same & (d > 0)]
    if len(positive) == 0:
        raise ValueError("no filament has more than one segment")
    step = float(np.median(positive))
    brk = np.r_[True, (~same) | (d / step > max_gap_steps)]
    chain_sorted = np.cumsum(brk) - 1
    first = tr[brk][chain_sorted]
    pos_sorted = np.rint((tr - first) / step).astype(np.int64)
    n_chain = int(chain_sorted[-1]) + 1
    length = np.zeros(n_chain, dtype=np.int64)
    np.maximum.at(length, chain_sorted, pos_sorted + 1)
    chain_filament = np.zeros(n_chain, dtype=np.int64)
    chain_filament[chain_sorted] = fl
    # numbered by length, the longest first (the positions that have all the
    # chains of a chunk are then the first ones: see _Chunk)
    rank = np.empty(n_chain, dtype=np.int64)
    rank[np.argsort(-length, kind="stable")] = np.arange(n_chain)
    pos_before = pos_sorted
    if backward is not None:
        # a filament is backward when most of its segments are; its chains run the other way
        b = np.asarray(backward, dtype=float)[order]
        fil_back = np.bincount(fl, weights=b, minlength=int(fl.max()) + 1) / np.maximum(
            np.bincount(fl, minlength=int(fl.max()) + 1), 1
        )
        flipped = fil_back[fl] > 0.5
        pos_sorted = np.where(
            flipped, length[chain_sorted] - 1 - pos_sorted, pos_sorted
        )
    chain = np.empty(n, dtype=np.int64)
    pos = np.empty(n, dtype=np.int64)
    chain[order] = rank[chain_sorted]
    pos[order] = pos_sorted
    chain_len = np.empty(n_chain, dtype=np.int64)
    chain_len[rank] = length
    chain_fil = np.empty(n_chain, dtype=np.int64)
    chain_fil[rank] = chain_filament
    unknown = cls < 0
    # of the segments that fall on one grid position the first along the filament counts
    # (as RELION does); the others carry no evidence, whatever order they were given in
    repeat = np.zeros(n, dtype=bool)
    repeat[order] = np.r_[
        False,
        (chain_sorted[1:] == chain_sorted[:-1]) & (pos_before[1:] == pos_before[:-1]),
    ]
    eps = emission_floor(prob, n_classes, temper)
    eps[unknown] = 1.0
    return Chains(
        step=step,
        n_classes=int(n_classes),
        filament=fil.astype(np.int64),
        filament_key=np.asarray(uniq, dtype=object),
        chain=chain,
        pos=pos,
        chain_len=chain_len,
        chain_filament=chain_fil,
        cls=np.where(unknown, -1, cls),
        eps=eps,
        repeat=repeat,
    )


# ── fused kernels ─────────────────────────────────────────────────────────
# The matrix products of the forward-backward are BLAS (several hundred GFlops
# on a node); what remains is elementwise work over the same (types, chains,
# classes) arrays, which as numpy expressions reads and writes them ten times
# over and ran 5 times slower than the products. Each of these does one
# position's elementwise work in a single pass, one chain at a time (the chain's
# T x K numbers stay in cache), over all threads.


@lazy_jit("njit", parallel=True, fastmath=True, nogil=True)
def _forward_step(G, s_prev, q, rho, sw, eps, cls, s_new, ll, n):
    """``G`` (T, n_max, K) holds ``alpha_{s-1} A`` for the first ``n`` chains; it
    becomes ``alpha_s``, normalised; ``s_new`` (T, n) the sum over classes of it;
    ``ll`` (n) gets the log of the normaliser."""
    T = G.shape[0]
    K = G.shape[2]
    for i in prange(n):
        stot = 0.0
        for t in range(T):
            stot += s_prev[t, i]
        c = cls[i]
        e = eps[i]
        z = 0.0
        for t in range(T):
            sin = sw * (stot - s_prev[t, i])
            for k in range(K):
                v = (1.0 - rho) * G[t, i, k] + sin * q[t, k]
                if k != c:
                    v *= e
                G[t, i, k] = v
                z += v
        inv = 1.0 / z
        ll[i] += np.log(z)
        for t in range(T):
            acc = 0.0
            for k in range(K):
                v = G[t, i, k] * inv
                G[t, i, k] = v
                acc += v
            s_new[t, i] = acc


@lazy_jit("njit", parallel=True, fastmath=True, nogil=True)
def _backward_prepare(beta, q, eps, cls, V, Q, n1):
    """``V = e * beta`` for the first ``n1`` chains, and ``Q[t] = sum_k q[t,k] V[t,k]``."""
    T = beta.shape[0]
    K = beta.shape[2]
    for i in prange(n1):
        c = cls[i]
        e = eps[i]
        for t in range(T):
            acc = 0.0
            for k in range(K):
                v = beta[t, i, k]
                if k != c:
                    v *= e
                V[t, i, k] = v
                acc += q[t, k] * v
            Q[t, i] = acc


@lazy_jit("njit", parallel=True, fastmath=True, nogil=True)
def _backward_finish(
    TMP,
    alpha,
    Q,
    wt,
    rho,
    sw,
    n,
    n1,
    collect,
    beta_out,
    a_out,
    g_blk,
    nsw_blk,
    ntr_blk,
    tpost,
    cbest,
    n_blocks,
):
    """Finish a position of the backward pass for the first ``n`` chains.

    ``TMP`` holds ``V A^T`` for the first ``n1`` chains (those that go on to the
    next position); the others have a backward message of ones. This makes the
    message ``beta_out`` (normalised), the posterior of the position (added to
    ``g_blk`` with the chain's weight, and written to ``tpost`` as the type posterior),
    and, when ``collect``, ``a_out`` (the forward message times the coefficient of
    the expected transitions) and the expected number of type switches. ``cbest``
    gets the class of the largest posterior (over the types)."""
    T = TMP.shape[0]
    K = TMP.shape[2]
    per = (n + n_blocks - 1) // n_blocks
    for b in prange(n_blocks):
        lo = b * per
        hi = min(n, lo + per)
        for i in range(lo, hi):
            go_on = i < n1
            qtot = 0.0
            if go_on:
                for t in range(T):
                    qtot += Q[t, i]
            zsum = 0.0
            bsum = 0.0
            for t in range(T):
                add = 0.0
                if go_on:
                    add = sw * (qtot - Q[t, i])
                for k in range(K):
                    if go_on:
                        v = (1.0 - rho) * TMP[t, i, k] + add
                    else:
                        v = 1.0
                    beta_out[t, i, k] = v
                    zsum += alpha[t, i, k] * v
                    bsum += v
            # the expected transitions: only for chains that go on
            if collect and go_on and zsum > 0.0:
                coef = wt[i] / zsum
                ntr_blk[b] += coef > 0.0
                for t in range(T):
                    st = 0.0
                    for k in range(K):
                        a = alpha[t, i, k]
                        a_out[t, i, k] = a * (1.0 - rho) * coef
                        st += a
                    nsw_blk[b] += st * sw * (qtot - Q[t, i]) * coef
            elif collect and go_on:
                for t in range(T):
                    for k in range(K):
                        a_out[t, i, k] = 0.0
            # the posterior of this position, and the normalised message
            ib = 1.0 / bsum
            inv = 1.0 / zsum if zsum > 0.0 else 0.0
            cm = np.zeros(K)
            for t in range(T):
                acc = 0.0
                for k in range(K):
                    gm = alpha[t, i, k] * beta_out[t, i, k] * inv
                    acc += gm
                    cm[k] += gm
                    g_blk[b, t, k] += wt[i] * gm
                    beta_out[t, i, k] *= ib
                tpost[i, t] = acc
            best = 0
            for k in range(1, K):
                if cm[k] > cm[best]:
                    best = k
            cbest[i] = best


_LOCAL = threading.local()
# the fused kernels run on all the threads of numba, and some of its threading
# layers abort when two Python threads launch parallel kernels at once: one fit at a time
_FIT_LOCK = threading.Lock()


def _buffer(name, shape, dtype=np.float32):
    """An array of ``shape`` from this thread's pool of buffers, not initialised.

    The arrays of the forward-backward are a few hundred MB, and the first write to
    fresh memory costs a page fault per 4 kB: on a node whose memory is mostly page
    cache that ran at a few hundred MB/s, against GB/s for memory already touched,
    and was most of the time of a pass. The buffers are therefore kept (the largest
    asked for, per name) and shared by the chunks, the passes and the fits of a
    thread; :func:`release_buffers` frees them.
    """
    pool = getattr(_LOCAL, "pool", None)
    if pool is None:
        pool = _LOCAL.pool = {}
    size = int(np.prod(shape))
    buf = pool.get(name)
    if buf is None or buf.size < size or buf.dtype != dtype:
        pool[name] = buf = np.empty(size, dtype=dtype)
    return buf[:size].reshape(shape)


def release_buffers():
    """Free the buffers that this thread's forward-backward passes keep."""
    _LOCAL.pool = {}


class _Chunk:
    """A group of chains, longest first, laid out position by position.

    The chains of a chunk are in order of length, so the ones that have a given
    position are always the first ``n_s``: arrays of a position are the first
    rows of those of the position before. Cell ``(s, r)`` is at ``off[s] + r``.
    """

    def __init__(self, chains, r0, r1):
        self.r0, self.r1 = r0, r1
        lens = chains.chain_len[r0:r1]
        n = r1 - r0
        self.n = n
        self.length = int(lens[0])
        ascending = lens[::-1]
        # chains longer than s
        self.n_s = n - np.searchsorted(ascending, np.arange(self.length), side="right")
        self.off = np.r_[0, np.cumsum(self.n_s)]
        total = int(self.off[-1])
        sel = np.flatnonzero((chains.chain >= r0) & (chains.chain < r1))
        flat = self.off[chains.pos[sel]] + (chains.chain[sel] - r0)
        # a position holding several segments takes the first one (along the filament)
        top = ~chains.repeat[sel]
        cells = flat[top]
        self.cls = np.full(total, -1, dtype=np.int32)
        self.eps = np.ones(total, dtype=np.float32)
        self.cls[cells] = chains.cls[sel][top]
        self.eps[cells] = chains.eps[sel][top]
        self.eps[cells[chains.cls[sel][top] < 0]] = 1.0
        self.segments = sel
        self.cell = flat
        self.filament = chains.chain_filament[r0:r1]

    def emission(self, s, n_classes):
        """``(n_s, K)`` likelihood of each class at position ``s``, per chain."""
        lo, hi = self.off[s], self.off[s + 1]
        c = self.cls[lo:hi]
        e = np.repeat(self.eps[lo:hi, None], n_classes, axis=1)
        rows = np.flatnonzero(c >= 0)
        e[rows, c[rows]] = 1.0
        return e


def _memory_budget():
    """Bytes the forward-backward may use for one chunk: a modest share of the free memory.

    Small chunks are as fast as large ones (a position's matrix products are over
    the chains of the chunk that reach it, and a few hundred are enough for BLAS)
    and touch little fresh memory; see :func:`_buffer`.
    """
    env = os.environ.get("HELICON_CONTEXT_MEMORY_GB")
    if env:
        return float(env) * 1024**3
    try:
        import psutil

        free = psutil.virtual_memory().available
    except Exception:
        free = 4 * 1024**3
    return float(min(max(0.02 * free, 64 * 1024**2), 384 * 1024**2))


def chunks_of(chains, n_types, memory=None, with_prior=False):
    """The chunks of a set of chains, sized to fit ``memory`` bytes (cached).

    A cell holds ``n_types * K`` single-precision forward probabilities; a chain
    needs, beside its cells, six working vectors of that size at its first
    position.

    Parameters
    ----------
    chains : Chains
    n_types : int
    memory : float, optional
        Bytes. Defaults to :func:`_memory_budget`.
    with_prior : bool, optional
        The backward message of every position is kept too (for the prior of the
        class of each segment): twice the memory a cell needs.

    Returns
    -------
    list of _Chunk
    """
    memory = _memory_budget() if memory is None else memory
    per_cell = 4.0 * n_types * chains.n_classes * (2.0 if with_prior else 1.0)
    max_cells = max(int(memory / per_cell), 1)
    key = (n_types, max_cells)
    if key in chains._chunks:
        return chains._chunks[key]
    out = []
    r0 = 0
    cum = np.cumsum(chains.chain_len + 6)
    while r0 < chains.n_chains:
        base = cum[r0 - 1] if r0 > 0 else 0
        r1 = int(np.searchsorted(cum, base + max_cells, side="right"))
        r1 = min(max(r1, r0 + 1), chains.n_chains)
        out.append(_Chunk(chains, r0, r1))
        r0 = r1
    chains._chunks.clear()  # one layout at a time
    chains._chunks[key] = out
    return out


@dataclass
class Model:
    """What is learnt: ``A`` (T, K, K), ``q`` (T, K), ``pi0`` (T * K), ``rho``."""

    A: np.ndarray
    q: np.ndarray
    pi0: np.ndarray
    rho: float


def _normalise_rows(a):
    s = a.sum(axis=-1, keepdims=True)
    return np.where(s > 0, a / np.where(s > 0, s, 1.0), 1.0 / a.shape[-1])


def nmf_model(chains, n_types, held_out=None, n_iter=300, seed=42):
    """The first model: a topic model of the classes of each filament.

    Kullback-Leibler non-negative matrix factorisation of the counts of the most
    likely class of the segments of each filament, ``X ~ W H`` with ``T`` topics;
    ``q_t = H_t``, the type weights of a filament are ``W_f``, and ``A_t`` counts
    the transitions of the filaments, each weighted by its ``W_f(t)``.

    Parameters
    ----------
    chains : Chains
    n_types : int
    held_out : numpy.ndarray, optional
        ``(F,)`` bool: filaments left out.
    n_iter : int, optional
        Multiplicative updates. Defaults to 300.
    seed : int, optional

    Returns
    -------
    Model
    """
    T, K, F = n_types, chains.n_classes, chains.n_filaments
    use = (chains.cls >= 0) & ~chains.repeat
    if held_out is not None:
        use &= ~held_out[chains.filament]
    X = np.zeros((F, K))
    np.add.at(X, (chains.filament[use], chains.cls[use]), 1.0)
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.5, 1.5, (F, T))
    H = rng.uniform(0.5, 1.5, (T, K))
    for _ in range(n_iter):
        R = X / np.maximum(W @ H, 1e-300)
        H *= (W.T @ R) / np.maximum(W.sum(0)[:, None], 1e-300)
        R = X / np.maximum(W @ H, 1e-300)
        W *= (R @ H.T) / np.maximum(H.sum(1)[None, :], 1e-300)
    q = _normalise_rows(H + 1e-6)
    Wn = W * H.sum(1)[None, :] + 1e-9
    Wn /= Wn.sum(1, keepdims=True)
    # transitions between neighbouring positions of a chain, both with a segment
    order = np.lexsort((chains.pos, chains.chain))
    ch, ps, cl, fl = (
        chains.chain[order],
        chains.pos[order],
        chains.cls[order],
        chains.filament[order],
    )
    nxt = (ch[1:] == ch[:-1]) & (ps[1:] == ps[:-1] + 1) & (cl[:-1] >= 0) & (cl[1:] >= 0)
    if held_out is not None:
        nxt &= ~held_out[fl[:-1]]
    a, b, f = cl[:-1][nxt], cl[1:][nxt], fl[:-1][nxt]
    cnt = np.empty((T, K, K))
    for t in range(T):
        cnt[t] = np.bincount(a * K + b, weights=Wn[f, t], minlength=K * K).reshape(K, K)
    mass = (Wn[fl[use[order]]]).sum(0) if use.any() else np.ones(T)
    mass = mass / mass.sum()
    A = _normalise_rows(cnt + q[:, None, :])
    pi0 = (mass[:, None] * q).reshape(-1)
    return Model(A=A, q=q, pi0=pi0, rho=1e-3)


def _forward_backward(
    chunk,
    n_classes,
    model,
    held_out,
    *,
    collect=True,
    posterior=False,
    backward=True,
    prior_steps=None,
):
    """One forward-backward pass over a chunk.

    Parameters
    ----------
    chunk : _Chunk
    n_classes : int
    model : Model
    held_out : numpy.ndarray or None
        ``(F,)`` bool: filaments whose chains count for the likelihood only.
    collect : bool, optional
        Collect the statistics for the M-step (expected transitions, switches and
        the state posteriors) from the chains that are not held out.
    posterior : bool, optional
        Return the posterior of the type of every cell.
    backward : bool, optional
        Run the backward pass; without it only the log-likelihood is found.
    prior_steps : int, optional
        With ``posterior``: also find, for every position, the prior of its class
        from the positions at least this many steps away on either side
        (:func:`_context_prior`).

    Returns
    -------
    dict
        ``ll_train`` and ``ll_test`` (log-likelihood of the chains not held out and
        of those held out), the statistics ``xi`` (T, K, K), ``g`` (T, K), ``nsw``
        and ``ntr`` when ``collect``, ``tpost`` (cells, T) and ``cbest`` (cells: the
        class of the largest posterior) when ``posterior``, and with ``prior_steps``
        ``ctx_prob`` and ``ctx_hit`` (cells).
    """
    T, K = model.q.shape
    A = np.ascontiguousarray(model.A, dtype=np.float32)
    AT = np.ascontiguousarray(A.transpose(0, 2, 1))
    q = np.ascontiguousarray(model.q, dtype=np.float32)
    rho = float(model.rho)
    sw = rho / (T - 1) if T > 1 else 0.0
    n_s, off, L = chunk.n_s, chunk.off, chunk.length
    test = (
        np.zeros(chunk.n, dtype=bool) if held_out is None else held_out[chunk.filament]
    )
    wt = (~test).astype(np.float32)
    ll = np.zeros(chunk.n)
    cls_pm, eps_pm = chunk.cls, chunk.eps
    n0 = int(n_s[0])

    # forward: alpha of every position in one array, sums over classes beside it
    store = _buffer("alpha", (T * K * int(off[-1]),))
    sums = _buffer("sums", (T * int(off[-1]),))

    def a_view(s):
        return store[T * K * off[s] : T * K * off[s + 1]].reshape(T, int(n_s[s]), K)

    def s_view(s):
        return sums[T * off[s] : T * off[s + 1]].reshape(T, int(n_s[s]))

    a0 = a_view(0)
    e0 = chunk.emission(0, K)
    a0[:] = model.pi0.reshape(T, 1, K).astype(np.float32) * e0[None]
    z = a0.sum(axis=(0, 2))
    ll += np.log(z)
    a0 /= z[None, :, None]
    s_view(0)[:] = a0.sum(axis=2)
    for s in range(1, L):
        n = int(n_s[s])
        ap = a_view(s - 1)
        cur = a_view(s)
        for t in range(T):
            np.matmul(ap[t, :n], A[t], out=cur[t])
        _forward_step(
            cur,
            s_view(s - 1),
            q,
            rho,
            sw,
            eps_pm[off[s] : off[s + 1]],
            cls_pm[off[s] : off[s + 1]],
            s_view(s),
            ll,
            n,
        )
    out = dict(
        ll_train=float(ll[~test].sum()),
        ll_test=float(ll[test].sum()),
    )
    if not backward:
        return out

    from scipy.linalg.blas import sgemm

    n_blocks = 32  # of chains, for the thread-local sums of the kernel
    xi = np.zeros((T, K, K))
    g_blk = np.zeros((n_blocks, T, K))
    nsw_blk = np.zeros(n_blocks)
    ntr_blk = np.zeros(n_blocks)
    tpost = np.zeros((int(off[-1]), T), dtype=np.float32) if posterior else None
    cbest_all = np.zeros(int(off[-1]), dtype=np.int32) if posterior else None
    cb = np.zeros(n0, dtype=np.int32)
    tp = np.zeros((n0, T), dtype=np.float32)
    keep_beta = posterior and prior_steps is not None
    bstore = _buffer("beta_all", (T * K * int(off[-1]),)) if keep_beta else None

    def b_view(s):
        return bstore[T * K * off[s] : T * K * off[s + 1]].reshape(T, int(n_s[s]), K)

    # the last position of the longest chains has no message from the right
    alpha_last = a_view(L - 1)
    n = int(n_s[L - 1])
    g_blk[0] += np.einsum("tnk,n->tk", alpha_last, wt[:n])
    if posterior:
        tpost[off[L - 1] : off[L]] = alpha_last.sum(axis=2).T
        cbest_all[off[L - 1] : off[L]] = alpha_last.sum(axis=0).argmax(axis=1)
    if keep_beta:
        b_view(L - 1)[:] = 1.0
    beta = _buffer("beta", (T, n0, K))
    beta.fill(1.0)
    beta_next = _buffer("beta_next", (T, n0, K))
    V = _buffer("V", (T, n0, K))
    TMP = _buffer("TMP", (T, n0, K))
    a_out = _buffer("a_out", (T, n0, K))
    Q = _buffer("Q", (T, n0))
    xi_raw = np.zeros((T, K, K), dtype=np.float32)
    for s in range(L - 2, -1, -1):
        n, n1 = int(n_s[s]), int(n_s[s + 1])
        lo1, hi1 = off[s + 1], off[s + 2]
        _backward_prepare(beta, q, eps_pm[lo1:hi1], cls_pm[lo1:hi1], V, Q, n1)
        for t in range(T):
            np.matmul(V[t, :n1], AT[t], out=TMP[t, :n1])
        _backward_finish(
            TMP,
            a_view(s),
            Q,
            wt,
            rho,
            sw,
            n,
            n1,
            collect,
            beta_next,
            a_out,
            g_blk,
            nsw_blk,
            ntr_blk,
            tp,
            cb,
            n_blocks,
        )
        if keep_beta:
            b_view(s)[:] = beta_next[:, :n]
        if collect:
            # the sum over positions of a^T V; A multiplies it once at the end
            for t in range(T):
                # in Fortran terms (the arrays are C ordered): xi_raw^T += V^T a
                sgemm(
                    1.0,
                    V[t, :n1].T,
                    a_out[t, :n1].T,
                    beta=1.0,
                    c=xi_raw[t].T,
                    trans_b=1,
                    overwrite_c=1,
                )
        if posterior:
            tpost[off[s] : off[s + 1]] = tp[:n]
            cbest_all[off[s] : off[s + 1]] = cb[:n]
        beta, beta_next = beta_next, beta
    xi += A * xi_raw
    out.update(
        xi=xi, g=g_blk.sum(axis=0), nsw=float(nsw_blk.sum()), ntr=float(ntr_blk.sum())
    )
    if posterior:
        out["tpost"] = tpost
        out["cbest"] = cbest_all
    if keep_beta:
        out["ctx_prob"], out["ctx_hit"] = _context_prior(
            chunk, model, a_view, b_view, int(prior_steps)
        )
    return out


def _context_prior(chunk, model, a_view, b_view, w):
    """The prior of the class of every position from the positions ``w`` or more steps away.

    The segments of a filament overlap a lot (the box is much longer than the
    step), so those next to a segment share most of its pixels and say nothing
    independent of its class. The prior of a position therefore uses only the
    positions at least ``w`` steps away on either side, as RELION's
    ``HelicalContext::update`` does:

        left_t(c)  = (1-rho)^w sum_c' alpha_{s-w}(t,c') A_t^w[c'][c]
                     + (1 - (1-rho)^w) / (T-1) * (sum of alpha_{s-w} of the other types) q_t(c)
        right_t(c) = (1-rho)^w sum_c' A_t^w[c][c'] (e beta)_{s+w}(t,c')
                     + (1 - (1-rho)^w) / (T-1) * (sum over the other types of q.(e beta))
        prior(c)   = sum_t left_t(c) right_t(c),  scaled to a largest value of 1, at least 1e-3

    (before the first ``w`` positions the left message is the initial distribution;
    after the last ``w`` the right message is 1).

    Returns
    -------
    ctx_prob : numpy.ndarray
        ``(cells,)`` the share of the prior of each position that is the prior of
        its assigned class (0 where there is none).
    ctx_hit : numpy.ndarray
        ``(cells,)`` bool: whether the class of the largest prior is the assigned one.
    """
    T, K = model.q.shape
    rho = float(model.rho)
    stay = (1.0 - rho) ** w
    switch = (1.0 - stay) / (T - 1) if T > 1 else 0.0
    Aw = np.stack([np.linalg.matrix_power(model.A[t], w) for t in range(T)]).astype(
        np.float32
    )
    AwT = np.ascontiguousarray(Aw.transpose(0, 2, 1))
    q = np.ascontiguousarray(model.q, dtype=np.float32)
    pi0 = model.pi0.reshape(T, 1, K).astype(np.float32)
    n_s, off, L = chunk.n_s, chunk.off, chunk.length
    ctx = np.zeros(int(off[-1]), dtype=np.float32)
    hit = np.zeros(int(off[-1]), dtype=bool)
    for s in range(L):
        n = int(n_s[s])
        if s - w >= 0:
            ap = a_view(s - w)[:, :n]
            S = ap.sum(axis=2)
            left = np.empty((T, n, K), dtype=np.float32)
            for t in range(T):
                np.matmul(ap[t], Aw[t], out=left[t])
            left *= stay
            left += (switch * (S.sum(axis=0)[None, :] - S))[:, :, None] * q[:, None, :]
        else:
            left = np.broadcast_to(pi0, (T, n, K))
        right = np.ones((T, n, K), dtype=np.float32)
        if s + w < L:
            n2 = int(n_s[s + w])
            v = chunk.emission(s + w, K)[None] * b_view(s + w)
            Q = np.einsum("tk,tnk->tn", q, v)
            r2 = np.empty((T, n2, K), dtype=np.float32)
            for t in range(T):
                np.matmul(v[t], AwT[t], out=r2[t])
            r2 *= stay
            r2 += (switch * (Q.sum(axis=0)[None, :] - Q))[:, :, None]
            right[:, :n2] = r2
        prior = (left * right).sum(axis=0)  # (n, K)
        top = prior.max(axis=1)
        prior = np.where(
            top[:, None] > 0,
            np.maximum(1e-3, prior / np.maximum(top, 1e-30)[:, None]),
            1.0,
        )
        assigned = chunk.cls[off[s] : off[s + 1]]
        rows = np.flatnonzero(assigned >= 0)
        ctx[off[s] + rows] = prior[rows, assigned[rows]] / prior[rows].sum(axis=1)
        hit[off[s] + rows] = prior[rows].argmax(axis=1) == assigned[rows]
    return ctx, hit


def fit(chains, n_types=4, max_fit_segments=None, **kw):
    """Learn the model by Baum-Welch and decode the types of every segment.

    Parameters
    ----------
    chains : Chains
    n_types : int, optional
        Number of types (one more if there are junk filaments). Defaults to 4.
    max_fit_segments : int, optional
        Learn from about this many segments only, in whole filaments chosen at
        random, then decode all of them with the model (one more pass). The model
        has ``n_types * K**2`` numbers, which a few hundred thousand segments
        determine well, and the time of a fit grows with the segments. Defaults to
        all.
    **kw
        See :func:`_fit`.

    Returns
    -------
    dict
        See :func:`_fit`; ``tpost`` is for all the segments of ``chains``.
    """
    with _FIT_LOCK:
        if (
            max_fit_segments is None
            or chains.n_segments <= max_fit_segments
            or kw.get("held_out") is not None
        ):
            return _fit(chains, n_types, **kw)
        sub = subsample(chains, max_fit_segments)
        learn = {k: v for k, v in kw.items() if k not in ("posterior", "held_out")}
        learnt = _fit(sub, n_types, posterior=False, **learn)
        decoded = _fit(
            chains,
            n_types,
            n_iter=0,
            model=learnt["model"],
            **{
                k: v
                for k, v in kw.items()
                if k in ("memory", "cpu", "posterior", "prior_steps")
            },
        )
        decoded["ll_train"] = learnt["ll_train"] * chains.n_segments / sub.n_segments
        decoded["n_fit_segments"] = sub.n_segments
        return decoded


def _fit(
    chains,
    n_types=4,
    n_iter=3,
    held_out=None,
    model=None,
    memory=None,
    cpu=None,
    posterior=True,
    prior_steps=None,
    verbose=False,
):
    """Learn the model by Baum-Welch and decode the types.

    Parameters
    ----------
    chains : Chains
    n_types : int, optional
        Number of types (one more if there are junk filaments). Defaults to 4.
    n_iter : int, optional
        Baum-Welch iterations. Defaults to 3.
    held_out : numpy.ndarray, optional
        ``(F,)`` bool: filaments that are not used to learn, whose log-likelihood
        is returned.
    model : Model, optional
        The model to start from. Defaults to :func:`nmf_model`.
    memory : float, optional
        Bytes for a chunk; see :func:`chunks_of`.
    cpu : int, optional
        Threads of the matrix products. Defaults to ``helicon.available_cpu()``.
    prior_steps : int, optional
        With ``posterior``: also return the context probability of every segment,
        from the segments at least this many steps away (:func:`_context_prior`).
    posterior : bool, optional
        Whether to return the posterior type of each segment (one more pass).
        Defaults to True.
    verbose : bool, optional

    Returns
    -------
    result : dict
        ``model``, ``ll_train``, ``ll_test`` (per held-out segment is up to the
        caller), ``n_test`` (segments held out) and, with ``posterior``,
        ``tpost`` ``(N, T)``, ``class_best`` ``(N,)`` (the class of the largest
        posterior) and, with ``prior_steps``, ``ctx_prob`` and ``ctx_hit`` ``(N,)``.
    """
    import time

    from threadpoolctl import threadpool_limits

    if cpu is None:
        import helicon

        cpu = helicon.available_cpu()
    if model is None or model.q.shape[0] != n_types:
        model = nmf_model(chains, n_types, held_out)
    K = chains.n_classes
    chunks = chunks_of(chains, n_types, memory, with_prior=prior_steps is not None)
    result = {}
    from helicon.lib.numba_compat import set_num_threads

    set_num_threads(cpu)
    with threadpool_limits(limits=cpu):
        for it in range(n_iter + 1):
            last = it == n_iter
            t0 = time.time()
            xi = np.zeros((n_types, K, K))
            g = np.zeros((n_types, K))
            nsw = ntr = ll_train = ll_test = 0.0
            tpost_cells = []
            for chunk in chunks:
                r = _forward_backward(
                    chunk,
                    K,
                    model,
                    held_out,
                    collect=not last,
                    posterior=last and posterior,
                    backward=(not last) or posterior,
                    prior_steps=prior_steps if last else None,
                )
                ll_train += r["ll_train"]
                ll_test += r["ll_test"]
                if "xi" in r:
                    xi += r["xi"]
                    g += r["g"]
                    nsw += r["nsw"]
                    ntr += r["ntr"]
                if "tpost" in r:
                    tpost_cells.append(r)
            if verbose:
                logger.info(
                    "Baum-Welch %d: log-likelihood %.1f, rho %.2g (%.1f s)",
                    it,
                    ll_train,
                    model.rho,
                    time.time() - t0,
                )
            if last:
                break
            rho = float(np.clip(nsw / ntr, RHO_MIN, RHO_MAX)) if ntr > 0 else model.rho
            q = _normalise_rows(g + 1e-6)
            A = _normalise_rows(xi + q[:, None, :])
            pi0 = (g.reshape(-1) + 0.0) / max(g.sum(), 1e-300)
            model = Model(A=A, q=q, pi0=pi0, rho=rho)
    result.update(
        model=model,
        ll_train=ll_train,
        ll_test=ll_test,
        n_test=0 if held_out is None else int(held_out[chains.filament].sum()),
    )
    if posterior and tpost_cells:
        tpost = np.zeros((chains.n_segments, n_types), dtype=np.float32)
        class_best = np.zeros(chains.n_segments, dtype=np.int32)
        ctx_prob = np.zeros(chains.n_segments, dtype=np.float32)
        ctx_hit = np.zeros(chains.n_segments, dtype=bool)
        for chunk, r in zip(chunks, tpost_cells):
            tpost[chunk.segments] = r["tpost"][chunk.cell]
            class_best[chunk.segments] = r["cbest"][chunk.cell]
            if "ctx_prob" in r:
                ctx_prob[chunk.segments] = r["ctx_prob"][chunk.cell]
                ctx_hit[chunk.segments] = r["ctx_hit"][chunk.cell]
        result["tpost"] = tpost
        result["class_best"] = class_best
        if prior_steps is not None:
            result["ctx_prob"], result["ctx_hit"] = ctx_prob, ctx_hit
    return result


def select_n_types(
    chains, t_min=3, t_max=8, n_fold=3, tol=0.01, n_iter=3, max_segments=None, **kw
):
    """The number of types by the likelihood of filaments held out of the fit.

    For each number of types, ``n_fold`` times, the model is learnt from the
    filaments of the other folds (a filament is in fold ``index % n_fold``) and the
    log-likelihood of the filaments held out is summed. The fewest types whose
    held-out log-likelihood per segment is within ``tol`` nats of the best is
    chosen: more types split real ones less reproducibly.

    Parameters
    ----------
    chains : Chains
    t_min, t_max : int, optional
        The numbers of types tried. Defaults to 3 to 8.
    n_fold : int, optional
        Defaults to 3.
    tol : float, optional
        Nats per segment. Defaults to 0.01.
    n_iter : int, optional
        Baum-Welch iterations of each fit. Defaults to 3.
    max_segments : int, optional
        Learn from about this many segments only, taken as whole filaments (the
        likelihood of held-out filaments is the same kind of number from a part).
        Defaults to all.
    **kw
        Passed to :func:`fit`.

    Returns
    -------
    best : int
    scores : dict
        Held-out log-likelihood per segment by number of types.
    """
    F = chains.n_filaments
    sub = chains
    if max_segments is not None and chains.n_segments > max_segments:
        sub = subsample(chains, max_segments)
    folds = np.arange(sub.n_filaments) % n_fold
    scores = {}
    for T in range(t_min, t_max + 1):
        total, count = 0.0, 0
        for fold in range(n_fold):
            held = folds == fold
            r = fit(
                sub,
                T,
                n_iter=n_iter,
                held_out=held,
                posterior=False,
                **kw,
            )
            total += r["ll_test"]
            count += int(held[sub.filament].sum())
        scores[T] = total / max(count, 1)
    top = max(scores, key=scores.get)
    best = next(T for T in sorted(scores) if scores[T] >= scores[top] - tol)
    return best, scores


def subsample(chains, max_segments, seed=0):
    """The chains of about ``max_segments`` segments in whole filaments.

    Parameters
    ----------
    chains : Chains
    max_segments : int
    seed : int, optional

    Returns
    -------
    Chains
        A new set of chains (re-laid out).
    """
    rng = np.random.default_rng(seed)
    per_filament = np.bincount(chains.filament, minlength=chains.n_filaments)
    order = rng.permutation(chains.n_filaments)
    keep = order[np.cumsum(per_filament[order]) <= max_segments]
    mask = np.isin(chains.filament, keep)
    sel = np.flatnonzero(mask)
    return _relayout(chains, sel)


def _relayout(chains, sel):
    """Chains of the segments ``sel`` of ``chains`` (their own filaments, chains and ranks)."""
    fil = chains.filament[sel]
    key, new_fil = np.unique(fil, return_inverse=True)
    ch = chains.chain[sel]
    uniq_ch, new_ch = np.unique(ch, return_inverse=True)
    length = np.zeros(len(uniq_ch), dtype=np.int64)
    np.maximum.at(length, new_ch, chains.pos[sel] + 1)
    rank = np.empty(len(uniq_ch), dtype=np.int64)
    rank[np.argsort(-length, kind="stable")] = np.arange(len(uniq_ch))
    chain_filament = np.empty(len(uniq_ch), dtype=np.int64)
    chain_filament[rank[new_ch]] = new_fil
    chain_len = np.empty(len(uniq_ch), dtype=np.int64)
    chain_len[rank] = length
    return Chains(
        step=chains.step,
        n_classes=chains.n_classes,
        filament=new_fil.astype(np.int64),
        filament_key=chains.filament_key[key],
        chain=rank[new_ch],
        pos=chains.pos[sel],
        chain_len=chain_len,
        chain_filament=chain_filament,
        cls=chains.cls[sel],
        eps=chains.eps[sel],
        repeat=chains.repeat[sel],
    )


def segment_types(chains, tpost):
    """The type of each segment from its posterior, and the sections along the chains.

    A section is a run of segments of one type along a chain. A run of fewer than
    :data:`MIN_RUN` segments between two runs of the same type is taken into
    them (the type does not flicker).

    Parameters
    ----------
    chains : Chains
    tpost : numpy.ndarray
        ``(N, T)`` posterior of the type of each segment.

    Returns
    -------
    type : numpy.ndarray
        ``(N,)`` the type of each segment, from 0.
    prob : numpy.ndarray
        ``(N,)`` its posterior probability.
    section : numpy.ndarray
        ``(N,)`` the section of each segment, from 0 (numbered along the chains).
    """
    t = tpost.argmax(axis=1)
    prob = tpost[np.arange(len(t)), t]
    order = np.lexsort((chains.pos, chains.chain))
    ch = chains.chain[order]
    ty = t[order].copy()
    new_run = np.r_[True, (ch[1:] != ch[:-1]) | (ty[1:] != ty[:-1])]
    run_id = np.cumsum(new_run) - 1
    starts = np.flatnonzero(new_run)
    run_len = np.diff(np.r_[starts, len(ty)])
    run_type = ty[starts]
    run_chain = ch[starts]
    # short interior runs between two of the same type: merge, chain by chain
    interior = np.flatnonzero(
        (run_len[1:-1] < MIN_RUN)
        & (run_type[:-2] == run_type[2:])
        & (run_chain[:-2] == run_chain[1:-1])
        & (run_chain[1:-1] == run_chain[2:])
    )
    if len(interior):
        chain_of_run = np.unique(run_chain[interior + 1])
        for c in chain_of_run:
            idx = np.flatnonzero(run_chain == c)
            runs = [[int(run_type[i]), int(run_len[i])] for i in idx]
            changed = True
            while changed:
                changed = False
                for r in range(1, len(runs) - 1):
                    if runs[r][1] < MIN_RUN and runs[r - 1][0] == runs[r + 1][0]:
                        runs[r - 1][1] += runs[r][1] + runs[r + 1][1]
                        del runs[r : r + 2]
                        changed = True
                        break
            lo = starts[idx[0]]
            n_after = sum(x[1] for x in runs)
            j = lo
            for ty_r, ln in runs:
                ty[j : j + ln] = ty_r
                j += ln
            assert j - lo == n_after
        new_run = np.r_[True, (ch[1:] != ch[:-1]) | (ty[1:] != ty[:-1])]
    section_sorted = np.cumsum(new_run) - 1
    out_type = np.empty_like(t)
    section = np.empty(len(t), dtype=np.int64)
    out_type[order] = ty
    section[order] = section_sorted
    return out_type, prob, section


def filament_types(chains, tpost):
    """The type of each filament: the one with the largest mean posterior.

    Parameters
    ----------
    chains : Chains
    tpost : numpy.ndarray
        ``(N, T)`` posterior of the type of each segment.

    Returns
    -------
    type : numpy.ndarray
        ``(F,)`` the type of each filament, from 0.
    share : numpy.ndarray
        ``(F,)`` its mean posterior over the filament's segments.
    """
    F, T = chains.n_filaments, tpost.shape[1]
    total = np.zeros((F, T))
    np.add.at(total, chains.filament, tpost)
    total /= np.maximum(total.sum(axis=1, keepdims=True), 1e-300)
    t = total.argmax(axis=1)
    return t, total[np.arange(F), t]


def context_groups(
    micrograph,
    tube,
    track,
    cls,
    prob,
    classes,
    keys,
    n_groups,
    max_fit_segments=400_000,
    n_iter=3,
    cpu=None,
    progress=None,
    select=True,
    select_segments=100_000,
    tol=0.01,
    counterparts=None,
    psi=None,
    psi_prior=None,
    diameter=None,
    detail=False,
    similarity=None,
    similarity_classes=None,
    compare_classes=None,
):
    """The filaments grouped by type for each of several numbers of types.

    Segments of the classes that are not in ``classes`` (rare or blank ones) keep
    their place on the grid of their filament but carry no evidence, and the
    classes are numbered again from 0, so the model is over the classes that matter.
    Each number of types is fitted on its own (the models are not nested). A
    filament is of the type with the largest mean posterior over its segments.

    Parameters
    ----------
    micrograph, tube, track, cls, prob : array_like
        Per segment: the micrograph, tube, position along the tube (A), the class
        (1-based) and the probability of that class (None: 0.9 for all).
    classes : sequence of int
        The class numbers (1-based) that are modelled.
    keys : sequence of tuple
        ``(micrograph, tube)`` of the filaments to label, in the order wanted.
    n_groups : sequence of int
        The numbers of types to fit.
    max_fit_segments : int, optional
        See :func:`fit`. Defaults to 400,000.
    n_iter : int, optional
        Baum-Welch iterations. Defaults to 3.
    cpu : int, optional
        See :func:`fit`.
    progress : callable, optional
        ``progress(done, total, message)`` after each number of types.
    select : bool, optional
        Also choose the number of types by the log-likelihood of filaments held out
        of the fit (:func:`select_n_types`, as RELION's ``--types_max`` does).
        Defaults to True.
    select_segments : int, optional
        The choice learns from about this many segments. Defaults to 100,000.
    tol : float, optional
        The fewest types within this many nats per segment of the best are chosen.
        Defaults to 0.01, as in RELION.
    counterparts : sequence of dict, optional
        Pairs of classes that are each other's 180-degree rotation, as
        ``helical_pitch_phase.pair_counterparts`` finds them (``a`` and ``b``, class
        numbers, 1-based). The class of the pair with fewer segments is taken as the
        other one, and its segments as facing backwards along their filament.
    psi, psi_prior : array_like, optional
        Per segment: the in-plane angle and its prior, degrees. A segment whose
        angle is turned by more than 90 degrees from its prior faces backwards
        (twice backwards, by its class and by its angle, is forwards).
    diameter : float, optional
        The particle (mask) diameter, A. The context probability of a segment is
        taken from the segments at least this far away (``--exclude`` of RELION), so
        that segments that share its own image do not predict it. Default: not
        computed.
    detail : bool, optional
        Also return, for each number of types, the tables of RELION's output
        (``info["detail"]``). Defaults to False.
    similarity : numpy.ndarray, optional
        ``(K, K)`` registered correlation of the class averages
        (:func:`helical_pitch_types.class_similarity`), in the order of
        ``similarity_classes``. With it, ``info["likeness"]`` has how alike the class
        averages of each pair of types look (:func:`type_likeness`). It does not
        change the number of types: on a data set of known structures the types of
        different structures were about as alike as those of one (see the benchmark).
    similarity_classes : sequence of int, optional
        The class numbers of the rows of ``similarity``. Defaults to ``classes``.
    compare_classes : sequence of int, optional
        The class numbers whose averages are compared. Classes that look like no other
        (junk, or a view of their own) say nothing about the structure, and a type that
        uses many of them looks unlike the rest only for that. Defaults to all of
        ``similarity_classes``.

    Returns
    -------
    labels : dict
        Number of types -> ``(len(keys),)`` the type of each filament, from 0
        (-1: a filament with no segment in the data).
    info : dict
        ``n_segments``, ``n_filaments``, ``step`` (A) and, by number of types, the
        log-likelihood per segment and ``rho``, and with ``select`` ``default_n`` (the
        number of types chosen) and ``held_out`` (the held-out log-likelihood per
        segment by number of types). With ``detail``, ``info["detail"][n]`` is a dict
        with ``labels`` (:func:`segment_labels`, one row per input segment, in the
        order of the input), ``types`` and ``filaments`` (:func:`type_tables`),
        ``period`` and ``period_score`` (:func:`fit_periods`) and, with ``diameter``,
        ``context_hit`` (the share of segments whose class the segments far along
        their filament predict), ``context_steps`` and ``context_angst``. With
        ``similarity``: ``likeness[n]`` (``(n, n)`` mean correlation between the
        classes of each pair of types). ``coverage``: how much of the class cycle a
        filament shows (:func:`filament_coverage`).
    """
    cls = np.asarray(cls).astype(np.int64)
    top = int(max(cls.max(), max(classes), 1))
    classes = [int(c) for c in classes]
    # a counterpart is its partner, faced the other way
    partner = np.arange(top + 1)
    is_back = np.zeros(top + 1, dtype=bool)
    counts = np.bincount(cls[cls > 0], minlength=top + 1)
    for d in counterparts or []:
        a, b = int(d["a"]), int(d["b"])
        if a > top or b > top or a not in classes or b not in classes:
            continue
        keep, gone = (a, b) if counts[a] >= counts[b] else (b, a)
        partner[gone] = keep
        is_back[gone] = True
    modelled = sorted({int(partner[c]) for c in classes})
    to_new = np.zeros(top + 1, dtype=np.int64)
    in_play = np.zeros(top + 1, dtype=bool)
    in_play[classes] = True
    to_new[modelled] = np.arange(1, len(modelled) + 1)
    new_cls = np.where(in_play[cls], to_new[partner[cls]], 0)
    backward = is_back[cls] & in_play[cls]
    if psi is not None and psi_prior is not None:
        turned = np.cos(np.deg2rad(np.asarray(psi) - np.asarray(psi_prior))) < 0
        backward = backward ^ turned
    chains = build_chains(
        micrograph,
        tube,
        track,
        new_cls,
        prob=prob,
        n_classes=len(modelled),
        backward=backward if backward.any() else None,
    )
    lookup = {k: i for i, k in enumerate(chains.filament_key)}
    row = np.array([lookup.get(f"{m}|{t}", -1) for m, t in keys], dtype=np.int64)
    labels, info = {}, dict(
        n_segments=chains.n_segments,
        n_filaments=chains.n_filaments,
        step=chains.step,
        ll={},
        rho={},
    )
    info["n_modelled_classes"] = len(modelled)
    info["backward_share"] = float(backward.mean())
    info["detail"] = {}
    info["likeness"] = {}
    info["coverage"] = filament_coverage(chains)
    S = None
    if similarity is not None:
        rows = [int(c) for c in (similarity_classes or classes)]
        where = {c: i for i, c in enumerate(rows)}
        S = np.asarray(similarity, dtype=float)[
            np.ix_([where[c] for c in modelled], [where[c] for c in modelled])
        ]
        compare = np.array(
            [compare_classes is None or c in set(compare_classes) for c in modelled]
        )
    prior_steps = None
    if detail and diameter:
        prior_steps = max(1, int(round(float(diameter) / chains.step)))
    n_groups = [int(n) for n in n_groups]
    if select and len(n_groups) > 1:
        if progress is not None:
            progress(0, len(n_groups), "choosing the number of types")
        best, scores = select_n_types(
            chains,
            1,
            max(n_groups),
            tol=tol,
            n_iter=n_iter,
            max_segments=select_segments,
            cpu=cpu,
        )
        info["default_n"], info["held_out"] = best, scores
    for i, n in enumerate(n_groups):
        r = fit(
            chains,
            int(n),
            n_iter=n_iter,
            max_fit_segments=max_fit_segments,
            cpu=cpu,
            prior_steps=prior_steps,
        )
        ft, _ = filament_types(chains, r["tpost"])
        if detail:
            info["detail"][int(n)] = _detail(
                chains, r, int(n), modelled, prior_steps, backward
            )
        labels[int(n)] = np.where(row >= 0, ft[np.maximum(row, 0)], -1)
        if S is not None:
            usage = class_usage(chains, r["tpost"])[:, compare]
            Sc = S[np.ix_(compare, compare)]
            info["likeness"][int(n)] = type_likeness(usage, Sc)
        info["ll"][int(n)] = r["ll_train"] / chains.n_segments
        info["rho"][int(n)] = r["model"].rho
        if progress is not None:
            progress(i + 1, len(n_groups), f"fitted {n} types")
    release_buffers()
    return labels, info


def _detail(chains, r, n, class_numbers, prior_steps, backward):
    """Labels, tables and periods of one fit (see ``detail`` of :func:`context_groups`)."""
    seg_type, seg_prob, section = segment_types(chains, r["tpost"])
    period, score = fit_periods(chains, seg_type, seg_prob, r["class_best"], n)
    ctx_prob = r.get("ctx_prob")
    types, filaments = type_tables(
        chains,
        seg_type,
        seg_prob,
        section,
        r["tpost"],
        r["model"],
        class_numbers,
        period,
        score,
    )
    out = dict(
        labels=segment_labels(seg_type, seg_prob, section, ctx_prob, chains),
        types=types,
        filaments=filaments,
        period=period,
        period_score=score,
        q=r["model"].q,
    )
    if prior_steps is not None and "ctx_hit" in r:
        out["context_hit"] = float(
            r["ctx_hit"][(chains.cls >= 0) & ~chains.repeat].mean()
        )
        out["context_steps"] = prior_steps
        out["context_angst"] = prior_steps * chains.step
    return out


# the numbers of types fitted for the Types card; the number it starts at is chosen by
# the held-out likelihood (context_groups), and this one is used where that is not
CONTEXT_MAX_TYPES = 8
CONTEXT_DEFAULT_TYPES = 5


def context_splits(params, classes, keys, n_max=CONTEXT_MAX_TYPES, progress=None, **kw):
    """The filaments grouped by context for 2 to ``n_max`` types, as the Types card shows splits.

    Parameters
    ----------
    params : pandas.DataFrame
        The segments of a 2D classification (``rlnMicrographName``,
        ``rlnHelicalTubeID``, ``rlnHelicalTrackLengthAngst``, ``rlnClassNumber`` and,
        if there, ``rlnMaxValueProbDistribution``).
    classes : sequence of int
        The classes (1-based) that are modelled.
    keys : sequence of tuple
        ``(micrograph, tube)`` of each filament, in the order of the rows of the
        card's usage matrix.
    n_max : int, optional
        Defaults to :data:`CONTEXT_MAX_TYPES`.
    progress : callable, optional
        ``progress(done, total, message)``.
    **kw
        Passed to :func:`context_groups`.

    Returns
    -------
    splits : list of dict
        For 2, 3, ... types: ``labels`` (the type of every filament, -1 for none),
        ``gain`` (the training log-likelihood per segment over that of one type
        fewer), and the fields ``describe_marker_groups`` reads (``markers`` empty,
        ``high`` 1: every group is a type of its own, none lacks another's markers).
    info : dict
        From :func:`context_groups`.
    """
    prob = (
        params["rlnMaxValueProbDistribution"].to_numpy(dtype=float)
        if "rlnMaxValueProbDistribution" in params
        else None
    )
    if "rlnAnglePsi" in params and "rlnAnglePsiPrior" in params:
        kw.setdefault("psi", params["rlnAnglePsi"].to_numpy(dtype=float))
        kw.setdefault("psi_prior", params["rlnAnglePsiPrior"].to_numpy(dtype=float))
    labels, info = context_groups(
        params["rlnMicrographName"],
        params["rlnHelicalTubeID"],
        params["rlnHelicalTrackLengthAngst"],
        params["rlnClassNumber"].astype(int),
        prob,
        classes,
        keys,
        list(range(2, int(n_max) + 1)),
        progress=progress,
        **kw,
    )
    splits, before = [], None
    for n in sorted(labels):
        ll = info["ll"][n]
        splits.append(
            dict(
                gain=0.0 if before is None else float(ll - before),
                low=0.0,
                high=1.0,
                markers=[],
                parent=0,
                labels=labels[n],
            )
        )
        before = ll
    return splits, info


def class_usage(chains, tpost):
    """The segments of each class in each type, weighted by the type posterior.

    Parameters
    ----------
    chains : Chains
    tpost : numpy.ndarray
        ``(N, T)`` posterior of the type of each segment.

    Returns
    -------
    numpy.ndarray
        ``(T, K)``.
    """
    use = (chains.cls >= 0) & ~chains.repeat
    return np.stack(
        [
            np.bincount(
                chains.cls[use], weights=tpost[use, t], minlength=chains.n_classes
            )
            for t in range(tpost.shape[1])
        ]
    )


def type_likeness(usage, similarity):
    """How alike the class averages of each pair of types look.

    The mean registered correlation between a class of one type and a class of the
    other, each weighted by its share of the type's segments (the average linkage
    of :func:`helical_pitch_types.pixel_groups`, between types). A class that two
    types share has a correlation of 1 with itself.

    Parameters
    ----------
    usage : numpy.ndarray
        ``(T, K)`` segments of each class in each type (:func:`class_usage`).
    similarity : numpy.ndarray
        ``(K, K)`` registered correlation of the class averages.

    Returns
    -------
    numpy.ndarray
        ``(T, T)`` symmetric.
    """
    w = usage / np.maximum(usage.sum(axis=1, keepdims=True), 1e-300)
    return w @ similarity @ w.T


def filament_coverage(chains, min_segments=20, min_share=0.1):
    """How much of the cycle of views a filament shows.

    Parameters
    ----------
    chains : Chains
    min_segments : int, optional
        Filaments with fewer segments are not counted. Defaults to 20.
    min_share : float, optional
        A class is shown by a filament if it has at least this share of its
        segments. Defaults to 0.1.

    Returns
    -------
    dict
        ``median_span`` and ``p90_span`` (A, along the filaments), ``median_classes``
        (classes a filament shows), ``n_classes`` (classes with at least 1% of all
        segments) and ``n_filaments`` counted.
    """
    use = chains.cls >= 0
    per = np.bincount(chains.filament, minlength=chains.n_filaments)
    # the span of the longest unbroken stretch of a filament
    span = np.zeros(chains.n_filaments)
    np.maximum.at(span, chains.chain_filament, chains.chain_len - 1.0)
    keep = per >= min_segments
    if not keep.any():
        return dict(
            median_span=0.0,
            p90_span=0.0,
            median_classes=0.0,
            n_classes=0,
            n_filaments=0,
        )
    table = np.zeros((chains.n_filaments, chains.n_classes))
    np.add.at(table, (chains.filament[use], chains.cls[use]), 1.0)
    share = table / np.maximum(table.sum(axis=1, keepdims=True), 1.0)
    shown = (share >= min_share).sum(axis=1)
    overall = table.sum(axis=0)
    main = int((overall >= 0.01 * overall.sum()).sum())
    spans = span[keep] * chains.step
    return dict(
        median_span=float(np.median(spans)),
        p90_span=float(np.quantile(spans, 0.9)),
        median_classes=float(np.median(shown[keep])),
        n_classes=main,
        n_filaments=int(keep.sum()),
    )


# ── periods, sections and the tables of types and filaments ───────────────────


@lazy_jit("njit", parallel=True, fastmath=True, nogil=True)
def _period_scores(x, ch, cl, phase0, n_classes, periods, rounds, out):
    """The agreement of the class sequences with a period, for each trial period.

    For a period ``C`` every segment has a phase ``2 pi x / C`` (``x`` its position
    along the filament), every class a phase of its own, and every filament one
    more (where it starts along the period). The class and filament phases are
    found by alternating (each the phase of the sum of the others' agreement),
    from the same random start for every ``C``; the score is the length of the mean
    agreement vector, 1 when every segment is at the phase of its class and filament.
    """
    n = x.shape[0]
    F = phase0.shape[0]
    for ic in prange(periods.shape[0]):
        C = periods[ic]
        e_re = np.empty(n)
        e_im = np.empty(n)
        for i in range(n):
            a = 2.0 * np.pi * x[i] / C
            e_re[i] = np.cos(a)
            e_im[i] = np.sin(a)
        p_re = np.cos(phase0)
        p_im = np.sin(phase0)
        t_re = np.ones(n_classes)
        t_im = np.zeros(n_classes)
        for it in range(rounds):
            z_re = np.zeros(n_classes)
            z_im = np.zeros(n_classes)
            for i in range(n):
                f = ch[i]
                k = cl[i]
                z_re[k] += e_re[i] * p_re[f] - e_im[i] * p_im[f]
                z_im[k] += e_re[i] * p_im[f] + e_im[i] * p_re[f]
            for k in range(n_classes):
                m = np.sqrt(z_re[k] ** 2 + z_im[k] ** 2)
                if m > 0.0:
                    t_re[k] = z_re[k] / m
                    t_im[k] = z_im[k] / m
                else:
                    t_re[k] = 1.0
                    t_im[k] = 0.0
            f_re = np.zeros(F)
            f_im = np.zeros(F)
            for i in range(n):
                f = ch[i]
                k = cl[i]
                f_re[f] += t_re[k] * e_re[i] + t_im[k] * e_im[i]
                f_im[f] += t_im[k] * e_re[i] - t_re[k] * e_im[i]
            for f in range(F):
                m = np.sqrt(f_re[f] ** 2 + f_im[f] ** 2)
                if m > 0.0:
                    p_re[f] = f_re[f] / m
                    p_im[f] = f_im[f] / m
                else:
                    p_re[f] = 1.0
                    p_im[f] = 0.0
        r_re = 0.0
        r_im = 0.0
        for i in range(n):
            f = ch[i]
            k = cl[i]
            # tk * conj(e * pf)
            q_re = e_re[i] * p_re[f] - e_im[i] * p_im[f]
            q_im = e_re[i] * p_im[f] + e_im[i] * p_re[f]
            r_re += t_re[k] * q_re + t_im[k] * q_im
            r_im += t_im[k] * q_re - t_re[k] * q_im
        out[ic] = np.sqrt(r_re**2 + r_im**2) / n


MAX_PERIOD_ANGST = 2000.0


def fit_periods(
    chains,
    seg_type,
    seg_prob,
    class_best,
    n_types,
    min_angst=100.0,
    max_angst=MAX_PERIOD_ANGST,
    step_angst=10.0,
    min_prob=0.9,
    min_per_chain=20,
    min_segments=500,
):
    """The period of the class sequence of each type.

    Along a helix of one type the class (the view) runs through its cycle with the
    distance along the filament: the crossover (or its half or a multiple of it).
    Over the confident segments of a type, in the chains with at least
    ``min_per_chain`` of them, the period is the one at which the classes and the
    filaments agree best (:func:`_period_scores`); the positions along a chain are
    those of the grid, in the direction of the molecule (see ``backward`` of
    :func:`build_chains`). As ``HelicalContext::fitPeriods`` of RELION.

    Parameters
    ----------
    chains : Chains
    seg_type, seg_prob : numpy.ndarray
        ``(N,)`` the type of each segment and its posterior (:func:`segment_types`).
    class_best : numpy.ndarray
        ``(N,)`` the class of the largest posterior of each segment (:func:`fit`).
    n_types : int
    min_angst, max_angst, step_angst : float, optional
        The periods tried, A. Default 100 to 2000 in steps of 10.
    min_prob, min_per_chain, min_segments : optional
        Segments are used if their type probability is at least ``min_prob``, in
        chains with at least ``min_per_chain`` such segments, and a type with fewer
        than ``min_segments`` in all has no period. Defaults 0.9, 20 and 500.

    Returns
    -------
    period, score : numpy.ndarray
        ``(n_types,)`` the period (A; 0 for none) and its score (0 to 1).
    """
    periods = np.arange(min_angst, max_angst + 1e-9, step_angst)
    period = np.zeros(n_types)
    score = np.zeros(n_types)
    for t in range(n_types):
        sure = (seg_type == t) & (seg_prob >= min_prob)
        per_chain = np.bincount(chains.chain[sure], minlength=chains.n_chains)
        use = sure & (per_chain[chains.chain] >= min_per_chain)
        if use.sum() < min_segments:
            continue
        _, ch = np.unique(chains.chain[use], return_inverse=True)
        phase0 = np.random.default_rng(1).uniform(0.0, 2.0 * np.pi, int(ch.max()) + 1)
        out = np.zeros(len(periods))
        _period_scores(
            (chains.pos[use] * chains.step).astype(np.float64),
            ch.astype(np.int64),
            class_best[use].astype(np.int64),
            phase0,
            chains.n_classes,
            periods,
            8,
            out,
        )
        k = int(np.argmax(out))
        period[t], score[t] = periods[k], out[k]
    return period, score


def segment_labels(seg_type, seg_prob, section, ctx_prob=None, chains=None):
    """The per-segment labels RELION writes, as a table in the order of the segments.

    Parameters
    ----------
    seg_type, seg_prob, section : numpy.ndarray
        From :func:`segment_types`.
    ctx_prob : numpy.ndarray, optional
        The context probability of the assigned class of each segment
        (:func:`fit` with ``prior_steps``).
    chains : Chains, optional
        With it, the sections are numbered as RELION does: along the filaments in
        the order of the input, and along each filament in the order of its
        segments in the input. Otherwise by the chains (longest first).

    Returns
    -------
    pandas.DataFrame
        ``rlnHelicalType`` and ``rlnHelicalSectionID`` (from 1),
        ``rlnHelicalTypeProbability`` and ``rlnHelicalContextProbability``.
    """
    import pandas as pd

    if chains is not None:
        n_sec = int(section.max()) + 1
        first = np.full(n_sec, len(section), dtype=np.int64)
        np.minimum.at(first, section, np.arange(len(section)))
        fil = np.zeros(n_sec, dtype=np.int64)
        fil[section] = chains.filament
        rank = np.empty(n_sec, dtype=np.int64)
        rank[np.lexsort((first, fil))] = np.arange(n_sec)
        section = rank[section]
    return pd.DataFrame(
        {
            "rlnHelicalType": seg_type.astype(np.int32) + 1,
            "rlnHelicalTypeProbability": np.asarray(seg_prob, dtype=np.float32),
            "rlnHelicalSectionID": section.astype(np.int64) + 1,
            "rlnHelicalContextProbability": (
                np.zeros(len(seg_type), dtype=np.float32)
                if ctx_prob is None
                else np.asarray(ctx_prob, dtype=np.float32)
            ),
        }
    )


def type_tables(
    chains,
    seg_type,
    seg_prob,
    section,
    tpost,
    model,
    class_numbers,
    period=None,
    score=None,
):
    """The tables of types and of filaments, as RELION writes them.

    Parameters
    ----------
    chains : Chains
    seg_type, seg_prob, section : numpy.ndarray
        From :func:`segment_types`.
    tpost : numpy.ndarray
        ``(N, T)`` the type posterior of each segment.
    model : Model
    class_numbers : sequence of int
        The class number of each class of the model.
    period, score : numpy.ndarray, optional
        From :func:`fit_periods`.

    Returns
    -------
    types : pandas.DataFrame
        One row per type: ``Type`` (from 1), ``NrSegments`` (the sum of the
        posteriors), ``NrFilaments`` and ``NrSections`` (those with a section of the
        type), ``MeanSectionAngst``, ``PeriodAngst``, ``PeriodScore`` and
        ``MainClasses`` (the classes that make up 90% of the type, most used first).
    filaments : pandas.DataFrame
        One row per filament: ``MicrographName``, ``HelicalTubeID``, ``NrSegments``,
        ``NrSections``, ``MeanTypeProbability`` and ``Sections`` (the runs along it,
        ``"type x segments"``).
    """
    import pandas as pd

    T = tpost.shape[1]
    order = np.lexsort((chains.pos, chains.chain))
    sec_o, typ_o = section[order], seg_type[order]
    pos_o, fil_o = chains.pos[order], chains.filament[order]
    first = np.r_[True, sec_o[1:] != sec_o[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(sec_o)] - 1
    sec_type = typ_o[starts]
    sec_fil = fil_o[starts]
    sec_len = (np.abs(pos_o[ends] - pos_o[starts]) + 1) * chains.step
    sec_n = ends - starts + 1
    rows = []
    for t in range(T):
        mine = sec_type == t
        order_q = np.argsort(-model.q[t], kind="stable")
        cum, main = 0.0, []
        for k in order_q:
            if cum >= 0.9:
                break
            cum += model.q[t, k]
            main.append(int(class_numbers[k]))
        rows.append(
            dict(
                Type=t + 1,
                NrSegments=float(tpost[:, t].sum()),
                NrFilaments=int(len(np.unique(sec_fil[mine]))),
                NrSections=int(mine.sum()),
                MeanSectionAngst=float(sec_len[mine].mean()) if mine.any() else 0.0,
                PeriodAngst=0.0 if period is None else float(period[t]),
                PeriodScore=0.0 if score is None else float(score[t]),
                MainClasses=",".join(str(c) for c in main),
            )
        )
    types = pd.DataFrame(rows)
    # the runs of sections along each filament, in order
    per_fil = pd.DataFrame(
        dict(fil=sec_fil, type=sec_type, n=sec_n, chain=chains.chain[order][starts])
    )
    fils = []
    mean_prob = np.bincount(
        chains.filament,
        weights=np.asarray(seg_prob, dtype=float),
        minlength=chains.n_filaments,
    ) / np.maximum(np.bincount(chains.filament, minlength=chains.n_filaments), 1)
    n_seg = np.bincount(chains.filament, minlength=chains.n_filaments)
    for f, grp in per_fil.groupby("fil", sort=True):
        micrograph, tube = str(chains.filament_key[f]).rsplit("|", 1)
        fils.append(
            dict(
                MicrographName=micrograph,
                HelicalTubeID=tube,
                NrSegments=int(n_seg[f]),
                NrSections=len(grp),
                MeanTypeProbability=float(mean_prob[f]),
                Sections=",".join(
                    f"{t + 1}x{n}" for t, n in zip(grp["type"], grp["n"])
                ),
            )
        )
    return types, pd.DataFrame(fils)
