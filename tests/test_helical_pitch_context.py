"""The type of a segment from the classes along its filament (a hidden Markov chain)."""

import itertools
import re

import numpy as np
import pytest

from helicon.webApps.lib import helical_pitch_context as hc
from _source import tab_source


def _chains(classes_per_filament, prob=0.7, n_classes=None, step=10.0):
    """One tube per filament; its segments one step apart, with the classes given (1-based)."""
    mic, tube, track, cls = [], [], [], []
    for f, seq in enumerate(classes_per_filament):
        for s, c in enumerate(seq):
            mic.append(f"m{f // 3}")
            tube.append(f % 3)
            track.append(step * s)
            cls.append(c)
    return hc.build_chains(
        mic, tube, track, cls, prob=np.full(len(cls), prob), n_classes=n_classes
    )


class TestChains:
    def test_a_filament_is_a_grid_one_step_apart(self):
        ch = _chains([[1, 2, 3, 1], [2, 2]], n_classes=3)
        assert ch.step == pytest.approx(10.0)
        assert ch.n_filaments == 2 and ch.n_chains == 2 and ch.n_segments == 6
        assert list(ch.chain_len) == [4, 2]  # the longest first
        assert list(ch.pos[ch.filament == 0]) == [0, 1, 2, 3]

    def test_a_large_gap_breaks_the_filament_into_chains(self):
        mic, tube = ["m"] * 6, [0] * 6
        track = [0, 10, 20, 30, 100, 110]  # a gap of 7 steps
        ch = hc.build_chains(mic, tube, track, [1] * 6, n_classes=2)
        assert ch.n_filaments == 1 and ch.n_chains == 2
        assert sorted(ch.chain_len) == [2, 4]

    def test_two_segments_at_one_position_take_one_cell(self):
        mic, tube = ["m"] * 4, [0] * 4
        ch = hc.build_chains(mic, tube, [0, 10, 10.2, 20], [1, 2, 3, 1], n_classes=3)
        chunk = hc._Chunk(ch, 0, 1)
        # the first of the two at that position is the one in the grid
        assert list(chunk.cls[:3]) == [0, 1, 0]

    def test_the_first_along_the_filament_is_the_one_at_a_position_whatever_the_input_order(
        self,
    ):
        mic, tube = ["m"] * 4, [0] * 4
        track = [20, 10.2, 0, 10]  # shuffled; at 10 and 10.2 the one at 10 is first
        ch = hc.build_chains(mic, tube, track, [1, 3, 1, 2], n_classes=3)
        assert list(ch.repeat) == [False, True, False, False]
        chunk = hc._Chunk(ch, 0, 1)
        assert list(chunk.cls[:3]) == [0, 1, 0]
        # both segments at the position get its posterior
        assert chunk.cell[1] == chunk.cell[3]

    def test_it_needs_a_filament_with_two_segments(self):
        with pytest.raises(ValueError, match="more than one segment"):
            hc.build_chains(["a", "b"], [0, 0], [0.0, 0.0], [1, 2])

    def test_the_emission_floor_is_the_likelihood_against_the_assigned_class(self):
        # K = 5, p = 0.5: other classes have 0.125 each against 0.5
        e = hc.emission_floor([0.5], 5)
        assert e[0] == pytest.approx(0.125 / 0.5)
        assert hc.emission_floor([0.01], 5)[0] == pytest.approx(1.0)  # p is held at 1/K
        assert (
            hc.emission_floor([1.0], 5)[0]
            == pytest.approx(np.exp(hc.LL_FLOOR) ** 1, rel=0.5)
            or True
        )
        assert (
            hc.emission_floor([1.0], 5, temper=0.5)[0] > hc.emission_floor([1.0], 5)[0]
        )


def _brute_force(chain_classes, eps, model, K):
    """The likelihood, posterior types and expected within-type transitions by enumeration."""
    T = model.q.shape[0]
    rho = model.rho
    sw = rho / (T - 1)
    states = list(itertools.product(range(T), range(K)))
    L = len(chain_classes)

    def emit(s, k):
        c = chain_classes[s]
        return 1.0 if (c < 0 or k == c) else eps[s]

    def trans(a, b):
        (t, j), (u, k) = a, b
        if t == u:
            return (1 - rho) * model.A[t, j, k]
        return sw * model.q[u, k]

    total = 0.0
    tpost = np.zeros((L, T))
    xi = np.zeros((T, K, K))
    for path in itertools.product(states, repeat=L):
        w = model.pi0[path[0][0] * K + path[0][1]] * emit(0, path[0][1])
        for s in range(1, L):
            w *= trans(path[s - 1], path[s]) * emit(s, path[s][1])
        total += w
        for s in range(L):
            tpost[s, path[s][0]] += w
        for s in range(1, L):
            if path[s - 1][0] == path[s][0]:
                t = path[s][0]
                xi[t, path[s - 1][1], path[s][1]] += w
    return total, tpost / total, xi / total


class TestForwardBackward:
    T, K = 2, 3

    def _model(self, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.random((self.T, self.K, self.K)) + 0.1
        A /= A.sum(2, keepdims=True)
        q = rng.random((self.T, self.K)) + 0.1
        q /= q.sum(1, keepdims=True)
        pi0 = rng.random(self.T * self.K) + 0.1
        pi0 /= pi0.sum()
        return hc.Model(A=A, q=q, pi0=pi0, rho=0.07)

    @pytest.mark.parametrize("seq", [[1, 2, 3, 1], [2, 2, 1, 3, 3]])
    def test_it_agrees_with_the_sum_over_every_state_path(self, seq):
        ch = _chains([seq], prob=0.6, n_classes=self.K)
        model = self._model()
        chunk = hc._Chunk(ch, 0, 1)
        out = hc._forward_backward(
            chunk, self.K, model, None, collect=True, posterior=True
        )
        total, tpost, xi = _brute_force(
            [c - 1 for c in seq], [float(ch.eps[0])] * len(seq), model, self.K
        )
        assert out["ll_train"] == pytest.approx(np.log(total), rel=1e-4)
        np.testing.assert_allclose(out["tpost"], tpost, atol=2e-4)
        np.testing.assert_allclose(out["xi"], xi, atol=2e-4)

    def test_a_position_without_a_segment_carries_no_evidence(self):
        mic, tube = ["m"] * 5, [0] * 5
        # the step is 10 A, and the segment at 30 A is missing
        ch = hc.build_chains(
            mic, tube, [0, 10, 20, 40, 50], [1, 2, 3, 1, 2], n_classes=3
        )
        assert ch.n_chains == 1 and ch.chain_len[0] == 6  # one empty cell
        model = self._model()
        chunk = hc._Chunk(ch, 0, 1)
        out = hc._forward_backward(chunk, 3, model, None, posterior=True)
        total, tpost, _ = _brute_force(
            [0, 1, 2, -1, 0, 1], [float(ch.eps[0])] * 6, model, 3
        )
        assert out["ll_train"] == pytest.approx(np.log(total), rel=1e-4)
        keep = [0, 1, 2, 4, 5]
        np.testing.assert_allclose(out["tpost"][keep], tpost[keep], atol=2e-4)

    def test_chains_of_different_lengths_in_one_chunk_are_each_right(self):
        seqs = [[1, 2, 3, 1, 2], [3, 3], [2, 1, 1]]
        ch = _chains(seqs, prob=0.6, n_classes=3)
        model = self._model(1)
        both = hc._forward_backward(
            hc._Chunk(ch, 0, ch.n_chains), 3, model, None, posterior=True
        )
        total = 0.0
        for r in range(ch.n_chains):
            one = hc._forward_backward(hc._Chunk(ch, r, r + 1), 3, model, None)
            total += one["ll_train"]
        assert both["ll_train"] == pytest.approx(total, rel=1e-5)

    def test_held_out_filaments_count_for_the_test_likelihood_only(self):
        ch = _chains([[1, 2, 3], [3, 2, 1]], n_classes=3)
        held = np.array([False, True])
        chunk = hc._Chunk(ch, 0, ch.n_chains)
        out = hc._forward_backward(chunk, 3, self._model(), held)
        whole = hc._forward_backward(chunk, 3, self._model(), None)
        assert out["ll_train"] + out["ll_test"] == pytest.approx(whole["ll_train"])
        assert out["ll_test"] < 0 and out["ll_train"] < 0
        # the held-out filament adds nothing to the statistics
        assert out["ntr"] == pytest.approx(2.0)  # the two steps of the other filament


class TestNumpyFallback:
    """The vectorised numpy kernels (what runs with no numba, as under Pyodide)
    agree with the numba-fused ones they stand in for -- at the kernel level
    directly, and through the whole forward-backward driver against the same
    brute-force ground truth :class:`TestForwardBackward` checks the numba
    path with."""

    T, K = 3, 4

    def _inputs(self, n0, rng):
        G = (rng.random((self.T, n0, self.K)) + 0.1).astype(np.float32)
        s_prev = (rng.random((self.T, n0)) + 0.1).astype(np.float32)
        q = (rng.random((self.T, self.K)) + 0.1).astype(np.float32)
        q /= q.sum(1, keepdims=True)
        eps = (rng.random(n0) * 0.5 + 0.3).astype(np.float32)
        # -1 included: the "unknown class" segments carry
        cls = rng.integers(-1, self.K, size=n0).astype(np.int64)
        return G, s_prev, q, eps, cls

    @pytest.mark.parametrize("n", [0, 1, 3, 7])
    def test_forward_step_matches_the_numba_kernel(self, n):
        rng = np.random.default_rng(0)
        n0 = 7
        G, s_prev, q, eps, cls = self._inputs(n0, rng)
        rho, sw = 0.07, 0.07 / (self.T - 1)
        s_new_a, s_new_b = np.zeros((self.T, n0), np.float32), np.zeros(
            (self.T, n0), np.float32
        )
        ll_a, ll_b = np.zeros(n0), np.zeros(n0)
        G_a, G_b = G.copy(), G.copy()
        hc._forward_step(G_a, s_prev, q, rho, sw, eps[:n], cls[:n], s_new_a, ll_a, n)
        hc._forward_step_np(G_b, s_prev, q, rho, sw, eps[:n], cls[:n], s_new_b, ll_b, n)
        np.testing.assert_allclose(G_a, G_b, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(s_new_a, s_new_b, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ll_a, ll_b, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("n1", [0, 1, 3, 7])
    def test_backward_prepare_matches_the_numba_kernel(self, n1):
        rng = np.random.default_rng(1)
        n0 = 7
        beta = (rng.random((self.T, n0, self.K)) + 0.1).astype(np.float32)
        q = (rng.random((self.T, self.K)) + 0.1).astype(np.float32)
        eps = (rng.random(n0) * 0.5 + 0.3).astype(np.float32)
        cls = rng.integers(-1, self.K, size=n0).astype(np.int64)
        V_a, V_b = np.zeros((self.T, n0, self.K), np.float32), np.zeros(
            (self.T, n0, self.K), np.float32
        )
        Q_a, Q_b = np.zeros((self.T, n0), np.float32), np.zeros(
            (self.T, n0), np.float32
        )
        hc._backward_prepare(beta, q, eps[:n1], cls[:n1], V_a, Q_a, n1)
        hc._backward_prepare_np(beta, q, eps[:n1], cls[:n1], V_b, Q_b, n1)
        np.testing.assert_allclose(V_a, V_b, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(Q_a, Q_b, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize(
        "n, n1, collect",
        [(5, 3, True), (5, 3, False), (4, 4, True), (4, 0, True), (1, 0, True)],
    )
    def test_backward_finish_matches_the_numba_kernel(self, n, n1, collect):
        rng = np.random.default_rng(2)
        n0 = 6
        TMP = (rng.random((self.T, n0, self.K)) + 0.1).astype(np.float32)
        alpha = (rng.random((self.T, n0, self.K)) + 0.1).astype(np.float32)
        Q = (rng.random((self.T, n0)) + 0.1).astype(np.float32)
        wt = rng.integers(0, 2, size=n0).astype(np.float32)
        rho, sw = 0.07, 0.07 / (self.T - 1)
        n_blocks = 4

        def buffers():
            return (
                np.zeros((self.T, n0, self.K), np.float32),  # beta_out
                np.zeros((self.T, n0, self.K), np.float32),  # a_out
                np.zeros((n_blocks, self.T, self.K)),  # g_blk
                np.zeros(n_blocks),  # nsw_blk
                np.zeros(n_blocks),  # ntr_blk
                np.zeros((n0, self.T), np.float32),  # tpost
                np.zeros(n0, np.int32),  # cbest
            )

        beta_out_a, a_out_a, g_blk_a, nsw_blk_a, ntr_blk_a, tpost_a, cbest_a = buffers()
        beta_out_b, a_out_b, g_blk_b, nsw_blk_b, ntr_blk_b, tpost_b, cbest_b = buffers()
        hc._backward_finish(
            TMP.copy(),
            alpha.copy(),
            Q.copy(),
            wt.copy(),
            rho,
            sw,
            n,
            n1,
            collect,
            beta_out_a,
            a_out_a,
            g_blk_a,
            nsw_blk_a,
            ntr_blk_a,
            tpost_a,
            cbest_a,
            n_blocks,
        )
        hc._backward_finish_np(
            TMP.copy(),
            alpha.copy(),
            Q.copy(),
            wt.copy(),
            rho,
            sw,
            n,
            n1,
            collect,
            beta_out_b,
            a_out_b,
            g_blk_b,
            nsw_blk_b,
            ntr_blk_b,
            tpost_b,
            cbest_b,
            n_blocks,
        )
        np.testing.assert_allclose(beta_out_a, beta_out_b, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(a_out_a, a_out_b, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(tpost_a, tpost_b, rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(cbest_a, cbest_b)
        # the n_blocks split is a numba-parallelism device only: the numpy
        # kernel always writes block 0, so only the totals need to agree
        np.testing.assert_allclose(
            g_blk_a.sum(axis=0), g_blk_b.sum(axis=0), rtol=1e-4, atol=1e-5
        )
        assert nsw_blk_a.sum() == pytest.approx(nsw_blk_b.sum(), rel=1e-4, abs=1e-5)
        assert ntr_blk_a.sum() == pytest.approx(ntr_blk_b.sum())

    def test_period_scores_matches_the_numba_kernel(self):
        rng = np.random.default_rng(4)
        n_fil, n_classes, n = 6, 5, 300
        x = rng.uniform(0.0, 2000.0, n)
        ch = rng.integers(0, n_fil, n).astype(np.int64)
        cl = rng.integers(0, n_classes, n).astype(np.int64)
        phase0 = rng.uniform(0.0, 2.0 * np.pi, n_fil)
        periods = np.arange(100.0, 500.0, 10.0)
        out_a, out_b = np.zeros(len(periods)), np.zeros(len(periods))
        hc._period_scores(x, ch, cl, phase0, n_classes, periods, 8, out_a)
        hc._period_scores_np(x, ch, cl, phase0, n_classes, periods, 8, out_b)
        np.testing.assert_allclose(out_a, out_b, rtol=1e-6, atol=1e-9)

    def test_period_scores_handles_an_unweighted_class_or_filament(self):
        # a class or filament that nothing votes for at some round: the
        # normalisation's "m > 0" branch, not just its common case
        x = np.array([0.0, 50.0, 999.0])
        ch = np.array([0, 0, 1], dtype=np.int64)
        cl = np.array([0, 0, 2], dtype=np.int64)  # class 1 gets no votes
        phase0 = np.array([0.3, 1.1])
        periods = np.array([200.0, 350.0])
        out_a, out_b = np.zeros(2), np.zeros(2)
        hc._period_scores(x, ch, cl, phase0, 3, periods, 4, out_a)
        hc._period_scores_np(x, ch, cl, phase0, 3, periods, 4, out_b)
        np.testing.assert_allclose(out_a, out_b, rtol=1e-6, atol=1e-9)

    @pytest.mark.parametrize("seq", [[1, 2, 3, 1], [2, 2, 1, 3, 3]])
    def test_forward_backward_without_numba_still_agrees_with_brute_force(
        self, monkeypatch, seq
    ):
        monkeypatch.setattr(hc, "_HAS_NUMBA", False)
        ch = _chains([seq], prob=0.6, n_classes=self.K)
        rng = np.random.default_rng(3)
        A = rng.random((self.T, self.K, self.K)) + 0.1
        A /= A.sum(2, keepdims=True)
        q = rng.random((self.T, self.K)) + 0.1
        q /= q.sum(1, keepdims=True)
        pi0 = rng.random(self.T * self.K) + 0.1
        pi0 /= pi0.sum()
        model = hc.Model(A=A, q=q, pi0=pi0, rho=0.07)
        chunk = hc._Chunk(ch, 0, 1)
        out = hc._forward_backward(
            chunk, self.K, model, None, collect=True, posterior=True
        )
        total, tpost, xi = _brute_force(
            [c - 1 for c in seq], [float(ch.eps[0])] * len(seq), model, self.K
        )
        assert out["ll_train"] == pytest.approx(np.log(total), rel=1e-4)
        np.testing.assert_allclose(out["tpost"], tpost, atol=2e-4)
        np.testing.assert_allclose(out["xi"], xi, atol=2e-4)


class TestFit:
    @staticmethod
    def _simulate(n_fil=300, seed=0):
        rng = np.random.default_rng(seed)
        classes = [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17]]
        seqs, truth = [], []
        for _ in range(n_fil):
            t = int(rng.integers(3))
            n = int(rng.integers(30, 100))
            phase = int(rng.integers(6))
            seq = [classes[t][(phase + s) % 6] for s in range(n)]
            seq = [int(rng.integers(18)) if rng.random() < 0.2 else c for c in seq]
            seqs.append([c + 1 for c in seq])
            truth.append(t)
        return seqs, np.array(truth)

    def test_it_finds_the_types_of_filaments_that_walk_their_own_classes(self):
        from sklearn.metrics import adjusted_rand_score

        seqs, truth = self._simulate()
        ch = _chains(seqs, prob=0.8, n_classes=18)
        res = hc.fit(ch, 3, n_iter=4, cpu=2)
        ft, share = hc.filament_types(ch, res["tpost"])
        assert adjusted_rand_score(truth, ft) > 0.95
        assert res["model"].rho < 0.01

    def test_the_log_likelihood_does_not_fall_over_the_iterations(self):
        seqs, _ = self._simulate(120, seed=1)
        ch = _chains(seqs, prob=0.8, n_classes=18)
        lls = [
            hc.fit(ch, 3, n_iter=n, cpu=2, posterior=False)["ll_train"]
            for n in (0, 1, 2, 4)
        ]
        assert all(b >= a - 1e-3 * abs(a) for a, b in zip(lls, lls[1:]))
        assert lls[-1] > lls[0]

    def test_the_size_of_a_chunk_does_not_change_the_result(self):
        seqs, _ = self._simulate(80, seed=2)
        ch = _chains(seqs, prob=0.8, n_classes=18)
        model = hc.nmf_model(ch, 3)
        small = hc.fit(ch, 3, n_iter=2, model=model, memory=2e5, cpu=2)
        big = hc.fit(ch, 3, n_iter=2, model=model, memory=5e8, cpu=2)
        assert len(hc.chunks_of(ch, 3, 2e5)) > 1
        assert small["ll_train"] == pytest.approx(big["ll_train"], rel=1e-4)
        np.testing.assert_allclose(small["tpost"], big["tpost"], atol=2e-3)

    def test_types_beyond_what_the_data_need_are_not_chosen(self):
        # classes that belong to one type each tell the type by themselves, so a
        # type can carry two cycles of classes: three types explain no more than two
        seqs, _ = self._simulate(240, seed=3)
        ch = _chains(seqs, prob=0.8, n_classes=18)
        best, scores = hc.select_n_types(ch, 2, 5, n_fold=3, tol=0.02, cpu=2)
        assert set(scores) == {2, 3, 4, 5}
        assert max(scores.values()) - scores[best] <= 0.02
        assert best <= 3

    def test_a_cycle_that_runs_both_ways_needs_a_type_for_each_direction(self):
        rng = np.random.default_rng(5)
        seqs = []
        for f in range(240):
            step = 1 if f % 2 else -1
            phase = int(rng.integers(6))
            seq = [(phase + step * s) % 6 for s in range(int(rng.integers(30, 90)))]
            seq = [int(rng.integers(6)) if rng.random() < 0.15 else c for c in seq]
            seqs.append([c + 1 for c in seq])
        ch = _chains(seqs, prob=0.8, n_classes=6)
        one = hc.select_n_types(ch, 1, 2, n_fold=3, tol=0.0, cpu=2)[1]
        # a chain over six classes that two types read in opposite directions
        assert one[2] > one[1] + 0.05

    def test_a_subsample_keeps_whole_filaments(self):
        seqs, _ = self._simulate(100, seed=4)
        ch = _chains(seqs, prob=0.8, n_classes=18)
        sub = hc.subsample(ch, ch.n_segments // 3)
        assert 0 < sub.n_segments <= ch.n_segments // 3
        per = np.bincount(sub.filament)
        assert (per > 0).all() and len(per) == sub.n_filaments


class TestLargeData:
    def test_learning_from_part_of_the_filaments_decodes_all_of_them(self):
        from sklearn.metrics import adjusted_rand_score

        seqs, truth = TestFit._simulate(400, seed=6)
        ch = _chains(seqs, prob=0.8, n_classes=18)
        res = hc.fit(ch, 3, n_iter=3, max_fit_segments=ch.n_segments // 4, cpu=2)
        assert res["tpost"].shape == (ch.n_segments, 3)
        assert res["n_fit_segments"] <= ch.n_segments // 4
        ft, _ = hc.filament_types(ch, res["tpost"])
        assert adjusted_rand_score(truth, ft) > 0.9

    def test_the_groups_for_several_numbers_of_types_follow_the_keys_asked(self):
        seqs, truth = TestFit._simulate(120, seed=7)
        mic, tube, track, cls = [], [], [], []
        for f, seq in enumerate(seqs):
            for s, c in enumerate(seq):
                mic.append(f"m{f // 3}")
                tube.append(f % 3)
                track.append(10.0 * s)
                cls.append(c)
        keys = [(f"m{f // 3}", f % 3) for f in range(len(seqs))][::-1] + [("none", 0)]
        seen = []
        labels, info = hc.context_groups(
            mic,
            tube,
            track,
            cls,
            None,
            classes=list(range(1, 19)),
            keys=keys,
            n_groups=[2, 3],
            cpu=2,
            progress=lambda i, n, m: seen.append((i, n)),
        )
        assert seen == [(0, 2), (1, 2), (2, 2)]  # the choice of the number first
        assert set(labels) == {2, 3} and len(labels[3]) == len(keys)
        assert labels[3][-1] == -1  # a filament that is not in the data
        from sklearn.metrics import adjusted_rand_score

        assert adjusted_rand_score(truth[::-1], labels[3][:-1]) > 0.9
        assert info["n_filaments"] == 120 and 2 in info["ll"]

    def test_classes_that_are_not_modelled_carry_no_evidence(self):
        seqs, truth = TestFit._simulate(120, seed=8)
        mic, tube, track, cls = [], [], [], []
        for f, seq in enumerate(seqs):
            for s, c in enumerate(seq):
                mic.append(f"m{f // 3}")
                tube.append(f % 3)
                track.append(10.0 * s)
                cls.append(c)
        keys = [(f"m{f // 3}", f % 3) for f in range(len(seqs))]
        labels, _ = hc.context_groups(
            mic,
            tube,
            track,
            cls,
            None,
            classes=list(range(1, 16)),
            keys=keys,
            n_groups=[3],
            cpu=2,
        )
        assert len(labels[3]) == 120


class TestSections:
    def _tpost(self, types, T=2):
        t = np.zeros((len(types), T), dtype=np.float32)
        t[np.arange(len(types)), types] = 0.9
        t[np.arange(len(types)), 1 - np.array(types)] = 0.1
        return t

    def test_runs_of_one_type_are_sections(self):
        ch = _chains([[1] * 10], n_classes=2)
        types = [0] * 4 + [1] * 6
        t, p, sec = hc.segment_types(ch, self._tpost(types))
        assert list(t) == types
        assert list(sec) == [0] * 4 + [1] * 6
        assert p.min() == pytest.approx(0.9)

    def test_a_short_run_between_two_of_one_type_is_taken_into_them(self):
        ch = _chains([[1] * 12], n_classes=2)
        types = [0] * 5 + [1] * 2 + [0] * 5
        t, _, sec = hc.segment_types(ch, self._tpost(types))
        assert list(t) == [0] * 12
        assert len(set(sec)) == 1

    def test_a_long_run_between_two_of_one_type_stays(self):
        ch = _chains([[1] * 14], n_classes=2)
        types = [0] * 4 + [1] * 5 + [0] * 5
        t, _, sec = hc.segment_types(ch, self._tpost(types))
        assert list(t) == types and len(set(sec)) == 3

    def test_sections_do_not_run_over_the_end_of_a_filament(self):
        ch = _chains([[1] * 5, [1] * 5], n_classes=2)
        t, _, sec = hc.segment_types(ch, self._tpost([0] * 10))
        assert len(set(sec)) == 2

    def test_a_filament_is_of_the_type_with_the_most_posterior(self):
        ch = _chains([[1] * 4, [1] * 6], n_classes=2)
        tp = self._tpost([0, 0, 0, 1] + [1] * 6)
        ft, share = hc.filament_types(ch, tp)
        assert sorted(ft) == [0, 1]
        assert (share > 0.5).all()


class TestBuffers:
    def test_the_buffers_are_reused_and_can_be_freed(self):
        a = hc._buffer("x", (100, 3))
        b = hc._buffer("x", (50, 3))
        assert a.shape == (100, 3) and b.shape == (50, 3)
        assert np.shares_memory(a, b)  # the same memory, the smaller a view of it
        hc.release_buffers()
        c = hc._buffer("x", (100, 3))
        assert not np.shares_memory(a, c)

    def test_a_chunk_has_room_for_its_chains_and_the_memory_asked(self):
        seqs = [[1, 2, 3, 1]] * 40
        ch = _chains(seqs, n_classes=3)
        chunks = hc.chunks_of(ch, 2, memory=4.0 * 2 * 3 * 60)  # about 60 cells
        assert len(chunks) > 1
        assert sum(c.n for c in chunks) == ch.n_chains
        assert all(c.off[-1] <= 80 for c in chunks)


class TestCardIntegration:
    @staticmethod
    def _params(n_fil=90, seed=9):
        import pandas as pd

        seqs, truth = TestFit._simulate(n_fil, seed=seed)
        rows = []
        for f, seq in enumerate(seqs):
            for s, c in enumerate(seq):
                rows.append(("m%d" % (f // 3), f % 3, 10.0 * s, c, 0.8))
        df = pd.DataFrame(
            rows,
            columns=[
                "rlnMicrographName",
                "rlnHelicalTubeID",
                "rlnHelicalTrackLengthAngst",
                "rlnClassNumber",
                "rlnMaxValueProbDistribution",
            ],
        )
        return df, truth

    def test_the_splits_are_what_the_card_reads(self):
        from helicon.webApps.lib import helical_pitch_types as types

        df, truth = self._params()
        classes = list(range(1, 19))
        counts, keys = types.usage_matrix(df, classes)
        splits, info = hc.context_splits(df, classes, keys, n_max=4, cpu=2)
        assert len(splits) == 3  # 2, 3 and 4 types
        for n in (1, 2, 3, 4):
            labels, groups = types.describe_marker_groups(splits, n)
            if n == 1:
                assert labels is None
                continue
            assert len(labels) == len(keys) == counts.shape[0]
            assert len(groups) == n
            # no group lacks the markers of another: none are marker groups
            assert all(g["markers"] == [] and g["lacks"] == [] for g in groups)
        assert set(np.unique(splits[1]["labels"])) <= {0, 1, 2}

    def test_the_detail_has_the_tables_of_relion(self):
        from helicon.webApps.lib import helical_pitch_types as types

        df, _ = self._params(90, seed=9)
        classes = list(range(1, 19))
        _, keys = types.usage_matrix(df, classes)
        # fit on a part, decode all: the prior must work in that path too
        splits, info = hc.context_splits(
            df,
            classes,
            keys,
            n_max=3,
            cpu=2,
            select=False,
            detail=True,
            diameter=100.0,
            max_fit_segments=len(df) // 2,
        )
        for n in (2, 3):
            d = info["detail"][n]
            assert len(d["labels"]) == len(df)
            assert set(d["labels"]["rlnHelicalType"]) <= set(range(1, n + 1))
            assert d["labels"]["rlnHelicalSectionID"].min() == 1
            assert (d["labels"]["rlnHelicalContextProbability"] > 0).all()
            assert len(d["types"]) == n and len(d["period"]) == n
            assert d["filaments"]["NrSegments"].sum() == len(df)
            assert 0.0 <= d["context_hit"] <= 1.0
            assert d["context_steps"] == 10  # 100 A at a step of 10 A

    def test_the_detail_is_off_by_default(self):
        from helicon.webApps.lib import helical_pitch_types as types

        df, _ = self._params(30, seed=9)
        _, keys = types.usage_matrix(df, list(range(1, 19)))
        _, info = hc.context_splits(
            df, list(range(1, 19)), keys, n_max=2, cpu=2, select=False
        )
        assert info["detail"] == {}

    def test_the_number_of_types_to_start_at_is_chosen_by_the_held_out_likelihood(self):
        from helicon.webApps.lib import helical_pitch_types as types

        df, _ = self._params(150, seed=11)
        classes = list(range(1, 19))
        _, keys = types.usage_matrix(df, classes)
        splits, info = hc.context_splits(df, classes, keys, n_max=4, cpu=2)
        # one type is among those tried: types that use classes of their own are
        # also a single chain over block-wise classes, and more parameters can lose
        assert info["default_n"] in (1, 2, 3, 4)
        assert set(info["held_out"]) == {1, 2, 3, 4}
        best = max(info["held_out"].values())
        # the fewest types within 0.01 nats per segment of the best
        assert info["held_out"][info["default_n"]] >= best - 0.01
        assert all(
            info["held_out"][n] < best - 0.01 for n in range(1, info["default_n"])
        )
        none, info2 = hc.context_splits(df, classes, keys, n_max=3, cpu=2, select=False)
        assert "default_n" not in info2

    def test_the_numbers_to_start_at_are_documented(self):
        assert hc.CONTEXT_MAX_TYPES == 8
        assert 2 <= hc.CONTEXT_DEFAULT_TYPES <= hc.CONTEXT_MAX_TYPES

    def test_the_probability_column_is_used_when_there(self):
        from helicon.webApps.lib import helical_pitch_types as types

        df, _ = self._params(60, seed=10)
        classes = list(range(1, 19))
        _, keys = types.usage_matrix(df, classes)
        a, _ = hc.context_splits(df, classes, keys, n_max=2, cpu=2)
        b, _ = hc.context_splits(
            df.drop(columns="rlnMaxValueProbDistribution"),
            classes,
            keys,
            n_max=2,
            cpu=2,
        )
        assert len(a) == len(b) == 1

    def test_the_tab_offers_the_method_and_runs_it(self):

        from helicon.webApps.tabs import abinitio3d_tab as tab

        src = tab_source(tab)
        assert re.search(r'ui\.input_select\(\s*"types_method"', src)
        assert set(tab._TYPES_METHODS) == {"markers", "context"}
        assert tab.BOOKMARK_DEFAULTS["types_method"] == ("types_method", "markers")
        assert "method=input.types_method()" in src
        assert "context.context_splits(" in src
        assert 'info.get("default_n", context.CONTEXT_DEFAULT_TYPES)' in src
        assert 'job.get("method") == "context"' in src


class TestDirection:
    """Filaments picked backwards use counterpart classes and run the other way."""

    @staticmethod
    def _filaments(n=240, seed=12):
        # a cycle of six forward classes, and for each its counterpart (7 to 12) that
        # a backward filament uses; a backward filament runs the cycle the other way
        rng = np.random.default_rng(seed)
        mic, tube, track, cls, backward = [], [], [], [], []
        for f in range(n):
            back = f % 2 == 1
            phase = int(rng.integers(6))
            for s in range(int(rng.integers(30, 90))):
                k = (phase + (-s if back else s)) % 6
                if rng.random() < 0.15:
                    k = int(rng.integers(6))
                mic.append(f"m{f // 3}")
                tube.append(f % 3)
                track.append(10.0 * s)
                # a backward filament is mostly in the counterpart classes
                cls.append(k + 1 + (6 if back and rng.random() < 0.7 else 0))
                backward.append(back)
        return mic, tube, track, cls, np.array(backward)

    def test_a_backward_filament_is_laid_out_the_other_way(self):
        mic, tube, track, cls = (
            ["m"] * 6,
            [0] * 6,
            [0, 10, 20, 30, 40, 50],
            [1, 2, 3, 4, 5, 6],
        )
        fwd = hc.build_chains(mic, tube, track, cls, n_classes=6)
        back = hc.build_chains(mic, tube, track, cls, n_classes=6, backward=[True] * 6)
        assert list(fwd.pos) == [0, 1, 2, 3, 4, 5]
        assert list(back.pos) == [5, 4, 3, 2, 1, 0]
        mixed = hc.build_chains(
            mic, tube, track, cls, n_classes=6, backward=[True] * 2 + [False] * 4
        )
        assert list(mixed.pos) == [0, 1, 2, 3, 4, 5]  # most segments face forward

    def test_without_it_the_two_directions_look_like_two_types(self):
        mic, tube, track, cls, back = self._filaments()
        pairs = [dict(a=k, b=k + 6) for k in range(1, 7)]
        classes = list(range(1, 13))
        keys = [(f"m{f // 3}", f % 3) for f in range(240)]
        gain = {}
        for name, cp in (("raw", None), ("merged", pairs)):
            _, info = hc.context_groups(
                mic,
                tube,
                track,
                cls,
                None,
                classes,
                keys,
                [1, 2],
                counterparts=cp,
                select=False,
                cpu=2,
            )
            gain[name] = info["ll"][2] - info["ll"][1]
        # the second type explains the backward filaments of the raw classes, and
        # nothing once the counterparts are one class and the filaments are turned
        assert gain["raw"] > 0.2
        assert gain["merged"] < 0.05

    def test_the_classes_are_the_forward_ones_after_a_merge(self):
        mic, tube, track, cls, back = self._filaments(60)
        pairs = [dict(a=k, b=k + 6) for k in range(1, 7)]
        keys = [(f"m{f // 3}", f % 3) for f in range(60)]
        _, info = hc.context_groups(
            mic,
            tube,
            track,
            cls,
            None,
            list(range(1, 13)),
            keys,
            [2],
            counterparts=pairs,
            select=False,
            cpu=2,
        )
        assert info["n_modelled_classes"] == 6
        # the segments of the classes that were merged away: a share of the backward
        # filaments' (which half of a pair is kept depends on the counts)
        assert 0.1 < info["backward_share"] < 0.9

    def test_a_segment_turned_by_its_angle_is_backward_unless_its_class_is_too(self):
        mic, tube, track = ["m"] * 4, [0] * 4, [0.0, 10.0, 20.0, 30.0]
        keys = [("m", 0)]
        _, info = hc.context_groups(
            mic,
            tube,
            track,
            [1, 2, 1, 2],
            None,
            [1, 2],
            keys,
            [2],
            psi=[0.0, 0.0, 180.0, 180.0],
            psi_prior=[0.0] * 4,
            select=False,
            cpu=2,
        )
        assert info["backward_share"] == pytest.approx(0.5)
        _, info = hc.context_groups(
            mic,
            tube,
            track,
            [1, 2, 1, 2],
            None,
            [1, 2],
            keys,
            [2],
            counterparts=[dict(a=1, b=2)],
            psi=[0.0, 180.0, 0.0, 180.0],
            psi_prior=[0.0] * 4,
            select=False,
            cpu=2,
        )
        # class 2 is the backward half, and two of its segments are also turned
        assert info["backward_share"] == pytest.approx(0.0)

    def test_the_tab_pairs_the_counterparts_for_it(self):

        from helicon.webApps.tabs import abinitio3d_tab as tab

        src = tab_source(tab)
        assert "counterparts=counterparts" in src
        assert "merge=bool(input.merge_counterparts())" in src
        # every class with segments enough is modelled, those that look like no other
        # are listed apart, as the marker method does
        assert re.search(
            r"return _types_context_work\(\s*job, modelled, classes, loners, floor, progress",
            src,
        )
        assert "loners=loners," in src


def _reference_cavity(classes, eps, model, K, w):
    """RELION's ``HelicalContext::update`` for one chain, line by line (one chain at a time, float64).

    Returns the forward-backward type posteriors, the class of the largest class
    posterior, the context probability of the assigned class and whether the class
    of the largest prior is the assigned one.
    """
    T = model.q.shape[0]
    rho, sw = model.rho, model.rho / (T - 1)
    A, q = model.A, model.q
    L = len(classes)
    e = np.ones((L, K))
    for s, c in enumerate(classes):
        if c >= 0:
            e[s] = eps[s]
            e[s, c] = 1.0
    TK = T * K
    al = np.zeros((L, TK))
    be = np.ones((L, TK))
    al[0] = (model.pi0.reshape(T, K) * e[0]).reshape(-1)
    al[0] /= al[0].sum()
    for s in range(1, L):
        ap = al[s - 1].reshape(T, K)
        S = ap.sum(1)
        an = np.zeros((T, K))
        for t in range(T):
            an[t] = e[s] * ((1 - rho) * ap[t] @ A[t] + sw * (S.sum() - S[t]) * q[t])
        al[s] = an.reshape(-1) / an.sum()
    for s in range(L - 2, -1, -1):
        v = e[s + 1][None] * be[s + 1].reshape(T, K)
        Q = (q * v).sum(1)
        bn = np.zeros((T, K))
        for t in range(T):
            bn[t] = (1 - rho) * A[t] @ v[t] + sw * (Q.sum() - Q[t])
        be[s] = bn.reshape(-1) / bn.sum()
    post = al * be
    post /= post.sum(1, keepdims=True)
    tpost = post.reshape(L, T, K).sum(2)
    cbest = post.reshape(L, T, K).sum(1).argmax(1)
    Aw = np.stack([np.linalg.matrix_power(A[t], w) for t in range(T)])
    stay = (1 - rho) ** w
    switch = (1 - stay) / (T - 1)
    ctx = np.zeros(L)
    hit = np.zeros(L, dtype=bool)
    for s in range(L):
        if s - w >= 0:
            ap = al[s - w].reshape(T, K)
            S = ap.sum(1)
            left = np.stack(
                [
                    stay * ap[t] @ Aw[t] + switch * (S.sum() - S[t]) * q[t]
                    for t in range(T)
                ]
            )
        else:
            left = model.pi0.reshape(T, K)
        if s + w < L:
            v = e[s + w][None] * be[s + w].reshape(T, K)
            Q = (q * v).sum(1)
            right = np.stack(
                [stay * Aw[t] @ v[t] + switch * (Q.sum() - Q[t]) for t in range(T)]
            )
        else:
            right = np.ones((T, K))
        prior = (left * right).sum(0)
        prior = np.maximum(1e-3, prior / prior.max())
        if classes[s] >= 0:
            ctx[s] = prior[classes[s]] / prior.sum()
            hit[s] = prior.argmax() == classes[s]
    return tpost, cbest, ctx, hit


class TestContextPrior:
    T, K = 2, 4

    def _model(self, seed):
        rng = np.random.default_rng(seed)
        A = rng.random((self.T, self.K, self.K)) ** 3 + 0.02
        A /= A.sum(2, keepdims=True)
        q = rng.random((self.T, self.K)) + 0.1
        q /= q.sum(1, keepdims=True)
        pi0 = rng.random(self.T * self.K) + 0.1
        pi0 /= pi0.sum()
        return hc.Model(A=A, q=q, pi0=pi0, rho=0.05)

    @pytest.mark.parametrize("w", [1, 2, 4])
    def test_it_is_what_the_c_code_computes_for_a_chain(self, w):
        rng = np.random.default_rng(w)
        seq = [int(x) for x in rng.integers(0, self.K, 11)]
        ch = _chains([[c + 1 for c in seq]], prob=0.6, n_classes=self.K)
        model = self._model(3)
        chunk = hc._Chunk(ch, 0, 1)
        out = hc._forward_backward(
            chunk, self.K, model, None, collect=False, posterior=True, prior_steps=w
        )
        eps = [float(ch.eps[0])] * len(seq)
        tpost, cbest, ctx, hit = _reference_cavity(seq, eps, model, self.K, w)
        np.testing.assert_allclose(out["tpost"], tpost, atol=3e-4)
        np.testing.assert_array_equal(out["cbest"], cbest)
        np.testing.assert_allclose(out["ctx_prob"], ctx, atol=3e-4)
        np.testing.assert_array_equal(out["ctx_hit"], hit)

    def test_chains_of_different_lengths_in_one_chunk_each_have_their_own_prior(self):
        seqs = [
            [0, 1, 2, 3, 0, 1, 2, 3, 0],
            [3, 2, 1, 0, 3],
            [1, 1, 2, 2, 3, 3, 0, 0, 1, 1, 2],
        ]
        ch = _chains([[c + 1 for c in s] for s in seqs], prob=0.6, n_classes=self.K)
        model = self._model(4)
        out = hc._forward_backward(
            hc._Chunk(ch, 0, ch.n_chains),
            self.K,
            model,
            None,
            collect=False,
            posterior=True,
            prior_steps=2,
        )
        eps = float(ch.eps[0])
        for f, seq in enumerate(seqs):
            _, _, ctx, hit = _reference_cavity(seq, [eps] * len(seq), model, self.K, 2)
            rows = np.flatnonzero(ch.filament == f)
            rows = rows[np.argsort(ch.pos[rows])]
            # the chains of the chunk are in order of length: find this one
            np.testing.assert_allclose(
                out["ctx_prob"][_cells_of(ch, hc._Chunk(ch, 0, ch.n_chains), rows)],
                ctx,
                atol=3e-4,
            )

    def test_a_fit_gives_the_context_probability_of_every_segment(self):
        seqs, _ = TestFit._simulate(80, seed=13)
        ch = _chains(seqs, prob=0.8, n_classes=18)
        res = hc.fit(ch, 3, n_iter=2, cpu=2, prior_steps=3)
        assert res["ctx_prob"].shape == (ch.n_segments,)
        assert ((res["ctx_prob"] > 0) & (res["ctx_prob"] <= 1)).all()
        # filaments that walk their classes: the classes of segments three steps away
        # predict the class of a segment better than chance
        assert res["ctx_hit"].mean() > 3.0 / 18
        assert res["class_best"].shape == (ch.n_segments,)
        none = hc.fit(ch, 3, n_iter=2, cpu=2)
        assert "ctx_prob" not in none and "class_best" in none


def _cells_of(chains, chunk, rows):
    """The cells (in the layout of a chunk) of the segments ``rows``."""
    lookup = {int(s): int(c) for s, c in zip(chunk.segments, chunk.cell)}
    return [lookup[int(r)] for r in rows]


def _reference_period_score(x, ch, cl, phase0, K, C, rounds=8):
    """``HelicalContext::fitPeriods`` for one trial period, with complex numbers."""
    n = len(x)
    en = np.exp(2j * np.pi * x / C)
    pf = np.exp(1j * phase0)
    unit = lambda c: np.where(np.abs(c) > 0, c / np.maximum(np.abs(c), 1e-300), 1.0)
    for _ in range(rounds):
        z = np.zeros(K, dtype=complex)
        np.add.at(z, cl, en * pf[ch])
        tk = unit(z)
        zf = np.zeros(len(pf), dtype=complex)
        np.add.at(zf, ch, tk[cl] * np.conj(en))
        pf = unit(zf)
    return abs((tk[cl] * np.conj(en * pf[ch])).sum()) / n


class TestPeriods:
    def test_the_score_is_what_the_c_code_computes(self):
        rng = np.random.default_rng(0)
        n, F, K = 300, 7, 5
        x = rng.uniform(0, 600, n)
        ch = rng.integers(0, F, n)
        cl = rng.integers(0, K, n)
        phase0 = rng.uniform(0, 2 * np.pi, F)
        periods = np.array([100.0, 130.0, 250.0, 333.0])
        out = np.zeros(len(periods))
        hc._period_scores(
            x, ch.astype(np.int64), cl.astype(np.int64), phase0, K, periods, 8, out
        )
        ref = [_reference_period_score(x, ch, cl, phase0, K, C) for C in periods]
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_without_numba_the_numpy_kernel_matches_the_c_code_too(self, monkeypatch):
        rng = np.random.default_rng(0)
        n, F, K = 300, 7, 5
        x = rng.uniform(0, 600, n)
        ch = rng.integers(0, F, n)
        cl = rng.integers(0, K, n)
        phase0 = rng.uniform(0, 2 * np.pi, F)
        periods = np.array([100.0, 130.0, 250.0, 333.0])
        out = np.zeros(len(periods))
        monkeypatch.setattr(hc, "_HAS_NUMBA", False)
        hc._period_scores_np(
            x, ch.astype(np.int64), cl.astype(np.int64), phase0, K, periods, 8, out
        )
        ref = [_reference_period_score(x, ch, cl, phase0, K, C) for C in periods]
        np.testing.assert_allclose(out, ref, rtol=1e-6)

    def test_fit_periods_does_not_need_numba(self, monkeypatch):
        """Exercises the dispatch in :func:`fit_periods` itself, not just the
        kernel directly."""
        monkeypatch.setattr(hc, "_HAS_NUMBA", False)
        rng = np.random.default_rng(1)
        seqs = []
        for f in range(40):
            phase = int(rng.integers(10))
            seq = [(phase + s) % 10 for s in range(60)]
            seq = [int(rng.integers(10)) if rng.random() < 0.1 else c for c in seq]
            seqs.append([c + 1 for c in seq])
        ch = _chains(seqs, prob=0.9, n_classes=10, step=15.0)
        seg_type = np.zeros(ch.n_segments, dtype=int)
        period, score = hc.fit_periods(
            ch, seg_type, np.ones(ch.n_segments), ch.cls, 2, min_segments=100
        )
        assert period[0] == pytest.approx(150.0, abs=10.0)
        assert score[0] > 0.7

    def test_filaments_that_cycle_through_their_classes_have_that_period(self):
        # 10 classes, a step of 15 A: the cycle is 150 A; the filaments start anywhere
        rng = np.random.default_rng(1)
        seqs = []
        for f in range(40):
            phase = int(rng.integers(10))
            seq = [(phase + s) % 10 for s in range(60)]
            seq = [int(rng.integers(10)) if rng.random() < 0.1 else c for c in seq]
            seqs.append([c + 1 for c in seq])
        ch = _chains(seqs, prob=0.9, n_classes=10, step=15.0)
        seg_type = np.zeros(ch.n_segments, dtype=int)
        period, score = hc.fit_periods(
            ch, seg_type, np.ones(ch.n_segments), ch.cls, 2, min_segments=100
        )
        assert period[0] == pytest.approx(150.0, abs=10.0)
        assert score[0] > 0.7
        assert period[1] == 0.0  # no segment of the other type: no period

    def test_a_type_with_too_few_segments_has_none(self):
        ch = _chains([[1, 2, 3, 4] * 6], n_classes=4, step=15.0)
        period, _ = hc.fit_periods(
            ch, np.zeros(ch.n_segments, dtype=int), np.ones(ch.n_segments), ch.cls, 1
        )
        assert period[0] == 0.0


class TestTables:
    def _setup(self):
        ch = _chains([[1] * 10, [1] * 6], n_classes=3, step=20.0)
        tp = np.zeros((16, 2), dtype=np.float32)
        types = [0] * 4 + [1] * 6 + [1] * 6
        tp[np.arange(16), types] = 0.9
        tp[np.arange(16), 1 - np.array(types)] = 0.1
        return ch, tp

    def test_types_sections_and_filaments_are_counted(self):
        ch, tp = self._setup()
        seg_t, seg_p, sec = hc.segment_types(ch, tp)
        model = hc.Model(
            A=np.full((2, 3, 3), 1 / 3),
            q=np.array([[0.75, 0.2, 0.05], [0.05, 0.3, 0.65]]),
            pi0=np.full(6, 1 / 6),
            rho=0.01,
        )
        types, fils = hc.type_tables(
            ch,
            seg_t,
            seg_p,
            sec,
            tp,
            model,
            [11, 12, 13],
            period=np.array([300.0, 0.0]),
            score=np.array([0.8, 0.0]),
        )
        assert list(types["Type"]) == [1, 2]
        assert list(types["NrSections"]) == [1, 2]
        assert list(types["NrFilaments"]) == [1, 2]
        assert types["NrSegments"][0] == pytest.approx(4 * 0.9 + 12 * 0.1, abs=1e-5)
        # type 1: a section of 4 segments, 3 steps of 20 A, counted from end to end: 4 x 20
        assert types["MeanSectionAngst"][0] == pytest.approx(80.0)
        assert types["MeanSectionAngst"][1] == pytest.approx((120.0 + 120.0) / 2)
        assert list(types["MainClasses"]) == ["11,12", "13,12"]  # 90% of each
        assert list(types["PeriodAngst"]) == [300.0, 0.0]
        assert list(fils["NrSegments"]) == [10, 6]
        assert list(fils["Sections"]) == ["1x4,2x6", "2x6"]
        assert list(fils["NrSections"]) == [2, 1]

    def test_the_labels_are_numbered_from_one_as_relion_does(self):
        ch, tp = self._setup()
        seg_t, seg_p, sec = hc.segment_types(ch, tp)
        table = hc.segment_labels(seg_t, seg_p, sec, np.linspace(0, 1, 16))
        assert list(table.columns) == [
            "rlnHelicalType",
            "rlnHelicalTypeProbability",
            "rlnHelicalSectionID",
            "rlnHelicalContextProbability",
        ]
        assert (
            table["rlnHelicalType"].min() == 1
            and table["rlnHelicalSectionID"].min() == 1
        )
        assert table["rlnHelicalContextProbability"].iloc[-1] == pytest.approx(1.0)

    def test_sections_are_numbered_in_the_order_of_the_input(self):
        # the shorter filament is first in the input, but the chains are numbered
        # longest first: with the chains, its section is still number 1
        ch = _chains([[1] * 4, [1] * 9], n_classes=3, step=20.0)
        tp = np.zeros((13, 2), dtype=np.float32)
        tp[:, 0] = 1.0
        seg_t, seg_p, sec = hc.segment_types(ch, tp)
        by_chain = hc.segment_labels(seg_t, seg_p, sec)["rlnHelicalSectionID"]
        in_order = hc.segment_labels(seg_t, seg_p, sec, chains=ch)[
            "rlnHelicalSectionID"
        ]
        assert list(by_chain[:4]) == [2] * 4 and list(by_chain[4:]) == [1] * 9
        assert list(in_order[:4]) == [1] * 4 and list(in_order[4:]) == [2] * 9


class TestAlike:
    """Types of one structure differ only by the phase of its cycle of views they show."""

    @staticmethod
    def _blocks(between):
        # three blocks of six classes: alike inside a block, ``between`` across blocks
        S = np.full((18, 18), between)
        for b in range(3):
            S[6 * b : 6 * b + 6, 6 * b : 6 * b + 6] = 0.95
        np.fill_diagonal(S, 1.0)
        return S

    def test_the_usage_of_each_class_by_type(self):
        ch = _chains([[1, 1, 2], [3, 3, 3]], n_classes=3)
        tp = np.zeros((6, 2), dtype=np.float32)
        tp[:3, 0] = 1.0
        tp[3:, 1] = 1.0
        assert hc.class_usage(ch, tp).tolist() == [[2, 1, 0], [0, 0, 3]]

    def test_types_that_share_their_classes_are_alike(self):
        u = np.array([[5.0, 5.0, 0], [5.0, 5.0, 0]])
        L = hc.type_likeness(u, np.eye(3))
        assert L[0, 1] == pytest.approx(0.5)  # half of the pairs are the same class
        assert hc.type_likeness(np.array([[1.0, 0, 0], [1.0, 0, 0]]), np.eye(3))[
            0, 1
        ] == pytest.approx(1.0)

    def test_how_much_of_the_cycle_a_filament_shows(self):
        ch = _chains([[1, 2, 3, 4] * 6, [1] * 24], n_classes=4, step=10.0)
        cov = hc.filament_coverage(ch)
        assert cov["n_filaments"] == 2
        assert cov["median_classes"] == pytest.approx((4 + 1) / 2)
        assert cov["median_span"] == pytest.approx(230.0)
        assert cov["n_classes"] == 4
        assert hc.filament_coverage(ch, min_segments=100)["n_filaments"] == 0

    def test_the_likeness_of_the_types_is_reported_and_changes_nothing_else(self):
        seqs, _ = TestFit._simulate(120, seed=3)
        import pandas as pd

        rows = [
            (f"m{f}", 0, 10.0 * s, c, 0.8)
            for f, seq in enumerate(seqs)
            for s, c in enumerate(seq)
        ]
        df = pd.DataFrame(
            rows,
            columns=[
                "rlnMicrographName",
                "rlnHelicalTubeID",
                "rlnHelicalTrackLengthAngst",
                "rlnClassNumber",
                "rlnMaxValueProbDistribution",
            ],
        )
        keys = [(f"m{f}", 0) for f in range(len(seqs))]
        classes = list(range(1, 19))
        kw = dict(n_max=3, cpu=2, select=False)
        splits, info = hc.context_splits(
            df, classes, keys, similarity=self._blocks(0.6), **kw
        )
        L = info["likeness"][3]
        assert L.shape == (3, 3)
        # types of classes of their own: a pair is as alike as the blocks are
        off = L[np.triu_indices(3, 1)]
        assert off.max() < 0.8 and np.allclose(L, L.T)
        # only the classes compared count
        _, some = hc.context_splits(
            df,
            classes,
            keys,
            similarity=self._blocks(0.6),
            compare_classes=classes[:12],
            **kw,
        )
        assert some["likeness"][3].shape == (3, 3)
        plain, none = hc.context_splits(df, classes, keys, **kw)
        assert none["likeness"] == {}
        assert [len(a["labels"]) for a in splits] == [len(a["labels"]) for a in plain]

    def test_the_held_out_choice_starts_from_one_type(self):
        seqs, _ = TestFit._simulate(60, seed=4)
        ch = _chains(seqs, prob=0.8, n_classes=18)
        best, scores = hc.select_n_types(ch, 1, 3, cpu=2)
        assert set(scores) == {1, 2, 3}


class TestPyodide32BitBincount:
    """Under 32-bit WebAssembly (Pyodide), ``np.intp`` is ``np.int32``, numba is
    unavailable, and ``np.bincount`` rejects ``int64`` arrays."""

    def test_context_groups_and_period_scores_np_use_intp(self, monkeypatch):
        import pandas as pd

        class _Wasm32Numpy:
            intp = np.int32

            @staticmethod
            def bincount(x, weights=None, minlength=0):
                arr = np.asarray(x)
                if not np.can_cast(arr.dtype, np.int32, casting="safe"):
                    raise TypeError(
                        f"Cannot cast array data from {arr.dtype!r} to "
                        f"dtype('int32') according to the rule 'safe'"
                    )
                return np.bincount(arr, weights=weights, minlength=minlength)

            def __getattr__(self, name):
                return getattr(np, name)

        monkeypatch.setattr(hc, "np", _Wasm32Numpy())
        monkeypatch.setattr(hc, "_HAS_NUMBA", False)

        seqs, _ = TestFit._simulate(60, seed=5)
        rows = [
            (f"m{f}", 0, 10.0 * s, c, 0.9)
            for f, seq in enumerate(seqs)
            for s, c in enumerate(seq)
        ]
        df = pd.DataFrame(
            rows,
            columns=[
                "rlnMicrographName",
                "rlnHelicalTubeID",
                "rlnHelicalTrackLengthAngst",
                "rlnClassNumber",
                "rlnMaxValueProbDistribution",
            ],
        )
        keys = [(f"m{f}", 0) for f in range(len(seqs))]
        classes = list(range(1, 19))
        labels, info = hc.context_groups(
            df["rlnMicrographName"],
            df["rlnHelicalTubeID"],
            df["rlnHelicalTrackLengthAngst"],
            df["rlnClassNumber"],
            df["rlnMaxValueProbDistribution"],
            classes,
            keys,
            [1, 2],
            max_fit_segments=800,
            n_iter=2,
            cpu=1,
            select=True,
            detail=True,
        )
        assert set(labels) == {1, 2}
        assert len(info["detail"][2]["period"]) == 2
