"""The joint solver fits without an intercept, unless asked for one.

The volume is the regression's coefficients alone; the intercept is fitted and
then dropped, so whatever it offsets stays in the map. On a volume only a few
voxels long (few symmetry equations) elastic net paid a large negative
intercept with a bright ring at the edge of the cylinder.
"""

import inspect

import numpy as np
import scipy.sparse as sp

from helicon.webApps.lib import denovo3d_solver, helical_pitch_map


def _system(offset):
    rng = np.random.default_rng(0)
    A = sp.csr_matrix(rng.random((400, 20)) * (rng.random((400, 20)) < 0.3))
    x = np.zeros(20)
    x[[3, 7, 11]] = [1.0, 0.5, 0.8]
    return A, A @ x + offset, x


class TestTheIntercept:
    def test_off_by_default_and_on_request(self):
        A, b, _ = _system(0.0)
        captured = {}
        orig = __import__("sklearn.linear_model", fromlist=["ElasticNet"]).ElasticNet

        class Spy(orig):
            def fit(self, X, y, *a, **k):
                captured["fit_intercept"] = self.fit_intercept
                return super().fit(X, y, *a, **k)

        import sklearn.linear_model as lm

        old = lm.ElasticNet
        lm.ElasticNet = Spy
        try:
            denovo3d_solver.solve_equations(
                A, b, None, None, algorithm=dict(model="elasticnet")
            )
            assert captured["fit_intercept"] is False
            denovo3d_solver.solve_equations(
                A,
                b,
                None,
                None,
                algorithm=dict(model="elasticnet", fit_intercept=True),
            )
            assert captured["fit_intercept"] is True
        finally:
            lm.ElasticNet = old

    def test_without_it_a_zero_background_system_is_recovered(self):
        A, b, x = _system(0.0)
        res, _ = denovo3d_solver.solve_equations(
            A,
            b,
            None,
            None,
            positive=True,
            algorithm=dict(model="elasticnet", alpha=1e-6, fit_intercept=False),
        )
        # within the shrinkage the L1 penalty applies to the largest values
        assert np.allclose(res, x, atol=0.05)


class TestAbInitioMaps:
    def test_the_joint_fit_has_no_intercept(self):
        src = inspect.getsource(helical_pitch_map._reconstruct_joint)
        assert 'dict(model="elasticnet", l1_ratio=0.5, fit_intercept=False)' in src
