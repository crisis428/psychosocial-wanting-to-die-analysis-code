"""Check White sandwich covariance on synthetic data; no study records are used."""
from pathlib import Path
import importlib.util
import sys
import unittest

import numpy as np
import statsmodels.api as sm

SCRIPT = Path(__file__).resolve().parents[1] / "analysis/02_regression/regression_analysis.py"
spec = importlib.util.spec_from_file_location("regression_analysis", SCRIPT)
regression = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = regression
spec.loader.exec_module(regression)


class CovarianceTest(unittest.TestCase):
    def test_white_sandwich(self):
        rng = np.random.default_rng(20260928)
        n = 400
        data = {
            "age_years": rng.integers(20, 70, n).astype(float),
            "gad7_total": rng.integers(0, 22, n).astype(float),
            "pss14_total": rng.normal(25, 6, n),
            "low_self_esteem_score": rng.normal(23, 4, n),
        }
        for name in regression.BINARY:
            data[name] = rng.binomial(1, 0.5, n).astype(float)
        for name in regression.ORDINAL:
            data[name] = rng.integers(1, 5, n).astype(float)
        logit = -1.3 + 0.06 * data["gad7_total"] + 0.25 * data["sex_female"]
        data[regression.OUTCOME] = rng.binomial(1, 1 / (1 + np.exp(-logit))).astype(float)
        bundle = regression.fit_bundle(data)
        self.assertEqual(bundle.model.cov_type, "HC0")
        probability = np.asarray(bundle.model.predict(bundle.X))
        weight = probability * (1 - probability)
        bread = np.linalg.inv(bundle.X.T @ (weight[:, None] * bundle.X))
        score = bundle.X * (bundle.y - probability)[:, None]
        expected = bread @ (score.T @ score) @ bread
        np.testing.assert_allclose(bundle.model.cov_params(), expected, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
