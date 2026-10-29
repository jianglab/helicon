"""The accuracy of the main methods on synthetic data does not fall (see quality_scenarios.py)."""

import json
from pathlib import Path

import pytest

import quality_scenarios as q

BASELINES = json.loads((Path(__file__).parent / "quality_baselines.json").read_text())


def _worse(name, value, baseline):
    """A message if ``value`` is worse than ``baseline`` by more than the tolerance."""
    direction, tolerance = q.spec_for(name)
    if direction == "max" and value > baseline + tolerance:
        return f"{name}: {value:.4f} is above the baseline {baseline:.4f} + {tolerance}"
    if direction == "min" and value < baseline - tolerance:
        return f"{name}: {value:.4f} is below the baseline {baseline:.4f} - {tolerance}"
    return None


class TestTheComparison:
    def test_lower_is_better_for_an_error(self):
        assert _worse("a.period_error_pct", 1.0, 0.5) is None
        assert _worse("a.period_error_pct", 1.4, 0.5) is not None

    def test_higher_is_better_for_an_agreement(self):
        assert _worse("a.ari", 0.97, 1.0) is None
        assert _worse("a.ari", 0.9, 1.0) is not None

    def test_getting_better_is_not_a_failure(self):
        assert _worse("a.period_error_pct", 0.0, 2.0) is None
        assert _worse("a.ari", 1.0, 0.5) is None

    def test_every_number_has_a_kind_and_the_file_has_every_number(self):
        for name in BASELINES:
            q.spec_for(name)
        assert {
            "ring.joint.clean.period_error_pct",
            "types.two_types.accuracy",
            "context.noisy.ari",
        } <= set(BASELINES)


class TestTheMethodsKeepTheirAccuracy:
    @pytest.mark.parametrize("scenario", list(q.SCENARIOS))
    def test_scenario(self, scenario):
        numbers = q.SCENARIOS[scenario]()
        missing = [
            n
            for n in BASELINES
            if n.split(".")[0] == _group(scenario) and n not in numbers
        ]
        assert not missing, f"no longer produced (refresh the baselines?): {missing}"
        unknown = [n for n in numbers if n not in BASELINES]
        assert not unknown, f"not in quality_baselines.json: {unknown}"
        problems = [
            m
            for n, v in numbers.items()
            if (m := _worse(n, v, BASELINES[n])) is not None
        ]
        assert not problems, "\n".join(problems)


def _group(scenario):
    return {
        "ring_period": "ring",
        "ring_map_period": "ring_map",
        "two_types": "types",
        "context_typing": "context",
        "true_fsc": "truefsc",
    }[scenario]
