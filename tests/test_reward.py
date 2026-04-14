"""Unit tests for the reward calculator.

Tests cover:
    - Correct reward component ranges.
    - Perfect detection scenario (R_det = 1.0).
    - All-false-positive scenario (R_fp = -1.0).
    - Throughput clamping to [0, 1].
    - Latency boundary at max_acceptable_latency_ms.
    - Policy stability penalty.
    - Composite reward within expected bounds.
"""

import pytest
import numpy as np

from src.environment.reward import RewardCalculator, RewardComponents


@pytest.fixture
def default_config():
    """Default reward configuration matching config/dqn_config.yaml."""
    return {
        "w_det": 0.40,
        "w_fp": 0.25,
        "w_thr": 0.15,
        "w_lat": 0.10,
        "w_stab": 0.10,
        "epsilon": 1e-8,
        "max_acceptable_latency_ms": 50.0,
        "max_changes_per_step": 5,
    }


@pytest.fixture
def calculator(default_config):
    return RewardCalculator(default_config)


@pytest.fixture
def baseline_metrics():
    """Metrics for a normal step with no attack and no policy changes."""
    return {
        "true_positives": 0,
        "false_negatives": 0,
        "false_positives": 0,
        "true_negatives": 100,
        "current_throughput_mbps": 95.0,
        "baseline_throughput_mbps": 100.0,
        "min_throughput_mbps": 10.0,
        "current_latency_ms": 5.0,
        "policy_changes": 0,
    }


class TestRewardComponents:
    """Test individual reward component calculations."""

    def test_perfect_detection(self, calculator):
        """R_detection should be 1.0 when all attacks are detected."""
        metrics = {
            "true_positives": 10,
            "false_negatives": 0,
            "false_positives": 0,
            "true_negatives": 90,
            "current_throughput_mbps": 100.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 0.0,
            "policy_changes": 0,
        }
        result = calculator.compute(metrics)
        assert result.detection == pytest.approx(1.0, abs=1e-6)

    def test_zero_detection(self, calculator):
        """R_detection should be ~0 when no attacks are detected."""
        metrics = {
            "true_positives": 0,
            "false_negatives": 10,
            "false_positives": 0,
            "true_negatives": 90,
            "current_throughput_mbps": 100.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 0.0,
            "policy_changes": 0,
        }
        result = calculator.compute(metrics)
        assert result.detection == pytest.approx(0.0, abs=1e-6)

    def test_false_positive_penalty(self, calculator):
        """R_false_positive should be negative when FPs exist."""
        metrics = {
            "true_positives": 0,
            "false_negatives": 0,
            "false_positives": 10,
            "true_negatives": 90,
            "current_throughput_mbps": 100.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 0.0,
            "policy_changes": 0,
        }
        result = calculator.compute(metrics)
        assert -1.0 <= result.false_positive < 0.0

    def test_no_false_positives(self, calculator):
        """R_false_positive should be ~0 when no FPs exist."""
        metrics = {
            "true_positives": 5,
            "false_negatives": 5,
            "false_positives": 0,
            "true_negatives": 90,
            "current_throughput_mbps": 100.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 0.0,
            "policy_changes": 0,
        }
        result = calculator.compute(metrics)
        assert result.false_positive == pytest.approx(0.0, abs=1e-6)

    def test_throughput_clamped_high(self, calculator):
        """R_throughput should be clamped to 1.0 when throughput exceeds baseline."""
        metrics = {
            "true_positives": 0, "false_negatives": 0,
            "false_positives": 0, "true_negatives": 100,
            "current_throughput_mbps": 120.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 5.0,
            "policy_changes": 0,
        }
        result = calculator.compute(metrics)
        assert result.throughput == pytest.approx(1.0, abs=1e-6)

    def test_throughput_clamped_low(self, calculator):
        """R_throughput should be 0 when throughput equals min."""
        metrics = {
            "true_positives": 0, "false_negatives": 0,
            "false_positives": 0, "true_negatives": 100,
            "current_throughput_mbps": 10.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 5.0,
            "policy_changes": 0,
        }
        result = calculator.compute(metrics)
        assert result.throughput == pytest.approx(0.0, abs=1e-6)

    def test_latency_at_max(self, calculator):
        """R_latency should be 0 when latency equals the max threshold."""
        metrics = {
            "true_positives": 0, "false_negatives": 0,
            "false_positives": 0, "true_negatives": 100,
            "current_throughput_mbps": 100.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 50.0,
            "policy_changes": 0,
        }
        result = calculator.compute(metrics)
        assert result.latency == pytest.approx(0.0, abs=1e-6)

    def test_latency_zero(self, calculator):
        """R_latency should be 1.0 when latency is 0."""
        metrics = {
            "true_positives": 0, "false_negatives": 0,
            "false_positives": 0, "true_negatives": 100,
            "current_throughput_mbps": 100.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 0.0,
            "policy_changes": 0,
        }
        result = calculator.compute(metrics)
        assert result.latency == pytest.approx(1.0, abs=1e-6)

    def test_stability_no_changes(self, calculator):
        """R_stability should be 0.0 when no policy changes occur."""
        metrics = {
            "true_positives": 0, "false_negatives": 0,
            "false_positives": 0, "true_negatives": 100,
            "current_throughput_mbps": 100.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 0.0,
            "policy_changes": 0,
        }
        result = calculator.compute(metrics)
        assert result.stability == pytest.approx(0.0, abs=1e-6)

    def test_stability_max_changes(self, calculator):
        """R_stability should be -1.0 when max policy changes are made."""
        metrics = {
            "true_positives": 0, "false_negatives": 0,
            "false_positives": 0, "true_negatives": 100,
            "current_throughput_mbps": 100.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 0.0,
            "policy_changes": 5,
        }
        result = calculator.compute(metrics)
        assert result.stability == pytest.approx(-1.0, abs=1e-6)


class TestCompositeReward:
    """Test the weighted composite reward."""

    def test_baseline_reward_positive(self, calculator, baseline_metrics):
        """Baseline (no attack, no action) should yield a positive reward."""
        result = calculator.compute(baseline_metrics)
        assert result.total > 0.0

    def test_reward_to_dict(self, calculator, baseline_metrics):
        """to_dict should return all 6 expected keys."""
        result = calculator.compute(baseline_metrics)
        d = result.to_dict()
        expected_keys = {
            "r_detection", "r_false_positive", "r_throughput",
            "r_latency", "r_stability", "r_total",
        }
        assert set(d.keys()) == expected_keys

    def test_reward_bounded(self, calculator):
        """Total reward should be bounded by the worst/best cases."""
        # Worst case: no detection, all FP, no throughput, max latency, max changes
        worst = {
            "true_positives": 0, "false_negatives": 10,
            "false_positives": 100, "true_negatives": 0,
            "current_throughput_mbps": 0.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 100.0,
            "policy_changes": 5,
        }
        result = calculator.compute(worst)
        assert result.total < 0.0

        # Best case: perfect detection, no FP, full throughput, zero latency, no changes
        best = {
            "true_positives": 10, "false_negatives": 0,
            "false_positives": 0, "true_negatives": 90,
            "current_throughput_mbps": 100.0,
            "baseline_throughput_mbps": 100.0,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": 0.0,
            "policy_changes": 0,
        }
        result = calculator.compute(best)
        assert result.total > 0.5
