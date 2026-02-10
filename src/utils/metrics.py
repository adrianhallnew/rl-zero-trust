"""Metric calculation utilities for the RL-driven adaptive security system.

Provides functions for computing security and performance metrics
used in evaluation against the defined thresholds.

Security metrics: detection rate, false positive rate, adaptation speed
Performance metrics: throughput degradation, latency overhead, convergence
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def detection_rate(true_positives: int, false_negatives: int) -> float:
    """Compute detection rate (true positive rate / recall).

    Args:
        true_positives: Number of correctly detected attacks.
        false_negatives: Number of missed attacks.

    Returns:
        Detection rate in [0, 1].
    """
    total = true_positives + false_negatives
    if total == 0:
        return 0.0
    return true_positives / total


def false_positive_rate(false_positives: int, true_negatives: int) -> float:
    """Compute false positive rate.

    Args:
        false_positives: Number of legitimate traffic incorrectly blocked.
        true_negatives: Number of correctly allowed legitimate traffic.

    Returns:
        False positive rate in [0, 1].
    """
    total = false_positives + true_negatives
    if total == 0:
        return 0.0
    return false_positives / total


def throughput_degradation(
    current_throughput: float,
    baseline_throughput: float,
) -> float:
    """Compute throughput degradation percentage.

    Args:
        current_throughput: Throughput with security system active (Mbps).
        baseline_throughput: Throughput without security system (Mbps).

    Returns:
        Degradation as percentage (0 = no degradation, 100 = total loss).
    """
    if baseline_throughput <= 0:
        return 0.0
    degradation = (baseline_throughput - current_throughput) / baseline_throughput * 100
    return max(0.0, degradation)


def latency_overhead(
    current_latency_ms: float,
    baseline_latency_ms: float,
) -> float:
    """Compute additional latency from security enforcement.

    Args:
        current_latency_ms: Latency with security system active.
        baseline_latency_ms: Latency without security system.

    Returns:
        Overhead in milliseconds.
    """
    return max(0.0, current_latency_ms - baseline_latency_ms)


def evaluate_thresholds(metrics: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """Evaluate metrics against the defined pass/fail thresholds.

    Args:
        metrics: Dictionary with metric name -> value pairs.
            Expected keys: detection_rate_ddos, detection_rate_portscan,
            detection_rate_spoofing, detection_rate_overall,
            false_positive_rate, adaptation_speed_s,
            throughput_degradation_pct, latency_overhead_ms

    Returns:
        Dictionary mapping metric name -> {value, threshold, status, rating}.
    """
    thresholds = {
        "detection_rate_ddos": {"min": 0.75, "good": 0.85, "excellent": 0.95},
        "detection_rate_portscan": {"min": 0.70, "good": 0.80, "excellent": 0.90},
        "detection_rate_spoofing": {"min": 0.65, "good": 0.80, "excellent": 0.90},
        "detection_rate_overall": {"min": 0.70, "good": 0.82, "excellent": 0.92},
        "false_positive_rate": {"max": 0.15, "good": 0.08, "excellent": 0.03},
        "adaptation_speed_s": {"max": 30, "good": 10, "excellent": 3},
        "throughput_degradation_pct": {"max": 25, "good": 15, "excellent": 5},
        "latency_overhead_ms": {"max": 50, "good": 20, "excellent": 5},
    }

    results = {}
    for name, value in metrics.items():
        if name not in thresholds:
            continue

        thresh = thresholds[name]

        if "min" in thresh:
            # Higher is better
            if value >= thresh["excellent"]:
                rating = "EXCELLENT"
            elif value >= thresh["good"]:
                rating = "GOOD"
            elif value >= thresh["min"]:
                rating = "PASS"
            else:
                rating = "FAIL"
            passed = value >= thresh["min"]
        else:
            # Lower is better
            if value <= thresh["excellent"]:
                rating = "EXCELLENT"
            elif value <= thresh["good"]:
                rating = "GOOD"
            elif value <= thresh["max"]:
                rating = "PASS"
            else:
                rating = "FAIL"
            passed = value <= thresh["max"]

        results[name] = {
            "value": value,
            "thresholds": thresh,
            "passed": passed,
            "rating": rating,
        }

    return results
