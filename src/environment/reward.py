"""Reward calculation for the RL-driven adaptive security system.

Implements the composite reward function:
    R(t) = w_det * R_detection + w_fp * R_false_positive + w_thr * R_throughput
           + w_lat * R_latency + w_stab * R_stability

Each component captures a different aspect of security and network health.
Weights are loaded from config/dqn_config.yaml (or ppo_config.yaml).

Reference: Section 4.4 of the system design (reward function specification).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Individual reward components for logging and analysis.

    Attributes:
        detection: True positive rate [0, 1].
        false_positive: Negative penalty for blocking legitimate traffic [-1, 0].
        throughput: Throughput maintenance score [0, 1].
        latency: Latency maintenance score [0, 1].
        stability: Negative penalty for policy oscillation [-1, 0].
        total: Weighted composite reward.
    """

    detection: float
    false_positive: float
    throughput: float
    latency: float
    stability: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to a plain dictionary for logging/CSV export."""
        return {
            "r_detection": self.detection,
            "r_false_positive": self.false_positive,
            "r_throughput": self.throughput,
            "r_latency": self.latency,
            "r_stability": self.stability,
            "r_total": self.total,
        }


class RewardCalculator:
    """Computes composite reward from network security and performance metrics.

    The reward balances five objectives: attack detection, false positive
    avoidance, throughput maintenance, latency control, and policy stability.

    Args:
        config: Dictionary containing reward weights and thresholds.
            Required keys: w_det, w_fp, w_thr, w_lat, w_stab,
            epsilon, max_acceptable_latency_ms, max_changes_per_step.

    Example:
        >>> from src.utils.config_loader import get_reward_config
        >>> calc = RewardCalculator(get_reward_config("dqn"))
        >>> metrics = {
        ...     'true_positives': 8, 'false_negatives': 2,
        ...     'false_positives': 1, 'true_negatives': 89,
        ...     'current_throughput_mbps': 85.0,
        ...     'baseline_throughput_mbps': 100.0,
        ...     'min_throughput_mbps': 10.0,
        ...     'current_latency_ms': 12.0,
        ...     'policy_changes': 2,
        ... }
        >>> result = calc.compute(metrics)
        >>> assert -1.0 <= result.total <= 1.0
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.w_det = float(config["w_det"])
        self.w_fp = float(config["w_fp"])
        self.w_thr = float(config["w_thr"])
        self.w_lat = float(config["w_lat"])
        self.w_stab = float(config["w_stab"])
        self.eps = float(config.get("epsilon", 1e-8))
        self.max_latency = float(config["max_acceptable_latency_ms"])
        self.max_changes = int(config["max_changes_per_step"])

        # Validate weights sum to approximately 1.0
        weight_sum = self.w_det + self.w_fp + self.w_thr + self.w_lat + self.w_stab
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(
                "Reward weights sum to %.4f (expected ~1.0). "
                "This may cause reward scale issues.",
                weight_sum,
            )

        logger.info(
            "RewardCalculator initialized: w=[%.2f, %.2f, %.2f, %.2f, %.2f]",
            self.w_det, self.w_fp, self.w_thr, self.w_lat, self.w_stab,
        )

    def compute(self, metrics: Dict[str, Any]) -> RewardComponents:
        """Compute composite reward from current step metrics.

        Args:
            metrics: Dictionary containing:
                - true_positives (int): Correctly identified attacks.
                - false_negatives (int): Missed attacks.
                - false_positives (int): Legitimate traffic wrongly blocked.
                - true_negatives (int): Correctly allowed legitimate traffic.
                - current_throughput_mbps (float): Current network throughput.
                - baseline_throughput_mbps (float): Expected throughput without security.
                - min_throughput_mbps (float): Minimum acceptable throughput.
                - current_latency_ms (float): Current round-trip latency.
                - policy_changes (int): Number of flow rule changes this step.

        Returns:
            RewardComponents with individual and total reward values.
        """
        tp = float(metrics["true_positives"])
        fn = float(metrics["false_negatives"])
        fp = float(metrics["false_positives"])
        tn = float(metrics["true_negatives"])

        # R_detection: [0, 1] — recall / true positive rate
        r_det = tp / (tp + fn + self.eps)

        # R_false_positive: [-1, 0] — penalty for blocking legitimate traffic
        r_fp = -1.0 * fp / (fp + tn + self.eps)

        # R_throughput: [0, 1] — clamped ratio of current vs baseline
        cur_thr = float(metrics["current_throughput_mbps"])
        base_thr = float(metrics["baseline_throughput_mbps"])
        min_thr = float(metrics["min_throughput_mbps"])
        r_thr = float(np.clip(
            (cur_thr - min_thr) / (base_thr - min_thr + self.eps),
            0.0, 1.0,
        ))

        # R_latency: [0, 1] — reward for keeping latency below threshold
        r_lat = max(0.0, 1.0 - (float(metrics["current_latency_ms"]) / self.max_latency))

        # R_stability: [-1, 0] — penalty for excessive policy oscillation
        changes = min(int(metrics["policy_changes"]), self.max_changes)
        r_stab = -1.0 * (changes / self.max_changes)

        # Weighted composite
        total = (
            self.w_det * r_det
            + self.w_fp * r_fp
            + self.w_thr * r_thr
            + self.w_lat * r_lat
            + self.w_stab * r_stab
        )

        components = RewardComponents(
            detection=r_det,
            false_positive=r_fp,
            throughput=r_thr,
            latency=r_lat,
            stability=r_stab,
            total=total,
        )

        logger.debug(
            "Reward: det=%.4f fp=%.4f thr=%.4f lat=%.4f stab=%.4f -> total=%.4f",
            r_det, r_fp, r_thr, r_lat, r_stab, total,
        )

        return components
