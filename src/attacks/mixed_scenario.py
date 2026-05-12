"""Multi-vector attack scenario orchestrator for the RL-driven adaptive security system.

Coordinates multiple simultaneous or sequential attack types to create
realistic mixed-threat scenarios for training and evaluation. Supports:

    - Random combinations of DDoS, port scan, and spoofing attacks.
    - Configurable attack onset timing and intervals.
    - Concurrent multi-vector attacks (e.g., DDoS + port scan together).
    - Phased attacks with escalation (scan -> spoof -> DDoS).

This orchestrator drives the environment's attack injection during both
simulation-mode training and live Mininet evaluation.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.attacks.ddos import DDoSAttack, DDoSConfig, INTENSITY_PROFILES
from src.attacks.port_scan import PortScanAttack, PortScanConfig, SPEED_PROFILES
from src.attacks.spoofing import SpoofingAttack, SpoofingConfig, SPOOF_PROFILES

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_THROUGHPUT = 100.0  # Mbps

# ── Attack-type-sensitive metric ranges for the orchestrator ─────────
# Each attack type maps action → {tp, fn, fp, thr, lat} ranges.
# The optimal action per type gets the best TP and throughput ranges.
# Ranges are [low, high) for randint, (low, high) for uniform.

def _r(tp, fn, fp, thr, lat):
    return {"tp": tp, "fn": fn, "fp": fp, "thr": thr, "lat": lat}

_ORCHESTRATOR_RANGES = {
    "ddos": {
        # Optimal: RATE_LIMIT(3)
        3: _r((8, 10), (0, 2), (0, 1), (70, 80), (8.0, 14.0)),
        1: _r((7, 9),  (1, 3), (0, 2), (35, 50), (5.0, 12.0)),
        2: _r((5, 7),  (3, 5), (0, 1), (55, 70), (10.0, 22.0)),
        0: _r((0, 2),  (7, 10), (0, 1), (20, 50), (20.0, 45.0)),
    },
    "portscan": {
        # Optimal: BLOCK(1)
        1: _r((8, 10), (0, 2), (0, 1), (40, 55), (5.0, 10.0)),
        3: _r((5, 7),  (3, 5), (0, 1), (60, 75), (8.0, 18.0)),
        2: _r((4, 6),  (4, 6), (0, 1), (55, 70), (10.0, 22.0)),
        0: _r((0, 2),  (7, 10), (0, 1), (20, 45), (18.0, 40.0)),
    },
    "spoofing": {
        # Optimal: REROUTE(2)
        2: _r((8, 10), (0, 2), (0, 1), (65, 75), (10.0, 16.0)),
        1: _r((6, 8),  (2, 4), (1, 3), (35, 50), (5.0, 12.0)),
        3: _r((5, 7),  (3, 5), (0, 2), (60, 75), (8.0, 18.0)),
        0: _r((0, 2),  (7, 10), (0, 1), (20, 50), (20.0, 45.0)),
    },
}


class ScenarioType(Enum):
    """Predefined attack scenario templates."""

    SINGLE_DDOS = "single_ddos"
    SINGLE_PORTSCAN = "single_portscan"
    SINGLE_SPOOFING = "single_spoofing"
    DDOS_PORTSCAN = "ddos_portscan"
    DDOS_SPOOFING = "ddos_spoofing"
    PORTSCAN_SPOOFING = "portscan_spoofing"
    ALL_COMBINED = "all_combined"
    RANDOM = "random"
    PHASED_ESCALATION = "phased_escalation"


@dataclass
class MixedScenarioConfig:
    """Configuration for the mixed attack scenario orchestrator.

    Attributes:
        scenario_type: Predefined scenario template or ``RANDOM``.
        attack_probability: Per-step probability of starting a new attack
            (used in random mode).
        max_concurrent_attacks: Maximum active attacks at once.
        min_gap_steps: Minimum steps between attack onsets.
        max_gap_steps: Maximum steps between attack onsets.
        normal_traffic_steps: Steps of normal traffic before first attack.
        seed: Random seed for reproducibility.
    """

    scenario_type: ScenarioType = ScenarioType.RANDOM
    attack_probability: float = 0.15
    max_concurrent_attacks: int = 2
    min_gap_steps: int = 5
    max_gap_steps: int = 30
    normal_traffic_steps: int = 10
    seed: int = 42


class AttackOrchestrator:
    """Coordinates multiple attack simulations for training episodes.

    Manages the lifecycle of attack objects, injection timing, and
    ground-truth label generation. Each environment episode should
    create a fresh orchestrator or call :meth:`reset`.

    Args:
        config: Orchestrator configuration.

    Example:
        >>> orch = AttackOrchestrator(MixedScenarioConfig(
        ...     scenario_type=ScenarioType.RANDOM, seed=42))
        >>> orch.reset()
        >>> state = np.zeros(65, dtype=np.float32)
        >>> for step in range(50):
        ...     state, info = orch.step(state)
        >>> assert orch.total_attack_steps > 0
    """

    def __init__(self, config: Optional[MixedScenarioConfig] = None) -> None:
        self.config = config or MixedScenarioConfig()
        self._rng = np.random.RandomState(self.config.seed)

        self._active_attacks: List[Any] = []  # Currently running attacks
        self._attack_history: List[Dict[str, Any]] = []
        self._current_step = 0
        self._last_attack_start = -self.config.min_gap_steps
        self._episode_attacks_started = 0

        # Metrics tracking
        self.total_attack_steps = 0
        self.total_normal_steps = 0
        self._attack_type_counts: Dict[str, int] = {
            "ddos": 0, "port_scan": 0, "spoofing": 0,
        }

        logger.info(
            "AttackOrchestrator created: scenario=%s, max_concurrent=%d",
            self.config.scenario_type.value, self.config.max_concurrent_attacks,
        )

    def reset(self) -> None:
        """Reset the orchestrator for a new episode."""
        self._active_attacks.clear()
        self._attack_history.clear()
        self._current_step = 0
        self._last_attack_start = -self.config.min_gap_steps
        self._episode_attacks_started = 0
        self.total_attack_steps = 0
        self.total_normal_steps = 0
        self._attack_type_counts = {
            "ddos": 0, "port_scan": 0, "spoofing": 0,
        }
        self._rng = np.random.RandomState(
            self.config.seed + hash(time.time()) % 10000
        )
        logger.debug("AttackOrchestrator reset")

    @property
    def any_attack_active(self) -> bool:
        """Whether any attack is currently running."""
        return len(self._active_attacks) > 0

    @property
    def active_attack_types(self) -> List[str]:
        """List of currently active attack type names."""
        return [a.attack_name for a in self._active_attacks]

    def step(
        self, state: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Advance one environment step with attack orchestration.

        Decides whether to start new attacks, advances active attacks,
        injects signatures into the state, and tracks ground truth.

        Args:
            state: Current state vector of shape ``(65,)``.

        Returns:
            Tuple of (modified_state, step_info).
        """
        self._current_step += 1
        modified_state = state.copy()

        # Maybe start new attacks
        if self._should_start_attack():
            self._start_new_attack()

        # Process active attacks
        still_active = []
        attack_active = False
        throughput_impacts = []
        latency_impacts = []

        for attack in self._active_attacks:
            if attack.step():
                still_active.append(attack)
                attack_active = True
                modified_state = attack.inject_signature(
                    modified_state, self._rng,
                )
                thr, lat = attack.get_traffic_impact(self._rng)
                throughput_impacts.append(thr)
                latency_impacts.append(lat)

        self._active_attacks = still_active

        # Track metrics
        if attack_active:
            self.total_attack_steps += 1
        else:
            self.total_normal_steps += 1

        # Aggregate traffic impact (worst case from all attacks)
        if throughput_impacts:
            throughput = min(throughput_impacts)
            latency = max(latency_impacts)
        else:
            throughput = self._rng.uniform(85.0, 100.0)
            latency = self._rng.uniform(2.0, 8.0)

        info = {
            "attack_active": attack_active,
            "active_attacks": self.active_attack_types,
            "num_active": len(self._active_attacks),
            "throughput_mbps": throughput,
            "latency_ms": latency,
            "step": self._current_step,
            "ground_truth": self._get_combined_ground_truth(),
        }

        return modified_state, info

    def get_step_metrics(
        self, action: int, attack_active: bool,
    ) -> Dict[str, Any]:
        """Compute confusion matrix metrics for the current step.

        Evaluates the agent's action against the current ground truth
        to produce TP/FP/TN/FN counts for reward computation.

        Args:
            action: Agent's chosen action (0-3).
            attack_active: Whether an attack is currently active.

        Returns:
            Dictionary with confusion matrix and traffic metrics.
        """
        tp, fn, fp, tn = 0, 0, 0, 0
        throughput = DEFAULT_BASELINE_THROUGHPUT
        latency = 5.0

        if attack_active:
            active_types = self.active_attack_types
            primary = active_types[0] if active_types else "unknown"

            # Classify attack category
            if "flood" in primary or "ddos" in primary:
                attack_cat = "ddos"
            elif "port" in primary or "scan" in primary:
                attack_cat = "portscan"
            elif "spoof" in primary or "arp" in primary:
                attack_cat = "spoofing"
            else:
                attack_cat = "ddos"

            # Attack-type-sensitive metric ranges
            # Optimal action per type gets the best TP/throughput ranges
            ranges = _ORCHESTRATOR_RANGES.get(attack_cat, _ORCHESTRATOR_RANGES["ddos"])
            r = ranges.get(action, ranges[0])
            tp = self._rng.randint(r["tp"][0], r["tp"][1])
            fn = self._rng.randint(r["fn"][0], r["fn"][1])
            fp = self._rng.randint(r["fp"][0], r["fp"][1])
            tn = 90 - fp
            throughput = self._rng.uniform(r["thr"][0], r["thr"][1])
            latency = self._rng.uniform(r["lat"][0], r["lat"][1])
        else:
            if action == 1:         # BLOCK during normal — worst FP
                tp, fn = 0, 0
                fp = self._rng.randint(5, 10)
                tn = 90 - fp
                throughput = DEFAULT_BASELINE_THROUGHPUT * self._rng.uniform(0.7, 0.85)
                latency = self._rng.uniform(5.0, 12.0)
            elif action == 3:       # RATE_LIMIT during normal — moderate FP
                tp, fn = 0, 0
                fp = self._rng.randint(3, 6)
                tn = 90 - fp
                throughput = DEFAULT_BASELINE_THROUGHPUT * self._rng.uniform(0.75, 0.9)
                latency = self._rng.uniform(4.0, 10.0)
            elif action == 2:       # REROUTE during normal — mild disruption
                tp, fn = 0, 0
                fp = self._rng.randint(1, 3)
                tn = 90 - fp
                throughput = DEFAULT_BASELINE_THROUGHPUT * self._rng.uniform(0.85, 0.95)
                latency = self._rng.uniform(4.0, 10.0)
            else:                   # ALLOW during normal — correct
                tp, fn, fp = 0, 0, 0
                tn = 90
                throughput = self._rng.uniform(85.0, 100.0)
                latency = self._rng.uniform(2.0, 6.0)

        policy_changes = 0 if action == 0 else 1

        return {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "current_throughput_mbps": throughput,
            "baseline_throughput_mbps": DEFAULT_BASELINE_THROUGHPUT,
            "min_throughput_mbps": 10.0,
            "current_latency_ms": latency,
            "policy_changes": policy_changes,
        }

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _should_start_attack(self) -> bool:
        """Decide whether to start a new attack this step."""
        if len(self._active_attacks) >= self.config.max_concurrent_attacks:
            return False

        gap = self._current_step - self._last_attack_start
        if gap < self.config.min_gap_steps:
            return False

        if self._current_step < self.config.normal_traffic_steps:
            return False

        scenario = self.config.scenario_type

        if scenario == ScenarioType.RANDOM:
            return self._rng.random() < self.config.attack_probability

        elif scenario == ScenarioType.PHASED_ESCALATION:
            # Phase 1: port scan, Phase 2: spoofing, Phase 3: DDoS
            if self._episode_attacks_started == 0 and gap >= 15:
                return True
            elif self._episode_attacks_started == 1 and gap >= 20:
                return True
            elif self._episode_attacks_started == 2 and gap >= 15:
                return True
            return False

        else:
            # Predefined scenarios: start attacks at configured gaps
            return (
                self._episode_attacks_started == 0
                or (gap >= self._rng.randint(
                    self.config.min_gap_steps, self.config.max_gap_steps + 1,
                ))
            )

    def _start_new_attack(self) -> None:
        """Create and start a new attack based on the scenario type."""
        scenario = self.config.scenario_type
        attack = None

        if scenario == ScenarioType.SINGLE_DDOS:
            attack = self._create_ddos()
        elif scenario == ScenarioType.SINGLE_PORTSCAN:
            attack = self._create_portscan()
        elif scenario == ScenarioType.SINGLE_SPOOFING:
            attack = self._create_spoofing()
        elif scenario == ScenarioType.PHASED_ESCALATION:
            # Escalation: portscan -> spoofing -> DDoS
            phase = self._episode_attacks_started
            if phase == 0:
                attack = self._create_portscan()
            elif phase == 1:
                attack = self._create_spoofing()
            else:
                attack = self._create_ddos()
        elif scenario == ScenarioType.RANDOM:
            attack = self._create_random_attack()
        elif scenario in (
            ScenarioType.DDOS_PORTSCAN,
            ScenarioType.DDOS_SPOOFING,
            ScenarioType.PORTSCAN_SPOOFING,
            ScenarioType.ALL_COMBINED,
        ):
            attack = self._create_from_combo(scenario)
        else:
            attack = self._create_random_attack()

        if attack is not None:
            attack.start()
            self._active_attacks.append(attack)
            self._last_attack_start = self._current_step
            self._episode_attacks_started += 1

            # Track attack type counts
            gt = attack.get_ground_truth()
            attack_type = gt.get("attack_type", "unknown")
            if attack_type in self._attack_type_counts:
                self._attack_type_counts[attack_type] += 1

            logger.debug(
                "Started attack #%d: %s at step %d",
                self._episode_attacks_started, attack.attack_name,
                self._current_step,
            )

    def _create_random_attack(self) -> Any:
        """Create a random attack type."""
        choice = self._rng.choice(["ddos", "port_scan", "spoofing"])
        if choice == "ddos":
            return self._create_ddos()
        elif choice == "port_scan":
            return self._create_portscan()
        else:
            return self._create_spoofing()

    def _create_ddos(self) -> DDoSAttack:
        """Create a DDoS attack with random configuration."""
        attack_type = self._rng.choice(["syn_flood", "udp_flood"])
        intensity = self._rng.choice(["low", "medium", "high"])
        target_sw = self._rng.randint(0, 5)
        return DDoSAttack(DDoSConfig(
            attack_type=attack_type,
            intensity=intensity,
            target_switch=target_sw,
            duration_steps=self._rng.randint(15, 40),
            seed=int(self._rng.randint(0, 100000)),
        ))

    def _create_portscan(self) -> PortScanAttack:
        """Create a port scan with random configuration."""
        scan_type = self._rng.choice(["tcp_connect", "syn_scan"])
        speed = self._rng.choice(["slow", "normal", "aggressive"])
        target_sw = self._rng.randint(0, 5)
        return PortScanAttack(PortScanConfig(
            scan_type=scan_type,
            speed=speed,
            target_switch=target_sw,
            randomize_ports=bool(self._rng.choice([True, False])),
            duration_steps=self._rng.randint(10, 30),
            seed=int(self._rng.randint(0, 100000)),
        ))

    def _create_spoofing(self) -> SpoofingAttack:
        """Create a spoofing attack with random configuration."""
        spoof_type = self._rng.choice(["ip_spoof", "mac_spoof", "arp_spoof"])
        target_sw = self._rng.randint(0, 5)
        return SpoofingAttack(SpoofingConfig(
            spoof_type=spoof_type,
            target_switch=target_sw,
            duration_steps=self._rng.randint(10, 25),
            seed=int(self._rng.randint(0, 100000)),
        ))

    def _create_from_combo(self, scenario: ScenarioType) -> Any:
        """Create an attack for a combination scenario."""
        idx = self._episode_attacks_started

        if scenario == ScenarioType.DDOS_PORTSCAN:
            return self._create_ddos() if idx % 2 == 0 else self._create_portscan()
        elif scenario == ScenarioType.DDOS_SPOOFING:
            return self._create_ddos() if idx % 2 == 0 else self._create_spoofing()
        elif scenario == ScenarioType.PORTSCAN_SPOOFING:
            return self._create_portscan() if idx % 2 == 0 else self._create_spoofing()
        elif scenario == ScenarioType.ALL_COMBINED:
            choices = [self._create_ddos, self._create_portscan, self._create_spoofing]
            return choices[idx % 3]()
        return self._create_random_attack()

    def _get_combined_ground_truth(self) -> Dict[str, Any]:
        """Get combined ground truth from all active attacks."""
        if not self._active_attacks:
            return {
                "attack_active": False,
                "attack_types": [],
                "num_active": 0,
            }

        return {
            "attack_active": True,
            "attack_types": [a.get_ground_truth() for a in self._active_attacks],
            "num_active": len(self._active_attacks),
        }

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get a summary of the episode's attack activity.

        Returns:
            Dictionary with episode-level attack statistics.
        """
        return {
            "total_steps": self._current_step,
            "total_attack_steps": self.total_attack_steps,
            "total_normal_steps": self.total_normal_steps,
            "attack_ratio": (
                self.total_attack_steps / max(1, self._current_step)
            ),
            "attacks_started": self._episode_attacks_started,
            "attack_type_counts": dict(self._attack_type_counts),
        }
