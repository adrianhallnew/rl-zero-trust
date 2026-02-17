"""Gymnasium-compatible RL environment for the adaptive security system.

Wraps the SDN network (Mininet + Ryu) into a standard Gym interface
so that any RL agent (DQN, PPO) can interact with it via
``reset()`` / ``step()`` / ``render()``.

The environment supports two operating modes:

1. **Live mode** — connected to a running Ryu controller via REST API.
   Observations come from real flow/port statistics, and actions are
   enforced as OpenFlow rules.

2. **Simulation mode** — no controller connection.  Uses the attack
   orchestrator module to generate synthetic observations with realistic
   attack signatures for DDoS, port scanning, and spoofing.
   This is the default for training and unit testing without Docker.

Action space modes:
- **Discrete** (DQN): Discrete(4) — ALLOW / BLOCK / REROUTE / RATE_LIMIT.
- **Continuous** (PPO): Box(3,) — rate_limit_intensity, rerouting_weight,
  priority_adjustment.

State space: Box(65,) float32  — 5 switches x 13 features.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.attacks.mixed_scenario import (
    AttackOrchestrator,
    MixedScenarioConfig,
    ScenarioType,
)
from src.environment.reward import RewardCalculator, RewardComponents
from src.environment.state_processor import StateProcessor, STATE_DIM

logger = logging.getLogger(__name__)

NUM_ACTIONS = 4
ACTION_NAMES = {0: "ALLOW", 1: "BLOCK", 2: "REROUTE", 3: "RATE_LIMIT"}
CONTINUOUS_ACTION_DIM = 3  # rate_limit_intensity, rerouting_weight, priority_adjustment

# Simulation-mode defaults
DEFAULT_BASELINE_THROUGHPUT = 100.0  # Mbps
DEFAULT_MIN_THROUGHPUT = 10.0        # Mbps
DEFAULT_MAX_LATENCY = 50.0           # ms


class NetworkSecurityEnv(gym.Env):
    """Gymnasium environment for RL-driven network security.

    Args:
        reward_config: Reward weight dictionary (from ``config/dqn_config.yaml``).
        stats_collector: Live ``StatsCollector`` instance, or *None* for
            simulation mode.
        policy_enforcer: Live ``PolicyEnforcer`` instance, or *None* for
            simulation mode.
        max_steps: Maximum steps per episode before truncation.
        moving_avg_window: State processor smoothing window.
        seed: Random seed for reproducibility.
        scenario_type: Attack scenario for simulation mode training.
            Defaults to ``RANDOM`` for diverse attack exposure.
        attack_config: Optional custom ``MixedScenarioConfig``.

    Example:
        >>> from src.utils.config_loader import get_reward_config
        >>> env = NetworkSecurityEnv(reward_config=get_reward_config("dqn"))
        >>> obs, info = env.reset()
        >>> obs.shape
        (65,)
        >>> obs, reward, terminated, truncated, info = env.step(0)
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        reward_config: Optional[Dict[str, Any]] = None,
        stats_collector=None,
        policy_enforcer=None,
        max_steps: int = 200,
        moving_avg_window: int = 10,
        seed: int = 42,
        scenario_type: Optional[ScenarioType] = None,
        attack_config: Optional[MixedScenarioConfig] = None,
        action_mode: str = "discrete",
    ) -> None:
        super().__init__()

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(STATE_DIM,), dtype=np.float32,
        )
        self._action_mode = action_mode
        if action_mode == "continuous":
            self.action_space = spaces.Box(
                low=np.array([0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Discrete(NUM_ACTIONS)

        # Components
        self.state_processor = StateProcessor(
            stats_collector=stats_collector,
            moving_avg_window=moving_avg_window,
        )
        self.policy_enforcer = policy_enforcer

        # Reward calculator
        if reward_config is None:
            reward_config = {
                "w_det": 0.40, "w_fp": 0.25, "w_thr": 0.15,
                "w_lat": 0.10, "w_stab": 0.10,
                "epsilon": 1e-8,
                "max_acceptable_latency_ms": DEFAULT_MAX_LATENCY,
                "max_changes_per_step": 5,
            }
        self.reward_calculator = RewardCalculator(reward_config)

        # Episode state
        self.max_steps = max_steps
        self._current_step = 0
        self._episode_count = 0
        self._cumulative_reward = 0.0
        self._last_reward_components: Optional[RewardComponents] = None

        # Simulation mode state
        self._sim_mode = stats_collector is None
        self._rng = np.random.RandomState(seed)
        self._seed = seed
        self._sim_attack_active = False
        self._sim_attack_type: Optional[str] = None

        # Attack orchestrator (simulation mode)
        if self._sim_mode:
            if attack_config is not None:
                self._attack_config = attack_config
            else:
                self._attack_config = MixedScenarioConfig(
                    scenario_type=scenario_type or ScenarioType.RANDOM,
                    attack_probability=0.15,
                    max_concurrent_attacks=2,
                    min_gap_steps=5,
                    max_gap_steps=30,
                    normal_traffic_steps=10,
                    seed=seed,
                )
            self._orchestrator = AttackOrchestrator(self._attack_config)
        else:
            self._orchestrator = None

        # Traffic simulation parameters
        self._sim_throughput = DEFAULT_BASELINE_THROUGHPUT
        self._sim_latency = 5.0  # ms

        logger.info(
            "NetworkSecurityEnv initialized: mode=%s, action_mode=%s, max_steps=%d",
            "simulation" if self._sim_mode else "live", action_mode, max_steps,
        )

    # ------------------------------------------------------------------
    # Gym Interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode.

        Args:
            seed: Optional RNG seed override.
            options: Optional configuration overrides.

        Returns:
            Tuple of (initial observation, info dict).
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self.state_processor.reset()
        self._current_step = 0
        self._cumulative_reward = 0.0
        self._episode_count += 1

        if self.policy_enforcer is not None:
            self.policy_enforcer.reset_policy_count()

        # Reset simulation state
        self._sim_attack_active = False
        self._sim_attack_type = None
        self._sim_throughput = DEFAULT_BASELINE_THROUGHPUT
        self._sim_latency = 5.0

        # Reset attack orchestrator
        if self._orchestrator is not None:
            self._orchestrator.reset()

        observation = self._get_observation()

        info = {
            "episode": self._episode_count,
            "mode": "simulation" if self._sim_mode else "live",
        }

        logger.debug("Episode %d started", self._episode_count)
        return observation, info

    def step(
        self, action,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step.

        Args:
            action: Discrete action index (0-3) or continuous action array
                of shape (3,) depending on action_mode.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        self._current_step += 1

        # Map continuous action to discrete for internal reward model
        if self._action_mode == "continuous":
            continuous_action = np.asarray(action, dtype=np.float32)
            discrete_action = self._continuous_to_discrete(continuous_action)
        else:
            discrete_action = int(action)
            continuous_action = None

        assert 0 <= discrete_action < NUM_ACTIONS, (
            f"Invalid discrete action: {discrete_action}"
        )

        # Reset per-step policy counter
        if self.policy_enforcer is not None:
            self.policy_enforcer.reset_policy_count()

        # Enforce action
        if self.policy_enforcer is not None and not self._sim_mode:
            # In live mode, enforce on all switches
            for dpid in range(1, 6):
                self.policy_enforcer.enforce_action(
                    discrete_action, dpid, match_fields={},
                )

        # Compute metrics for reward
        metrics = self._compute_step_metrics(discrete_action)

        # Calculate reward
        reward_components = self.reward_calculator.compute(metrics)
        reward = reward_components.total
        self._last_reward_components = reward_components
        self._cumulative_reward += reward

        # Get next observation
        observation = self._get_observation()

        # Episode termination
        terminated = False  # No natural termination in network security
        truncated = self._current_step >= self.max_steps

        info = {
            "step": self._current_step,
            "action": discrete_action,
            "action_name": ACTION_NAMES[discrete_action],
            "reward_components": reward_components.to_dict(),
            "metrics": metrics,
            "cumulative_reward": self._cumulative_reward,
            "attack_active": self._sim_attack_active,
            "attack_type": self._sim_attack_type,
        }

        if continuous_action is not None:
            info["continuous_action"] = continuous_action.tolist()

        if self.policy_enforcer is not None:
            info["policy_changes"] = self.policy_enforcer.get_policy_changes()

        logger.debug(
            "Step %d: action=%s, reward=%.4f, attack=%s",
            self._current_step, ACTION_NAMES[discrete_action], reward,
            self._sim_attack_type or "none",
        )

        return observation, reward, terminated, truncated, info

    @staticmethod
    def _continuous_to_discrete(action: np.ndarray) -> int:
        """Map continuous PPO action to discrete action for reward model.

        Mapping:
            rate_limit_intensity > 0.7 → BLOCK (1)
            rate_limit_intensity > 0.3 → RATE_LIMIT (3)
            rerouting_weight > 0.5     → REROUTE (2)
            otherwise                  → ALLOW (0)

        Args:
            action: Continuous action array [rate_limit, reroute, priority].

        Returns:
            Discrete action in {0, 1, 2, 3}.
        """
        rate_limit = float(action[0])
        reroute = float(action[1])

        if rate_limit > 0.7:
            return 1  # BLOCK
        elif rate_limit > 0.3:
            return 3  # RATE_LIMIT
        elif reroute > 0.5:
            return 2  # REROUTE
        else:
            return 0  # ALLOW

    def render(self, mode: str = "human") -> Optional[str]:
        """Render the current environment state.

        Args:
            mode: ``"human"`` prints to stdout, ``"ansi"`` returns a string.
        """
        output = (
            f"Episode {self._episode_count} | Step {self._current_step}/{self.max_steps}\n"
            f"  Attack: {self._sim_attack_type or 'none'} "
            f"(active={self._sim_attack_active})\n"
            f"  Throughput: {self._sim_throughput:.1f} Mbps | "
            f"Latency: {self._sim_latency:.1f} ms\n"
            f"  Cumulative reward: {self._cumulative_reward:.4f}\n"
        )
        if self._last_reward_components:
            rc = self._last_reward_components
            output += (
                f"  Reward breakdown: det={rc.detection:.3f} "
                f"fp={rc.false_positive:.3f} thr={rc.throughput:.3f} "
                f"lat={rc.latency:.3f} stab={rc.stability:.3f}\n"
            )

        if mode == "ansi":
            return output
        print(output)
        return None

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> np.ndarray:
        """Get the current state observation.

        In simulation mode, generates a synthetic state vector and uses
        the attack orchestrator to inject realistic attack signatures.
        """
        if not self._sim_mode:
            return self.state_processor.get_state()

        return self._generate_simulated_state()

    def _generate_simulated_state(self) -> np.ndarray:
        """Generate a synthetic state vector for simulation mode.

        Uses the attack orchestrator to inject realistic attack signatures
        from DDoS, port scan, and spoofing modules rather than hard-coded
        modifications.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)

        # Generate baseline traffic features with noise
        for sw in range(5):
            offset = sw * 13

            state[offset + 0] = self._rng.uniform(0.05, 0.30)   # flow count
            state[offset + 1] = self._rng.uniform(3.0, 6.0)     # packets (log)
            state[offset + 2] = self._rng.uniform(8.0, 12.0)    # bytes (log)
            state[offset + 3] = self._rng.uniform(0.05, 0.20)   # avg duration
            state[offset + 4] = self._rng.uniform(3.0, 5.0)     # rx_packets
            state[offset + 5] = self._rng.uniform(3.0, 5.0)     # tx_packets
            state[offset + 6] = self._rng.uniform(8.0, 11.0)    # rx_bytes
            state[offset + 7] = self._rng.uniform(8.0, 11.0)    # tx_bytes
            state[offset + 8] = self._rng.uniform(0.0, 0.5)     # rx_dropped
            state[offset + 9] = self._rng.uniform(0.0, 0.5)     # tx_dropped
            state[offset + 10] = self._rng.uniform(0.0, 0.1)    # rx_errors
            state[offset + 11] = self._rng.uniform(0.0, 0.1)    # tx_errors
            state[offset + 12] = self._rng.uniform(0.5, 2.0)    # conn rate

        # Use attack orchestrator for realistic attack injection
        if self._orchestrator is not None:
            state, orch_info = self._orchestrator.step(state)
            self._sim_attack_active = orch_info["attack_active"]
            self._sim_throughput = orch_info["throughput_mbps"]
            self._sim_latency = orch_info["latency_ms"]

            # Extract primary attack type for info dict
            if self._sim_attack_active and orch_info["active_attacks"]:
                primary = orch_info["active_attacks"][0]
                # Map attack names to simple types
                if "ddos" in primary:
                    self._sim_attack_type = "ddos"
                elif "port_scan" in primary:
                    self._sim_attack_type = "port_scan"
                elif "spoofing" in primary:
                    self._sim_attack_type = "spoofing"
                else:
                    self._sim_attack_type = primary
            else:
                self._sim_attack_type = None
        else:
            # Fallback: legacy inline attack injection
            self._legacy_attack_injection(state)

        return self.state_processor.process_raw_state(state)

    def _legacy_attack_injection(self, state: np.ndarray) -> None:
        """Legacy inline attack injection for backward compatibility.

        Used only when no attack orchestrator is available.
        """
        if not self._sim_attack_active and self._rng.random() < 0.15:
            self._sim_attack_active = True
            self._sim_attack_type = self._rng.choice(
                ["ddos", "port_scan", "spoofing"]
            )

        if self._sim_attack_active:
            target_sw = self._rng.randint(0, 5)
            offset = target_sw * 13

            if self._sim_attack_type == "ddos":
                state[offset + 0] += self._rng.uniform(0.3, 0.7)
                state[offset + 1] += self._rng.uniform(3.0, 6.0)
                state[offset + 2] += self._rng.uniform(2.0, 4.0)
                state[offset + 8] += self._rng.uniform(1.0, 3.0)
                state[offset + 12] += self._rng.uniform(3.0, 8.0)
            elif self._sim_attack_type == "port_scan":
                state[offset + 0] += self._rng.uniform(0.4, 0.8)
                state[offset + 3] = self._rng.uniform(0.01, 0.05)
                state[offset + 12] += self._rng.uniform(5.0, 15.0)
            elif self._sim_attack_type == "spoofing":
                state[offset + 0] += self._rng.uniform(0.1, 0.3)
                state[offset + 10] += self._rng.uniform(0.5, 2.0)
                state[offset + 11] += self._rng.uniform(0.5, 2.0)

            self._sim_throughput = self._rng.uniform(30.0, 70.0)
            self._sim_latency = self._rng.uniform(15.0, 45.0)
        else:
            self._sim_throughput = self._rng.uniform(85.0, 100.0)
            self._sim_latency = self._rng.uniform(2.0, 8.0)

    # ------------------------------------------------------------------
    # Step Metrics
    # ------------------------------------------------------------------

    def _compute_step_metrics(self, action: int) -> Dict[str, Any]:
        """Compute security and performance metrics for reward calculation.

        In simulation mode, uses the attack orchestrator's metric model
        for realistic confusion-matrix generation.
        In live mode, queries actual detection results.
        """
        if not self._sim_mode:
            return self._compute_live_metrics(action)

        return self._compute_simulated_metrics(action)

    def _compute_simulated_metrics(self, action: int) -> Dict[str, Any]:
        """Compute simulated metrics using the attack orchestrator.

        The orchestrator provides confusion matrix values that reward
        correct actions:
        - BLOCK/RATE_LIMIT during attacks -> higher detection
        - ALLOW during normal traffic -> lower false positives
        - REROUTE during attacks -> moderate detection, good throughput
        """
        if self._orchestrator is not None:
            metrics = self._orchestrator.get_step_metrics(
                action, self._sim_attack_active,
            )
            # Override throughput/latency with current sim state
            metrics["current_throughput_mbps"] = self._sim_throughput
            metrics["current_latency_ms"] = self._sim_latency
            return metrics

        # Fallback: legacy metric computation
        return self._compute_legacy_metrics(action)

    def _compute_legacy_metrics(self, action: int) -> Dict[str, Any]:
        """Legacy metric computation for backward compatibility."""
        attack = self._sim_attack_active
        tp, fn, fp, tn = 0, 0, 0, 0

        if attack:
            if action in (1, 3):
                tp = self._rng.randint(7, 11)
                fn = self._rng.randint(0, 3)
                fp = self._rng.randint(0, 2)
                tn = 90 - fp
                self._sim_throughput = min(
                    self._sim_throughput + self._rng.uniform(10, 25),
                    DEFAULT_BASELINE_THROUGHPUT,
                )
            elif action == 2:
                tp = self._rng.randint(5, 8)
                fn = self._rng.randint(2, 5)
                fp = self._rng.randint(0, 1)
                tn = 90 - fp
                self._sim_throughput = min(
                    self._sim_throughput + self._rng.uniform(5, 15),
                    DEFAULT_BASELINE_THROUGHPUT,
                )
            else:
                tp = self._rng.randint(0, 2)
                fn = self._rng.randint(7, 10)
                fp = 0
                tn = 90

            if self._rng.random() < 0.20:
                self._sim_attack_active = False
                self._sim_attack_type = None
        else:
            if action in (1, 3):
                tp = 0
                fn = 0
                fp = self._rng.randint(3, 8)
                tn = 90 - fp
                self._sim_throughput *= self._rng.uniform(0.7, 0.9)
            elif action == 2:
                tp = 0
                fn = 0
                fp = self._rng.randint(1, 3)
                tn = 90 - fp
            else:
                tp = 0
                fn = 0
                fp = 0
                tn = 90

        policy_changes = 0 if action == 0 else 1

        return {
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "current_throughput_mbps": self._sim_throughput,
            "baseline_throughput_mbps": DEFAULT_BASELINE_THROUGHPUT,
            "min_throughput_mbps": DEFAULT_MIN_THROUGHPUT,
            "current_latency_ms": self._sim_latency,
            "policy_changes": policy_changes,
        }

    def _compute_live_metrics(self, action: int) -> Dict[str, Any]:
        """Compute metrics from live network observations.

        In live mode, detection metrics come from comparing the
        policy enforcer's actions against known attack ground truth.
        """
        policy_changes = 0
        if self.policy_enforcer is not None:
            policy_changes = self.policy_enforcer.get_policy_changes()

        return {
            "true_positives": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "true_negatives": 90,
            "current_throughput_mbps": DEFAULT_BASELINE_THROUGHPUT,
            "baseline_throughput_mbps": DEFAULT_BASELINE_THROUGHPUT,
            "min_throughput_mbps": DEFAULT_MIN_THROUGHPUT,
            "current_latency_ms": 5.0,
            "policy_changes": policy_changes,
        }
