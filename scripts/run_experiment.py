"""Full experiment orchestrator for Sprint 7 comprehensive evaluation.

Runs all 7 controlled experiments with multiple random seeds, computes
mean ± standard deviation across seeds, and saves structured results to
results/experiments/.

Experiments:
    1. DQN vs Static Policy Baseline      -- all 4 attack scenarios
    2. PPO vs Static Policy Baseline      -- all 4 attack scenarios
    3. DQN vs PPO Head-to-Head            -- same scenarios, direct comparison
    4. Zero-Trust Overhead Measurement    -- with / without OpenZiti overhead
    5. Scalability Test                   -- varying attack_probability (0.1→0.5)
    6. Mixed Attack Resilience            -- varying max_concurrent_attacks (1→3)
    7. Policy Stability Test              -- long-duration episodes (500 steps)

Usage:
    python -m scripts.run_experiment                        # all experiments
    python -m scripts.run_experiment --experiment 3         # head-to-head only
    python -m scripts.run_experiment --quick                # 10 eps/seed (fast)
    python -m scripts.run_experiment --seeds 42 123 456     # explicit seeds
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── project root on path ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.attacks.mixed_scenario import MixedScenarioConfig, ScenarioType
from src.environment.network_env import NetworkSecurityEnv
from src.utils.config_loader import get_hyperparameters, get_reward_config
from src.utils.logger import setup_logging, get_logger
from src.utils.metrics import (
    detection_rate,
    false_positive_rate,
    throughput_degradation,
    latency_overhead,
    evaluate_thresholds,
)

logger = get_logger(__name__)

# ── constants ─────────────────────────────────────────────────────────────
DEFAULT_SEEDS: List[int] = [42, 123, 456]
DEFAULT_EPISODES: int = 30
DEFAULT_MAX_STEPS: int = 200
STABILITY_MAX_STEPS: int = 500
STABILITY_EPISODES: int = 20
SCALABILITY_EPISODES: int = 20
RESILIENCE_EPISODES: int = 20

ATTACK_PROBS: List[float] = [0.10, 0.20, 0.30, 0.40, 0.50]
CONCURRENT_LEVELS: List[int] = [1, 2, 3]
OPENZITI_LATENCY_OVERHEAD_MS: float = 5.0   # simulated mTLS round-trip delta
OPENZITI_THROUGHPUT_PENALTY_PCT: float = 2.0  # encryption overhead

EVAL_SCENARIOS: List[Tuple[str, ScenarioType]] = [
    ("ddos",      ScenarioType.SINGLE_DDOS),
    ("port_scan", ScenarioType.SINGLE_PORTSCAN),
    ("spoofing",  ScenarioType.SINGLE_SPOOFING),
    ("mixed",     ScenarioType.ALL_COMBINED),
]

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "experiments")


# ── agent loading ─────────────────────────────────────────────────────────

def load_dqn(reward_config: Dict[str, Any]) -> DQNAgent:
    """Load trained DQN agent from checkpoint."""
    cfg = get_hyperparameters("dqn")
    agent = DQNAgent(state_dim=65, config=cfg)
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "dqn")
    if os.path.isdir(ckpt):
        try:
            agent.load(ckpt)
            logger.info("DQN loaded from %s", ckpt)
        except Exception as exc:
            logger.warning("DQN checkpoint load failed (%s) — using untrained agent.", exc)
    else:
        logger.warning("DQN checkpoint dir not found: %s", ckpt)
    return agent


def load_ppo(reward_config: Dict[str, Any]) -> PPOAgent:
    """Load trained PPO agent from checkpoint."""
    cfg = get_hyperparameters("ppo")
    agent = PPOAgent(state_dim=65, config=cfg)
    ckpt = os.path.join(PROJECT_ROOT, "checkpoints", "ppo")
    if os.path.isdir(ckpt):
        try:
            agent.load(ckpt)
            logger.info("PPO loaded from %s", ckpt)
        except Exception as exc:
            logger.warning("PPO checkpoint load failed (%s) — using untrained agent.", exc)
    else:
        logger.warning("PPO checkpoint dir not found: %s", ckpt)
    return agent


# ── action selection ──────────────────────────────────────────────────────

def _select_action(agent: Any, state: np.ndarray, agent_type: str) -> int:
    """Return a discrete action from either DQN or PPO agent."""
    if agent_type == "dqn":
        return agent.select_action(state, greedy=True)
    # PPO returns (continuous_action, log_prob, value)
    cont, _, _ = agent.select_action(state, deterministic=True)
    return PPOAgent.continuous_to_discrete(cont)


def _static_action(_state: np.ndarray) -> int:
    """Static baseline: always ALLOW (action 0)."""
    return 0


# ── core evaluation engine ────────────────────────────────────────────────

def _run_single_seed(
    agent: Any,
    agent_type: str,          # "dqn" | "ppo" | "static"
    scenario_type: ScenarioType,
    episodes: int,
    max_steps: int,
    reward_config: Dict[str, Any],
    seed: int,
    attack_probability: float = 0.20,
    max_concurrent: int = 2,
    latency_offset_ms: float = 0.0,
    throughput_penalty_pct: float = 0.0,
) -> Dict[str, float]:
    """Run *episodes* evaluation episodes for one agent/scenario/seed.

    Args:
        agent:               Trained agent (or None for static).
        agent_type:          "dqn", "ppo", or "static".
        scenario_type:       Attack scenario to use.
        episodes:            Number of episodes to run.
        max_steps:           Maximum steps per episode.
        reward_config:       Reward weight dictionary.
        seed:                Environment random seed.
        attack_probability:  Probability of attack at each step.
        max_concurrent:      Maximum simultaneous attack types.
        latency_offset_ms:   Extra latency to simulate overhead (exp 4).
        throughput_penalty:  Extra throughput reduction % (exp 4).

    Returns:
        Dict of aggregated scalar metrics for this seed.
    """
    attack_cfg = MixedScenarioConfig(
        scenario_type=scenario_type,
        attack_probability=attack_probability,
        max_concurrent_attacks=max_concurrent,
        min_gap_steps=3,
        max_gap_steps=20,
        normal_traffic_steps=5,
        seed=seed,
    )

    all_tp = all_fn = all_fp = all_tn = 0
    all_rewards: List[float] = []
    all_throughputs: List[float] = []
    all_latencies: List[float] = []
    all_adapt_times: List[float] = []
    all_policy_switches: List[int] = []

    for ep in range(episodes):
        ep_seed = seed + ep
        env = NetworkSecurityEnv(
            reward_config=reward_config,
            max_steps=max_steps,
            seed=ep_seed,
            attack_config=attack_cfg,
        )
        state, _ = env.reset()
        ep_reward = 0.0
        first_attack: Optional[int] = None
        first_response: Optional[int] = None
        prev_action: Optional[int] = None
        action_switches = 0

        for step in range(1, max_steps + 1):
            if agent_type == "static":
                action = _static_action(state)
            else:
                action = _select_action(agent, state, agent_type)

            # Count policy oscillation
            if prev_action is not None and action != prev_action:
                action_switches += 1
            prev_action = action

            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            m = info["metrics"]
            all_tp += m["true_positives"]
            all_fn += m["false_negatives"]
            all_fp += m["false_positives"]
            all_tn += m["true_negatives"]

            # Apply overhead offsets (experiment 4)
            thr = m["current_throughput_mbps"] * (1.0 - throughput_penalty_pct / 100.0)
            lat = m["current_latency_ms"] + latency_offset_ms
            all_throughputs.append(thr)
            all_latencies.append(lat)

            # Adaptation timing
            if info.get("attack_active", False):
                if first_attack is None:
                    first_attack = step
                if action in (1, 2, 3) and first_response is None:
                    first_response = step

            state = next_state
            if terminated or truncated:
                break

        all_rewards.append(ep_reward)
        all_policy_switches.append(action_switches)
        if first_attack is not None and first_response is not None:
            all_adapt_times.append(float(first_response - first_attack))

    avg_thr = float(np.mean(all_throughputs)) if all_throughputs else 100.0
    avg_lat = float(np.mean(all_latencies)) if all_latencies else 5.0
    avg_adapt = float(np.mean(all_adapt_times)) if all_adapt_times else 0.0
    avg_switches = float(np.mean(all_policy_switches)) if all_policy_switches else 0.0

    return {
        "detection_rate":            detection_rate(all_tp, all_fn),
        "false_positive_rate":       false_positive_rate(all_fp, all_tn),
        "avg_reward":                float(np.mean(all_rewards)),
        "std_reward":                float(np.std(all_rewards)),
        "avg_throughput_mbps":       avg_thr,
        "throughput_degradation_pct": throughput_degradation(avg_thr, 100.0),
        "avg_latency_ms":            avg_lat,
        "latency_overhead_ms":       latency_overhead(avg_lat, 5.0),
        "adaptation_speed_s":        avg_adapt,
        "avg_policy_switches":       avg_switches,
        "oscillation_rate":          avg_switches / max(max_steps, 1),
        "total_tp": all_tp, "total_fn": all_fn,
        "total_fp": all_fp, "total_tn": all_tn,
    }


def _run_multi_seed(
    agent: Any,
    agent_type: str,
    scenario_type: ScenarioType,
    episodes: int,
    max_steps: int,
    reward_config: Dict[str, Any],
    seeds: List[int],
    **kwargs: Any,
) -> Dict[str, float]:
    """Run evaluation across multiple seeds; return mean ± std aggregation."""
    per_seed: List[Dict[str, float]] = []
    for seed in seeds:
        result = _run_single_seed(
            agent, agent_type, scenario_type,
            episodes, max_steps, reward_config, seed, **kwargs,
        )
        per_seed.append(result)

    scalar_keys = [k for k, v in per_seed[0].items() if isinstance(v, (int, float))]
    agg: Dict[str, float] = {}
    n = len(per_seed)
    for k in scalar_keys:
        vals = [r[k] for r in per_seed]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        agg[k] = mean_v
        agg[f"{k}_std"] = std_v
        agg[f"{k}_ci95"] = 1.96 * std_v / np.sqrt(n) if n > 1 else 0.0
    return agg


# ── result persistence ────────────────────────────────────────────────────

def _save_csv(data: List[Dict[str, Any]], path: str) -> None:
    """Save list-of-dicts to CSV."""
    if not data:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    logger.info("Saved CSV: %s", path)


def _save_json(data: Any, path: str) -> None:
    """Save data to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved JSON: %s", path)


# ── experiment implementations ────────────────────────────────────────────

def experiment_1_dqn_vs_baseline(
    dqn_agent: DQNAgent,
    reward_config: Dict[str, Any],
    seeds: List[int],
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Exp 1: DQN vs Static Policy Baseline across all attack scenarios."""
    logger.info("=" * 60)
    logger.info("Experiment 1: DQN vs Static Baseline")
    logger.info("=" * 60)
    rows: List[Dict[str, Any]] = []

    for scenario_name, scenario_type in EVAL_SCENARIOS:
        logger.info("  Scenario: %s", scenario_name)
        dqn_m = _run_multi_seed(
            dqn_agent, "dqn", scenario_type, episodes, max_steps, reward_config, seeds,
        )
        static_m = _run_multi_seed(
            None, "static", scenario_type, episodes, max_steps, reward_config, seeds,
        )

        improvement_pp = (dqn_m["detection_rate"] - static_m["detection_rate"]) * 100
        logger.info(
            "    DQN det=%.1f%% | Static det=%.1f%% | Δ=+%.1f pp",
            dqn_m["detection_rate"] * 100,
            static_m["detection_rate"] * 100,
            improvement_pp,
        )

        rows.append({
            "scenario": scenario_name, "agent": "DQN",
            **{k: round(v, 4) for k, v in dqn_m.items()},
        })
        rows.append({
            "scenario": scenario_name, "agent": "Static",
            **{k: round(v, 4) for k, v in static_m.items()},
        })

    _save_csv(rows, os.path.join(RESULTS_DIR, "exp1_dqn_vs_baseline.csv"))
    return {"rows": rows}


def experiment_2_ppo_vs_baseline(
    ppo_agent: PPOAgent,
    reward_config: Dict[str, Any],
    seeds: List[int],
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Exp 2: PPO vs Static Policy Baseline across all attack scenarios."""
    logger.info("=" * 60)
    logger.info("Experiment 2: PPO vs Static Baseline")
    logger.info("=" * 60)
    rows: List[Dict[str, Any]] = []

    for scenario_name, scenario_type in EVAL_SCENARIOS:
        logger.info("  Scenario: %s", scenario_name)
        ppo_m = _run_multi_seed(
            ppo_agent, "ppo", scenario_type, episodes, max_steps, reward_config, seeds,
        )
        static_m = _run_multi_seed(
            None, "static", scenario_type, episodes, max_steps, reward_config, seeds,
        )
        logger.info(
            "    PPO det=%.1f%% | Static det=%.1f%%",
            ppo_m["detection_rate"] * 100,
            static_m["detection_rate"] * 100,
        )
        rows.append({"scenario": scenario_name, "agent": "PPO", **{k: round(v, 4) for k, v in ppo_m.items()}})
        rows.append({"scenario": scenario_name, "agent": "Static", **{k: round(v, 4) for k, v in static_m.items()}})

    _save_csv(rows, os.path.join(RESULTS_DIR, "exp2_ppo_vs_baseline.csv"))
    return {"rows": rows}


def experiment_3_head_to_head(
    dqn_agent: DQNAgent,
    ppo_agent: PPOAgent,
    reward_config: Dict[str, Any],
    seeds: List[int],
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Exp 3: DQN vs PPO head-to-head, same seeds and scenarios."""
    logger.info("=" * 60)
    logger.info("Experiment 3: DQN vs PPO Head-to-Head")
    logger.info("=" * 60)
    rows: List[Dict[str, Any]] = []

    for scenario_name, scenario_type in EVAL_SCENARIOS:
        logger.info("  Scenario: %s", scenario_name)
        dqn_m = _run_multi_seed(
            dqn_agent, "dqn", scenario_type, episodes, max_steps, reward_config, seeds,
        )
        ppo_m = _run_multi_seed(
            ppo_agent, "ppo", scenario_type, episodes, max_steps, reward_config, seeds,
        )
        logger.info(
            "    DQN det=%.1f%% ± %.1f%%  |  PPO det=%.1f%% ± %.1f%%",
            dqn_m["detection_rate"] * 100, dqn_m["detection_rate_std"] * 100,
            ppo_m["detection_rate"] * 100, ppo_m["detection_rate_std"] * 100,
        )
        rows.append({"scenario": scenario_name, "agent": "DQN", **{k: round(v, 4) for k, v in dqn_m.items()}})
        rows.append({"scenario": scenario_name, "agent": "PPO", **{k: round(v, 4) for k, v in ppo_m.items()}})

    _save_csv(rows, os.path.join(RESULTS_DIR, "exp3_head_to_head.csv"))
    return {"rows": rows}


def experiment_4_zero_trust_overhead(
    dqn_agent: DQNAgent,
    ppo_agent: PPOAgent,
    reward_config: Dict[str, Any],
    seeds: List[int],
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Exp 4: Measure zero-trust (OpenZiti) network overhead.

    Simulates OpenZiti overlay by applying:
    - +5 ms latency per step (mTLS round-trip overhead)
    - +2% throughput reduction (encryption CPU overhead)

    Compares without-ZT vs with-ZT for both agents.
    """
    logger.info("=" * 60)
    logger.info("Experiment 4: Zero-Trust Overhead Measurement")
    logger.info("=" * 60)
    rows: List[Dict[str, Any]] = []
    scenario_name = "mixed"
    scenario_type = ScenarioType.ALL_COMBINED

    for agent_obj, agent_type in [(dqn_agent, "dqn"), (ppo_agent, "ppo")]:
        # Without OpenZiti
        no_zt = _run_multi_seed(
            agent_obj, agent_type, scenario_type, episodes, max_steps, reward_config, seeds,
            latency_offset_ms=0.0, throughput_penalty_pct=0.0,
        )
        # With simulated OpenZiti overhead
        with_zt = _run_multi_seed(
            agent_obj, agent_type, scenario_type, episodes, max_steps, reward_config, seeds,
            latency_offset_ms=OPENZITI_LATENCY_OVERHEAD_MS,
            throughput_penalty_pct=OPENZITI_THROUGHPUT_PENALTY_PCT,
        )
        lat_delta = with_zt["avg_latency_ms"] - no_zt["avg_latency_ms"]
        thr_delta = with_zt["throughput_degradation_pct"] - no_zt["throughput_degradation_pct"]
        logger.info(
            "  %s → Latency overhead: +%.1f ms | Throughput penalty: +%.1f%%",
            agent_type.upper(), lat_delta, thr_delta,
        )
        rows.append({"agent": agent_type.upper(), "zero_trust": False, **{k: round(v, 4) for k, v in no_zt.items()}})
        rows.append({"agent": agent_type.upper(), "zero_trust": True, **{k: round(v, 4) for k, v in with_zt.items()}})

    _save_csv(rows, os.path.join(RESULTS_DIR, "exp4_zero_trust_overhead.csv"))
    return {"rows": rows, "zt_latency_ms": OPENZITI_LATENCY_OVERHEAD_MS}


def experiment_5_scalability(
    dqn_agent: DQNAgent,
    ppo_agent: PPOAgent,
    reward_config: Dict[str, Any],
    seeds: List[int],
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Exp 5: Scalability — detection rate vs increasing attack load."""
    logger.info("=" * 60)
    logger.info("Experiment 5: Scalability Test (attack_probability sweep)")
    logger.info("=" * 60)
    rows: List[Dict[str, Any]] = []
    scenario_type = ScenarioType.ALL_COMBINED

    for prob in ATTACK_PROBS:
        for agent_obj, agent_type in [(dqn_agent, "dqn"), (ppo_agent, "ppo")]:
            m = _run_multi_seed(
                agent_obj, agent_type, scenario_type,
                episodes, max_steps, reward_config, seeds,
                attack_probability=prob,
            )
            logger.info(
                "  attack_prob=%.2f  %s  det=%.1f%% ± %.1f%%  fp=%.2f%%",
                prob, agent_type.upper(),
                m["detection_rate"] * 100, m["detection_rate_std"] * 100,
                m["false_positive_rate"] * 100,
            )
            rows.append({
                "attack_probability": prob,
                "agent": agent_type.upper(),
                **{k: round(v, 4) for k, v in m.items()},
            })

    _save_csv(rows, os.path.join(RESULTS_DIR, "exp5_scalability.csv"))
    return {"rows": rows, "attack_probs": ATTACK_PROBS}


def experiment_6_resilience(
    dqn_agent: DQNAgent,
    ppo_agent: PPOAgent,
    reward_config: Dict[str, Any],
    seeds: List[int],
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Exp 6: Resilience — detection vs max concurrent attacks (1→3)."""
    logger.info("=" * 60)
    logger.info("Experiment 6: Mixed Attack Resilience")
    logger.info("=" * 60)
    rows: List[Dict[str, Any]] = []
    scenario_type = ScenarioType.ALL_COMBINED

    for concurrent in CONCURRENT_LEVELS:
        for agent_obj, agent_type in [(dqn_agent, "dqn"), (ppo_agent, "ppo")]:
            m = _run_multi_seed(
                agent_obj, agent_type, scenario_type,
                episodes, max_steps, reward_config, seeds,
                max_concurrent=concurrent,
            )
            logger.info(
                "  concurrent=%d  %s  det=%.1f%% ± %.1f%%",
                concurrent, agent_type.upper(),
                m["detection_rate"] * 100, m["detection_rate_std"] * 100,
            )
            rows.append({
                "max_concurrent": concurrent,
                "agent": agent_type.upper(),
                **{k: round(v, 4) for k, v in m.items()},
            })

    _save_csv(rows, os.path.join(RESULTS_DIR, "exp6_resilience.csv"))
    return {"rows": rows, "concurrent_levels": CONCURRENT_LEVELS}


def experiment_7_stability(
    dqn_agent: DQNAgent,
    ppo_agent: PPOAgent,
    reward_config: Dict[str, Any],
    seeds: List[int],
    episodes: int,
    max_steps: int,
) -> Dict[str, Any]:
    """Exp 7: Policy Stability — long-duration episodes, measure oscillation."""
    logger.info("=" * 60)
    logger.info("Experiment 7: Policy Stability (long-duration runs)")
    logger.info("=" * 60)
    rows: List[Dict[str, Any]] = []
    scenario_type = ScenarioType.ALL_COMBINED
    long_steps = STABILITY_MAX_STEPS

    for agent_obj, agent_type in [(dqn_agent, "dqn"), (ppo_agent, "ppo")]:
        m = _run_multi_seed(
            agent_obj, agent_type, scenario_type,
            episodes, long_steps, reward_config, seeds,
        )
        logger.info(
            "  %s  oscillation_rate=%.4f  det=%.1f%%  reward=%.2f",
            agent_type.upper(),
            m["oscillation_rate"],
            m["detection_rate"] * 100,
            m["avg_reward"],
        )
        rows.append({
            "agent": agent_type.upper(),
            "max_steps": long_steps,
            **{k: round(v, 4) for k, v in m.items()},
        })

    _save_csv(rows, os.path.join(RESULTS_DIR, "exp7_stability.csv"))
    return {"rows": rows, "max_steps": long_steps}


# ── summary builder ───────────────────────────────────────────────────────

def build_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key statistics from all experiments into a single summary."""
    summary: Dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Exp 1 — DQN vs baseline (mixed scenario)
    if "exp1" in results:
        rows = results["exp1"]["rows"]
        dqn_mixed  = next((r for r in rows if r["scenario"] == "mixed" and r["agent"] == "DQN"),  {})
        stat_mixed = next((r for r in rows if r["scenario"] == "mixed" and r["agent"] == "Static"), {})
        summary["exp1"] = {
            "dqn_detection_mean": dqn_mixed.get("detection_rate"),
            "dqn_detection_std":  dqn_mixed.get("detection_rate_std"),
            "static_detection":   stat_mixed.get("detection_rate"),
            "improvement_pp":     round(
                (dqn_mixed.get("detection_rate", 0) - stat_mixed.get("detection_rate", 0)) * 100, 2
            ),
        }

    # Exp 2 — PPO vs baseline
    if "exp2" in results:
        rows = results["exp2"]["rows"]
        ppo_mixed  = next((r for r in rows if r["scenario"] == "mixed" and r["agent"] == "PPO"),    {})
        stat_mixed = next((r for r in rows if r["scenario"] == "mixed" and r["agent"] == "Static"), {})
        summary["exp2"] = {
            "ppo_detection_mean": ppo_mixed.get("detection_rate"),
            "ppo_detection_std":  ppo_mixed.get("detection_rate_std"),
            "static_detection":   stat_mixed.get("detection_rate"),
            "improvement_pp":     round(
                (ppo_mixed.get("detection_rate", 0) - stat_mixed.get("detection_rate", 0)) * 100, 2
            ),
        }

    # Exp 3 — head-to-head
    if "exp3" in results:
        rows = results["exp3"]["rows"]
        dqn_m = next((r for r in rows if r["scenario"] == "mixed" and r["agent"] == "dqn"), {})
        ppo_m = next((r for r in rows if r["scenario"] == "mixed" and r["agent"] == "ppo"), {})
        summary["exp3"] = {
            "dqn": {k: dqn_m.get(k) for k in ["detection_rate", "detection_rate_std", "false_positive_rate", "avg_reward"]},
            "ppo": {k: ppo_m.get(k) for k in ["detection_rate", "detection_rate_std", "false_positive_rate", "avg_reward"]},
        }

    # Exp 4 — zero-trust overhead
    if "exp4" in results:
        rows = results["exp4"]["rows"]
        dqn_no  = next((r for r in rows if r["agent"] == "DQN"  and not r.get("zero_trust")), {})
        dqn_zt  = next((r for r in rows if r["agent"] == "DQN"  and r.get("zero_trust")),     {})
        summary["exp4"] = {
            "zt_latency_overhead_ms":    results["exp4"].get("zt_latency_ms"),
            "dqn_latency_no_zt":         dqn_no.get("avg_latency_ms"),
            "dqn_latency_with_zt":       dqn_zt.get("avg_latency_ms"),
            "detection_unchanged":        dqn_no.get("detection_rate") == dqn_zt.get("detection_rate"),
        }

    # Exp 5 — scalability extremes
    if "exp5" in results:
        rows = results["exp5"]["rows"]
        low  = [r for r in rows if r["attack_probability"] == 0.10 and r["agent"] == "DQN"]
        high = [r for r in rows if r["attack_probability"] == 0.50 and r["agent"] == "DQN"]
        if low and high:
            summary["exp5"] = {
                "detection_at_0.10": low[0].get("detection_rate"),
                "detection_at_0.50": high[0].get("detection_rate"),
                "graceful_degradation": high[0].get("detection_rate", 0) >= 0.70,
            }

    # Exp 6 — resilience
    if "exp6" in results:
        rows = results["exp6"]["rows"]
        c1 = next((r for r in rows if r["max_concurrent"] == 1 and r["agent"] == "DQN"), {})
        c3 = next((r for r in rows if r["max_concurrent"] == 3 and r["agent"] == "DQN"), {})
        summary["exp6"] = {
            "detection_1_concurrent": c1.get("detection_rate"),
            "detection_3_concurrent": c3.get("detection_rate"),
        }

    # Exp 7 — stability
    if "exp7" in results:
        rows = results["exp7"]["rows"]
        dqn_s = next((r for r in rows if r["agent"] == "DQN"), {})
        ppo_s = next((r for r in rows if r["agent"] == "PPO"), {})
        summary["exp7"] = {
            "dqn_oscillation_rate": dqn_s.get("oscillation_rate"),
            "ppo_oscillation_rate": ppo_s.get("oscillation_rate"),
        }

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sprint 7 comprehensive experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "1", "2", "3", "4", "5", "6", "7"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Random seeds (default: 42 123 456)",
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help="Episodes per seed per scenario (default: 30)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS,
        help="Max steps per episode (default: 200)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 10 episodes, 1 seed (for testing)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(level=args.log_level)

    if args.quick:
        seeds = [42]
        episodes = 10
        max_steps = 200
        logger.info("QUICK mode: 1 seed, 10 episodes")
    else:
        seeds = args.seeds
        episodes = args.episodes
        max_steps = args.max_steps

    os.makedirs(RESULTS_DIR, exist_ok=True)
    logger.info("Results directory: %s", RESULTS_DIR)
    logger.info("Seeds: %s | Episodes: %d | Max steps: %d", seeds, episodes, max_steps)

    # Load configs and agents
    reward_config_dqn = get_reward_config("dqn")
    reward_config = reward_config_dqn

    t0 = time.time()
    logger.info("Loading DQN agent...")
    dqn_agent = load_dqn(reward_config)
    logger.info("Loading PPO agent...")
    ppo_agent = load_ppo(reward_config)

    run = args.experiment
    all_results: Dict[str, Any] = {}

    if run in ("all", "1"):
        all_results["exp1"] = experiment_1_dqn_vs_baseline(
            dqn_agent, reward_config, seeds, episodes, max_steps,
        )

    if run in ("all", "2"):
        all_results["exp2"] = experiment_2_ppo_vs_baseline(
            ppo_agent, reward_config, seeds, episodes, max_steps,
        )

    if run in ("all", "3"):
        all_results["exp3"] = experiment_3_head_to_head(
            dqn_agent, ppo_agent, reward_config, seeds, episodes, max_steps,
        )

    if run in ("all", "4"):
        all_results["exp4"] = experiment_4_zero_trust_overhead(
            dqn_agent, ppo_agent, reward_config, seeds, episodes, max_steps,
        )

    if run in ("all", "5"):
        all_results["exp5"] = experiment_5_scalability(
            dqn_agent, ppo_agent, reward_config, seeds,
            SCALABILITY_EPISODES if not args.quick else 5, max_steps,
        )

    if run in ("all", "6"):
        all_results["exp6"] = experiment_6_resilience(
            dqn_agent, ppo_agent, reward_config, seeds,
            RESILIENCE_EPISODES if not args.quick else 5, max_steps,
        )

    if run in ("all", "7"):
        all_results["exp7"] = experiment_7_stability(
            dqn_agent, ppo_agent, reward_config, seeds,
            STABILITY_EPISODES if not args.quick else 5,
            STABILITY_MAX_STEPS,
        )

    # Build and save summary
    summary = build_summary(all_results)
    _save_json(summary, os.path.join(RESULTS_DIR, "summary.json"))

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("All experiments complete in %.1f minutes.", elapsed / 60)
    logger.info("Results in: %s", RESULTS_DIR)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
