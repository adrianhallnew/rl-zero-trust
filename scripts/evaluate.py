"""Evaluation script for the RL-driven adaptive security system.

Loads a trained DQN or PPO model and runs evaluation episodes against each
attack type (DDoS, port scan, spoofing, mixed). Computes all security
and performance metrics against the defined thresholds, generates
publication-quality charts, and exports results to CSV.

Usage:
    # Evaluate DQN (default, backward compatible)
    python -m scripts.evaluate

    # Evaluate PPO agent
    python -m scripts.evaluate --agent ppo

    # Evaluate with static baseline comparison
    python -m scripts.evaluate --agent ppo --static-baseline

    # Evaluate specific checkpoint
    python -m scripts.evaluate --agent dqn --checkpoint checkpoints/dqn/dqn_model_ep250.keras

    # Custom evaluation episodes
    python -m scripts.evaluate --episodes 50 --max-steps 200
"""

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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

# Attack scenarios to evaluate
EVAL_SCENARIOS = [
    ("ddos", ScenarioType.SINGLE_DDOS),
    ("port_scan", ScenarioType.SINGLE_PORTSCAN),
    ("spoofing", ScenarioType.SINGLE_SPOOFING),
    ("mixed", ScenarioType.ALL_COMBINED),
]

# Agent-specific defaults
AGENT_DEFAULTS = {
    "dqn": {
        "checkpoint": "checkpoints/dqn",
        "results_dir": "results/dqn",
        "config_key": "dqn",
    },
    "ppo": {
        "checkpoint": "checkpoints/ppo",
        "results_dir": "results/ppo",
        "config_key": "ppo",
    },
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained RL agent against attack scenarios",
    )
    parser.add_argument(
        "--agent", type=str, default="dqn",
        choices=["dqn", "ppo"],
        help="Agent type to evaluate (default: dqn)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint directory or .keras file "
             "(default: checkpoints/{agent})",
    )
    parser.add_argument(
        "--episodes", type=int, default=30,
        help="Number of evaluation episodes per scenario",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None,
        help="Directory to save evaluation results (default: results/{agent})",
    )
    parser.add_argument(
        "--charts-dir", type=str, default="results/charts",
        help="Directory to save charts",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--static-baseline", action="store_true",
        help="Also run static baseline comparison (always ALLOW)",
    )
    args = parser.parse_args()

    # Apply agent-specific defaults if not explicitly set
    defaults = AGENT_DEFAULTS[args.agent]
    if args.checkpoint is None:
        args.checkpoint = defaults["checkpoint"]
    if args.results_dir is None:
        args.results_dir = defaults["results_dir"]

    return args


def select_agent_action(agent, state: np.ndarray, agent_type: str) -> int:
    """Select a discrete action from the agent in evaluation mode.

    Handles the different action selection APIs for DQN and PPO agents.

    Args:
        agent: Trained DQN or PPO agent instance.
        state: Current observation vector.
        agent_type: Either "dqn" or "ppo".

    Returns:
        Discrete action in {0, 1, 2, 3}.
    """
    if agent_type == "dqn":
        return agent.select_action(state, greedy=True)
    else:
        # PPO returns (continuous_action, log_prob, value)
        continuous_action, _, _ = agent.select_action(
            state, deterministic=True,
        )
        return PPOAgent.continuous_to_discrete(continuous_action)


def run_evaluation_episodes(
    agent,
    agent_type: str,
    scenario_type: ScenarioType,
    num_episodes: int,
    max_steps: int,
    reward_config: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Run evaluation episodes for a specific attack scenario.

    Args:
        agent: Trained DQN or PPO agent instance.
        agent_type: Either "dqn" or "ppo".
        scenario_type: Attack scenario to evaluate against.
        num_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
        reward_config: Reward configuration dictionary.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with aggregated evaluation metrics.
    """
    # Aggregated metrics
    all_tp, all_fn, all_fp, all_tn = 0, 0, 0, 0
    all_rewards = []
    all_throughputs = []
    all_latencies = []
    all_adaptation_times = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    episode_attack_steps = []
    episode_correct_actions = []

    # Per-episode tracking for attack-type-specific detection
    attack_tp = {"ddos": 0, "port_scan": 0, "spoofing": 0}
    attack_fn = {"ddos": 0, "port_scan": 0, "spoofing": 0}

    for ep in range(num_episodes):
        ep_seed = seed + ep

        attack_config = MixedScenarioConfig(
            scenario_type=scenario_type,
            attack_probability=0.20,
            max_concurrent_attacks=2,
            min_gap_steps=3,
            max_gap_steps=20,
            normal_traffic_steps=5,
            seed=ep_seed,
        )

        env = NetworkSecurityEnv(
            reward_config=reward_config,
            max_steps=max_steps,
            seed=ep_seed,
            attack_config=attack_config,
        )

        state, info = env.reset()
        ep_reward = 0.0
        ep_attack_steps = 0
        ep_correct = 0
        first_attack_step = None
        first_response_step = None

        for step in range(1, max_steps + 1):
            action = select_agent_action(agent, state, agent_type)
            next_state, reward, terminated, truncated, step_info = env.step(action)

            ep_reward += reward
            action_counts[action] += 1

            # Track metrics from step info
            metrics = step_info["metrics"]
            all_tp += metrics["true_positives"]
            all_fn += metrics["false_negatives"]
            all_fp += metrics["false_positives"]
            all_tn += metrics["true_negatives"]
            all_throughputs.append(metrics["current_throughput_mbps"])
            all_latencies.append(metrics["current_latency_ms"])

            # Track attack-specific detection
            attack_active = step_info.get("attack_active", False)
            attack_type = step_info.get("attack_type", None)

            if attack_active:
                ep_attack_steps += 1
                if first_attack_step is None:
                    first_attack_step = step

                # Track per-type detection
                if attack_type and attack_type in attack_tp:
                    if action in (1, 2, 3):  # Any defensive action
                        attack_tp[attack_type] += 1
                        if first_response_step is None:
                            first_response_step = step
                    else:
                        attack_fn[attack_type] += 1

                # Track correct actions
                if action in (1, 3):  # BLOCK or RATE_LIMIT
                    ep_correct += 1
            else:
                if action == 0:  # ALLOW during normal
                    ep_correct += 1

            state = next_state
            if terminated or truncated:
                break

        all_rewards.append(ep_reward)
        episode_attack_steps.append(ep_attack_steps)
        episode_correct_actions.append(ep_correct / max(step, 1))

        # Adaptation speed: steps from attack start to first defensive action
        if first_attack_step is not None and first_response_step is not None:
            adaptation_steps = first_response_step - first_attack_step
            # Assume ~1 second per step for timing
            all_adaptation_times.append(float(adaptation_steps))

    # Compute aggregate metrics
    overall_dr = detection_rate(all_tp, all_fn)
    overall_fpr = false_positive_rate(all_fp, all_tn)
    avg_throughput = float(np.mean(all_throughputs)) if all_throughputs else 100.0
    avg_latency = float(np.mean(all_latencies)) if all_latencies else 5.0
    avg_adaptation = float(np.mean(all_adaptation_times)) if all_adaptation_times else 30.0

    # Per-type detection rates
    type_detection = {}
    for atype in ["ddos", "port_scan", "spoofing"]:
        tp_t = attack_tp[atype]
        fn_t = attack_fn[atype]
        type_detection[atype] = detection_rate(tp_t, fn_t)

    total_actions = sum(action_counts.values()) or 1

    return {
        "detection_rate_overall": overall_dr,
        "detection_rate_ddos": type_detection.get("ddos", 0.0),
        "detection_rate_portscan": type_detection.get("port_scan", 0.0),
        "detection_rate_spoofing": type_detection.get("spoofing", 0.0),
        "false_positive_rate": overall_fpr,
        "avg_throughput_mbps": avg_throughput,
        "throughput_degradation_pct": throughput_degradation(avg_throughput, 100.0),
        "avg_latency_ms": avg_latency,
        "latency_overhead_ms": latency_overhead(avg_latency, 5.0),
        "adaptation_speed_s": avg_adaptation,
        "avg_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "avg_correct_action_pct": float(np.mean(episode_correct_actions)) * 100,
        "action_distribution": {
            k: v / total_actions for k, v in action_counts.items()
        },
        "total_episodes": num_episodes,
        "total_attack_steps": sum(episode_attack_steps),
        "confusion_matrix": {
            "tp": all_tp, "fn": all_fn, "fp": all_fp, "tn": all_tn,
        },
    }


def run_static_baseline(
    scenario_type: ScenarioType,
    num_episodes: int,
    max_steps: int,
    reward_config: Dict[str, Any],
    seed: int = 42,
) -> Dict[str, Any]:
    """Run a static baseline (always ALLOW) for comparison.

    Args:
        scenario_type: Attack scenario.
        num_episodes: Number of episodes.
        max_steps: Steps per episode.
        reward_config: Reward config.
        seed: Random seed.

    Returns:
        Aggregated metrics for the static baseline.
    """
    all_tp, all_fn, all_fp, all_tn = 0, 0, 0, 0
    all_rewards = []

    for ep in range(num_episodes):
        ep_seed = seed + ep + 10000  # Different seed offset

        attack_config = MixedScenarioConfig(
            scenario_type=scenario_type,
            attack_probability=0.20,
            max_concurrent_attacks=2,
            min_gap_steps=3,
            max_gap_steps=20,
            normal_traffic_steps=5,
            seed=ep_seed,
        )

        env = NetworkSecurityEnv(
            reward_config=reward_config,
            max_steps=max_steps,
            seed=ep_seed,
            attack_config=attack_config,
        )

        state, _ = env.reset()
        ep_reward = 0.0

        for step in range(1, max_steps + 1):
            action = 0  # Static baseline: always ALLOW
            _, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward

            metrics = step_info["metrics"]
            all_tp += metrics["true_positives"]
            all_fn += metrics["false_negatives"]
            all_fp += metrics["false_positives"]
            all_tn += metrics["true_negatives"]

            if terminated or truncated:
                break

        all_rewards.append(ep_reward)

    return {
        "detection_rate_overall": detection_rate(all_tp, all_fn),
        "false_positive_rate": false_positive_rate(all_fp, all_tn),
        "avg_reward": float(np.mean(all_rewards)),
        "confusion_matrix": {
            "tp": all_tp, "fn": all_fn, "fp": all_fp, "tn": all_tn,
        },
    }


def generate_charts(
    results: Dict[str, Dict[str, Any]],
    baseline_results: Dict[str, Dict[str, Any]],
    charts_dir: str,
    agent_name: str = "DQN",
) -> None:
    """Generate publication-quality evaluation charts.

    Args:
        results: Per-scenario evaluation results.
        baseline_results: Static baseline results for comparison.
        charts_dir: Output directory for charts.
        agent_name: Agent display name for chart titles (e.g. "DQN", "PPO").
    """
    from src.utils.visualization import _save_figure, _ensure_output_dir
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_output_dir(charts_dir)
    prefix = agent_name.lower()

    # --- 1. Detection Rate by Attack Type ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    attack_types = ["DDoS", "Port Scan", "Spoofing", "Overall"]

    # Get rates from the mixed scenario (most comprehensive)
    mixed = results.get("mixed", {})
    agent_rates = [
        mixed.get("detection_rate_ddos", 0) * 100,
        mixed.get("detection_rate_portscan", 0) * 100,
        mixed.get("detection_rate_spoofing", 0) * 100,
        mixed.get("detection_rate_overall", 0) * 100,
    ]
    thresholds_min = [75, 70, 65, 70]  # Minimum acceptable

    x = np.arange(len(attack_types))
    width = 0.35

    color = "#4CAF50" if agent_name == "PPO" else "#2196F3"
    bars1 = ax.bar(x - width / 2, agent_rates, width, label=f"{agent_name} Agent",
                    color=color, edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width / 2, thresholds_min, width, label="Min. Threshold",
                    color="#FF9800", edgecolor="black", alpha=0.6)

    for bar, val in zip(bars1, agent_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xlabel("Attack Type")
    ax.set_ylabel("Detection Rate (%)")
    ax.set_title(f"{agent_name} Detection Rate by Attack Type", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(attack_types)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save_figure(fig, f"{prefix}_detection_by_type", charts_dir)

    # --- 2. Confusion Matrix Heatmap ---
    cm = mixed.get("confusion_matrix", {"tp": 0, "fn": 0, "fp": 0, "tn": 0})
    cm_array = np.array([
        [cm["tp"], cm["fn"]],
        [cm["fp"], cm["tn"]],
    ])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    im = ax.imshow(cm_array, cmap="Blues", aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Attack", "Predicted Normal"])
    ax.set_yticklabels(["Actual Attack", "Actual Normal"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_array[i, j]}", ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm_array[i, j] > cm_array.max() / 2 else "black")

    ax.set_title(f"{agent_name} Confusion Matrix (Mixed Scenario)", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save_figure(fig, f"{prefix}_confusion_matrix", charts_dir)

    # --- 3. Agent vs Static Baseline ---
    if baseline_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        baseline_mixed = baseline_results.get("mixed", {})
        methods = [f"{agent_name} Agent", "Static Baseline"]
        det_rates = [
            mixed.get("detection_rate_overall", 0) * 100,
            baseline_mixed.get("detection_rate_overall", 0) * 100,
        ]
        fp_rates = [
            mixed.get("false_positive_rate", 0) * 100,
            baseline_mixed.get("false_positive_rate", 0) * 100,
        ]

        colors = [color, "#9E9E9E"]

        bars = axes[0].bar(methods, det_rates, color=colors, edgecolor="black", alpha=0.85)
        for bar, val in zip(bars, det_rates):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=12)
        axes[0].set_ylabel("Detection Rate (%)")
        axes[0].set_title("Detection Rate Comparison", fontweight="bold")
        axes[0].set_ylim(0, 110)
        axes[0].axhline(y=70, color="r", linestyle="--", label="Min. threshold (70%)")
        axes[0].legend()
        axes[0].grid(True, axis="y", alpha=0.3)

        bars = axes[1].bar(methods, fp_rates, color=colors, edgecolor="black", alpha=0.85)
        for bar, val in zip(bars, fp_rates):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f"{val:.1f}%", ha="center", va="bottom", fontsize=12)
        axes[1].set_ylabel("False Positive Rate (%)")
        axes[1].set_title("False Positive Rate Comparison", fontweight="bold")
        axes[1].set_ylim(0, max(max(fp_rates) * 1.5, 20))
        axes[1].axhline(y=15, color="r", linestyle="--", label="Max. threshold (15%)")
        axes[1].legend()
        axes[1].grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        _save_figure(fig, f"{prefix}_vs_static_baseline", charts_dir)

    logger.info("Evaluation charts saved to %s", charts_dir)


def evaluate(args: argparse.Namespace) -> None:
    """Run the full evaluation pipeline."""
    setup_logging(level=args.log_level)

    agent_type = args.agent
    defaults = AGENT_DEFAULTS[agent_type]
    agent_name = agent_type.upper()

    config = get_hyperparameters(defaults["config_key"])
    reward_config = get_reward_config(defaults["config_key"])
    seed = config.get("seed", 42)

    logger.info("=" * 60)
    logger.info("%s Evaluation -- RL-Driven Adaptive Security System", agent_name)
    logger.info("=" * 60)
    logger.info("Agent: %s | Episodes per scenario: %d | Steps/episode: %d",
                agent_name, args.episodes, args.max_steps)

    # Load trained agent
    if agent_type == "dqn":
        agent = DQNAgent(state_dim=65, config=config)
        checkpoint_path = args.checkpoint
        if os.path.isdir(checkpoint_path):
            agent.load(checkpoint_path)
        elif os.path.isfile(checkpoint_path):
            agent.load(os.path.dirname(checkpoint_path))
        else:
            logger.warning(
                "No checkpoint found at %s. Using untrained agent.",
                checkpoint_path,
            )
    elif agent_type == "ppo":
        agent = PPOAgent(state_dim=65, config=config)
        checkpoint_path = args.checkpoint
        if os.path.isdir(checkpoint_path):
            agent.load(checkpoint_path)
        elif os.path.isfile(checkpoint_path):
            agent.load(os.path.dirname(checkpoint_path))
        else:
            logger.warning(
                "No checkpoint found at %s. Using untrained agent.",
                checkpoint_path,
            )

    # Run evaluation for each scenario
    results = {}
    baseline_results = {}

    for scenario_name, scenario_type in EVAL_SCENARIOS:
        logger.info("-" * 40)
        logger.info("Evaluating scenario: %s", scenario_name)

        result = run_evaluation_episodes(
            agent=agent,
            agent_type=agent_type,
            scenario_type=scenario_type,
            num_episodes=args.episodes,
            max_steps=args.max_steps,
            reward_config=reward_config,
            seed=seed,
        )
        results[scenario_name] = result

        # Evaluate against thresholds
        threshold_results = evaluate_thresholds({
            "detection_rate_ddos": result["detection_rate_ddos"],
            "detection_rate_portscan": result["detection_rate_portscan"],
            "detection_rate_spoofing": result["detection_rate_spoofing"],
            "detection_rate_overall": result["detection_rate_overall"],
            "false_positive_rate": result["false_positive_rate"],
            "adaptation_speed_s": result["adaptation_speed_s"],
            "throughput_degradation_pct": result["throughput_degradation_pct"],
            "latency_overhead_ms": result["latency_overhead_ms"],
        })

        # Log results
        logger.info("  Detection Rate (overall): %.1f%% [%s]",
                     result["detection_rate_overall"] * 100,
                     threshold_results.get("detection_rate_overall", {}).get("rating", "N/A"))
        logger.info("  Detection Rate (DDoS): %.1f%%",
                     result["detection_rate_ddos"] * 100)
        logger.info("  Detection Rate (Port Scan): %.1f%%",
                     result["detection_rate_portscan"] * 100)
        logger.info("  Detection Rate (Spoofing): %.1f%%",
                     result["detection_rate_spoofing"] * 100)
        logger.info("  False Positive Rate: %.1f%% [%s]",
                     result["false_positive_rate"] * 100,
                     threshold_results.get("false_positive_rate", {}).get("rating", "N/A"))
        logger.info("  Adaptation Speed: %.1fs [%s]",
                     result["adaptation_speed_s"],
                     threshold_results.get("adaptation_speed_s", {}).get("rating", "N/A"))
        logger.info("  Avg Reward: %.4f (std=%.4f)",
                     result["avg_reward"], result["std_reward"])

        # Static baseline
        if args.static_baseline:
            baseline = run_static_baseline(
                scenario_type=scenario_type,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                reward_config=reward_config,
                seed=seed,
            )
            baseline_results[scenario_name] = baseline
            logger.info("  [Baseline] Detection: %.1f%%, FP: %.1f%%",
                         baseline["detection_rate_overall"] * 100,
                         baseline["false_positive_rate"] * 100)

    # Save results to CSV
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "evaluation_results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "scenario", "detection_overall", "detection_ddos",
            "detection_portscan", "detection_spoofing",
            "false_positive_rate", "adaptation_speed_s",
            "throughput_degradation_pct", "latency_overhead_ms",
            "avg_reward", "std_reward",
        ])
        for name, r in results.items():
            writer.writerow([
                name,
                round(r["detection_rate_overall"], 4),
                round(r["detection_rate_ddos"], 4),
                round(r["detection_rate_portscan"], 4),
                round(r["detection_rate_spoofing"], 4),
                round(r["false_positive_rate"], 4),
                round(r["adaptation_speed_s"], 2),
                round(r["throughput_degradation_pct"], 2),
                round(r["latency_overhead_ms"], 2),
                round(r["avg_reward"], 4),
                round(r["std_reward"], 4),
            ])

    logger.info("Results saved to: %s", csv_path)

    # Generate charts
    generate_charts(results, baseline_results, args.charts_dir, agent_name)

    # Print summary table
    logger.info("=" * 60)
    logger.info("%s EVALUATION SUMMARY", agent_name)
    logger.info("=" * 60)
    logger.info("%-15s %-10s %-10s %-10s %-10s %-8s",
                "Scenario", "Det.Rate", "FP Rate", "Adapt.Spd", "Reward", "Status")
    logger.info("-" * 60)

    all_pass = True
    for name, r in results.items():
        det = r["detection_rate_overall"]
        fpr = r["false_positive_rate"]
        status = "PASS" if det >= 0.70 and fpr <= 0.15 else "FAIL"
        if status == "FAIL":
            all_pass = False
        logger.info("%-15s %-10.1f%% %-10.1f%% %-10.1fs %-10.3f %s",
                     name, det * 100, fpr * 100,
                     r["adaptation_speed_s"], r["avg_reward"], status)

    logger.info("=" * 60)
    logger.info("Overall: %s", "ALL PASS" if all_pass else "SOME FAILURES")
    logger.info("=" * 60)


if __name__ == "__main__":
    evaluate(parse_args())
