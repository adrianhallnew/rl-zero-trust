"""Comparative analysis of DQN vs PPO agents.

Loads evaluation results from both agents, computes 6 key metrics,
determines a winner per metric, checks the PPO parity requirement
(≥3 of 6 metrics), and generates comparative charts and a markdown report.

Usage:
    python -m scripts.compare_agents
    python -m scripts.compare_agents --dqn-results results/dqn/evaluation_results.csv
    python -m scripts.compare_agents --ppo-results results/ppo/evaluation_results.csv
"""

import argparse
import csv
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to path for absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.visualization import (
    plot_comparative_detection,
    plot_comparative_learning_curves,
    plot_comparative_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Thresholds from project specification
THRESHOLDS = {
    "detection_rate": 0.70,       # >= 70%
    "false_positive_rate": 0.15,  # <= 15%
    "adaptation_speed_s": 30.0,   # <= 30 seconds
}

# 6 key comparison metrics and whether higher is better
METRIC_SPEC = {
    "detection_overall": {"label": "Detection Rate", "higher_better": True},
    "false_positive_rate": {"label": "False Positive Rate", "higher_better": False},
    "adaptation_speed_s": {"label": "Adaptation Speed", "higher_better": False},
    "throughput_degradation_pct": {"label": "Throughput Degradation", "higher_better": False},
    "latency_overhead_ms": {"label": "Latency Overhead", "higher_better": False},
    "avg_reward": {"label": "Average Reward", "higher_better": True},
}


def load_evaluation_results(csv_path: str) -> Dict[str, Dict[str, float]]:
    """Load evaluation results CSV into a dict of scenario -> metrics.

    Args:
        csv_path: Path to evaluation_results.csv.

    Returns:
        Dict mapping scenario name -> metric dict.
    """
    results = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scenario = row["scenario"]
            results[scenario] = {
                k: float(v) for k, v in row.items() if k != "scenario"
            }
    return results


def compute_aggregate_metrics(
    results: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute aggregate metrics across all scenarios.

    Uses the 'mixed' scenario as primary (if available), otherwise
    averages across all scenarios.

    Args:
        results: Scenario -> metrics dict.

    Returns:
        Aggregated metric dict.
    """
    if "mixed" in results:
        return results["mixed"].copy()

    # Average across scenarios
    all_keys = list(next(iter(results.values())).keys())
    agg = {}
    for key in all_keys:
        values = [r[key] for r in results.values() if key in r]
        agg[key] = np.mean(values)
    return agg


def compare_metrics(
    dqn_metrics: Dict[str, float],
    ppo_metrics: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Compare DQN vs PPO across the 6 key metrics.

    Args:
        dqn_metrics: Aggregated DQN metrics.
        ppo_metrics: Aggregated PPO metrics.

    Returns:
        List of dicts with metric name, DQN value, PPO value, winner.
    """
    comparisons = []
    for metric_key, spec in METRIC_SPEC.items():
        dqn_val = dqn_metrics.get(metric_key, 0.0)
        ppo_val = ppo_metrics.get(metric_key, 0.0)

        if spec["higher_better"]:
            winner = "PPO" if ppo_val >= dqn_val else "DQN"
        else:
            winner = "PPO" if ppo_val <= dqn_val else "DQN"

        comparisons.append({
            "metric": spec["label"],
            "key": metric_key,
            "dqn": dqn_val,
            "ppo": ppo_val,
            "winner": winner,
            "higher_better": spec["higher_better"],
        })

    return comparisons


def check_ppo_parity(comparisons: List[Dict[str, Any]]) -> Tuple[bool, int]:
    """Check if PPO matches or exceeds DQN on >= 3 of 6 metrics.

    Args:
        comparisons: Output of compare_metrics().

    Returns:
        (passed, ppo_wins) tuple.
    """
    ppo_wins = sum(1 for c in comparisons if c["winner"] == "PPO")
    return ppo_wins >= 3, ppo_wins


def check_thresholds(
    metrics: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """Check if metrics meet project thresholds.

    Args:
        metrics: Aggregated metrics dict.

    Returns:
        Dict mapping threshold name -> {value, threshold, passed}.
    """
    results = {}
    dr = metrics.get("detection_overall", 0.0)
    results["detection_rate"] = {
        "value": dr, "threshold": 0.70, "passed": dr >= 0.70,
    }
    fp = metrics.get("false_positive_rate", 0.0)
    results["false_positive_rate"] = {
        "value": fp, "threshold": 0.15, "passed": fp <= 0.15,
    }
    adapt = metrics.get("adaptation_speed_s", 999.0)
    results["adaptation_speed"] = {
        "value": adapt, "threshold": 30.0, "passed": adapt <= 30.0,
    }
    return results


def generate_report(
    dqn_results: Dict[str, Dict[str, float]],
    ppo_results: Dict[str, Dict[str, float]],
    dqn_agg: Dict[str, float],
    ppo_agg: Dict[str, float],
    comparisons: List[Dict[str, Any]],
    parity_passed: bool,
    ppo_wins: int,
    output_dir: str,
) -> str:
    """Generate a markdown comparison report.

    Args:
        dqn_results: Per-scenario DQN results.
        ppo_results: Per-scenario PPO results.
        dqn_agg: Aggregated DQN metrics.
        ppo_agg: Aggregated PPO metrics.
        comparisons: Metric comparisons.
        parity_passed: Whether PPO parity check passed.
        ppo_wins: Number of metrics PPO wins.
        output_dir: Output directory.

    Returns:
        Path to report file.
    """
    report_path = os.path.join(output_dir, "comparative_report.md")

    dqn_thresholds = check_thresholds(dqn_agg)
    ppo_thresholds = check_thresholds(ppo_agg)

    lines = [
        "# DQN vs PPO Comparative Analysis",
        "",
        "## Summary",
        "",
        f"PPO parity check: **{'PASSED' if parity_passed else 'FAILED'}** "
        f"({ppo_wins}/6 metrics favoring PPO, threshold: 3/6)",
        "",
        "## Metric Comparison (Mixed Scenario)",
        "",
        "| Metric | DQN | PPO | Winner |",
        "|--------|-----|-----|--------|",
    ]

    for c in comparisons:
        star = " ★" if c["winner"] == "PPO" else ""
        if c["key"] in ("detection_overall",):
            dqn_str = f"{c['dqn']:.1%}"
            ppo_str = f"{c['ppo']:.1%}"
        elif c["key"] == "false_positive_rate":
            dqn_str = f"{c['dqn']:.2%}"
            ppo_str = f"{c['ppo']:.2%}"
        elif c["key"] == "avg_reward":
            dqn_str = f"{c['dqn']:.2f}"
            ppo_str = f"{c['ppo']:.2f}"
        elif c["key"] == "latency_overhead_ms":
            dqn_str = f"{c['dqn']:.1f} ms"
            ppo_str = f"{c['ppo']:.1f} ms"
        elif c["key"] == "throughput_degradation_pct":
            dqn_str = f"{c['dqn']:.1f}%"
            ppo_str = f"{c['ppo']:.1f}%"
        else:
            dqn_str = f"{c['dqn']:.2f}"
            ppo_str = f"{c['ppo']:.2f}"

        lines.append(
            f"| {c['metric']} | {dqn_str} | {ppo_str} | "
            f"{c['winner']}{star} |"
        )

    lines += [
        "",
        "## Threshold Compliance",
        "",
        "| Threshold | DQN | PPO | Requirement |",
        "|-----------|-----|-----|-------------|",
    ]

    for name in ["detection_rate", "false_positive_rate", "adaptation_speed"]:
        dqn_t = dqn_thresholds[name]
        ppo_t = ppo_thresholds[name]
        if name == "detection_rate":
            req = "≥ 70%"
            dqn_val = f"{dqn_t['value']:.1%} {'✓' if dqn_t['passed'] else '✗'}"
            ppo_val = f"{ppo_t['value']:.1%} {'✓' if ppo_t['passed'] else '✗'}"
        elif name == "false_positive_rate":
            req = "≤ 15%"
            dqn_val = f"{dqn_t['value']:.2%} {'✓' if dqn_t['passed'] else '✗'}"
            ppo_val = f"{ppo_t['value']:.2%} {'✓' if ppo_t['passed'] else '✗'}"
        else:
            req = "≤ 30s"
            dqn_val = f"{dqn_t['value']:.1f}s {'✓' if dqn_t['passed'] else '✗'}"
            ppo_val = f"{ppo_t['value']:.1f}s {'✓' if ppo_t['passed'] else '✗'}"

        lines.append(
            f"| {name.replace('_', ' ').title()} | {dqn_val} | "
            f"{ppo_val} | {req} |"
        )

    lines += [
        "",
        "## Per-Scenario Detection Rates",
        "",
        "| Scenario | DQN | PPO |",
        "|----------|-----|-----|",
    ]

    for scenario in ["ddos", "port_scan", "spoofing", "mixed"]:
        dqn_dr = dqn_results.get(scenario, {}).get("detection_overall", 0.0)
        ppo_dr = ppo_results.get(scenario, {}).get("detection_overall", 0.0)
        lines.append(f"| {scenario} | {dqn_dr:.1%} | {ppo_dr:.1%} |")

    lines += [
        "",
        "## Per-Scenario Average Rewards",
        "",
        "| Scenario | DQN | PPO |",
        "|----------|-----|-----|",
    ]

    for scenario in ["ddos", "port_scan", "spoofing", "mixed"]:
        dqn_r = dqn_results.get(scenario, {}).get("avg_reward", 0.0)
        ppo_r = ppo_results.get(scenario, {}).get("avg_reward", 0.0)
        lines.append(f"| {scenario} | {dqn_r:.2f} | {ppo_r:.2f} |")

    lines += ["", "---", f"*Generated by scripts/compare_agents.py*", ""]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare DQN and PPO agent evaluation results.",
    )
    parser.add_argument(
        "--dqn-results",
        default="results/dqn/evaluation_results.csv",
        help="Path to DQN evaluation CSV.",
    )
    parser.add_argument(
        "--ppo-results",
        default="results/ppo/evaluation_results.csv",
        help="Path to PPO evaluation CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/comparative",
        help="Output directory for charts and report.",
    )
    parser.add_argument(
        "--dqn-training-log",
        default="results/dqn/training_log.csv",
        help="Path to DQN training log CSV (for learning curves).",
    )
    parser.add_argument(
        "--ppo-training-log",
        default="results/ppo/training_log.csv",
        help="Path to PPO training log CSV (for learning curves).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    logger.info("Loading DQN results from: %s", args.dqn_results)
    dqn_results = load_evaluation_results(args.dqn_results)

    logger.info("Loading PPO results from: %s", args.ppo_results)
    ppo_results = load_evaluation_results(args.ppo_results)

    # Aggregate metrics (use mixed scenario)
    dqn_agg = compute_aggregate_metrics(dqn_results)
    ppo_agg = compute_aggregate_metrics(ppo_results)

    # Compare
    comparisons = compare_metrics(dqn_agg, ppo_agg)
    parity_passed, ppo_wins = check_ppo_parity(comparisons)

    logger.info("=" * 60)
    logger.info("DQN vs PPO Comparison Results")
    logger.info("=" * 60)
    for c in comparisons:
        logger.info(
            "  %-25s  DQN=%.4f  PPO=%.4f  Winner=%s",
            c["metric"], c["dqn"], c["ppo"], c["winner"],
        )
    logger.info("-" * 60)
    logger.info(
        "PPO parity: %s (%d/6 metrics)",
        "PASSED" if parity_passed else "FAILED", ppo_wins,
    )

    # Generate charts
    logger.info("Generating comparative charts...")

    # 1. Detection rate comparison
    dqn_det = {
        "ddos": dqn_results["ddos"]["detection_ddos"],
        "port_scan": dqn_results["port_scan"]["detection_portscan"],
        "spoofing": dqn_results["spoofing"]["detection_spoofing"],
        "overall": dqn_agg["detection_overall"],
    }
    ppo_det = {
        "ddos": ppo_results["ddos"]["detection_ddos"],
        "port_scan": ppo_results["port_scan"]["detection_portscan"],
        "spoofing": ppo_results["spoofing"]["detection_spoofing"],
        "overall": ppo_agg["detection_overall"],
    }
    chart1 = plot_comparative_detection(dqn_det, ppo_det, args.output_dir)
    logger.info("  Detection chart: %s", chart1)

    # 2. Learning curves (from training logs)
    try:
        dqn_log = np.genfromtxt(
            args.dqn_training_log, delimiter=",", skip_header=1,
        )
        ppo_log = np.genfromtxt(
            args.ppo_training_log, delimiter=",", skip_header=1,
        )
        dqn_rewards = dqn_log[:, 2]  # total_reward column
        ppo_rewards = ppo_log[:, 2]
        chart2 = plot_comparative_learning_curves(
            dqn_rewards, ppo_rewards, window=20, output_dir=args.output_dir,
        )
        logger.info("  Learning curves: %s", chart2)
    except Exception as e:
        logger.warning("  Could not generate learning curves: %s", e)

    # 3. Summary comparison
    chart3 = plot_comparative_summary(dqn_agg, ppo_agg, args.output_dir)
    logger.info("  Summary chart: %s", chart3)

    # Generate report
    report_path = generate_report(
        dqn_results, ppo_results,
        dqn_agg, ppo_agg,
        comparisons, parity_passed, ppo_wins,
        args.output_dir,
    )
    logger.info("Report saved to: %s", report_path)

    # Save comparison CSV
    csv_path = os.path.join(args.output_dir, "comparison_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "dqn", "ppo", "winner"],
        )
        writer.writeheader()
        for c in comparisons:
            writer.writerow({
                "metric": c["metric"],
                "dqn": c["dqn"],
                "ppo": c["ppo"],
                "winner": c["winner"],
            })
    logger.info("Comparison CSV: %s", csv_path)

    return 0 if parity_passed else 1


if __name__ == "__main__":
    sys.exit(main())
