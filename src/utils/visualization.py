"""Matplotlib chart generators for the RL-driven adaptive security system.

Provides visualization utilities for:
- Network topology diagrams
- Training curves (reward, loss, Q-values)
- Performance metrics (detection rate, throughput, latency)
- Comparative charts (DQN vs PPO)

All charts are publication-quality with proper labels, legends, and font sizes.
Output formats: PNG (300 DPI) and SVG.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless/container use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# Publication-quality defaults
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 7),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "charts")


def _ensure_output_dir(output_dir: Optional[str] = None) -> str:
    """Ensure the output directory exists.

    Args:
        output_dir: Directory path. Uses default results/charts if None.

    Returns:
        Resolved directory path.
    """
    path = output_dir or RESULTS_DIR
    os.makedirs(path, exist_ok=True)
    return path


def _save_figure(fig: plt.Figure, filename: str, output_dir: Optional[str] = None) -> str:
    """Save a figure in both PNG and SVG formats.

    Args:
        fig: Matplotlib figure to save.
        filename: Base filename without extension.
        output_dir: Output directory. Uses default if None.

    Returns:
        Path to the saved PNG file.
    """
    path = _ensure_output_dir(output_dir)

    png_path = os.path.join(path, f"{filename}.png")
    svg_path = os.path.join(path, f"{filename}.svg")

    fig.savefig(png_path, format="png")
    fig.savefig(svg_path, format="svg")
    plt.close(fig)

    logger.info("Saved chart: %s (.png + .svg)", filename)
    return png_path


def draw_topology(output_dir: Optional[str] = None) -> str:
    """Generate a network topology visualization diagram.

    Draws the zero-trust network topology with:
    - 5 OVS switches (s1 core, s2-s5 edge)
    - 15 hosts (h1-h15, 3 per switch)
    - Link connections
    - IP address labels

    Args:
        output_dir: Directory to save the figure. Uses default if None.

    Returns:
        Path to the saved PNG file.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(-1.5, 11.5)
    ax.set_ylim(-1.5, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Zero-Trust Network Topology\n5 OVS Switches, 15 Hosts (OpenFlow 1.3)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    # --- Switch positions ---
    switch_positions = {
        "s1": (5, 7),       # Core switch (top center)
        "s2": (1, 4),       # Edge switches (row below)
        "s3": (3.5, 4),
        "s4": (6.5, 4),
        "s5": (9, 4),
    }

    # --- Host positions (3 per switch, below each switch) ---
    host_positions = {}
    host_ip = {}

    # s1 hosts: h1, h2, h3 (above and to sides of s1)
    for i, (hx, hy) in enumerate([(3.5, 8.5), (5, 8.5), (6.5, 8.5)]):
        h_name = f"h{i + 1}"
        host_positions[h_name] = (hx, hy)
        host_ip[h_name] = f"10.0.0.{i + 1}"

    # s2-s5 hosts: below each edge switch
    host_idx = 4
    for sw_name in ["s2", "s3", "s4", "s5"]:
        sx, sy = switch_positions[sw_name]
        offsets = [(-0.8, -1.5), (0, -1.5), (0.8, -1.5)]
        for dx, dy in offsets:
            h_name = f"h{host_idx}"
            host_positions[h_name] = (sx + dx, sy + dy)
            host_ip[h_name] = f"10.0.0.{host_idx}"
            host_idx += 1

    # --- Draw links ---
    # Core to edge switch links
    for sw_name in ["s2", "s3", "s4", "s5"]:
        x1, y1 = switch_positions["s1"]
        x2, y2 = switch_positions[sw_name]
        ax.plot([x1, x2], [y1, y2], "b-", linewidth=2, alpha=0.6, zorder=1)

    # Switch to host links
    all_switch_hosts = {
        "s1": ["h1", "h2", "h3"],
        "s2": ["h4", "h5", "h6"],
        "s3": ["h7", "h8", "h9"],
        "s4": ["h10", "h11", "h12"],
        "s5": ["h13", "h14", "h15"],
    }
    for sw_name, hosts in all_switch_hosts.items():
        sx, sy = switch_positions[sw_name]
        for h_name in hosts:
            hx, hy = host_positions[h_name]
            ax.plot([sx, hx], [sy, hy], "g-", linewidth=1.2, alpha=0.5, zorder=1)

    # --- Draw switches ---
    for sw_name, (x, y) in switch_positions.items():
        color = "#2196F3" if sw_name == "s1" else "#42A5F5"
        label_extra = " (core)" if sw_name == "s1" else ""
        rect = mpatches.FancyBboxPatch(
            (x - 0.4, y - 0.3), 0.8, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="black",
            linewidth=1.5,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, y, f"{sw_name}{label_extra}", ha="center", va="center",
                fontsize=11, fontweight="bold", color="white", zorder=4)

    # --- Draw hosts ---
    for h_name, (x, y) in host_positions.items():
        circle = plt.Circle((x, y), 0.3, facecolor="#66BB6A", edgecolor="black",
                            linewidth=1, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.02, h_name, ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=4)
        ax.text(x, y - 0.55, host_ip[h_name], ha="center", va="center",
                fontsize=7, color="#555555", zorder=4)

    # --- Draw controller ---
    ctrl_x, ctrl_y = 9.5, 7.5
    rect = mpatches.FancyBboxPatch(
        (ctrl_x - 0.7, ctrl_y - 0.3), 1.4, 0.6,
        boxstyle="round,pad=0.1",
        facecolor="#FF7043",
        edgecolor="black",
        linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(rect)
    ax.text(ctrl_x, ctrl_y, "Ryu Controller", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=4)
    ax.text(ctrl_x, ctrl_y - 0.55, "172.20.0.20:6633", ha="center", va="center",
            fontsize=7, color="#555555", zorder=4)

    # Dashed line from controller to core switch
    ax.annotate(
        "", xy=(switch_positions["s1"][0] + 0.4, switch_positions["s1"][1]),
        xytext=(ctrl_x - 0.7, ctrl_y),
        arrowprops=dict(arrowstyle="->", color="#FF7043", lw=2, ls="dashed"),
        zorder=2,
    )
    ax.text(7.5, 7.6, "OpenFlow 1.3", ha="center", va="center",
            fontsize=8, color="#FF7043", style="italic", zorder=4)

    # --- Legend ---
    legend_elements = [
        mpatches.Patch(facecolor="#2196F3", edgecolor="black", label="Core Switch"),
        mpatches.Patch(facecolor="#42A5F5", edgecolor="black", label="Edge Switch"),
        mpatches.Patch(facecolor="#66BB6A", edgecolor="black", label="Host"),
        mpatches.Patch(facecolor="#FF7043", edgecolor="black", label="SDN Controller"),
        plt.Line2D([0], [0], color="b", linewidth=2, alpha=0.6, label="Trunk Link (100Mbps)"),
        plt.Line2D([0], [0], color="g", linewidth=1.2, alpha=0.5, label="Access Link (100Mbps)"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=9,
              framealpha=0.9, edgecolor="gray")

    # --- Annotation ---
    ax.text(0.5, 0.3,
            "Link params: 100 Mbps, 2ms delay, 0% loss\n"
            "All switches: OVS with OpenFlow 1.3\n"
            "Host subnet: 10.0.0.0/24",
            ha="left", va="bottom", fontsize=8, color="#777777",
            transform=ax.transAxes)

    return _save_figure(fig, "network_topology", output_dir)


def plot_training_reward(
    rewards: List[float],
    title: str = "Training Reward Curve",
    filename: str = "training_reward",
    window: int = 20,
    output_dir: Optional[str] = None,
) -> str:
    """Plot training reward over episodes with moving average.

    Args:
        rewards: List of per-episode total rewards.
        title: Chart title.
        filename: Output filename (without extension).
        window: Moving average window size.
        output_dir: Output directory.

    Returns:
        Path to saved PNG file.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    episodes = range(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.3, color="blue", label="Per-episode")

    # Moving average
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(rewards) + 1), ma, color="red",
                linewidth=2, label=f"{window}-episode moving avg")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save_figure(fig, filename, output_dir)


def plot_throughput_comparison(
    labels: List[str],
    throughputs: List[float],
    title: str = "Throughput Comparison",
    filename: str = "throughput_comparison",
    output_dir: Optional[str] = None,
) -> str:
    """Plot a bar chart comparing throughput across host pairs.

    Args:
        labels: List of test labels (e.g., "h1->h2").
        throughputs: List of throughput values in Mbps.
        title: Chart title.
        filename: Output filename.
        output_dir: Output directory.

    Returns:
        Path to saved PNG file.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x = np.arange(len(labels))
    bars = ax.bar(x, throughputs, color="#42A5F5", edgecolor="black", alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Host Pair")
    ax.set_ylabel("Throughput (Mbps)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    return _save_figure(fig, filename, output_dir)


if __name__ == "__main__":
    # Generate topology visualization as a standalone script
    logging.basicConfig(level=logging.INFO)
    path = draw_topology()
    print(f"Topology diagram saved to: {path}")
