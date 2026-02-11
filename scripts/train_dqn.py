"""DQN training entrypoint for the adaptive security system.

Runs the full DQN training loop with TensorBoard logging, periodic
checkpointing, and CSV metric export. Can operate in simulation mode
(no Docker containers required) or live mode (connected to Ryu).

Usage:
    # Simulation mode (default — for development and testing)
    python -m scripts.train_dqn

    # Live mode (requires Docker containers running)
    python -m scripts.train_dqn --live

    # Custom config overrides
    python -m scripts.train_dqn --episodes 100 --max-steps 100
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.dqn_agent import DQNAgent
from src.environment.network_env import NetworkSecurityEnv
from src.utils.config_loader import get_hyperparameters, get_reward_config
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DQN agent for network security",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Connect to live Ryu controller (requires Docker)",
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Override max_episodes from config",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max_steps_per_episode from config",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/dqn",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/dqn",
    )
    parser.add_argument(
        "--log-dir", type=str, default="results/logs/dqn",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Run the DQN training loop."""
    setup_logging(level=args.log_level)

    # Load configuration
    dqn_config = get_hyperparameters("dqn")
    reward_config = get_reward_config("dqn")

    max_episodes = args.episodes or dqn_config.get("max_episodes", 500)
    max_steps = args.max_steps or dqn_config.get("max_steps_per_episode", 200)
    checkpoint_freq = 50  # Save every N episodes
    log_freq = 10         # Log metrics every N episodes

    logger.info("=" * 60)
    logger.info("DQN Training — RL-Driven Adaptive Security System")
    logger.info("=" * 60)
    logger.info("Episodes: %d | Steps/episode: %d", max_episodes, max_steps)
    logger.info("Mode: %s", "LIVE" if args.live else "SIMULATION")

    # Create environment
    stats_collector = None
    policy_enforcer = None

    if args.live:
        from src.sdn.stats_collector import StatsCollector
        from src.sdn.policy_enforcer import PolicyEnforcer

        stats_collector = StatsCollector()
        policy_enforcer = PolicyEnforcer()

        logger.info("Waiting for Ryu controller...")
        if not stats_collector.wait_for_controller(timeout=60):
            logger.error("Ryu controller not reachable. Exiting.")
            sys.exit(1)

    env = NetworkSecurityEnv(
        reward_config=reward_config,
        stats_collector=stats_collector,
        policy_enforcer=policy_enforcer,
        max_steps=max_steps,
        seed=dqn_config.get("seed", 42),
    )

    # Create agent
    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        config=dqn_config,
    )

    # Setup TensorBoard
    agent.setup_tensorboard(args.log_dir)

    # Setup results directory
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "training_log.csv")
    csv_fields = [
        "episode", "steps", "total_reward", "avg_reward", "epsilon",
        "avg_loss", "r_detection", "r_false_positive", "r_throughput",
        "r_latency", "r_stability", "action_0_pct", "action_1_pct",
        "action_2_pct", "action_3_pct", "buffer_size", "wall_time_s",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    # Training loop
    all_rewards = []
    training_start = time.time()

    for episode in range(1, max_episodes + 1):
        state, info = env.reset()
        episode_reward = 0.0
        episode_losses = []
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        reward_sums = {
            "r_detection": 0.0, "r_false_positive": 0.0,
            "r_throughput": 0.0, "r_latency": 0.0, "r_stability": 0.0,
        }

        ep_start = time.time()

        for step in range(1, max_steps + 1):
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, step_info = env.step(action)

            # Store and train
            loss = agent.step(state, action, reward, next_state,
                              terminated or truncated)
            if loss is not None:
                episode_losses.append(loss)

            # Track metrics
            episode_reward += reward
            action_counts[action] += 1
            for key in reward_sums:
                reward_sums[key] += step_info["reward_components"].get(key, 0.0)

            state = next_state

            if terminated or truncated:
                break

        ep_time = time.time() - ep_start
        all_rewards.append(episode_reward)
        avg_loss = float(np.mean(episode_losses)) if episode_losses else 0.0

        # Log to CSV
        total_actions = sum(action_counts.values()) or 1
        row = {
            "episode": episode,
            "steps": step,
            "total_reward": round(episode_reward, 6),
            "avg_reward": round(episode_reward / step, 6),
            "epsilon": round(agent.epsilon, 6),
            "avg_loss": round(avg_loss, 6),
            "r_detection": round(reward_sums["r_detection"] / step, 4),
            "r_false_positive": round(reward_sums["r_false_positive"] / step, 4),
            "r_throughput": round(reward_sums["r_throughput"] / step, 4),
            "r_latency": round(reward_sums["r_latency"] / step, 4),
            "r_stability": round(reward_sums["r_stability"] / step, 4),
            "action_0_pct": round(100 * action_counts[0] / total_actions, 1),
            "action_1_pct": round(100 * action_counts[1] / total_actions, 1),
            "action_2_pct": round(100 * action_counts[2] / total_actions, 1),
            "action_3_pct": round(100 * action_counts[3] / total_actions, 1),
            "buffer_size": len(agent.replay_buffer),
            "wall_time_s": round(ep_time, 2),
        }

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        # TensorBoard logging
        if episode % log_freq == 0:
            agent.log_scalars({
                "reward/episode_total": episode_reward,
                "reward/episode_avg": episode_reward / step,
                "training/epsilon": agent.epsilon,
                "training/loss": avg_loss,
                "training/buffer_size": len(agent.replay_buffer),
                "actions/allow_pct": action_counts[0] / total_actions,
                "actions/block_pct": action_counts[1] / total_actions,
                "actions/reroute_pct": action_counts[2] / total_actions,
                "actions/rate_limit_pct": action_counts[3] / total_actions,
            }, step=episode)

            # Compute moving average
            window = min(50, len(all_rewards))
            avg_50 = float(np.mean(all_rewards[-window:]))

            logger.info(
                "Episode %d/%d: reward=%.3f (avg50=%.3f), "
                "eps=%.3f, loss=%.5f, buffer=%d, time=%.1fs",
                episode, max_episodes, episode_reward, avg_50,
                agent.epsilon, avg_loss, len(agent.replay_buffer), ep_time,
            )

        # Checkpoint
        if episode % checkpoint_freq == 0:
            agent.save(args.checkpoint_dir, episode=episode)

    # Final save
    total_time = time.time() - training_start
    agent.save(args.checkpoint_dir)

    # Save reward history
    np.save(
        os.path.join(args.results_dir, "rewards.npy"),
        np.array(all_rewards, dtype=np.float32),
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(
        "  Episodes: %d | Total steps: %d | Time: %.1f min",
        max_episodes, agent.total_steps, total_time / 60,
    )
    logger.info(
        "  Final avg reward (last 50): %.4f",
        float(np.mean(all_rewards[-50:])),
    )
    logger.info("  Results saved to: %s", args.results_dir)
    logger.info("  Checkpoints saved to: %s", args.checkpoint_dir)
    logger.info("  TensorBoard logs: %s", args.log_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    train(parse_args())
