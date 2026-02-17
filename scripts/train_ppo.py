"""PPO training entrypoint for the adaptive security system.

Runs the full PPO training loop with TensorBoard logging, periodic
checkpointing, and CSV metric export. Uses on-policy rollouts with
Generalised Advantage Estimation (GAE).

Usage:
    # Simulation mode (default — for development and testing)
    python -m scripts.train_ppo

    # Live mode (requires Docker containers running)
    python -m scripts.train_ppo --live

    # Custom config overrides
    python -m scripts.train_ppo --timesteps 50000 --max-steps 200
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.ppo_agent import PPOAgent
from src.environment.network_env import NetworkSecurityEnv
from src.utils.config_loader import get_hyperparameters, get_reward_config
from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent for network security",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Connect to live Ryu controller (requires Docker)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total_timesteps from config",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max steps per episode",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints/ppo",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/ppo",
    )
    parser.add_argument(
        "--log-dir", type=str, default="results/logs/ppo",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    """Run the PPO training loop."""
    setup_logging(level=args.log_level)

    # Load configuration
    ppo_config = get_hyperparameters("ppo")
    reward_config = get_reward_config("ppo")

    total_timesteps = args.timesteps or ppo_config.get("total_timesteps", 100000)
    max_steps = args.max_steps or 200
    n_steps = ppo_config.get("n_steps", 2048)
    checkpoint_freq = 50  # Save every N episodes
    log_freq = 10         # Log metrics every N episodes

    logger.info("=" * 60)
    logger.info("PPO Training -- RL-Driven Adaptive Security System")
    logger.info("=" * 60)
    logger.info(
        "Total timesteps: %d | Steps/rollout: %d | Steps/episode: %d",
        total_timesteps, n_steps, max_steps,
    )
    logger.info("Mode: %s", "LIVE" if args.live else "SIMULATION")

    # Create environment (continuous action mode for PPO)
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
        seed=ppo_config.get("seed", 42),
        action_mode="continuous",
    )

    # Create agent
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        config=ppo_config,
    )

    # Setup TensorBoard
    agent.setup_tensorboard(args.log_dir)

    # Setup results directory
    os.makedirs(args.results_dir, exist_ok=True)
    csv_path = os.path.join(args.results_dir, "training_log.csv")
    csv_fields = [
        "episode", "steps", "total_reward", "avg_reward",
        "actor_loss", "critic_loss", "entropy", "clip_fraction",
        "r_detection", "r_false_positive", "r_throughput",
        "r_latency", "r_stability",
        "action_allow_pct", "action_block_pct",
        "action_reroute_pct", "action_rate_limit_pct",
        "avg_rate_limit", "avg_reroute_weight", "avg_priority",
        "total_timesteps", "wall_time_s",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    # Training loop
    all_rewards = []
    training_start = time.time()
    global_step = 0
    episode = 0

    # Latest training metrics (from most recent PPO update)
    latest_train_metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "clip_fraction": 0.0,
    }

    state, info = env.reset()

    while global_step < total_timesteps:
        episode += 1
        episode_reward = 0.0
        reward_sums = {
            "r_detection": 0.0, "r_false_positive": 0.0,
            "r_throughput": 0.0, "r_latency": 0.0, "r_stability": 0.0,
        }
        # Track discrete action distribution from continuous mapping
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        continuous_actions = []
        ep_steps = 0

        ep_start = time.time()

        for step in range(1, max_steps + 1):
            # Select action from policy
            action, log_prob, value = agent.select_action(state)

            # Step environment
            next_state, reward, terminated, truncated, step_info = env.step(action)

            # Store transition
            agent.store_transition(
                state, action, log_prob, reward, value,
                terminated or truncated,
            )

            # Track metrics
            episode_reward += reward
            global_step += 1
            ep_steps += 1

            discrete_action = step_info["action"]
            action_counts[discrete_action] += 1
            continuous_actions.append(action.copy())

            for key in reward_sums:
                reward_sums[key] += step_info["reward_components"].get(key, 0.0)

            # PPO training update when rollout buffer is full
            if len(agent.rollout_buffer) >= agent.n_steps:
                last_value = agent.get_value(next_state)
                latest_train_metrics = agent.train(last_value=last_value)

                # TensorBoard: training metrics
                agent.log_scalars({
                    "training/actor_loss": latest_train_metrics["actor_loss"],
                    "training/critic_loss": latest_train_metrics["critic_loss"],
                    "training/entropy": latest_train_metrics["entropy"],
                    "training/clip_fraction": latest_train_metrics["clip_fraction"],
                }, step=global_step)

            state = next_state

            if terminated or truncated:
                state, info = env.reset()
                break

        if global_step >= total_timesteps:
            break

        ep_time = time.time() - ep_start
        all_rewards.append(episode_reward)

        # Compute continuous action statistics
        ca = np.array(continuous_actions) if continuous_actions else np.zeros((1, 3))
        avg_rate_limit = float(np.mean(ca[:, 0]))
        avg_reroute = float(np.mean(ca[:, 1]))
        avg_priority = float(np.mean(ca[:, 2]))

        # Log to CSV
        total_actions = sum(action_counts.values()) or 1
        row = {
            "episode": episode,
            "steps": ep_steps,
            "total_reward": round(episode_reward, 6),
            "avg_reward": round(episode_reward / max(ep_steps, 1), 6),
            "actor_loss": round(latest_train_metrics["actor_loss"], 6),
            "critic_loss": round(latest_train_metrics["critic_loss"], 6),
            "entropy": round(latest_train_metrics["entropy"], 6),
            "clip_fraction": round(latest_train_metrics["clip_fraction"], 6),
            "r_detection": round(reward_sums["r_detection"] / max(ep_steps, 1), 4),
            "r_false_positive": round(reward_sums["r_false_positive"] / max(ep_steps, 1), 4),
            "r_throughput": round(reward_sums["r_throughput"] / max(ep_steps, 1), 4),
            "r_latency": round(reward_sums["r_latency"] / max(ep_steps, 1), 4),
            "r_stability": round(reward_sums["r_stability"] / max(ep_steps, 1), 4),
            "action_allow_pct": round(100 * action_counts[0] / total_actions, 1),
            "action_block_pct": round(100 * action_counts[1] / total_actions, 1),
            "action_reroute_pct": round(100 * action_counts[2] / total_actions, 1),
            "action_rate_limit_pct": round(100 * action_counts[3] / total_actions, 1),
            "avg_rate_limit": round(avg_rate_limit, 4),
            "avg_reroute_weight": round(avg_reroute, 4),
            "avg_priority": round(avg_priority, 4),
            "total_timesteps": global_step,
            "wall_time_s": round(ep_time, 2),
        }

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        # TensorBoard: episode metrics
        if episode % log_freq == 0:
            agent.log_scalars({
                "reward/episode_total": episode_reward,
                "reward/episode_avg": episode_reward / max(ep_steps, 1),
                "actions/allow_pct": action_counts[0] / total_actions,
                "actions/block_pct": action_counts[1] / total_actions,
                "actions/reroute_pct": action_counts[2] / total_actions,
                "actions/rate_limit_pct": action_counts[3] / total_actions,
                "continuous/avg_rate_limit": avg_rate_limit,
                "continuous/avg_reroute_weight": avg_reroute,
                "continuous/avg_priority": avg_priority,
            }, step=episode)

            window = min(50, len(all_rewards))
            avg_50 = float(np.mean(all_rewards[-window:]))

            logger.info(
                "Episode %d: reward=%.3f (avg50=%.3f), "
                "actor_loss=%.5f, critic_loss=%.5f, entropy=%.4f, "
                "steps=%d/%d, time=%.1fs",
                episode, episode_reward, avg_50,
                latest_train_metrics["actor_loss"],
                latest_train_metrics["critic_loss"],
                latest_train_metrics["entropy"],
                global_step, total_timesteps, ep_time,
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
    logger.info("PPO Training complete!")
    logger.info(
        "  Episodes: %d | Total steps: %d | Time: %.1f min",
        episode, global_step, total_time / 60,
    )
    logger.info(
        "  Final avg reward (last 50): %.4f",
        float(np.mean(all_rewards[-50:])) if all_rewards else 0.0,
    )
    logger.info("  Results saved to: %s", args.results_dir)
    logger.info("  Checkpoints saved to: %s", args.checkpoint_dir)
    logger.info("  TensorBoard logs: %s", args.log_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    train(parse_args())
