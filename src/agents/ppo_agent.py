"""Proximal Policy Optimization (PPO) agent for the adaptive security system.

Implements the PPO-Clip algorithm (Schulman et al. 2017) with:
    - Actor-critic architecture with shared feature layers.
    - Continuous action space (rate limit intensity, rerouting weight,
      priority adjustment) via diagonal Gaussian policy.
    - Clipped surrogate objective (epsilon = 0.2).
    - Generalised Advantage Estimation (GAE, lambda = 0.95).
    - Entropy bonus for sustained exploration.
    - Gradient clipping for training stability.
    - TensorBoard logging of actor loss, critic loss, entropy, and
      explained variance.

Hyperparameters are loaded from ``config/ppo_config.yaml``.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.agents.networks import (
    build_ppo_actor_network,
    build_ppo_critic_network,
    LOG_STD_MIN,
    LOG_STD_MAX,
)

logger = logging.getLogger(__name__)

# Suppress TF INFO messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# PPO continuous action dimensions
# 0: rate_limit_intensity [0, 1]
# 1: rerouting_weight     [0, 1]
# 2: priority_adjustment  [-1, 1]
ACTION_DIM = 3
ACTION_NAMES = {
    0: "rate_limit_intensity",
    1: "rerouting_weight",
    2: "priority_adjustment",
}


class RolloutBuffer:
    """Stores trajectory data collected during rollouts for PPO training.

    PPO is an on-policy algorithm, so the buffer is fully consumed after
    each training update and then cleared.

    Args:
        capacity: Maximum number of steps to store.
        state_dim: Observation vector dimensionality.
        action_dim: Number of continuous action dimensions.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)

        self._pos = 0
        self._full = False

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.states[self._pos] = state
        self.actions[self._pos] = action
        self.log_probs[self._pos] = log_prob
        self.rewards[self._pos] = reward
        self.values[self._pos] = value
        self.dones[self._pos] = float(done)

        self._pos += 1
        if self._pos >= self.capacity:
            self._full = True

    def compute_gae(
        self,
        last_value: float,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalised Advantage Estimation (GAE).

        Fills ``self.advantages`` and ``self.returns`` in-place.

        Args:
            last_value: V(s_T+1) for the state after the last stored step.
            gamma: Discount factor.
            gae_lambda: GAE smoothing parameter.
        """
        size = self._pos
        gae = 0.0

        for t in reversed(range(size)):
            if t == size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns[:size] = self.advantages[:size] + self.values[:size]

    def get_batches(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        """Yield mini-batches from the stored rollout data.

        Args:
            batch_size: Number of transitions per mini-batch.

        Returns:
            List of dictionaries, each containing batch arrays.
        """
        size = self._pos
        indices = np.arange(size)
        np.random.shuffle(indices)

        batches = []
        for start in range(0, size, batch_size):
            end = min(start + batch_size, size)
            batch_idx = indices[start:end]
            batches.append({
                "states": self.states[batch_idx],
                "actions": self.actions[batch_idx],
                "old_log_probs": self.log_probs[batch_idx],
                "advantages": self.advantages[batch_idx],
                "returns": self.returns[batch_idx],
                "values": self.values[batch_idx],
            })
        return batches

    def clear(self) -> None:
        """Reset the buffer for the next rollout."""
        self._pos = 0
        self._full = False

    def __len__(self) -> int:
        return self._pos


class PPOAgent:
    """PPO agent with actor-critic architecture and continuous actions.

    Args:
        state_dim: Observation vector dimensionality (default 65).
        action_dim: Number of continuous action dimensions (default 3).
        config: PPO hyperparameter dictionary from ``config/ppo_config.yaml``.

    Attributes:
        actor: Actor network outputting Gaussian means and log-stds.
        critic: Critic network outputting scalar V(s).
        total_steps: Cumulative environment steps taken.
    """

    def __init__(
        self,
        state_dim: int = 65,
        action_dim: int = ACTION_DIM,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Core hyperparameters
        self.learning_rate = float(cfg.get("learning_rate", 0.0003))
        self.clip_epsilon = float(cfg.get("clip_epsilon", 0.2))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))

        # Training
        self.n_epochs = int(cfg.get("n_epochs", 10))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.n_steps = int(cfg.get("n_steps", 2048))
        self.entropy_coef = float(cfg.get("entropy_coefficient", 0.01))
        self.value_loss_coef = float(cfg.get("value_loss_coefficient", 0.5))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.total_timesteps = int(cfg.get("total_timesteps", 100000))

        # Network architecture
        shared_layers = cfg.get("shared_layers", [128])
        actor_layers = cfg.get("actor_layers", [128, 128, 64])
        critic_layers = cfg.get("critic_layers", [128, 128, 64])
        activation = cfg.get("activation", "relu")

        # Build networks
        self.actor = build_ppo_actor_network(
            state_dim=state_dim,
            action_dim=action_dim,
            shared_layers=shared_layers,
            actor_layers=actor_layers,
            activation=activation,
        )
        self.critic = build_ppo_critic_network(
            state_dim=state_dim,
            shared_layers=shared_layers,
            critic_layers=critic_layers,
            activation=activation,
            learning_rate=self.learning_rate,
        )

        # Optimisers (actor uses custom training, not model.compile)
        self.actor_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
        )
        self.critic_optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(
            capacity=self.n_steps,
            state_dim=state_dim,
            action_dim=action_dim,
        )

        # Reproducibility
        seed = int(cfg.get("seed", 42))
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self._rng = np.random.RandomState(seed)

        # Counters
        self.total_steps = 0
        self._update_count = 0

        # TensorBoard writer
        self._tb_writer: Optional[tf.summary.FileWriter] = None

        logger.info(
            "PPOAgent initialized: state_dim=%d, action_dim=%d, "
            "lr=%.5f, clip_eps=%.2f, gamma=%.3f, gae_lambda=%.3f, "
            "n_steps=%d, n_epochs=%d, batch_size=%d",
            state_dim, action_dim, self.learning_rate,
            self.clip_epsilon, self.gamma, self.gae_lambda,
            self.n_steps, self.n_epochs, self.batch_size,
        )

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """Sample an action from the Gaussian policy.

        Args:
            state: Current observation vector of shape ``(state_dim,)``.
            deterministic: If True, return the mean action (no sampling).

        Returns:
            Tuple of (action, log_probability, value_estimate).
        """
        state_batch = np.expand_dims(state, axis=0).astype(np.float32)

        # Forward pass through actor
        means, log_stds = self.actor.predict(state_batch, verbose=0)
        means = means[0]  # shape: (action_dim,)
        log_stds = log_stds[0]

        # Clamp log_stds for numerical stability
        log_stds = np.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)
        stds = np.exp(log_stds)

        # Value estimate
        value = float(self.critic.predict(state_batch, verbose=0)[0, 0])

        if deterministic:
            action = means
            log_prob = self._gaussian_log_prob(means, means, log_stds)
        else:
            # Sample from diagonal Gaussian
            action = means + stds * self._rng.randn(self.action_dim).astype(
                np.float32
            )
            log_prob = self._gaussian_log_prob(action, means, log_stds)

        # Clip actions to valid ranges
        action = self._clip_action(action)

        return action, float(log_prob), value

    def get_value(self, state: np.ndarray) -> float:
        """Get the critic's value estimate for a state.

        Args:
            state: Observation vector of shape ``(state_dim,)``.

        Returns:
            Scalar value estimate V(s).
        """
        state_batch = np.expand_dims(state, axis=0).astype(np.float32)
        return float(self.critic.predict(state_batch, verbose=0)[0, 0])

    @staticmethod
    def _gaussian_log_prob(
        action: np.ndarray,
        mean: np.ndarray,
        log_std: np.ndarray,
    ) -> float:
        """Compute log probability under a diagonal Gaussian.

        Args:
            action: Action vector.
            mean: Gaussian mean vector.
            log_std: Log standard deviation vector.

        Returns:
            Scalar log probability (sum over action dimensions).
        """
        std = np.exp(log_std)
        var = std ** 2
        log_prob = -0.5 * (
            ((action - mean) ** 2) / (var + 1e-8)
            + 2 * log_std
            + np.log(2 * np.pi)
        )
        return float(np.sum(log_prob))

    @staticmethod
    def _clip_action(action: np.ndarray) -> np.ndarray:
        """Clip continuous actions to their valid ranges.

        Action dimensions:
            0: rate_limit_intensity [0, 1]
            1: rerouting_weight     [0, 1]
            2: priority_adjustment  [-1, 1]

        The actor outputs tanh-bounded means in [-1, 1], which we then
        rescale actions 0 and 1 to [0, 1].
        """
        clipped = action.copy()
        # Rescale dims 0 and 1 from [-1,1] to [0,1]
        clipped[0] = np.clip((action[0] + 1.0) / 2.0, 0.0, 1.0)
        clipped[1] = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)
        # Dim 2 stays in [-1, 1]
        clipped[2] = np.clip(action[2], -1.0, 1.0)
        return clipped

    # ------------------------------------------------------------------
    # Continuous Action → Discrete Mapping (for environment compatibility)
    # ------------------------------------------------------------------

    @staticmethod
    def continuous_to_discrete(action: np.ndarray) -> int:
        """Map continuous PPO action to a discrete action for the environment.

        This allows PPO to work with the same environment reward model
        as DQN during simulation mode, enabling fair comparison.

        Mapping logic:
            - If rate_limit_intensity > 0.7 → BLOCK (1)
            - If rate_limit_intensity > 0.3 → RATE_LIMIT (3)
            - If rerouting_weight > 0.5     → REROUTE (2)
            - Otherwise                     → ALLOW (0)

        Args:
            action: Continuous action vector [rate_limit, reroute, priority].

        Returns:
            Discrete action in {0, 1, 2, 3}.
        """
        rate_limit = action[0]  # Already in [0, 1] after clipping
        reroute = action[1]     # Already in [0, 1]

        if rate_limit > 0.7:
            return 1  # BLOCK
        elif rate_limit > 0.3:
            return 3  # RATE_LIMIT
        elif reroute > 0.5:
            return 2  # REROUTE
        else:
            return 0  # ALLOW

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Store a transition in the rollout buffer.

        Args:
            state: Observation before the action.
            action: Continuous action taken.
            log_prob: Log probability of the action.
            reward: Reward received.
            value: Critic value estimate V(s).
            done: Whether the episode ended.
        """
        self.rollout_buffer.add(state, action, log_prob, reward, value, done)
        self.total_steps += 1

    def train(self, last_value: float = 0.0) -> Dict[str, float]:
        """Perform a PPO training update on the collected rollout.

        Should be called after ``n_steps`` transitions have been stored.

        Args:
            last_value: V(s_T+1) for bootstrapping advantage at the
                final stored step.

        Returns:
            Dictionary of training metrics (actor_loss, critic_loss,
            entropy, explained_variance, clip_fraction).
        """
        # Compute GAE advantages
        self.rollout_buffer.compute_gae(
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )

        # Normalise advantages (improves training stability)
        size = len(self.rollout_buffer)
        advs = self.rollout_buffer.advantages[:size]
        adv_mean = advs.mean()
        adv_std = advs.std() + 1e-8
        self.rollout_buffer.advantages[:size] = (advs - adv_mean) / adv_std

        # Training metrics accumulators
        all_actor_losses = []
        all_critic_losses = []
        all_entropies = []
        all_clip_fractions = []

        for epoch in range(self.n_epochs):
            batches = self.rollout_buffer.get_batches(self.batch_size)

            for batch in batches:
                metrics = self._train_step(batch)
                all_actor_losses.append(metrics["actor_loss"])
                all_critic_losses.append(metrics["critic_loss"])
                all_entropies.append(metrics["entropy"])
                all_clip_fractions.append(metrics["clip_fraction"])

        self._update_count += 1

        # Clear buffer for next rollout
        self.rollout_buffer.clear()

        # Compute explained variance
        size_prev = size  # Buffer was just cleared
        values = self.rollout_buffer.values[:size_prev] if size_prev > 0 else np.array([0.0])
        returns = self.rollout_buffer.returns[:size_prev] if size_prev > 0 else np.array([0.0])

        result = {
            "actor_loss": float(np.mean(all_actor_losses)),
            "critic_loss": float(np.mean(all_critic_losses)),
            "entropy": float(np.mean(all_entropies)),
            "clip_fraction": float(np.mean(all_clip_fractions)),
            "update_count": self._update_count,
        }

        logger.info(
            "PPO update %d: actor_loss=%.5f, critic_loss=%.5f, "
            "entropy=%.5f, clip_frac=%.3f",
            self._update_count, result["actor_loss"],
            result["critic_loss"], result["entropy"],
            result["clip_fraction"],
        )

        return result

    def _train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Perform a single gradient update on a mini-batch.

        Args:
            batch: Dictionary with states, actions, old_log_probs,
                advantages, returns, values.

        Returns:
            Dictionary of per-step training metrics.
        """
        states = tf.constant(batch["states"], dtype=tf.float32)
        actions_raw = tf.constant(batch["actions"], dtype=tf.float32)
        old_log_probs = tf.constant(batch["old_log_probs"], dtype=tf.float32)
        advantages = tf.constant(batch["advantages"], dtype=tf.float32)
        returns = tf.constant(batch["returns"], dtype=tf.float32)

        # --- Actor update ---
        with tf.GradientTape() as tape:
            means, log_stds = self.actor(states, training=True)
            log_stds = tf.clip_by_value(log_stds, LOG_STD_MIN, LOG_STD_MAX)
            stds = tf.exp(log_stds)

            # Reverse the action clipping to get raw actions for log prob
            # We stored clipped actions, so we need to reverse-map them
            # For simplicity, compute log prob on raw Gaussian outputs
            # using the stored (clipped) actions as approximation
            var = stds ** 2 + 1e-8
            log_probs = -0.5 * tf.reduce_sum(
                ((actions_raw - means) ** 2) / var
                + 2 * log_stds
                + np.log(2 * np.pi),
                axis=-1,
            )

            # Policy ratio
            ratio = tf.exp(log_probs - old_log_probs)

            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = tf.clip_by_value(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            ) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            # Entropy bonus (diagonal Gaussian entropy)
            entropy = tf.reduce_mean(
                0.5 * tf.reduce_sum(
                    tf.math.log(2 * np.pi * np.e * var), axis=-1
                )
            )

            total_actor_loss = actor_loss - self.entropy_coef * entropy

        # Compute and apply actor gradients with clipping
        actor_grads = tape.gradient(
            total_actor_loss, self.actor.trainable_variables,
        )
        if self.max_grad_norm > 0:
            actor_grads, _ = tf.clip_by_global_norm(
                actor_grads, self.max_grad_norm,
            )
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables),
        )

        # --- Critic update ---
        with tf.GradientTape() as tape:
            values_pred = tf.squeeze(self.critic(states, training=True), axis=-1)
            critic_loss = tf.reduce_mean((returns - values_pred) ** 2)

        critic_grads = tape.gradient(
            critic_loss, self.critic.trainable_variables,
        )
        if self.max_grad_norm > 0:
            critic_grads, _ = tf.clip_by_global_norm(
                critic_grads, self.max_grad_norm,
            )
        self.critic_optimizer.apply_gradients(
            zip(critic_grads, self.critic.trainable_variables),
        )

        # Clip fraction (for monitoring)
        clip_fraction = float(tf.reduce_mean(
            tf.cast(
                tf.abs(ratio - 1.0) > self.clip_epsilon,
                tf.float32,
            )
        ))

        return {
            "actor_loss": float(actor_loss),
            "critic_loss": float(critic_loss),
            "entropy": float(entropy),
            "clip_fraction": clip_fraction,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str, episode: Optional[int] = None) -> str:
        """Save actor and critic weights to disk.

        Args:
            directory: Target directory for checkpoint files.
            episode: Optional episode number for the filename.

        Returns:
            Path to the saved directory.
        """
        os.makedirs(directory, exist_ok=True)
        suffix = f"_ep{episode}" if episode is not None else ""

        actor_path = os.path.join(directory, f"ppo_actor{suffix}.keras")
        critic_path = os.path.join(directory, f"ppo_critic{suffix}.keras")

        self.actor.save(actor_path)
        self.critic.save(critic_path)

        # Save metadata
        meta_path = os.path.join(directory, f"ppo_meta{suffix}.npz")
        np.savez(
            meta_path,
            total_steps=self.total_steps,
            update_count=self._update_count,
        )

        logger.info("PPO model saved to %s (episode=%s)", directory, episode)
        return directory

    def load(self, directory: str, episode: Optional[int] = None) -> None:
        """Load actor and critic weights from disk.

        Args:
            directory: Directory containing checkpoint files.
            episode: Optional episode number in the filename.
        """
        suffix = f"_ep{episode}" if episode is not None else ""

        actor_path = os.path.join(directory, f"ppo_actor{suffix}.keras")
        critic_path = os.path.join(directory, f"ppo_critic{suffix}.keras")

        self.actor = keras.models.load_model(actor_path)
        self.critic = keras.models.load_model(critic_path)

        meta_path = os.path.join(directory, f"ppo_meta{suffix}.npz")
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            self.total_steps = int(meta["total_steps"])
            self._update_count = int(meta["update_count"])
            logger.info(
                "Restored PPO metadata: steps=%d, updates=%d",
                self.total_steps, self._update_count,
            )

        logger.info("PPO model loaded from %s", directory)

    # ------------------------------------------------------------------
    # TensorBoard Logging
    # ------------------------------------------------------------------

    def setup_tensorboard(self, log_dir: str) -> None:
        """Initialise TensorBoard summary writer.

        Args:
            log_dir: Directory for TensorBoard event files.
        """
        os.makedirs(log_dir, exist_ok=True)
        self._tb_writer = tf.summary.create_file_writer(log_dir)
        logger.info("TensorBoard writer created at %s", log_dir)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Write scalar metrics to TensorBoard.

        Args:
            metrics: Dictionary of metric name -> value pairs.
            step: Global step number for the x-axis.
        """
        if self._tb_writer is None:
            return
        with self._tb_writer.as_default():
            for name, value in metrics.items():
                tf.summary.scalar(name, value, step=step)
            self._tb_writer.flush()
