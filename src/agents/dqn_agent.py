"""Deep Q-Network (DQN) agent for the adaptive security system.

Implements the core DQN algorithm with:
    - Experience replay buffer (uniform sampling).
    - Separate online and target networks with soft (Polyak) updates.
    - Epsilon-greedy exploration with linear decay.
    - TensorBoard logging of loss, Q-values, and epsilon.

Hyperparameters are loaded from ``config/dqn_config.yaml``.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.agents.networks import (
    build_dqn_network,
    build_dqn_target_network,
    soft_update_target,
)
from src.agents.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

# Suppress TF INFO messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

NUM_ACTIONS = 4  # ALLOW, BLOCK, REROUTE, RATE_LIMIT


class DQNAgent:
    """DQN agent with experience replay and target network.

    Args:
        state_dim: Observation vector dimensionality (default 65).
        config: DQN hyperparameter dictionary. If *None*, sensible
            defaults matching ``config/dqn_config.yaml`` are used.

    Attributes:
        online_model: The Q-network used for action selection and training.
        target_model: Periodically updated copy for stable TD targets.
        replay_buffer: Experience storage for batch training.
        epsilon: Current exploration probability.
        total_steps: Cumulative environment steps taken.
    """

    def __init__(
        self,
        state_dim: int = 65,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}

        # Hyperparameters
        self.state_dim = state_dim
        self.num_actions = NUM_ACTIONS
        self.gamma = float(cfg.get("gamma", 0.99))
        self.tau = float(cfg.get("tau", 0.005))
        self.batch_size = int(cfg.get("batch_size", 64))
        self.train_freq = int(cfg.get("train_freq", 4))
        self.min_buffer_size = int(cfg.get("min_buffer_size", 1000))
        self.learning_rate = float(cfg.get("learning_rate", 0.001))

        # Exploration
        self.epsilon_start = float(cfg.get("epsilon_start", 1.0))
        self.epsilon_end = float(cfg.get("epsilon_end", 0.05))
        self.epsilon_decay_steps = int(cfg.get("epsilon_decay_steps", 50000))
        self.epsilon = self.epsilon_start

        # Replay buffer
        buffer_size = int(cfg.get("buffer_size", 100_000))
        seed = int(cfg.get("seed", 42))
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size, state_dim=state_dim, seed=seed,
        )

        # Networks
        hidden_layers = cfg.get("hidden_layers", [128, 128, 64])
        activation = cfg.get("activation", "relu")

        self.online_model = build_dqn_network(
            state_dim=state_dim,
            num_actions=self.num_actions,
            hidden_layers=hidden_layers,
            activation=activation,
            learning_rate=self.learning_rate,
        )
        self.target_model = build_dqn_target_network(self.online_model)

        # Reproducibility (local RNG only — don't pollute global state)
        tf.random.set_seed(seed)
        self._rng = np.random.RandomState(seed)

        # Counters
        self.total_steps = 0
        self._train_step_count = 0

        # TensorBoard writer (lazy-initialised)
        self._tb_writer: Optional[tf.summary.FileWriter] = None

        logger.info(
            "DQNAgent initialized: state_dim=%d, actions=%d, "
            "gamma=%.3f, tau=%.4f, lr=%.5f, eps=[%.2f->%.2f over %d steps]",
            state_dim, self.num_actions, self.gamma, self.tau,
            self.learning_rate, self.epsilon_start, self.epsilon_end,
            self.epsilon_decay_steps,
        )

    # ------------------------------------------------------------------
    # Action Selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Choose an action using epsilon-greedy policy.

        Args:
            state: Current observation vector of shape ``(state_dim,)``.
            greedy: If *True*, always select the best action (no exploration).

        Returns:
            Integer action in {0, 1, 2, 3}.
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.randint(0, self.num_actions))

        state_batch = np.expand_dims(state, axis=0).astype(np.float32)
        q_values = self.online_model.predict(state_batch, verbose=0)
        return int(np.argmax(q_values[0]))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for all actions given a single state.

        Args:
            state: Observation vector of shape ``(state_dim,)``.

        Returns:
            Array of shape ``(num_actions,)`` with Q-value per action.
        """
        state_batch = np.expand_dims(state, axis=0).astype(np.float32)
        return self.online_model.predict(state_batch, verbose=0)[0]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store a transition and optionally train.

        This method should be called after every environment step.

        Args:
            state: Observation before the action.
            action: Action taken.
            reward: Reward received.
            next_state: Observation after the action.
            done: Whether the episode ended.

        Returns:
            Training loss if a training step occurred, else *None*.
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1
        self._update_epsilon()

        loss = None
        if (
            self.total_steps % self.train_freq == 0
            and len(self.replay_buffer) >= self.min_buffer_size
        ):
            loss = self._train_batch()
            soft_update_target(self.online_model, self.target_model, self.tau)

        return loss

    def _train_batch(self) -> float:
        """Sample a batch and perform one gradient update.

        Returns:
            Mean squared error loss for the batch.
        """
        batch = self.replay_buffer.sample(self.batch_size)

        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Compute TD targets using the target network
        next_q = self.target_model.predict(next_states, verbose=0)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1.0 - dones)

        # Get current Q-value predictions and update only the taken actions
        current_q = self.online_model.predict(states, verbose=0)
        for i, a in enumerate(actions):
            current_q[i, a] = targets[i]

        # Train
        history = self.online_model.fit(
            states, current_q,
            batch_size=self.batch_size,
            epochs=1,
            verbose=0,
        )

        loss = float(history.history["loss"][0])
        self._train_step_count += 1

        if self._train_step_count % 100 == 0:
            logger.debug(
                "Train step %d: loss=%.6f, eps=%.4f, buffer=%d",
                self._train_step_count, loss, self.epsilon,
                len(self.replay_buffer),
            )

        return loss

    # ------------------------------------------------------------------
    # Exploration Schedule
    # ------------------------------------------------------------------

    def _update_epsilon(self) -> None:
        """Linearly decay epsilon from start to end over configured steps."""
        fraction = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = (
            self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str, episode: Optional[int] = None) -> str:
        """Save model weights to disk.

        Args:
            directory: Target directory for checkpoint files.
            episode: Optional episode number for the filename.

        Returns:
            Path to the saved checkpoint directory.
        """
        os.makedirs(directory, exist_ok=True)
        suffix = f"_ep{episode}" if episode is not None else ""
        path = os.path.join(directory, f"dqn_model{suffix}.keras")
        self.online_model.save(path)
        logger.info("DQN model saved to %s", path)

        # Save metadata
        meta_path = os.path.join(directory, f"dqn_meta{suffix}.npz")
        np.savez(
            meta_path,
            epsilon=self.epsilon,
            total_steps=self.total_steps,
            train_step_count=self._train_step_count,
        )
        return path

    def load(self, directory: str, episode: Optional[int] = None) -> None:
        """Load model weights from disk.

        Args:
            directory: Directory containing checkpoint files.
            episode: Optional episode number in the filename.
        """
        suffix = f"_ep{episode}" if episode is not None else ""
        path = os.path.join(directory, f"dqn_model{suffix}.keras")
        self.online_model = keras.models.load_model(path)
        self.target_model = build_dqn_target_network(self.online_model)
        logger.info("DQN model loaded from %s", path)

        meta_path = os.path.join(directory, f"dqn_meta{suffix}.npz")
        if os.path.exists(meta_path):
            meta = np.load(meta_path)
            self.epsilon = float(meta["epsilon"])
            self.total_steps = int(meta["total_steps"])
            self._train_step_count = int(meta["train_step_count"])
            logger.info(
                "Restored metadata: eps=%.4f, steps=%d",
                self.epsilon, self.total_steps,
            )

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
