"""Experience replay buffer for DQN training.

Implements a fixed-size circular buffer backed by pre-allocated NumPy
arrays for memory efficiency.  Stores (state, action, reward, next_state,
done) transitions and supports uniform random batch sampling.

Design constraints (from hardware_constraints):
    - 100 000 transitions maximum (~100 K).
    - float32 storage to stay under 4 GB container limit.
    - State dimension: 65 (5 switches * 13 features).
"""

import logging
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Fixed-size circular experience replay buffer.

    Pre-allocates contiguous NumPy arrays for each field so that
    insertion is O(1) and batch sampling is fast via fancy indexing.

    Args:
        capacity: Maximum number of transitions to store.
        state_dim: Dimensionality of the observation vector.
        seed: Random seed for reproducible sampling.

    Example:
        >>> buf = ReplayBuffer(capacity=10000, state_dim=65, seed=42)
        >>> buf.add(state=np.zeros(65), action=0, reward=1.0,
        ...         next_state=np.zeros(65), done=False)
        >>> batch = buf.sample(batch_size=32)
        >>> batch["states"].shape
        (32, 65)
    """

    def __init__(
        self,
        capacity: int = 100_000,
        state_dim: int = 65,
        seed: int = 42,
    ) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self._rng = np.random.RandomState(seed)

        # Pre-allocate arrays (float32 for memory efficiency)
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._pos = 0       # Next write position
        self._size = 0      # Current number of stored transitions

        mem_mb = (
            self.states.nbytes + self.actions.nbytes + self.rewards.nbytes
            + self.next_states.nbytes + self.dones.nbytes
        ) / (1024 * 1024)
        logger.info(
            "ReplayBuffer initialized: capacity=%d, state_dim=%d, mem=%.1f MB",
            capacity, state_dim, mem_mb,
        )

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition.

        Args:
            state: Current observation vector.
            action: Discrete action taken (0–3).
            reward: Scalar reward received.
            next_state: Observation after the action.
            done: Whether the episode terminated.
        """
        self.states[self._pos] = state
        self.actions[self._pos] = action
        self.rewards[self._pos] = reward
        self.next_states[self._pos] = next_state
        self.dones[self._pos] = float(done)

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with keys: ``states``, ``actions``, ``rewards``,
            ``next_states``, ``dones``.  Each value is a NumPy array
            with the batch dimension first.

        Raises:
            ValueError: If fewer transitions than ``batch_size`` are stored.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer "
                f"with {self._size} stored."
            )

        indices = self._rng.randint(0, self._size, size=batch_size)

        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }

    def __len__(self) -> int:
        """Return the current number of stored transitions."""
        return self._size

    @property
    def is_ready(self) -> bool:
        """Whether the buffer has enough transitions for a training batch.

        The minimum threshold is defined externally (``min_buffer_size``
        in ``config/dqn_config.yaml``), but as a safety check we require
        at least 1 transition.
        """
        return self._size > 0

    def clear(self) -> None:
        """Reset the buffer to empty without releasing memory."""
        self._pos = 0
        self._size = 0
        logger.debug("ReplayBuffer cleared")
