"""State processor for the RL-driven adaptive security system.

Transforms raw network statistics from the Ryu SDN controller REST API
into a normalised observation vector suitable for the RL agent. Maintains
a moving-average history buffer to provide the agent with temporal context.

State vector layout (per switch, 13 features):
    [0]  total_flows          — normalised to [0, 1] by /100
    [1]  total_packets        — log1p scaled
    [2]  total_bytes          — log1p scaled
    [3]  avg_duration         — normalised to [0, 1] by /300s
    [4]  rx_packets           — log1p scaled
    [5]  tx_packets           — log1p scaled
    [6]  rx_bytes             — log1p scaled
    [7]  tx_bytes             — log1p scaled
    [8]  rx_dropped           — log1p scaled
    [9]  tx_dropped           — log1p scaled
    [10] rx_errors            — log1p scaled
    [11] tx_errors            — log1p scaled
    [12] connection_rate      — flows / elapsed seconds

Total state dimension: 5 switches * 13 features = 65.
"""

import logging
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

NUM_SWITCHES = 5
FEATURES_PER_SWITCH = 13
STATE_DIM = NUM_SWITCHES * FEATURES_PER_SWITCH  # 65


class StateProcessor:
    """Converts raw network statistics into RL observation vectors.

    Wraps a ``StatsCollector`` to pull live data from the Ryu REST API,
    normalise it, and optionally maintain an exponential moving average
    for smoother observations.

    Args:
        stats_collector: An instance of ``src.sdn.stats_collector.StatsCollector``.
            If *None*, the processor operates in **offline mode** and accepts
            raw state arrays directly via :meth:`process_raw_state`.
        moving_avg_window: Size of the moving-average buffer. Set to 1 to
            disable smoothing.
    """

    def __init__(
        self,
        stats_collector=None,
        moving_avg_window: int = 10,
    ) -> None:
        self.collector = stats_collector
        self.window = max(1, moving_avg_window)
        self._history: deque = deque(maxlen=self.window)
        self._last_state: Optional[np.ndarray] = None
        logger.info(
            "StateProcessor initialized (window=%d, collector=%s)",
            self.window,
            "live" if stats_collector else "offline",
        )

    @staticmethod
    def get_state_dim() -> int:
        """Return the dimensionality of the observation vector."""
        return STATE_DIM

    def get_state(self) -> np.ndarray:
        """Fetch and process the current network state.

        If a live ``StatsCollector`` is attached, queries the Ryu REST API.
        Otherwise returns a zero vector (useful for unit testing).

        Returns:
            Float32 numpy array of shape ``(65,)``.
        """
        if self.collector is not None:
            raw = self.collector.get_state_vector()
        else:
            raw = np.zeros(STATE_DIM, dtype=np.float32)

        return self._apply_moving_average(raw)

    def process_raw_state(self, raw_state: np.ndarray) -> np.ndarray:
        """Process an externally provided raw state vector.

        Useful for replaying recorded data or unit testing without a
        live controller connection.

        Args:
            raw_state: Float32 array of shape ``(65,)``.

        Returns:
            Smoothed float32 array of shape ``(65,)``.
        """
        if raw_state.shape != (STATE_DIM,):
            raise ValueError(
                f"Expected state shape ({STATE_DIM},), got {raw_state.shape}"
            )
        return self._apply_moving_average(raw_state.astype(np.float32))

    def _apply_moving_average(self, state: np.ndarray) -> np.ndarray:
        """Apply simple moving average over the history buffer.

        Args:
            state: Current raw observation.

        Returns:
            Averaged observation vector.
        """
        self._history.append(state.copy())
        self._last_state = state.copy()

        if len(self._history) == 1:
            return state

        stacked = np.stack(list(self._history), axis=0)
        averaged = stacked.mean(axis=0).astype(np.float32)
        return averaged

    def reset(self) -> None:
        """Clear the history buffer (call at the start of each episode)."""
        self._history.clear()
        self._last_state = None
        if self.collector is not None:
            self.collector.reset()
        logger.debug("StateProcessor history cleared")

    def get_last_raw_state(self) -> Optional[np.ndarray]:
        """Return the most recent un-averaged raw state, or None."""
        return self._last_state
