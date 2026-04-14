"""Unit tests for the experience replay buffer.

Tests cover:
    - Adding transitions (single and multiple).
    - Circular overwrite when capacity is reached.
    - Batch sampling correctness.
    - Edge cases (empty buffer, sample > size).
    - Memory layout (float32 dtype).
"""

import pytest
import numpy as np

from src.agents.replay_buffer import ReplayBuffer

STATE_DIM = 65


@pytest.fixture
def buffer():
    """Small buffer for fast tests."""
    return ReplayBuffer(capacity=100, state_dim=STATE_DIM, seed=42)


@pytest.fixture
def filled_buffer():
    """Buffer pre-filled with 50 transitions."""
    buf = ReplayBuffer(capacity=100, state_dim=STATE_DIM, seed=42)
    for i in range(50):
        state = np.full(STATE_DIM, float(i), dtype=np.float32)
        buf.add(
            state=state,
            action=i % 4,
            reward=float(i) * 0.1,
            next_state=state + 1.0,
            done=(i % 10 == 9),
        )
    return buf


class TestBufferAdd:
    """Test adding transitions to the buffer."""

    def test_add_single(self, buffer):
        """Adding one transition should increase size to 1."""
        state = np.zeros(STATE_DIM, dtype=np.float32)
        buffer.add(state, action=0, reward=1.0,
                    next_state=state, done=False)
        assert len(buffer) == 1

    def test_add_multiple(self, buffer):
        """Adding N transitions should increase size to N."""
        state = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(10):
            buffer.add(state, 1, 0.5, state, False)
        assert len(buffer) == 10

    def test_circular_overwrite(self):
        """Buffer should wrap around when capacity is exceeded."""
        buf = ReplayBuffer(capacity=10, state_dim=STATE_DIM, seed=0)
        for i in range(15):
            state = np.full(STATE_DIM, float(i), dtype=np.float32)
            buf.add(state, 0, 0.0, state, False)
        assert len(buf) == 10
        # The oldest entry (i=0..4) should be overwritten
        # Position 0 should now hold i=10
        assert buf.states[0, 0] == pytest.approx(10.0)

    def test_dtype_float32(self, buffer):
        """States and rewards should be stored as float32."""
        state = np.ones(STATE_DIM, dtype=np.float64)  # intentionally float64
        buffer.add(state, 0, 1.0, state, False)
        assert buffer.states.dtype == np.float32
        assert buffer.rewards.dtype == np.float32


class TestBufferSample:
    """Test batch sampling from the buffer."""

    def test_sample_shape(self, filled_buffer):
        """Sampled batch should have correct shapes."""
        batch = filled_buffer.sample(batch_size=16)
        assert batch["states"].shape == (16, STATE_DIM)
        assert batch["actions"].shape == (16,)
        assert batch["rewards"].shape == (16,)
        assert batch["next_states"].shape == (16, STATE_DIM)
        assert batch["dones"].shape == (16,)

    def test_sample_values_in_range(self, filled_buffer):
        """Sampled actions and dones should be in valid ranges."""
        batch = filled_buffer.sample(batch_size=32)
        assert np.all(batch["actions"] >= 0)
        assert np.all(batch["actions"] < 4)
        assert np.all((batch["dones"] == 0.0) | (batch["dones"] == 1.0))

    def test_sample_from_empty_raises(self, buffer):
        """Sampling from an empty buffer should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot sample"):
            buffer.sample(batch_size=1)

    def test_sample_exceeds_size_raises(self, filled_buffer):
        """Sampling more than stored should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot sample"):
            filled_buffer.sample(batch_size=100)

    def test_sample_reproducibility(self, filled_buffer):
        """Two buffers with the same seed should produce identical samples."""
        buf2 = ReplayBuffer(capacity=100, state_dim=STATE_DIM, seed=42)
        for i in range(50):
            state = np.full(STATE_DIM, float(i), dtype=np.float32)
            buf2.add(state, i % 4, float(i) * 0.1, state + 1.0, i % 10 == 9)

        batch1 = filled_buffer.sample(16)
        batch2 = buf2.sample(16)
        np.testing.assert_array_equal(batch1["actions"], batch2["actions"])


class TestBufferProperties:
    """Test buffer utility methods."""

    def test_is_ready_empty(self, buffer):
        """is_ready should be False for empty buffer."""
        assert not buffer.is_ready

    def test_is_ready_filled(self, filled_buffer):
        """is_ready should be True for non-empty buffer."""
        assert filled_buffer.is_ready

    def test_clear(self, filled_buffer):
        """clear should reset size to 0."""
        filled_buffer.clear()
        assert len(filled_buffer) == 0
        assert not filled_buffer.is_ready
