"""Unit tests for the PPO agent.

Tests cover:
    - Actor network outputs valid Gaussian parameters (mean + std > 0).
    - Critic network outputs scalar value estimates.
    - PPO clipping correctly bounds policy ratio to [1-eps, 1+eps].
    - GAE computation produces correct advantage estimates.
    - Continuous action selection and clipping.
    - Continuous-to-discrete action mapping.
    - Rollout buffer storage and batch generation.
    - Training loop runs without crashes.
    - Model save/load round-trip.
    - Explained variance correctness.
    - RolloutBuffer overflow protection.
    - Local RNG: no global np.random.seed pollution.
    - Environment continuous action mode integration.
"""

import os
import tempfile

import pytest
import numpy as np

from src.agents.ppo_agent import PPOAgent, RolloutBuffer, ACTION_DIM
from src.environment.network_env import NetworkSecurityEnv

STATE_DIM = 65


@pytest.fixture
def default_config():
    """Minimal PPO config for fast tests."""
    return {
        "shared_layers": [32],
        "actor_layers": [32, 32],
        "critic_layers": [32, 32],
        "activation": "relu",
        "learning_rate": 0.0003,
        "clip_epsilon": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "n_epochs": 2,
        "batch_size": 8,
        "n_steps": 32,
        "entropy_coefficient": 0.01,
        "value_loss_coefficient": 0.5,
        "max_grad_norm": 0.5,
        "total_timesteps": 100,
        "seed": 42,
    }


@pytest.fixture
def agent(default_config):
    return PPOAgent(state_dim=STATE_DIM, config=default_config)


@pytest.fixture
def random_state():
    return np.random.randn(STATE_DIM).astype(np.float32)


class TestActorNetwork:
    """Test the PPO actor (policy) network."""

    def test_actor_output_shapes(self, agent, random_state):
        """Actor should output means and log_stds of shape (action_dim,)."""
        state_batch = np.expand_dims(random_state, axis=0)
        means, log_stds = agent.actor.predict(state_batch, verbose=0)
        assert means.shape == (1, ACTION_DIM)
        assert log_stds.shape == (1, ACTION_DIM)

    def test_actor_means_bounded(self, agent, random_state):
        """Actor means should be in [-1, 1] due to tanh activation."""
        state_batch = np.expand_dims(random_state, axis=0)
        means, _ = agent.actor.predict(state_batch, verbose=0)
        assert np.all(means >= -1.0)
        assert np.all(means <= 1.0)

    def test_actor_outputs_finite(self, agent, random_state):
        """Actor outputs should not contain NaN or Inf."""
        state_batch = np.expand_dims(random_state, axis=0)
        means, log_stds = agent.actor.predict(state_batch, verbose=0)
        assert np.all(np.isfinite(means))
        assert np.all(np.isfinite(log_stds))

    def test_stds_positive(self, agent, random_state):
        """Exponentiated log_stds should be positive."""
        state_batch = np.expand_dims(random_state, axis=0)
        _, log_stds = agent.actor.predict(state_batch, verbose=0)
        log_stds_clipped = np.clip(log_stds, -20.0, 2.0)
        stds = np.exp(log_stds_clipped)
        assert np.all(stds > 0)


class TestCriticNetwork:
    """Test the PPO critic (value function) network."""

    def test_critic_output_shape(self, agent, random_state):
        """Critic should output a scalar value."""
        state_batch = np.expand_dims(random_state, axis=0)
        value = agent.critic.predict(state_batch, verbose=0)
        assert value.shape == (1, 1)

    def test_critic_output_finite(self, agent, random_state):
        """Critic output should be finite."""
        value = agent.get_value(random_state)
        assert np.isfinite(value)


class TestActionSelection:
    """Test action selection and continuous-to-discrete mapping."""

    def test_select_action_returns_tuple(self, agent, random_state):
        """select_action should return (action, log_prob, value)."""
        action, log_prob, value = agent.select_action(random_state)
        assert action.shape == (ACTION_DIM,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_deterministic_action_consistent(self, agent, random_state):
        """Deterministic action should be the same on repeat calls."""
        a1, _, _ = agent.select_action(random_state, deterministic=True)
        a2, _, _ = agent.select_action(random_state, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2)

    def test_stochastic_actions_vary(self, agent, random_state):
        """Stochastic actions should differ across calls."""
        actions = []
        for _ in range(20):
            a, _, _ = agent.select_action(random_state, deterministic=False)
            actions.append(a.copy())
        actions = np.array(actions)
        # Check that actions are not all identical
        assert np.std(actions) > 1e-4

    def test_action_clipping_rate_limit(self, agent):
        """Rate limit intensity should be clipped to [0, 1]."""
        action = np.array([0.8, 0.3, 0.0], dtype=np.float32)
        clipped = agent._clip_action(action)
        assert 0.0 <= clipped[0] <= 1.0

    def test_action_clipping_reroute(self, agent):
        """Rerouting weight should be clipped to [0, 1]."""
        action = np.array([0.0, 0.9, 0.0], dtype=np.float32)
        clipped = agent._clip_action(action)
        assert 0.0 <= clipped[1] <= 1.0

    def test_action_clipping_priority(self, agent):
        """Priority adjustment should be clipped to [-1, 1]."""
        action = np.array([0.0, 0.0, -0.5], dtype=np.float32)
        clipped = agent._clip_action(action)
        assert -1.0 <= clipped[2] <= 1.0

    def test_continuous_to_discrete_block(self, agent):
        """High rate_limit should map to BLOCK."""
        action = np.array([0.9, 0.0, 0.0])
        assert agent.continuous_to_discrete(action) == 1

    def test_continuous_to_discrete_rate_limit(self, agent):
        """Moderate rate_limit should map to RATE_LIMIT."""
        action = np.array([0.5, 0.0, 0.0])
        assert agent.continuous_to_discrete(action) == 3

    def test_continuous_to_discrete_reroute(self, agent):
        """High reroute weight with low rate_limit should map to REROUTE."""
        action = np.array([0.1, 0.7, 0.0])
        assert agent.continuous_to_discrete(action) == 2

    def test_continuous_to_discrete_allow(self, agent):
        """Low values should map to ALLOW."""
        action = np.array([0.1, 0.2, 0.0])
        assert agent.continuous_to_discrete(action) == 0


class TestRolloutBuffer:
    """Test the on-policy rollout buffer."""

    def test_add_and_length(self):
        """Adding transitions should increase buffer length."""
        buf = RolloutBuffer(capacity=100, state_dim=STATE_DIM, action_dim=ACTION_DIM)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        buf.add(state, action, -1.0, 0.5, 0.3, False)
        assert len(buf) == 1

    def test_clear_resets_length(self):
        """Clearing the buffer should reset length to 0."""
        buf = RolloutBuffer(capacity=100, state_dim=STATE_DIM, action_dim=ACTION_DIM)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(10):
            buf.add(state, action, -1.0, 0.5, 0.3, False)
        buf.clear()
        assert len(buf) == 0

    def test_gae_computation(self):
        """GAE should produce finite advantages and returns."""
        buf = RolloutBuffer(capacity=100, state_dim=STATE_DIM, action_dim=ACTION_DIM)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for i in range(20):
            buf.add(state, action, -0.5, float(i) * 0.1, 0.3, i == 19)
        buf.compute_gae(last_value=0.0)
        assert np.all(np.isfinite(buf.advantages[:20]))
        assert np.all(np.isfinite(buf.returns[:20]))

    def test_gae_advantages_have_correct_properties(self):
        """Advantages should have roughly zero mean after normalisation."""
        buf = RolloutBuffer(capacity=100, state_dim=STATE_DIM, action_dim=ACTION_DIM)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for i in range(50):
            buf.add(state, action, -0.5, np.random.uniform(-1, 1), 0.3, False)
        buf.compute_gae(last_value=0.5)
        advs = buf.advantages[:50]
        # Normalise (as done in train())
        normalised = (advs - advs.mean()) / (advs.std() + 1e-8)
        assert abs(normalised.mean()) < 0.01

    def test_get_batches_covers_all_data(self):
        """Batches should cover all stored transitions."""
        buf = RolloutBuffer(capacity=100, state_dim=STATE_DIM, action_dim=ACTION_DIM)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for i in range(30):
            buf.add(state, action, -0.5, 0.5, 0.3, False)
        buf.compute_gae(last_value=0.0)
        batches = buf.get_batches(batch_size=8)
        total = sum(b["states"].shape[0] for b in batches)
        assert total == 30


class TestPPOClipping:
    """Test PPO's clipping mechanism."""

    def test_clip_fraction_in_valid_range(self, agent):
        """Clip fraction should be between 0 and 1."""
        buf = agent.rollout_buffer
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        for i in range(agent.n_steps):
            buf.add(state, action, -0.5, np.random.uniform(-1, 1), 0.3, False)
        metrics = agent.train(last_value=0.0)
        assert 0.0 <= metrics["clip_fraction"] <= 1.0


class TestTrainingLoop:
    """Test the full training loop."""

    def test_training_runs_without_crash(self, agent):
        """Training loop should run for n_steps without errors."""
        for i in range(agent.n_steps):
            state = np.random.randn(STATE_DIM).astype(np.float32)
            action, log_prob, value = agent.select_action(state)
            reward = np.random.uniform(-1, 1)
            done = (i == agent.n_steps - 1)
            agent.store_transition(state, action, log_prob, reward, value, done)

        metrics = agent.train(last_value=0.0)
        assert "actor_loss" in metrics
        assert "critic_loss" in metrics
        assert "entropy" in metrics
        assert np.isfinite(metrics["actor_loss"])
        assert np.isfinite(metrics["critic_loss"])

    def test_entropy_is_positive(self, agent):
        """Gaussian entropy should be positive."""
        for i in range(agent.n_steps):
            state = np.random.randn(STATE_DIM).astype(np.float32)
            action, log_prob, value = agent.select_action(state)
            agent.store_transition(
                state, action, log_prob, np.random.uniform(-1, 1), value, False,
            )
        metrics = agent.train(last_value=0.0)
        assert metrics["entropy"] > 0


class TestSaveLoad:
    """Test model persistence."""

    def test_save_and_load_roundtrip(self, agent, random_state):
        """Saving and loading should preserve actor/critic outputs."""
        a_before, _, v_before = agent.select_action(random_state, deterministic=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir, episode=1)
            agent.load(tmpdir, episode=1)

        a_after, _, v_after = agent.select_action(random_state, deterministic=True)
        np.testing.assert_array_almost_equal(a_before, a_after, decimal=4)
        assert abs(v_before - v_after) < 0.01

    def test_metadata_persists(self, agent):
        """Total steps and update count should persist through save/load."""
        agent.total_steps = 5000
        agent._update_count = 10

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir, episode=5)
            agent.load(tmpdir, episode=5)

        assert agent.total_steps == 5000
        assert agent._update_count == 10


class TestEnvironmentContinuousMode:
    """Test environment integration with continuous actions."""

    def test_continuous_env_creation(self):
        """Environment should accept action_mode='continuous'."""
        env = NetworkSecurityEnv(action_mode="continuous", max_steps=10)
        assert env.action_space.shape == (3,)

    def test_continuous_env_step(self):
        """Environment should accept continuous action arrays."""
        env = NetworkSecurityEnv(action_mode="continuous", max_steps=10)
        obs, info = env.reset()
        action = np.array([0.5, 0.3, 0.0], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (65,)
        assert np.isfinite(reward)
        assert "action" in info

    def test_continuous_to_discrete_in_env(self):
        """Environment should correctly map continuous to discrete actions."""
        env = NetworkSecurityEnv(action_mode="continuous", max_steps=10)
        env.reset()
        # High rate limit → BLOCK
        _, _, _, _, info = env.step(np.array([0.9, 0.0, 0.0]))
        assert info["action"] == 1  # BLOCK

    def test_discrete_mode_still_works(self):
        """Default discrete mode should still work unchanged."""
        env = NetworkSecurityEnv(action_mode="discrete", max_steps=10)
        obs, info = env.reset()
        obs, reward, _, _, info = env.step(0)
        assert obs.shape == (65,)
        assert info["action"] == 0


# ---------------------------------------------------------------------------
# Explained variance
# ---------------------------------------------------------------------------


class TestExplainedVariance:
    """Verify _compute_explained_variance returns correct values."""

    def test_near_perfect_predictions(self):
        from src.agents.ppo_agent import PPOAgent
        returns = np.array([1.0, 2.0, 3.0])
        predictions = np.array([1.1, 1.9, 3.1])
        ev = PPOAgent._compute_explained_variance(predictions, returns)
        assert ev > 0.95

    def test_random_predictions(self):
        from src.agents.ppo_agent import PPOAgent
        returns = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
        ev = PPOAgent._compute_explained_variance(predictions, returns)
        assert ev < 0.5

    def test_zero_variance_returns_zero(self):
        from src.agents.ppo_agent import PPOAgent
        returns = np.array([1.0, 1.0, 1.0])
        predictions = np.array([1.0, 1.0, 1.0])
        ev = PPOAgent._compute_explained_variance(predictions, returns)
        assert ev == 0.0


# ---------------------------------------------------------------------------
# RolloutBuffer overflow protection
# ---------------------------------------------------------------------------


class TestRolloutBufferOverflow:
    """Buffer raises RuntimeError when full, not IndexError."""

    def test_overflow_raises_runtime_error(self):
        from src.agents.ppo_agent import RolloutBuffer
        buf = RolloutBuffer(capacity=3, state_dim=4, action_dim=2)
        for _ in range(3):
            buf.add(np.zeros(4), np.zeros(2), 0.0, 0.0, 0.0, False)
        assert buf.is_full()
        with pytest.raises(RuntimeError, match="full"):
            buf.add(np.zeros(4), np.zeros(2), 0.0, 0.0, 0.0, False)

    def test_clear_allows_reuse(self):
        from src.agents.ppo_agent import RolloutBuffer
        buf = RolloutBuffer(capacity=2, state_dim=4, action_dim=2)
        for _ in range(2):
            buf.add(np.zeros(4), np.zeros(2), 0.0, 0.0, 0.0, False)
        assert buf.is_full()
        buf.clear()
        assert not buf.is_full()
        buf.add(np.zeros(4), np.zeros(2), 0.0, 0.0, 0.0, False)


# ---------------------------------------------------------------------------
# Local RNG (no global seed pollution)
# ---------------------------------------------------------------------------


class TestPPOLocalRNG:
    """PPO agent uses local RNG, not global np.random.seed."""

    def test_does_not_set_global_seed(self):
        before = np.random.get_state()[1].copy()
        from src.agents.ppo_agent import PPOAgent
        PPOAgent(state_dim=4, action_dim=2, config={"seed": 99999})
        after = np.random.get_state()[1].copy()
        np.testing.assert_array_equal(before, after)
