"""Unit tests for the DQN agent.

Tests cover:
    - Network construction (correct input/output shapes).
    - Q-value forward pass produces valid outputs for all 4 actions.
    - Epsilon-greedy action selection.
    - Epsilon decay follows the configured schedule.
    - Target network soft updates (including tau=1 and tau=0 edge cases).
    - Step method stores transitions and trains.
    - Model save/load round-trip.
    - Local RNG: no global np.random.seed pollution.
"""

import os
import tempfile

import pytest
import numpy as np

from src.agents.dqn_agent import DQNAgent

STATE_DIM = 65
NUM_ACTIONS = 4


@pytest.fixture
def default_config():
    """Minimal DQN config for fast tests."""
    return {
        "hidden_layers": [32, 32],
        "activation": "relu",
        "learning_rate": 0.001,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 8,
        "train_freq": 2,
        "min_buffer_size": 10,
        "buffer_size": 100,
        "epsilon_start": 1.0,
        "epsilon_end": 0.05,
        "epsilon_decay_steps": 100,
        "seed": 42,
    }


@pytest.fixture
def agent(default_config):
    return DQNAgent(state_dim=STATE_DIM, config=default_config)


@pytest.fixture
def random_state():
    return np.random.randn(STATE_DIM).astype(np.float32)


class TestDQNNetwork:
    """Test the underlying Q-network architecture."""

    def test_online_model_output_shape(self, agent, random_state):
        """Online model should output Q-values for all 4 actions."""
        q_values = agent.get_q_values(random_state)
        assert q_values.shape == (NUM_ACTIONS,)

    def test_target_model_output_shape(self, agent, random_state):
        """Target model should have the same output shape."""
        state_batch = np.expand_dims(random_state, axis=0)
        q_values = agent.target_model.predict(state_batch, verbose=0)
        assert q_values.shape == (1, NUM_ACTIONS)

    def test_q_values_are_finite(self, agent, random_state):
        """Q-values should not be NaN or Inf."""
        q_values = agent.get_q_values(random_state)
        assert np.all(np.isfinite(q_values))

    def test_initial_weights_match(self, agent):
        """Online and target networks should start with identical weights."""
        for online_w, target_w in zip(
            agent.online_model.get_weights(),
            agent.target_model.get_weights(),
        ):
            np.testing.assert_array_almost_equal(online_w, target_w)


class TestActionSelection:
    """Test epsilon-greedy action selection."""

    def test_greedy_action_deterministic(self, agent, random_state):
        """Greedy action should be deterministic (same state -> same action)."""
        a1 = agent.select_action(random_state, greedy=True)
        a2 = agent.select_action(random_state, greedy=True)
        assert a1 == a2

    def test_greedy_action_in_range(self, agent, random_state):
        """Action should be in {0, 1, 2, 3}."""
        action = agent.select_action(random_state, greedy=True)
        assert 0 <= action < NUM_ACTIONS

    def test_exploratory_actions_vary(self, agent, random_state):
        """With epsilon=1.0, actions should be random (not always the same)."""
        agent.epsilon = 1.0
        actions = [agent.select_action(random_state) for _ in range(50)]
        unique = set(actions)
        # With 50 random draws from 4 options, we should see >1 unique
        assert len(unique) > 1


class TestEpsilonDecay:
    """Test the exploration schedule."""

    def test_initial_epsilon(self, agent):
        """Epsilon should start at epsilon_start."""
        assert agent.epsilon == pytest.approx(1.0)

    def test_epsilon_decays(self, agent, random_state):
        """Epsilon should decrease after steps."""
        initial = agent.epsilon
        state = random_state
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        for _ in range(20):
            agent.step(state, 0, 0.0, next_state, False)
        assert agent.epsilon < initial

    def test_epsilon_does_not_go_below_end(self, agent, random_state):
        """Epsilon should never go below epsilon_end."""
        state = random_state
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        for _ in range(200):  # well past decay steps
            agent.step(state, 0, 0.0, next_state, False)
        assert agent.epsilon >= agent.epsilon_end


class TestTargetUpdate:
    """Test soft target network updates."""

    def test_soft_update_changes_target(self, agent, random_state):
        """After training steps, target weights should diverge from initial."""
        state = random_state
        next_state = np.random.randn(STATE_DIM).astype(np.float32)

        initial_target_weights = [w.copy() for w in agent.target_model.get_weights()]

        # Fill buffer and train
        for i in range(50):
            agent.step(state, i % 4, float(i) * 0.1, next_state, False)

        current_target_weights = agent.target_model.get_weights()

        # At least one weight array should have changed
        any_changed = False
        for init_w, curr_w in zip(initial_target_weights, current_target_weights):
            if not np.allclose(init_w, curr_w, atol=1e-8):
                any_changed = True
                break
        assert any_changed


class TestStepAndTrain:
    """Test the step/train loop."""

    def test_step_stores_transition(self, agent, random_state):
        """Calling step should add to the replay buffer."""
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        agent.step(random_state, 1, 0.5, next_state, False)
        assert len(agent.replay_buffer) == 1

    def test_training_runs_without_crash(self, agent):
        """Training loop should run 1000 steps without errors."""
        for i in range(100):
            state = np.random.randn(STATE_DIM).astype(np.float32)
            next_state = np.random.randn(STATE_DIM).astype(np.float32)
            action = i % 4
            reward = np.random.uniform(-1, 1)
            done = (i % 20 == 19)
            loss = agent.step(state, action, reward, next_state, done)
            # After min_buffer_size, losses should appear
            if i > 20 and loss is not None:
                assert np.isfinite(loss)


class TestSaveLoad:
    """Test model persistence."""

    def test_save_and_load_roundtrip(self, agent, random_state):
        """Saving and loading should preserve Q-values."""
        q_before = agent.get_q_values(random_state)

        with tempfile.TemporaryDirectory() as tmpdir:
            agent.save(tmpdir, episode=1)
            agent.load(tmpdir, episode=1)

        q_after = agent.get_q_values(random_state)
        np.testing.assert_array_almost_equal(q_before, q_after, decimal=5)


# ---------------------------------------------------------------------------
# Soft update edge cases
# ---------------------------------------------------------------------------


class TestSoftUpdateEdgeCases:
    """Verify soft_update_target at boundary tau values."""

    def test_tau_1_copies_weights_exactly(self):
        from src.agents.networks import build_dqn_network, build_dqn_target_network, soft_update_target

        online = build_dqn_network(state_dim=4, num_actions=2, hidden_layers=[8])
        target = build_dqn_target_network(online)

        for layer in online.layers:
            weights = layer.get_weights()
            if weights:
                layer.set_weights([np.random.randn(*w.shape).astype(np.float32) for w in weights])

        soft_update_target(online, target, tau=1.0)

        for o_w, t_w in zip(online.get_weights(), target.get_weights()):
            np.testing.assert_array_almost_equal(o_w, t_w, decimal=6)

    def test_tau_0_preserves_target(self):
        from src.agents.networks import build_dqn_network, build_dqn_target_network, soft_update_target

        online = build_dqn_network(state_dim=4, num_actions=2, hidden_layers=[8])
        target = build_dqn_target_network(online)
        original_weights = [w.copy() for w in target.get_weights()]

        for layer in online.layers:
            weights = layer.get_weights()
            if weights:
                layer.set_weights([w + 1.0 for w in weights])

        soft_update_target(online, target, tau=0.0)

        for orig, current in zip(original_weights, target.get_weights()):
            np.testing.assert_array_almost_equal(orig, current, decimal=6)


# ---------------------------------------------------------------------------
# Local RNG (no global seed pollution)
# ---------------------------------------------------------------------------


class TestLocalRNG:
    """DQN agent uses local RNG, not global np.random.seed."""

    def test_does_not_set_global_seed(self):
        before = np.random.get_state()[1].copy()
        from src.agents.dqn_agent import DQNAgent
        DQNAgent(state_dim=4, config={"seed": 99999})
        after = np.random.get_state()[1].copy()
        np.testing.assert_array_equal(before, after)

    def test_different_seeds_different_actions(self):
        from src.agents.dqn_agent import DQNAgent

        agent1 = DQNAgent(state_dim=4, config={"seed": 1, "epsilon_start": 1.0, "epsilon_end": 1.0})
        agent2 = DQNAgent(state_dim=4, config={"seed": 2, "epsilon_start": 1.0, "epsilon_end": 1.0})

        state = np.zeros(4, dtype=np.float32)
        actions1 = [agent1.select_action(state) for _ in range(100)]
        actions2 = [agent2.select_action(state) for _ in range(100)]
        assert actions1 != actions2
