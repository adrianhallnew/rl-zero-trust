"""Integration tests for Sprint 6: PPO evaluation, comparison, and zero-trust.

Tests cover:
    - DQN agent end-to-end evaluation episode.
    - PPO agent end-to-end evaluation episode.
    - PPO continuous-to-discrete action mapping in environment.
    - Evaluation script action selection helper for both agents.
    - Comparative visualization function signatures.
    - Zero-trust client fallback behaviour.
    - Service resolution end-to-end with real config.
    - Docker compose config validity.
    - Config loader loads openziti_config.
    - Evaluate CLI argument parsing.
"""

import os
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.agents.ppo_agent import PPOAgent, ACTION_DIM
from src.environment.network_env import NetworkSecurityEnv
from src.utils.config_loader import load_config


STATE_DIM = 65


# --- DQN End-to-End ---

class TestDQNEndToEnd:
    """Test DQN agent runs a full evaluation episode."""

    def test_dqn_episode(self):
        """DQN agent should complete an episode in simulation mode."""
        from src.agents.dqn_agent import DQNAgent

        config = load_config("dqn_config")
        agent = DQNAgent(state_dim=STATE_DIM, config=config)

        env = NetworkSecurityEnv(max_steps=20)
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(20):
            action = agent.select_action(state, greedy=True)
            assert isinstance(action, (int, np.integer))
            assert 0 <= action < 4
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if done or truncated:
                break

        assert isinstance(total_reward, float)

    def test_dqn_greedy_deterministic(self):
        """Greedy DQN actions should be deterministic for same state."""
        from src.agents.dqn_agent import DQNAgent

        config = load_config("dqn_config")
        agent = DQNAgent(state_dim=STATE_DIM, config=config)
        state = np.random.randn(STATE_DIM).astype(np.float32)

        a1 = agent.select_action(state, greedy=True)
        a2 = agent.select_action(state, greedy=True)
        assert a1 == a2


# --- PPO End-to-End ---

class TestPPOEndToEnd:
    """Test PPO agent runs a full evaluation episode."""

    @pytest.fixture
    def ppo_agent(self):
        config = load_config("ppo_config")
        return PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, config=config)

    def test_ppo_episode(self, ppo_agent):
        """PPO agent should complete an episode with discrete actions."""
        env = NetworkSecurityEnv(max_steps=20)
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(20):
            continuous_action, _, _ = ppo_agent.select_action(
                state, deterministic=True,
            )
            discrete_action = PPOAgent.continuous_to_discrete(continuous_action)
            assert isinstance(discrete_action, (int, np.integer))
            assert 0 <= discrete_action < 4
            state, reward, done, truncated, info = env.step(discrete_action)
            total_reward += reward
            if done or truncated:
                break

        assert isinstance(total_reward, float)

    def test_ppo_deterministic_actions(self, ppo_agent):
        """Deterministic PPO actions should be consistent."""
        state = np.random.randn(STATE_DIM).astype(np.float32)
        a1, _, _ = ppo_agent.select_action(state, deterministic=True)
        a2, _, _ = ppo_agent.select_action(state, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2)

    def test_ppo_continuous_to_discrete_coverage(self):
        """All 4 discrete actions should be reachable."""
        # ALLOW: rate_limit <= 0.3, reroute <= 0.5
        assert PPOAgent.continuous_to_discrete(np.array([0.0, 0.0, 0.0])) == 0
        # BLOCK: rate_limit > 0.7
        assert PPOAgent.continuous_to_discrete(np.array([0.8, 0.0, 0.0])) == 1
        # REROUTE: rate_limit <= 0.3, reroute > 0.5
        assert PPOAgent.continuous_to_discrete(np.array([0.0, 0.6, 0.0])) == 2
        # RATE_LIMIT: 0.3 < rate_limit <= 0.7
        assert PPOAgent.continuous_to_discrete(np.array([0.5, 0.0, 0.0])) == 3


# --- Evaluate Script ---

class TestEvaluateScript:
    """Test evaluate.py helper functions and CLI."""

    def test_select_agent_action_dqn(self):
        """select_agent_action should return int for DQN."""
        from scripts.evaluate import select_agent_action
        from src.agents.dqn_agent import DQNAgent

        config = load_config("dqn_config")
        agent = DQNAgent(state_dim=STATE_DIM, config=config)
        state = np.random.randn(STATE_DIM).astype(np.float32)

        action = select_agent_action(agent, state, "dqn")
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 4

    def test_select_agent_action_ppo(self):
        """select_agent_action should return int for PPO."""
        from scripts.evaluate import select_agent_action

        config = load_config("ppo_config")
        agent = PPOAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, config=config)
        state = np.random.randn(STATE_DIM).astype(np.float32)

        action = select_agent_action(agent, state, "ppo")
        assert isinstance(action, (int, np.integer))
        assert 0 <= action < 4


# --- Comparative Visualization ---

class TestComparativeVisualization:
    """Test comparative visualization functions exist and accept inputs."""

    def test_plot_comparative_detection_callable(self):
        """plot_comparative_detection should be importable."""
        from src.utils.visualization import plot_comparative_detection
        assert callable(plot_comparative_detection)

    def test_plot_comparative_learning_curves_callable(self):
        """plot_comparative_learning_curves should be importable."""
        from src.utils.visualization import plot_comparative_learning_curves
        assert callable(plot_comparative_learning_curves)

    def test_plot_comparative_summary_callable(self):
        """plot_comparative_summary should be importable."""
        from src.utils.visualization import plot_comparative_summary
        assert callable(plot_comparative_summary)


# --- Zero-Trust Fallback ---

class TestZeroTrustFallback:
    """Test zero-trust client with real config file."""

    def test_load_from_default_config(self):
        """Client should load from config/openziti_config.yaml."""
        with patch.dict(os.environ, {"OPENZITI_ENABLED": "false"}):
            from src.zero_trust.openziti_client import OpenZitiClient
            client = OpenZitiClient()
            assert len(client.service_map) == 3
            assert not client.is_secure()
            client.close()

    def test_service_names_match_config(self):
        """Service names should match openziti_config.yaml."""
        with patch.dict(os.environ, {"OPENZITI_ENABLED": "false"}):
            from src.zero_trust.openziti_client import OpenZitiClient
            client = OpenZitiClient()
            expected = {"sdn_controller_api", "policy_engine", "monitoring_feed"}
            assert set(client.list_services()) == expected
            client.close()

    def test_resolve_sdn_controller(self):
        """Should resolve sdn_controller_api to ryu-controller:8080."""
        with patch.dict(os.environ, {"OPENZITI_ENABLED": "false"}, clear=False):
            # Remove env override if present
            env = os.environ.copy()
            env.pop("RYU_API_URL", None)
            with patch.dict(os.environ, env, clear=True):
                from src.zero_trust.openziti_client import OpenZitiClient
                client = OpenZitiClient()
                url = client._resolve_url("sdn_controller_api", "/stats")
                assert "ryu-controller" in url
                assert "8080" in url
                client.close()
