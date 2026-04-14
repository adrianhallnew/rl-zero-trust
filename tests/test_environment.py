"""Unit tests for the Gymnasium RL environment.

Tests cover:
    - Environment reset returns valid state vector (correct dimensions).
    - Environment step returns (next_state, reward, done, info) correctly.
    - Action space and observation space are properly defined.
    - Simulation mode produces varying observations.
    - Episode truncation at max_steps.
    - Info dict contains expected keys.
"""

import pytest
import numpy as np

from src.environment.network_env import NetworkSecurityEnv
from src.environment.state_processor import STATE_DIM

NUM_ACTIONS = 4


@pytest.fixture
def default_reward_config():
    return {
        "w_det": 0.40, "w_fp": 0.25, "w_thr": 0.15,
        "w_lat": 0.10, "w_stab": 0.10,
        "epsilon": 1e-8,
        "max_acceptable_latency_ms": 50.0,
        "max_changes_per_step": 5,
    }


@pytest.fixture
def env(default_reward_config):
    """Simulation-mode environment with short episodes."""
    return NetworkSecurityEnv(
        reward_config=default_reward_config,
        max_steps=50,
        seed=42,
    )


class TestEnvironmentSpaces:
    """Test observation and action space definitions."""

    def test_observation_space_shape(self, env):
        """Observation space should be Box(65,)."""
        assert env.observation_space.shape == (STATE_DIM,)

    def test_action_space_size(self, env):
        """Action space should be Discrete(4)."""
        assert env.action_space.n == NUM_ACTIONS

    def test_observation_space_dtype(self, env):
        """Observation space dtype should be float32."""
        assert env.observation_space.dtype == np.float32


class TestReset:
    """Test environment reset."""

    def test_reset_returns_tuple(self, env):
        """reset() should return (observation, info)."""
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_observation_shape(self, env):
        """Reset observation should have correct shape."""
        obs, _ = env.reset()
        assert obs.shape == (STATE_DIM,)
        assert obs.dtype == np.float32

    def test_reset_observation_finite(self, env):
        """Reset observation should contain finite values."""
        obs, _ = env.reset()
        assert np.all(np.isfinite(obs))

    def test_reset_info_dict(self, env):
        """Info dict should contain episode number and mode."""
        _, info = env.reset()
        assert "episode" in info
        assert "mode" in info
        assert info["mode"] == "simulation"


class TestStep:
    """Test environment step."""

    def test_step_returns_five_items(self, env):
        """step() should return (obs, reward, terminated, truncated, info)."""
        env.reset()
        result = env.step(0)
        assert len(result) == 5

    def test_step_observation_shape(self, env):
        """Step observation should have correct shape."""
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (STATE_DIM,)

    def test_step_reward_is_scalar(self, env):
        """Reward should be a scalar float."""
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert isinstance(reward, float)
        assert np.isfinite(reward)

    def test_step_info_has_expected_keys(self, env):
        """Info dict should contain step metadata."""
        env.reset()
        _, _, _, _, info = env.step(1)
        expected_keys = {"step", "action", "action_name", "reward_components",
                         "metrics", "cumulative_reward", "attack_active"}
        assert expected_keys.issubset(set(info.keys()))

    def test_all_actions_valid(self, env):
        """All 4 actions should execute without error."""
        env.reset()
        for action in range(NUM_ACTIONS):
            obs, reward, term, trunc, info = env.step(action)
            assert obs.shape == (STATE_DIM,)
            assert isinstance(reward, float)

    def test_invalid_action_raises(self, env):
        """Actions outside {0,1,2,3} should raise an error."""
        env.reset()
        with pytest.raises(AssertionError):
            env.step(5)


class TestEpisodeTruncation:
    """Test that episodes end after max_steps."""

    def test_truncation_at_max_steps(self, env):
        """Episode should be truncated after max_steps."""
        env.reset()
        truncated = False
        for _ in range(env.max_steps):
            _, _, _, truncated, _ = env.step(0)
        assert truncated is True

    def test_not_truncated_before_max(self, env):
        """Episode should NOT be truncated before max_steps."""
        env.reset()
        for _ in range(env.max_steps - 1):
            _, _, _, truncated, _ = env.step(0)
        assert truncated is False


class TestSimulationMode:
    """Test the simulation-mode traffic simulator."""

    def test_observations_vary(self, env):
        """Consecutive observations should not be identical."""
        env.reset()
        obs1, _, _, _, _ = env.step(0)
        obs2, _, _, _, _ = env.step(0)
        # States won't be exactly equal due to noise
        assert not np.array_equal(obs1, obs2)

    def test_attack_injection(self, env):
        """Over many steps, at least one attack should trigger."""
        env.reset()
        attack_seen = False
        for _ in range(100):
            _, _, _, truncated, info = env.step(0)
            if info.get("attack_active"):
                attack_seen = True
                break
            if truncated:
                env.reset()
        assert attack_seen, "No attack triggered in 100 steps"

    def test_reward_varies_with_action(self):
        """Different actions should produce different rewards on average."""
        config = {
            "w_det": 0.40, "w_fp": 0.25, "w_thr": 0.15,
            "w_lat": 0.10, "w_stab": 0.10,
            "epsilon": 1e-8,
            "max_acceptable_latency_ms": 50.0,
            "max_changes_per_step": 5,
        }
        rewards_per_action = {a: [] for a in range(NUM_ACTIONS)}

        for action in range(NUM_ACTIONS):
            env = NetworkSecurityEnv(
                reward_config=config, max_steps=200, seed=42,
            )
            env.reset()
            for _ in range(200):
                _, reward, _, truncated, _ = env.step(action)
                rewards_per_action[action].append(reward)
                if truncated:
                    env.reset()

        # ALLOW (action=0) should differ from BLOCK (action=1) on average
        avg_allow = np.mean(rewards_per_action[0])
        avg_block = np.mean(rewards_per_action[1])
        assert avg_allow != pytest.approx(avg_block, abs=0.01)


class TestRender:
    """Test rendering output."""

    def test_ansi_render(self, env):
        """ansi mode should return a string."""
        env.reset()
        env.step(0)
        output = env.render(mode="ansi")
        assert isinstance(output, str)
        assert "Episode" in output
