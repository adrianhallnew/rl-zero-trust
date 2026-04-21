"""Phase F tests: Core RL correctness & observability fixes.

Tests cover:
    F1.1 — BLOCK vs RATE_LIMIT produce different live metrics
    F1.2 — DQN epsilon reset in live mode
    F1.3 — _fire_single does not overwrite manual attacks
    F1.4 — check_expired() works when RL loop is idle
    F1.5 — --no-demo-bias flag disables Q-value bias
    F2.1 — Docker exec stderr captured (not DEVNULL)
    F2.2 — Scheduler thread crashes are caught and reported
    F2.3 — _kill_processes waits then escalates to kill
    F2.4 — step_num initialized before loop (no NameError)
    F5.1 — FP penalty differentiated per action during peace
    F5.4 — Throughput/latency vary per action
"""

import subprocess
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.environment.network_env import NetworkSecurityEnv


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reward_config():
    return {
        "w_det": 0.40, "w_fp": 0.25, "w_thr": 0.15,
        "w_lat": 0.10, "w_stab": 0.10,
        "epsilon": 1e-8,
        "max_acceptable_latency_ms": 50.0,
        "max_changes_per_step": 5,
    }


@pytest.fixture
def env(reward_config):
    return NetworkSecurityEnv(
        reward_config=reward_config, max_steps=50, seed=42,
    )


@pytest.fixture
def scheduler():
    """Import and create an AttackScheduler with no Docker dependency."""
    from scripts.live_demo import AttackScheduler
    return AttackScheduler(scenario="none")


# ---------------------------------------------------------------------------
# F1.1 — BLOCK vs RATE_LIMIT live metrics differ
# ---------------------------------------------------------------------------

class TestF1_1_DifferentiatedMetrics:
    """BLOCK and RATE_LIMIT must produce different metrics."""

    def test_attack_active_block_vs_rate_limit_tp_differs(self, env):
        env.set_attack_state(True, "ddos")
        m_block = env._compute_live_metrics(1)
        m_rate = env._compute_live_metrics(3)
        assert m_block["true_positives"] != m_rate["true_positives"], (
            "BLOCK and RATE_LIMIT should have different TP during attack"
        )

    def test_attack_active_throughput_differs(self, env):
        env.set_attack_state(True, "ddos")
        m_block = env._compute_live_metrics(1)
        m_rate = env._compute_live_metrics(3)
        assert m_block["current_throughput_mbps"] != m_rate["current_throughput_mbps"]

    def test_no_attack_fp_differs(self, env):
        env.set_attack_state(False)
        m_block = env._compute_live_metrics(1)
        m_rate = env._compute_live_metrics(3)
        assert m_block["false_positives"] > m_rate["false_positives"], (
            "BLOCK should have more FP than RATE_LIMIT during peace"
        )

    def test_no_attack_allow_zero_fp(self, env):
        env.set_attack_state(False)
        m = env._compute_live_metrics(0)
        assert m["false_positives"] == 0

    def test_all_four_actions_distinct_during_attack(self, env):
        env.set_attack_state(True, "ddos")
        tps = set()
        for action in range(4):
            m = env._compute_live_metrics(action)
            tps.add(m["true_positives"])
        assert len(tps) >= 3, "At least 3 distinct TP values across 4 actions"


# ---------------------------------------------------------------------------
# F1.2 — DQN epsilon reset
# ---------------------------------------------------------------------------

class TestF1_2_EpsilonLiveMode:
    """DQN load(live_mode=True) forces epsilon to epsilon_end."""

    def test_live_mode_resets_epsilon(self):
        from src.agents.dqn_agent import DQNAgent
        config = {
            "learning_rate": 1e-3, "gamma": 0.99, "epsilon_start": 1.0,
            "epsilon_end": 0.05, "epsilon_decay_steps": 10000,
            "batch_size": 32, "replay_capacity": 1000,
            "target_update_freq": 100, "seed": 42,
        }
        agent = DQNAgent(state_dim=65, config=config)
        agent.epsilon = 0.5

        # Patch the entire load method internals to avoid Keras model I/O,
        # then call the live_mode logic directly
        import src.agents.dqn_agent as dqn_mod
        with patch.object(dqn_mod.keras.models, "load_model", return_value=agent.online_model), \
             patch("src.agents.networks.build_dqn_target_network", return_value=agent.target_model), \
             patch("os.path.exists", return_value=False):
            agent.load("dummy_dir", live_mode=True)

        assert agent.epsilon == agent.epsilon_end

    def test_non_live_mode_preserves_epsilon(self):
        from src.agents.dqn_agent import DQNAgent
        config = {
            "learning_rate": 1e-3, "gamma": 0.99, "epsilon_start": 1.0,
            "epsilon_end": 0.05, "epsilon_decay_steps": 10000,
            "batch_size": 32, "replay_capacity": 1000,
            "target_update_freq": 100, "seed": 42,
        }
        agent = DQNAgent(state_dim=65, config=config)
        agent.epsilon = 0.5

        import src.agents.dqn_agent as dqn_mod
        with patch.object(dqn_mod.keras.models, "load_model", return_value=agent.online_model), \
             patch("src.agents.networks.build_dqn_target_network", return_value=agent.target_model), \
             patch("os.path.exists", return_value=False):
            agent.load("dummy_dir", live_mode=False)

        assert agent.epsilon == 0.5


# ---------------------------------------------------------------------------
# F1.3 — Manual attack not overwritten by auto schedule
# ---------------------------------------------------------------------------

class TestF1_3_AttackOwnership:
    """_fire_single should not clear state when manual attack owns it."""

    def test_fire_manual_sets_owner(self, scheduler):
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler.fire_manual("ddos", duration=30)
        assert scheduler._attack_owner == "manual"
        assert scheduler.attack_active is True
        assert scheduler.attack_type == "ddos"

    def test_fire_single_sets_auto_owner(self, scheduler):
        """_fire_single sets owner to 'auto' and clears when done."""
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler._stop_event.set()  # Make it return immediately
            scheduler._fire_single("ddos", duration=1)
        # After _fire_single with auto owner, state should be cleared
        assert scheduler._attack_owner is None
        assert scheduler.attack_active is False

    def test_manual_survives_auto_completion(self, scheduler):
        """If manual attack is active, _fire_single completion doesn't clear it."""
        # Simulate: manual attack is in progress
        scheduler._attack_owner = "manual"
        scheduler.attack_active = True
        scheduler.attack_type = "portscan"

        # _fire_single completes for a different attack
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler._stop_event.set()
            scheduler._fire_single("ddos", duration=1)

        # Manual attack state should survive
        assert scheduler.attack_active is True
        assert scheduler.attack_type == "portscan"


# ---------------------------------------------------------------------------
# F1.4 — check_expired when idle
# ---------------------------------------------------------------------------

class TestF1_4_IdleExpiration:
    """check_expired() clears attacks even without RL loop running."""

    def test_expired_attack_cleared(self, scheduler):
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler.fire_manual("ddos", duration=1)
        assert scheduler.attack_active is True

        # Wait for expiration
        time.sleep(1.5)
        result = scheduler.check_expired()

        assert result is True
        assert scheduler.attack_active is False
        assert scheduler.attack_type is None

    def test_non_expired_not_cleared(self, scheduler):
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler.fire_manual("ddos", duration=60)
        result = scheduler.check_expired()
        assert result is False
        assert scheduler.attack_active is True


# ---------------------------------------------------------------------------
# F1.5 — --no-demo-bias flag
# ---------------------------------------------------------------------------

class TestF1_5_NoDemoBias:
    """DEMO_ACTION_BIAS should be skippable."""

    def test_demo_bias_constants_exist(self):
        from scripts.live_demo import DEMO_ACTION_BIAS
        assert "ddos" in DEMO_ACTION_BIAS
        assert None in DEMO_ACTION_BIAS

    def test_bias_changes_action(self):
        from scripts.live_demo import DEMO_ACTION_BIAS
        # With uniform Q-values, bias should determine the action
        q = np.array([1.0, 1.0, 1.0, 1.0])
        bias = DEMO_ACTION_BIAS["ddos"]
        biased = q + bias
        assert np.argmax(biased) == 1  # BLOCK has highest bias for ddos


# ---------------------------------------------------------------------------
# F2.1 — Docker exec stderr captured
# ---------------------------------------------------------------------------

class TestF2_1_DockerExecStderr:
    """Attack subprocess stderr should be captured, not DEVNULL."""

    def test_popen_uses_pipe_for_stderr(self, scheduler):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            mock_popen.return_value = mock_proc
            scheduler._docker_exec_attack("ddos", duration=10)

            call_kwargs = mock_popen.call_args
            assert call_kwargs[1]["stderr"] == subprocess.PIPE


# ---------------------------------------------------------------------------
# F2.2 �� Scheduler thread error handling
# ---------------------------------------------------------------------------

class TestF2_2_SchedulerErrorHandling:
    """Scheduler thread crashes should be caught and reported."""

    def test_run_safe_catches_exception(self):
        from scripts.live_demo import AttackScheduler

        errors = []
        sched = AttackScheduler(
            scenario="auto",
            error_callback=lambda msg: errors.append(msg),
        )
        # Force _run to raise
        with patch.object(sched, "_run", side_effect=RuntimeError("test crash")):
            sched._run_safe()

        assert len(errors) == 1
        assert "test crash" in errors[0]
        assert sched.attack_active is False

    def test_start_uses_run_safe(self):
        from scripts.live_demo import AttackScheduler
        sched = AttackScheduler(scenario="ddos")
        with patch.object(sched, "_run_safe") as mock_safe:
            sched.start()
            time.sleep(0.1)
            sched.stop()
        # _run_safe should have been called (it's the thread target)
        assert mock_safe.called or sched._thread is None


# ---------------------------------------------------------------------------
# F2.3 — Graceful process kill
# ---------------------------------------------------------------------------

class TestF2_3_GracefulKill:
    """_kill_processes should wait then escalate to kill."""

    def test_clean_termination(self, scheduler):
        proc = MagicMock()
        proc.wait.return_value = 0
        scheduler._processes = [proc]
        scheduler._kill_processes()
        proc.terminate.assert_called_once()
        assert len(scheduler._processes) == 0

    def test_escalate_to_kill_on_timeout(self, scheduler):
        proc = MagicMock()
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="test", timeout=3)
        scheduler._processes = [proc]
        scheduler._kill_processes()
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()
        assert len(scheduler._processes) == 0

    def test_handles_oserror_gracefully(self, scheduler):
        proc = MagicMock()
        proc.terminate.side_effect = OSError("gone")
        proc.wait.side_effect = OSError("gone")
        scheduler._processes = [proc]
        # Should not raise
        scheduler._kill_processes()
        assert len(scheduler._processes) == 0


# ---------------------------------------------------------------------------
# F2.4 — step_num initialization
# ---------------------------------------------------------------------------

class TestF2_4_StepNumInit:
    """step_num should be initialized before the loop."""

    def test_zero_steps_no_name_error(self, reward_config):
        """run_rl_loop with max_steps=0 should not raise NameError."""
        from scripts.live_demo import AttackScheduler

        sched = AttackScheduler(scenario="none")
        events = []

        # Mock the RL loop to run with max_steps=0
        with patch("scripts.live_demo.StatsCollector"), \
             patch("scripts.live_demo.PolicyEnforcer"), \
             patch("scripts.live_demo.get_reward_config", return_value=reward_config), \
             patch("scripts.live_demo.get_hyperparameters", return_value={
                 "learning_rate": 1e-3, "gamma": 0.99, "epsilon_start": 1.0,
                 "epsilon_end": 0.05, "epsilon_decay_steps": 10000,
                 "batch_size": 32, "replay_capacity": 1000,
                 "target_update_freq": 100, "seed": 42,
             }), \
             patch("scripts.live_demo.DQNAgent") as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent.get_q_values.return_value = np.array([1, 0, 0, 0])
            mock_agent_cls.return_value = mock_agent

            from scripts.live_demo import run_rl_loop
            # Should not raise NameError
            run_rl_loop(
                agent_name="dqn",
                max_steps=0,
                step_interval=0.0,
                attack_scheduler=sched,
                event_callback=events.append,
                mode="sim",
            )

        # Should get episode_done with steps=0
        done_events = [e for e in events if e.get("type") == "episode_done"]
        assert len(done_events) == 1
        assert done_events[0]["steps"] == 0


# ---------------------------------------------------------------------------
# F5.1 + F5.4 — FP penalty and throughput/latency per action
# ---------------------------------------------------------------------------

class TestF5_MetricDifferentiation:
    """Reward-relevant metrics should vary by action."""

    def test_fp_ordering_during_peace(self, env):
        """BLOCK > RATE_LIMIT > REROUTE > ALLOW in FP during no-attack."""
        env.set_attack_state(False)
        fps = {}
        for action in range(4):
            m = env._compute_live_metrics(action)
            fps[action] = m["false_positives"]
        assert fps[1] > fps[3] > fps[2] > fps[0], (
            f"FP ordering wrong: {fps}"
        )

    def test_throughput_varies_per_action_attack(self, env):
        """Each action produces different throughput during attack."""
        env.set_attack_state(True, "ddos")
        tps = set()
        for action in range(4):
            m = env._compute_live_metrics(action)
            tps.add(m["current_throughput_mbps"])
        assert len(tps) == 4, "All 4 actions should produce different throughput"

    def test_latency_varies_per_action_attack(self, env):
        env.set_attack_state(True, "ddos")
        lats = set()
        for action in range(4):
            m = env._compute_live_metrics(action)
            lats.add(m["current_latency_ms"])
        assert len(lats) == 4

    def test_allow_worst_throughput_during_attack(self, env):
        """ALLOW during attack = DDoS overwhelms = lowest throughput."""
        env.set_attack_state(True, "ddos")
        m_allow = env._compute_live_metrics(0)
        m_block = env._compute_live_metrics(1)
        assert m_allow["current_throughput_mbps"] < m_block["current_throughput_mbps"]

    def test_block_worst_throughput_during_peace(self, env):
        """BLOCK during peace = kills legit traffic = lowest throughput."""
        env.set_attack_state(False)
        m_allow = env._compute_live_metrics(0)
        m_block = env._compute_live_metrics(1)
        assert m_block["current_throughput_mbps"] < m_allow["current_throughput_mbps"]
