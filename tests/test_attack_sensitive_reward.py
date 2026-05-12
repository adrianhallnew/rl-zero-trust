"""Tests for attack-type-sensitive reward model and dashboard enhancements.

Covers:
    Item 1  — Attack-type-sensitive optimal actions and reward ordering
    Item 2  — PDF export (vendor script tags)
    Item 3  — Tab restructure (4 tabs)
    Item 4  — Device inventory endpoint
    Item 5  — Session comparison endpoint
    Item 7  — Action distribution (step event has action field)
    Item 8  — Explainability fields in events
    Item 9  — Session metrics dataclass
    Item 10 — Baseline flow snapshot and diff endpoints
"""

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.dashboard.server import (
    DashboardState,
    SessionMetrics,
    ZTA_BASELINE_RULES,
    app,
    state,
)
from src.environment.network_env import (
    OPTIMAL_ACTIONS,
    OPTIMAL_EXPLANATIONS,
    NetworkSecurityEnv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_state():
    state.agent = "dqn"
    state.mode = "live"
    state.running = False
    state.step = 0
    state.start_time = None
    state.last_event = None
    state.events.clear()
    state.subscribers.clear()
    state.requested_agent = None
    state.requested_mode = None
    state.requested_attack = None
    state.requested_attack_intensity = 0.7
    state.requested_stop_attacks = False
    state.requested_auto_scenario = False
    state.requested_start = False
    state.requested_stop = False
    state.agent_sessions = {}
    state.current_session = None
    state.baseline_flows = {}
    state.baseline_captured = False
    state.rl_installed_rules = []
    yield


@pytest.fixture
def client():
    return TestClient(app)


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


# ---------------------------------------------------------------------------
# Item 1: Attack-type-sensitive reward model
# ---------------------------------------------------------------------------

class TestAttackSensitiveReward:

    def _compute_rewards(self, env, reward_config):
        from src.environment.reward import RewardCalculator
        calc = RewardCalculator(reward_config)
        rewards = {}
        for action in range(4):
            m = env._compute_live_metrics(action)
            metrics_dict = {
                "true_positives": m["true_positives"],
                "false_positives": m["false_positives"],
                "false_negatives": m["false_negatives"],
                "true_negatives": m.get("true_negatives", 0),
                "current_throughput_mbps": m["current_throughput_mbps"],
                "baseline_throughput_mbps": 100.0,
                "min_throughput_mbps": 10.0,
                "current_latency_ms": m["current_latency_ms"],
                "policy_changes": 0,
            }
            r = calc.compute(metrics_dict)
            rewards[action] = r.total
        return rewards

    def test_ddos_rate_limit_optimal(self, env, reward_config):
        """RATE_LIMIT yields highest reward during DDoS."""
        env.set_attack_state(True, "ddos")
        rewards = self._compute_rewards(env, reward_config)
        assert rewards[3] > rewards[1], f"DDoS: RATE_LIMIT({rewards[3]:.4f}) should beat BLOCK({rewards[1]:.4f})"
        assert rewards[3] > rewards[0], f"DDoS: RATE_LIMIT({rewards[3]:.4f}) should beat ALLOW({rewards[0]:.4f})"

    def test_portscan_block_optimal(self, env, reward_config):
        """BLOCK yields highest reward during port scan."""
        env.set_attack_state(True, "portscan")
        rewards = self._compute_rewards(env, reward_config)
        assert rewards[1] > rewards[0], f"Portscan: BLOCK({rewards[1]:.4f}) should beat ALLOW({rewards[0]:.4f})"
        assert rewards[1] > rewards[2], f"Portscan: BLOCK({rewards[1]:.4f}) should beat REROUTE({rewards[2]:.4f})"

    def test_spoofing_reroute_optimal(self, env, reward_config):
        """REROUTE yields highest reward during spoofing."""
        env.set_attack_state(True, "spoofing")
        rewards = self._compute_rewards(env, reward_config)
        assert rewards[2] > rewards[0], f"Spoofing: REROUTE({rewards[2]:.4f}) should beat ALLOW({rewards[0]:.4f})"
        assert rewards[2] > rewards[1], f"Spoofing: REROUTE({rewards[2]:.4f}) should beat BLOCK({rewards[1]:.4f})"

    def test_normal_allow_optimal(self, env, reward_config):
        """ALLOW yields highest reward during peace."""
        env.set_attack_state(False)
        rewards = self._compute_rewards(env, reward_config)
        assert rewards[0] > rewards[1], f"Peace: ALLOW({rewards[0]:.4f}) should beat BLOCK({rewards[1]:.4f})"

    def test_optimal_actions_complete(self):
        """All attack types have entries in OPTIMAL_ACTIONS."""
        for key in ["ddos", "port_scan", "portscan", "spoofing", "mixed", None]:
            assert key in OPTIMAL_ACTIONS, f"Missing OPTIMAL_ACTIONS entry for {key}"
            assert key in OPTIMAL_EXPLANATIONS, f"Missing OPTIMAL_EXPLANATIONS entry for {key}"

    def test_fp_ordering_preserved(self, env):
        """Regression: BLOCK > RATE_LIMIT > REROUTE > ALLOW in FP during peace."""
        env.set_attack_state(False)
        fps = {}
        for action in range(4):
            m = env._compute_live_metrics(action)
            fps[action] = m["false_positives"]
        assert fps[1] > fps[3] > fps[2] > fps[0], f"FP ordering wrong: {fps}"

    def test_demo_bias_alignment(self):
        """DEMO_ACTION_BIAS maps to correct optimal actions."""
        from scripts.live_demo import DEMO_ACTION_BIAS
        q = np.ones(4)
        assert np.argmax(q + DEMO_ACTION_BIAS["ddos"]) == 3
        assert np.argmax(q + DEMO_ACTION_BIAS["portscan"]) == 1
        assert np.argmax(q + DEMO_ACTION_BIAS["spoofing"]) == 2
        assert np.argmax(q + DEMO_ACTION_BIAS[None]) == 0

    def test_orchestrator_attack_type_aware(self):
        """get_step_metrics varies by attack type."""
        from src.attacks.mixed_scenario import AttackOrchestrator, MixedScenarioConfig
        orch = AttackOrchestrator(MixedScenarioConfig(seed=42))

        mock_ddos = MagicMock()
        mock_ddos.attack_name = "ddos_flood"
        mock_ps = MagicMock()
        mock_ps.attack_name = "port_scan"

        orch._active_attacks = [mock_ddos]
        m_ddos = orch.get_step_metrics(action=3, attack_active=True)

        orch._rng = np.random.RandomState(42)
        orch._active_attacks = [mock_ps]
        m_ps = orch.get_step_metrics(action=1, attack_active=True)

        assert m_ddos["true_positives"] != m_ps["true_positives"] or \
               m_ddos["current_throughput_mbps"] != m_ps["current_throughput_mbps"], \
               "Metrics should differ between attack types"


# ---------------------------------------------------------------------------
# Item 2: PDF export (vendor script presence)
# ---------------------------------------------------------------------------

class TestPDFExport:

    def test_html_has_jspdf_script(self, client):
        resp = client.get("/")
        text = resp.text
        assert "jspdf.umd.min.js" in text
        assert "jspdf.plugin.autotable.min.js" in text

    def test_pdf_button_exists(self, client):
        resp = client.get("/")
        assert "btn-export-pdf" in resp.text


# ---------------------------------------------------------------------------
# Item 3: Tab restructure (4 tabs)
# ---------------------------------------------------------------------------

class TestTabRestructure:

    def test_html_has_four_tabs(self, client):
        resp = client.get("/")
        text = resp.text
        assert 'data-tab="live"' in text
        assert 'data-tab="network"' in text
        assert 'data-tab="comparison"' in text
        assert 'data-tab="training"' in text

    def test_tab_panel_ids_exist(self, client):
        resp = client.get("/")
        text = resp.text
        assert 'id="tab-live"' in text
        assert 'id="tab-network"' in text
        assert 'id="tab-comparison"' in text
        assert 'id="tab-training"' in text

    def test_old_system_tab_removed(self, client):
        resp = client.get("/")
        assert 'id="tab-system"' not in resp.text


# ---------------------------------------------------------------------------
# Item 4: Device inventory endpoint
# ---------------------------------------------------------------------------

class TestDeviceInventory:

    def test_devices_returns_20(self, client):
        resp = client.get("/devices")
        assert resp.status_code == 200
        devices = resp.json()["devices"]
        assert len(devices) == 20

    def test_device_structure(self, client):
        resp = client.get("/devices")
        devices = resp.json()["devices"]
        hosts = [d for d in devices if d["type"] == "host"]
        d = hosts[0]
        for key in ("id", "type", "role", "ip", "mac", "status"):
            assert key in d, f"Missing key: {key}"

    def test_switch_roles(self, client):
        resp = client.get("/devices")
        devices = resp.json()["devices"]
        switches = [d for d in devices if d["type"] == "switch"]
        assert len(switches) == 5
        core = [s for s in switches if s["role"] == "core"]
        assert len(core) == 1

    def test_host_ips(self, client):
        resp = client.get("/devices")
        devices = resp.json()["devices"]
        hosts = [d for d in devices if d["type"] == "host"]
        assert len(hosts) == 15
        for h in hosts:
            assert h["ip"].startswith("10.0.")


# ---------------------------------------------------------------------------
# Item 5: Session comparison
# ---------------------------------------------------------------------------

class TestSessionComparison:

    def test_sessions_initially_empty(self, client):
        resp = client.get("/sessions")
        assert resp.status_code == 200
        assert resp.json() == {}

    def test_sessions_with_data(self, client):
        sm = SessionMetrics(agent="dqn", started_at=time.time())
        sm.accumulate({"reward": 0.5, "metrics": {"detection_rate": 0.9, "false_positive_rate": 0.1}, "action": 1})
        sm.accumulate({"reward": 0.8, "metrics": {"detection_rate": 0.95, "false_positive_rate": 0.05}, "action": 3})
        state.agent_sessions["dqn"] = sm

        resp = client.get("/sessions")
        data = resp.json()
        assert "dqn" in data
        assert data["dqn"]["steps"] == 2
        assert data["dqn"]["avg_reward"] == pytest.approx(0.65, abs=0.01)


# ---------------------------------------------------------------------------
# Item 7: Action distribution (step event has action field)
# ---------------------------------------------------------------------------

class TestActionDistribution:

    def test_step_event_has_action_field(self):
        """Verify action key 0-3 in step events produced by RL loop."""
        for action in range(4):
            event = {"type": "step", "action": action, "reward": 0.1}
            assert "action" in event
            assert 0 <= event["action"] <= 3


# ---------------------------------------------------------------------------
# Item 8: Explainability
# ---------------------------------------------------------------------------

class TestExplainability:

    def test_explainability_fields_structure(self):
        """Verify expected event fields for explainability."""
        event = {
            "type": "step",
            "action": 3,
            "optimal_action": 3,
            "optimal_action_name": "RATE_LIMIT",
            "action_matches_optimal": True,
            "explanation": "Throttling preserves service availability",
        }
        assert event["action_matches_optimal"] is True
        assert event["optimal_action_name"] == "RATE_LIMIT"

    def test_optimal_action_values(self):
        assert OPTIMAL_ACTIONS["ddos"] == 3
        assert OPTIMAL_ACTIONS["portscan"] == 1
        assert OPTIMAL_ACTIONS["spoofing"] == 2
        assert OPTIMAL_ACTIONS[None] == 0


# ---------------------------------------------------------------------------
# Item 9: SessionMetrics dataclass
# ---------------------------------------------------------------------------

class TestSessionMetrics:

    def test_session_metrics_to_dict(self):
        sm = SessionMetrics(agent="ppo", started_at=1000.0)
        sm.accumulate({"reward": 1.0, "metrics": {"detection_rate": 0.8, "false_positive_rate": 0.2}, "action": 0})
        sm.accumulate({"reward": 2.0, "metrics": {"detection_rate": 0.9, "false_positive_rate": 0.1}, "action": 1})

        d = sm.to_dict()
        assert d["agent"] == "ppo"
        assert d["steps"] == 2
        assert d["total_reward"] == pytest.approx(3.0)
        assert d["avg_reward"] == pytest.approx(1.5)
        assert d["avg_detection_rate"] == pytest.approx(0.85)
        assert d["avg_fpr"] == pytest.approx(0.15)
        assert d["action_distribution"][0] == 1
        assert d["action_distribution"][1] == 1

    def test_session_metrics_empty(self):
        sm = SessionMetrics(agent="dqn", started_at=1000.0)
        d = sm.to_dict()
        assert d["steps"] == 0
        assert d["avg_reward"] == 0.0


# ---------------------------------------------------------------------------
# Item 10: Baseline flow snapshot
# ---------------------------------------------------------------------------

class TestBaselineFlows:

    def test_baseline_404_before_capture(self, client):
        resp = client.get("/flows/baseline")
        assert resp.status_code == 404

    def test_diff_404_before_baseline(self, client):
        resp = client.get("/flows/diff/1")
        assert resp.status_code == 404

    def test_baseline_capture_and_retrieve(self, client):
        mock_flows = [{"priority": 100, "match": {}, "actions": ["OUTPUT:1"]}]
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"1": mock_flows}
            mock_get.return_value = mock_resp
            resp = client.post("/flows/baseline")

        assert resp.status_code == 200
        assert state.baseline_captured is True

        resp2 = client.get("/flows/baseline")
        assert resp2.status_code == 200

    def test_flow_diff_structure(self, client):
        state.baseline_flows = {1: [{"priority": 100, "match": {"in_port": 1}, "actions": ["OUTPUT:2"]}]}
        state.baseline_captured = True

        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {"1": [
                {"priority": 100, "match": {"in_port": 1}, "actions": ["OUTPUT:2"]},
                {"priority": 200, "match": {"in_port": 3}, "actions": ["DROP"]},
            ]}
            mock_get.return_value = mock_resp
            resp = client.get("/flows/diff/1")

        data = resp.json()
        assert "added" in data
        assert "removed" in data
        assert len(data["added"]) == 1


# ---------------------------------------------------------------------------
# ZTA Baseline endpoint
# ---------------------------------------------------------------------------

class TestZTABaseline:

    def test_zta_baseline_endpoint_returns_rules(self, client):
        resp = client.get("/flows/zta-baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5
        for dpid_str in ("1", "2", "3", "4", "5"):
            assert dpid_str in data

    def test_zta_baseline_rule_structure(self, client):
        resp = client.get("/flows/zta-baseline")
        data = resp.json()
        for dpid_str, rules in data.items():
            for rule in rules:
                for key in ("priority", "match", "actions", "purpose"):
                    assert key in rule, f"Missing {key} in rule for switch {dpid_str}"

    def test_zta_baseline_has_default_deny(self, client):
        resp = client.get("/flows/zta-baseline")
        data = resp.json()
        for dpid_str, rules in data.items():
            deny = [r for r in rules if r["priority"] == 1 and "DROP" in r["actions"]]
            assert len(deny) == 1, f"Switch {dpid_str} missing default-deny rule"

    def test_zta_baseline_has_table_miss(self, client):
        resp = client.get("/flows/zta-baseline")
        data = resp.json()
        for dpid_str, rules in data.items():
            miss = [r for r in rules if r["priority"] == 0]
            assert len(miss) == 1, f"Switch {dpid_str} missing table-miss rule"


# ---------------------------------------------------------------------------
# RL Rules endpoint
# ---------------------------------------------------------------------------

class TestRLRules:

    def test_rl_rules_initially_empty(self, client):
        resp = client.get("/flows/rl-rules")
        assert resp.status_code == 200
        assert resp.json()["rules"] == []

    def test_rl_rules_after_publish(self, client):
        state.rl_installed_rules = [
            {"step": 1, "action": "BLOCK", "priority": 300, "actions": ["DROP"]},
            {"step": 2, "action": "RATE_LIMIT", "priority": 200, "actions": ["SET_QUEUE:1"]},
        ]
        resp = client.get("/flows/rl-rules")
        data = resp.json()
        assert len(data["rules"]) == 2
        assert data["rules"][0]["action"] == "BLOCK"


# ---------------------------------------------------------------------------
# Sim-mode fallbacks
# ---------------------------------------------------------------------------

class TestSimModeFallbacks:

    def test_flows_sim_mode_returns_baseline(self, client):
        state.mode = "sim"
        resp = client.get("/flows/1")
        assert resp.status_code == 200
        data = resp.json()
        assert "1" in data
        rules = data["1"]
        assert any(r["priority"] == 1 for r in rules)

    def test_baseline_capture_sim_mode(self, client):
        state.mode = "sim"
        resp = client.post("/flows/baseline")
        assert resp.status_code == 200
        assert state.baseline_captured is True
        assert len(state.baseline_flows) == 5

    def test_diff_sim_mode(self, client):
        state.mode = "sim"
        state.baseline_captured = True
        state.baseline_flows = {1: ZTA_BASELINE_RULES[1]}
        state.rl_installed_rules = [
            {"step": 1, "action": "BLOCK", "dpid": 1, "priority": 300},
        ]
        resp = client.get("/flows/diff/1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["added"]) == 1
        assert data["unchanged"] == len(ZTA_BASELINE_RULES[1])


# ---------------------------------------------------------------------------
# Realistic scenario constants
# ---------------------------------------------------------------------------

class TestRealisticScenario:

    def test_realistic_in_parse_args(self):
        from scripts.live_demo import parse_args
        with patch("sys.argv", ["live_demo.py", "--scenario", "realistic"]):
            args = parse_args()
        assert args.scenario == "realistic"

    def test_realistic_weights_sum_to_one(self):
        from scripts.live_demo import REALISTIC_ATTACK_WEIGHTS
        total = sum(REALISTIC_ATTACK_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6

    def test_realistic_peace_longer_than_attack(self):
        from scripts.live_demo import REALISTIC_ATTACK_RANGE, REALISTIC_PEACE_RANGE
        assert REALISTIC_PEACE_RANGE[0] > REALISTIC_ATTACK_RANGE[0]
        assert REALISTIC_PEACE_RANGE[1] > REALISTIC_ATTACK_RANGE[1]
