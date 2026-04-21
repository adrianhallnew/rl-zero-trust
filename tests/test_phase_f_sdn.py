"""Phase F tests: SDN policy enforcement & metric correctness.

Tests cover:
    F3.1 — Reroute port cycling (2, 3, 4)
    F3.2 — Queue validation with fallback
    F3.3 — ALLOW installs permissive rule
    F5.2 — Attack-type-aware confusion matrix
    F5.3 — Fewer than 5 switches → sentinel values
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.sdn.policy_enforcer import (
    ACTION_ALLOW,
    ACTION_BLOCK,
    ACTION_RATE_LIMIT,
    ACTION_REROUTE,
    PRIORITY_SECURITY,
    PolicyEnforcer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def enforcer():
    return PolicyEnforcer(ryu_api_url="http://localhost:8080")


# ---------------------------------------------------------------------------
# F3.1 — Reroute port cycling
# ---------------------------------------------------------------------------

class TestF3_1_ReroutePortCycling:
    """Core switch reroute should cycle through ports 2, 3, 4."""

    def test_core_switch_cycles_ports(self, enforcer):
        ports = [enforcer._pick_reroute_port(1) for _ in range(6)]
        assert ports == [2, 3, 4, 2, 3, 4]

    def test_edge_switch_always_port_1(self, enforcer):
        for dpid in (2, 3, 4, 5):
            assert enforcer._pick_reroute_port(dpid) == 1

    def test_reroute_index_persists(self, enforcer):
        enforcer._pick_reroute_port(1)  # index -> 1
        enforcer._pick_reroute_port(2)  # edge, no index change
        port = enforcer._pick_reroute_port(1)  # index 1 -> port 3
        assert port == 3


# ---------------------------------------------------------------------------
# F3.2 — Queue validation
# ---------------------------------------------------------------------------

class TestF3_2_QueueValidation:
    """RATE_LIMIT should validate queue existence."""

    def test_validate_queue_caches_result(self, enforcer):
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = [{"queue_id": 1}]
            mock_get.return_value = mock_resp

            result1 = enforcer._validate_queue(1)
            result2 = enforcer._validate_queue(1)

        assert result1 is True
        assert result2 is True
        # Should only call API once (cached)
        assert mock_get.call_count == 1

    def test_validate_queue_missing(self, enforcer):
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = []
            mock_get.return_value = mock_resp

            result = enforcer._validate_queue(1)

        assert result is False

    def test_rate_limit_falls_back_to_reroute_on_missing_queue(self, enforcer):
        enforcer._queue_validated[1] = False  # Pre-cache as missing

        with patch.object(enforcer, "_enforce_reroute", return_value=True) as mock_rr:
            result = enforcer._enforce_rate_limit(1, {"dl_type": 2048})

        mock_rr.assert_called_once()
        assert result is True

    def test_rate_limit_uses_queue_when_valid(self, enforcer):
        enforcer._queue_validated[1] = True  # Pre-cache as valid

        with patch.object(enforcer, "_add_flow", return_value=True) as mock_add:
            result = enforcer._enforce_rate_limit(1, {"dl_type": 2048})

        assert result is True
        call_args = mock_add.call_args[0][0]
        actions = call_args["actions"]
        assert any(a["type"] == "SET_QUEUE" for a in actions)


# ---------------------------------------------------------------------------
# F3.3 — ALLOW installs permissive rule
# ---------------------------------------------------------------------------

class TestF3_3_AllowPermissiveRule:
    """ALLOW should delete + install NORMAL output rule."""

    def test_allow_calls_delete_then_add(self, enforcer):
        with patch.object(enforcer, "_delete_flow", return_value=True) as mock_del, \
             patch.object(enforcer, "_add_flow", return_value=True) as mock_add:
            result = enforcer._enforce_allow(1, {"dl_type": 2048})

        mock_del.assert_called_once()
        mock_add.assert_called_once()
        assert result is True

    def test_allow_installs_normal_output(self, enforcer):
        with patch.object(enforcer, "_delete_flow", return_value=True), \
             patch.object(enforcer, "_add_flow", return_value=True) as mock_add:
            enforcer._enforce_allow(1, {"dl_type": 2048})

        flow = mock_add.call_args[0][0]
        assert flow["actions"] == [{"type": "OUTPUT", "port": "NORMAL"}]
        assert flow["priority"] == PRIORITY_SECURITY - 10


# ---------------------------------------------------------------------------
# F5.2 — Attack-type-aware confusion matrix
# ---------------------------------------------------------------------------

class TestF5_2_AttackTypeMetrics:
    """Confusion matrix should vary by attack type."""

    @pytest.fixture
    def orchestrator(self):
        from src.attacks.mixed_scenario import (
            AttackOrchestrator,
            MixedScenarioConfig,
            ScenarioType,
        )
        config = MixedScenarioConfig(scenario_type=ScenarioType.RANDOM)
        return AttackOrchestrator(config=config)

    def test_ddos_higher_tp_than_spoofing(self, orchestrator):
        """DDoS should be easier to detect (higher TP) than spoofing."""
        # Simulate DDoS active
        ddos_attack = MagicMock()
        ddos_attack.attack_name = "syn_flood"
        ddos_attack.is_active.return_value = True
        orchestrator._active_attacks = [ddos_attack]

        ddos_tps = []
        for _ in range(50):
            m = orchestrator.get_step_metrics(1, True)
            ddos_tps.append(m["true_positives"])

        # Simulate spoofing active
        spoof_attack = MagicMock()
        spoof_attack.attack_name = "spoofing"
        spoof_attack.is_active.return_value = True
        orchestrator._active_attacks = [spoof_attack]

        spoof_tps = []
        for _ in range(50):
            m = orchestrator.get_step_metrics(1, True)
            spoof_tps.append(m["true_positives"])

        assert np.mean(ddos_tps) > np.mean(spoof_tps), (
            f"DDoS mean TP ({np.mean(ddos_tps):.1f}) should exceed "
            f"spoofing ({np.mean(spoof_tps):.1f})"
        )

    def test_block_vs_rate_limit_metrics_differ(self, orchestrator):
        """BLOCK and RATE_LIMIT should produce different metrics."""
        attack = MagicMock()
        attack.attack_name = "syn_flood"
        attack.is_active.return_value = True
        orchestrator._active_attacks = [attack]

        m_block = orchestrator.get_step_metrics(1, True)
        m_rate = orchestrator.get_step_metrics(3, True)

        # Throughput should differ (RATE_LIMIT gentler)
        # Run multiple times due to randomness
        block_throughputs = []
        rate_throughputs = []
        for _ in range(50):
            mb = orchestrator.get_step_metrics(1, True)
            mr = orchestrator.get_step_metrics(3, True)
            block_throughputs.append(mb["current_throughput_mbps"])
            rate_throughputs.append(mr["current_throughput_mbps"])

        assert np.mean(rate_throughputs) > np.mean(block_throughputs), (
            "RATE_LIMIT should have better throughput than BLOCK"
        )


# ---------------------------------------------------------------------------
# F5.3 — Fewer than 5 switches → sentinel values
# ---------------------------------------------------------------------------

class TestF5_3_SwitchSentinels:
    """Missing switches should produce -1.0 sentinel values."""

    def test_fewer_switches_fills_sentinels(self):
        from src.sdn.stats_collector import StatsCollector
        collector = StatsCollector(ryu_api_url="http://localhost:8080")

        # Mock collect_all_stats to return only 3 switches
        three_switch_stats = {
            1: {"flows": [], "ports": []},
            2: {"flows": [], "ports": []},
            3: {"flows": [], "ports": []},
        }

        with patch.object(collector, "collect_all_stats", return_value=three_switch_stats), \
             patch.object(collector, "get_aggregate_flow_stats", return_value={
                 "total_flows": 0, "total_packets": 0, "total_bytes": 0, "avg_duration": 0,
             }), \
             patch.object(collector, "get_aggregate_port_stats", return_value={
                 "total_rx_packets": 0, "total_tx_packets": 0,
                 "total_rx_bytes": 0, "total_tx_bytes": 0,
                 "total_rx_dropped": 0, "total_tx_dropped": 0,
                 "total_rx_errors": 0, "total_tx_errors": 0,
             }):
            state = collector.get_state_vector()

        assert state.shape == (65,)
        assert state.dtype == np.float32
        # Switches 4-5 (indices 39-64) should be -1.0 sentinel
        assert np.all(state[39:65] == -1.0), (
            f"Missing switch features should be -1.0, got {state[39:65]}"
        )

    def test_full_switches_no_sentinels(self):
        from src.sdn.stats_collector import StatsCollector
        collector = StatsCollector(ryu_api_url="http://localhost:8080")

        five_switch_stats = {i: {"flows": [], "ports": []} for i in range(1, 6)}

        with patch.object(collector, "collect_all_stats", return_value=five_switch_stats), \
             patch.object(collector, "get_aggregate_flow_stats", return_value={
                 "total_flows": 10, "total_packets": 100, "total_bytes": 5000, "avg_duration": 5,
             }), \
             patch.object(collector, "get_aggregate_port_stats", return_value={
                 "total_rx_packets": 50, "total_tx_packets": 50,
                 "total_rx_bytes": 2500, "total_tx_bytes": 2500,
                 "total_rx_dropped": 0, "total_tx_dropped": 0,
                 "total_rx_errors": 0, "total_tx_errors": 0,
             }):
            state = collector.get_state_vector()

        assert state.shape == (65,)
        # No sentinel values when all 5 switches present
        assert not np.any(state == -1.0)
