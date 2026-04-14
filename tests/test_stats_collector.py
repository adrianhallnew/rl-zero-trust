"""StatsCollector tests: normalization, caching, reset, and timeout handling.

Covers log1p scaling, /100 normalization, /300 duration capping,
connection rate calculation, state vector shape/dtype, double-fetch
elimination, connection rate reset, and timeout fallback to cache.
"""

import time
from unittest.mock import patch

import numpy as np
import pytest
import requests as requests_lib

from src.sdn.stats_collector import StatsCollector


@pytest.fixture
def collector():
    return StatsCollector("http://fake:8080")


def _make_mock_get(flow_count=10, packet_count=100, byte_count=1000, duration=5,
                   rx_packets=50, tx_packets=30):
    """Create a mock _get function returning consistent stats."""
    def mock_get(endpoint):
        if endpoint == "/stats/switches":
            return [1, 2, 3, 4, 5]
        if endpoint.startswith("/stats/flow/"):
            dpid = endpoint.split("/")[-1]
            return {dpid: [
                {"packet_count": packet_count, "byte_count": byte_count, "duration_sec": duration}
            ] * flow_count}
        if endpoint.startswith("/stats/port/"):
            dpid = endpoint.split("/")[-1]
            return {dpid: [{
                "rx_packets": rx_packets, "tx_packets": tx_packets,
                "rx_bytes": 5000, "tx_bytes": 3000,
                "rx_dropped": 2, "tx_dropped": 1,
                "rx_errors": 0, "tx_errors": 0,
            }]}
        return {}
    return mock_get


class TestStateVectorShape:
    """Verify state vector has correct shape and dtype."""

    def test_shape_is_65(self, collector):
        collector._get = _make_mock_get()
        state = collector.get_state_vector()
        assert state.shape == (65,)

    def test_dtype_is_float32(self, collector):
        collector._get = _make_mock_get()
        state = collector.get_state_vector()
        assert state.dtype == np.float32


class TestNormalization:
    """Verify normalization logic for each feature type."""

    def test_flow_count_normalized_by_100(self, collector):
        collector._get = _make_mock_get(flow_count=50)
        state = collector.get_state_vector()
        # Switch 1, feature 0: total_flows / 100 = 50/100 = 0.5
        assert abs(state[0] - 0.5) < 0.01

    def test_flow_count_capped_at_1(self, collector):
        collector._get = _make_mock_get(flow_count=200)
        state = collector.get_state_vector()
        assert state[0] == 1.0

    def test_packet_count_log1p_scaled(self, collector):
        collector._get = _make_mock_get(flow_count=1, packet_count=100)
        state = collector.get_state_vector()
        # Feature 1: log1p(100) ≈ 4.615
        expected = np.log1p(100)
        assert abs(state[1] - expected) < 0.1

    def test_duration_normalized_by_300(self, collector):
        collector._get = _make_mock_get(flow_count=1, duration=150)
        state = collector.get_state_vector()
        # Feature 3: 150/300 = 0.5
        assert abs(state[3] - 0.5) < 0.01

    def test_duration_capped_at_1(self, collector):
        collector._get = _make_mock_get(flow_count=1, duration=600)
        state = collector.get_state_vector()
        assert state[3] == 1.0


class TestConnectionRate:
    """Verify connection rate uses elapsed time correctly."""

    def test_connection_rate_positive(self, collector):
        collector._get = _make_mock_get(flow_count=5)
        state = collector.get_state_vector()
        # Feature 12: total_flows / elapsed
        # _last_poll_time was 0, so elapsed = time.time() (large)
        # After collect_all_stats, poll_time is set, so rate = 5 / elapsed
        assert state[12] > 0


class TestDoubleFetchElimination:
    """get_state_vector() reuses cached data — no redundant API calls."""

    def test_no_redundant_api_calls(self, collector):
        call_count = {"get": 0}

        def mock_get(endpoint):
            call_count["get"] += 1
            if endpoint == "/stats/switches":
                return [1, 2, 3, 4, 5]
            if endpoint.startswith("/stats/flow/"):
                return {endpoint.split("/")[-1]: [
                    {"packet_count": 10, "byte_count": 100, "duration_sec": 5}
                ]}
            if endpoint.startswith("/stats/port/"):
                return {endpoint.split("/")[-1]: [{
                    "rx_packets": 1, "tx_packets": 1, "rx_bytes": 0, "tx_bytes": 0,
                    "rx_dropped": 0, "tx_dropped": 0, "rx_errors": 0, "tx_errors": 0,
                }]}
            return {}

        collector._get = mock_get
        collector.get_state_vector()
        # 1 switch list + 5 flow + 5 port = 11 calls total (not 21)
        assert call_count["get"] == 11


class TestConnectionRateReset:
    """StateProcessor.reset() clears connection rate accumulation."""

    def test_reset_clears_collector_state(self):
        from src.environment.state_processor import StateProcessor

        collector = StatsCollector("http://fake:8080")
        collector._last_poll_time = time.time() - 1000
        collector._cumulative_flows = 999

        processor = StateProcessor(stats_collector=collector)
        processor.reset()

        assert collector._last_poll_time == 0.0
        assert collector._cumulative_flows == 0


class TestTimeoutCachedData:
    """Ryu API timeout returns cached data instead of raising."""

    def test_timeout_returns_cached(self, collector):
        collector._last_stats = {1: {"flows": [{"packet_count": 42}], "ports": []}}

        def mock_get(url, timeout=None):
            raise requests_lib.Timeout("timed out")

        with patch("src.sdn.stats_collector.requests.get", side_effect=mock_get):
            result = collector._get("/stats/switches")

        assert result == collector._last_stats
