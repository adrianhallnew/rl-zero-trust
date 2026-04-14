"""PolicyEnforcer tests: REST API calls, action log cap, error handling.

Covers flow rule installation, deletion, reroute port selection,
error handling, policy_change_count, and action log bounding.
"""

from unittest.mock import patch, MagicMock

import pytest
import requests

from src.sdn.policy_enforcer import (
    PolicyEnforcer,
    ACTION_ALLOW, ACTION_BLOCK, ACTION_REROUTE, ACTION_RATE_LIMIT,
)


@pytest.fixture
def enforcer():
    return PolicyEnforcer(ryu_api_url="http://fake:8080")


class TestEnforceAction:
    """Test enforce_action for each action type."""

    def test_block_posts_correct_body(self, enforcer):
        posted = []

        def mock_post(url, json=None, timeout=None):
            posted.append({"url": url, "json": json})
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        with patch("src.sdn.policy_enforcer.requests.post", side_effect=mock_post):
            result = enforcer.enforce_action(
                ACTION_BLOCK, dpid=2, match_fields={"dl_type": 2048, "nw_src": "10.0.1.1"},
            )

        assert result is True
        assert len(posted) == 1
        body = posted[0]["json"]
        assert body["dpid"] == 2
        assert body["actions"] == []  # DROP
        assert body["match"]["dl_type"] == 2048

    def test_allow_sends_delete(self, enforcer):
        posted = []

        def mock_post(url, json=None, timeout=None):
            posted.append({"url": url, "json": json})
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        with patch("src.sdn.policy_enforcer.requests.post", side_effect=mock_post):
            enforcer.enforce_action(
                ACTION_ALLOW, dpid=1, match_fields={"dl_type": 2048},
            )

        assert len(posted) == 1
        assert "/delete" in posted[0]["url"]

    def test_reroute_has_output_action(self, enforcer):
        posted = []

        def mock_post(url, json=None, timeout=None):
            posted.append(body := json)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        with patch("src.sdn.policy_enforcer.requests.post", side_effect=mock_post):
            enforcer.enforce_action(
                ACTION_REROUTE, dpid=1, match_fields={"dl_type": 2048},
            )

        assert any(a["type"] == "OUTPUT" for a in posted[0]["actions"])

    def test_rate_limit_has_set_queue(self, enforcer):
        posted = []

        def mock_post(url, json=None, timeout=None):
            posted.append(json)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        with patch("src.sdn.policy_enforcer.requests.post", side_effect=mock_post):
            enforcer.enforce_action(
                ACTION_RATE_LIMIT, dpid=3, match_fields={"dl_type": 2048},
            )

        actions = posted[0]["actions"]
        assert any(a.get("type") == "SET_QUEUE" for a in actions)

    def test_empty_match_gets_defaults(self, enforcer):
        posted = []

        def mock_post(url, json=None, timeout=None):
            posted.append(json)
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        with patch("src.sdn.policy_enforcer.requests.post", side_effect=mock_post):
            enforcer.enforce_action(ACTION_BLOCK, dpid=2, match_fields={})

        match = posted[0]["match"]
        assert "dl_type" in match
        assert match["dl_type"] == 2048


class TestPickReroutePort:
    """Test reroute port selection logic."""

    def test_core_switch_returns_port_2(self):
        assert PolicyEnforcer._pick_reroute_port(1) == 2

    def test_edge_switch_returns_port_1(self):
        for dpid in [2, 3, 4, 5]:
            assert PolicyEnforcer._pick_reroute_port(dpid) == 1


class TestErrorHandling:
    """Test behavior when Ryu API is unreachable."""

    def test_request_exception_returns_false(self, enforcer):
        with patch("src.sdn.policy_enforcer.requests.post", side_effect=requests.RequestException("fail")):
            result = enforcer.enforce_action(
                ACTION_BLOCK, dpid=1, match_fields={"dl_type": 2048},
            )
        assert result is False

    def test_policy_count_no_increment_on_failure(self, enforcer):
        with patch("src.sdn.policy_enforcer.requests.post", side_effect=requests.RequestException("fail")):
            enforcer.enforce_action(ACTION_BLOCK, dpid=1, match_fields={"dl_type": 2048})
        assert enforcer.policy_change_count == 0

    def test_policy_count_increments_on_success(self, enforcer):
        def mock_post(url, json=None, timeout=None):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            return resp

        with patch("src.sdn.policy_enforcer.requests.post", side_effect=mock_post):
            enforcer.enforce_action(ACTION_BLOCK, dpid=1, match_fields={"dl_type": 2048})
            enforcer.enforce_action(ACTION_BLOCK, dpid=1, match_fields={"dl_type": 2048})
        assert enforcer.policy_change_count == 2


class TestActionLogCap:
    """_action_log stays bounded at 500 entries."""

    def test_action_log_capped_at_500(self, enforcer):
        enforcer._add_flow = lambda entry: True
        enforcer._delete_flow = lambda dpid, match: True

        for i in range(600):
            enforcer.enforce_action(
                action=ACTION_BLOCK, dpid=1,
                match_fields={"dl_type": 2048, "nw_src": f"10.0.0.{i % 256}"},
            )

        assert len(enforcer._action_log) == 500
