"""Phase F tests: Dashboard security & robustness.

Tests cover:
    F4.1 — XSS prevention (server-side attack type validation)
    F4.2 — Path traversal prevention in chart serving
    F4.3 — SSE disconnect detection improvement
    F4.4 — Enriched session export
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.dashboard.server import app, state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset dashboard state between tests."""
    state.agent = "dqn"
    state.mode = "live"
    state.running = False
    state.step = 0
    state.start_time = time.time()
    state.events.clear()
    state.last_event = None
    state.requested_agent = None
    state.requested_mode = None
    state.requested_attack = None
    state.requested_stop_attacks = False
    yield


# ---------------------------------------------------------------------------
# F4.1 — XSS prevention (server-side validation)
# ---------------------------------------------------------------------------

class TestF4_1_XSSPrevention:
    """Attack type endpoint should reject invalid types."""

    def test_valid_attack_types_accepted(self, client):
        for atype in ("ddos", "portscan", "spoofing", "mixed", "stop"):
            resp = client.post(f"/attack/{atype}")
            assert resp.status_code == 200, f"Valid type {atype} rejected"

    def test_xss_payload_rejected(self, client):
        """XSS payloads with special chars may be rejected by router (404)
        or by validation (400). Either is acceptable — they must not return 200."""
        resp = client.post("/attack/<script>alert(1)</script>")
        assert resp.status_code in (400, 404)

    def test_arbitrary_string_rejected(self, client):
        resp = client.post("/attack/notarealattack")
        assert resp.status_code == 400
        assert "Unknown attack type" in resp.json()["error"]

    def test_empty_attack_type_rejected(self, client):
        resp = client.post("/attack/")
        assert resp.status_code in (400, 404, 405)


# ---------------------------------------------------------------------------
# F4.2 — Path traversal prevention
# ---------------------------------------------------------------------------

class TestF4_2_PathTraversal:
    """Chart serving should prevent directory traversal."""

    def test_traversal_blocked(self, client):
        resp = client.get("/charts/../../../etc/passwd")
        assert resp.status_code in (403, 404)

    def test_double_traversal_blocked(self, client):
        resp = client.get("/charts/..%2F..%2F..%2Fetc%2Fpasswd")
        assert resp.status_code in (403, 404)

    def test_nonexistent_chart_404(self, client):
        resp = client.get("/charts/nonexistent_chart.svg")
        assert resp.status_code == 404

    def test_valid_chart_path_format(self, client):
        """Valid path format should not trigger 403 (may 404 if file missing)."""
        resp = client.get("/charts/dqn_reward.svg")
        # Should be 404 (not found) not 403 (traversal blocked)
        assert resp.status_code in (200, 404)


# ---------------------------------------------------------------------------
# F4.4 — Enriched session export
# ---------------------------------------------------------------------------

class TestF4_4_EnrichedExport:
    """Export should include full session context."""

    def test_export_contains_all_fields(self, client):
        state.agent = "ppo"
        state.mode = "sim"
        state.running = True
        state.step = 42
        state.start_time = time.time() - 60

        resp = client.get("/export")
        data = resp.json()

        assert data["agent"] == "ppo"
        assert data["mode"] == "sim"
        assert data["running"] is True
        assert data["total_steps"] == 42
        assert data["start_time"] is not None
        assert data["elapsed_seconds"] is not None
        assert 55 < data["elapsed_seconds"] < 65  # ~60s
        assert "events" in data
        assert "exported_at" in data

    def test_export_with_events(self, client):
        state.events = [
            {"type": "step", "step": 1, "reward": 0.5},
            {"type": "step", "step": 2, "reward": -0.3},
        ]
        state.last_event = state.events[-1]

        resp = client.get("/export")
        data = resp.json()

        assert len(data["events"]) == 2
        assert data["last_event"]["step"] == 2

    def test_export_empty_session(self, client):
        state.start_time = None
        resp = client.get("/export")
        data = resp.json()

        assert data["elapsed_seconds"] is None
        assert data["events"] == []
