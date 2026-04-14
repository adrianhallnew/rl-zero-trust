"""Tests for the dashboard FastAPI server.

Covers:
    - GET /status fields and state reflection
    - POST /attack valid/invalid types, intensity clamping
    - POST /mode valid/invalid
    - POST /scenario/auto
    - GET /export event collection
    - Memory cap (1000 events max)
    - episode_done event schema
    - SSE subscribe/unsubscribe lifecycle
    - Chart listing returns SVG files
    - Chart path traversal blocked
    - Concurrent publish_sync thread safety
"""

import asyncio
import threading
import time

import pytest
from fastapi.testclient import TestClient

from src.dashboard.server import DashboardState, app, state


@pytest.fixture(autouse=True)
def reset_state():
    """Reset singleton state before each test."""
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
    yield


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

class TestStatusEndpoint:
    """GET /status returns correct fields."""

    def test_status_returns_all_fields(self, client):
        state.agent = "ppo"
        state.mode = "sim"
        state.running = True
        state.step = 42
        state.start_time = time.time() - 10.0

        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent"] == "ppo"
        assert data["mode"] == "sim"
        assert data["running"] is True
        assert data["step"] == 42
        assert isinstance(data["elapsed_seconds"], float)
        assert data["elapsed_seconds"] >= 10.0
        assert "last_event" in data

    def test_status_elapsed_none_when_not_started(self, client):
        resp = client.get("/status")
        data = resp.json()
        assert data["elapsed_seconds"] is None


# ---------------------------------------------------------------------------
# POST /attack
# ---------------------------------------------------------------------------

class TestAttackEndpoints:
    """POST /attack/{type} validation and state effects."""

    def test_valid_attack_types(self, client):
        for atype in ("ddos", "portscan", "spoofing", "mixed"):
            resp = client.post(f"/attack/{atype}")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"

    def test_invalid_attack_returns_400(self, client):
        resp = client.post("/attack/unknown")
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_stop_attack_sets_flag(self, client):
        resp = client.post("/attack/stop")
        assert resp.status_code == 200
        assert state.requested_stop_attacks is True

    def test_intensity_clamped_to_range(self, client):
        resp = client.post("/attack/ddos?intensity=5.0")
        assert resp.status_code == 200
        assert state.requested_attack_intensity <= 1.0

        resp = client.post("/attack/ddos?intensity=0.01")
        assert resp.status_code == 200
        assert state.requested_attack_intensity >= 0.1

    def test_attack_sets_requested_type(self, client):
        client.post("/attack/portscan")
        assert state.requested_attack == "portscan"


# ---------------------------------------------------------------------------
# POST /mode
# ---------------------------------------------------------------------------

class TestModeEndpoints:
    """POST /mode/{mode} validation."""

    def test_valid_modes(self, client):
        for mode in ("live", "sim"):
            resp = client.post(f"/mode/{mode}")
            assert resp.status_code == 200
            assert state.requested_mode == mode

    def test_invalid_mode_returns_400(self, client):
        resp = client.post("/mode/invalid")
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# POST /scenario/auto
# ---------------------------------------------------------------------------

class TestAutoScenario:
    """POST /scenario/auto sets flag."""

    def test_auto_scenario_sets_flag(self, client):
        resp = client.post("/scenario/auto")
        assert resp.status_code == 200
        assert state.requested_auto_scenario is True


# ---------------------------------------------------------------------------
# GET /export
# ---------------------------------------------------------------------------

class TestExport:
    """GET /export returns session data."""

    def test_export_contains_events(self, client):
        for i in range(3):
            state.publish_sync({
                "type": "step",
                "step": i + 1,
                "timestamp": time.time(),
            })
        resp = client.get("/export")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["events"]) == 3
        assert data["events"][0]["step"] == 1
        assert data["events"][2]["step"] == 3

    def test_export_includes_metadata(self, client):
        state.agent = "ppo"
        state.mode = "sim"
        state.step = 50
        resp = client.get("/export")
        data = resp.json()
        assert data["agent"] == "ppo"
        assert data["mode"] == "sim"
        assert data["total_steps"] == 50
        assert "exported_at" in data


# ---------------------------------------------------------------------------
# Memory cap
# ---------------------------------------------------------------------------

class TestMemoryCap:
    """Events list stays bounded at 1000."""

    def test_cap_at_1000(self):
        for i in range(1100):
            state.publish_sync({
                "type": "step",
                "step": i + 1,
                "timestamp": time.time(),
            })
        assert len(state.events) == 1000
        assert state.events[0]["step"] == 101
        assert state.events[-1]["step"] == 1100

    def test_async_publish_also_capped(self):
        loop = asyncio.new_event_loop()
        try:
            for i in range(1100):
                loop.run_until_complete(state.publish({
                    "type": "step",
                    "step": i + 1,
                    "timestamp": time.time(),
                }))
            assert len(state.events) == 1000
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# episode_done schema
# ---------------------------------------------------------------------------

class TestEpisodeDoneSchema:
    """episode_done event has required fields."""

    def test_schema_preserved(self):
        event = {
            "type": "episode_done",
            "timestamp": time.time(),
            "steps": 200,
            "cumulative_reward": 45.5,
        }
        state.publish_sync(event)
        stored = state.events[0]
        assert stored["type"] == "episode_done"
        assert isinstance(stored["steps"], int)
        assert stored["steps"] == 200
        assert isinstance(stored["cumulative_reward"], float)


# ---------------------------------------------------------------------------
# SSE subscribe/unsubscribe
# ---------------------------------------------------------------------------

class TestSubscription:
    """Subscribe and unsubscribe lifecycle."""

    def test_subscribe_creates_queue(self):
        q = state.subscribe()
        assert q in state.subscribers
        assert isinstance(q, asyncio.Queue)

    def test_unsubscribe_removes_queue(self):
        q = state.subscribe()
        state.unsubscribe(q)
        assert q not in state.subscribers

    def test_unsubscribe_nonexistent_is_safe(self):
        q = asyncio.Queue()
        state.unsubscribe(q)  # should not raise

    def test_multiple_subscribers(self):
        q1 = state.subscribe()
        q2 = state.subscribe()
        assert len(state.subscribers) == 2
        state.unsubscribe(q1)
        assert len(state.subscribers) == 1
        assert q2 in state.subscribers


# ---------------------------------------------------------------------------
# Concurrent publish_sync thread safety
# ---------------------------------------------------------------------------

class TestConcurrentPublish:
    """publish_sync from multiple threads doesn't corrupt state."""

    def test_concurrent_publish_no_data_loss(self):
        errors = []
        n_per_thread = 100
        n_threads = 4

        def publish_batch(thread_id):
            try:
                for i in range(n_per_thread):
                    state.publish_sync({
                        "type": "step",
                        "thread": thread_id,
                        "step": i,
                        "timestamp": time.time(),
                    })
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=publish_batch, args=(t,))
            for t in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(state.events) == n_per_thread * n_threads


# ---------------------------------------------------------------------------
# Chart listing
# ---------------------------------------------------------------------------

class TestChartListing:
    """GET /charts returns SVG files."""

    def test_charts_endpoint_returns_list(self, client):
        resp = client.get("/charts")
        assert resp.status_code == 200
        data = resp.json()
        assert "charts" in data
        # All entries should be SVG
        for path in data["charts"]:
            assert path.endswith(".svg"), f"Expected .svg, got: {path}"


# ---------------------------------------------------------------------------
# Chart path traversal
# ---------------------------------------------------------------------------

class TestChartSecurity:
    """Chart endpoint blocks path traversal."""

    def test_path_traversal_blocked(self, client):
        resp = client.get("/charts/../../../etc/passwd")
        assert resp.status_code in (403, 404)
