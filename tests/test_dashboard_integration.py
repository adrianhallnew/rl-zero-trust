"""D3: Integration tests for Dashboard + RL Loop.

Tests the FastAPI app with mock RL events covering:
    - Status endpoint reflects published state
    - Export returns all published events in order
    - Agent switch via POST /agent/{type}
    - Mode switch via POST /mode/{type}
    - Attack trigger and stop flow
    - Auto scenario flag
    - Control start/stop lifecycle
    - Config and summary endpoints
    - Charts listing
    - publish_sync updates last_event
"""

import time

import pytest
from starlette.testclient import TestClient


@pytest.fixture(autouse=True)
def reset_state():
    """Reset DashboardState for each test."""
    from src.dashboard.server import state
    state.events.clear()
    state.step = 0
    state.agent = "dqn"
    state.mode = "live"
    state.running = False
    state.start_time = None
    state.last_event = None
    state.requested_agent = None
    state.requested_mode = None
    state.requested_attack = None
    state.requested_attack_intensity = 0.7
    state.requested_stop_attacks = False
    state.requested_auto_scenario = False
    state.requested_start = False
    state.requested_stop = False
    yield
    state.events.clear()


@pytest.fixture
def client():
    from src.dashboard.server import app
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def test_status_endpoint_defaults(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent"] == "dqn"
    assert data["running"] is False


def test_status_reflects_state_changes(client):
    from src.dashboard.server import state
    state.agent = "ppo"
    state.running = True
    state.step = 99
    resp = client.get("/status")
    data = resp.json()
    assert data["agent"] == "ppo"
    assert data["step"] == 99
    assert data["running"] is True


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def test_export_returns_events_in_order(client):
    from src.dashboard.server import state
    for i in range(5):
        state.publish_sync({
            "type": "step",
            "step": i + 1,
            "timestamp": time.time(),
        })
    resp = client.get("/export")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["events"]) == 5
    steps = [e["step"] for e in data["events"]]
    assert steps == [1, 2, 3, 4, 5]


def test_export_empty_when_no_events(client):
    resp = client.get("/export")
    data = resp.json()
    assert data["events"] == []
    assert data["total_steps"] == 0


# ---------------------------------------------------------------------------
# Agent switch
# ---------------------------------------------------------------------------

def test_agent_switch_sets_requested(client):
    from src.dashboard.server import state
    resp = client.post("/agent/ppo")
    assert resp.status_code == 200
    assert state.requested_agent == "ppo"


def test_agent_switch_invalid_returns_400(client):
    resp = client.post("/agent/invalid")
    assert resp.status_code == 400


def test_agent_switch_publishes_toast(client):
    from src.dashboard.server import state
    client.post("/agent/dqn")
    toast_events = [e for e in state.events if e.get("type") == "toast"]
    assert len(toast_events) >= 1
    assert "DQN" in toast_events[0]["message"]


# ---------------------------------------------------------------------------
# Mode switch
# ---------------------------------------------------------------------------

def test_mode_switch_valid(client):
    from src.dashboard.server import state
    resp = client.post("/mode/sim")
    assert resp.status_code == 200
    assert state.requested_mode == "sim"


def test_mode_switch_invalid_returns_400(client):
    resp = client.post("/mode/invalid")
    assert resp.status_code == 400


def test_mode_switch_publishes_toast(client):
    from src.dashboard.server import state
    client.post("/mode/live")
    toast_events = [e for e in state.events if e.get("type") == "toast"]
    assert len(toast_events) >= 1
    assert "Live" in toast_events[0]["message"]


# ---------------------------------------------------------------------------
# Attack trigger and stop
# ---------------------------------------------------------------------------

def test_attack_trigger(client):
    from src.dashboard.server import state
    resp = client.post("/attack/ddos?intensity=0.8")
    assert resp.status_code == 200
    assert state.requested_attack == "ddos"
    assert abs(state.requested_attack_intensity - 0.8) < 0.01


def test_attack_stop(client):
    from src.dashboard.server import state
    resp = client.post("/attack/stop")
    assert resp.status_code == 200
    assert state.requested_stop_attacks is True


def test_attack_stop_publishes_toast(client):
    from src.dashboard.server import state
    client.post("/attack/stop")
    toast_events = [e for e in state.events if e.get("type") == "toast"]
    assert len(toast_events) >= 1
    assert "stop" in toast_events[0]["message"].lower()


# ---------------------------------------------------------------------------
# Auto scenario
# ---------------------------------------------------------------------------

def test_auto_scenario(client):
    from src.dashboard.server import state
    resp = client.post("/scenario/auto")
    assert resp.status_code == 200
    assert state.requested_auto_scenario is True


# ---------------------------------------------------------------------------
# Control start/stop
# ---------------------------------------------------------------------------

def test_control_start(client):
    from src.dashboard.server import state
    resp = client.post("/control/start")
    assert resp.status_code == 200
    assert state.requested_start is True


def test_control_stop(client):
    from src.dashboard.server import state
    resp = client.post("/control/stop")
    assert resp.status_code == 200
    assert state.requested_stop is True


def test_start_then_stop_lifecycle(client):
    from src.dashboard.server import state
    client.post("/control/start")
    assert state.requested_start is True
    client.post("/control/stop")
    assert state.requested_stop is True


# ---------------------------------------------------------------------------
# Config and summary
# ---------------------------------------------------------------------------

def test_config_endpoint(client):
    resp = client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    # Should load at least one config
    assert "dqn" in data or "ppo" in data


def test_summary_endpoint(client):
    resp = client.get("/summary")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Charts listing
# ---------------------------------------------------------------------------

def test_charts_list_endpoint(client):
    resp = client.get("/charts")
    assert resp.status_code == 200
    data = resp.json()
    assert "charts" in data


# ---------------------------------------------------------------------------
# publish_sync updates last_event
# ---------------------------------------------------------------------------

def test_publish_sync_updates_last_event():
    from src.dashboard.server import state
    event = {"type": "step", "step": 1, "timestamp": time.time()}
    state.publish_sync(event)
    assert state.last_event == event


def test_publish_sync_preserves_event_order():
    from src.dashboard.server import state
    for i in range(10):
        state.publish_sync({"type": "step", "step": i, "timestamp": time.time()})
    steps = [e["step"] for e in state.events]
    assert steps == list(range(10))
