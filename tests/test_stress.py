"""D1: Stress tests for attack cycling and concurrent dashboard operations.

Covers:
    - Rapid attack cycling (fire/stop <1s transitions)
    - Concurrent publish from multiple threads
    - Subscribe/unsubscribe under publish load
    - Rapid agent switch requests
    - Memory cap under concurrent writes
"""

import asyncio
import threading
import time
from unittest.mock import patch, MagicMock

import pytest

from scripts.live_demo import AttackScheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_dashboard_state():
    """Reset dashboard state before each test."""
    from src.dashboard.server import state
    state.events.clear()
    state.subscribers.clear()
    state.last_event = None
    state.requested_agent = None
    yield
    state.events.clear()
    state.subscribers.clear()


# ---------------------------------------------------------------------------
# Rapid attack cycling
# ---------------------------------------------------------------------------

class TestRapidAttackCycling:
    """D1: Rapid attack cycling with <1s transitions."""

    def test_rapid_cycle_all_types_no_exceptions(self):
        scheduler = AttackScheduler(scenario="none")
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            for atype in ["ddos", "portscan", "spoofing", "mixed"] * 10:
                scheduler.fire_manual(atype, duration=2, intensity="low")
                assert scheduler.attack_active is True
                assert scheduler.attack_type == atype
                scheduler.stop_attacks()
                assert scheduler.attack_active is False

    def test_processes_stay_bounded(self):
        scheduler = AttackScheduler(scenario="none")
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0

        with patch.object(scheduler, "_docker_exec_attack", return_value=mock_proc):
            for _ in range(50):
                scheduler.fire_manual("ddos", duration=2, intensity="low")
                scheduler.stop_attacks()
        assert len(scheduler._processes) == 0

    def test_flag_consistency_after_50_cycles(self):
        scheduler = AttackScheduler(scenario="none")
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            for _ in range(50):
                scheduler.fire_manual("ddos", duration=1, intensity="low")
                scheduler.stop_attacks()
            assert scheduler.attack_active is False
            assert scheduler.attack_type is None
            assert scheduler._attack_end_time is None


# ---------------------------------------------------------------------------
# Concurrent publish thread safety
# ---------------------------------------------------------------------------

class TestConcurrentPublishStress:
    """publish_sync from many threads under load."""

    def test_1000_events_from_4_threads(self):
        from src.dashboard.server import state
        errors = []

        def publish_250(tid):
            try:
                for i in range(250):
                    state.publish_sync({
                        "type": "step",
                        "thread": tid,
                        "step": i,
                        "timestamp": time.time(),
                    })
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=publish_250, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # With 1000 events and cap at 1000, all should be retained
        assert len(state.events) == 1000

    def test_exceeding_cap_from_threads(self):
        from src.dashboard.server import state
        def publish_400(tid):
            for i in range(400):
                state.publish_sync({
                    "type": "step",
                    "thread": tid,
                    "step": i,
                    "timestamp": time.time(),
                })

        threads = [threading.Thread(target=publish_400, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 1600 published, cap is 1000
        assert len(state.events) <= 1000


# ---------------------------------------------------------------------------
# Subscribe/unsubscribe under load
# ---------------------------------------------------------------------------

class TestSubscribeUnderLoad:
    """Subscribe and unsubscribe while events are being published."""

    def test_subscribe_unsubscribe_during_publish(self):
        from src.dashboard.server import state
        errors = []

        def publisher():
            try:
                for i in range(200):
                    state.publish_sync({
                        "type": "step", "step": i, "timestamp": time.time(),
                    })
            except Exception as e:
                errors.append(e)

        def subscriber_churn():
            try:
                for _ in range(50):
                    q = state.subscribe()
                    time.sleep(0.001)
                    state.unsubscribe(q)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=publisher)
        t2 = threading.Thread(target=subscriber_churn)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors


# ---------------------------------------------------------------------------
# Rapid agent switch requests
# ---------------------------------------------------------------------------

class TestRapidAgentSwitch:
    """Rapid POST /agent switches don't crash or corrupt state."""

    def test_rapid_switches(self):
        from starlette.testclient import TestClient
        from src.dashboard.server import app, state

        with TestClient(app) as client:
            for _ in range(20):
                resp = client.post("/agent/ppo")
                assert resp.status_code == 200
                resp = client.post("/agent/dqn")
                assert resp.status_code == 200

        # Final requested_agent should be one of the valid agents
        assert state.requested_agent in ("dqn", "ppo")
