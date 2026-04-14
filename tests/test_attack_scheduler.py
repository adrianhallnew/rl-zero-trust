"""Tests for AttackScheduler: timing, expiry, manual fire, stop, and sequencing.

Covers D2 spec (fire_manual/stop_attacks lifecycle) plus timing logic
(check_expired), auto-schedule sequencing, and concurrent attack guards.
"""

import time
from unittest.mock import patch, MagicMock

import pytest

from scripts.live_demo import AttackScheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scheduler():
    """A scheduler with no scenario (manual-only)."""
    return AttackScheduler(scenario="none")


# ---------------------------------------------------------------------------
# D2: fire_manual / stop_attacks lifecycle
# ---------------------------------------------------------------------------

class TestFireManual:
    """fire_manual sets flags and spawns correct processes."""

    def test_sets_active_flag_and_type(self, scheduler):
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler.fire_manual("ddos", duration=5)
        assert scheduler.attack_active is True
        assert scheduler.attack_type == "ddos"

    def test_sets_end_time(self, scheduler):
        before = time.time()
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler.fire_manual("portscan", duration=10)
        after = time.time()
        assert scheduler._attack_end_time >= before + 10
        assert scheduler._attack_end_time <= after + 10

    def test_mixed_fires_three_attack_types(self, scheduler):
        fired = []
        with patch.object(
            scheduler, "_docker_exec_attack",
            side_effect=lambda atype, dur, intensity=None: fired.append(atype),
        ):
            scheduler.fire_manual("mixed", duration=10)
        assert set(fired) == {"ddos", "portscan", "spoofing"}
        assert scheduler.attack_type == "mixed"

    def test_fire_manual_kills_previous_attack(self, scheduler):
        proc1 = MagicMock()
        proc1.poll.return_value = None  # still running
        scheduler._processes = [proc1]
        scheduler.attack_active = True
        scheduler.attack_type = "ddos"

        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler.fire_manual("portscan", duration=5)

        proc1.terminate.assert_called_once()
        assert scheduler.attack_type == "portscan"

    def test_intensity_passed_through(self, scheduler):
        calls = []
        with patch.object(
            scheduler, "_docker_exec_attack",
            side_effect=lambda atype, dur, intensity=None: calls.append(
                (atype, dur, intensity)
            ),
        ):
            scheduler.fire_manual("ddos", duration=15, intensity="high")
        assert calls == [("ddos", 15, "high")]


class TestStopAttacks:
    """stop_attacks clears all state and terminates processes."""

    def test_clears_all_flags(self, scheduler):
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler.fire_manual("ddos", duration=5)
        scheduler.stop_attacks()
        assert scheduler.attack_active is False
        assert scheduler.attack_type is None
        assert scheduler._attack_end_time is None

    def test_terminates_tracked_processes(self, scheduler):
        procs = [MagicMock() for _ in range(3)]
        scheduler._processes = list(procs)
        scheduler.stop_attacks()
        for p in procs:
            p.terminate.assert_called_once()
        assert len(scheduler._processes) == 0

    def test_stop_is_idempotent(self, scheduler):
        scheduler.stop_attacks()
        scheduler.stop_attacks()
        assert scheduler.attack_active is False


# ---------------------------------------------------------------------------
# Timing: check_expired
# ---------------------------------------------------------------------------

class TestCheckExpired:
    """check_expired clears attack state when duration elapses."""

    def test_unexpired_attack_stays_active(self, scheduler):
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler.fire_manual("ddos", duration=60)
        assert scheduler.check_expired() is False
        assert scheduler.attack_active is True

    def test_expired_attack_cleared(self, scheduler):
        with patch.object(scheduler, "_docker_exec_attack", return_value=None):
            scheduler.fire_manual("ddos", duration=1)
        # Force expiry
        scheduler._attack_end_time = time.time() - 1
        assert scheduler.check_expired() is True
        assert scheduler.attack_active is False
        assert scheduler.attack_type is None

    def test_no_attack_returns_false(self, scheduler):
        assert scheduler.check_expired() is False

    def test_already_inactive_returns_false(self, scheduler):
        scheduler._attack_end_time = time.time() - 10
        scheduler.attack_active = False
        assert scheduler.check_expired() is False


# ---------------------------------------------------------------------------
# Auto-schedule sequencing
# ---------------------------------------------------------------------------

class TestAutoSchedule:
    """Auto scenario executes schedule entries in order."""

    def test_schedule_fires_in_order(self):
        schedule = [
            (0, 1, "ddos", "low"),
            (1, 1, "portscan", "low"),
            (2, 1, "spoofing", "low"),
        ]
        sched = AttackScheduler(scenario="auto", schedule=schedule)
        fired = []

        def mock_fire(atype, duration=30, intensity=None):
            fired.append(atype)
            sched.attack_active = True
            sched.attack_type = atype
            # Simulate instant completion
            sched.attack_active = False
            sched.attack_type = None

        with patch.object(sched, "_fire_single", side_effect=mock_fire):
            sched._run_auto_schedule()

        assert fired == ["ddos", "portscan", "spoofing"]

    def test_stop_event_aborts_schedule(self):
        schedule = [
            (0, 1, "ddos", "low"),
            (0, 1, "portscan", "low"),
        ]
        sched = AttackScheduler(scenario="auto", schedule=schedule)
        fired = []

        def mock_fire(atype, duration=30, intensity=None):
            fired.append(atype)
            sched._stop_event.set()  # abort after first

        with patch.object(sched, "_fire_single", side_effect=mock_fire):
            sched._run_auto_schedule()

        assert fired == ["ddos"]


# ---------------------------------------------------------------------------
# Process reaping
# ---------------------------------------------------------------------------

class TestProcessReaping:
    """_reap_processes removes finished subprocesses."""

    def test_removes_finished_keeps_running(self, scheduler):
        alive = MagicMock()
        alive.poll.return_value = None
        dead = MagicMock()
        dead.poll.return_value = 0
        scheduler._processes = [alive, dead]
        scheduler._reap_processes()
        assert scheduler._processes == [alive]

    def test_empty_list_is_noop(self, scheduler):
        scheduler._reap_processes()
        assert scheduler._processes == []
