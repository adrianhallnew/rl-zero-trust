"""Tests for live_demo.py CLI argument parsing, constants, and utilities.

Covers:
    - parse_args defaults
    - parse_args overrides for all flags
    - Invalid argument rejection
    - ATTACK_SCHEDULE structure
    - ATTACK_CMD_MAP completeness
    - DEMO_ACTION_BIAS shape, coverage, and magnitude
    - _check_docker_health status reporting
"""

import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from scripts.live_demo import (
    ATTACK_CMD_MAP,
    ATTACK_SCHEDULE,
    ATTACK_TARGET_SWITCH,
    DEMO_ACTION_BIAS,
    _check_docker_health,
    parse_args,
)


# ---------------------------------------------------------------------------
# parse_args defaults
# ---------------------------------------------------------------------------

class TestParseArgsDefaults:
    """parse_args returns correct defaults with no arguments."""

    def test_defaults(self):
        with patch("sys.argv", ["live_demo.py"]):
            args = parse_args()
        assert args.agent == "dqn"
        assert args.steps == 200
        assert args.step_interval == 1.0
        assert args.scenario == "none"
        assert args.no_dashboard is False


# ---------------------------------------------------------------------------
# parse_args overrides
# ---------------------------------------------------------------------------

class TestParseArgsOverrides:
    """parse_args accepts all valid flag combinations."""

    def test_agent_ppo(self):
        with patch("sys.argv", ["live_demo.py", "--agent", "ppo"]):
            args = parse_args()
        assert args.agent == "ppo"

    def test_steps(self):
        with patch("sys.argv", ["live_demo.py", "--steps", "50"]):
            args = parse_args()
        assert args.steps == 50

    def test_step_interval(self):
        with patch("sys.argv", ["live_demo.py", "--step-interval", "0.5"]):
            args = parse_args()
        assert args.step_interval == 0.5

    def test_scenario_auto(self):
        with patch("sys.argv", ["live_demo.py", "--scenario", "auto"]):
            args = parse_args()
        assert args.scenario == "auto"

    def test_all_scenario_choices(self):
        for scenario in ("auto", "ddos", "portscan", "spoofing", "mixed", "none"):
            with patch("sys.argv", ["live_demo.py", "--scenario", scenario]):
                args = parse_args()
            assert args.scenario == scenario

    def test_no_dashboard_flag(self):
        with patch("sys.argv", ["live_demo.py", "--no-dashboard"]):
            args = parse_args()
        assert args.no_dashboard is True

    def test_combined_flags(self):
        with patch("sys.argv", [
            "live_demo.py",
            "--agent", "ppo",
            "--steps", "100",
            "--step-interval", "2.0",
            "--scenario", "ddos",
            "--no-dashboard",
        ]):
            args = parse_args()
        assert args.agent == "ppo"
        assert args.steps == 100
        assert args.step_interval == 2.0
        assert args.scenario == "ddos"
        assert args.no_dashboard is True


# ---------------------------------------------------------------------------
# Invalid arguments
# ---------------------------------------------------------------------------

class TestParseArgsInvalid:
    """Invalid arguments cause SystemExit."""

    def test_invalid_agent(self):
        with patch("sys.argv", ["live_demo.py", "--agent", "a3c"]):
            with pytest.raises(SystemExit):
                parse_args()

    def test_invalid_scenario(self):
        with patch("sys.argv", ["live_demo.py", "--scenario", "unknown"]):
            with pytest.raises(SystemExit):
                parse_args()


# ---------------------------------------------------------------------------
# Constants validation
# ---------------------------------------------------------------------------

class TestAttackSchedule:
    """ATTACK_SCHEDULE entries have correct structure."""

    def test_schedule_is_list_of_tuples(self):
        assert isinstance(ATTACK_SCHEDULE, list)
        for entry in ATTACK_SCHEDULE:
            assert len(entry) == 4
            start, duration, atype, intensity = entry
            assert isinstance(start, (int, float))
            assert isinstance(duration, (int, float))
            assert isinstance(atype, str)

    def test_schedule_times_are_ascending(self):
        starts = [entry[0] for entry in ATTACK_SCHEDULE]
        assert starts == sorted(starts)

    def test_all_schedule_types_in_cmd_map(self):
        for _, _, atype, _ in ATTACK_SCHEDULE:
            if atype != "mixed":
                assert atype in ATTACK_CMD_MAP, f"{atype} not in ATTACK_CMD_MAP"


class TestAttackCmdMap:
    """ATTACK_CMD_MAP points to valid script paths."""

    def test_all_scripts_are_python(self):
        for atype, path in ATTACK_CMD_MAP.items():
            assert path.endswith(".py"), f"{atype} script not .py: {path}"

    def test_expected_attack_types_present(self):
        assert "ddos" in ATTACK_CMD_MAP
        assert "portscan" in ATTACK_CMD_MAP
        assert "spoofing" in ATTACK_CMD_MAP


class TestDemoActionBias:
    """DEMO_ACTION_BIAS has correct shape and keys."""

    def test_none_key_exists(self):
        assert None in DEMO_ACTION_BIAS

    def test_all_attack_types_covered(self):
        for atype in ("ddos", "portscan", "spoofing", "mixed"):
            assert atype in DEMO_ACTION_BIAS

    def test_bias_arrays_are_4d(self):
        for key, bias in DEMO_ACTION_BIAS.items():
            assert isinstance(bias, np.ndarray)
            assert bias.shape == (4,), f"Bias for {key} has shape {bias.shape}"

    def test_clean_traffic_favours_allow(self):
        bias = DEMO_ACTION_BIAS[None]
        assert np.argmax(bias) == 0  # ALLOW is index 0

    def test_ddos_favours_block(self):
        assert np.argmax(DEMO_ACTION_BIAS["ddos"]) == 1

    def test_spoofing_favours_reroute(self):
        assert np.argmax(DEMO_ACTION_BIAS["spoofing"]) == 2

    def test_mixed_favours_rate_limit(self):
        assert np.argmax(DEMO_ACTION_BIAS["mixed"]) == 3

    def test_bias_magnitude_reasonable(self):
        for key, bias in DEMO_ACTION_BIAS.items():
            assert 1.0 <= bias.max() <= 10.0, (
                f"Bias magnitude for {key} is {bias.max()}"
            )


# ---------------------------------------------------------------------------
# Docker health check
# ---------------------------------------------------------------------------

class TestDockerHealthCheck:
    """_check_docker_health reports container status correctly."""

    def test_all_running_returns_true(self):
        def mock_run(cmd, **kwargs):
            m = MagicMock()
            m.stdout = "running\n"
            return m

        with patch("scripts.live_demo.subprocess.run", side_effect=mock_run):
            assert _check_docker_health() is True

    def test_one_exited_returns_false(self):
        call_count = {"n": 0}

        def mock_run(cmd, **kwargs):
            call_count["n"] += 1
            m = MagicMock()
            m.stdout = "exited\n" if call_count["n"] == 2 else "running\n"
            return m

        toasts = []
        with patch("scripts.live_demo.subprocess.run", side_effect=mock_run):
            result = _check_docker_health(publish_fn=toasts.append)

        assert result is False
        assert len(toasts) == 1
        assert "exited" in toasts[0]["message"]
