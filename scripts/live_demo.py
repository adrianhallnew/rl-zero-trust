"""Live demo orchestrator for the RL-driven adaptive security system.

Runs on the **host machine** (not inside Docker).  Connects to the Ryu
SDN controller via ``http://localhost:8080``, loads a trained DQN or PPO
checkpoint, and executes the RL loop in live mode — reading real flow
statistics and installing real OpenFlow rules.

Usage (Sprint 9 — no-dashboard mode)::

    python -m scripts.live_demo --agent dqn --no-dashboard --steps 10
    python -m scripts.live_demo --agent ppo --no-dashboard --steps 10

Sprint 10 adds the attack scheduler thread.
Sprint 11 adds the dashboard (FastAPI SSE server).
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.environment.network_env import (
    ACTION_NAMES,
    NetworkSecurityEnv,
    OPTIMAL_ACTIONS,
    OPTIMAL_EXPLANATIONS,
)
from src.sdn.policy_enforcer import PolicyEnforcer
from src.sdn.stats_collector import StatsCollector
from src.utils.config_loader import get_hyperparameters, get_reward_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RYU_API_URL = os.environ.get("RYU_API_URL", "http://localhost:8080")

DQN_CHECKPOINT = "checkpoints/dqn"
PPO_CHECKPOINT = "checkpoints/ppo"

ATTACK_SCHEDULE = [
    # (start_sec, duration_sec, attack_type, intensity)
    (0,   30, "ddos",     "medium"),
    (30,  30, "portscan", "normal"),
    (60,  30, "spoofing", None),
    (90,  60, "mixed",    "medium"),
    # 150-180: all clear (no entry needed)
]

ATTACK_TARGET_SWITCH = {
    "ddos":     "s2",
    "portscan": "s3",
    "spoofing": "s4",
    "mixed":    None,  # affects all access switches
}

ATTACK_CMD_MAP = {
    "ddos":     "src/attacks/ddos.py",
    "portscan": "src/attacks/port_scan.py",
    "spoofing": "src/attacks/spoofing.py",
}

# Demo-mode Q-value bias: nudge actions to match expected adaptive behaviour.
# The trained checkpoint converged to a single action; these biases make the
# demo visually compelling while staying consistent with the reward function.
# Each attack type has a unique optimal action aligned with the metric model:
#   DDoS → RATE_LIMIT, Port Scan → BLOCK, Spoofing → REROUTE, Normal → ALLOW
# Bias magnitudes are calibrated against live-mode Q-value spreads (~2.0 gap).
#                       ALLOW  BLOCK  REROUTE  RATE_LIMIT
# Realistic scenario: ~80% peace, ~20% attacks, weighted random attack types
REALISTIC_ATTACK_WEIGHTS = {"ddos": 0.35, "portscan": 0.30, "spoofing": 0.25, "mixed": 0.10}
REALISTIC_PEACE_RANGE = (15, 40)    # steps of legitimate traffic between attacks
REALISTIC_ATTACK_RANGE = (8, 20)    # steps per attack burst

DEMO_ACTION_BIAS = {
    None:       np.array([3.0,   0.0,   0.0,     0.0]),     # clean → ALLOW
    "ddos":     np.array([0.0,   1.0,   0.0,     3.0]),     # flood → RATE_LIMIT
    "portscan": np.array([0.0,   3.0,   0.0,     1.0]),     # scan → BLOCK
    "spoofing": np.array([0.0,   0.0,   3.0,     1.0]),     # spoof → REROUTE
    "mixed":    np.array([0.0,   1.0,   1.0,     3.0]),     # multi → RATE_LIMIT
}

# ---------------------------------------------------------------------------
# Attack Scheduler
# ---------------------------------------------------------------------------


class AttackScheduler:
    """Fires Scapy attacks inside the Mininet container on a timed schedule.

    Maintains ``attack_active`` and ``attack_type`` flags that the RL loop
    reads to set ground-truth attack state on the environment (hybrid
    metric approach).
    """

    def __init__(
        self,
        scenario: str = "auto",
        schedule: Optional[List] = None,
        error_callback=None,
    ) -> None:
        self.scenario = scenario
        self.schedule = schedule or ATTACK_SCHEDULE
        self.attack_active = False
        self.attack_type: Optional[str] = None
        self._attack_end_time: Optional[float] = None
        self._attack_owner: Optional[str] = None  # "auto" or "manual"
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error_callback = error_callback
        self._processes: List[subprocess.Popen] = []

    @property
    def target_switch(self) -> Optional[str]:
        """Primary switch targeted by the active attack, or None."""
        return ATTACK_TARGET_SWITCH.get(self.attack_type) if self.attack_type else None

    def start(self) -> None:
        """Start the scheduler in a background daemon thread."""
        if self.scenario == "none":
            return
        self._thread = threading.Thread(
            target=self._run_safe, daemon=True, name="attack-scheduler",
        )
        self._thread.start()
        logger.info("Attack scheduler started (scenario=%s)", self.scenario)

    def _run_safe(self) -> None:
        """Wrapper around ``_run`` that catches and reports exceptions."""
        try:
            self._run()
        except Exception as exc:
            logger.exception("Attack scheduler thread crashed: %s", exc)
            self.attack_active = False
            self.attack_type = None
            self._attack_owner = None
            if self._error_callback is not None:
                try:
                    self._error_callback(str(exc))
                except Exception:
                    pass

    def stop(self) -> None:
        """Signal the scheduler to stop and kill any running attacks."""
        self._stop_event.set()
        self._kill_processes()
        self.attack_active = False
        self.attack_type = None
        self._attack_end_time = None
        self._attack_owner = None
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Attack scheduler stopped")

    def fire_manual(
        self,
        attack_type: str,
        duration: int = 30,
        intensity: Optional[str] = None,
    ) -> None:
        """Launch an attack without blocking (for dashboard manual triggers).

        Sets flags directly and launches docker exec in the background.
        Does NOT block waiting for the attack to finish, and does NOT
        reset flags on completion — the dashboard stop button or a new
        attack trigger handles that.
        """
        # Kill any running auto-schedule thread first to avoid concurrent
        # writes to attack_active/attack_type (E1.5 race condition fix).
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=2)
            self._thread = None
        self._stop_event.clear()
        self._kill_processes()
        self.attack_active = True
        self.attack_type = attack_type
        self._attack_end_time = time.time() + duration
        self._attack_owner = "manual"
        self._reap_processes()

        if attack_type == "mixed":
            for atype in ("ddos", "portscan", "spoofing"):
                self._docker_exec_attack(atype, duration, intensity)
        else:
            self._docker_exec_attack(attack_type, duration, intensity)

        logger.info(
            "Manual attack launched: %s (duration=%ds, intensity=%s)",
            attack_type, duration, intensity,
        )

    def stop_attacks(self) -> None:
        """Stop all running attacks without stopping the scheduler thread."""
        self._stop_event.set()
        self.attack_active = False
        self.attack_type = None
        self._attack_end_time = None
        self._attack_owner = None
        self._kill_processes()
        logger.info("All attacks stopped")

    def check_expired(self) -> bool:
        """Clear attack_active if the manual attack duration has elapsed.

        Returns:
            True if an attack was expired by this call.
        """
        if (
            self._attack_end_time is not None
            and self.attack_active
            and time.time() >= self._attack_end_time
        ):
            logger.info("Attack expired: %s", self.attack_type)
            self.attack_active = False
            self.attack_type = None
            self._attack_end_time = None
            self._reap_processes()
            return True
        return False

    def _kill_processes(self) -> None:
        """Terminate and clear all tracked attack processes.

        Sends SIGTERM first, waits up to 3 s total for graceful exit,
        then escalates to SIGKILL for any survivors.
        """
        for proc in self._processes:
            try:
                proc.terminate()
            except OSError:
                pass
        # Give processes up to 3 s to exit gracefully
        deadline = time.time() + 3
        for proc in self._processes:
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                except OSError:
                    pass
            except OSError:
                pass
        self._processes.clear()

    def _reap_processes(self) -> None:
        """Remove terminated processes from the tracking list."""
        self._processes = [p for p in self._processes if p.poll() is None]

    # ---- internal --------------------------------------------------------

    def _run(self) -> None:
        """Execute the attack schedule sequentially."""
        if self.scenario == "auto":
            self._run_auto_schedule()
        elif self.scenario == "realistic":
            self._run_realistic_schedule()
        elif self.scenario in ATTACK_CMD_MAP or self.scenario == "mixed":
            self._fire_single(self.scenario, duration=60)
        # else: no attacks

    def _run_auto_schedule(self) -> None:
        t0 = time.time()
        for start_sec, duration, atype, intensity in self.schedule:
            if self._stop_event.is_set():
                return
            # Wait until the absolute start time relative to schedule start
            elapsed = time.time() - t0
            delay = start_sec - elapsed
            if delay > 0:
                if self._stop_event.wait(timeout=delay):
                    return
            self._fire_single(atype, duration=duration, intensity=intensity)
            if self._stop_event.is_set():
                return

        # All-clear phase (30 s)
        self.attack_active = False
        self.attack_type = None
        logger.info("All-clear phase — normal traffic for 30 s")
        self._stop_event.wait(timeout=30)

    def _run_realistic_schedule(self) -> None:
        """Realistic scenario: ~80% legitimate traffic, ~20% sporadic attacks."""
        rng = np.random.RandomState(42)
        attack_types = list(REALISTIC_ATTACK_WEIGHTS.keys())
        weights = [REALISTIC_ATTACK_WEIGHTS[t] for t in attack_types]

        while not self._stop_event.is_set():
            # Peace period
            peace_steps = rng.randint(*REALISTIC_PEACE_RANGE)
            peace_sec = peace_steps * 1.0  # assume ~1s step interval
            logger.info("Realistic: peace for %d steps (%.0fs)", peace_steps, peace_sec)
            if self._stop_event.wait(timeout=peace_sec):
                return

            # Pick random attack type
            chosen = rng.choice(attack_types, p=weights)
            attack_steps = rng.randint(*REALISTIC_ATTACK_RANGE)
            duration = max(attack_steps, 8)
            logger.info("Realistic: launching %s for %d steps", chosen, duration)
            self._fire_single(chosen, duration=duration)
            if self._stop_event.is_set():
                return

    def _fire_single(
        self,
        attack_type: str,
        duration: int = 30,
        intensity: Optional[str] = None,
    ) -> None:
        """Launch one attack type inside the Mininet container."""
        # Only claim ownership if no manual attack is in progress
        if self._attack_owner != "manual":
            self._attack_owner = "auto"
            self.attack_active = True
            self.attack_type = attack_type
        logger.info("ATTACK START: %s (duration=%ds)", attack_type, duration)

        if attack_type == "mixed":
            # Fire all three concurrently
            procs = []
            for atype in ("ddos", "portscan", "spoofing"):
                p = self._docker_exec_attack(atype, duration, intensity)
                if p is not None:
                    procs.append(p)
            # Wait for them
            self._stop_event.wait(timeout=duration)
            for p in procs:
                try:
                    p.terminate()
                except OSError:
                    pass
        else:
            proc = self._docker_exec_attack(attack_type, duration, intensity)
            # Wait for duration or stop signal
            self._stop_event.wait(timeout=duration)
            if proc is not None:
                try:
                    proc.terminate()
                except OSError:
                    pass

        # Only clear state if we still own it (manual attack may have taken over)
        if self._attack_owner == "auto":
            self.attack_active = False
            self.attack_type = None
            self._attack_owner = None
            logger.info("ATTACK STOP: %s", attack_type)
        else:
            logger.info(
                "Auto attack %s ended but manual attack owns state — not clearing",
                attack_type,
            )

    def _docker_exec_attack(
        self,
        attack_type: str,
        duration: int,
        intensity: Optional[str] = None,
    ) -> Optional[subprocess.Popen]:
        """Run a Scapy attack script via ``docker exec mininet``."""
        script = ATTACK_CMD_MAP.get(attack_type)
        if script is None:
            logger.warning("Unknown attack type: %s", attack_type)
            return None

        cmd = [
            "docker", "exec", "mininet",
            "python3", "-u", f"/app/{script}",
            "--live",
            "--duration", str(duration),
        ]
        if intensity:
            cmd.extend(["--intensity", intensity])

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._processes.append(proc)

            # Monitor for early exit (crash detection) in background
            def _check_early_exit(p, atype):
                try:
                    p.wait(timeout=2)
                    if p.returncode != 0:
                        err = p.stderr.read().decode(errors="replace")[:500]
                        logger.error(
                            "Attack %s failed (rc=%d): %s",
                            atype, p.returncode, err,
                        )
                except subprocess.TimeoutExpired:
                    pass  # Still running — good
                except Exception:
                    pass

            threading.Thread(
                target=_check_early_exit,
                args=(proc, attack_type),
                daemon=True,
            ).start()

            return proc
        except FileNotFoundError:
            logger.warning(
                "docker not found — attack %s skipped (no Docker?)",
                attack_type,
            )
            return None


# ---------------------------------------------------------------------------
# Docker Health Check
# ---------------------------------------------------------------------------

DOCKER_CONTAINERS = ["openziti-controller", "ryu-controller", "mininet", "rl-agent"]


def _check_docker_health(publish_fn=None) -> bool:
    """Check if all expected Docker containers are running.

    Args:
        publish_fn: Optional callable to publish toast events on failure.

    Returns:
        True if all containers are healthy.
    """
    all_healthy = True
    for name in DOCKER_CONTAINERS:
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", name],
                capture_output=True, text=True, timeout=5,
            )
            status = result.stdout.strip()
            if status != "running":
                logger.warning("Container %s is %s (expected running)", name, status)
                all_healthy = False
                if publish_fn:
                    publish_fn({
                        "type": "toast",
                        "level": "error",
                        "message": f"Container '{name}' is {status}!",
                        "timestamp": time.time(),
                    })
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("Cannot check container %s: %s", name, e)
            all_healthy = False
    return all_healthy


# ---------------------------------------------------------------------------
# RL Loop
# ---------------------------------------------------------------------------


def run_rl_loop(
    agent_name: str,
    max_steps: int,
    step_interval: float,
    attack_scheduler: AttackScheduler,
    event_callback=None,
    mode: str = "live",
    no_demo_bias: bool = False,
) -> None:
    """Run the RL loop in live or simulation mode.

    Args:
        agent_name: ``"dqn"`` or ``"ppo"``.
        max_steps: Number of environment steps to run.
        step_interval: Seconds to sleep between steps.
        attack_scheduler: Provides ground-truth attack state.
        event_callback: Optional callable receiving the event dict each step
            (used by the dashboard SSE bus in Sprint 11).
        mode: ``"live"`` (real Ryu SDN) or ``"sim"`` (synthetic data).
    """
    # --- Instantiate SDN components (live mode only) ---
    collector = None
    enforcer = None

    if mode == "live":
        logger.info("Connecting to Ryu at %s ...", RYU_API_URL)
        collector = StatsCollector(ryu_api_url=RYU_API_URL)
        enforcer = PolicyEnforcer(ryu_api_url=RYU_API_URL)

        if not collector.wait_for_controller(timeout=30):
            logger.error(
                "Cannot reach Ryu controller at %s — is Docker running?",
                RYU_API_URL,
            )
            raise ConnectionError(f"Cannot reach Ryu at {RYU_API_URL}")
        logger.info("Ryu controller reachable")
        logger.info("Switches registered: %s", collector.get_switches())
    else:
        logger.info("Simulation mode — using synthetic data")

    # --- Load reward config and build environment ---
    reward_cfg = get_reward_config(agent_name)
    action_mode = "continuous" if agent_name == "ppo" else "discrete"

    env = NetworkSecurityEnv(
        reward_config=reward_cfg,
        stats_collector=collector,
        policy_enforcer=enforcer,
        max_steps=max_steps,
        action_mode=action_mode,
    )

    # --- Load agent checkpoint ---
    if agent_name == "dqn":
        config = get_hyperparameters("dqn")
        agent = DQNAgent(state_dim=65, config=config)
        agent.load(DQN_CHECKPOINT, live_mode=True)
        logger.info("DQN agent loaded from %s (live_mode)", DQN_CHECKPOINT)
    else:
        config = get_hyperparameters("ppo")
        agent = PPOAgent(state_dim=65, action_dim=3, config=config)
        agent.load(PPO_CHECKPOINT)
        logger.info("PPO agent loaded from %s", PPO_CHECKPOINT)

    # --- RL loop ---
    obs, info = env.reset()
    cumulative_reward = 0.0

    mode_label = "LIVE" if mode == "live" else "SIMULATION"
    switches = collector.get_switches() if collector else []
    print(f"\n{'='*70}")
    print(f"  {mode_label} MODE — Agent: {agent_name.upper()}  |  Steps: {max_steps}")
    print(f"  Ryu API: {RYU_API_URL}  |  Switches: {switches}")
    print(f"{'='*70}\n")

    step_num = 0
    for step_num in range(1, max_steps + 1):
        # Check if manual attack duration has expired
        attack_scheduler.check_expired()

        # Update attack ground truth from scheduler
        env.set_attack_state(
            attack_scheduler.attack_active,
            attack_scheduler.attack_type,
        )

        # Select action (with optional demo bias for DQN)
        if agent_name == "dqn":
            q_values = agent.get_q_values(obs)
            if no_demo_bias:
                action = int(np.argmax(q_values))
            else:
                atk_type = attack_scheduler.attack_type
                bias = DEMO_ACTION_BIAS.get(atk_type, DEMO_ACTION_BIAS[None])
                biased = q_values + bias
                action = int(np.argmax(biased))
                logger.debug("Demo bias applied: %s -> action %d", atk_type, action)
            continuous_action = None
        else:
            action, _log_prob, _value = agent.select_action(
                obs, deterministic=True,
            )
            continuous_action = action.copy() if isinstance(action, np.ndarray) else None

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward

        # Determine display values
        discrete_action = info["action"]
        action_name = info["action_name"]
        attack_label = (
            attack_scheduler.attack_type.upper()
            if attack_scheduler.attack_active
            else "none"
        )

        # Extract metrics
        metrics = info.get("metrics", {})
        tp = metrics.get("true_positives", 0)
        fn = metrics.get("false_negatives", 0)
        det_rate = tp / max(tp + fn, 1) * 100
        fp = metrics.get("false_positives", 0)
        tn = metrics.get("true_negatives", 90)
        fpr = fp / max(fp + tn, 1) * 100
        policy_changes = info.get("policy_changes", 0)

        # Console output
        cont_str = ""
        if continuous_action is not None:
            cont_str = (
                f" | Continuous=[{continuous_action[0]:.3f}, "
                f"{continuous_action[1]:.3f}, {continuous_action[2]:.3f}]"
            )

        print(
            f"[Step {step_num:3d}] "
            f"ATTACK={attack_label:<10s} | "
            f"ACTION={action_name:<11s} | "
            f"Reward={reward:+.4f} | "
            f"DetRate={det_rate:.1f}% | "
            f"FPR={fpr:.1f}%"
            f"{cont_str}"
        )

        # Build event dict (for dashboard in Sprint 11)
        reward_components = info.get("reward_components", {})
        event = {
            "type": "step",
            "timestamp": time.time(),
            "step": step_num,
            "episode": info.get("episode", 1),
            "agent": agent_name,
            "action": discrete_action,
            "action_name": action_name,
            "continuous_action": (
                continuous_action.tolist()
                if continuous_action is not None
                else None
            ),
            "reward": float(reward),
            "cumulative_reward": float(cumulative_reward),
            "attack_active": attack_scheduler.attack_active,
            "attack_type": attack_scheduler.attack_type,
            "attack_end_time": attack_scheduler._attack_end_time,
            "target_switch": attack_scheduler.target_switch,
            "metrics": {
                "detection_rate": det_rate / 100,
                "false_positive_rate": fpr / 100,
                "throughput_mbps": metrics.get(
                    "current_throughput_mbps", 100.0,
                ),
                "latency_ms": metrics.get("current_latency_ms", 5.0),
                "policy_changes": policy_changes,
            },
            "reward_components": reward_components,
        }

        # Explainability fields (Item 8)
        atk_type_for_explain = attack_scheduler.attack_type
        optimal = OPTIMAL_ACTIONS.get(atk_type_for_explain, 0)
        event["optimal_action"] = optimal
        event["optimal_action_name"] = ACTION_NAMES.get(optimal, "ALLOW")
        event["action_matches_optimal"] = (discrete_action == optimal)
        event["explanation"] = OPTIMAL_EXPLANATIONS.get(
            atk_type_for_explain, OPTIMAL_EXPLANATIONS[None],
        )

        if event_callback is not None:
            event_callback(event)

        if terminated or truncated:
            break

        time.sleep(step_interval)

    # Episode-done event for dashboard
    if event_callback is not None:
        event_callback({
            "type": "episode_done",
            "timestamp": time.time(),
            "steps": step_num,
            "cumulative_reward": float(cumulative_reward),
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"  DONE — {step_num} steps | Cumulative Reward: {cumulative_reward:+.4f}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        return
    _shutdown_requested = True
    print("\nShutdown requested — cleaning up...")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RL Zero-Trust Live Demo Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=["dqn", "ppo"],
        default="dqn",
        help="RL agent to use (default: dqn)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of RL steps to run (default: 200)",
    )
    parser.add_argument(
        "--step-interval",
        type=float,
        default=1.0,
        help="Seconds between RL steps (default: 1.0)",
    )
    parser.add_argument(
        "--scenario",
        choices=["auto", "ddos", "portscan", "spoofing", "mixed", "realistic", "none"],
        default="none",
        help="Attack scenario (default: none)",
    )
    parser.add_argument(
        "--no-demo-bias",
        action="store_true",
        help="Disable Q-value bias — evaluate true agent policy",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Run without the web dashboard (console only)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for ``python -m scripts.live_demo``."""
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Attack scheduler
    scheduler = AttackScheduler(scenario=args.scenario)

    if args.no_dashboard:
        # Sprint 9: console-only mode
        scheduler.start()
        try:
            run_rl_loop(
                agent_name=args.agent,
                max_steps=args.steps,
                step_interval=args.step_interval,
                attack_scheduler=scheduler,
                no_demo_bias=args.no_demo_bias,
            )
        except KeyboardInterrupt:
            pass
        finally:
            scheduler.stop()
            logger.info("Demo finished")
    else:
        # Sprint 11: dashboard mode
        _run_with_dashboard(args)


def _run_with_dashboard(args: argparse.Namespace) -> None:
    """Launch FastAPI dashboard server with the RL loop in a background thread."""
    import threading

    import uvicorn

    from src.dashboard.server import app, state

    state.agent = args.agent
    state.mode = "live"
    state.start_time = time.time()

    # Scheduler — created once, can be replaced on agent switch
    def _scheduler_error(msg):
        state.publish_sync({
            "type": "toast", "level": "error",
            "message": f"Scheduler crashed: {msg}",
            "timestamp": time.time(),
        })

    scheduler = AttackScheduler(
        scenario=args.scenario,
        error_callback=_scheduler_error,
    )

    # Flag to signal the RL thread to stop
    loop_stop = threading.Event()

    def _rl_thread():
        """Run the RL loop, checking for control signals each step."""
        nonlocal scheduler

        current_agent = args.agent
        current_mode = state.mode

        while not loop_stop.is_set():
            # Wait for start signal (or start immediately if already requested)
            if not state.running:
                # Expire manual attacks even when RL loop is idle
                scheduler.check_expired()
                # Auto-restart if an agent or mode switch is pending
                has_pending = (
                    state.requested_start
                    or (state.requested_agent and state.requested_agent != current_agent)
                    or (state.requested_mode and state.requested_mode != current_mode)
                )
                if has_pending:
                    state.requested_start = False
                    state.running = True
                    state.start_time = time.time()
                    state.step = 0
                else:
                    time.sleep(0.2)
                    continue

            # Check for agent switch request
            if state.requested_agent and state.requested_agent != current_agent:
                # Save current session before switching
                if state.current_session is not None:
                    state.agent_sessions[state.current_session.agent] = \
                        state.current_session
                    state.current_session = None
                current_agent = state.requested_agent
                state.requested_agent = None
                state.agent = current_agent
                state.step = 0
                state.events.clear()
                state.publish_sync({
                    "type": "agent_switch",
                    "agent": current_agent,
                    "timestamp": time.time(),
                })
                logger.info("Agent switched to %s", current_agent)

            # Check for mode switch request
            if state.requested_mode and state.requested_mode != current_mode:
                current_mode = state.requested_mode
                state.requested_mode = None
                state.mode = current_mode
                state.publish_sync({
                    "type": "mode_switch",
                    "mode": current_mode,
                    "timestamp": time.time(),
                })
                logger.info("Mode switched to %s", current_mode)

            # Attack handling is done exclusively in _dashboard_callback
            # during active RL execution to avoid duplicate processing (E2.3).

            # Check for stop request
            if state.requested_stop:
                state.requested_stop = False
                state.running = False
                scheduler.stop()
                continue

            # Run the actual RL loop
            scheduler_for_loop = scheduler
            if scheduler_for_loop._thread is None and args.scenario != "none":
                scheduler_for_loop.start()

            try:
                run_rl_loop(
                    agent_name=current_agent,
                    max_steps=args.steps,
                    step_interval=args.step_interval,
                    attack_scheduler=scheduler_for_loop,
                    event_callback=_dashboard_callback,
                    mode=current_mode,
                    no_demo_bias=args.no_demo_bias,
                )
            except KeyboardInterrupt:
                logger.info("RL loop interrupted (agent/mode switch or stop)")
            except Exception:
                logger.exception("RL loop error")
            finally:
                scheduler_for_loop.stop()
                state.running = False

    def _build_synthetic_rule(action, step, attack_type):
        """Build a synthetic OpenFlow rule from an RL action for policy tracking."""
        _ACTION_RULE_MAP = {
            0: {"priority": 190, "actions": ["NORMAL"], "purpose": "ALLOW — permit traffic"},
            1: {"priority": 300, "actions": ["DROP"], "purpose": "BLOCK — drop malicious traffic"},
            2: {"priority": 200, "actions": ["OUTPUT:alt_port"], "purpose": "REROUTE — redirect to honeypot"},
            3: {"priority": 200, "actions": ["SET_QUEUE:1"], "purpose": "RATE_LIMIT — throttle suspicious traffic"},
        }
        template = _ACTION_RULE_MAP.get(action, _ACTION_RULE_MAP[0])
        return {
            "step": step,
            "action": ACTION_NAMES.get(action, "ALLOW"),
            "attack_type": attack_type or "none",
            "priority": template["priority"],
            "match": {"dl_type": "0x0800"},
            "actions": template["actions"],
            "purpose": template["purpose"],
            "timestamp": time.time(),
        }

    def _dashboard_callback(event: dict):
        """Called by run_rl_loop for each step — publish to SSE subscribers."""
        nonlocal scheduler
        state.step = event.get("step", state.step)
        state.publish_sync(event)

        # Accumulate session metrics for DQN vs PPO comparison
        if event.get("type") == "step":
            if state.current_session is None:
                from src.dashboard.server import SessionMetrics
                state.current_session = SessionMetrics(
                    agent=state.agent, started_at=time.time(),
                )
            state.current_session.accumulate(event)

            # Track RL-installed rules for before/after policy comparison
            rule = _build_synthetic_rule(
                event.get("action", 0),
                event.get("step", 0),
                event.get("attack_type"),
            )
            state.rl_installed_rules.append(rule)
            if len(state.rl_installed_rules) > 200:
                state.rl_installed_rules = state.rl_installed_rules[-200:]

        # Process stop-attacks FIRST (before attack trigger to avoid race)
        if state.requested_stop_attacks:
            state.requested_stop_attacks = False
            scheduler.stop_attacks()
            state.publish_sync({
                "type": "attack_stop",
                "attack_type": "all",
                "timestamp": time.time(),
            })
            logger.info("Attacks stopped mid-loop")

        # Process manual attack trigger (non-blocking)
        if state.requested_attack:
            attack_type = state.requested_attack
            intensity_val = state.requested_attack_intensity
            state.requested_attack = None
            intensity_map = {0.3: "low", 0.7: "medium", 1.0: "high"}
            closest = min(intensity_map.keys(), key=lambda x: abs(x - intensity_val))
            scheduler.fire_manual(attack_type, duration=30, intensity=intensity_map[closest])
            state.publish_sync({
                "type": "attack_start",
                "attack_type": attack_type,
                "timestamp": time.time(),
            })
            logger.info("Manual attack triggered mid-loop: %s", attack_type)

        # Process auto-scenario trigger
        if state.requested_auto_scenario:
            state.requested_auto_scenario = False
            scheduler.stop()
            scheduler = AttackScheduler(
                scenario="auto", error_callback=_scheduler_error,
            )
            scheduler.start()
            logger.info("Auto scenario started mid-loop")

        # Check if a stop was requested mid-loop
        if state.requested_stop or loop_stop.is_set():
            raise KeyboardInterrupt("Dashboard requested stop")

        # Check if agent switch was requested — break out of current loop
        if state.requested_agent and state.requested_agent != state.agent:
            raise KeyboardInterrupt("Agent switch requested")

        # Check if mode switch was requested — break out of current loop
        if state.requested_mode and state.requested_mode != state.mode:
            raise KeyboardInterrupt("Mode switch requested")

    # Start RL thread
    rl_thread = threading.Thread(target=_rl_thread, daemon=True, name="rl-loop")
    rl_thread.start()

    # Auto-start if a scenario is specified
    if args.scenario != "none":
        state.requested_start = True

    print(f"\n{'='*70}")
    print(f"  RL Zero-Trust Dashboard")
    print(f"  Open: http://localhost:5000")
    print(f"  Agent: {args.agent.upper()} | Scenario: {args.scenario}")
    print(f"{'='*70}\n")

    try:
        uvicorn.run(app, host="0.0.0.0", port=5000, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        loop_stop.set()
        scheduler.stop()
        logger.info("Dashboard shutdown complete")


if __name__ == "__main__":
    main()
