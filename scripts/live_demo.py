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

ATTACK_CMD_MAP = {
    "ddos":     "src/attacks/ddos.py",
    "portscan": "src/attacks/port_scan.py",
    "spoofing": "src/attacks/spoofing.py",
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
    ) -> None:
        self.scenario = scenario
        self.schedule = schedule or ATTACK_SCHEDULE
        self.attack_active = False
        self.attack_type: Optional[str] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._processes: List[subprocess.Popen] = []

    def start(self) -> None:
        """Start the scheduler in a background daemon thread."""
        if self.scenario == "none":
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="attack-scheduler",
        )
        self._thread.start()
        logger.info("Attack scheduler started (scenario=%s)", self.scenario)

    def stop(self) -> None:
        """Signal the scheduler to stop and kill any running attacks."""
        self._stop_event.set()
        for proc in self._processes:
            try:
                proc.terminate()
            except OSError:
                pass
        self._processes.clear()
        self.attack_active = False
        self.attack_type = None
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("Attack scheduler stopped")

    # ---- internal --------------------------------------------------------

    def _run(self) -> None:
        """Execute the attack schedule sequentially."""
        if self.scenario == "auto":
            self._run_auto_schedule()
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

    def _fire_single(
        self,
        attack_type: str,
        duration: int = 30,
        intensity: Optional[str] = None,
    ) -> None:
        """Launch one attack type inside the Mininet container."""
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

        self.attack_active = False
        self.attack_type = None
        logger.info("ATTACK STOP: %s", attack_type)

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
                stderr=subprocess.DEVNULL,
            )
            self._processes.append(proc)
            return proc
        except FileNotFoundError:
            logger.warning(
                "docker not found — attack %s skipped (no Docker?)",
                attack_type,
            )
            return None


# ---------------------------------------------------------------------------
# RL Loop
# ---------------------------------------------------------------------------


def run_rl_loop(
    agent_name: str,
    max_steps: int,
    step_interval: float,
    attack_scheduler: AttackScheduler,
    event_callback=None,
) -> None:
    """Run the live-mode RL loop.

    Args:
        agent_name: ``"dqn"`` or ``"ppo"``.
        max_steps: Number of environment steps to run.
        step_interval: Seconds to sleep between steps.
        attack_scheduler: Provides ground-truth attack state.
        event_callback: Optional callable receiving the event dict each step
            (used by the dashboard SSE bus in Sprint 11).
    """
    # --- Instantiate SDN components ---
    logger.info("Connecting to Ryu at %s ...", RYU_API_URL)
    collector = StatsCollector(ryu_api_url=RYU_API_URL)
    enforcer = PolicyEnforcer(ryu_api_url=RYU_API_URL)

    if not collector.wait_for_controller(timeout=30):
        logger.error(
            "Cannot reach Ryu controller at %s — is Docker running?",
            RYU_API_URL,
        )
        sys.exit(1)
    logger.info("Ryu controller reachable")

    switches = collector.get_switches()
    logger.info("Switches registered: %s", switches)

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
        agent.load(DQN_CHECKPOINT)
        logger.info("DQN agent loaded from %s", DQN_CHECKPOINT)
    else:
        config = get_hyperparameters("ppo")
        agent = PPOAgent(state_dim=65, action_dim=3, config=config)
        agent.load(PPO_CHECKPOINT)
        logger.info("PPO agent loaded from %s", PPO_CHECKPOINT)

    # --- RL loop ---
    obs, info = env.reset()
    cumulative_reward = 0.0

    print(f"\n{'='*70}")
    print(f"  LIVE MODE — Agent: {agent_name.upper()}  |  Steps: {max_steps}")
    print(f"  Ryu API: {RYU_API_URL}  |  Switches: {switches}")
    print(f"{'='*70}\n")

    for step_num in range(1, max_steps + 1):
        # Update attack ground truth from scheduler
        env.set_attack_state(
            attack_scheduler.attack_active,
            attack_scheduler.attack_type,
        )

        # Select action
        if agent_name == "dqn":
            action = agent.select_action(obs, greedy=True)
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

        if event_callback is not None:
            event_callback(event)

        if terminated or truncated:
            break

        time.sleep(step_interval)

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
        choices=["auto", "ddos", "portscan", "spoofing", "mixed", "none"],
        default="none",
        help="Attack scenario (default: none)",
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
            )
        except KeyboardInterrupt:
            pass
        finally:
            scheduler.stop()
            logger.info("Demo finished")
    else:
        # Sprint 11: dashboard mode (placeholder)
        print("Dashboard mode not yet implemented — use --no-dashboard")
        sys.exit(1)


if __name__ == "__main__":
    main()
