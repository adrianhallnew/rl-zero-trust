"""Port scanning attack simulation for the RL-driven adaptive security system.

Implements reconnaissance attack patterns including:
    - TCP connect scan: Full TCP handshake probes across port ranges.
    - SYN scan (half-open): SYN-only probes that avoid completing handshake.

Each attack has two modes:
    1. **Live mode** — uses Scapy to craft probe packets inside Mininet.
    2. **Simulation mode** — generates synthetic state-vector signatures
       mimicking SDN controller observations during port scanning.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Scan speed presets
SPEED_SLOW = "slow"
SPEED_NORMAL = "normal"
SPEED_AGGRESSIVE = "aggressive"

SPEED_PROFILES = {
    SPEED_SLOW: {
        "ports_per_second": 10,
        "delay_between_probes_ms": 100,
        "flow_count_increase": (0.3, 0.5),
        "duration_decrease": (0.01, 0.03),
        "conn_rate_increase": (3.0, 7.0),
        "packet_count_increase": (1.0, 2.0),
        "throughput_range": (70.0, 90.0),
        "latency_range": (5.0, 15.0),
    },
    SPEED_NORMAL: {
        "ports_per_second": 100,
        "delay_between_probes_ms": 10,
        "flow_count_increase": (0.4, 0.8),
        "duration_decrease": (0.01, 0.05),
        "conn_rate_increase": (5.0, 15.0),
        "packet_count_increase": (2.0, 4.0),
        "throughput_range": (60.0, 85.0),
        "latency_range": (8.0, 20.0),
    },
    SPEED_AGGRESSIVE: {
        "ports_per_second": 1000,
        "delay_between_probes_ms": 1,
        "flow_count_increase": (0.6, 1.2),
        "duration_decrease": (0.005, 0.02),
        "conn_rate_increase": (10.0, 25.0),
        "packet_count_increase": (3.0, 6.0),
        "throughput_range": (45.0, 70.0),
        "latency_range": (12.0, 30.0),
    },
}


@dataclass
class PortScanConfig:
    """Configuration for a port scanning simulation.

    Attributes:
        scan_type: ``"tcp_connect"`` or ``"syn_scan"``.
        speed: ``"slow"``, ``"normal"``, or ``"aggressive"``.
        target_switch: Target switch index (0-4).  ``None`` for random.
        target_ip: Target host IP for live mode.
        source_ip: Scanner source IP for live mode.
        port_range: Tuple of (start_port, end_port) to scan.
        randomize_ports: Randomize port order to evade sequential detection.
        duration_steps: Duration in environment steps for simulation mode.
        seed: Random seed for reproducibility.
    """

    scan_type: str = "tcp_connect"
    speed: str = SPEED_NORMAL
    target_switch: Optional[int] = None
    target_ip: str = "10.0.0.1"
    source_ip: str = "10.0.0.7"
    port_range: Tuple[int, int] = (1, 1024)
    randomize_ports: bool = False
    duration_steps: int = 25
    seed: int = 42


class PortScanAttack:
    """Port scanning attack simulator with live and simulation modes.

    Args:
        config: Port scan configuration.

    Example:
        >>> attack = PortScanAttack(PortScanConfig(speed="normal"))
        >>> state = np.zeros(65, dtype=np.float32)
        >>> modified = attack.inject_signature(state, rng=np.random.RandomState(42))
        >>> assert modified[12] > state[12]  # connection rate increased
    """

    def __init__(self, config: Optional[PortScanConfig] = None) -> None:
        self.config = config or PortScanConfig()
        self.profile = SPEED_PROFILES[self.config.speed]
        self._active = False
        self._steps_elapsed = 0
        self._start_time: Optional[float] = None

        logger.info(
            "PortScanAttack created: type=%s, speed=%s, ports=%s-%s",
            self.config.scan_type, self.config.speed,
            self.config.port_range[0], self.config.port_range[1],
        )

    @property
    def is_active(self) -> bool:
        """Whether the scan is currently active."""
        return self._active

    @property
    def attack_name(self) -> str:
        """Human-readable attack name."""
        return f"port_scan_{self.config.scan_type}"

    def start(self) -> None:
        """Start the scan."""
        self._active = True
        self._steps_elapsed = 0
        self._start_time = time.time()
        logger.info("Port scan started: %s", self.attack_name)

    def stop(self) -> None:
        """Stop the scan."""
        self._active = False
        elapsed = time.time() - self._start_time if self._start_time else 0
        logger.info(
            "Port scan stopped after %d steps (%.1fs)",
            self._steps_elapsed, elapsed,
        )

    def step(self) -> bool:
        """Advance one simulation step.

        Returns:
            True if the scan is still active, False if it has ended.
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
        """Inject port scan signatures into a state vector.

        Port scans are characterised by many short-lived flows with
        very brief durations and a high connection rate, but relatively
        low byte volume per flow.

        Args:
            state: Current state vector of shape ``(65,)``.
            rng: Random number generator.
            target_switch: Override target switch (0-4).

        Returns:
            Modified state vector with port scan signatures.
        """
        modified = state.copy()
        sw = target_switch if target_switch is not None else self.config.target_switch
        if sw is None:
            sw = rng.randint(0, 5)

        offset = sw * 13
        p = self.profile

        # Many short-lived flows (key port scan signature)
        modified[offset + 0] += rng.uniform(*p["flow_count_increase"])
        # Low packet count per flow but many flows
        modified[offset + 1] += rng.uniform(*p["packet_count_increase"])
        # Very low byte count (probes are tiny)
        modified[offset + 2] += rng.uniform(0.5, 1.5)
        # Very short flow duration (key signature)
        modified[offset + 3] = rng.uniform(*p["duration_decrease"])
        # Connection rate spikes dramatically
        modified[offset + 12] += rng.uniform(*p["conn_rate_increase"])

        if self.config.scan_type == "syn_scan":
            # SYN scan: even shorter flows, slightly higher error rate
            # because RST packets are common
            modified[offset + 3] = rng.uniform(0.001, 0.01)
            modified[offset + 10] += rng.uniform(0.2, 0.8)  # rx_errors (RSTs)
            modified[offset + 11] += rng.uniform(0.1, 0.4)  # tx_errors

        elif self.config.scan_type == "tcp_connect":
            # TCP connect: slightly longer flows (full handshake)
            modified[offset + 3] = rng.uniform(0.02, 0.08)
            # Moderate tx_packets from completed handshakes
            modified[offset + 5] += rng.uniform(0.5, 1.5)

        return modified

    def get_traffic_impact(
        self, rng: np.random.RandomState,
    ) -> Tuple[float, float]:
        """Get throughput and latency during the scan.

        Port scans have less throughput impact than DDoS but still
        degrade performance due to connection table pressure.

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
        """Get ground truth labels for the current scan state.

        Returns:
            Dictionary with attack metadata for evaluation.
        """
        return {
            "attack_active": self._active,
            "attack_type": "port_scan",
            "attack_subtype": self.config.scan_type,
            "speed": self.config.speed,
            "steps_elapsed": self._steps_elapsed,
            "target_switch": self.config.target_switch,
        }

    # ------------------------------------------------------------------
    # Live Mode (Scapy)
    # ------------------------------------------------------------------

    def generate_live_packets(
        self, num_ports: int = 20,
    ) -> Optional[List[Any]]:
        """Generate Scapy probe packets for live network injection.

        Args:
            num_ports: Number of ports to probe in this batch.

        Returns:
            List of Scapy packets, or None if scapy is not available.
        """
        try:
            from scapy.all import IP, TCP, RandShort
        except ImportError:
            logger.warning("Scapy not available; live packet generation disabled")
            return None

        packets = []
        start_port, end_port = self.config.port_range
        ports = list(range(start_port, min(start_port + num_ports, end_port + 1)))

        if self.config.randomize_ports:
            rng = np.random.RandomState(self.config.seed + self._steps_elapsed)
            rng.shuffle(ports)

        for port in ports:
            if self.config.scan_type == "syn_scan":
                pkt = (
                    IP(src=self.config.source_ip, dst=self.config.target_ip)
                    / TCP(sport=RandShort(), dport=port, flags="S")
                )
            else:  # tcp_connect
                pkt = (
                    IP(src=self.config.source_ip, dst=self.config.target_ip)
                    / TCP(sport=RandShort(), dport=port, flags="S")
                )
            packets.append(pkt)

        logger.debug(
            "Generated %d %s probe packets for ports %d-%d",
            len(packets), self.config.scan_type, ports[0], ports[-1],
        )
        return packets
