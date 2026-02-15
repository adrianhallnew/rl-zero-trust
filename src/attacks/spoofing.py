"""Traffic spoofing attack simulation for the RL-driven adaptive security system.

Implements identity falsification attack patterns including:
    - IP spoofing: Packets with forged source IP addresses.
    - MAC spoofing: Frames with forged source MAC addresses.
    - ARP spoofing: Poisoned ARP replies to redirect traffic.

Each attack has two modes:
    1. **Live mode** — uses Scapy to craft spoofed packets inside Mininet.
    2. **Simulation mode** — generates synthetic state-vector signatures
       reflecting what the SDN controller would observe.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Spoofing technique types
SPOOF_IP = "ip_spoof"
SPOOF_MAC = "mac_spoof"
SPOOF_ARP = "arp_spoof"

SPOOF_PROFILES = {
    SPOOF_IP: {
        "flow_count_increase": (0.1, 0.3),
        "packet_count_increase": (1.0, 3.0),
        "byte_count_increase": (1.0, 2.5),
        "error_increase": (0.5, 2.0),
        "drop_increase": (0.3, 1.0),
        "throughput_range": (55.0, 80.0),
        "latency_range": (8.0, 25.0),
    },
    SPOOF_MAC: {
        "flow_count_increase": (0.15, 0.35),
        "packet_count_increase": (1.0, 2.5),
        "byte_count_increase": (1.0, 2.0),
        "error_increase": (0.8, 2.5),
        "drop_increase": (0.5, 1.5),
        "throughput_range": (50.0, 75.0),
        "latency_range": (10.0, 30.0),
    },
    SPOOF_ARP: {
        "flow_count_increase": (0.05, 0.15),
        "packet_count_increase": (0.5, 1.5),
        "byte_count_increase": (0.3, 1.0),
        "error_increase": (1.0, 3.0),
        "drop_increase": (0.8, 2.0),
        "throughput_range": (40.0, 70.0),
        "latency_range": (15.0, 35.0),
    },
}


@dataclass
class SpoofingConfig:
    """Configuration for a spoofing attack simulation.

    Attributes:
        spoof_type: ``"ip_spoof"``, ``"mac_spoof"``, or ``"arp_spoof"``.
        target_switch: Target switch index (0-4).  ``None`` for random.
        target_ip: IP of the host to impersonate or target.
        spoofed_ip: The forged source IP address.
        spoofed_mac: The forged source MAC address.
        gateway_ip: Gateway IP for ARP spoofing.
        duration_steps: Duration in environment steps for simulation mode.
        seed: Random seed for reproducibility.
    """

    spoof_type: str = SPOOF_IP
    target_switch: Optional[int] = None
    target_ip: str = "10.0.0.1"
    spoofed_ip: str = "10.0.0.99"
    spoofed_mac: str = "00:de:ad:be:ef:00"
    gateway_ip: str = "10.0.0.254"
    duration_steps: int = 20
    seed: int = 42


class SpoofingAttack:
    """Spoofing attack simulator with live and simulation modes.

    Args:
        config: Spoofing attack configuration.

    Example:
        >>> attack = SpoofingAttack(SpoofingConfig(spoof_type="ip_spoof"))
        >>> state = np.zeros(65, dtype=np.float32)
        >>> modified = attack.inject_signature(state, rng=np.random.RandomState(42))
        >>> assert modified[10] > state[10]  # rx_errors increased
    """

    def __init__(self, config: Optional[SpoofingConfig] = None) -> None:
        self.config = config or SpoofingConfig()
        self.profile = SPOOF_PROFILES[self.config.spoof_type]
        self._active = False
        self._steps_elapsed = 0
        self._start_time: Optional[float] = None

        logger.info(
            "SpoofingAttack created: type=%s, target_sw=%s",
            self.config.spoof_type, self.config.target_switch,
        )

    @property
    def is_active(self) -> bool:
        """Whether the attack is currently active."""
        return self._active

    @property
    def attack_name(self) -> str:
        """Human-readable attack name."""
        return f"spoofing_{self.config.spoof_type}"

    def start(self) -> None:
        """Start the attack."""
        self._active = True
        self._steps_elapsed = 0
        self._start_time = time.time()
        logger.info("Spoofing attack started: %s", self.attack_name)

    def stop(self) -> None:
        """Stop the attack."""
        self._active = False
        elapsed = time.time() - self._start_time if self._start_time else 0
        logger.info(
            "Spoofing attack stopped after %d steps (%.1fs)",
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
        """Inject spoofing signatures into a state vector.

        Spoofing attacks are characterised by increased error rates
        (mismatched MACs/IPs trigger controller alerts), moderate flow
        increases, and elevated drop rates from switch confusion.

        Args:
            state: Current state vector of shape ``(65,)``.
            rng: Random number generator.
            target_switch: Override target switch (0-4).

        Returns:
            Modified state vector with spoofing signatures.
        """
        modified = state.copy()
        sw = target_switch if target_switch is not None else self.config.target_switch
        if sw is None:
            sw = rng.randint(0, 5)

        offset = sw * 13
        p = self.profile

        # Common spoofing signatures: flow & packet increases
        modified[offset + 0] += rng.uniform(*p["flow_count_increase"])
        modified[offset + 1] += rng.uniform(*p["packet_count_increase"])
        modified[offset + 2] += rng.uniform(*p["byte_count_increase"])

        # Key spoofing signatures: elevated errors
        modified[offset + 10] += rng.uniform(*p["error_increase"])  # rx_errors
        modified[offset + 11] += rng.uniform(*p["error_increase"])  # tx_errors

        # Elevated drops from switch MAC table confusion
        modified[offset + 8] += rng.uniform(*p["drop_increase"])  # rx_dropped
        modified[offset + 9] += rng.uniform(*p["drop_increase"])  # tx_dropped

        if self.config.spoof_type == SPOOF_ARP:
            # ARP spoofing causes MAC table instability across multiple switches
            # Propagate to adjacent switches
            for adj_sw in range(5):
                if adj_sw != sw:
                    adj_offset = adj_sw * 13
                    modified[adj_offset + 10] += rng.uniform(0.1, 0.5)
                    modified[adj_offset + 11] += rng.uniform(0.1, 0.5)
                    modified[adj_offset + 8] += rng.uniform(0.1, 0.3)

        elif self.config.spoof_type == SPOOF_MAC:
            # MAC spoofing: frequent MAC table updates trigger more flows
            modified[offset + 0] += rng.uniform(0.05, 0.15)
            modified[offset + 12] += rng.uniform(1.0, 3.0)  # conn rate increase

        elif self.config.spoof_type == SPOOF_IP:
            # IP spoofing: unusual source IPs create many unique flows
            modified[offset + 0] += rng.uniform(0.1, 0.25)
            modified[offset + 12] += rng.uniform(0.5, 2.0)

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
            "attack_type": "spoofing",
            "attack_subtype": self.config.spoof_type,
            "steps_elapsed": self._steps_elapsed,
            "target_switch": self.config.target_switch,
        }

    # ------------------------------------------------------------------
    # Live Mode (Scapy)
    # ------------------------------------------------------------------

    def generate_live_packets(self) -> Optional[List[Any]]:
        """Generate Scapy packets for live network injection.

        Returns:
            List of Scapy packets, or None if scapy is not available.
        """
        try:
            from scapy.all import IP, TCP, UDP, Ether, ARP, RandShort
        except ImportError:
            logger.warning("Scapy not available; live packet generation disabled")
            return None

        packets = []

        if self.config.spoof_type == SPOOF_IP:
            # Packets with forged source IP
            for _ in range(10):
                pkt = (
                    IP(src=self.config.spoofed_ip, dst=self.config.target_ip)
                    / TCP(sport=RandShort(), dport=80, flags="S")
                )
                packets.append(pkt)

        elif self.config.spoof_type == SPOOF_MAC:
            # Frames with forged source MAC
            for _ in range(10):
                pkt = (
                    Ether(src=self.config.spoofed_mac)
                    / IP(src=self.config.spoofed_ip, dst=self.config.target_ip)
                    / TCP(sport=RandShort(), dport=80, flags="S")
                )
                packets.append(pkt)

        elif self.config.spoof_type == SPOOF_ARP:
            # ARP poisoning: claim to be the gateway
            pkt = (
                Ether(dst="ff:ff:ff:ff:ff:ff")
                / ARP(
                    op=2,  # ARP reply
                    psrc=self.config.gateway_ip,
                    hwsrc=self.config.spoofed_mac,
                    pdst=self.config.target_ip,
                )
            )
            packets.append(pkt)

        logger.debug(
            "Generated %d live %s packets",
            len(packets), self.config.spoof_type,
        )
        return packets
