"""Attack simulation modules for the RL-driven adaptive security system.

Provides DDoS, port scanning, and spoofing attack simulations for
training and evaluating the RL agent. Each module supports both
live (Scapy) and simulation (synthetic state) modes.

Modules:
    ddos: SYN flood, UDP flood attacks.
    port_scan: TCP connect, SYN scan reconnaissance.
    spoofing: IP, MAC, and ARP spoofing.
    mixed_scenario: Multi-vector attack orchestration.
"""

from src.attacks.ddos import DDoSAttack, DDoSConfig
from src.attacks.port_scan import PortScanAttack, PortScanConfig
from src.attacks.spoofing import SpoofingAttack, SpoofingConfig
from src.attacks.mixed_scenario import (
    AttackOrchestrator,
    MixedScenarioConfig,
    ScenarioType,
)

__all__ = [
    "DDoSAttack",
    "DDoSConfig",
    "PortScanAttack",
    "PortScanConfig",
    "SpoofingAttack",
    "SpoofingConfig",
    "AttackOrchestrator",
    "MixedScenarioConfig",
    "ScenarioType",
]
