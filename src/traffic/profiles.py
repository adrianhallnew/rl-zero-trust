"""Traffic profile definitions for the RL-driven adaptive security system.

Defines named traffic profiles used for baseline testing and evaluation.
Each profile specifies protocol, duration, bandwidth, and connection parameters
for iPerf3-based traffic generation.

Profiles are loaded from config/network_config.yaml or can be used
as standalone dataclasses.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrafficProfile:
    """Configuration for a single traffic generation profile.

    Attributes:
        name: Human-readable profile name.
        protocol: Transport protocol ("TCP" or "UDP").
        duration_seconds: Test duration in seconds.
        bandwidth_mbps: Target bandwidth in Mbps (None = unlimited for TCP).
        server_port: iPerf3 server listening port.
        num_streams: Number of parallel streams (iPerf3 -P flag).
        reverse: If True, server sends to client (iPerf3 -R flag).
        json_output: If True, produce JSON output (iPerf3 -J flag).
        interval_seconds: Report interval in seconds (iPerf3 -i flag).
    """

    name: str
    protocol: str = "TCP"
    duration_seconds: int = 60
    bandwidth_mbps: Optional[float] = None
    server_port: int = 5201
    num_streams: int = 1
    reverse: bool = False
    json_output: bool = True
    interval_seconds: float = 1.0

    def to_iperf_args(self) -> List[str]:
        """Convert profile to iPerf3 client command-line arguments.

        Returns:
            List of iPerf3 CLI arguments (excluding 'iperf3' and server IP).
        """
        args = [
            "-t", str(self.duration_seconds),
            "-p", str(self.server_port),
            "-P", str(self.num_streams),
            "-i", str(self.interval_seconds),
        ]

        if self.protocol.upper() == "UDP":
            args.append("-u")

        if self.bandwidth_mbps is not None:
            args.extend(["-b", f"{self.bandwidth_mbps}M"])

        if self.reverse:
            args.append("-R")

        if self.json_output:
            args.append("-J")

        return args


# ---------------------------------------------------------------
# Predefined Traffic Profiles
# ---------------------------------------------------------------

BASELINE_TCP = TrafficProfile(
    name="baseline_tcp",
    protocol="TCP",
    duration_seconds=30,
    bandwidth_mbps=50,
    server_port=5201,
    num_streams=1,
)

BASELINE_UDP = TrafficProfile(
    name="baseline_udp",
    protocol="UDP",
    duration_seconds=30,
    bandwidth_mbps=10,
    server_port=5202,
    num_streams=1,
)

LIGHT_TRAFFIC = TrafficProfile(
    name="light_traffic",
    protocol="TCP",
    duration_seconds=60,
    bandwidth_mbps=5,
    server_port=5201,
    num_streams=1,
)

HEAVY_TRAFFIC = TrafficProfile(
    name="heavy_traffic",
    protocol="TCP",
    duration_seconds=60,
    bandwidth_mbps=80,
    server_port=5201,
    num_streams=4,
)

MULTI_FLOW = TrafficProfile(
    name="multi_flow",
    protocol="TCP",
    duration_seconds=60,
    bandwidth_mbps=10,
    server_port=5201,
    num_streams=5,
)

SUSTAINED_LOAD = TrafficProfile(
    name="sustained_load",
    protocol="TCP",
    duration_seconds=300,
    bandwidth_mbps=30,
    server_port=5201,
    num_streams=2,
)


# Registry of all predefined profiles
PROFILES: Dict[str, TrafficProfile] = {
    "baseline_tcp": BASELINE_TCP,
    "baseline_udp": BASELINE_UDP,
    "light_traffic": LIGHT_TRAFFIC,
    "heavy_traffic": HEAVY_TRAFFIC,
    "multi_flow": MULTI_FLOW,
    "sustained_load": SUSTAINED_LOAD,
}


def get_profile(name: str) -> TrafficProfile:
    """Retrieve a named traffic profile.

    Args:
        name: Profile name (e.g., 'baseline_tcp', 'heavy_traffic').

    Returns:
        TrafficProfile instance.

    Raises:
        KeyError: If the profile name is not found.
    """
    if name not in PROFILES:
        available = ", ".join(PROFILES.keys())
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")
    return PROFILES[name]


def create_profile_from_config(config: Dict) -> TrafficProfile:
    """Create a TrafficProfile from a config dictionary.

    Args:
        config: Dictionary with profile parameters (from YAML config).

    Returns:
        TrafficProfile instance.
    """
    return TrafficProfile(
        name=config.get("name", "custom"),
        protocol=config.get("protocol", "TCP"),
        duration_seconds=config.get("duration_seconds", 60),
        bandwidth_mbps=config.get("bandwidth_mbps"),
        server_port=config.get("server_port", 5201),
        num_streams=config.get("num_concurrent_flows", 1),
    )
