"""Flow statistics collector for the RL-driven adaptive security system.

Retrieves flow and port statistics from the Ryu SDN controller via its
REST API (ofctl_rest module). Data is formatted for consumption by the
RL agent's state processor.

Ryu REST API endpoints used:
    GET /stats/switches          — List connected switch DPIDs
    GET /stats/flow/<dpid>       — Flow entries for a switch
    GET /stats/port/<dpid>       — Port statistics for a switch
    GET /stats/desc/<dpid>       — Switch description

Reference: https://ryu.readthedocs.io/en/latest/app/ofctl_rest.html
"""

import logging
import time
from typing import Any, Dict, List, Optional

import requests
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_RYU_API_URL = "http://172.20.0.20:8080"
REQUEST_TIMEOUT = 10  # seconds


class StatsCollector:
    """Collects and processes network statistics from the Ryu REST API.

    Provides methods to query switch, flow, and port statistics, and
    aggregates them into feature vectors suitable for RL agent observation.

    Args:
        ryu_api_url: Base URL for the Ryu REST API.
        poll_interval: Seconds between automatic stat collection cycles.

    Example:
        >>> collector = StatsCollector("http://172.20.0.20:8080")
        >>> switches = collector.get_switches()
        >>> flow_stats = collector.get_flow_stats(switches[0])
        >>> state_vector = collector.get_state_vector()
    """

    def __init__(
        self,
        ryu_api_url: str = DEFAULT_RYU_API_URL,
        poll_interval: float = 5.0,
    ) -> None:
        self.api_url = ryu_api_url.rstrip("/")
        self.poll_interval = poll_interval
        self._last_stats = {}
        self._last_poll_time = 0.0
        self._cumulative_flows = 0
        logger.info("StatsCollector initialized (API: %s)", self.api_url)

    # ---------------------------------------------------------------
    # Raw API Methods
    # ---------------------------------------------------------------

    def _get(self, endpoint: str) -> Any:
        """Make a GET request to the Ryu REST API.

        Args:
            endpoint: API path (e.g., '/stats/switches').

        Returns:
            Parsed JSON response.

        Raises:
            ConnectionError: If the controller is unreachable.
            requests.HTTPError: If the API returns an error status.
        """
        url = f"{self.api_url}{endpoint}"
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            logger.error("Cannot connect to Ryu API at %s", url)
            raise ConnectionError(f"Ryu API unreachable at {self.api_url}")
        except requests.Timeout:
            logger.warning("Ryu API request timed out: %s — using cached data", url)
            return self._last_stats if self._last_stats else {}
        except requests.HTTPError as e:
            logger.error("Ryu API error: %s %s", e.response.status_code, url)
            raise

    def get_switches(self) -> List[int]:
        """Get list of connected switch datapath IDs.

        Returns:
            List of integer DPIDs for all connected switches.
        """
        dpids = self._get("/stats/switches")
        logger.debug("Connected switches: %s", dpids)
        return dpids

    def get_switch_desc(self, dpid: int) -> Dict[str, Any]:
        """Get description of a specific switch.

        Args:
            dpid: Switch datapath ID.

        Returns:
            Dictionary with switch description fields.
        """
        data = self._get(f"/stats/desc/{dpid}")
        return data.get(str(dpid), {})

    def get_flow_stats(self, dpid: int) -> List[Dict[str, Any]]:
        """Get flow entry statistics for a specific switch.

        Args:
            dpid: Switch datapath ID.

        Returns:
            List of flow entry dictionaries with packet/byte counts, duration, etc.
        """
        data = self._get(f"/stats/flow/{dpid}")
        flows = data.get(str(dpid), [])
        logger.debug("Switch %d: %d flow entries", dpid, len(flows))
        return flows

    def get_port_stats(self, dpid: int) -> List[Dict[str, Any]]:
        """Get port statistics for a specific switch.

        Args:
            dpid: Switch datapath ID.

        Returns:
            List of port statistics dictionaries with rx/tx packets, bytes,
            dropped, errors.
        """
        data = self._get(f"/stats/port/{dpid}")
        ports = data.get(str(dpid), [])
        logger.debug("Switch %d: %d port stats", dpid, len(ports))
        return ports

    # ---------------------------------------------------------------
    # Aggregated Statistics
    # ---------------------------------------------------------------

    def collect_all_stats(self) -> Dict[int, Dict[str, Any]]:
        """Collect flow and port statistics from all connected switches.

        Returns:
            Dictionary mapping dpid -> {"flows": [...], "ports": [...]}.
        """
        stats = {}
        try:
            switches = self.get_switches()
        except (ConnectionError, requests.RequestException) as e:
            logger.error("Failed to get switch list: %s", e)
            return stats

        for dpid in switches:
            try:
                flows = self.get_flow_stats(dpid)
                ports = self.get_port_stats(dpid)
                stats[dpid] = {"flows": flows, "ports": ports}
            except (ConnectionError, requests.RequestException) as e:
                logger.warning("Failed to get stats for switch %d: %s", dpid, e)
                stats[dpid] = {"flows": [], "ports": []}

        self._last_stats = stats
        self._last_poll_time = time.time()
        logger.debug("Collected stats from %d switches", len(stats))
        return stats

    def get_aggregate_flow_stats(
        self, dpid: int, flows: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Compute aggregate flow statistics for a switch.

        Args:
            dpid: Switch datapath ID.
            flows: Pre-fetched flow entries (avoids redundant API call).
                If None, fetches from the Ryu API.

        Returns:
            Dictionary with aggregated metrics:
                - total_flows: Number of active flow entries
                - total_packets: Sum of packet counts across all flows
                - total_bytes: Sum of byte counts across all flows
                - avg_duration: Mean flow duration in seconds
                - max_duration: Maximum flow duration
        """
        if flows is None:
            flows = self.get_flow_stats(dpid)

        if not flows:
            return {
                "total_flows": 0,
                "total_packets": 0,
                "total_bytes": 0,
                "avg_duration": 0.0,
                "max_duration": 0.0,
            }

        packet_counts = [f.get("packet_count", 0) for f in flows]
        byte_counts = [f.get("byte_count", 0) for f in flows]
        durations = [f.get("duration_sec", 0) for f in flows]

        return {
            "total_flows": len(flows),
            "total_packets": sum(packet_counts),
            "total_bytes": sum(byte_counts),
            "avg_duration": float(np.mean(durations)) if durations else 0.0,
            "max_duration": float(max(durations)) if durations else 0.0,
        }

    def get_aggregate_port_stats(
        self, dpid: int, ports: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, float]:
        """Compute aggregate port statistics for a switch.

        Args:
            dpid: Switch datapath ID.
            ports: Pre-fetched port stats (avoids redundant API call).
                If None, fetches from the Ryu API.

        Returns:
            Dictionary with aggregated port metrics:
                - total_rx_packets, total_tx_packets
                - total_rx_bytes, total_tx_bytes
                - total_rx_dropped, total_tx_dropped
                - total_rx_errors, total_tx_errors
        """
        if ports is None:
            ports = self.get_port_stats(dpid)

        if not ports:
            return {
                "total_rx_packets": 0, "total_tx_packets": 0,
                "total_rx_bytes": 0, "total_tx_bytes": 0,
                "total_rx_dropped": 0, "total_tx_dropped": 0,
                "total_rx_errors": 0, "total_tx_errors": 0,
            }

        return {
            "total_rx_packets": sum(p.get("rx_packets", 0) for p in ports),
            "total_tx_packets": sum(p.get("tx_packets", 0) for p in ports),
            "total_rx_bytes": sum(p.get("rx_bytes", 0) for p in ports),
            "total_tx_bytes": sum(p.get("tx_bytes", 0) for p in ports),
            "total_rx_dropped": sum(p.get("rx_dropped", 0) for p in ports),
            "total_tx_dropped": sum(p.get("tx_dropped", 0) for p in ports),
            "total_rx_errors": sum(p.get("rx_errors", 0) for p in ports),
            "total_tx_errors": sum(p.get("tx_errors", 0) for p in ports),
        }

    # ---------------------------------------------------------------
    # State Vector for RL Agent
    # ---------------------------------------------------------------

    def get_state_vector(self) -> np.ndarray:
        """Build a normalized state vector from current network statistics.

        The state vector is used as the observation for the RL agent.
        It includes per-switch aggregate flow and port statistics.

        State vector features per switch (13 features):
            [0] total_flows (normalized)
            [1] total_packets (log-scaled)
            [2] total_bytes (log-scaled)
            [3] avg_duration (normalized to [0,1] by max 300s)
            [4] total_rx_packets (log-scaled)
            [5] total_tx_packets (log-scaled)
            [6] total_rx_bytes (log-scaled)
            [7] total_tx_bytes (log-scaled)
            [8] total_rx_dropped (log-scaled)
            [9] total_tx_dropped (log-scaled)
            [10] total_rx_errors (log-scaled)
            [11] total_tx_errors (log-scaled)
            [12] connection_rate (flows / time since last poll)

        Total state size: 5 switches * 13 features = 65

        Returns:
            Numpy array of shape (65,) with float32 values.
        """
        all_stats = self.collect_all_stats()
        switches = sorted(all_stats.keys())

        # Ensure we always have 5 switches worth of features
        features_per_switch = 13
        num_switches = 5
        state = np.zeros(num_switches * features_per_switch, dtype=np.float32)

        for i, dpid in enumerate(switches[:num_switches]):
            sw = all_stats.get(dpid, {})
            flow_agg = self.get_aggregate_flow_stats(dpid, flows=sw.get("flows"))
            port_agg = self.get_aggregate_port_stats(dpid, ports=sw.get("ports"))

            offset = i * features_per_switch

            # Flow features
            state[offset + 0] = min(flow_agg["total_flows"] / 100.0, 1.0)
            state[offset + 1] = np.log1p(flow_agg["total_packets"])
            state[offset + 2] = np.log1p(flow_agg["total_bytes"])
            state[offset + 3] = min(flow_agg["avg_duration"] / 300.0, 1.0)

            # Port features
            state[offset + 4] = np.log1p(port_agg["total_rx_packets"])
            state[offset + 5] = np.log1p(port_agg["total_tx_packets"])
            state[offset + 6] = np.log1p(port_agg["total_rx_bytes"])
            state[offset + 7] = np.log1p(port_agg["total_tx_bytes"])
            state[offset + 8] = np.log1p(port_agg["total_rx_dropped"])
            state[offset + 9] = np.log1p(port_agg["total_tx_dropped"])
            state[offset + 10] = np.log1p(port_agg["total_rx_errors"])
            state[offset + 11] = np.log1p(port_agg["total_tx_errors"])

            # Connection rate (new flows per second since last poll)
            elapsed = time.time() - self._last_poll_time if self._last_poll_time > 0 else 1.0
            elapsed = max(elapsed, 0.1)
            state[offset + 12] = flow_agg["total_flows"] / elapsed

        return state

    def reset(self) -> None:
        """Reset cached stats and timing (call at the start of each episode)."""
        self._last_stats = {}
        self._last_poll_time = 0.0
        self._cumulative_flows = 0
        logger.debug("StatsCollector caches reset")

    def get_state_dim(self) -> int:
        """Return the dimensionality of the state vector.

        Returns:
            Integer size of the state observation vector.
        """
        return 5 * 13  # 5 switches, 13 features each

    # ---------------------------------------------------------------
    # Health Check
    # ---------------------------------------------------------------

    def is_controller_reachable(self) -> bool:
        """Check if the Ryu SDN controller REST API is reachable.

        Returns:
            True if the controller responds, False otherwise.
        """
        try:
            self.get_switches()
            return True
        except Exception:
            return False

    def wait_for_controller(self, timeout: float = 60.0, interval: float = 2.0) -> bool:
        """Wait for the Ryu controller to become available.

        Args:
            timeout: Maximum seconds to wait.
            interval: Seconds between retry attempts.

        Returns:
            True if controller became reachable within timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            if self.is_controller_reachable():
                logger.info("Ryu controller is reachable at %s", self.api_url)
                return True
            logger.debug("Waiting for Ryu controller at %s...", self.api_url)
            time.sleep(interval)

        logger.error("Ryu controller not reachable after %.0fs", timeout)
        return False
