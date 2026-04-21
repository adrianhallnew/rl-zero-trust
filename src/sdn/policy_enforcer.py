"""Policy enforcer for translating RL actions to OpenFlow rules.

Translates discrete (DQN) and continuous (PPO) RL agent actions into
concrete OpenFlow flow modifications on the Ryu SDN controller via its
REST API (``/stats/flowentry/{add,delete}``).

Action mappings:
    DQN (discrete):
        0 = ALLOW   -- Permit traffic flow normally
        1 = BLOCK   -- Drop packets matching malicious flow
        2 = REROUTE -- Redirect traffic through alternative path
        3 = RATE_LIMIT -- Throttle suspicious traffic to 50% bandwidth

    PPO (continuous — Sprint 5):
        rate_limit_intensity: [0.0, 1.0]
        rerouting_weight: [0.0, 1.0]
        priority_adjustment: [-1.0, 1.0]
"""

import json
import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# DQN discrete action constants
ACTION_ALLOW = 0
ACTION_BLOCK = 1
ACTION_REROUTE = 2
ACTION_RATE_LIMIT = 3

ACTION_NAMES = {
    ACTION_ALLOW: "ALLOW",
    ACTION_BLOCK: "BLOCK",
    ACTION_REROUTE: "REROUTE",
    ACTION_RATE_LIMIT: "RATE_LIMIT",
}

# OpenFlow priority levels (must match ryu_app.py)
PRIORITY_SECURITY = 200
PRIORITY_EMERGENCY = 300

# REST API timeout
REQUEST_TIMEOUT = 10


class PolicyEnforcer:
    """Translates RL agent actions into OpenFlow rules.

    Interfaces with the Ryu SDN controller REST API to install, modify,
    and remove flow entries based on RL policy decisions.

    Args:
        ryu_api_url: Base URL for the Ryu REST API.
        rate_limit_bw_fraction: Bandwidth fraction for RATE_LIMIT action
            (0.5 = throttle to 50%).
    """

    def __init__(
        self,
        ryu_api_url: str = "http://172.20.0.20:8080",
        rate_limit_bw_fraction: float = 0.5,
    ) -> None:
        self.api_url = ryu_api_url.rstrip("/")
        self.rate_limit_fraction = rate_limit_bw_fraction
        self.policy_change_count = 0
        self._action_log: List[Dict[str, Any]] = []
        self._reroute_index: int = 0
        self._queue_validated: Dict[int, bool] = {}
        logger.info("PolicyEnforcer initialized (API: %s)", self.api_url)

    # ------------------------------------------------------------------
    # DQN Discrete Actions
    # ------------------------------------------------------------------

    # Default subnet mapping per switch DPID (tree topology)
    SWITCH_SUBNETS = {
        1: ("10.0.0.0", "255.255.0.0"),       # core — all traffic
        2: ("10.0.1.0", "255.255.255.0"),      # edge s2
        3: ("10.0.2.0", "255.255.255.0"),      # edge s3
        4: ("10.0.3.0", "255.255.255.0"),      # edge s4
        5: ("10.0.4.0", "255.255.255.0"),      # edge s5
    }

    def enforce_action(
        self,
        action: int,
        dpid: int,
        match_fields: Dict[str, Any],
    ) -> bool:
        """Enforce a discrete RL action on a specific switch.

        Translates the DQN action index into a concrete OpenFlow rule
        and pushes it to the Ryu controller via REST API.

        Args:
            action: DQN action index (0-3).
            dpid: Target switch datapath ID.
            match_fields: Flow match criteria (e.g. ``{"ipv4_src": "10.0.0.1"}``).

        Returns:
            True if enforcement succeeded, False otherwise.
        """
        # Ensure match fields are never empty (wildcard) — restrict to
        # at least dl_type + the switch's subnet if nothing is specified.
        if not match_fields:
            subnet = self.SWITCH_SUBNETS.get(dpid, ("10.0.0.0", "255.255.0.0"))
            match_fields = {
                "dl_type": 2048,  # 0x0800 = IPv4
                "nw_src": subnet[0],
                "nw_src_mask": subnet[1],
            }

        action_name = ACTION_NAMES.get(action, "UNKNOWN")
        logger.info(
            "Enforcing %s (id=%d) on switch %d, match=%s",
            action_name, action, dpid, match_fields,
        )

        success = False

        if action == ACTION_ALLOW:
            success = self._enforce_allow(dpid, match_fields)
        elif action == ACTION_BLOCK:
            success = self._enforce_block(dpid, match_fields)
        elif action == ACTION_REROUTE:
            success = self._enforce_reroute(dpid, match_fields)
        elif action == ACTION_RATE_LIMIT:
            success = self._enforce_rate_limit(dpid, match_fields)
        else:
            logger.error("Unknown action: %d", action)
            return False

        if success:
            self.policy_change_count += 1
            self._action_log.append({
                "action": action,
                "action_name": action_name,
                "dpid": dpid,
                "match_fields": match_fields,
            })
            if len(self._action_log) > 500:
                self._action_log = self._action_log[-500:]

        return success

    def _enforce_allow(self, dpid: int, match_fields: Dict[str, Any]) -> bool:
        """Remove restrictive rules and install a permissive forwarding rule.

        Deletes any security rules (BLOCK, RATE_LIMIT, REROUTE), then
        installs an explicit NORMAL output rule at lower priority to
        avoid falling through to table-miss (which would send every
        packet to the controller, causing CPU overload under load).
        """
        self._delete_flow(dpid, match_fields)
        flow_entry = self._build_flow_entry(
            dpid=dpid,
            match=match_fields,
            actions=[{"type": "OUTPUT", "port": "NORMAL"}],
            priority=PRIORITY_SECURITY - 10,
        )
        return self._add_flow(flow_entry)

    def _enforce_block(self, dpid: int, match_fields: Dict[str, Any]) -> bool:
        """Install a high-priority drop rule.

        An empty actions list in OpenFlow means DROP.
        """
        flow_entry = self._build_flow_entry(
            dpid=dpid,
            match=match_fields,
            actions=[],  # empty = DROP
            priority=PRIORITY_EMERGENCY,
        )
        return self._add_flow(flow_entry)

    def _enforce_reroute(self, dpid: int, match_fields: Dict[str, Any]) -> bool:
        """Install a flow entry that redirects traffic to an alternative port.

        In the tree topology, the core switch s1 (dpid=1) has ports
        connecting to edge switches s2-s5 (ports 1-4). Rerouting sends
        traffic out a different edge switch port to isolate the threat.
        """
        # Determine an alternative output port.  For the core switch,
        # cycle to the next port; for edge switches, send to the core.
        out_port = self._pick_reroute_port(dpid)

        flow_entry = self._build_flow_entry(
            dpid=dpid,
            match=match_fields,
            actions=[{"type": "OUTPUT", "port": out_port}],
            priority=PRIORITY_SECURITY,
        )
        return self._add_flow(flow_entry)

    def _enforce_rate_limit(self, dpid: int, match_fields: Dict[str, Any]) -> bool:
        """Install a flow entry that queues traffic for rate limiting.

        Uses OpenFlow SET_QUEUE to push matching traffic into a
        bandwidth-limited queue. Queue 1 is pre-configured for 50%
        throughput in the Mininet topology.

        If queue validation fails, falls back to REROUTE so the
        action still has a real effect on the network.
        """
        if not self._validate_queue(dpid, queue_id=1):
            logger.warning(
                "Queue 1 unavailable on switch %d — falling back to REROUTE",
                dpid,
            )
            return self._enforce_reroute(dpid, match_fields)

        flow_entry = self._build_flow_entry(
            dpid=dpid,
            match=match_fields,
            actions=[
                {"type": "SET_QUEUE", "queue_id": 1},
                {"type": "OUTPUT", "port": "NORMAL"},
            ],
            priority=PRIORITY_SECURITY,
        )
        return self._add_flow(flow_entry)

    def _validate_queue(self, dpid: int, queue_id: int = 1) -> bool:
        """Check if a queue is configured on a switch.

        Results are cached per-dpid to avoid repeated API calls.

        Args:
            dpid: Switch datapath ID.
            queue_id: Queue ID to check.

        Returns:
            True if the queue exists.
        """
        if dpid in self._queue_validated:
            return self._queue_validated[dpid]

        try:
            url = f"{self.api_url}/qos/queue/{dpid}"
            resp = requests.get(url, timeout=3)
            if resp.status_code == 200:
                queues = resp.json()
                valid = any(
                    q.get("queue_id") == queue_id
                    for q in (queues if isinstance(queues, list) else [])
                )
            else:
                # Ryu QoS REST API not available — assume queue exists
                # (common when using simple_switch_13 without qos module)
                valid = True
        except Exception:
            # Can't verify — assume queue exists to avoid unnecessary fallbacks
            valid = True

        self._queue_validated[dpid] = valid
        if not valid:
            logger.warning(
                "Queue %d not configured on switch %d", queue_id, dpid,
            )
        return valid

    # ------------------------------------------------------------------
    # REST API Helpers
    # ------------------------------------------------------------------

    def _build_flow_entry(
        self,
        dpid: int,
        match: Dict[str, Any],
        actions: List[Dict[str, Any]],
        priority: int = PRIORITY_SECURITY,
        idle_timeout: int = 30,
        hard_timeout: int = 300,
    ) -> Dict[str, Any]:
        """Build a Ryu REST API flow entry body.

        Args:
            dpid: Switch datapath ID.
            match: OFPMatch-style dictionary.
            actions: List of action dictionaries.
            priority: Flow entry priority.
            idle_timeout: Seconds of inactivity before removal.
            hard_timeout: Absolute seconds before removal.

        Returns:
            Dictionary suitable for POST to ``/stats/flowentry/add``.
        """
        return {
            "dpid": dpid,
            "table_id": 0,
            "priority": priority,
            "idle_timeout": idle_timeout,
            "hard_timeout": hard_timeout,
            "match": match,
            "actions": actions,
        }

    def _add_flow(self, flow_entry: Dict[str, Any]) -> bool:
        """POST a flow entry to the Ryu REST API.

        Args:
            flow_entry: Flow entry body dictionary.

        Returns:
            True if the request succeeded.
        """
        url = f"{self.api_url}/stats/flowentry/add"
        try:
            resp = requests.post(
                url,
                json=flow_entry,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            logger.debug("Flow added: dpid=%d", flow_entry["dpid"])
            return True
        except requests.RequestException as e:
            logger.error("Failed to add flow: %s", e)
            return False

    def _delete_flow(self, dpid: int, match_fields: Dict[str, Any]) -> bool:
        """DELETE a flow entry via the Ryu REST API.

        Args:
            dpid: Switch datapath ID.
            match_fields: Match criteria for the flow to delete.

        Returns:
            True if the request succeeded.
        """
        url = f"{self.api_url}/stats/flowentry/delete"
        body = {
            "dpid": dpid,
            "table_id": 0,
            "match": match_fields,
        }
        try:
            resp = requests.post(
                url,
                json=body,
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            logger.debug("Flow deleted: dpid=%d, match=%s", dpid, match_fields)
            return True
        except requests.RequestException as e:
            logger.error("Failed to delete flow: %s", e)
            return False

    def _pick_reroute_port(self, dpid: int) -> int:
        """Select an alternative output port for rerouting.

        For the core switch (dpid=1), cycle through edge ports 2-4
        to provide actual path diversity. For edge switches, send to
        the core (port 1).

        Args:
            dpid: Switch datapath ID.

        Returns:
            Alternative output port number.
        """
        if dpid == 1:
            # Core switch: cycle through edge ports for path diversity
            ports = [2, 3, 4]
            port = ports[self._reroute_index % len(ports)]
            self._reroute_index += 1
            return port
        # Edge switches: uplink to core is always port 1
        return 1

    # ------------------------------------------------------------------
    # Counters & Logging
    # ------------------------------------------------------------------

    def reset_policy_count(self) -> None:
        """Reset the policy change counter (called at each RL step)."""
        self.policy_change_count = 0

    def get_policy_changes(self) -> int:
        """Get the number of policy changes in the current step."""
        return self.policy_change_count

    def get_action_log(self) -> List[Dict[str, Any]]:
        """Return the full action log for analysis."""
        return list(self._action_log)

    def clear_action_log(self) -> None:
        """Clear the action log."""
        self._action_log.clear()
