"""Policy enforcer for translating RL actions to OpenFlow rules.

Translates discrete (DQN) and continuous (PPO) RL agent actions into
concrete OpenFlow flow modifications on the Ryu SDN controller.

Action mappings:
    DQN (discrete):
        0 = ALLOW   — Permit traffic flow normally
        1 = BLOCK   — Drop packets matching malicious flow
        2 = REROUTE — Redirect traffic through alternative path
        3 = RATE_LIMIT — Throttle suspicious traffic to 50% bandwidth

    PPO (continuous):
        rate_limit_intensity: [0.0, 1.0]
        rerouting_weight: [0.0, 1.0]
        priority_adjustment: [-1.0, 1.0]

Full implementation: Sprint 3 (DQN) and Sprint 5 (PPO).
"""

import logging
from typing import Any, Dict, List, Optional

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


class PolicyEnforcer:
    """Translates RL agent actions into OpenFlow rules.

    Interfaces with the Ryu SDN controller to install, modify, and
    remove flow entries based on RL policy decisions.

    Args:
        ryu_api_url: Base URL for the Ryu REST API.

    Note:
        Full implementation in Sprint 3. This is a structural stub
        providing the interface that the RL environment will use.
    """

    def __init__(self, ryu_api_url: str = "http://172.20.0.20:8080") -> None:
        self.api_url = ryu_api_url.rstrip("/")
        self.policy_change_count = 0
        logger.info("PolicyEnforcer initialized (API: %s)", self.api_url)

    def enforce_action(
        self,
        action: int,
        dpid: int,
        match_fields: Dict[str, Any],
    ) -> bool:
        """Enforce a discrete RL action on a specific switch.

        Args:
            action: DQN action index (0-3).
            dpid: Target switch datapath ID.
            match_fields: Flow match criteria.

        Returns:
            True if enforcement succeeded.
        """
        action_name = ACTION_NAMES.get(action, "UNKNOWN")
        logger.info(
            "Enforcing action %s (id=%d) on switch %d, match=%s",
            action_name, action, dpid, match_fields,
        )
        self.policy_change_count += 1

        # Stub: full implementation in Sprint 3
        raise NotImplementedError("PolicyEnforcer.enforce_action — Sprint 3")

    def reset_policy_count(self) -> None:
        """Reset the policy change counter (called at each RL step)."""
        self.policy_change_count = 0

    def get_policy_changes(self) -> int:
        """Get the number of policy changes in the current step.

        Returns:
            Number of flow modifications made this step.
        """
        return self.policy_change_count
