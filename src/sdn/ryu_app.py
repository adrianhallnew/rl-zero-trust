"""Ryu SDN controller application for the RL-driven adaptive security system.

Implements an OpenFlow 1.3 L2 learning switch with:
- MAC address learning and forwarding
- Configurable flow entry timeouts (idle/hard)
- Priority-based flow table management for future policy enforcement
- Switch and host event logging
- Table-miss flow entry installation

This application runs alongside ryu.app.ofctl_rest for REST API access
to flow and port statistics.

Usage:
    ryu-manager src/sdn/ryu_app.py ryu.app.ofctl_rest --ofp-tcp-listen-port 6633
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ether_types, arp, ipv4
from ryu.lib import hub

import logging
import time

LOG = logging.getLogger(__name__)

# Flow entry priorities (higher = more important)
PRIORITY_TABLE_MISS = 0          # Catch-all: send to controller
PRIORITY_DEFAULT = 100           # Normal learned forwarding
PRIORITY_SECURITY = 200          # Security policy enforcement (Sprint 3+)
PRIORITY_EMERGENCY = 300         # Emergency block rules (Sprint 3+)

# Flow entry timeouts (seconds)
IDLE_TIMEOUT = 30                # Remove flow if idle for 30s
HARD_TIMEOUT = 300               # Remove flow after 5 minutes regardless

# Statistics polling interval
STATS_POLL_INTERVAL = 5          # Poll flow/port stats every 5 seconds


class ZeroTrustSwitch(app_manager.RyuApp):
    """OpenFlow 1.3 L2 learning switch for zero-trust security system.

    Features:
    - MAC learning table per switch (datapath)
    - Automatic flow entry installation for known destinations
    - Table-miss rule sends unknown packets to controller
    - Periodic flow and port statistics polling
    - Structured logging for all switch events

    Attributes:
        mac_to_port: Per-switch MAC address to port mapping.
        datapaths: Connected switch datapath objects.
        switch_stats: Latest flow/port statistics per switch.
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}       # {dpid: {mac: port}}
        self.datapaths = {}         # {dpid: datapath}
        self.switch_stats = {}      # {dpid: {"flows": [...], "ports": [...]}}
        self._monitor_thread = hub.spawn(self._stats_monitor)
        LOG.info("ZeroTrustSwitch initialized (OpenFlow 1.3)")

    # ---------------------------------------------------------------
    # Switch Connection Handling
    # ---------------------------------------------------------------

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle new switch connection — install table-miss flow entry.

        The table-miss entry matches all packets with the lowest priority
        and sends them to the controller for MAC learning.
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id

        self.datapaths[dpid] = datapath
        self.mac_to_port.setdefault(dpid, {})
        self.switch_stats[dpid] = {"flows": [], "ports": []}

        LOG.info(
            "Switch connected: dpid=%s (0x%x), n_buffers=%d, n_tables=%d",
            dpid, dpid, ev.msg.n_buffers, ev.msg.n_tables,
        )

        # Install table-miss flow entry (priority=0, match=all, action=output:CONTROLLER)
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER,
            ofproto.OFPCML_NO_BUFFER,
        )]
        self._add_flow(datapath, PRIORITY_TABLE_MISS, match, actions,
                        idle_timeout=0, hard_timeout=0)
        LOG.info("Table-miss entry installed on switch %s", dpid)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER])
    def state_change_handler(self, ev):
        """Track switch connect/disconnect events."""
        datapath = ev.datapath
        dpid = datapath.id

        if ev.state == MAIN_DISPATCHER:
            if dpid not in self.datapaths:
                self.datapaths[dpid] = datapath
                LOG.info("Switch %s entered MAIN state (ready)", dpid)
        else:
            if dpid in self.datapaths:
                del self.datapaths[dpid]
                LOG.info("Switch %s disconnected", dpid)

    # ---------------------------------------------------------------
    # Packet Processing (L2 Learning Switch)
    # ---------------------------------------------------------------

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle packets sent to controller (table-miss or explicit output).

        Learns the source MAC address and port, then either:
        - Forwards to the known destination port, or
        - Floods to all ports if the destination is unknown.
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None:
            return

        # Ignore LLDP and IPv6 multicast (reduce noise)
        if eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return
        if eth.dst[:5] == "33:33":
            return

        src_mac = eth.src
        dst_mac = eth.dst

        # Learn source MAC
        if src_mac not in self.mac_to_port[dpid]:
            LOG.info(
                "Host discovered: mac=%s on switch=%s port=%d",
                src_mac, dpid, in_port,
            )
        self.mac_to_port[dpid][src_mac] = in_port

        # Determine output port
        if dst_mac in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst_mac]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]

        # Install a flow entry to avoid future PacketIn for this dst
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
                in_port=in_port,
                eth_dst=dst_mac,
                eth_src=src_mac,
            )
            self._add_flow(
                datapath, PRIORITY_DEFAULT, match, actions,
                idle_timeout=IDLE_TIMEOUT,
                hard_timeout=HARD_TIMEOUT,
                buffer_id=msg.buffer_id if msg.buffer_id != ofproto.OFP_NO_BUFFER else None,
            )

            # If the switch buffered the packet, we're done
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                return

        # Send the packet out
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id if msg.buffer_id != ofproto.OFP_NO_BUFFER else ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None,
        )
        datapath.send_msg(out)

    # ---------------------------------------------------------------
    # Flow Entry Management
    # ---------------------------------------------------------------

    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0, buffer_id=None):
        """Install a flow entry on a switch.

        Args:
            datapath: Switch datapath object.
            priority: Flow entry priority (higher = matched first).
            match: OFPMatch object defining packet matching criteria.
            actions: List of OFPAction objects to apply.
            idle_timeout: Seconds before removal if no matching packets.
            hard_timeout: Seconds before removal regardless of activity.
            buffer_id: Optional buffer ID for buffered packets.
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions,
        )]

        kwargs = {
            "datapath": datapath,
            "priority": priority,
            "match": match,
            "instructions": inst,
            "idle_timeout": idle_timeout,
            "hard_timeout": hard_timeout,
        }
        if buffer_id is not None:
            kwargs["buffer_id"] = buffer_id

        mod = parser.OFPFlowMod(**kwargs)
        datapath.send_msg(mod)

    def add_security_flow(self, dpid, match_fields, actions, priority=PRIORITY_SECURITY):
        """Install a security policy flow entry (for RL agent integration).

        This method is called by the policy enforcer (Sprint 3+) to translate
        RL agent actions into OpenFlow rules.

        Args:
            dpid: Switch datapath ID.
            match_fields: Dictionary of match fields (e.g., eth_src, ipv4_dst).
            actions: List of action dictionaries.
            priority: Flow entry priority (default: PRIORITY_SECURITY).

        Returns:
            True if the flow was installed, False if the switch is not connected.
        """
        if dpid not in self.datapaths:
            LOG.warning("Cannot install security flow: switch %s not connected", dpid)
            return False

        datapath = self.datapaths[dpid]
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        match = parser.OFPMatch(**match_fields)

        ofp_actions = []
        for action in actions:
            if action.get("type") == "OUTPUT":
                ofp_actions.append(parser.OFPActionOutput(action["port"]))
            elif action.get("type") == "DROP":
                pass  # Empty action list = drop
            elif action.get("type") == "SET_QUEUE":
                ofp_actions.append(parser.OFPActionSetQueue(action["queue_id"]))

        self._add_flow(
            datapath, priority, match, ofp_actions,
            idle_timeout=IDLE_TIMEOUT,
            hard_timeout=HARD_TIMEOUT,
        )

        LOG.info(
            "Security flow installed: switch=%s, priority=%d, match=%s",
            dpid, priority, match_fields,
        )
        return True

    def delete_flows(self, dpid, match_fields=None):
        """Delete flow entries matching the given criteria.

        Args:
            dpid: Switch datapath ID.
            match_fields: Optional dict of match fields. If None, deletes all
                non-table-miss entries.

        Returns:
            True if the delete was sent, False if switch not connected.
        """
        if dpid not in self.datapaths:
            LOG.warning("Cannot delete flows: switch %s not connected", dpid)
            return False

        datapath = self.datapaths[dpid]
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        if match_fields:
            match = parser.OFPMatch(**match_fields)
        else:
            match = parser.OFPMatch()

        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=match,
        )
        datapath.send_msg(mod)

        LOG.info("Flow delete sent: switch=%s, match=%s", dpid, match_fields)
        return True

    # ---------------------------------------------------------------
    # Statistics Monitoring
    # ---------------------------------------------------------------

    def _stats_monitor(self):
        """Periodic statistics polling thread.

        Requests flow and port statistics from all connected switches
        at regular intervals for the RL agent's state observations.
        """
        while True:
            for dpid, datapath in list(self.datapaths.items()):
                self._request_flow_stats(datapath)
                self._request_port_stats(datapath)
            hub.sleep(STATS_POLL_INTERVAL)

    def _request_flow_stats(self, datapath):
        """Send a flow statistics request to a switch."""
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    def _request_port_stats(self, datapath):
        """Send a port statistics request to a switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply from a switch."""
        dpid = ev.msg.datapath.id
        flows = []
        for stat in ev.msg.body:
            flows.append({
                "table_id": stat.table_id,
                "priority": stat.priority,
                "match": str(stat.match),
                "packet_count": stat.packet_count,
                "byte_count": stat.byte_count,
                "duration_sec": stat.duration_sec,
                "duration_nsec": stat.duration_nsec,
                "idle_timeout": stat.idle_timeout,
                "hard_timeout": stat.hard_timeout,
            })
        self.switch_stats.setdefault(dpid, {})["flows"] = flows

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Handle port statistics reply from a switch."""
        dpid = ev.msg.datapath.id
        ports = []
        for stat in ev.msg.body:
            ports.append({
                "port_no": stat.port_no,
                "rx_packets": stat.rx_packets,
                "tx_packets": stat.tx_packets,
                "rx_bytes": stat.rx_bytes,
                "tx_bytes": stat.tx_bytes,
                "rx_dropped": stat.rx_dropped,
                "tx_dropped": stat.tx_dropped,
                "rx_errors": stat.rx_errors,
                "tx_errors": stat.tx_errors,
            })
        self.switch_stats.setdefault(dpid, {})["ports"] = ports

    # ---------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------

    def get_topology_info(self):
        """Return a summary of the current network topology state.

        Returns:
            Dictionary with connected switches, learned MACs, and flow counts.
        """
        info = {
            "connected_switches": list(self.datapaths.keys()),
            "num_switches": len(self.datapaths),
            "mac_tables": {},
            "flow_counts": {},
        }
        for dpid in self.datapaths:
            info["mac_tables"][dpid] = dict(self.mac_to_port.get(dpid, {}))
            stats = self.switch_stats.get(dpid, {})
            info["flow_counts"][dpid] = len(stats.get("flows", []))

        return info
