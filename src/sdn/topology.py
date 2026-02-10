"""Custom Mininet topology for the RL-driven adaptive security system.

Implements a tree-like topology with 5 OVS switches and 15 hosts,
all using OpenFlow 1.3 with a remote Ryu SDN controller.

Topology Layout:
                        [s1] (core)
                      /  |   |   \\
                   [s2] [s3] [s4] [s5] (edge)
                   /|\\  /|\\  /|\\  /|\\
                  h   h  h  h  h  h  h  h  h  h  h  h
    s1: h1, h2, h3        (3 hosts directly on core)
    s2: h4, h5, h6        (3 hosts)
    s3: h7, h8, h9        (3 hosts)
    s4: h10, h11, h12     (3 hosts)
    s5: h13, h14, h15     (3 hosts)

Total: 5 switches, 15 hosts (3 per switch)

Link parameters are configurable via config/network_config.yaml.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSKernelSwitch, RemoteController
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel

logger = logging.getLogger(__name__)

# Default link parameters (overridden by config)
DEFAULT_BW_MBPS = 100
DEFAULT_DELAY_MS = 2
DEFAULT_LOSS_PERCENT = 0
DEFAULT_MAX_QUEUE = 1000

NUM_SWITCHES = 5
HOSTS_PER_SWITCH = 3
TOTAL_HOSTS = NUM_SWITCHES * HOSTS_PER_SWITCH
HOST_SUBNET = "10.0.0.0/24"


class ZeroTrustTopology(Topo):
    """Tree-like topology with 5 switches and 15 hosts for zero-trust security.

    Each switch hosts 3 hosts. Switch s1 acts as the core switch,
    connected to edge switches s2-s5 via trunk links. All switches
    use OpenFlow 1.3.

    Args:
        link_config: Dictionary with link parameters (bandwidth, delay, loss).
            If None, uses default values from network_config.yaml or constants.
    """

    def build(self, link_config: Optional[Dict[str, Any]] = None) -> None:
        """Build the network topology.

        Args:
            link_config: Optional dict with keys 'bandwidth_mbps', 'delay_ms',
                'loss_percent', 'max_queue_size' for link parameters.
        """
        if link_config is None:
            link_config = {}

        bw = link_config.get("bandwidth_mbps", DEFAULT_BW_MBPS)
        delay = f"{link_config.get('delay_ms', DEFAULT_DELAY_MS)}ms"
        loss = link_config.get("loss_percent", DEFAULT_LOSS_PERCENT)
        max_queue = link_config.get("max_queue_size", DEFAULT_MAX_QUEUE)

        link_opts = {
            "bw": bw,
            "delay": delay,
            "loss": loss,
            "max_queue_size": max_queue,
        }

        # --- Add switches (OpenFlow 1.3) ---
        switches = []
        for i in range(1, NUM_SWITCHES + 1):
            switch = self.addSwitch(
                f"s{i}",
                cls=OVSKernelSwitch,
                protocols="OpenFlow13",
            )
            switches.append(switch)
            logger.info("Added switch s%d", i)

        s1, s2, s3, s4, s5 = switches

        # --- Core-to-edge trunk links ---
        for edge_sw in [s2, s3, s4, s5]:
            self.addLink(s1, edge_sw, cls=TCLink, **link_opts)
            logger.info("Linked %s <-> %s (bw=%dMbps, delay=%s)", s1, edge_sw, bw, delay)

        # --- Add hosts and connect to switches ---
        host_idx = 1
        for switch in switches:
            for _ in range(HOSTS_PER_SWITCH):
                ip = f"10.0.0.{host_idx}/24"
                mac = f"00:00:00:00:00:{host_idx:02x}"
                host = self.addHost(
                    f"h{host_idx}",
                    ip=ip,
                    mac=mac,
                )
                self.addLink(host, switch, cls=TCLink, **link_opts)
                logger.info("Added h%d (ip=%s, mac=%s) -> %s", host_idx, ip, mac, switch)
                host_idx += 1

        logger.info(
            "Topology built: %d switches, %d hosts",
            NUM_SWITCHES,
            TOTAL_HOSTS,
        )


def create_network(
    controller_ip: str = "127.0.0.1",
    controller_port: int = 6633,
    link_config: Optional[Dict[str, Any]] = None,
) -> Mininet:
    """Create and return a Mininet network instance with the zero-trust topology.

    Args:
        controller_ip: IP address of the remote Ryu SDN controller.
        controller_port: OpenFlow port of the controller.
        link_config: Optional link parameter overrides.

    Returns:
        Configured Mininet network instance (not yet started).
    """
    topo = ZeroTrustTopology(link_config=link_config)

    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(
            name,
            ip=controller_ip,
            port=controller_port,
            protocols="OpenFlow13",
        ),
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=False,  # We set MACs explicitly
        autoStaticArp=False,
    )

    logger.info(
        "Network created with remote controller at %s:%d",
        controller_ip,
        controller_port,
    )
    return net


def start_network(net: Mininet) -> None:
    """Start the Mininet network and verify connectivity.

    Args:
        net: Mininet network instance to start.
    """
    import time as _time
    import subprocess as _sp

    logger.info("Starting network...")
    net.start()

    # Wait for switches to connect to the controller
    logger.info("Waiting for switches to connect to controller...")
    for switch in net.switches:
        if hasattr(switch, "waitConnected"):
            switch.waitConnected(timeout=30)
        else:
            # Fallback: poll OVS until controller connection is established
            deadline = _time.time() + 30
            while _time.time() < deadline:
                try:
                    out = _sp.check_output(
                        ["ovs-vsctl", "get", "controller", switch.name, "is_connected"],
                        stderr=_sp.DEVNULL,
                    ).decode().strip()
                    if out == "true":
                        break
                except _sp.CalledProcessError:
                    pass
                _time.sleep(1)
        logger.info("Switch %s connected (dpid=%s)", switch.name, switch.dpid)

    logger.info("All %d switches connected.", len(net.switches))
    logger.info("Network started with %d hosts.", len(net.hosts))


def stop_network(net: Mininet) -> None:
    """Stop and clean up the Mininet network.

    Args:
        net: Mininet network instance to stop.
    """
    logger.info("Stopping network...")
    net.stop()
    logger.info("Network stopped and cleaned up.")


def run_ping_test(net: Mininet) -> float:
    """Run a full ping test between all hosts and return packet loss percentage.

    Args:
        net: Running Mininet network instance.

    Returns:
        Packet loss percentage (0.0 = full connectivity).
    """
    logger.info("Running full connectivity test (pingAll)...")
    loss = net.pingAll(timeout="5")
    logger.info("Ping test complete. Packet loss: %.1f%%", loss)
    return loss


def _load_config() -> Dict[str, Any]:
    """Load network configuration from YAML file.

    Returns:
        Network configuration dictionary, or empty dict on failure.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from src.utils.config_loader import get_network_config
        return get_network_config()
    except Exception as e:
        logger.warning("Could not load network config: %s. Using defaults.", e)
        return {}


def main() -> None:
    """Entry point for running the topology standalone.

    Usage (inside Mininet container):
        python3 /app/src/sdn/topology.py [--cli]

    Environment variables:
        RYU_CONTROLLER_IP: Controller IP (default: 172.20.0.20)
        RYU_CONTROLLER_PORT: Controller port (default: 6633)
    """
    setLogLevel("info")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load configuration
    config = _load_config()
    link_config = config.get("links", {}).get("default", {})

    # Controller connection from environment or config
    controller_ip = os.environ.get(
        "RYU_CONTROLLER_IP",
        config.get("topology", {}).get("controller", {}).get("ip", "172.20.0.20"),
    )
    controller_port = int(os.environ.get(
        "RYU_CONTROLLER_PORT",
        config.get("topology", {}).get("controller", {}).get("port", 6633),
    ))

    logger.info("Controller target: %s:%d", controller_ip, controller_port)

    net = create_network(
        controller_ip=controller_ip,
        controller_port=controller_port,
        link_config=link_config,
    )

    try:
        start_network(net)

        # Run basic connectivity test
        loss = run_ping_test(net)
        if loss > 0:
            logger.warning("Connectivity issues detected: %.1f%% packet loss", loss)
        else:
            logger.info("Full connectivity verified (0%% loss)")

        # Drop into CLI if requested or by default
        if "--cli" in sys.argv or len(sys.argv) == 1:
            logger.info("Entering Mininet CLI. Type 'exit' to quit.")
            CLI(net)
        else:
            # Keep running for automated testing
            logger.info("Topology running. Press Ctrl+C to stop.")
            import signal
            signal.pause()

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stop_network(net)


if __name__ == "__main__":
    main()
