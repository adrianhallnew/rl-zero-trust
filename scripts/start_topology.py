"""Start the Mininet topology inside the mininet container and keep it alive.

Runs inside the mininet container via:
    docker exec mininet python3 /app/scripts/start_topology.py

Responsibilities:
    1. Import and instantiate ZeroTrustTopology
    2. Start Mininet with RemoteController pointing to RYU_CONTROLLER_IP:6633
    3. Wait for all 5 switches to register with Ryu (poll /stats/switches)
    4. Launch iPerf3 baseline traffic between host pairs to populate flow tables
    5. Log TOPOLOGY_READY when Ryu sees all 5 switches
    6. Keep running (Mininet must stay alive for the demo)
"""

import logging
import os
import signal
import subprocess
import sys
import time

import requests

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mininet.log import setLogLevel

from src.sdn.topology import create_network, start_network, stop_network

logger = logging.getLogger("start_topology")
# Ensure our logger has a handler even if Mininet's import configured the root logger
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False  # Prevent duplicate messages from root logger

# Configuration from environment
RYU_CONTROLLER_IP = os.environ.get("RYU_CONTROLLER_IP", "172.20.0.20")
RYU_CONTROLLER_PORT = int(os.environ.get("RYU_CONTROLLER_PORT", "6633"))
RYU_API_URL = os.environ.get(
    "RYU_API_URL", f"http://{RYU_CONTROLLER_IP}:8080"
)

NUM_EXPECTED_SWITCHES = 5
SWITCH_WAIT_TIMEOUT = 120  # seconds
SWITCH_POLL_INTERVAL = 3  # seconds

# iPerf3 baseline traffic settings
IPERF_DURATION = 5  # seconds per pair
IPERF_BANDWIDTH = "10M"  # 10 Mbps baseline


def wait_for_ryu_switches(
    api_url: str,
    expected: int,
    timeout: int = SWITCH_WAIT_TIMEOUT,
    interval: int = SWITCH_POLL_INTERVAL,
) -> bool:
    """Poll Ryu REST API until all expected switches are registered.

    Args:
        api_url: Ryu REST API base URL (e.g. http://172.20.0.20:8080).
        expected: Number of switches expected.
        timeout: Maximum wait time in seconds.
        interval: Polling interval in seconds.

    Returns:
        True if all switches registered within timeout, False otherwise.
    """
    endpoint = f"{api_url}/stats/switches"
    deadline = time.time() + timeout
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        try:
            resp = requests.get(endpoint, timeout=5)
            resp.raise_for_status()
            switches = resp.json()
            count = len(switches)
            logger.info(
                "[Attempt %d] Ryu reports %d/%d switches: %s",
                attempt, count, expected, switches,
            )
            if count >= expected:
                return True
        except requests.ConnectionError:
            logger.info(
                "[Attempt %d] Ryu not reachable at %s — retrying...",
                attempt, endpoint,
            )
        except Exception as e:
            logger.warning(
                "[Attempt %d] Error polling Ryu: %s", attempt, e,
            )
        time.sleep(interval)

    logger.error(
        "Timed out waiting for %d switches after %ds", expected, timeout,
    )
    return False


def verify_flow_tables(api_url: str) -> bool:
    """Verify that at least switch 1 has non-empty flow entries.

    Args:
        api_url: Ryu REST API base URL.

    Returns:
        True if flow entries exist on switch 1.
    """
    try:
        resp = requests.get(f"{api_url}/stats/flow/1", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        # Response is {"1": [flow_entries...]}
        flows = data.get("1", data.get(1, []))
        flow_count = len(flows)
        logger.info("Switch 1 has %d flow entries", flow_count)
        return flow_count > 0
    except Exception as e:
        logger.warning("Could not verify flow tables: %s", e)
        return False


def launch_iperf_baseline(net) -> None:
    """Launch iPerf3 baseline traffic between host pairs to populate flow tables.

    Runs short iPerf3 sessions between selected host pairs so that
    the Ryu controller learns MAC addresses and installs forwarding rules.

    Args:
        net: Running Mininet network instance.
    """
    # Host pairs: one pair per edge switch to ensure all switches get traffic
    pairs = [
        ("h1", "h4"),   # s1 <-> s2
        ("h1", "h7"),   # s1 <-> s3
        ("h1", "h10"),  # s1 <-> s4
        ("h1", "h13"),  # s1 <-> s5
        ("h4", "h7"),   # s2 <-> s3 (via s1)
    ]

    logger.info("Starting iPerf3 baseline traffic (%d pairs)...", len(pairs))

    for server_name, client_name in pairs:
        server = net.get(server_name)
        client = net.get(client_name)

        if server is None or client is None:
            logger.warning(
                "Could not find hosts %s/%s — skipping", server_name, client_name,
            )
            continue

        server_ip = server.IP()

        # Start iPerf3 server in background
        server.cmd(f"iperf3 -s -D -p 5201 --one-off &")
        time.sleep(0.5)

        # Run client
        logger.info(
            "  iPerf3: %s (%s) -> %s (%s) for %ds at %s",
            client_name, client.IP(), server_name, server_ip,
            IPERF_DURATION, IPERF_BANDWIDTH,
        )
        result = client.cmd(
            f"iperf3 -c {server_ip} -p 5201 -t {IPERF_DURATION} "
            f"-b {IPERF_BANDWIDTH} --connect-timeout 5000 2>&1"
        )

        if "error" in result.lower() or "refused" in result.lower():
            logger.warning("  iPerf3 %s->%s had issues: %s", client_name, server_name, result.strip()[:200])
        else:
            logger.info("  iPerf3 %s->%s completed OK", client_name, server_name)

        # Clean up server
        server.cmd("kill %iperf3 2>/dev/null; wait 2>/dev/null")
        time.sleep(0.3)

    logger.info("Baseline traffic generation complete.")


def main() -> None:
    """Start the topology, wait for Ryu, generate baseline traffic, and keep alive."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    setLogLevel("info")

    logger.info("=" * 60)
    logger.info("SPRINT 8 — Start Topology Script")
    logger.info("=" * 60)
    logger.info("Controller: %s:%d", RYU_CONTROLLER_IP, RYU_CONTROLLER_PORT)
    logger.info("Ryu API:    %s", RYU_API_URL)
    logger.info("Expected switches: %d", NUM_EXPECTED_SWITCHES)

    # Step 1: Create the Mininet network
    logger.info("")
    logger.info("Step 1: Creating Mininet network...")
    net = create_network(
        controller_ip=RYU_CONTROLLER_IP,
        controller_port=RYU_CONTROLLER_PORT,
    )

    # Graceful shutdown handler
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            logger.info("Shutdown signal received — stopping network...")
            stop_network(net)
            sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Step 2: Start the network (connects OVS switches to Ryu)
        logger.info("")
        logger.info("Step 2: Starting Mininet network...")
        start_network(net)
        logger.info("Mininet started with %d switches, %d hosts",
                     len(net.switches), len(net.hosts))

        # Step 3: Wait for Ryu to see all switches via REST API
        logger.info("")
        logger.info("Step 3: Waiting for Ryu to register all %d switches...",
                     NUM_EXPECTED_SWITCHES)
        if not wait_for_ryu_switches(RYU_API_URL, NUM_EXPECTED_SWITCHES):
            logger.error("TOPOLOGY_FAILED: Ryu did not see all switches")
            logger.error("Diagnostics:")
            # Print OVS status for debugging
            for sw in net.switches:
                out = sw.cmd("ovs-vsctl show")
                logger.error("  %s: %s", sw.name, out.strip()[:300])
            stop_network(net)
            sys.exit(1)

        logger.info("All %d switches registered with Ryu.", NUM_EXPECTED_SWITCHES)

        # Step 4: Quick connectivity check
        logger.info("")
        logger.info("Step 4: Running connectivity check (pingAll)...")
        loss = net.pingAll(timeout="5")
        if loss > 50:
            logger.warning("High packet loss: %.1f%% — some paths may not work", loss)
        else:
            logger.info("Connectivity check: %.1f%% loss", loss)

        # Step 5: Generate baseline traffic with iPerf3
        logger.info("")
        logger.info("Step 5: Generating iPerf3 baseline traffic...")
        launch_iperf_baseline(net)

        # Step 6: Verify flow tables are populated
        logger.info("")
        logger.info("Step 6: Verifying flow tables...")
        if verify_flow_tables(RYU_API_URL):
            logger.info("Flow tables verified — entries present on switch 1")
        else:
            logger.warning("Flow tables may be empty — baseline traffic may not have generated flows")

        # TOPOLOGY_READY
        logger.info("")
        logger.info("=" * 60)
        logger.info("TOPOLOGY_READY")
        logger.info("=" * 60)
        logger.info("Switches: %d | Hosts: %d | Packet loss: %.1f%%",
                     len(net.switches), len(net.hosts), loss)
        logger.info("Mininet is running. Press Ctrl+C or send SIGTERM to stop.")
        logger.info("")

        # Step 7: Keep alive — Mininet must stay running for the demo
        while not shutdown_requested:
            time.sleep(5)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error("Fatal error: %s", e, exc_info=True)
    finally:
        if not shutdown_requested:
            stop_network(net)


if __name__ == "__main__":
    main()
