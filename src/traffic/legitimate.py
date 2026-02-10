"""iPerf3 wrapper for legitimate baseline traffic generation.

Provides classes and functions to generate realistic network traffic
between Mininet hosts using iPerf3. Supports TCP and UDP profiles
with configurable bandwidth, duration, and parallel streams.

This module is used inside the Mininet container where hosts are
accessible via Mininet's Host.cmd() / Host.popen() interface.

Usage (inside Mininet container with Mininet Python API):
    >>> from src.traffic.legitimate import TrafficGenerator
    >>> from mininet.net import Mininet
    >>> gen = TrafficGenerator()
    >>> result = gen.run_iperf_test(net, "h1", "h2", "baseline_tcp")
    >>> print(f"Throughput: {result['throughput_mbps']:.2f} Mbps")

Usage (standalone subprocess mode):
    >>> gen = TrafficGenerator()
    >>> gen.start_server_subprocess("10.0.0.1", port=5201)
    >>> result = gen.run_client_subprocess("10.0.0.1", "10.0.0.2", "baseline_tcp")
"""

import json
import logging
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from src.traffic.profiles import TrafficProfile, get_profile, BASELINE_TCP

logger = logging.getLogger(__name__)

# Timeout buffer beyond test duration (seconds)
IPERF_TIMEOUT_BUFFER = 15


class IperfResult:
    """Parsed result from an iPerf3 test run.

    Attributes:
        raw: Raw JSON output from iPerf3.
        throughput_mbps: Achieved throughput in Mbps.
        retransmits: TCP retransmit count (TCP only).
        jitter_ms: Jitter in milliseconds (UDP only).
        lost_packets: Number of lost packets (UDP only).
        lost_percent: Packet loss percentage (UDP only).
        duration_seconds: Actual test duration.
        protocol: Protocol used ("TCP" or "UDP").
        success: Whether the test completed successfully.
        error: Error message if test failed.
    """

    def __init__(self, raw_json: Optional[Dict] = None, error: Optional[str] = None):
        self.raw = raw_json or {}
        self.success = error is None and bool(raw_json)
        self.error = error
        self.throughput_mbps = 0.0
        self.retransmits = 0
        self.jitter_ms = 0.0
        self.lost_packets = 0
        self.lost_percent = 0.0
        self.duration_seconds = 0.0
        self.protocol = "TCP"

        if self.success:
            self._parse()

    def _parse(self) -> None:
        """Parse iPerf3 JSON output into structured fields."""
        try:
            end = self.raw.get("end", {})

            # Determine protocol from test structure
            if "sum_sent" in end:
                self._parse_tcp(end)
            elif "sum" in end:
                self._parse_udp(end)
            elif "streams" in end:
                streams = end["streams"]
                if streams and "udp" in streams[0]:
                    self._parse_udp_stream(end)
                else:
                    self._parse_tcp_stream(end)

        except (KeyError, TypeError, IndexError) as e:
            logger.warning("Failed to parse iPerf3 result: %s", e)
            self.success = False
            self.error = f"Parse error: {e}"

    def _parse_tcp(self, end: Dict) -> None:
        """Parse TCP results from iPerf3 JSON."""
        self.protocol = "TCP"
        sent = end.get("sum_sent", {})
        self.throughput_mbps = sent.get("bits_per_second", 0) / 1e6
        self.retransmits = sent.get("retransmits", 0)
        self.duration_seconds = sent.get("seconds", 0)

    def _parse_tcp_stream(self, end: Dict) -> None:
        """Parse TCP stream results."""
        self.protocol = "TCP"
        streams = end.get("streams", [])
        if streams:
            sender = streams[0].get("sender", {})
            self.throughput_mbps = sender.get("bits_per_second", 0) / 1e6
            self.retransmits = sender.get("retransmits", 0)
            self.duration_seconds = sender.get("seconds", 0)

    def _parse_udp(self, end: Dict) -> None:
        """Parse UDP results from iPerf3 JSON."""
        self.protocol = "UDP"
        summary = end.get("sum", {})
        self.throughput_mbps = summary.get("bits_per_second", 0) / 1e6
        self.jitter_ms = summary.get("jitter_ms", 0)
        self.lost_packets = summary.get("lost_packets", 0)
        self.lost_percent = summary.get("lost_percent", 0)
        self.duration_seconds = summary.get("seconds", 0)

    def _parse_udp_stream(self, end: Dict) -> None:
        """Parse UDP stream results."""
        self.protocol = "UDP"
        streams = end.get("streams", [])
        if streams:
            udp = streams[0].get("udp", {})
            self.throughput_mbps = udp.get("bits_per_second", 0) / 1e6
            self.jitter_ms = udp.get("jitter_ms", 0)
            self.lost_packets = udp.get("lost_packets", 0)
            self.lost_percent = udp.get("lost_percent", 0)
            self.duration_seconds = udp.get("seconds", 0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary for logging/serialization.

        Returns:
            Dictionary with all result fields.
        """
        return {
            "success": self.success,
            "protocol": self.protocol,
            "throughput_mbps": round(self.throughput_mbps, 2),
            "retransmits": self.retransmits,
            "jitter_ms": round(self.jitter_ms, 3),
            "lost_packets": self.lost_packets,
            "lost_percent": round(self.lost_percent, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "error": self.error,
        }

    def __repr__(self) -> str:
        if self.success:
            return (
                f"IperfResult({self.protocol}, "
                f"throughput={self.throughput_mbps:.2f}Mbps, "
                f"duration={self.duration_seconds:.1f}s)"
            )
        return f"IperfResult(failed: {self.error})"


class TrafficGenerator:
    """Generates legitimate network traffic using iPerf3.

    Supports two modes of operation:
    1. Mininet mode: Uses host.cmd()/host.popen() for traffic between virtual hosts.
    2. Subprocess mode: Uses subprocess.Popen() for standalone operation.

    Args:
        default_profile: Default traffic profile name to use.
    """

    def __init__(self, default_profile: str = "baseline_tcp") -> None:
        self.default_profile = default_profile
        self._server_procs: Dict[str, subprocess.Popen] = {}
        logger.info("TrafficGenerator initialized (default_profile=%s)", default_profile)

    # ---------------------------------------------------------------
    # Mininet Mode (host.cmd / host.popen)
    # ---------------------------------------------------------------

    def run_iperf_test(
        self,
        net: Any,
        server_host: str,
        client_host: str,
        profile_name: Optional[str] = None,
    ) -> IperfResult:
        """Run an iPerf3 test between two Mininet hosts.

        Args:
            net: Mininet network instance.
            server_host: Server host name (e.g., 'h1').
            client_host: Client host name (e.g., 'h2').
            profile_name: Traffic profile name. If None, uses default.

        Returns:
            IperfResult with test metrics.
        """
        profile = get_profile(profile_name or self.default_profile)
        server = net.get(server_host)
        client = net.get(client_host)

        if server is None or client is None:
            return IperfResult(error=f"Host not found: {server_host} or {client_host}")

        server_ip = server.IP()
        logger.info(
            "Starting iPerf3 test: %s (%s) -> %s, profile=%s",
            client_host, server_host, server_ip, profile.name,
        )

        # Start iPerf3 server on the server host
        server_cmd = f"iperf3 -s -p {profile.server_port} -D --one-off"
        server.cmd(server_cmd)
        time.sleep(1)  # Wait for server to bind

        # Build client command
        client_args = profile.to_iperf_args()
        client_cmd = f"iperf3 -c {server_ip} {' '.join(client_args)}"

        logger.debug("Server cmd: %s", server_cmd)
        logger.debug("Client cmd: %s", client_cmd)

        # Run client and capture output
        try:
            output = client.cmd(client_cmd)

            if profile.json_output:
                result_json = json.loads(output)
                result = IperfResult(raw_json=result_json)
            else:
                result = IperfResult(error="Non-JSON output not supported for parsing")

        except json.JSONDecodeError as e:
            logger.error("Failed to parse iPerf3 JSON output: %s", e)
            logger.debug("Raw output: %s", output[:500])
            result = IperfResult(error=f"JSON parse error: {e}")
        except Exception as e:
            logger.error("iPerf3 test failed: %s", e)
            result = IperfResult(error=str(e))

        logger.info(
            "iPerf3 result (%s -> %s): %s",
            client_host, server_host, result,
        )
        return result

    def run_multi_flow_test(
        self,
        net: Any,
        host_pairs: List[Tuple[str, str]],
        profile_name: Optional[str] = None,
    ) -> List[IperfResult]:
        """Run iPerf3 tests between multiple host pairs sequentially.

        Args:
            net: Mininet network instance.
            host_pairs: List of (server_host, client_host) tuples.
            profile_name: Traffic profile to use for all tests.

        Returns:
            List of IperfResult objects, one per host pair.
        """
        results = []
        for server_host, client_host in host_pairs:
            result = self.run_iperf_test(net, server_host, client_host, profile_name)
            results.append(result)
        return results

    def start_background_traffic(
        self,
        net: Any,
        server_host: str,
        client_host: str,
        profile_name: Optional[str] = None,
    ) -> Optional[Any]:
        """Start background iPerf3 traffic between two hosts (non-blocking).

        Args:
            net: Mininet network instance.
            server_host: Server host name.
            client_host: Client host name.
            profile_name: Traffic profile name.

        Returns:
            Popen object for the client process, or None on failure.
        """
        profile = get_profile(profile_name or self.default_profile)
        server = net.get(server_host)
        client = net.get(client_host)

        if server is None or client is None:
            logger.error("Host not found: %s or %s", server_host, client_host)
            return None

        server_ip = server.IP()

        # Start server in daemon mode
        server.cmd(f"iperf3 -s -p {profile.server_port} -D")
        time.sleep(0.5)

        # Start client in background
        client_args = profile.to_iperf_args()
        client_cmd = f"iperf3 -c {server_ip} {' '.join(client_args)}"
        proc = client.popen(client_cmd, shell=True)

        logger.info(
            "Background traffic started: %s -> %s (profile=%s, pid=%d)",
            client_host, server_host, profile.name, proc.pid,
        )
        return proc

    # ---------------------------------------------------------------
    # Subprocess Mode (standalone, no Mininet)
    # ---------------------------------------------------------------

    def start_server_subprocess(
        self,
        bind_ip: str = "0.0.0.0",
        port: int = 5201,
    ) -> subprocess.Popen:
        """Start an iPerf3 server as a subprocess.

        Args:
            bind_ip: IP address to bind the server to.
            port: Port to listen on.

        Returns:
            Popen object for the server process.
        """
        cmd = ["iperf3", "-s", "-B", bind_ip, "-p", str(port), "--one-off"]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._server_procs[f"{bind_ip}:{port}"] = proc
        logger.info("iPerf3 server started on %s:%d (pid=%d)", bind_ip, port, proc.pid)
        return proc

    def run_client_subprocess(
        self,
        server_ip: str,
        profile_name: Optional[str] = None,
    ) -> IperfResult:
        """Run an iPerf3 client as a subprocess.

        Args:
            server_ip: IP address of the iPerf3 server.
            profile_name: Traffic profile name.

        Returns:
            IperfResult with test metrics.
        """
        profile = get_profile(profile_name or self.default_profile)
        client_args = profile.to_iperf_args()

        cmd = ["iperf3", "-c", server_ip] + client_args
        timeout = profile.duration_seconds + IPERF_TIMEOUT_BUFFER

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                return IperfResult(error=f"iperf3 exit code {result.returncode}: {result.stderr}")

            if profile.json_output:
                result_json = json.loads(result.stdout)
                return IperfResult(raw_json=result_json)
            else:
                return IperfResult(error="Non-JSON output not supported")

        except subprocess.TimeoutExpired:
            return IperfResult(error=f"iPerf3 timed out after {timeout}s")
        except json.JSONDecodeError as e:
            return IperfResult(error=f"JSON parse error: {e}")
        except FileNotFoundError:
            return IperfResult(error="iperf3 not found in PATH")

    def cleanup(self) -> None:
        """Terminate all running iPerf3 server processes."""
        for key, proc in self._server_procs.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
                logger.info("Stopped iPerf3 server: %s", key)
            except Exception as e:
                logger.warning("Failed to stop iPerf3 server %s: %s", key, e)
                proc.kill()
        self._server_procs.clear()


def generate_baseline_report(
    net: Any,
    num_pairs: int = 5,
    profile_name: str = "baseline_tcp",
) -> Dict[str, Any]:
    """Generate a baseline throughput report for the network.

    Tests traffic between random host pairs and computes aggregate statistics.

    Args:
        net: Mininet network instance.
        num_pairs: Number of host pairs to test.
        profile_name: Traffic profile to use.

    Returns:
        Dictionary with individual and aggregate results.
    """
    import random

    hosts = [h.name for h in net.hosts]
    if len(hosts) < 2:
        return {"error": "Need at least 2 hosts for baseline test"}

    gen = TrafficGenerator(default_profile=profile_name)
    results = []

    # Generate random host pairs
    pairs = []
    for _ in range(num_pairs):
        pair = random.sample(hosts, 2)
        pairs.append((pair[0], pair[1]))

    logger.info("Running baseline test with %d host pairs", num_pairs)

    for server, client in pairs:
        result = gen.run_iperf_test(net, server, client, profile_name)
        results.append({
            "server": server,
            "client": client,
            "result": result.to_dict(),
        })

    # Aggregate statistics
    throughputs = [r["result"]["throughput_mbps"] for r in results if r["result"]["success"]]

    report = {
        "profile": profile_name,
        "num_tests": len(results),
        "successful_tests": len(throughputs),
        "individual_results": results,
        "aggregate": {
            "mean_throughput_mbps": round(sum(throughputs) / len(throughputs), 2) if throughputs else 0,
            "min_throughput_mbps": round(min(throughputs), 2) if throughputs else 0,
            "max_throughput_mbps": round(max(throughputs), 2) if throughputs else 0,
        },
    }

    logger.info(
        "Baseline report: %d/%d tests passed, avg throughput=%.2f Mbps",
        report["successful_tests"],
        report["num_tests"],
        report["aggregate"]["mean_throughput_mbps"],
    )

    return report
