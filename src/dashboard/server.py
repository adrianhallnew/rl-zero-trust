"""FastAPI dashboard server for the RL Zero-Trust live demo.

Serves the single-file web dashboard, streams live RL events via SSE,
and exposes control endpoints for attacks, agent switching, and mode
toggling.

Designed to run **in-process** with ``live_demo.py`` — the RL loop
runs in a background thread and publishes events to an asyncio queue
that this server drains into SSE connections.
"""

import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_STATIC_DIR = _HERE / "static"
_CHARTS_DIR = _PROJECT_ROOT / "results" / "charts"
_CONFIG_DIR = _PROJECT_ROOT / "config"
_SUMMARY_JSON = _PROJECT_ROOT / "results" / "experiments" / "summary.json"

RYU_API_URL = os.environ.get("RYU_API_URL", "http://localhost:8080")

# ---------------------------------------------------------------------------
# ZTA Baseline Rules (static, matches ryu_app.py ZeroTrustSwitch defaults)
# ---------------------------------------------------------------------------

def _build_zta_baseline() -> Dict[int, List[Dict[str, Any]]]:
    """Build the static ZTA baseline rules for all 5 switches.

    Mirrors what ryu_app.py installs:
      - Priority 0: table-miss → send to controller
      - Priority 1: default deny (DROP) — zero-trust posture
      - Priority 100: ARP flood for L2 discovery
      - Priority 100: learned forwarding rules (eth_src+eth_dst→output)
    """
    rules: Dict[int, List[Dict[str, Any]]] = {}

    # Topology: s1 (core) connects to s2-s5 (access) on ports 1-4
    # Each access switch connects to core on port 1, hosts on ports 2-4
    # 15 hosts: h1-h3→s2, h4-h6→s3, h7-h9→s4, h10-h12→s5, h13-h15→s2(extra)
    switch_ports = {
        1: {  # core: ports 1-4 → s2,s3,s4,s5
            "forwarding": [
                {"in_port": 1, "eth_dst": "00:00:00:00:00:01", "out_port": 1},
                {"in_port": 2, "eth_dst": "00:00:00:00:00:04", "out_port": 2},
                {"in_port": 3, "eth_dst": "00:00:00:00:00:07", "out_port": 3},
                {"in_port": 4, "eth_dst": "00:00:00:00:00:0a", "out_port": 4},
            ],
        },
        2: {  # access: port 1→core, ports 2-4→h1,h2,h3
            "forwarding": [
                {"in_port": 2, "eth_dst": "00:00:00:00:00:01", "out_port": 2},
                {"in_port": 3, "eth_dst": "00:00:00:00:00:02", "out_port": 3},
                {"in_port": 1, "eth_dst": "00:00:00:00:00:03", "out_port": 1},
            ],
        },
        3: {  # access: port 1→core, ports 2-4→h4,h5,h6
            "forwarding": [
                {"in_port": 2, "eth_dst": "00:00:00:00:00:04", "out_port": 2},
                {"in_port": 3, "eth_dst": "00:00:00:00:00:05", "out_port": 3},
                {"in_port": 1, "eth_dst": "00:00:00:00:00:06", "out_port": 1},
            ],
        },
        4: {  # access: port 1→core, ports 2-4→h7,h8,h9
            "forwarding": [
                {"in_port": 2, "eth_dst": "00:00:00:00:00:07", "out_port": 2},
                {"in_port": 3, "eth_dst": "00:00:00:00:00:08", "out_port": 3},
                {"in_port": 1, "eth_dst": "00:00:00:00:00:09", "out_port": 1},
            ],
        },
        5: {  # access: port 1→core, ports 2-4→h10,h11,h12
            "forwarding": [
                {"in_port": 2, "eth_dst": "00:00:00:00:00:0a", "out_port": 2},
                {"in_port": 3, "eth_dst": "00:00:00:00:00:0b", "out_port": 3},
                {"in_port": 1, "eth_dst": "00:00:00:00:00:0c", "out_port": 1},
            ],
        },
    }

    for dpid in range(1, 6):
        sw_rules = [
            {
                "priority": 0, "match": {},
                "actions": ["OUTPUT:CONTROLLER"],
                "purpose": "Table-miss — send unknown packets to controller",
            },
            {
                "priority": 1, "match": {},
                "actions": ["DROP"],
                "purpose": "Default deny — zero-trust baseline policy",
            },
            {
                "priority": 100, "match": {"dl_type": "0x0806"},
                "actions": ["FLOOD"],
                "purpose": "ARP broadcast for L2 host discovery",
            },
        ]
        for fwd in switch_ports[dpid]["forwarding"]:
            sw_rules.append({
                "priority": 100,
                "match": {
                    "in_port": fwd["in_port"],
                    "eth_dst": fwd["eth_dst"],
                },
                "actions": [f"OUTPUT:{fwd['out_port']}"],
                "purpose": "Learned L2 forwarding",
                "idle_timeout": 30,
                "hard_timeout": 300,
            })
        rules[dpid] = sw_rules

    return rules


ZTA_BASELINE_RULES: Dict[int, List[Dict[str, Any]]] = _build_zta_baseline()

# ---------------------------------------------------------------------------
# Application state (shared with live_demo.py via DashboardState)
# ---------------------------------------------------------------------------


@dataclass
class SessionMetrics:
    """Running metrics for one agent session (Item 9)."""

    agent: str
    started_at: float
    steps: int = 0
    total_reward: float = 0.0
    action_counts: Dict[int, int] = field(
        default_factory=lambda: {0: 0, 1: 0, 2: 0, 3: 0},
    )
    detection_rate_sum: float = 0.0
    fpr_sum: float = 0.0
    throughput_sum: float = 0.0
    latency_sum: float = 0.0

    @property
    def avg_reward(self) -> float:
        return self.total_reward / max(self.steps, 1)

    @property
    def avg_detection_rate(self) -> float:
        return self.detection_rate_sum / max(self.steps, 1)

    @property
    def avg_fpr(self) -> float:
        return self.fpr_sum / max(self.steps, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent,
            "started_at": self.started_at,
            "steps": self.steps,
            "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            "avg_detection_rate": self.avg_detection_rate,
            "avg_fpr": self.avg_fpr,
            "avg_throughput": self.throughput_sum / max(self.steps, 1),
            "avg_latency": self.latency_sum / max(self.steps, 1),
            "action_distribution": dict(self.action_counts),
        }

    def accumulate(self, event: Dict[str, Any]) -> None:
        """Accumulate a step event into running totals."""
        self.steps += 1
        self.total_reward += event.get("reward", 0.0)
        action = event.get("action", 0)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        metrics = event.get("metrics", {})
        self.detection_rate_sum += metrics.get("detection_rate", 0.0)
        self.fpr_sum += metrics.get("false_positive_rate", 0.0)
        self.throughput_sum += metrics.get("throughput_mbps", 0.0)
        self.latency_sum += metrics.get("latency_ms", 0.0)


class DashboardState:
    """Mutable shared state between the RL loop thread and the FastAPI server.

    All writes happen from the RL thread; all reads from the async server.
    Simple attributes are atomic on CPython (GIL), so no locks needed for
    individual field reads.
    """

    def __init__(self) -> None:
        # Current run info
        self.agent: str = "dqn"
        self.mode: str = "live"          # "live" or "sim"
        self.running: bool = False
        self.step: int = 0
        self.start_time: Optional[float] = None
        self.last_event: Optional[Dict[str, Any]] = None

        # Session history for export (protected by _lock)
        self.events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        # SSE subscribers (asyncio queues, one per connected browser tab)
        self.subscribers: List[asyncio.Queue] = []

        # Event loop reference for thread-safe publishing
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Control signals (read by RL thread)
        self.requested_agent: Optional[str] = None
        self.requested_mode: Optional[str] = None
        self.requested_attack: Optional[str] = None
        self.requested_attack_intensity: float = 0.7
        self.requested_stop_attacks: bool = False
        self.requested_auto_scenario: bool = False
        self.requested_start: bool = False
        self.requested_stop: bool = False

        # Session comparison memory (Item 9)
        self.agent_sessions: Dict[str, SessionMetrics] = {}
        self.current_session: Optional[SessionMetrics] = None

        # Baseline flow snapshot (Item 10)
        self.baseline_flows: Dict[int, List[Dict]] = {}
        self.baseline_captured: bool = False

        # RL-installed rules tracker (before/after policy comparison)
        self.rl_installed_rules: List[Dict[str, Any]] = []

    async def publish(self, event: Dict[str, Any]) -> None:
        """Push an event to all SSE subscribers."""
        self.last_event = event
        with self._lock:
            self.events.append(event)
            if len(self.events) > 1000:
                self.events = self.events[-1000:]
        dead: List[asyncio.Queue] = []
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    def publish_sync(self, event: Dict[str, Any]) -> None:
        """Thread-safe publish from the RL loop (non-async context).

        Uses ``loop.call_soon_threadsafe`` to schedule queue puts on
        the event loop thread, ensuring ``asyncio.Queue`` waiters are
        properly notified.
        """
        self.last_event = event
        with self._lock:
            self.events.append(event)
            if len(self.events) > 1000:
                self.events = self.events[-1000:]

        loop = self._loop
        for q in list(self.subscribers):
            try:
                if loop is not None and loop.is_running():
                    loop.call_soon_threadsafe(q.put_nowait, event)
                else:
                    q.put_nowait(event)
            except (asyncio.QueueFull, RuntimeError):
                pass

    def subscribe(self) -> asyncio.Queue:
        """Create a new SSE subscriber queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        try:
            self.subscribers.remove(q)
        except ValueError:
            pass


# Singleton — imported by live_demo.py
state = DashboardState()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Capture the running event loop for thread-safe publish_sync."""
    state._loop = asyncio.get_running_loop()
    try:
        yield
    except asyncio.CancelledError:
        pass
    finally:
        state._loop = None


app = FastAPI(title="RL Zero-Trust Dashboard", docs_url=None, redoc_url=None, lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/vendor", StaticFiles(directory=str(_STATIC_DIR / "vendor")), name="vendor")


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the dashboard HTML."""
    html_path = _STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Routes — SSE
# ---------------------------------------------------------------------------


@app.get("/events")
async def sse_stream(request: Request):
    """Server-Sent Events stream of live RL step data."""
    q = state.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            state.unsubscribe(q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Routes — Status & Control
# ---------------------------------------------------------------------------


@app.get("/status")
async def get_status():
    """Current system status."""
    elapsed = None
    if state.start_time is not None:
        elapsed = time.time() - state.start_time
    return {
        "agent": state.agent,
        "mode": state.mode,
        "running": state.running,
        "step": state.step,
        "elapsed_seconds": elapsed,
        "last_event": state.last_event,
    }


@app.post("/agent/{agent_type}")
async def switch_agent(agent_type: str):
    """Request an agent switch (DQN <-> PPO)."""
    if agent_type not in ("dqn", "ppo"):
        return JSONResponse(
            {"error": f"Unknown agent: {agent_type}"}, status_code=400,
        )
    state.requested_agent = agent_type
    await state.publish({
        "type": "toast",
        "level": "info",
        "message": f"Switching to {agent_type.upper()} agent...",
        "timestamp": time.time(),
    })
    return {"status": "ok", "requested_agent": agent_type}


@app.post("/mode/{mode}")
async def switch_mode(mode: str):
    """Switch between live and simulation mode."""
    if mode not in ("live", "sim"):
        return JSONResponse(
            {"error": f"Unknown mode: {mode}"}, status_code=400,
        )
    state.requested_mode = mode
    await state.publish({
        "type": "toast",
        "level": "info",
        "message": f"Switching to {'Live' if mode == 'live' else 'Simulation'} mode...",
        "timestamp": time.time(),
    })
    return {"status": "ok", "requested_mode": mode}


@app.post("/attack/{attack_type}")
async def trigger_attack(attack_type: str, intensity: float = 0.7):
    """Manually trigger an attack."""
    valid = ("ddos", "portscan", "spoofing", "mixed", "stop")
    if attack_type not in valid:
        return JSONResponse(
            {"error": f"Unknown attack type: {attack_type}"}, status_code=400,
        )
    if attack_type == "stop":
        state.requested_stop_attacks = True
        await state.publish({
            "type": "toast",
            "level": "warning",
            "message": "Stopping all attacks...",
            "timestamp": time.time(),
        })
        return {"status": "ok", "action": "stop_all"}

    state.requested_attack = attack_type
    state.requested_attack_intensity = max(0.1, min(1.0, intensity))
    await state.publish({
        "type": "toast",
        "level": "warning",
        "message": f"Launching {attack_type.upper()} attack (intensity: {intensity:.1f})...",
        "timestamp": time.time(),
    })
    return {"status": "ok", "attack": attack_type, "intensity": intensity}


@app.post("/scenario/auto")
async def trigger_auto_scenario():
    """Run the full auto attack scenario."""
    state.requested_auto_scenario = True
    await state.publish({
        "type": "toast",
        "level": "info",
        "message": "Starting auto scenario: DDoS \u2192 Scan \u2192 Spoof \u2192 Mixed \u2192 Clear",
        "timestamp": time.time(),
    })
    return {"status": "ok", "scenario": "auto"}


@app.post("/control/start")
async def start_loop():
    """Start the RL loop."""
    state.requested_start = True
    return {"status": "ok"}


@app.post("/control/stop")
async def stop_loop():
    """Stop the RL loop."""
    state.requested_stop = True
    await state.publish({
        "type": "toast",
        "level": "info",
        "message": "Stopping RL loop...",
        "timestamp": time.time(),
    })
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routes — Data
# ---------------------------------------------------------------------------


@app.get("/export")
async def export_session():
    """Download the current session as JSON with full context."""
    elapsed = None
    if state.start_time is not None:
        elapsed = time.time() - state.start_time

    with state._lock:
        events_copy = list(state.events)

    return JSONResponse({
        "exported_at": time.time(),
        "agent": state.agent,
        "mode": state.mode,
        "running": state.running,
        "total_steps": state.step,
        "start_time": state.start_time,
        "elapsed_seconds": elapsed,
        "last_event": state.last_event,
        "events": events_copy,
    })


@app.get("/config")
async def get_config():
    """Return current agent configurations from YAML."""
    configs = {}
    for name in ("dqn", "ppo"):
        cfg_path = _CONFIG_DIR / f"{name}_config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                configs[name] = yaml.safe_load(f)
    return configs


@app.get("/summary")
async def get_summary():
    """Return experiment summary from results/experiments/summary.json."""
    if _SUMMARY_JSON.exists():
        with open(_SUMMARY_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.get("/charts/{filepath:path}")
async def serve_chart(filepath: str):
    """Serve a chart image from results/charts/."""
    # Security: validate resolved path BEFORE any filesystem access
    chart_path = (_CHARTS_DIR / filepath).resolve()
    try:
        chart_path.relative_to(_CHARTS_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path"}, status_code=403)
    if not chart_path.exists() or not chart_path.is_file():
        return JSONResponse({"error": "Chart not found"}, status_code=404)
    media = "image/png" if chart_path.suffix == ".png" else "image/svg+xml"
    return FileResponse(chart_path, media_type=media)


@app.get("/charts")
async def list_charts():
    """List available chart files."""
    charts = []
    if _CHARTS_DIR.exists():
        for p in sorted(_CHARTS_DIR.rglob("*.svg")):
            rel = p.relative_to(_CHARTS_DIR).as_posix()
            charts.append(rel)
    return {"charts": charts}


def _flow_key(flow: Dict) -> str:
    """Create a hashable key from a flow's priority + match."""
    return f"{flow.get('priority', 0)}:{json.dumps(flow.get('match', {}), sort_keys=True)}"


@app.get("/flows/zta-baseline")
async def get_zta_baseline():
    """Return the static ZTA baseline rules (before RL)."""
    return {str(k): v for k, v in ZTA_BASELINE_RULES.items()}


@app.get("/flows/rl-rules")
async def get_rl_rules():
    """Return rules installed by the RL agent."""
    return {"rules": list(state.rl_installed_rules)}


@app.get("/flows/baseline")
async def get_baseline():
    """Return the captured baseline flow rules."""
    if not state.baseline_captured:
        return JSONResponse({"error": "No baseline captured yet"}, status_code=404)
    return {
        "baseline": {str(k): v for k, v in state.baseline_flows.items()},
    }


@app.post("/flows/baseline")
async def capture_baseline():
    """Capture current flow rules as baseline for diff comparison."""
    if state.mode == "sim":
        state.baseline_flows = {
            dpid: list(rules) for dpid, rules in ZTA_BASELINE_RULES.items()
        }
        state.baseline_captured = True
        total = sum(len(f) for f in state.baseline_flows.values())
        await state.publish({
            "type": "toast", "level": "success",
            "message": f"Baseline captured (sim): {total} ZTA rules across 5 switches",
            "timestamp": time.time(),
        })
        return {"status": "ok", "switches": 5, "total_rules": total}

    import requests as http_requests
    captured: Dict[int, list] = {}
    for dpid in range(1, 6):
        try:
            resp = http_requests.get(
                f"{RYU_API_URL}/stats/flow/{dpid}", timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            captured[dpid] = data.get(str(dpid), [])
        except Exception:
            captured[dpid] = []
    state.baseline_flows = captured
    state.baseline_captured = True
    total = sum(len(f) for f in captured.values())
    await state.publish({
        "type": "toast", "level": "success",
        "message": f"Baseline captured: {total} rules across 5 switches",
        "timestamp": time.time(),
    })
    return {"status": "ok", "switches": len(captured), "total_rules": total}


@app.get("/flows/diff/{dpid}")
async def get_flow_diff(dpid: int):
    """Compare current flows against baseline for a switch."""
    if not state.baseline_captured:
        return JSONResponse({"error": "No baseline captured"}, status_code=404)

    if state.mode == "sim":
        baseline = state.baseline_flows.get(dpid, [])
        rl_rules = [
            r for r in state.rl_installed_rules
            if r.get("dpid", 0) == dpid or r.get("dpid") is None
        ]
        return {
            "dpid": dpid,
            "added": rl_rules,
            "removed": [],
            "modified": [],
            "unchanged": len(baseline),
        }

    import requests as http_requests
    try:
        resp = http_requests.get(
            f"{RYU_API_URL}/stats/flow/{dpid}", timeout=5,
        )
        resp.raise_for_status()
        current_flows = resp.json().get(str(dpid), [])
    except Exception as exc:
        return JSONResponse(
            {"error": f"Cannot reach Ryu: {exc}"}, status_code=502,
        )

    baseline = state.baseline_flows.get(dpid, [])
    baseline_set = {_flow_key(f): f for f in baseline}
    current_set = {_flow_key(f): f for f in current_flows}

    added = [current_set[k] for k in current_set if k not in baseline_set]
    removed = [baseline_set[k] for k in baseline_set if k not in current_set]
    modified = []
    for k in current_set:
        if k in baseline_set and current_set[k] != baseline_set[k]:
            modified.append({"baseline": baseline_set[k], "current": current_set[k]})

    unchanged = len(current_set) - len(added) - len(modified)
    return {
        "dpid": dpid,
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": unchanged,
    }


@app.get("/flows/{dpid}")
async def get_flows(dpid: int):
    """Proxy to Ryu flow stats for the policy rules view."""
    if state.mode == "sim":
        return {str(dpid): ZTA_BASELINE_RULES.get(dpid, [])}

    import requests as http_requests
    try:
        resp = http_requests.get(
            f"{RYU_API_URL}/stats/flow/{dpid}", timeout=5,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return JSONResponse(
            {"error": f"Cannot reach Ryu: {exc}"}, status_code=502,
        )


# ---------------------------------------------------------------------------
# Session comparison (Item 9)
# ---------------------------------------------------------------------------

@app.get("/sessions")
async def get_sessions():
    """Return stored session metrics for DQN vs PPO comparison."""
    result = {}
    for name, session in state.agent_sessions.items():
        result[name] = session.to_dict()
    if state.current_session is not None:
        result[state.current_session.agent] = state.current_session.to_dict()
    return result


# ---------------------------------------------------------------------------
# Device inventory (Item 4)
# ---------------------------------------------------------------------------

@app.get("/devices")
async def get_devices():
    """Return device inventory for all switches and hosts."""
    devices = []
    for i in range(1, 6):
        role = "core" if i == 1 else "access"
        devices.append({
            "type": "switch", "id": f"s{i}", "dpid": i,
            "role": role, "protocol": "OpenFlow 1.3",
            "status": "active",
        })
    for i in range(1, 16):
        switch_idx = 1 if i <= 3 else 2 + (i - 4) // 3
        role = "server" if i <= 3 else "endpoint"
        devices.append({
            "type": "host", "id": f"h{i}",
            "ip": f"10.0.0.{i}",
            "mac": f"00:00:00:00:00:{i:02x}",
            "switch": f"s{switch_idx}",
            "role": role, "status": "active",
        })
    return {"devices": devices}


