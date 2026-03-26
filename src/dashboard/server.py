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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, Request
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
# Application state (shared with live_demo.py via DashboardState)
# ---------------------------------------------------------------------------


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

        # Session history for export
        self.events: List[Dict[str, Any]] = []

        # SSE subscribers (asyncio queues, one per connected browser tab)
        self.subscribers: List[asyncio.Queue] = []

        # Control signals (read by RL thread)
        self.requested_agent: Optional[str] = None
        self.requested_mode: Optional[str] = None
        self.requested_attack: Optional[str] = None
        self.requested_attack_intensity: float = 0.7
        self.requested_stop_attacks: bool = False
        self.requested_auto_scenario: bool = False
        self.requested_start: bool = False
        self.requested_stop: bool = False

    async def publish(self, event: Dict[str, Any]) -> None:
        """Push an event to all SSE subscribers."""
        self.last_event = event
        self.events.append(event)
        if len(self.events) > 1000:
            self.events = self.events[-1000:]
        dead: List[asyncio.Queue] = []
        for q in self.subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self.subscribers.remove(q)

    def publish_sync(self, event: Dict[str, Any]) -> None:
        """Thread-safe publish from the RL loop (non-async context)."""
        self.last_event = event
        self.events.append(event)
        if len(self.events) > 1000:
            self.events = self.events[-1000:]
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except (asyncio.QueueFull, Exception):
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

app = FastAPI(title="RL Zero-Trust Dashboard", docs_url=None, redoc_url=None)


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
                    # Keep-alive comment to prevent proxy/browser timeout
                    yield ": keepalive\n\n"
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
    """Download the current session as JSON."""
    return JSONResponse({
        "exported_at": time.time(),
        "agent": state.agent,
        "mode": state.mode,
        "total_steps": state.step,
        "events": state.events,
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
    chart_path = _CHARTS_DIR / filepath
    if not chart_path.exists() or not chart_path.is_file():
        return JSONResponse({"error": "Chart not found"}, status_code=404)
    # Security: ensure the resolved path is under _CHARTS_DIR
    try:
        chart_path.resolve().relative_to(_CHARTS_DIR.resolve())
    except ValueError:
        return JSONResponse({"error": "Invalid path"}, status_code=403)
    media = "image/png" if chart_path.suffix == ".png" else "image/svg+xml"
    return FileResponse(chart_path, media_type=media)


@app.get("/charts")
async def list_charts():
    """List available chart files."""
    charts = []
    if _CHARTS_DIR.exists():
        for p in sorted(_CHARTS_DIR.rglob("*.png")):
            rel = p.relative_to(_CHARTS_DIR).as_posix()
            charts.append(rel)
    return {"charts": charts}


@app.get("/flows/{dpid}")
async def get_flows(dpid: int):
    """Proxy to Ryu flow stats for the policy rules view."""
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
