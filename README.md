# RL-Zero-Trust

Reinforcement Learning-Driven Adaptive Security System for Zero-Trust Networks

## Overview

This system uses reinforcement learning (DQN and PPO) to automatically adjust security policies in zero-trust networks. It detects and responds to DDoS, port scanning, and traffic spoofing attacks in real-time using an SDN-based architecture secured by OpenZiti overlay networking.

**Author:** Adrian David Justin Hall (TP075220)
**Supervisor:** Mr. Shahab Alizadeh
**Institution:** Asia Pacific University of Technology and Innovation
**Programme:** B.Sc. (Hons) Information Technology

## Architecture

```
Network Traffic --> Mininet Virtual Switches --> Ryu SDN Controller (OpenFlow 1.3)
                                                        |
                                                        v
                                              RL Agent (DQN / PPO)
                                                        |
                                                        v
                                              Policy Update --> Flow Entry Installation
                                                        |
                                                        v
                                              Reward Calculation --> Training Loop
```

All inter-component communication is secured via OpenZiti (mTLS, identity-based routing, dark services).

## Interactive Dashboard

A browser-based dashboard at `http://localhost:5000` provides real-time monitoring and control:

- **Live Monitor** — D3.js force-directed topology, reward/action timelines, sparklines, confusion matrix, attack controls, agent switching (DQN/PPO mid-demo)
- **Training Results** — Pre-computed charts, DQN vs PPO comparison table, summary KPIs
- **System Overview** — Architecture diagram, reward formula, agent configs, live flow rules

```bash
# Launch dashboard + RL loop (live mode, requires Docker)
python -m scripts.live_demo --agent dqn --steps 200

# Launch in simulation mode (no Docker needed)
python -m scripts.live_demo --agent dqn --mode sim --steps 200
```

All frontend assets (Tailwind, DaisyUI, Chart.js, D3.js, FontAwesome, JetBrains Mono) are bundled locally in `src/dashboard/static/vendor/` — no internet required.

## Prerequisites

- **OS:** Windows 10/11 with WSL2 enabled
- **Docker:** Docker Desktop 4.x with WSL2 backend
- **RAM:** 16 GB minimum (11 GB allocated to containers)
- **CPU:** 4+ cores recommended (no GPU required)
- **Storage:** 10 GB free disk space

## Quick Start

### 1. Clone and Configure

```bash
git clone <repository-url>
cd rl-zero-trust

# Create environment file from template
cp .env.example .env
# Edit .env and set a strong ZITI_PWD
```

### 2. Build and Start Services

```bash
# Build all container images
docker compose build

# Start all services (startup order is handled automatically)
docker compose up -d

# Verify all containers are running
docker compose ps

# Check resource allocation
docker stats --no-stream
```

### 3. Verify Services

```bash
# Check Ryu REST API
curl http://localhost:8080/stats/switches

# Check container logs
docker compose logs ryu-controller
docker compose logs mininet
docker compose logs rl-agent
docker compose logs openziti-controller
```

### 4. Stop Services

```bash
docker compose down

# Remove volumes (full cleanup)
docker compose down -v
```

## Container Services

| Service | Container | Port(s) | Memory | Purpose |
|---------|-----------|---------|--------|---------|
| OpenZiti | openziti-controller | 1280, 3022, 6262 | 1 GB | Zero-trust overlay (mTLS, identities) |
| Ryu | ryu-controller | 6633, 8080 | 2 GB | SDN controller (OpenFlow 1.3 + REST API) |
| Mininet | mininet | - | 4 GB | Network simulation (5 switches, 15 hosts) |
| RL Agent | rl-agent | - | 4 GB | DQN/PPO training and inference |

## Project Structure

```
rl-zero-trust/
├── docker-compose.yml          # Service orchestration
├── requirements.txt            # Python dependencies (pinned)
├── config/                     # YAML configuration files
│   ├── dqn_config.yaml         # DQN hyperparameters
│   ├── ppo_config.yaml         # PPO hyperparameters
│   ├── network_config.yaml     # Topology & traffic profiles
│   └── openziti_config.yaml    # Zero-trust identities & policies
├── docker/                     # Dockerfiles per service
│   ├── mininet/Dockerfile
│   ├── ryu/Dockerfile
│   ├── openziti/Dockerfile
│   └── rl_agent/Dockerfile
├── src/                        # Source code
│   ├── environment/            # Gym-compatible RL environment
│   ├── agents/                 # DQN and PPO implementations
│   ├── sdn/                    # Stats collector & policy enforcer
│   ├── dashboard/              # FastAPI server + single-file HTML dashboard
│   │   ├── server.py           # SSE streaming, 15 REST endpoints
│   │   └── static/             # index.html + vendor/ (offline assets)
│   ├── attacks/                # Attack simulation (DDoS, scan, spoof)
│   ├── traffic/                # Legitimate traffic generation
│   ├── zero_trust/             # OpenZiti SDK wrapper
│   └── utils/                  # Logging, config, metrics, visualization
├── scripts/                    # Training, evaluation, and live demo entrypoints
│   └── live_demo.py            # Dashboard + RL loop launcher
├── tests/                      # 306 automated tests (unit, integration, stress)
├── notebooks/                  # Jupyter notebooks for analysis
├── results/                    # Training outputs and charts (generated, gitignored)
├── checkpoints/                # Model checkpoints (generated, gitignored)
├── captures/                   # Packet captures (generated, gitignored)
└── docs/                       # Architecture and sprint documentation
```

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Network Simulation | Mininet | Latest (Ubuntu 22.04 apt) |
| SDN Controller | Ryu | 4.34 (Python 3.9) |
| Zero-Trust Overlay | OpenZiti | Latest |
| RL Framework | TensorFlow (CPU) | 2.15.0 |
| Numerical Computing | NumPy | 1.26.4 |
| Containerization | Docker + Compose | Latest |
| Traffic Generation | iPerf3 + Scapy | 2.5.0 |

## Development Sprints

| Sprint | Weeks | Focus |
|--------|-------|-------|
| 1 | 1-2 | Environment Setup & Architecture Design |
| 2 | 3-4 | Network Simulation & SDN Controller Integration |
| 3 | 5-6 | DQN Agent Implementation |
| 4 | 7-8 | Attack Scenarios & DQN Training |
| 5 | 9-10 | PPO Agent Implementation |
| 6 | 11-12 | Zero-Trust Integration & PPO Training |
| 7 | 13-14 | System Evaluation & Documentation |
| 8 | 15-16 | Docker Stack Validation |
| 9 | 17-18 | Live Mode RL Loop |
| 10 | 19-20 | Attack Integration |
| 11 | 21-22 | Interactive Dashboard + 45-item fix audit |

## Testing

```bash
# Run full test suite (306 tests)
python -m pytest tests/ -v

# Run specific suite
python -m pytest tests/test_phase_a.py -v
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| test_dqn.py | 18 | Network, action selection, epsilon, save/load, soft update, local RNG |
| test_ppo.py | 36 | Actor/critic, GAE, clipping, explained variance, buffer overflow, local RNG |
| test_environment.py | 19 | Gym spaces, reset, step, truncation, simulation |
| test_attacks.py | 45 | DDoS, portscan, spoofing, orchestrator |
| test_integration.py | 13 | E2E episodes, evaluation, config loading |
| test_dashboard_server.py | 22 | Endpoints, SSE subscribe, memory cap, thread safety |
| test_dashboard_integration.py | 22 | Agent/mode switch, attack flow, control lifecycle |
| test_attack_scheduler.py | 16 | Timing, expiry, manual fire, auto-schedule, process reaping |
| test_policy_enforcer.py | 11 | Flow rule REST API, error handling, action log cap |
| test_stats_collector.py | 11 | Normalization, double-fetch elimination, timeout cache, reset |
| test_stress.py | 7 | Concurrent publish, subscribe churn, rapid cycling |
| test_live_demo_cli.py | 25 | CLI arg parsing, constants, bias validation, Docker health |
| test_chart_generation.py | 13 | Chart helpers, CSV/JSON loading, SVG output |
| test_openziti_client.py | 23 | Zero-trust client, access control, HTTP methods |
| test_replay_buffer.py | 12 | Circular buffer, sampling, overflow |
| test_reward.py | 13 | Reward components, composite bounds |

## Troubleshooting

### Containers fail to start
```bash
# Check Docker Desktop is running and WSL2 is enabled
docker info

# Rebuild from scratch
docker compose build --no-cache
docker compose up -d
```

### Mininet container exits immediately
The Mininet container requires privileged mode for network namespaces. Ensure `privileged: true` is set in docker-compose.yml.

### Ryu REST API not responding
```bash
# Check controller logs
docker compose logs ryu-controller

# Verify OpenFlow port isn't in use
netstat -tlnp | grep 6633
```

### Out of memory
```bash
# Check current allocation
docker stats --no-stream

# Reduce container limits in docker-compose.yml if needed
```

### Dashboard not loading assets
All vendor assets are bundled locally. If the dashboard shows unstyled content, verify `src/dashboard/static/vendor/` exists and contains `js/`, `css/`, `webfonts/`, and `fonts/` directories.