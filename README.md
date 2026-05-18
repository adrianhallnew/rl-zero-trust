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

Four Docker containers on a bridge network (172.20.0.0/24):

| Service | IP | Purpose |
|---------|----|---------|
| openziti-controller | 172.20.0.10 | Zero-trust overlay (mTLS, identity-based routing) |
| ryu-controller | 172.20.0.20 | SDN controller (OpenFlow 1.3 + REST API) |
| mininet | 172.20.0.30 | Network simulation (5 switches, 15 hosts) |
| rl-agent | 172.20.0.40 | DQN/PPO training and inference |

## Prerequisites

- Windows 10/11 with WSL2 enabled
- Docker Desktop 4.x with WSL2 backend
- Python 3.11 (for host-side scripts and dashboard)
- 16 GB RAM minimum (11 GB allocated to containers)
- 4+ cores recommended (no GPU required)
- 10 GB free disk space

## Setup

### 1. Clone and install Python dependencies

```bash
git clone <repository-url>
cd rl-zero-trust
pip install -r requirements.txt
```

### 2. Create the environment file

```bash
cp .env.example .env
```

Open `.env` and set a strong value for `ZITI_PWD`. Do not commit this file.

### 3. Build and start Docker services

```bash
docker compose build
docker compose up -d
```

Services start in order: OpenZiti, then Ryu, then Mininet, then RL Agent.

### 4. Verify services are running

```bash
docker compose ps
docker stats --no-stream
curl http://localhost:8080/stats/switches
```

### 5. Stop services

```bash
docker compose down

# Full cleanup including volumes:
docker compose down -v
```

## Container Services

| Service | Container | Ports | Memory Limit | Purpose |
|---------|-----------|-------|--------------|---------|
| OpenZiti | openziti-controller | 1280, 3022, 6262 | 1 GB | Zero-trust overlay (mTLS, identities) |
| Ryu | ryu-controller | 6633, 8080 | 2 GB | SDN controller (OpenFlow 1.3 + REST API) |
| Mininet | mininet | none | 4 GB | Network simulation (5 switches, 15 hosts) |
| RL Agent | rl-agent | none | 4 GB | DQN/PPO training and inference |

## Commands

### Interactive Dashboard

The browser-based dashboard at `http://localhost:5000` provides real-time monitoring, attack controls, agent switching (DQN/PPO mid-demo), topology visualization, and training results.

```bash
# Live mode (requires Docker stack running)
python -m scripts.live_demo --agent dqn --steps 200

# Simulation mode (no Docker needed)
python -m scripts.live_demo --agent ppo --steps 200 --no-dashboard

# With attack scenario
python -m scripts.live_demo --agent dqn --steps 300 --scenario auto

# Slower step interval (2 seconds between RL steps)
python -m scripts.live_demo --agent ppo --steps 100 --step-interval 2.0

# Disable demo bias to evaluate true agent policy
python -m scripts.live_demo --agent dqn --steps 200 --no-demo-bias
```

**live_demo.py options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--agent {dqn,ppo}` | dqn | RL agent to use |
| `--steps N` | 200 | Number of RL steps |
| `--step-interval N` | 1.0 | Seconds between steps |
| `--scenario` | none | Attack scenario: auto, ddos, portscan, spoofing, mixed, realistic, none |
| `--no-demo-bias` | off | Disable Q-value bias for true policy evaluation |
| `--no-dashboard` | off | Console-only mode (no web UI) |

### Training

```bash
# Train DQN agent (simulation)
python -m scripts.train_dqn

# Train DQN agent (live, requires Docker)
python -m scripts.train_dqn --live --episodes 500

# Train PPO agent (simulation)
python -m scripts.train_ppo

# Train PPO agent (live, requires Docker)
python -m scripts.train_ppo --live --timesteps 50000

# Custom output directories
python -m scripts.train_dqn --checkpoint-dir checkpoints/dqn_v2 --results-dir results/dqn_v2
```

**train_dqn.py options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--live` | off | Connect to live Ryu controller (requires Docker) |
| `--episodes N` | config | Override max_episodes from dqn_config.yaml |
| `--max-steps N` | config | Override max_steps_per_episode |
| `--checkpoint-dir PATH` | checkpoints/dqn | Model save directory |
| `--results-dir PATH` | results/dqn | Training results directory |
| `--log-dir PATH` | auto | Log file directory |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |

**train_ppo.py options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--live` | off | Connect to live Ryu controller (requires Docker) |
| `--timesteps N` | config | Override total_timesteps from ppo_config.yaml |
| `--max-steps N` | config | Override max steps per episode |
| `--checkpoint-dir PATH` | checkpoints/ppo | Model save directory |
| `--results-dir PATH` | results/ppo | Training results directory |
| `--log-dir PATH` | auto | Log file directory |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |

### Evaluation

```bash
# Evaluate DQN agent
python -m scripts.evaluate --agent dqn

# Evaluate PPO with custom checkpoint
python -m scripts.evaluate --agent ppo --checkpoint checkpoints/ppo/best_model.keras

# Include static baseline comparison
python -m scripts.evaluate --agent dqn --static-baseline --episodes 50
```

**evaluate.py options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--agent {dqn,ppo}` | dqn | Agent type to evaluate |
| `--checkpoint PATH` | checkpoints/{agent} | Model checkpoint path |
| `--episodes N` | auto | Evaluation episodes per scenario |
| `--max-steps N` | auto | Max steps per episode |
| `--results-dir PATH` | results/{agent} | Results output directory |
| `--charts-dir PATH` | auto | Charts output directory |
| `--static-baseline` | off | Also run static baseline (always ALLOW) |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |

### Agent Comparison

```bash
# Compare DQN vs PPO from default result paths
python -m scripts.compare_agents

# Compare with custom paths
python -m scripts.compare_agents \
  --dqn-results results/dqn/evaluation.csv \
  --ppo-results results/ppo/evaluation.csv \
  --output-dir results/comparison
```

**compare_agents.py options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--dqn-results PATH` | auto | DQN evaluation CSV |
| `--ppo-results PATH` | auto | PPO evaluation CSV |
| `--output-dir PATH` | auto | Output directory for charts and report |
| `--dqn-training-log PATH` | auto | DQN training log CSV (for learning curves) |
| `--ppo-training-log PATH` | auto | PPO training log CSV (for learning curves) |

### Experiments

```bash
# Run all 7 experiments (full suite)
python -m scripts.run_experiment

# Run a single experiment
python -m scripts.run_experiment --experiment 3

# Quick mode (10 episodes, 1 seed) for testing
python -m scripts.run_experiment --quick

# Custom seeds and episode count
python -m scripts.run_experiment --seeds 42 123 456 789 --episodes 50
```

**run_experiment.py options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment {all,1..7}` | all | Which experiment to run |
| `--seeds N [N ...]` | 42 123 456 | Random seeds for reproducibility |
| `--episodes N` | 30 | Episodes per seed per scenario |
| `--max-steps N` | 200 | Max steps per episode |
| `--quick` | off | Quick mode: 10 episodes, 1 seed |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |

### Chart Generation

```bash
# Generate publication charts from training/evaluation data
python -m scripts.generate_charts

# Custom data paths
python -m scripts.generate_charts \
  --dqn-eval results/dqn/evaluation.csv \
  --ppo-eval results/ppo/evaluation.csv \
  --charts-dir results/charts
```

**generate_charts.py options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--charts-dir PATH` | auto | Output directory for charts |
| `--exp-dir PATH` | auto | Experiment data directory |
| `--dqn-eval PATH` | auto | DQN evaluation CSV |
| `--ppo-eval PATH` | auto | PPO evaluation CSV |
| `--dqn-log PATH` | auto | DQN training log CSV |
| `--ppo-log PATH` | auto | PPO training log CSV |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR |

### Testing

```bash
# Run full test suite (306 tests)
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_dqn.py -v

# Run a single test by name
python -m pytest tests/test_dqn.py -k "test_action_selection" -v

# Run with short output
python -m pytest tests/ -q
```

### Docker Operations

```bash
# View logs for a specific container
docker compose logs ryu-controller
docker compose logs mininet
docker compose logs rl-agent
docker compose logs openziti-controller

# Follow logs in real-time
docker compose logs -f ryu-controller

# Rebuild a single service
docker compose build ryu-controller
docker compose up -d ryu-controller

# Rebuild everything from scratch
docker compose build --no-cache
docker compose up -d

# Shell into a container
docker exec -it mininet bash
docker exec -it rl-agent bash

# Check resource usage
docker stats --no-stream
```

**Windows/Git Bash note:** Prefix `docker exec` commands with `MSYS_NO_PATHCONV=1` to prevent path mangling:

```bash
MSYS_NO_PATHCONV=1 docker exec mininet python3 -u /app/scripts/start_topology.py
```

## Project Structure

```
rl-zero-trust/
├── docker-compose.yml          # Service orchestration
├── requirements.txt            # Python dependencies (pinned)
├── .env.example                # Environment variable template
├── config/                     # YAML configuration files
│   ├── dqn_config.yaml         # DQN hyperparameters
│   ├── ppo_config.yaml         # PPO hyperparameters
│   ├── network_config.yaml     # Topology and traffic profiles
│   └── openziti_config.yaml    # Zero-trust identities and policies
├── docker/                     # Dockerfiles per service
│   ├── mininet/Dockerfile
│   ├── ryu/Dockerfile
│   ├── openziti/Dockerfile
│   └── rl_agent/Dockerfile
├── src/                        # Source code
│   ├── environment/            # Gym-compatible RL environment
│   ├── agents/                 # DQN and PPO implementations
│   ├── sdn/                    # Stats collector and policy enforcer
│   ├── dashboard/              # FastAPI server + single-file HTML dashboard
│   │   ├── server.py           # SSE streaming, 15 REST endpoints
│   │   └── static/             # index.html + vendor/ (offline assets)
│   ├── attacks/                # Attack simulation (DDoS, scan, spoof)
│   ├── traffic/                # Legitimate traffic generation
│   ├── zero_trust/             # OpenZiti SDK wrapper
│   └── utils/                  # Logging, config, metrics, visualization
├── scripts/                    # Entrypoint scripts
│   ├── live_demo.py            # Dashboard + RL loop launcher
│   ├── train_dqn.py            # DQN training
│   ├── train_ppo.py            # PPO training
│   ├── evaluate.py             # Agent evaluation against attack scenarios
│   ├── compare_agents.py       # DQN vs PPO comparison charts and report
│   ├── run_experiment.py       # Full experiment suite (7 experiments)
│   ├── generate_charts.py      # Publication-quality chart generation
│   └── start_topology.py       # Mininet topology launcher (runs in container)
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
| RL Framework | TensorFlow (CPU) | 2.20.0 |
| RL Environment | Gymnasium | 1.2.3 |
| Numerical Computing | NumPy | 1.26.4 |
| Dashboard Backend | FastAPI + Uvicorn | Latest |
| Dashboard Frontend | Tailwind CSS, DaisyUI, Chart.js, D3.js, FontAwesome 6 | CDN-free (vendored) |
| Containerization | Docker + Compose | Latest |
| Traffic Generation | iPerf3 + Scapy | 2.5.0 |
| Testing | pytest | 7.3.1 |

## Test Suites

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
# Verify Docker Desktop is running and WSL2 is enabled
docker info

# Rebuild from scratch
docker compose build --no-cache
docker compose up -d
```

### Mininet container exits immediately

The Mininet container requires privileged mode for network namespaces. Verify that `privileged: true` is set in docker-compose.yml.

### Ryu REST API not responding

```bash
# Check controller logs
docker compose logs ryu-controller

# Verify OpenFlow port is not in use
netstat -tlnp | grep 6633
```

### Out of memory

```bash
# Check current allocation
docker stats --no-stream

# Reduce container limits in docker-compose.yml if needed
```

### Dashboard not loading assets

All vendor assets are bundled locally in `src/dashboard/static/vendor/`. If the dashboard shows unstyled content, verify that directory contains `js/`, `css/`, `webfonts/`, and `fonts/` subdirectories.

### sch_htb quantum warnings

These are benign in WSL2 Docker environments and can be ignored.

### iPerf3 baseline flows disappearing

Expected behavior. Baseline flows have a 30-second idle timeout and expire when traffic stops.

## Development Sprints

| Sprint | Weeks | Focus |
|--------|-------|-------|
| 1 | 1-2 | Environment Setup and Architecture Design |
| 2 | 3-4 | Network Simulation and SDN Controller Integration |
| 3 | 5-6 | DQN Agent Implementation |
| 4 | 7-8 | Attack Scenarios and DQN Training |
| 5 | 9-10 | PPO Agent Implementation |
| 6 | 11-12 | Zero-Trust Integration and PPO Training |
| 7 | 13-14 | System Evaluation and Documentation |
| 8 | 15-16 | Docker Stack Validation |
| 9 | 17-18 | Live Mode RL Loop |
| 10 | 19-20 | Attack Integration |
| 11 | 21-22 | Interactive Dashboard and Fix Audit |
