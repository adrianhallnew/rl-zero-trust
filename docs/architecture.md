# System Architecture Document

## RL-Driven Adaptive Security System for Zero-Trust Networks

**Version:** 1.0
**Sprint:** 1 (Environment Setup & Architecture Design)
**Author:** Adrian David Justin Hall (TP075220)

---

## 1. System Overview

The system implements a reinforcement learning-driven adaptive security framework for software-defined zero-trust networks. It bridges the gap between static security policies and evolving threats by using DQN and PPO algorithms to automatically adjust network security policies in response to detected threats.

```
+------------------------------------------------------------------+
|                     Docker Compose Environment                    |
|                                                                   |
|  +------------------+    +------------------+                     |
|  |   OpenZiti        |    |   Ryu SDN        |                    |
|  |   Controller      |<-->|   Controller     |                    |
|  |                   |    |                  |                     |
|  |  - PKI/CA         |    |  - OpenFlow 1.3  |                    |
|  |  - mTLS Certs     |    |  - REST API      |                    |
|  |  - Edge Router    |    |  - Flow Tables   |                    |
|  |  - Dark Services  |    |  - L2 Switching  |                    |
|  |                   |    |                  |                     |
|  |  Port: 1280       |    |  Port: 6633/8080 |                    |
|  |  Memory: 1 GB     |    |  Memory: 2 GB    |                    |
|  +------------------+    +--------+---------+                     |
|                                    |                              |
|                            OpenFlow 1.3                           |
|                                    |                              |
|  +------------------+    +--------+---------+                     |
|  |   RL Agent        |    |   Mininet         |                   |
|  |                   |<-->|   Network Sim     |                   |
|  |  - DQN Agent      |    |                   |                   |
|  |  - PPO Agent      |    |  - 5 Switches     |                   |
|  |  - TensorFlow     |    |  - 15 Hosts       |                   |
|  |  - Training Loop  |    |  - Open vSwitch   |                   |
|  |  - TensorBoard    |    |  - iPerf3/Scapy   |                   |
|  |                   |    |                   |                    |
|  |  Memory: 4 GB     |    |  Memory: 4 GB     |                   |
|  +------------------+    +-------------------+                    |
|                                                                   |
|  Network: zt-network (172.20.0.0/24)                             |
+------------------------------------------------------------------+
```

---

## 2. Component Communication Diagram

### 2.1 Data Flow (RL Training Loop)

```
Step 1: OBSERVE
  Mininet Switches --[flow stats]--> Ryu REST API --[HTTP GET]--> RL Agent
  (port stats, flow counts, byte counts, connection rates)

Step 2: DECIDE
  RL Agent: state_vector = StateProcessor.process(raw_stats)
  RL Agent: action = DQN.select_action(state_vector)  [or PPO]

Step 3: ACT
  RL Agent --[HTTP POST]--> Ryu REST API --[OpenFlow]--> Switch Flow Table
  Actions: ALLOW | BLOCK | REROUTE | RATE_LIMIT

Step 4: EVALUATE
  RewardCalculator.compute(metrics) --> reward signal
  Components: detection + false_positive + throughput + latency + stability

Step 5: LEARN
  DQN: store (s, a, r, s') in replay buffer --> sample batch --> train
  PPO: collect trajectory --> compute advantages --> update policy
```

### 2.2 Network Communication Matrix

| Source | Destination | Protocol | Port | Purpose | Security |
|--------|-------------|----------|------|---------|----------|
| RL Agent | Ryu Controller | HTTP | 8080 | Flow stats query, policy push | OpenZiti mTLS (Sprint 6) |
| Ryu Controller | Mininet Switches | OpenFlow | 6633 | Flow table management | Docker bridge |
| Mininet Hosts | Mininet Hosts | TCP/UDP | Various | Legitimate + attack traffic | Mininet internal |
| RL Agent | TensorBoard | HTTP | 6006 | Training visualization | Local only |
| All Components | OpenZiti | HTTPS | 1280 | Identity verification | mTLS |

### 2.3 Docker Networking

```
Docker Bridge Network: zt-network (172.20.0.0/24)

  172.20.0.10  openziti-controller   (Controller API + Edge Router)
  172.20.0.20  ryu-controller        (OpenFlow + REST API)
  172.20.0.30  mininet               (Network Simulation)
  172.20.0.40  rl-agent              (Training Agent)
```

---

## 3. Mininet Network Topology

### 3.1 Tree Topology (depth=2, fanout=5)

```
                            [s1] (Root Switch)
                     /    /    |    \    \
                   /    /      |      \    \
                [s2]  [s3]   [s4]   [s5]  [s6]
               / | \  / | \  / | \  / | \  / | \
              h1-h3  h4-h6  h7-h9 h10-12 h13-15
```

- **Switches:** 6 (1 root + 5 leaf) -- adjusted from spec to achieve 15 hosts
- **Hosts:** 15 (3 hosts per leaf switch)
- **Host subnet:** 10.0.0.0/24 (h1=10.0.0.1 ... h15=10.0.0.15)
- **OpenFlow version:** 1.3
- **Controller connection:** Remote (Ryu at 172.20.0.20:6633)

### 3.2 Link Parameters

| Parameter | Default | Degraded |
|-----------|---------|----------|
| Bandwidth | 100 Mbps | 50 Mbps |
| Delay | 2 ms | 10 ms |
| Packet Loss | 0% | 1% |
| Queue Size | 1000 packets | 1000 packets |

---

## 4. RL Environment Design

### 4.1 State Space

The observation vector is constructed from Ryu REST API flow statistics:

| Feature | Dimension | Source | Range |
|---------|-----------|--------|-------|
| Per-flow packet count | N_flows | /stats/flow/{dpid} | [0, inf) |
| Per-flow byte count | N_flows | /stats/flow/{dpid} | [0, inf) |
| Per-flow duration | N_flows | /stats/flow/{dpid} | [0, inf) |
| Per-port rx_packets | N_ports | /stats/port/{dpid} | [0, inf) |
| Per-port tx_packets | N_ports | /stats/port/{dpid} | [0, inf) |
| Per-port rx_dropped | N_ports | /stats/port/{dpid} | [0, inf) |
| Per-port tx_dropped | N_ports | /stats/port/{dpid} | [0, inf) |
| Active flow count | N_switches | /stats/flow/{dpid} | [0, inf) |
| Connection rate | 1 | Computed | [0, inf) |
| Moving averages | K | Computed | [0, inf) |

All features are normalized to [0, 1] by `StateProcessor` using min-max scaling with running statistics.

### 4.2 Action Space

**DQN (Discrete):**

| Action ID | Name | Effect |
|-----------|------|--------|
| 0 | ALLOW | Permit traffic normally |
| 1 | BLOCK | Drop matching packets (install drop flow) |
| 2 | REROUTE | Redirect via alternative switch path |
| 3 | RATE_LIMIT | Throttle to 50% bandwidth (meter band) |

**PPO (Continuous):**

| Dimension | Name | Range | Effect |
|-----------|------|-------|--------|
| 0 | Rate limit intensity | [0.0, 1.0] | 0=no limit, 1=full block |
| 1 | Rerouting weight | [0.0, 1.0] | Probability of alt path |
| 2 | Priority adjustment | [-1.0, 1.0] | QoS priority change |

### 4.3 Reward Function

```
R(t) = 0.40 * R_detection
     + 0.25 * R_false_positive
     + 0.15 * R_throughput
     + 0.10 * R_latency
     + 0.10 * R_stability
```

See `config/dqn_config.yaml` for full specification.

---

## 5. Docker Service Architecture

### 5.1 Startup Order

```
1. openziti-controller  (zero-trust control plane)
       |
       v
2. ryu-controller       (SDN control plane)
       |
       v
3. mininet              (network data plane)
       |
       v
4. rl-agent             (ML training plane)
```

### 5.2 Resource Allocation

| Service | Memory Limit | Memory Reserved | CPU | Privileged |
|---------|-------------|-----------------|-----|------------|
| openziti-controller | 1 GB | 512 MB | shared | No |
| ryu-controller | 2 GB | 512 MB | shared | No |
| mininet | 4 GB | 1 GB | shared | Yes |
| rl-agent | 4 GB | 1 GB | shared | No |
| **Total** | **11 GB** | **3 GB** | - | - |

### 5.3 Volume Mounts

| Service | Host Path | Container Path | Mode | Purpose |
|---------|-----------|---------------|------|---------|
| ryu-controller | ./src | /app/src | ro | Source code |
| ryu-controller | ./config | /app/config | ro | Configuration |
| mininet | ./src | /app/src | rw | Source + attack scripts |
| mininet | ./config | /app/config | ro | Configuration |
| mininet | ./captures | /app/captures | rw | Packet captures |
| rl-agent | ./src | /app/src | rw | Source code |
| rl-agent | ./config | /app/config | ro | Configuration |
| rl-agent | ./scripts | /app/scripts | rw | Training scripts |
| rl-agent | ./results | /app/results | rw | Training outputs |
| rl-agent | ./checkpoints | /app/checkpoints | rw | Model saves |

---

## 6. Security Architecture (Zero-Trust)

### 6.1 OpenZiti Integration (Sprint 6)

All inter-component communication will be secured via OpenZiti overlay:

```
+------------------+         +------------------+
|   RL Agent       |         |  Ryu Controller  |
|                  |         |                  |
|  [Ziti SDK]      |<--mTLS-->|  [Ziti SDK]     |
|  Identity: rl-   |         |  Identity: ryu-  |
|  agent-identity  |         |  ctrl-identity   |
+------------------+         +------------------+
        |                            |
        +---------- mTLS -----------+
                     |
            +------------------+
            | OpenZiti         |
            | Controller       |
            |                  |
            | - PKI/CA         |
            | - Identity Mgmt |
            | - Service Mesh  |
            +------------------+
```

### 6.2 Dark Services Model

- No inbound ports exposed on any application service
- All connections initiated outbound through OpenZiti SDK
- Identity verified via X.509 certificates (mTLS)
- Service access controlled by service policies
- Posture checks verify endpoint security before authorization

### 6.3 Services

| Service | Description | Host | Consumers |
|---------|-------------|------|-----------|
| sdn-controller-api | Ryu REST API | ryu-controller | rl-agent |
| policy-engine | RL policy endpoint | rl-agent | ryu-controller |
| monitoring-feed | Network stats feed | mininet | rl-agent |

---

## 7. Attack Scenarios (Sprints 4+)

| Attack Type | Method | Source | Target | Detection Features |
|-------------|--------|--------|--------|-------------------|
| DDoS (SYN Flood) | Scapy SYN packets | 3+ hosts | Single host | High packet rate, SYN flag ratio |
| DDoS (UDP Flood) | Scapy UDP packets | 3+ hosts | Single host | High byte rate, no responses |
| Port Scan (TCP Connect) | Sequential TCP probes | 1 host | Port range | Many short connections |
| Port Scan (SYN Scan) | Scapy SYN-only probes | 1 host | Port range | SYN-only, no ACK follow-up |
| IP Spoofing | Forged source IP | 1 host | Network | Mismatched source IPs |
| MAC Spoofing | Forged source MAC | 1 host | Local segment | MAC-IP inconsistency |
| ARP Spoofing | False ARP replies | 1 host | Local segment | ARP reply storms |
| Mixed | Simultaneous multi-vector | Multiple | Multiple | Combined feature anomalies |
