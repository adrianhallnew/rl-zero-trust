# Sprint 2 Review — Network Simulation & SDN Controller Integration

**Date**: 2026-02-10
**Sprint Goal**: Implement Mininet topology, Ryu SDN controller, traffic generation, and statistics collection.

## Completed User Stories

- [x] As a network simulator, I want a tree-like topology with 5 switches and 15 hosts so that I can model realistic network segments
- [x] As an SDN controller, I want L2 MAC learning and OpenFlow 1.3 support so that I can manage traffic flows
- [x] As a traffic generator, I want iPerf3 wrappers with configurable profiles so that I can produce realistic baseline traffic
- [x] As an RL agent, I want flow/port statistics from the REST API so that I can observe network state
- [x] As a researcher, I want a topology visualization diagram for the dissertation

## Deliverables

| File | Purpose |
|------|---------|
| `src/sdn/topology.py` | Custom Mininet topology (5 switches, 15 hosts, 3/switch) |
| `src/sdn/ryu_app.py` | Ryu L2 switch with MAC learning, OpenFlow 1.3, stats polling |
| `src/sdn/stats_collector.py` | REST API client for flow/port statistics |
| `src/sdn/policy_enforcer.py` | Stub for Sprint 3 RL action → OpenFlow translation |
| `src/traffic/legitimate.py` | iPerf3 wrapper (Mininet mode + subprocess mode) |
| `src/traffic/profiles.py` | Traffic profile definitions (6 profiles) |
| `src/utils/visualization.py` | Publication-quality topology diagram generator |
| `src/utils/metrics.py` | Security/performance metric calculations |
| `results/charts/network_topology.png` | Topology visualization (PNG + SVG) |

## Topology Design

```
                    [s1] (core)
                  /  |   |   \
               [s2] [s3] [s4] [s5] (edge)
               /|\  /|\  /|\  /|\
              3h   3h   3h   3h    (3 hosts each)

  s1: h1(10.0.0.1), h2(10.0.0.2), h3(10.0.0.3)
  s2: h4(10.0.0.4), h5(10.0.0.5), h6(10.0.0.6)
  s3: h7(10.0.0.7), h8(10.0.0.8), h9(10.0.0.9)
  s4: h10(10.0.0.10), h11(10.0.0.11), h12(10.0.0.12)
  s5: h13(10.0.0.13), h14(10.0.0.14), h15(10.0.0.15)

  Links: 100 Mbps, 2ms delay, 0% loss
  All switches: OVS with OpenFlow 1.3
```

## Testing Cycle — Validation Commands

### 1. All 15 hosts can ping each other (full connectivity)
```bash
# Start the topology (inside Mininet container)
docker exec -it mininet python3 /app/src/sdn/topology.py

# In Mininet CLI:
mininet> pingall
# Expected: 0% packet loss
```

### 2. Ryu controller detects all switches and hosts via LLDP
```bash
# Check connected switches
curl http://localhost:8080/stats/switches
# Expected: JSON list with 5 switch DPIDs [1, 2, 3, 4, 5]
```

### 3. Flow entries installed correctly
```bash
docker exec mininet ovs-ofctl dump-flows s1 -O OpenFlow13
docker exec mininet ovs-ofctl dump-flows s2 -O OpenFlow13
# Expected: Table-miss entry + learned MAC forwarding entries
```

### 4. iPerf3 achieves expected throughput between host pairs
```bash
# In Mininet CLI (after starting topology):
mininet> h1 iperf3 -s -D
mininet> h4 iperf3 -c 10.0.0.1 -t 10 -J
# Expected: ~100 Mbps throughput (link bandwidth)
```

### 5. Flow statistics retrievable via REST API
```bash
# Get flow stats for switch 1
curl http://localhost:8080/stats/flow/1
# Expected: JSON with flow entries including packet_count, byte_count

# Get port stats for switch 1
curl http://localhost:8080/stats/port/1
# Expected: JSON with per-port rx/tx counters
```

### 6. Packet captures confirm expected traffic patterns
```bash
docker exec mininet tcpdump -i s1-eth1 -c 20 -n
# Expected: ARP and IP traffic between hosts on the topology
```

### 7. Network survives 30-minute continuous traffic without memory leaks
```bash
# Run sustained load profile
# In Mininet CLI:
mininet> h1 iperf3 -s -D
mininet> h4 iperf3 -c 10.0.0.1 -t 1800 &

# Monitor container memory
docker stats --no-stream
# Expected: Memory stays within 4GB limit for Mininet container
```

## Sprint Review Artifact Checklist

- [x] Topology visualization diagram (`results/charts/network_topology.png`)
- [ ] iPerf3 throughput results table (requires running containers)
- [ ] REST API JSON sample response (requires running containers)

## Issues Encountered

1. [Document any issues during container testing]

## Supervisor Feedback

- [Record feedback from sprint review meeting]

## Items Carried to Sprint 3

- PolicyEnforcer full implementation (RL action → OpenFlow rules)
- RL environment integration with stats_collector
- DQN agent implementation
