# Mininet Network Simulation Container
# Base: Ubuntu 22.04 LTS
# Purpose: Virtual SDN environment with tree topology (5 switches, 15 hosts)

FROM ubuntu:22.04

LABEL maintainer="Adrian David Justin Hall (TP075220)"
LABEL description="Mininet SDN simulation environment for RL-driven zero-trust security"

ENV DEBIAN_FRONTEND=noninteractive

# Install Mininet, Open vSwitch, and network tools; apply OS-level security patches
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    mininet \
    openvswitch-switch \
    openvswitch-common \
    python3 \
    python3-pip \
    python3-dev \
    iperf3 \
    tcpdump \
    net-tools \
    iputils-ping \
    iproute2 \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for attack simulation and traffic generation
RUN pip3 install --no-cache-dir \
    scapy==2.7.0 \
    requests==2.32.3 \
    PyYAML==6.0.1

WORKDIR /app

COPY docker/mininet/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=10s \
    CMD ovs-vsctl show > /dev/null 2>&1 || exit 1

ENTRYPOINT ["/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
