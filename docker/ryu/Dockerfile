# Ryu SDN Controller Container
# Base: Python 3.9 (required for Ryu compatibility)
# Purpose: OpenFlow 1.3 controller with REST API for flow management

FROM python:3.9-slim

LABEL maintainer="Adrian David Justin Hall (TP075220)"
LABEL description="Ryu SDN controller for OpenFlow-based policy enforcement"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ryu and dependencies
# Ryu 4.34 requires older setuptools (get_script_args removed in >=58)
# Use --no-build-isolation so pip uses our pinned setuptools
# eventlet 0.30.2 is the last version with ALREADY_HANDLED in wsgi module
RUN pip install --no-cache-dir "setuptools==57.5.0" "wheel" && \
    pip install --no-cache-dir --no-build-isolation \
    ryu==4.34 \
    eventlet==0.30.2 \
    PyYAML==6.0.1 \
    requests==2.31.0

WORKDIR /app

# OpenFlow controller port and REST API port
EXPOSE 6633 8080

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=15s \
    CMD curl -sf http://localhost:8080/stats/switches || exit 1

# Run custom L2 switch (ZeroTrustSwitch) alongside REST API
CMD ["ryu-manager", \
     "src/sdn/ryu_app.py", \
     "ryu.app.ofctl_rest", \
     "--verbose", \
     "--ofp-tcp-listen-port", "6633"]
