# OpenZiti Zero-Trust Controller Container
# Base: Ubuntu 22.04 LTS
# Purpose: Identity-based routing, mTLS, dark services between components

FROM ubuntu:22.04

LABEL maintainer="Adrian David Justin Hall (TP075220)"
LABEL description="OpenZiti zero-trust overlay network controller"

ENV DEBIAN_FRONTEND=noninteractive
ENV ZITI_HOME=/openziti

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Install OpenZiti CLI from official repository
RUN curl -sSLf https://get.openziti.io/install.bash | bash -s openziti

# Create data directory
RUN mkdir -p ${ZITI_HOME}

WORKDIR ${ZITI_HOME}

COPY docker/openziti/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Controller API, Edge Router, Management
EXPOSE 1280 3022 6262

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=60s \
    CMD curl -sk https://localhost:1280/edge/client/v1/version > /dev/null 2>&1 || exit 1

ENTRYPOINT ["/entrypoint.sh"]
