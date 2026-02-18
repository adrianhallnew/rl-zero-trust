#!/bin/bash
# OpenZiti Identity & Service Provisioning Script
#
# Creates identities, services, and policies for the RL-Zero-Trust system.
# Run after the OpenZiti controller has fully started.
#
# Usage:
#   docker exec openziti-controller /openziti/setup_identities.sh

set -euo pipefail

ZITI_CLI="/usr/local/bin/ziti"
ZITI_CTRL="https://localhost:1280"
ZITI_USER="${ZITI_USER:-admin}"
ZITI_PASS="${ZITI_PWD:-admin}"
IDENTITY_DIR="/openziti/identities"

echo "=== OpenZiti Identity & Service Setup ==="
echo "Controller: ${ZITI_CTRL}"

# Wait for controller to be ready
echo "[1/6] Waiting for controller..."
for i in $(seq 1 30); do
    if "${ZITI_CLI}" edge login "${ZITI_CTRL}" \
        -u "${ZITI_USER}" -p "${ZITI_PASS}" \
        --yes 2>/dev/null; then
        echo "  Controller ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ERROR: Controller not ready after 30 attempts."
        exit 1
    fi
    sleep 2
done

# Create identity directory
mkdir -p "${IDENTITY_DIR}"

# --- Identities ---
echo "[2/6] Creating identities..."
declare -A IDENTITIES=(
    ["rl-agent"]="rl-agent-identity"
    ["ryu-controller"]="ryu-controller-identity"
    ["mininet-monitor"]="mininet-monitor-identity"
)

for HOST_NAME in "${!IDENTITIES[@]}"; do
    IDENTITY_NAME="${IDENTITIES[$HOST_NAME]}"
    ENROLLMENT_FILE="${IDENTITY_DIR}/${HOST_NAME}.json"

    if "${ZITI_CLI}" edge list identities "filter name=\"${IDENTITY_NAME}\"" 2>/dev/null | grep -q "${IDENTITY_NAME}"; then
        echo "  Identity '${IDENTITY_NAME}' already exists, skipping."
    else
        echo "  Creating identity: ${IDENTITY_NAME}"
        "${ZITI_CLI}" edge create identity "${IDENTITY_NAME}" \
            --role-attributes "${HOST_NAME}" \
            -o "${ENROLLMENT_FILE}"

        echo "  Enrolling identity: ${IDENTITY_NAME}"
        "${ZITI_CLI}" edge enroll "${ENROLLMENT_FILE}"
    fi
done

# --- Services ---
echo "[3/6] Creating services..."
declare -A SERVICES=(
    ["sdn-controller-api"]="ryu-controller:8080"
    ["policy-engine"]="rl-agent:5000"
    ["monitoring-feed"]="mininet:9090"
)

for SVC_NAME in "${!SERVICES[@]}"; do
    HOST_PORT="${SERVICES[$SVC_NAME]}"
    HOST="${HOST_PORT%%:*}"
    PORT="${HOST_PORT##*:}"

    if "${ZITI_CLI}" edge list services "filter name=\"${SVC_NAME}\"" 2>/dev/null | grep -q "${SVC_NAME}"; then
        echo "  Service '${SVC_NAME}' already exists, skipping."
    else
        echo "  Creating service: ${SVC_NAME} -> ${HOST}:${PORT}"
        "${ZITI_CLI}" edge create service "${SVC_NAME}" \
            --role-attributes "${SVC_NAME}"
    fi
done

# --- Service Edge Router Policies ---
echo "[4/6] Creating service edge router policies..."
"${ZITI_CLI}" edge create service-edge-router-policy "all-services-public" \
    --service-roles "#all" \
    --edge-router-roles "#all" 2>/dev/null || echo "  Policy already exists."

# --- Bind Policies (hosting) ---
echo "[5/6] Creating bind (host) policies..."
declare -A BIND_POLICIES=(
    ["sdn-controller-api-bind"]="sdn-controller-api:ryu-controller"
    ["policy-engine-bind"]="policy-engine:rl-agent"
    ["monitoring-feed-bind"]="monitoring-feed:mininet-monitor"
)

for POLICY_NAME in "${!BIND_POLICIES[@]}"; do
    SVC_HOST="${BIND_POLICIES[$POLICY_NAME]}"
    SVC="${SVC_HOST%%:*}"
    HOST_ATTR="${SVC_HOST##*:}"

    "${ZITI_CLI}" edge create service-policy "${POLICY_NAME}" Bind \
        --service-roles "@${SVC}" \
        --identity-roles "#${HOST_ATTR}" 2>/dev/null || echo "  ${POLICY_NAME} already exists."
done

# --- Dial Policies (access) ---
echo "[6/6] Creating dial (access) policies..."
declare -A DIAL_POLICIES=(
    ["rl-agent-to-sdn"]="sdn-controller-api:rl-agent"
    ["rl-agent-to-monitoring"]="monitoring-feed:rl-agent"
    ["monitoring-to-policy"]="policy-engine:mininet-monitor"
)

for POLICY_NAME in "${!DIAL_POLICIES[@]}"; do
    SVC_CALLER="${DIAL_POLICIES[$POLICY_NAME]}"
    SVC="${SVC_CALLER%%:*}"
    CALLER_ATTR="${SVC_CALLER##*:}"

    "${ZITI_CLI}" edge create service-policy "${POLICY_NAME}" Dial \
        --service-roles "@${SVC}" \
        --identity-roles "#${CALLER_ATTR}" 2>/dev/null || echo "  ${POLICY_NAME} already exists."
done

echo ""
echo "=== Setup Complete ==="
echo "Identities: ${!IDENTITIES[*]}"
echo "Services:   ${!SERVICES[*]}"
echo "Identity files in: ${IDENTITY_DIR}"
