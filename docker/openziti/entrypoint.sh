#!/bin/bash
set -e

ZITI_HOME="${ZITI_HOME:-/openziti}"
ZITI_CTRL_ADDRESS="${ZITI_CTRL_ADVERTISED_ADDRESS:-openziti-controller}"
ZITI_CTRL_PORT="${ZITI_CTRL_ADVERTISED_PORT:-1280}"
ZITI_ROUTER_ADDRESS="${ZITI_ROUTER_ADVERTISED_ADDRESS:-openziti-controller}"
ZITI_ROUTER_PORT="${ZITI_ROUTER_PORT:-3022}"
ZITI_PASSWORD="${ZITI_PWD:-admin}"

echo "=== OpenZiti Controller Starting ==="
echo "Controller address: ${ZITI_CTRL_ADDRESS}:${ZITI_CTRL_PORT}"
echo "Router address: ${ZITI_ROUTER_ADDRESS}:${ZITI_ROUTER_PORT}"

# Run OpenZiti quickstart (creates PKI, controller, and edge router)
# The quickstart handles initialization automatically
exec ziti edge quickstart \
    --ctrl-address "${ZITI_CTRL_ADDRESS}" \
    --ctrl-port "${ZITI_CTRL_PORT}" \
    --router-address "${ZITI_ROUTER_ADDRESS}" \
    --router-port "${ZITI_ROUTER_PORT}" \
    --password "${ZITI_PASSWORD}" \
    --home "${ZITI_HOME}"
