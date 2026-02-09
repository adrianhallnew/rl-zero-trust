#!/bin/bash
set -e

echo "=== Mininet Container Starting ==="

# Start Open vSwitch service
echo "Starting Open vSwitch..."
service openvswitch-switch start

# Wait for OVS to be ready
echo "Waiting for OVS to initialize..."
ovs-vsctl --timeout=10 show

echo "Open vSwitch ready."
echo "Mininet container initialized successfully."

# Execute the CMD or any passed command
exec "$@"
