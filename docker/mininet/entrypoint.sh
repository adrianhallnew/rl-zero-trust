#!/bin/bash
set -e

echo "=== Mininet Container Starting ==="

# Start ovsdb-server (database for OVS configuration)
echo "Starting ovsdb-server..."
mkdir -p /var/run/openvswitch /var/log/openvswitch

# Initialize OVS database if not exists
if [ ! -f /etc/openvswitch/conf.db ]; then
    ovsdb-tool create /etc/openvswitch/conf.db /usr/share/openvswitch/vswitch.ovsschema
fi

ovsdb-server /etc/openvswitch/conf.db \
    --remote=punix:/var/run/openvswitch/db.sock \
    --pidfile=/var/run/openvswitch/ovsdb-server.pid \
    --detach --log-file=/var/log/openvswitch/ovsdb-server.log

# Wait for ovsdb socket
for i in $(seq 1 10); do
    if [ -S /var/run/openvswitch/db.sock ]; then
        break
    fi
    sleep 0.5
done

# Initialize OVS DB schema version
ovs-vsctl --no-wait init

# Start ovs-vswitchd (the actual switch daemon)
# On WSL2, 'service openvswitch-switch start' may fail to load the kernel
# module, but ovs-vswitchd can still run using the existing kernel datapath
# or the userspace datapath. Start it directly to avoid modprobe issues.
echo "Starting ovs-vswitchd..."
ovs-vswitchd \
    --pidfile=/var/run/openvswitch/ovs-vswitchd.pid \
    --detach \
    --log-file=/var/log/openvswitch/ovs-vswitchd.log

# Wait for OVS to be fully ready
echo "Waiting for OVS to initialize..."
ovs-vsctl --timeout=10 show

echo "Open vSwitch ready (ovsdb-server + ovs-vswitchd)."
echo "Mininet container initialized successfully."

# Execute the CMD or any passed command
exec "$@"
