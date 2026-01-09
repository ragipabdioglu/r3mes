#!/bin/bash
# R3MES Node Restart Script with Logging

echo "🛑 Stopping R3MES node..."

# Kill all remesd processes
pkill -9 remesd 2>/dev/null
pkill -9 remes 2>/dev/null

# Wait a bit
sleep 2

# Kill processes on ports
lsof -ti :26657 | xargs kill -9 2>/dev/null
lsof -ti :1317 | xargs kill -9 2>/dev/null
lsof -ti :9090 | xargs kill -9 2>/dev/null

sleep 1

echo "🚀 Starting R3MES node with new binary..."
cd ~/R3MES/remes

# Start node and look for dashboard route registration log
remesd start 2>&1 | tee /tmp/remesd.log &
NODE_PID=$!

# Wait a bit for node to start
sleep 5

# Check if dashboard routes were registered
if grep -q "Dashboard API routes registered" /tmp/remesd.log 2>/dev/null; then
    echo "✅ Dashboard routes registered successfully!"
else
    echo "⚠️  Dashboard routes registration log not found"
    echo "Checking node logs..."
    tail -20 /tmp/remesd.log
fi

echo ""
echo "Node PID: $NODE_PID"
echo "Log file: /tmp/remesd.log"
echo ""
echo "Test API endpoints:"
echo "  curl http://localhost:1317/api/dashboard/status"
echo "  curl http://localhost:1317/api/dashboard/locations"

