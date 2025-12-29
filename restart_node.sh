#!/bin/bash
# R3MES Node Restart Script

echo "🛑 Stopping R3MES node..."

# Kill all remesd processes
pkill -9 remesd 2>/dev/null
pkill -9 remes 2>/dev/null

# Wait a bit for processes to terminate
sleep 2

# Check if ports are still in use
if lsof -i :26657 >/dev/null 2>&1; then
    echo "⚠️  Port 26657 still in use, killing process..."
    lsof -ti :26657 | xargs kill -9 2>/dev/null
fi

if lsof -i :1317 >/dev/null 2>&1; then
    echo "⚠️  Port 1317 still in use, killing process..."
    lsof -ti :1317 | xargs kill -9 2>/dev/null
fi

if lsof -i :9090 >/dev/null 2>&1; then
    echo "⚠️  Port 9090 still in use, killing process..."
    lsof -ti :9090 | xargs kill -9 2>/dev/null
fi

# Wait a bit more
sleep 1

# Verify no processes are running
if pgrep -f remesd >/dev/null; then
    echo "❌ Some remesd processes are still running!"
    ps aux | grep remesd | grep -v grep
    exit 1
else
    echo "✅ All remesd processes stopped"
fi

# Optional: Reset node data (UNCOMMENT IF YOU WANT TO RESET)
# echo "🗑️  Resetting node data..."
# rm -rf ~/.remesd/data
# rm -rf ~/.remesd/wasm

echo "🚀 Starting R3MES node..."
cd ~/R3MES/remes
remesd start

