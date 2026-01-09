#!/bin/bash
# Collect Debug Information Script
# Collects debug logs, profiles, and state dumps for analysis

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEBUG_DIR="$HOME/.r3mes"
OUTPUT_DIR="$DEBUG_DIR/debug_collection_$(date +%Y%m%d_%H%M%S)"

echo "========================================="
echo "R3MES Debug Information Collection"
echo "========================================="
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Collect log files
echo "Collecting log files..."
if [ -d "$DEBUG_DIR/logs" ]; then
    cp -r "$DEBUG_DIR/logs" "$OUTPUT_DIR/logs" 2>/dev/null || true
fi
if [ -f "$DEBUG_DIR/debug.log" ]; then
    cp "$DEBUG_DIR/debug.log" "$OUTPUT_DIR/" 2>/dev/null || true
fi
if [ -f "$PROJECT_ROOT/backend_prod.log" ]; then
    cp "$PROJECT_ROOT/backend_prod.log" "$OUTPUT_DIR/" 2>/dev/null || true
fi
if [ -f "$PROJECT_ROOT/frontend_prod.log" ]; then
    cp "$PROJECT_ROOT/frontend_prod.log" "$OUTPUT_DIR/" 2>/dev/null || true
fi
echo "  ✓ Log files collected"

# Collect profile files
echo "Collecting profile files..."
if [ -d "$DEBUG_DIR/profiles" ]; then
    cp -r "$DEBUG_DIR/profiles" "$OUTPUT_DIR/profiles" 2>/dev/null || true
    echo "  ✓ Profile files collected"
else
    echo "  ⚠ No profile directory found"
fi

# Collect trace files
echo "Collecting trace files..."
if [ -d "$DEBUG_DIR/traces" ]; then
    cp -r "$DEBUG_DIR/traces" "$OUTPUT_DIR/traces" 2>/dev/null || true
    echo "  ✓ Trace files collected"
else
    echo "  ⚠ No trace directory found"
fi

# Collect system information
echo "Collecting system information..."
{
    echo "=== System Information ==="
    uname -a
    echo ""
    echo "=== Environment Variables ==="
    env | grep -i r3mes || true
    echo ""
    echo "=== Disk Usage ==="
    df -h
    echo ""
    echo "=== Memory Usage ==="
    free -h || vm_stat || true
    echo ""
    echo "=== Process List (R3MES related) ==="
    ps aux | grep -i r3mes || ps aux | grep -i remes || true
} > "$OUTPUT_DIR/system_info.txt" 2>&1
echo "  ✓ System information collected"

# Create archive
echo ""
echo "Creating archive..."
ARCHIVE_FILE="$DEBUG_DIR/debug_collection_$(date +%Y%m%d_%H%M%S).tar.gz"
cd "$DEBUG_DIR"
tar -czf "$ARCHIVE_FILE" "debug_collection_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
if [ -f "$ARCHIVE_FILE" ]; then
    echo "  ✓ Archive created: $ARCHIVE_FILE"
    echo ""
    echo "Collection complete! Archive size: $(du -h "$ARCHIVE_FILE" | cut -f1)"
else
    echo "  ⚠ Failed to create archive, but files are in: $OUTPUT_DIR"
fi

echo ""
echo "========================================="
echo "Debug information collection complete!"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
if [ -f "$ARCHIVE_FILE" ]; then
    echo "Archive file: $ARCHIVE_FILE"
fi
echo ""
