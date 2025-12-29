#!/bin/bash
#
# Create Engine Package Script
#
# Packages engine.exe into a ZIP file with version metadata and SHA256 checksum.
# Creates a release package ready for CDN upload.
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINER_ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
INPUT_EXE=""
VERSION="1.0.0"
OUTPUT_DIR="$MINER_ENGINE_DIR/releases"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_EXE="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --input <engine.exe> [--version <version>] [--output-dir <dir>]"
            echo ""
            echo "Options:"
            echo "  --input <file>      Path to engine.exe (required)"
            echo "  --version <version> Version number (default: 1.0.0)"
            echo "  --output-dir <dir>  Output directory (default: releases/)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate input
if [[ -z "$INPUT_EXE" ]]; then
    echo -e "${RED}Error: --input is required${NC}"
    exit 1
fi

if [[ ! -f "$INPUT_EXE" ]]; then
    echo -e "${RED}Error: Input file not found: $INPUT_EXE${NC}"
    exit 1
fi

# Convert to absolute path
INPUT_EXE="$(cd "$(dirname "$INPUT_EXE")" && pwd)/$(basename "$INPUT_EXE")"
OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "R3MES Engine Package Creator"
echo "=========================================="
echo "Input:  $INPUT_EXE"
echo "Version: $VERSION"
echo "Output:  $OUTPUT_DIR"
echo "=========================================="
echo ""

# Get file size
FILE_SIZE=$(stat -f%z "$INPUT_EXE" 2>/dev/null || stat -c%s "$INPUT_EXE" 2>/dev/null)
FILE_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $FILE_SIZE/1024/1024}")

echo "File size: ${FILE_SIZE_MB} MB"
echo ""

# Calculate SHA256 checksum
echo "Calculating SHA256 checksum..."
CHECKSUM=$(sha256sum "$INPUT_EXE" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$INPUT_EXE" 2>/dev/null | cut -d' ' -f1)

if [[ -z "$CHECKSUM" ]]; then
    echo -e "${RED}Error: Failed to calculate checksum${NC}"
    exit 1
fi

echo "Checksum: $CHECKSUM"
echo ""

# Create version metadata JSON
VERSION_JSON="$OUTPUT_DIR/version-$VERSION.json"
cat > "$VERSION_JSON" <<EOF
{
  "version": "$VERSION",
  "filename": "engine.exe",
  "size_bytes": $FILE_SIZE,
  "size_mb": $FILE_SIZE_MB,
  "sha256_checksum": "$CHECKSUM",
  "created_at": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "platform": "windows",
  "architecture": "x86_64"
}
EOF

echo "✅ Version metadata created: $VERSION_JSON"

# Create ZIP package
ZIP_NAME="engine-v${VERSION}.zip"
ZIP_PATH="$OUTPUT_DIR/$ZIP_NAME"

echo "Creating ZIP package..."
cd "$(dirname "$INPUT_EXE")"
zip -q "$ZIP_PATH" "$(basename "$INPUT_EXE")" "$(dirname "$INPUT_EXE")/version-$VERSION.json" 2>/dev/null || {
    # Fallback if zip command not available, try tar
    cd "$OUTPUT_DIR"
    tar -czf "${ZIP_NAME%.zip}.tar.gz" -C "$(dirname "$INPUT_EXE")" "$(basename "$INPUT_EXE")" "version-$VERSION.json" 2>/dev/null || {
        echo -e "${YELLOW}Warning: zip/tar command not available, skipping package creation${NC}"
        ZIP_PATH=""
    }
}

if [[ -n "$ZIP_PATH" && -f "$ZIP_PATH" ]]; then
    ZIP_SIZE=$(stat -f%z "$ZIP_PATH" 2>/dev/null || stat -c%s "$ZIP_PATH" 2>/dev/null)
    ZIP_SIZE_MB=$(awk "BEGIN {printf \"%.2f\", $ZIP_SIZE/1024/1024}")
    echo "✅ ZIP package created: $ZIP_PATH (${ZIP_SIZE_MB} MB)"
else
    echo -e "${YELLOW}⚠️  ZIP package creation skipped${NC}"
fi

# Create checksum file
CHECKSUM_FILE="$OUTPUT_DIR/engine-v${VERSION}.sha256"
echo "$CHECKSUM  $(basename "$INPUT_EXE")" > "$CHECKSUM_FILE"
echo "✅ Checksum file created: $CHECKSUM_FILE"

# Create manifest.json entry
MANIFEST_PATH="$OUTPUT_DIR/manifest.json"
if [[ -f "$MANIFEST_PATH" ]]; then
    # Update existing manifest
    echo "Updating manifest.json..."
    # Use jq if available, otherwise create simple JSON
    if command -v jq &> /dev/null; then
        jq --arg version "$VERSION" \
           --arg url "https://releases.r3mes.network/$ZIP_NAME" \
           --arg checksum "$CHECKSUM" \
           --arg size "$FILE_SIZE" \
           '.engine.versions[$version] = {
             "version": $version,
             "download_url": $url,
             "checksum": $checksum,
             "size_bytes": ($size | tonumber),
             "required": true
           }' "$MANIFEST_PATH" > "$MANIFEST_PATH.tmp" && mv "$MANIFEST_PATH.tmp" "$MANIFEST_PATH"
    else
        echo -e "${YELLOW}Warning: jq not available, manifest.json not updated${NC}"
    fi
else
    # Create new manifest
    echo "Creating manifest.json..."
    cat > "$MANIFEST_PATH" <<EOF
{
  "launcher": {
    "version": "1.0.0",
    "download_url": "https://releases.r3mes.network/launcher-v1.0.0.exe",
    "checksum": "sha256:...",
    "required": true
  },
  "engine": {
    "version": "$VERSION",
    "download_url": "https://releases.r3mes.network/$ZIP_NAME",
    "checksum": "sha256:$CHECKSUM",
    "required": true,
    "size_bytes": $FILE_SIZE,
    "versions": {
      "$VERSION": {
        "version": "$VERSION",
        "download_url": "https://releases.r3mes.network/$ZIP_NAME",
        "checksum": "sha256:$CHECKSUM",
        "size_bytes": $FILE_SIZE,
        "required": true
      }
    }
  }
}
EOF
    echo "✅ Manifest created: $MANIFEST_PATH"
fi

# Summary
echo ""
echo "=========================================="
echo "✅ Package creation completed!"
echo "=========================================="
echo "Executable: $INPUT_EXE"
echo "Version: $VERSION"
echo "Checksum: $CHECKSUM"
if [[ -n "$ZIP_PATH" && -f "$ZIP_PATH" ]]; then
    echo "Package: $ZIP_PATH"
fi
echo "Manifest: $MANIFEST_PATH"
echo ""
echo "Next steps:"
echo "1. Upload package to CDN: $ZIP_NAME"
echo "2. Upload manifest: manifest.json"
echo "3. Update CDN URL in manifest if needed"
echo "=========================================="

