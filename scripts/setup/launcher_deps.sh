#!/bin/bash
# Desktop Launcher Dependencies Installer

echo "📦 R3MES Desktop Launcher Dependencies"
echo "======================================="
echo ""

# Ubuntu/Debian için güncel paket isimleri
echo "Kurulacak paketler:"
echo "  - libnss3"
echo "  - libatk-bridge2.0-0t64 (veya libatk-bridge2.0-0)"
echo "  - libdrm2"
echo "  - libxkbcommon0"
echo "  - libxcomposite1"
echo "  - libxdamage1"
echo "  - libxfixes3"
echo "  - libxrandr2"
echo "  - libgbm1"
echo "  - libasound2t64 (veya libasound2)"
echo ""

echo "Kurulum komutu:"
echo "sudo apt-get update && sudo apt-get install -y \\"
echo "  libnss3 \\"
echo "  libatk-bridge2.0-0t64 \\"
echo "  libdrm2 \\"
echo "  libxkbcommon0 \\"
echo "  libxcomposite1 \\"
echo "  libxdamage1 \\"
echo "  libxfixes3 \\"
echo "  libxrandr2 \\"
echo "  libgbm1 \\"
echo "  libasound2t64"
echo ""

echo "Veya tek tek:"
echo "sudo apt-get install -y libnss3 libatk-bridge2.0-0t64 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2t64"

