#!/bin/bash
# R3MES Desktop Launcher Installation Script
# This script sets up the desktop launcher for production

set -e

echo "=========================================="
echo "R3MES Desktop Launcher Installation"
echo "=========================================="
echo ""

# Check Node.js
echo "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ first."
    exit 1
fi

NODE_VERSION=$(node --version | sed 's/v//')
echo "✅ Node.js $NODE_VERSION detected"
echo ""

# Install dependencies
echo "Installing dependencies..."
cd "$(dirname "$0")/../desktop-launcher"
npm install

echo "✅ Dependencies installed"
echo ""

# Check if web dashboard is running
echo "Checking web dashboard..."
if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "⚠️  Web dashboard not running on http://localhost:3000"
    echo "   Please start the web dashboard first:"
    echo "   cd ../web-dashboard && npm run dev"
    echo ""
    read -p "Continue anyway? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
else
    echo "✅ Web dashboard is running"
    echo ""
fi

# Create desktop entry (Linux)
if [ "$(uname)" != "Darwin" ] && [ "$(uname)" != "MINGW"* ]; then
    echo "Creating desktop entry..."
    DESKTOP_FILE="$HOME/.local/share/applications/r3mes-launcher.desktop"
    mkdir -p "$HOME/.local/share/applications"
    
    cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Name=R3MES Launcher
Comment=R3MES Desktop Launcher
Exec=$(which node) $(pwd)/src/main.js
Icon=$(pwd)/src/assets/icon.png
Terminal=false
Type=Application
Categories=Network;Blockchain;
EOF

    chmod +x "$DESKTOP_FILE"
    echo "✅ Desktop entry created: $DESKTOP_FILE"
    echo ""
fi

# Create startup script
echo "Creating startup script..."
cat > start-launcher.sh <<EOF
#!/bin/bash
cd "$(dirname "$0")"
npm start
EOF

chmod +x start-launcher.sh
echo "✅ Startup script created"
echo ""

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Start launcher:"
echo "   npm start"
echo ""
echo "2. Or use startup script:"
echo "   ./start-launcher.sh"
echo ""
echo "3. For auto-start on boot (Linux):"
echo "   Add to ~/.config/autostart/r3mes-launcher.desktop"
echo ""
echo "Note: Desktop launcher is currently a skeleton."
echo "      Full production features coming soon."
echo ""
echo "=========================================="

