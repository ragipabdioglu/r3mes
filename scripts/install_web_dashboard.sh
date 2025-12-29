#!/bin/bash
# R3MES Web Dashboard Installation Script
# This script sets up the web dashboard for production

set -e

echo "=========================================="
echo "R3MES Web Dashboard Installation"
echo "=========================================="
echo ""

# Check Node.js
echo "Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ first."
    echo "   Download from: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | sed 's/v//')
echo "✅ Node.js $NODE_VERSION detected"
echo ""

# Check npm
echo "Checking npm..."
if ! command -v npm &> /dev/null; then
    echo "❌ npm not found."
    exit 1
fi

echo "✅ npm detected"
echo ""

# Install dependencies
echo "Installing dependencies..."
cd "$(dirname "$0")/../web-dashboard"
npm install

echo "✅ Dependencies installed"
echo ""

# Build
echo "Building production bundle..."
npm run build

if [ ! -d ".next" ]; then
    echo "❌ Build failed. .next directory not found."
    exit 1
fi

echo "✅ Build successful"
echo ""

# Install PM2 (if not installed)
if ! command -v pm2 &> /dev/null; then
    echo "Installing PM2..."
    sudo npm install -g pm2
    echo "✅ PM2 installed"
    echo ""
fi

# Create PM2 ecosystem file
echo "Creating PM2 ecosystem file..."
cat > ecosystem.config.js <<EOF
module.exports = {
  apps: [{
    name: 'r3mes-dashboard',
    script: 'npm',
    args: 'start',
    cwd: '$(pwd)',
    instances: 1,
    exec_mode: 'fork',
    env: {
      NODE_ENV: 'production',
      PORT: 3000,
      NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:1317',
      NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:1317',
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    autorestart: true,
    watch: false,
    max_memory_restart: '1G',
  }]
};
EOF

echo "✅ PM2 ecosystem file created"
echo ""

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Configure environment variables:"
echo "   export NEXT_PUBLIC_API_URL=https://api.r3mes.network"
echo "   export NEXT_PUBLIC_WS_URL=wss://api.r3mes.network"
echo ""
echo "2. Start with PM2:"
echo "   pm2 start ecosystem.config.js"
echo "   pm2 save"
echo "   pm2 startup"
echo ""
echo "3. Setup Nginx reverse proxy (see DEPLOYMENT_GUIDE.md)"
echo ""
echo "4. Setup SSL (Let's Encrypt):"
echo "   sudo certbot --nginx -d dashboard.r3mes.network"
echo ""
echo "=========================================="

