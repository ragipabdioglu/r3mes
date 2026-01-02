#!/bin/sh
set -e

# R3MES Validator Entrypoint Script

REMES_HOME="/root/.remes"
CHAIN_ID="${CHAIN_ID:-r3mes-testnet-1}"
MONIKER="${MONIKER:-r3mes-validator}"

# Initialize if not already done
if [ ! -f "$REMES_HOME/config/config.toml" ]; then
    echo "🚀 Initializing R3MES node..."
    remesd init "$MONIKER" --chain-id "$CHAIN_ID" --home "$REMES_HOME"
    
    # Configure for external access
    sed -i 's/laddr = "tcp:\/\/127.0.0.1:26657"/laddr = "tcp:\/\/0.0.0.0:26657"/' "$REMES_HOME/config/config.toml"
    sed -i 's/cors_allowed_origins = \[\]/cors_allowed_origins = ["*"]/' "$REMES_HOME/config/config.toml"
    
    # Enable API
    sed -i 's/enable = false/enable = true/' "$REMES_HOME/config/app.toml"
    sed -i 's/swagger = false/swagger = true/' "$REMES_HOME/config/app.toml"
    sed -i 's/address = "tcp:\/\/localhost:1317"/address = "tcp:\/\/0.0.0.0:1317"/' "$REMES_HOME/config/app.toml"
    
    # Enable gRPC
    sed -i 's/address = "localhost:9090"/address = "0.0.0.0:9090"/' "$REMES_HOME/config/app.toml"
    
    # Set minimum gas prices
    sed -i 's/minimum-gas-prices = ""/minimum-gas-prices = "0.025ur3mes"/' "$REMES_HOME/config/app.toml"
    
    echo "✅ Node initialized successfully"
fi

# Check if genesis exists
if [ -f "$REMES_HOME/config/genesis.json" ]; then
    echo "📜 Genesis file found"
else
    echo "⚠️  No genesis file found, using default"
fi

echo "🔗 Starting R3MES node..."
echo "   Chain ID: $CHAIN_ID"
echo "   Moniker: $MONIKER"
echo "   Home: $REMES_HOME"

# Execute the command
exec "$@"
