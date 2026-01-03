#!/bin/bash
set -e

echo "=== R3MES Validator Setup ==="

# 1. Init
echo "[1/8] Initializing node..."
remesd init r3mes-validator --chain-id r3mes-testnet-1 --home /root/.remes

# 2. Keys
echo "[2/8] Adding keys..."
echo "visual orchard mother post ridge gym undo sell anger weather they you kick weekend swallow patrol wise useless actual merry bottom alone foil diesel" | remesd keys add validator --recover --keyring-backend test --home /root/.remes
echo "wasp promote level wait team soccer helmet nerve boat math pizza bring connect because match ill limit arrest isolate body age motion soup truck" | remesd keys add faucet --recover --keyring-backend test --home /root/.remes
echo "sadness umbrella enough private wreck mosquito left nature since laundry drum fossil imitate audit either rhythm blossom under snack holiday velvet target cup weird" | remesd keys add treasury --recover --keyring-backend test --home /root/.remes

# 3. Genesis accounts
echo "[3/8] Adding genesis accounts..."
remesd genesis add-genesis-account validator 100000000000ur3mes --keyring-backend test --home /root/.remes
remesd genesis add-genesis-account faucet 1000000000000ur3mes --keyring-backend test --home /root/.remes
remesd genesis add-genesis-account treasury 8900000000000ur3mes --keyring-backend test --home /root/.remes

# 4. Gentx
echo "[4/8] Creating gentx..."
remesd genesis gentx validator 100000000000ur3mes --chain-id r3mes-testnet-1 --keyring-backend test --home /root/.remes

# 5. Collect
echo "[5/8] Collecting gentxs..."
remesd genesis collect-gentxs --home /root/.remes

# 6. Validate
echo "[6/8] Validating genesis..."
remesd genesis validate-genesis --home /root/.remes

# 7. Config
echo "[7/8] Configuring node..."
sed -i 's|laddr = "tcp://127.0.0.1:26657"|laddr = "tcp://0.0.0.0:26657"|' /root/.remes/config/config.toml
sed -i 's|cors_allowed_origins = \[\]|cors_allowed_origins = ["*"]|' /root/.remes/config/config.toml
sed -i 's|address = "tcp://localhost:1317"|address = "tcp://0.0.0.0:1317"|' /root/.remes/config/app.toml
sed -i 's|address = "localhost:9090"|address = "0.0.0.0:9090"|' /root/.remes/config/app.toml
sed -i '0,/enable = false/s//enable = true/' /root/.remes/config/app.toml
sed -i 's|swagger = false|swagger = true|' /root/.remes/config/app.toml

# 8. Start
echo "[8/8] Starting validator..."
exec remesd start --home /root/.remes
