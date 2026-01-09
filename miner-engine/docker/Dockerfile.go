# R3MES Blockchain Node - Go Dockerfile
# Cosmos SDK v0.50.x, CometBFT v0.38.27, Go 1.22

FROM golang:1.22-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
  git \
  make \
  bash \
  ca-certificates \
  tzdata

# Set working directory
WORKDIR /build

# Copy go mod files
COPY remes/go.mod remes/go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY remes/ ./

# Build arguments for platform support
ARG TARGETOS=linux
ARG TARGETARCH=amd64

# Build remesd binary
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
  -mod=readonly \
  -tags 'netgo,ledger,musl,static_build' \
  -ldflags '-extldflags "-static" -w -s -X github.com/cosmos/cosmos-sdk/version.Name=remes -X github.com/cosmos/cosmos-sdk/version.AppName=remesd' \
  -o /build/remesd \
  ./cmd/remesd

# Final stage
FROM alpine:latest

# Install runtime dependencies
RUN apk add --no-cache \
  ca-certificates \
  tzdata \
  curl \
  bash \
  jq

# Copy binary from builder
COPY --from=builder /build/remesd /usr/local/bin/remesd

# Create app directory and data directory
WORKDIR /app
RUN mkdir -p /app/.remesd

# Create genesis initialization script
RUN echo '#!/bin/bash\n\
  set -e\n\
  \n\
  CHAIN_ID="${CHAIN_ID:-remes-testnet-1}"\n\
  HOME_DIR="/app/.remesd"\n\
  \n\
  # Check if already initialized\n\
  if [ -f "$HOME_DIR/config/genesis.json" ]; then\n\
  echo "Genesis already exists, skipping initialization..."\n\
  else\n\
  echo "Initializing blockchain with chain-id: $CHAIN_ID"\n\
  \n\
  # Initialize chain\n\
  remesd init testnet-node --chain-id "$CHAIN_ID" --home "$HOME_DIR"\n\
  \n\
  # Create validator key\n\
  remesd keys add validator --keyring-backend test --home "$HOME_DIR" 2>&1 || true\n\
  \n\
  # Get validator address\n\
  VALIDATOR_ADDR=$(remesd keys show validator -a --keyring-backend test --home "$HOME_DIR")\n\
  echo "Validator address: $VALIDATOR_ADDR"\n\
  \n\
  # Add genesis account with initial balance\n\
  remesd genesis add-genesis-account "$VALIDATOR_ADDR" 1000000000000stake --keyring-backend test --home "$HOME_DIR"\n\
  \n\
  # Create genesis transaction\n\
  remesd genesis gentx validator 100000000000stake \\\n\
  --chain-id "$CHAIN_ID" \\\n\
  --keyring-backend test \\\n\
  --home "$HOME_DIR"\n\
  \n\
  # Collect genesis transactions\n\
  remesd genesis collect-gentxs --home "$HOME_DIR"\n\
  \n\
  # Update config for external access\n\
  sed -i "s/127.0.0.1:26657/0.0.0.0:26657/g" "$HOME_DIR/config/config.toml"\n\
  sed -i "s/localhost:26657/0.0.0.0:26657/g" "$HOME_DIR/config/config.toml"\n\
  \n\
  # Enable API and gRPC\n\
  sed -i "s/enable = false/enable = true/g" "$HOME_DIR/config/app.toml"\n\
  sed -i "s/swagger = false/swagger = true/g" "$HOME_DIR/config/app.toml"\n\
  sed -i "s/address = \"localhost:9090\"/address = \"0.0.0.0:9090\"/g" "$HOME_DIR/config/app.toml"\n\
  sed -i "s/address = \"tcp:\\/\\/localhost:1317\"/address = \"tcp:\\/\\/0.0.0.0:1317\"/g" "$HOME_DIR/config/app.toml"\n\
  \n\
  echo "Genesis initialization complete!"\n\
  fi\n\
  \n\
  # Start the node\n\
  echo "Starting remesd..."\n\
  exec remesd start --home "$HOME_DIR"\n\
  ' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# Default command (entrypoint handles start)
CMD []

