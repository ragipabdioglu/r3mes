# R3MES Environment Variables Example

Copy these variables to your `.env` file and customize for your environment.

## Environment Mode
```bash
R3MES_ENV=development  # Options: development, staging, production
```

## Database Configuration
```bash
DATABASE_TYPE=sqlite  # Options: sqlite, postgresql
DATABASE_URL=postgresql://user:password@localhost:5432/r3mes  # Required for postgresql
DATABASE_PATH=backend/database.db  # For SQLite
```

## Redis Configuration
```bash
# Optional in dev, required in production
REDIS_URL=redis://localhost:6379/0
```

## Blockchain Configuration
```bash
# In production, these MUST be set and MUST NOT use localhost
BLOCKCHAIN_RPC_URL=http://localhost:26657  # Development only
BLOCKCHAIN_GRPC_URL=localhost:9090  # Development only
BLOCKCHAIN_REST_URL=http://localhost:1317  # Development only
```

## Backend API Configuration
```bash
BACKEND_PORT=8000
CORS_ALLOWED_ORIGINS=http://localhost:3000  # Development only, comma-separated
```

## IPFS Configuration
```bash
IPFS_GATEWAY_URL=https://ipfs.io/ipfs/
```

## Model Configuration
```bash
BASE_MODEL_PATH=checkpoints/base_model
MODEL_DOWNLOAD_DIR=~/.r3mes/models
R3MES_USE_MOCK_MODEL=false
```

## Logging Configuration
```bash
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE=  # Optional, enables file logging if set
```

## Security Configuration
```bash
API_KEY_SECRET=your-secret-key-here-min-32-chars  # Must be at least 32 characters
JWT_SECRET=your-jwt-secret-key-min-32-chars  # Must be at least 32 characters
```

## Genesis Tokenomics Parameters

These are used by `scripts/genesis_mainnet_config.sh`:

### Mint Module
```bash
MINT_INFLATION_RATE=0.10
MINT_INFLATION_MAX=0.20
MINT_INFLATION_MIN=0.05
MINT_GOAL_BONDED=0.67
MINT_BLOCKS_PER_YEAR=6311520
```

### Staking Module
```bash
STAKING_UNBONDING_TIME=1814400s
STAKING_MAX_VALIDATORS=100
STAKING_MAX_ENTRIES=7
STAKING_HISTORICAL_ENTRIES=10000
```

### Slashing Module
```bash
SLASHING_SIGNED_BLOCKS_WINDOW=10000
SLASHING_MIN_SIGNED_PER_WINDOW=0.05
SLASHING_DOWNTIME_JAIL_DURATION=600s
SLASHING_SLASH_FRACTION_DOUBLE_SIGN=0.05
SLASHING_SLASH_FRACTION_DOWNTIME=0.0001
```

### Distribution Module
```bash
DISTRIBUTION_COMMUNITY_TAX=0.02
DISTRIBUTION_BASE_PROPOSER_REWARD=0.01
DISTRIBUTION_BONUS_PROPOSER_REWARD=0.04
DISTRIBUTION_WITHDRAW_ADDR_ENABLED=true
```

### Governance Module
```bash
GOVERNANCE_MIN_DEPOSIT=1000000uremes
GOVERNANCE_MAX_DEPOSIT_PERIOD=172800s
GOVERNANCE_VOTING_PERIOD=1209600s
GOVENRANCE_QUORUM=0.40
GOVERNANCE_THRESHOLD=0.50
GOVERNANCE_VETO_THRESHOLD=0.334
```

### Crisis Module
```bash
CRISIS_CONSTANT_FEE=1000uremes
```

