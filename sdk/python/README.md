# R3MES Python SDK

Official Python SDK for interacting with the R3MES decentralized AI training network.

## Installation

```bash
pip install r3mes-sdk
```

## Quick Start

```python
import asyncio
from r3mes import R3MESClient, Wallet

async def main():
    # Create client
    async with R3MESClient() as client:
        # Get network statistics
        stats = await client.get_network_stats()
        print(f"Active miners: {stats['active_miners']}")
        print(f"Total users: {stats['total_users']}")
        
        # Chat with AI (requires wallet with credits)
        wallet = Wallet.from_mnemonic("your mnemonic phrase here")
        async for token in client.chat("Hello, R3MES!", wallet):
            print(token, end="")

asyncio.run(main())
```

## Features

- **AI Chat**: Stream AI responses with credit-based billing
- **Network Stats**: Query network statistics and health
- **Miner Operations**: Get miner stats, earnings history, leaderboard
- **Serving Nodes**: List and query serving nodes
- **Proposer Operations**: Query aggregations and gradient pools
- **Blockchain Queries**: Get blocks, validators, balances, transactions

## Clients

### R3MESClient

Main client for general operations:

```python
from r3mes import R3MESClient

async with R3MESClient(
    rpc_url="https://rpc.r3mes.network",
    rest_url="https://api.r3mes.network",
    backend_url="https://backend.r3mes.network",
) as client:
    # Get user info
    user = await client.get_user_info("remes1...")
    print(f"Credits: {user['credits']}")
```

### MinerClient

For miner-specific operations:

```python
from r3mes import MinerClient

async with MinerClient() as miner:
    # Get miner stats
    stats = await miner.get_stats("remes1...")
    
    # Get earnings history
    earnings = await miner.get_earnings_history("remes1...", limit=100)
    
    # Get leaderboard
    leaderboard = await miner.get_leaderboard(limit=10, period="week")
```

### ServingClient

For serving node operations:

```python
from r3mes import ServingClient

async with ServingClient() as serving:
    # List all serving nodes
    nodes = await serving.list_nodes()
    
    # Get node details
    node = await serving.get_node("remes1...")
```

### BlockchainClient

For blockchain queries:

```python
from r3mes import BlockchainClient

async with BlockchainClient() as blockchain:
    # Get latest block
    block = await blockchain.get_latest_block()
    
    # Get validators
    validators = await blockchain.get_validators()
    
    # Get balance
    balance = await blockchain.get_balance("remes1...")
```

## Wallet Management

```python
from r3mes import Wallet

# Generate new wallet
wallet, mnemonic = Wallet.generate()
print(f"Address: {wallet.address}")
print(f"Mnemonic: {mnemonic}")  # Save this securely!

# Import from mnemonic
wallet = Wallet.from_mnemonic("word1 word2 ... word24")
```

## Error Handling

```python
from r3mes import (
    R3MESError,
    ConnectionError,
    InsufficientCreditsError,
    RateLimitError,
)

try:
    async for token in client.chat("Hello!", wallet):
        print(token, end="")
except InsufficientCreditsError:
    print("Not enough credits! Mine some blocks first.")
except RateLimitError:
    print("Too many requests. Please wait.")
except ConnectionError as e:
    print(f"Connection failed: {e}")
except R3MESError as e:
    print(f"R3MES error: {e}")
```

## Configuration

Environment variables:

- `R3MES_RPC_URL`: RPC endpoint (default: https://rpc.r3mes.network)
- `R3MES_REST_URL`: REST API endpoint (default: https://api.r3mes.network)
- `R3MES_BACKEND_URL`: Backend endpoint (default: https://backend.r3mes.network)

## Requirements

- Python >= 3.10
- aiohttp >= 3.9.0
- bip39 >= 2.0.0
- cosmpy >= 0.8.0

## License

MIT License - see LICENSE file for details.

## Links

- [Documentation](https://docs.r3mes.network/sdk/python)
- [GitHub](https://github.com/r3mes/sdk-python)
- [R3MES Network](https://r3mes.network)
