# R3MES Go SDK

Official Go SDK for interacting with the R3MES decentralized AI training network.

## Installation

```bash
go get github.com/r3mes/sdk-go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"
    "log"

    r3mes "github.com/r3mes/sdk-go"
)

func main() {
    // Create client with default config
    client, err := r3mes.NewClient(r3mes.DefaultConfig())
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    ctx := context.Background()

    // Get network stats
    stats, err := client.GetNetworkStats(ctx)
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Active miners: %d\n", stats.ActiveMiners)
    fmt.Printf("Total users: %d\n", stats.TotalUsers)

    // Get user info
    user, err := client.GetUserInfo(ctx, "remes1...")
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("Credits: %.2f\n", user.Credits)
}
```

## Features

- **Network Stats**: Query network statistics and health
- **User Info**: Get user information and credits
- **Miner Operations**: Get miner stats, earnings history, leaderboard
- **Blockchain Queries**: Get blocks, validators, balances, transactions

## Configuration

```go
config := r3mes.Config{
    RPCEndpoint:     "https://rpc.r3mes.network",
    RESTEndpoint:    "https://api.r3mes.network",
    BackendEndpoint: "https://backend.r3mes.network",
    Timeout:         30 * time.Second,
}

client, err := r3mes.NewClient(config)
```

## Clients

### Main Client

```go
// Get network stats
stats, err := client.GetNetworkStats(ctx)

// Get user info
user, err := client.GetUserInfo(ctx, "remes1...")

// Get miner stats
miner, err := client.GetMinerStats(ctx, "remes1...")

// Get latest block
block, err := client.GetLatestBlock(ctx)

// Get balance
balances, err := client.GetBalance(ctx, "remes1...")
```

### Miner Client

```go
minerClient := client.Miner()

// Get earnings history
earnings, err := minerClient.GetEarningsHistory(ctx, "remes1...", 100, 0)

// Get hashrate history
hashrate, err := minerClient.GetHashrateHistory(ctx, "remes1...", 24)

// Get leaderboard
leaderboard, err := minerClient.GetLeaderboard(ctx, 100, "all")

// Get active task pool
pool, err := minerClient.GetActivePool(ctx)
```

### Blockchain Client

```go
blockchainClient := client.Blockchain()

// Get block by height
block, err := blockchainClient.GetBlock(ctx, 12345)

// Get validators
validators, err := blockchainClient.GetValidators(ctx, "BOND_STATUS_BONDED", 100)

// Get transaction
tx, err := blockchainClient.GetTransaction(ctx, "TXHASH...")

// Get node status
status, err := blockchainClient.GetStatus(ctx)
```

## Error Handling

```go
import "errors"

user, err := client.GetUserInfo(ctx, "remes1...")
if err != nil {
    if errors.Is(err, r3mes.ErrUserNotFound) {
        fmt.Println("User not found")
    } else if errors.Is(err, r3mes.ErrConnectionFailed) {
        fmt.Println("Connection failed")
    } else {
        fmt.Printf("Error: %v\n", err)
    }
}
```

## Available Errors

- `ErrUserNotFound` - User not found
- `ErrMinerNotFound` - Miner not found
- `ErrNodeNotFound` - Node not found
- `ErrInsufficientCredits` - Insufficient credits
- `ErrInvalidWallet` - Invalid wallet address
- `ErrTransactionFailed` - Transaction failed
- `ErrConnectionFailed` - Connection failed
- `ErrTimeout` - Operation timed out
- `ErrRateLimited` - Rate limit exceeded
- `ErrBlockNotFound` - Block not found
- `ErrTransactionNotFound` - Transaction not found

## Types

### NetworkStats

```go
type NetworkStats struct {
    ActiveMiners int64   `json:"active_miners"`
    TotalUsers   int64   `json:"total_users"`
    TotalCredits float64 `json:"total_credits"`
    BlockHeight  int64   `json:"block_height"`
}
```

### MinerStats

```go
type MinerStats struct {
    WalletAddress    string  `json:"wallet_address"`
    Hashrate         float64 `json:"hashrate"`
    TotalEarnings    float64 `json:"total_earnings"`
    BlocksFound      int64   `json:"blocks_found"`
    UptimePercentage float64 `json:"uptime_percentage"`
    IsActive         bool    `json:"is_active"`
}
```

### BlockInfo

```go
type BlockInfo struct {
    Height    int64  `json:"height"`
    Hash      string `json:"hash"`
    Timestamp string `json:"timestamp"`
    Proposer  string `json:"proposer"`
    TxCount   int    `json:"tx_count"`
}
```

## Requirements

- Go >= 1.21

## License

MIT License - see LICENSE file for details.

## Links

- [Documentation](https://docs.r3mes.network/sdk/go)
- [GitHub](https://github.com/r3mes/sdk-go)
- [R3MES Network](https://r3mes.network)
