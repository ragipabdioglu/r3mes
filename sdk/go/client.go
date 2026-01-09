// Package r3mes provides the official Go SDK for the R3MES network.
//
// Example usage:
//
//	client, err := r3mes.NewClient(r3mes.Config{
//	    RPCEndpoint: "https://rpc.r3mes.network",
//	    RESTEndpoint: "https://api.r3mes.network",
//	})
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer client.Close()
//
//	stats, err := client.GetNetworkStats(context.Background())
package r3mes

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Config holds the configuration for the R3MES client.
type Config struct {
	// RPCEndpoint is the Tendermint RPC endpoint URL.
	RPCEndpoint string

	// RESTEndpoint is the Cosmos REST API endpoint URL.
	RESTEndpoint string

	// BackendEndpoint is the R3MES backend inference service URL.
	BackendEndpoint string

	// Timeout is the HTTP request timeout.
	Timeout time.Duration
}

// DefaultConfig returns the default configuration.
func DefaultConfig() Config {
	return Config{
		RPCEndpoint:     "https://rpc.r3mes.network",
		RESTEndpoint:    "https://api.r3mes.network",
		BackendEndpoint: "https://backend.r3mes.network",
		Timeout:         30 * time.Second,
	}
}

// Client is the main R3MES SDK client.
type Client struct {
	config     Config
	httpClient *http.Client
}

// NewClient creates a new R3MES client with the given configuration.
func NewClient(config Config) (*Client, error) {
	if config.RPCEndpoint == "" {
		config.RPCEndpoint = DefaultConfig().RPCEndpoint
	}
	if config.RESTEndpoint == "" {
		config.RESTEndpoint = DefaultConfig().RESTEndpoint
	}
	if config.BackendEndpoint == "" {
		config.BackendEndpoint = DefaultConfig().BackendEndpoint
	}
	if config.Timeout == 0 {
		config.Timeout = DefaultConfig().Timeout
	}

	return &Client{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}, nil
}

// Close closes the client and releases resources.
func (c *Client) Close() error {
	c.httpClient.CloseIdleConnections()
	return nil
}

// NetworkStats represents network statistics.
type NetworkStats struct {
	ActiveMiners int64   `json:"active_miners"`
	TotalUsers   int64   `json:"total_users"`
	TotalCredits float64 `json:"total_credits"`
	BlockHeight  int64   `json:"block_height"`
}

// GetNetworkStats retrieves network statistics.
func (c *Client) GetNetworkStats(ctx context.Context) (*NetworkStats, error) {
	url := fmt.Sprintf("%s/network/stats", c.config.BackendEndpoint)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var stats NetworkStats
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &stats, nil
}

// UserInfo represents user information.
type UserInfo struct {
	WalletAddress string  `json:"wallet_address"`
	Credits       float64 `json:"credits"`
	IsMiner       bool    `json:"is_miner"`
}

// GetUserInfo retrieves user information by wallet address.
func (c *Client) GetUserInfo(ctx context.Context, walletAddress string) (*UserInfo, error) {
	url := fmt.Sprintf("%s/user/info/%s", c.config.BackendEndpoint, walletAddress)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, ErrUserNotFound
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var info UserInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &info, nil
}

// MinerStats represents miner statistics.
type MinerStats struct {
	WalletAddress    string  `json:"wallet_address"`
	Hashrate         float64 `json:"hashrate"`
	TotalEarnings    float64 `json:"total_earnings"`
	BlocksFound      int64   `json:"blocks_found"`
	UptimePercentage float64 `json:"uptime_percentage"`
	IsActive         bool    `json:"is_active"`
}

// GetMinerStats retrieves miner statistics by wallet address.
func (c *Client) GetMinerStats(ctx context.Context, walletAddress string) (*MinerStats, error) {
	url := fmt.Sprintf("%s/miner/stats/%s", c.config.BackendEndpoint, walletAddress)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return &MinerStats{
			WalletAddress: walletAddress,
			IsActive:      false,
		}, nil
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var stats MinerStats
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &stats, nil
}

// BlockInfo represents block information.
type BlockInfo struct {
	Height    int64  `json:"height"`
	Hash      string `json:"hash"`
	Timestamp string `json:"timestamp"`
	Proposer  string `json:"proposer"`
	TxCount   int    `json:"tx_count"`
}

// GetLatestBlock retrieves the latest block.
func (c *Client) GetLatestBlock(ctx context.Context) (*BlockInfo, error) {
	url := fmt.Sprintf("%s/block", c.config.RPCEndpoint)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Result struct {
			Block struct {
				Header struct {
					Height          string `json:"height"`
					Time            string `json:"time"`
					ProposerAddress string `json:"proposer_address"`
				} `json:"header"`
				Data struct {
					Txs []string `json:"txs"`
				} `json:"data"`
			} `json:"block"`
			BlockID struct {
				Hash string `json:"hash"`
			} `json:"block_id"`
		} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	var height int64
	fmt.Sscanf(result.Result.Block.Header.Height, "%d", &height)

	return &BlockInfo{
		Height:    height,
		Hash:      result.Result.BlockID.Hash,
		Timestamp: result.Result.Block.Header.Time,
		Proposer:  result.Result.Block.Header.ProposerAddress,
		TxCount:   len(result.Result.Block.Data.Txs),
	}, nil
}

// Balance represents account balance.
type Balance struct {
	Denom  string `json:"denom"`
	Amount string `json:"amount"`
}

// GetBalance retrieves account balance.
func (c *Client) GetBalance(ctx context.Context, address string) ([]Balance, error) {
	url := fmt.Sprintf("%s/cosmos/bank/v1beta1/balances/%s", c.config.RESTEndpoint, address)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Balances []Balance `json:"balances"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Balances, nil
}
