package r3mes

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
)

// MinerClient provides methods for miner-related operations.
type MinerClient struct {
	client *Client
}

// NewMinerClient creates a new MinerClient.
func (c *Client) Miner() *MinerClient {
	return &MinerClient{client: c}
}

// EarningsRecord represents a single earnings record.
type EarningsRecord struct {
	Amount     float64 `json:"amount"`
	RecordedAt string  `json:"recorded_at"`
	Source     string  `json:"source"`
}

// GetEarningsHistory retrieves miner earnings history.
func (m *MinerClient) GetEarningsHistory(ctx context.Context, walletAddress string, limit, offset int) ([]EarningsRecord, error) {
	params := url.Values{}
	params.Set("limit", strconv.Itoa(limit))
	params.Set("offset", strconv.Itoa(offset))

	reqURL := fmt.Sprintf("%s/miner/earnings/%s?%s", m.client.config.BackendEndpoint, walletAddress, params.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := m.client.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Earnings []EarningsRecord `json:"earnings"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Earnings, nil
}

// HashrateRecord represents a single hashrate record.
type HashrateRecord struct {
	Hashrate   float64 `json:"hashrate"`
	RecordedAt string  `json:"recorded_at"`
}

// GetHashrateHistory retrieves miner hashrate history.
func (m *MinerClient) GetHashrateHistory(ctx context.Context, walletAddress string, hours int) ([]HashrateRecord, error) {
	params := url.Values{}
	params.Set("hours", strconv.Itoa(hours))

	reqURL := fmt.Sprintf("%s/miner/hashrate/%s?%s", m.client.config.BackendEndpoint, walletAddress, params.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := m.client.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Hashrate []HashrateRecord `json:"hashrate"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Hashrate, nil
}

// LeaderboardEntry represents a single leaderboard entry.
type LeaderboardEntry struct {
	Rank          int     `json:"rank"`
	WalletAddress string  `json:"wallet_address"`
	TotalEarnings float64 `json:"total_earnings"`
	Hashrate      float64 `json:"hashrate"`
	BlocksFound   int64   `json:"blocks_found"`
}

// GetLeaderboard retrieves the miner leaderboard.
func (m *MinerClient) GetLeaderboard(ctx context.Context, limit int, period string) ([]LeaderboardEntry, error) {
	params := url.Values{}
	params.Set("limit", strconv.Itoa(limit))
	params.Set("period", period)

	reqURL := fmt.Sprintf("%s/leaderboard?%s", m.client.config.BackendEndpoint, params.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := m.client.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Miners []LeaderboardEntry `json:"miners"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Miners, nil
}

// TaskPool represents a task pool.
type TaskPool struct {
	ID              int64  `json:"id"`
	DatasetIPFSHash string `json:"dataset_ipfs_hash"`
	TotalTasks      int    `json:"total_tasks"`
	CompletedTasks  int    `json:"completed_tasks"`
	Status          string `json:"status"`
	CreatedAt       string `json:"created_at"`
}

// GetActivePool retrieves the currently active task pool.
func (m *MinerClient) GetActivePool(ctx context.Context) (*TaskPool, error) {
	reqURL := fmt.Sprintf("%s/training/active-pool", m.client.config.BackendEndpoint)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := m.client.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, nil // No active pool
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var pool TaskPool
	if err := json.NewDecoder(resp.Body).Decode(&pool); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &pool, nil
}
