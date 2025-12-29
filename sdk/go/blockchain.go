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

// BlockchainClient provides methods for blockchain-related operations.
type BlockchainClient struct {
	client *Client
}

// NewBlockchainClient creates a new BlockchainClient.
func (c *Client) Blockchain() *BlockchainClient {
	return &BlockchainClient{client: c}
}

// GetBlock retrieves a block by height.
func (b *BlockchainClient) GetBlock(ctx context.Context, height int64) (*BlockInfo, error) {
	params := url.Values{}
	params.Set("height", strconv.FormatInt(height, 10))

	reqURL := fmt.Sprintf("%s/block?%s", b.client.config.RPCEndpoint, params.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := b.client.httpClient.Do(req)
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

	var blockHeight int64
	fmt.Sscanf(result.Result.Block.Header.Height, "%d", &blockHeight)

	return &BlockInfo{
		Height:    blockHeight,
		Hash:      result.Result.BlockID.Hash,
		Timestamp: result.Result.Block.Header.Time,
		Proposer:  result.Result.Block.Header.ProposerAddress,
		TxCount:   len(result.Result.Block.Data.Txs),
	}, nil
}

// Validator represents a validator.
type Validator struct {
	OperatorAddress string `json:"operator_address"`
	Moniker         string `json:"moniker"`
	Tokens          string `json:"tokens"`
	Status          string `json:"status"`
	Jailed          bool   `json:"jailed"`
	Commission      string `json:"commission"`
}

// GetValidators retrieves validators.
func (b *BlockchainClient) GetValidators(ctx context.Context, status string, limit int) ([]Validator, error) {
	params := url.Values{}
	params.Set("status", status)
	params.Set("pagination.limit", strconv.Itoa(limit))

	reqURL := fmt.Sprintf("%s/cosmos/staking/v1beta1/validators?%s", b.client.config.RESTEndpoint, params.Encode())

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := b.client.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Validators []struct {
			OperatorAddress string `json:"operator_address"`
			Description     struct {
				Moniker string `json:"moniker"`
			} `json:"description"`
			Tokens     string `json:"tokens"`
			Status     string `json:"status"`
			Jailed     bool   `json:"jailed"`
			Commission struct {
				CommissionRates struct {
					Rate string `json:"rate"`
				} `json:"commission_rates"`
			} `json:"commission"`
		} `json:"validators"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	validators := make([]Validator, len(result.Validators))
	for i, v := range result.Validators {
		validators[i] = Validator{
			OperatorAddress: v.OperatorAddress,
			Moniker:         v.Description.Moniker,
			Tokens:          v.Tokens,
			Status:          v.Status,
			Jailed:          v.Jailed,
			Commission:      v.Commission.CommissionRates.Rate,
		}
	}

	return validators, nil
}

// Transaction represents a transaction.
type Transaction struct {
	Hash      string `json:"hash"`
	Height    int64  `json:"height"`
	Code      uint32 `json:"code"`
	RawLog    string `json:"raw_log"`
	Timestamp string `json:"timestamp"`
	GasUsed   int64  `json:"gas_used"`
	GasWanted int64  `json:"gas_wanted"`
}

// GetTransaction retrieves a transaction by hash.
func (b *BlockchainClient) GetTransaction(ctx context.Context, txHash string) (*Transaction, error) {
	reqURL := fmt.Sprintf("%s/cosmos/tx/v1beta1/txs/%s", b.client.config.RESTEndpoint, txHash)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := b.client.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, ErrTransactionNotFound
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		TxResponse struct {
			TxHash    string `json:"txhash"`
			Height    string `json:"height"`
			Code      uint32 `json:"code"`
			RawLog    string `json:"raw_log"`
			Timestamp string `json:"timestamp"`
			GasUsed   string `json:"gas_used"`
			GasWanted string `json:"gas_wanted"`
		} `json:"tx_response"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	var height, gasUsed, gasWanted int64
	fmt.Sscanf(result.TxResponse.Height, "%d", &height)
	fmt.Sscanf(result.TxResponse.GasUsed, "%d", &gasUsed)
	fmt.Sscanf(result.TxResponse.GasWanted, "%d", &gasWanted)

	return &Transaction{
		Hash:      result.TxResponse.TxHash,
		Height:    height,
		Code:      result.TxResponse.Code,
		RawLog:    result.TxResponse.RawLog,
		Timestamp: result.TxResponse.Timestamp,
		GasUsed:   gasUsed,
		GasWanted: gasWanted,
	}, nil
}

// NodeStatus represents node status.
type NodeStatus struct {
	NodeID            string `json:"node_id"`
	Network           string `json:"network"`
	LatestBlockHeight int64  `json:"latest_block_height"`
	LatestBlockTime   string `json:"latest_block_time"`
	CatchingUp        bool   `json:"catching_up"`
}

// GetStatus retrieves node status.
func (b *BlockchainClient) GetStatus(ctx context.Context) (*NodeStatus, error) {
	reqURL := fmt.Sprintf("%s/status", b.client.config.RPCEndpoint)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := b.client.httpClient.Do(req)
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
			NodeInfo struct {
				ID      string `json:"id"`
				Network string `json:"network"`
			} `json:"node_info"`
			SyncInfo struct {
				LatestBlockHeight string `json:"latest_block_height"`
				LatestBlockTime   string `json:"latest_block_time"`
				CatchingUp        bool   `json:"catching_up"`
			} `json:"sync_info"`
		} `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	var height int64
	fmt.Sscanf(result.Result.SyncInfo.LatestBlockHeight, "%d", &height)

	return &NodeStatus{
		NodeID:            result.Result.NodeInfo.ID,
		Network:           result.Result.NodeInfo.Network,
		LatestBlockHeight: height,
		LatestBlockTime:   result.Result.SyncInfo.LatestBlockTime,
		CatchingUp:        result.Result.SyncInfo.CatchingUp,
	}, nil
}
