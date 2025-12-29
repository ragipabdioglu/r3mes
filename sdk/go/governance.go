// Package r3mes provides governance and staking operations for the R3MES network.
package r3mes

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Proposal represents a governance proposal.
type Proposal struct {
	ProposalID       string    `json:"proposal_id"`
	Content          any       `json:"content"`
	Status           string    `json:"status"`
	FinalTallyResult Tally     `json:"final_tally_result"`
	SubmitTime       string    `json:"submit_time"`
	DepositEndTime   string    `json:"deposit_end_time"`
	TotalDeposit     []Balance `json:"total_deposit"`
	VotingStartTime  string    `json:"voting_start_time"`
	VotingEndTime    string    `json:"voting_end_time"`
}

// Tally represents vote tally results.
type Tally struct {
	Yes        string `json:"yes"`
	Abstain    string `json:"abstain"`
	No         string `json:"no"`
	NoWithVeto string `json:"no_with_veto"`
}

// Vote represents a vote on a proposal.
type Vote struct {
	ProposalID string `json:"proposal_id"`
	Voter      string `json:"voter"`
	Option     string `json:"option"`
}

// GetProposals retrieves governance proposals.
func (c *Client) GetProposals(ctx context.Context, status string, limit int) ([]Proposal, error) {
	url := fmt.Sprintf("%s/cosmos/gov/v1beta1/proposals?pagination.limit=%d", c.config.RESTEndpoint, limit)
	if status != "" {
		url += fmt.Sprintf("&proposal_status=%s", status)
	}

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
		Proposals []Proposal `json:"proposals"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Proposals, nil
}

// GetProposal retrieves a specific proposal by ID.
func (c *Client) GetProposal(ctx context.Context, proposalID string) (*Proposal, error) {
	url := fmt.Sprintf("%s/cosmos/gov/v1beta1/proposals/%s", c.config.RESTEndpoint, proposalID)

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
		return nil, ErrProposalNotFound
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Proposal Proposal `json:"proposal"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result.Proposal, nil
}

// GetProposalTally retrieves the current tally for a proposal.
func (c *Client) GetProposalTally(ctx context.Context, proposalID string) (*Tally, error) {
	url := fmt.Sprintf("%s/cosmos/gov/v1beta1/proposals/%s/tally", c.config.RESTEndpoint, proposalID)

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
		Tally Tally `json:"tally"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result.Tally, nil
}

// GetValidatorsForGovernance retrieves validators (uses Validator type from blockchain.go).
func (c *Client) GetValidatorsForGovernance(ctx context.Context, status string, limit int) ([]Validator, error) {
	url := fmt.Sprintf("%s/cosmos/staking/v1beta1/validators?status=%s&pagination.limit=%d",
		c.config.RESTEndpoint, status, limit)

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

// Delegation represents a delegation.
type Delegation struct {
	DelegatorAddress string  `json:"delegator_address"`
	ValidatorAddress string  `json:"validator_address"`
	Shares           string  `json:"shares"`
	Balance          Balance `json:"balance"`
}

// GetDelegations retrieves delegations for a delegator.
func (c *Client) GetDelegations(ctx context.Context, delegatorAddress string, limit int) ([]Delegation, error) {
	url := fmt.Sprintf("%s/cosmos/staking/v1beta1/delegations/%s?pagination.limit=%d",
		c.config.RESTEndpoint, delegatorAddress, limit)

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
		DelegationResponses []struct {
			Delegation struct {
				DelegatorAddress string `json:"delegator_address"`
				ValidatorAddress string `json:"validator_address"`
				Shares           string `json:"shares"`
			} `json:"delegation"`
			Balance Balance `json:"balance"`
		} `json:"delegation_responses"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	delegations := make([]Delegation, len(result.DelegationResponses))
	for i, d := range result.DelegationResponses {
		delegations[i] = Delegation{
			DelegatorAddress: d.Delegation.DelegatorAddress,
			ValidatorAddress: d.Delegation.ValidatorAddress,
			Shares:           d.Delegation.Shares,
			Balance:          d.Balance,
		}
	}

	return delegations, nil
}

// StakingPool represents the staking pool.
type StakingPool struct {
	BondedTokens    string `json:"bonded_tokens"`
	NotBondedTokens string `json:"not_bonded_tokens"`
}

// GetStakingPool retrieves the staking pool information.
func (c *Client) GetStakingPool(ctx context.Context) (*StakingPool, error) {
	url := fmt.Sprintf("%s/cosmos/staking/v1beta1/pool", c.config.RESTEndpoint)

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
		Pool StakingPool `json:"pool"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result.Pool, nil
}

// Rewards represents delegation rewards.
type Rewards struct {
	Rewards []struct {
		ValidatorAddress string    `json:"validator_address"`
		Reward           []Balance `json:"reward"`
	} `json:"rewards"`
	Total []Balance `json:"total"`
}

// GetRewards retrieves delegation rewards for a delegator.
func (c *Client) GetRewards(ctx context.Context, delegatorAddress string) (*Rewards, error) {
	url := fmt.Sprintf("%s/cosmos/distribution/v1beta1/delegators/%s/rewards",
		c.config.RESTEndpoint, delegatorAddress)

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

	var rewards Rewards
	if err := json.NewDecoder(resp.Body).Decode(&rewards); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &rewards, nil
}
