package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

// Proposal represents a governance proposal
type Proposal struct {
	ID           string `json:"proposal_id"`
	Title        string `json:"title"`
	Description  string `json:"description"`
	Status       string `json:"status"`
	VotingStart  string `json:"voting_start_time"`
	VotingEnd    string `json:"voting_end_time"`
	TotalYes     string `json:"yes_votes"`
	TotalNo      string `json:"no_votes"`
	TotalAbstain string `json:"abstain_votes"`
	TotalVeto    string `json:"no_with_veto_votes"`
}

// VoteOption represents a vote choice
type VoteOption string

const (
	VoteYes        VoteOption = "VOTE_OPTION_YES"
	VoteNo         VoteOption = "VOTE_OPTION_NO"
	VoteAbstain    VoteOption = "VOTE_OPTION_ABSTAIN"
	VoteNoWithVeto VoteOption = "VOTE_OPTION_NO_WITH_VETO"
)

func newGovernanceCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:     "governance",
		Aliases: []string{"gov"},
		Short:   "Governance operations",
		Long:    "Participate in R3MES network governance.",
	}

	cmd.AddCommand(newGovVoteCmd())
	cmd.AddCommand(newGovProposalsCmd())
	cmd.AddCommand(newGovProposalCmd())

	return cmd
}

func newGovVoteCmd() *cobra.Command {
	var (
		memo     string
		gasLimit uint64
	)

	cmd := &cobra.Command{
		Use:   "vote <proposal_id> <yes|no|abstain|no_with_veto>",
		Short: "Vote on a proposal",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			proposalID := args[0]
			vote := args[1]
			return voteOnProposalCobra(ctx, proposalID, vote, memo, gasLimit)
		},
	}

	cmd.Flags().StringVar(&memo, "memo", "", "Transaction memo")
	cmd.Flags().Uint64Var(&gasLimit, "gas", 200000, "Gas limit")

	return cmd
}

func newGovProposalsCmd() *cobra.Command {
	var status string

	cmd := &cobra.Command{
		Use:   "proposals",
		Short: "List governance proposals",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return listProposalsCobra(ctx, status)
		},
	}

	cmd.Flags().StringVar(&status, "status", "", "Filter by status (voting, passed, rejected, deposit)")

	return cmd
}

func newGovProposalCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "proposal <proposal_id>",
		Short: "Get proposal details",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return getProposalCobra(ctx, args[0])
		},
	}
}

func voteOnProposalCobra(ctx context.Context, proposalID, vote, memo string, gasLimit uint64) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	if config.RPCEndpoint == "" {
		return fmt.Errorf("RPC endpoint not configured. Run: r3mes config init")
	}

	// Validate vote option
	voteOption, err := parseVoteOption(vote)
	if err != nil {
		return err
	}

	// Load wallet
	wallet, err := LoadDefaultWallet(config)
	if err != nil {
		return fmt.Errorf("no wallet found. Create one with: r3mes wallet create")
	}

	logger.Info("Voting on proposal",
		zap.String("proposal_id", proposalID),
		zap.String("vote", string(voteOption)),
		zap.String("voter", wallet.Address))

	// Get private key
	privateKey, err := getPrivateKey(wallet)
	if err != nil {
		return fmt.Errorf("failed to get private key: %w", err)
	}

	// Get account info
	accountNum, sequence, err := getAccountInfo(ctx, config, wallet.Address)
	if err != nil {
		return fmt.Errorf("failed to get account info: %w", err)
	}

	// Build vote message
	proposalIDInt, _ := strconv.ParseUint(proposalID, 10, 64)
	voteMsg := buildVoteMsg(proposalIDInt, wallet.Address, voteOption)

	// Create and sign transaction
	builder := &TxBuilder{
		ChainID:    config.ChainID,
		AccountNum: accountNum,
		Sequence:   sequence,
		GasLimit:   gasLimit,
		Memo:       memo,
	}

	signedTx, err := builder.BuildAndSign([]json.RawMessage{voteMsg}, privateKey)
	if err != nil {
		return fmt.Errorf("failed to sign transaction: %w", err)
	}

	// Broadcast
	result, err := broadcastTx(ctx, config, signedTx)
	if err != nil {
		return fmt.Errorf("failed to broadcast: %w", err)
	}

	if result.TxResponse.Code != 0 {
		return fmt.Errorf("transaction failed: %s", result.TxResponse.RawLog)
	}

	logger.Info("Vote submitted successfully",
		zap.String("txhash", result.TxResponse.TxHash))

	fmt.Printf("✅ Vote submitted!\n")
	fmt.Printf("   Proposal: #%s\n", proposalID)
	fmt.Printf("   Vote: %s\n", vote)
	fmt.Printf("   TxHash: %s\n", result.TxResponse.TxHash)

	return nil
}

func listProposalsCobra(ctx context.Context, statusFilter string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	if config.RPCEndpoint == "" {
		return fmt.Errorf("RPC endpoint not configured")
	}

	logger.Debug("Fetching proposals", zap.String("status", statusFilter))

	url := fmt.Sprintf("%s/cosmos/gov/v1beta1/proposals", getRESTEndpoint(config.RPCEndpoint))
	if statusFilter != "" {
		statusCode := mapStatusFilter(statusFilter)
		if statusCode != "" {
			url += "?proposal_status=" + statusCode
		}
	}

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to fetch proposals: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var result struct {
		Proposals []struct {
			ProposalID string `json:"proposal_id"`
			Content    struct {
				Type        string `json:"@type"`
				Title       string `json:"title"`
				Description string `json:"description"`
			} `json:"content"`
			Status          string `json:"status"`
			VotingStartTime string `json:"voting_start_time"`
			VotingEndTime   string `json:"voting_end_time"`
		} `json:"proposals"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	if config.JSONOutput {
		data, _ := json.MarshalIndent(result.Proposals, "", "  ")
		fmt.Println(string(data))
		return nil
	}

	fmt.Println("\nGovernance Proposals:")
	if len(result.Proposals) == 0 {
		fmt.Println("  No proposals found")
	} else {
		for _, p := range result.Proposals {
			statusIcon := getStatusIcon(p.Status)
			fmt.Printf("  #%s: %s %s [%s]\n", p.ProposalID, statusIcon, p.Content.Title, formatStatus(p.Status))
		}
	}

	return nil
}

func getProposalCobra(ctx context.Context, proposalID string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	if config.RPCEndpoint == "" {
		return fmt.Errorf("RPC endpoint not configured")
	}

	logger.Debug("Fetching proposal", zap.String("id", proposalID))

	url := fmt.Sprintf("%s/cosmos/gov/v1beta1/proposals/%s", getRESTEndpoint(config.RPCEndpoint), proposalID)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to fetch proposal: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if config.JSONOutput {
		var prettyJSON map[string]any
		json.Unmarshal(body, &prettyJSON)
		formatted, _ := json.MarshalIndent(prettyJSON, "", "  ")
		fmt.Println(string(formatted))
		return nil
	}

	var result struct {
		Proposal struct {
			ProposalID string `json:"proposal_id"`
			Content    struct {
				Title       string `json:"title"`
				Description string `json:"description"`
			} `json:"content"`
			Status           string `json:"status"`
			VotingStartTime  string `json:"voting_start_time"`
			VotingEndTime    string `json:"voting_end_time"`
			FinalTallyResult struct {
				Yes        string `json:"yes"`
				No         string `json:"no"`
				Abstain    string `json:"abstain"`
				NoWithVeto string `json:"no_with_veto"`
			} `json:"final_tally_result"`
		} `json:"proposal"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	p := result.Proposal
	fmt.Printf("\nProposal #%s\n", p.ProposalID)
	fmt.Printf("Title: %s\n", p.Content.Title)
	fmt.Printf("Status: %s %s\n", getStatusIcon(p.Status), formatStatus(p.Status))
	fmt.Printf("\nDescription:\n%s\n", p.Content.Description)

	if p.VotingStartTime != "" {
		fmt.Printf("\nVoting Period:\n")
		fmt.Printf("  Start: %s\n", formatTime(p.VotingStartTime))
		fmt.Printf("  End:   %s\n", formatTime(p.VotingEndTime))
	}

	if p.FinalTallyResult.Yes != "" {
		fmt.Printf("\nTally Results:\n")
		fmt.Printf("  Yes:          %s\n", p.FinalTallyResult.Yes)
		fmt.Printf("  No:           %s\n", p.FinalTallyResult.No)
		fmt.Printf("  Abstain:      %s\n", p.FinalTallyResult.Abstain)
		fmt.Printf("  No with Veto: %s\n", p.FinalTallyResult.NoWithVeto)
	}

	return nil
}

func parseVoteOption(vote string) (VoteOption, error) {
	switch strings.ToLower(vote) {
	case "yes", "y":
		return VoteYes, nil
	case "no", "n":
		return VoteNo, nil
	case "abstain", "a":
		return VoteAbstain, nil
	case "no_with_veto", "veto", "nwv":
		return VoteNoWithVeto, nil
	default:
		return "", fmt.Errorf("invalid vote option: %s. Use: yes, no, abstain, or no_with_veto", vote)
	}
}

func buildVoteMsg(proposalID uint64, voter string, option VoteOption) json.RawMessage {
	msg := map[string]any{
		"@type":       "/cosmos.gov.v1beta1.MsgVote",
		"proposal_id": strconv.FormatUint(proposalID, 10),
		"voter":       voter,
		"option":      string(option),
	}
	data, _ := json.Marshal(msg)
	return data
}

func mapStatusFilter(status string) string {
	switch strings.ToLower(status) {
	case "voting":
		return "PROPOSAL_STATUS_VOTING_PERIOD"
	case "passed":
		return "PROPOSAL_STATUS_PASSED"
	case "rejected":
		return "PROPOSAL_STATUS_REJECTED"
	case "deposit":
		return "PROPOSAL_STATUS_DEPOSIT_PERIOD"
	default:
		return ""
	}
}

func formatStatus(status string) string {
	switch status {
	case "PROPOSAL_STATUS_VOTING_PERIOD":
		return "Voting"
	case "PROPOSAL_STATUS_PASSED":
		return "Passed"
	case "PROPOSAL_STATUS_REJECTED":
		return "Rejected"
	case "PROPOSAL_STATUS_DEPOSIT_PERIOD":
		return "Deposit"
	case "PROPOSAL_STATUS_FAILED":
		return "Failed"
	default:
		return status
	}
}

func getStatusIcon(status string) string {
	switch status {
	case "PROPOSAL_STATUS_VOTING_PERIOD":
		return "🗳️"
	case "PROPOSAL_STATUS_PASSED":
		return "✅"
	case "PROPOSAL_STATUS_REJECTED":
		return "❌"
	case "PROPOSAL_STATUS_DEPOSIT_PERIOD":
		return "💰"
	default:
		return "❓"
	}
}

func formatTime(timeStr string) string {
	t, err := time.Parse(time.RFC3339, timeStr)
	if err != nil {
		return timeStr
	}
	return t.Local().Format("2006-01-02 15:04:05")
}
