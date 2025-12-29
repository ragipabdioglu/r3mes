package keeper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"strings"

	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// FinalizeDatasetProposals finalizes dataset proposals that have passed voting period
// This should be called in EndBlocker
func (k Keeper) FinalizeDatasetProposals(ctx sdk.Context) error {
	// Iterate through all proposals
	iter, err := k.DatasetProposals.Iterate(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to iterate proposals: %w", err)
	}
	defer iter.Close()

	for ; iter.Valid(); iter.Next() {
		proposal, err := iter.Value()
		if err != nil {
			continue
		}

		// Skip if already finalized
		if proposal.Status == "executed" || proposal.Status == "rejected" {
			continue
		}

		// Check if voting period has ended
		if ctx.BlockTime().Before(proposal.VotingPeriodEnd) {
			continue
		}

		// Finalize proposal
		if err := k.finalizeProposal(ctx, proposal); err != nil {
			ctx.Logger().Error(fmt.Sprintf("Failed to finalize proposal %d: %v", proposal.ProposalId, err))
			continue
		}
	}

	return nil
}

// finalizeProposal finalizes a single proposal based on voting results
func (k Keeper) finalizeProposal(ctx sdk.Context, proposal types.DatasetProposal) error {
	totalVotesInt, _ := sdkmath.NewIntFromString(proposal.TotalVotes)
	yesVotesInt, _ := sdkmath.NewIntFromString(proposal.YesVotes)
	noVotesInt, _ := sdkmath.NewIntFromString(proposal.NoVotes)

	// Threshold: Simple majority + minimum quorum (1M tokens)
	quorumThreshold := sdkmath.NewInt(1000000)
	passThreshold := totalVotesInt.Quo(sdkmath.NewInt(2)) // 50% + 1

	if totalVotesInt.GTE(quorumThreshold) && yesVotesInt.GT(noVotesInt) && yesVotesInt.GT(passThreshold) {
		// Proposal passed - add to approved datasets
		datasetID, err := k.ApprovedDatasetID.Next(ctx)
		if err != nil {
			return fmt.Errorf("failed to generate dataset ID: %w", err)
		}

		// Create audit trail entry
		approvalEntry := &types.AuditTrailEntry{
			EntryType:   "approval",
			BlockHeight: ctx.BlockHeight(),
			TxHash:      hex.EncodeToString(ctx.TxBytes()),
			Actor:       proposal.Proposer,
			Description: "Dataset approved through governance voting",
			Timestamp:   ctx.BlockTime(),
		}

		approvedDataset := types.ApprovedDataset{
			DatasetId:              datasetID,
			DatasetIpfsHash:        proposal.DatasetIpfsHash,
			Metadata:               proposal.Metadata,
			ApprovalHeight:         ctx.BlockHeight(),
			ApprovalTxHash:         hex.EncodeToString(ctx.TxBytes()),
			Proposer:               proposal.Proposer,
			Status:                 "active",
			IsOfficialTrainingData: true,
			RemovalHeight:          0,
			RemovalTxHash:          "",
			AuditTrail:             []*types.AuditTrailEntry{approvalEntry},
		}

		if err := k.ApprovedDatasets.Set(ctx, datasetID, approvedDataset); err != nil {
			return fmt.Errorf("failed to store approved dataset: %w", err)
		}

		proposal.Status = "executed"
	} else {
		// Proposal rejected
		proposal.Status = "rejected"
	}

	// Update proposal status
	if err := k.DatasetProposals.Set(ctx, proposal.ProposalId, proposal); err != nil {
		return fmt.Errorf("failed to update proposal: %w", err)
	}

	return nil
}

// CalculateVotingPower calculates voting power for a voter
// Supports stake-weighted and quadratic voting
func (k Keeper) CalculateVotingPower(
	ctx sdk.Context,
	voter sdk.AccAddress,
	votingMethod string, // "stake_weighted", "quadratic", "simple"
) (sdkmath.Int, error) {
	switch votingMethod {
	case "stake_weighted":
		// Stake-weighted: voting power = sqrt(stake)
		balance := k.bankKeeper.GetBalance(ctx, voter, "stake")
		if balance.Amount.IsZero() {
			return sdkmath.ZeroInt(), nil
		}
		// Simplified: use stake directly (in production, use sqrt for quadratic)
		return balance.Amount, nil

	case "quadratic":
		// Quadratic voting: voting power = sqrt(stake)
		balance := k.bankKeeper.GetBalance(ctx, voter, "stake")
		if balance.Amount.IsZero() {
			return sdkmath.ZeroInt(), nil
		}
		// Simplified: use sqrt (in production, implement proper sqrt)
		// For now, use stake directly
		return balance.Amount, nil

	case "simple":
		// Simple: 1 vote per address
		return sdkmath.OneInt(), nil

	default:
		// Default: stake-weighted
		balance := k.bankKeeper.GetBalance(ctx, voter, "stake")
		return balance.Amount, nil
	}
}

// VerifyDatasetIntegrity verifies dataset integrity using cryptographic checksum
func (k Keeper) VerifyDatasetIntegrity(
	ctx context.Context,
	datasetIPFSHash string,
	expectedChecksum string,
) (bool, error) {
	// 1. Validate inputs
	if datasetIPFSHash == "" {
		return false, fmt.Errorf("dataset IPFS hash cannot be empty")
	}
	if expectedChecksum == "" {
		return false, fmt.Errorf("expected checksum cannot be empty")
	}

	// 2. Check if IPFS manager is available
	if k.ipfsManager == nil {
		// CRITICAL: Fail-closed security model
		// In production, IPFS manager must be configured for dataset verification
		// If not available, we must reject the verification to prevent security vulnerabilities
		isTestMode := os.Getenv("R3MES_TEST_MODE") == "true"
		if isTestMode {
			// Only allow in test mode with explicit flag
			// Note: ctx is context.Context, not sdk.Context, so we can't use Logger()
			// Just return error with clear message
			return false, fmt.Errorf("IPFS manager not configured - dataset verification cannot proceed (TEST MODE ONLY)")
		}
		// Production mode: fail-fast
		return false, fmt.Errorf("IPFS manager not configured - dataset verification required in production")
	}

	// 3. Download dataset from IPFS
	datasetData, err := k.ipfsManager.RetrieveContent(ctx, datasetIPFSHash)
	if err != nil {
		return false, fmt.Errorf("failed to retrieve dataset from IPFS: %w", err)
	}

	if len(datasetData) == 0 {
		return false, fmt.Errorf("dataset retrieved from IPFS is empty")
	}

	// 4. Calculate SHA256 checksum
	calculatedChecksum := CalculateDatasetChecksum(datasetData)

	// 5. Compare with expected checksum (case-insensitive comparison)
	expectedChecksumLower := strings.ToLower(expectedChecksum)
	calculatedChecksumLower := strings.ToLower(calculatedChecksum)

	if expectedChecksumLower != calculatedChecksumLower {
		return false, fmt.Errorf(
			"checksum mismatch: expected %s, got %s",
			expectedChecksum,
			calculatedChecksum,
		)
	}

	// 6. Integrity verified
	return true, nil
}

// CalculateDatasetChecksum calculates SHA256 checksum of dataset
func CalculateDatasetChecksum(datasetData []byte) string {
	hash := sha256.Sum256(datasetData)
	return hex.EncodeToString(hash[:])
}

// ValidateDatasetMetadata validates dataset metadata
func (k Keeper) ValidateDatasetMetadata(metadata types.DatasetMetadata) error {
	if metadata.Name == "" {
		return fmt.Errorf("dataset name cannot be empty")
	}
	if metadata.Description == "" {
		return fmt.Errorf("dataset description cannot be empty")
	}
	if metadata.SizeBytes == 0 {
		return fmt.Errorf("dataset size cannot be zero")
	}
	if metadata.NumSamples == 0 {
		return fmt.Errorf("dataset must have at least one sample")
	}
	if metadata.Checksum == "" {
		return fmt.Errorf("dataset checksum is required")
	}
	return nil
}

