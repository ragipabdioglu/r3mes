package keeper

import (
	"context"
	"fmt"
	"strings"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// SubmitAggregation handles MsgSubmitAggregation
// Processes off-chain aggregation results and stores them on-chain
func (k msgServer) SubmitAggregation(ctx context.Context, msg *types.MsgSubmitAggregation) (*types.MsgSubmitAggregationResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate proposer address
	proposerAddr, err := k.addressCodec.StringToBytes(msg.Proposer)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid proposer address: %s", msg.Proposer)
	}
	_ = proposerAddr // Address validated

	// 2. Validate aggregated gradient IPFS hash
	aggregatedHash := strings.TrimSpace(msg.AggregatedGradientIpfsHash)
	if aggregatedHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "aggregated gradient IPFS hash cannot be empty")
	}
	if !strings.HasPrefix(aggregatedHash, "Qm") && !strings.HasPrefix(aggregatedHash, "bafy") && !strings.HasPrefix(aggregatedHash, "bafk") {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", aggregatedHash)
	}

	// 3. Validate Merkle root
	merkleRoot := strings.TrimSpace(msg.MerkleRoot)
	if merkleRoot == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "merkle root cannot be empty")
	}

	// 4. Validate participant gradient IDs
	if len(msg.ParticipantGradientIds) == 0 {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "participant gradient IDs cannot be empty")
	}

	// 5. Verify all participant gradients exist and are in pending status
	for _, gradientID := range msg.ParticipantGradientIds {
		gradient, err := k.StoredGradients.Get(ctx, gradientID)
		if err != nil {
			return nil, errorsmod.Wrapf(err, "gradient %d not found", gradientID)
		}
		if gradient.Status != "pending" {
			return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "gradient %d is not in pending status (current: %s)", gradientID, gradient.Status)
		}
	}

	// 6. Validate model version
	if msg.ModelVersion == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidModelVersion, "model version cannot be empty")
	}

	// 7. Verify Merkle root matches participant gradients
	// Reconstruct Merkle tree from stored gradient hashes
	gradientHashes := make([]string, 0, len(msg.ParticipantGradientIds))
	for _, gradientID := range msg.ParticipantGradientIds {
		gradient, err := k.StoredGradients.Get(ctx, gradientID)
		if err != nil {
			return nil, errorsmod.Wrapf(err, "gradient %d not found for Merkle verification", gradientID)
		}
		gradientHashes = append(gradientHashes, gradient.GradientHash)
	}

	// Calculate expected Merkle root
	calculatedRoot, err := k.CalculateMerkleRoot(gradientHashes)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to calculate Merkle root")
	}

	// Verify claimed root matches calculated root
	if calculatedRoot != merkleRoot {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidGradientHash,
			"merkle root mismatch: expected %s, got %s",
			calculatedRoot,
			merkleRoot,
		)
	}

	// 8. Generate unique aggregation ID
	aggregationID, err := k.AggregationID.Next(ctx)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to generate aggregation ID")
	}

	// 9. Calculate challenge deadline - get from params
	params, err := k.Params.Get(ctx)
	if err != nil {
		// If params not set, use default
		params = types.DefaultParams()
	}
	challengePeriodBlocks := params.ChallengePeriodBlocks
	if challengePeriodBlocks == 0 {
		// Fallback to default if not set
		challengePeriodBlocks = 100
	}
	challengeDeadlineHeight := sdkCtx.BlockHeight() + challengePeriodBlocks

	// 10. Create AggregationRecord
	aggregationRecord := types.AggregationRecord{
		AggregationId:              aggregationID,
		Proposer:                   msg.Proposer,
		AggregatedGradientIpfsHash: aggregatedHash,
		MerkleRoot:                 merkleRoot,
		ParticipantGradientIds:     msg.ParticipantGradientIds,
		TrainingRoundId:            msg.TrainingRoundId,
		ModelVersion:               msg.ModelVersion,
		SubmittedAtHeight:          sdkCtx.BlockHeight(),
		Status:                     "pending", // Will be finalized after challenge period
		ChallengeDeadlineHeight:    challengeDeadlineHeight,
	}

	// 11. Store aggregation record on-chain
	if err := k.AggregationRecords.Set(ctx, aggregationID, aggregationRecord); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store aggregation record")
	}

	// 11.5. Add aggregation to deadline index for efficient end-blocker lookup
	if err := k.AddAggregationToDeadlineIndex(ctx, challengeDeadlineHeight, aggregationID); err != nil {
		// Log error but don't fail transaction (index is for optimization)
		sdkCtx.Logger().Warn("Failed to add aggregation to deadline index", "aggregation_id", aggregationID, "error", err)
	}

	// 12. Update stored gradients status to "aggregated" (atomic operation)
	// Re-verify status before update to prevent race conditions
	for _, gradientID := range msg.ParticipantGradientIds {
		gradient, err := k.StoredGradients.Get(ctx, gradientID)
		if err != nil {
			// Should not happen as we verified above, but handle gracefully
			continue
		}
		
		// Atomic status update: only update if still "pending"
		// This prevents race conditions where the same gradient is used in multiple aggregations
		if gradient.Status != "pending" {
			return nil, errorsmod.Wrapf(
				types.ErrInvalidGradientHash,
				"gradient %d is no longer in pending status (current: %s), may have been used in another aggregation",
				gradientID,
				gradient.Status,
			)
		}
		
		// Update status atomically
		gradient.Status = "aggregated"
		if err := k.StoredGradients.Set(ctx, gradientID, gradient); err != nil {
			return nil, errorsmod.Wrapf(err, "failed to update gradient %d status", gradientID)
		}
	}

	// 13. Distribute proposer reward
	// Reward distribution is critical for economic incentives
	// If it fails, the transaction should fail to ensure consistency
	participantCount := uint64(len(msg.ParticipantGradientIds))
	if err := k.DistributeProposerReward(ctx, msg.Proposer, participantCount); err != nil {
		return nil, errorsmod.Wrap(err, "failed to distribute proposer reward")
	}

	// 14. Update global model state if this aggregation should become the new global model
	// Strategy: Update if this aggregation has a higher training round ID than current global model,
	// or same training round but this is the latest aggregation
	if err := k.UpdateGlobalModelStateIfNeeded(sdkCtx, aggregationRecord); err != nil {
		// Log error but don't fail transaction (model update is not critical for aggregation acceptance)
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to update global model state: %v", err))
	}

	// 15. Emit event for aggregation submission
	sdkCtx.EventManager().EmitEvent(
		types.NewEventAggregationSubmitted(
			msg.Proposer,
			aggregationID,
			msg.TrainingRoundId,
		),
	)

	return &types.MsgSubmitAggregationResponse{
		AggregationId: aggregationID,
	}, nil
}

