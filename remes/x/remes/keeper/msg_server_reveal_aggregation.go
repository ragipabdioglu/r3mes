package keeper

import (
	"context"
	"crypto/sha256"
	"fmt"
	"strings"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// RevealAggregation handles MsgRevealAggregation
// Proposers reveal their aggregation result and verify commitment
func (k msgServer) RevealAggregation(ctx context.Context, msg *types.MsgRevealAggregation) (*types.MsgRevealAggregationResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate proposer address
	proposerAddr, err := k.addressCodec.StringToBytes(msg.Proposer)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid proposer address: %s", msg.Proposer)
	}
	_ = proposerAddr // Address validated

	// 2. Retrieve commitment
	commitment, err := k.AggregationCommitments.Get(ctx, msg.CommitmentId)
	if err != nil {
		return nil, errorsmod.Wrapf(err, "commitment %d not found", msg.CommitmentId)
	}

	// 3. Verify proposer matches
	if commitment.Proposer != msg.Proposer {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "proposer mismatch: expected %s, got %s", commitment.Proposer, msg.Proposer)
	}

	// 4. Verify commitment is still in "committed" status
	if commitment.Status != "committed" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "commitment %d is not in committed status (current: %s)", msg.CommitmentId, commitment.Status)
	}

	// 5. Verify reveal deadline has not passed
	if sdkCtx.BlockHeight() > commitment.RevealDeadlineHeight {
		// Mark commitment as expired
		commitment.Status = "expired"
		if err := k.AggregationCommitments.Set(ctx, msg.CommitmentId, commitment); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to mark commitment as expired: %v", err))
		}
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "reveal deadline has passed (deadline: %d, current: %d)", commitment.RevealDeadlineHeight, sdkCtx.BlockHeight())
	}

	// 6. Verify training round ID matches
	if commitment.TrainingRoundId != msg.TrainingRoundId {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "training round ID mismatch: expected %d, got %d", commitment.TrainingRoundId, msg.TrainingRoundId)
	}

	// 7. Verify participant gradient IDs match
	if len(commitment.ParticipantGradientIds) != len(msg.ParticipantGradientIds) {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "participant gradient IDs count mismatch: expected %d, got %d", len(commitment.ParticipantGradientIds), len(msg.ParticipantGradientIds))
	}
	for i, id := range commitment.ParticipantGradientIds {
		if id != msg.ParticipantGradientIds[i] {
			return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "participant gradient IDs mismatch at index %d: expected %d, got %d", i, id, msg.ParticipantGradientIds[i])
		}
	}

	// 8. Validate aggregated gradient IPFS hash
	aggregatedHash := strings.TrimSpace(msg.AggregatedGradientIpfsHash)
	if aggregatedHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "aggregated gradient IPFS hash cannot be empty")
	}
	if !strings.HasPrefix(aggregatedHash, "Qm") && !strings.HasPrefix(aggregatedHash, "bafy") && !strings.HasPrefix(aggregatedHash, "bafk") {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", aggregatedHash)
	}

	// 9. Validate Merkle root
	merkleRoot := strings.TrimSpace(msg.MerkleRoot)
	if merkleRoot == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "merkle root cannot be empty")
	}

	// 10. Validate salt
	salt := strings.TrimSpace(msg.Salt)
	if salt == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "salt cannot be empty")
	}

	// 11. Verify commitment hash matches revealed data
	// Reconstruct commitment using revealed salt
	// Commitment: hash(proposer + aggregated_hash + merkle_root + participant_ids + salt)
	commitmentData := fmt.Sprintf("%s:%s:%s:%v:%s", msg.Proposer, aggregatedHash, merkleRoot, msg.ParticipantGradientIds, salt)
	commitmentHashBytes := sha256.Sum256([]byte(commitmentData))
	commitmentHash := fmt.Sprintf("%x", commitmentHashBytes)

	if commitmentHash != commitment.CommitmentHash {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidGradientHash,
			"commitment hash mismatch: expected %s, got %s",
			commitment.CommitmentHash,
			commitmentHash,
		)
	}

	// 12. Verify Merkle root matches participant gradients
	gradientHashes := make([]string, 0, len(msg.ParticipantGradientIds))
	for _, gradientID := range msg.ParticipantGradientIds {
		gradient, err := k.StoredGradients.Get(ctx, gradientID)
		if err != nil {
			return nil, errorsmod.Wrapf(err, "gradient %d not found for Merkle verification", gradientID)
		}
		gradientHashes = append(gradientHashes, gradient.GradientHash)
	}

	calculatedRoot, err := k.CalculateMerkleRoot(gradientHashes)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to calculate Merkle root")
	}

	if calculatedRoot != merkleRoot {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidGradientHash,
			"merkle root mismatch: expected %s, got %s",
			calculatedRoot,
			merkleRoot,
		)
	}

	// 13. Verify all participant gradients are still in pending status
	for _, gradientID := range msg.ParticipantGradientIds {
		gradient, err := k.StoredGradients.Get(ctx, gradientID)
		if err != nil {
			return nil, errorsmod.Wrapf(err, "gradient %d not found", gradientID)
		}
		if gradient.Status != "pending" {
			return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "gradient %d is not in pending status (current: %s)", gradientID, gradient.Status)
		}
	}

	// 14. Validate model version
	if msg.ModelVersion == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidModelVersion, "model version cannot be empty")
	}

	// 15. Generate unique aggregation ID
	aggregationID, err := k.AggregationID.Next(ctx)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to generate aggregation ID")
	}

	// 16. Calculate challenge deadline - get from params
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

	// 17. Create AggregationRecord
	aggregationRecord := types.AggregationRecord{
		AggregationId:              aggregationID,
		Proposer:                   msg.Proposer,
		AggregatedGradientIpfsHash: aggregatedHash,
		MerkleRoot:                 merkleRoot,
		ParticipantGradientIds:     msg.ParticipantGradientIds,
		TrainingRoundId:            msg.TrainingRoundId,
		ModelVersion:               msg.ModelVersion,
		SubmittedAtHeight:          sdkCtx.BlockHeight(),
		Status:                     "pending",
		ChallengeDeadlineHeight:    challengeDeadlineHeight,
	}

	// 18. Store aggregation record on-chain
	if err := k.AggregationRecords.Set(ctx, aggregationID, aggregationRecord); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store aggregation record")
	}

	// 18.5. Add aggregation to deadline index for efficient end-blocker lookup
	if err := k.AddAggregationToDeadlineIndex(ctx, challengeDeadlineHeight, aggregationID); err != nil {
		// Log error but don't fail transaction (index is for optimization)
		sdkCtx.Logger().Warn("Failed to add aggregation to deadline index", "aggregation_id", aggregationID, "error", err)
	}

	// 19. Update commitment status to "revealed"
	commitment.Status = "revealed"
	commitment.RevealedAggregationId = aggregationID
	if err := k.AggregationCommitments.Set(ctx, msg.CommitmentId, commitment); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update commitment status")
	}

	// 20. Update stored gradients status to "aggregated"
	for _, gradientID := range msg.ParticipantGradientIds {
		gradient, err := k.StoredGradients.Get(ctx, gradientID)
		if err != nil {
			continue
		}
		if gradient.Status != "pending" {
			return nil, errorsmod.Wrapf(
				types.ErrInvalidGradientHash,
				"gradient %d is no longer in pending status (current: %s)",
				gradientID,
				gradient.Status,
			)
		}
		gradient.Status = "aggregated"
		if err := k.StoredGradients.Set(ctx, gradientID, gradient); err != nil {
			return nil, errorsmod.Wrapf(err, "failed to update gradient %d status", gradientID)
		}
	}

	// 21. Distribute proposer reward
	participantCount := uint64(len(msg.ParticipantGradientIds))
	if err := k.DistributeProposerReward(ctx, msg.Proposer, participantCount); err != nil {
		return nil, errorsmod.Wrap(err, "failed to distribute proposer reward")
	}

	// 22. Update global model state if needed
	if err := k.UpdateGlobalModelStateIfNeeded(sdkCtx, aggregationRecord); err != nil {
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to update global model state: %v", err))
	}

	// 23. Emit event for reveal
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeRevealAggregation,
			sdk.NewAttribute(types.AttributeKeyProposer, msg.Proposer),
			sdk.NewAttribute(types.AttributeKeyCommitmentID, fmt.Sprintf("%d", msg.CommitmentId)),
			sdk.NewAttribute(types.AttributeKeyAggregationID, fmt.Sprintf("%d", aggregationID)),
			sdk.NewAttribute(types.AttributeKeyTrainingRoundID, fmt.Sprintf("%d", msg.TrainingRoundId)),
		),
	)

	return &types.MsgRevealAggregationResponse{
		AggregationId:      aggregationID,
		CommitmentVerified: true,
	}, nil
}

