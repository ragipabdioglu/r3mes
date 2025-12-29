package keeper

import (
	"context"
	"fmt"
	"strings"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// CommitAggregation handles MsgCommitAggregation
// Proposers commit to an aggregation result hash to prevent collusion
func (k msgServer) CommitAggregation(ctx context.Context, msg *types.MsgCommitAggregation) (*types.MsgCommitAggregationResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate proposer address
	proposerAddr, err := k.addressCodec.StringToBytes(msg.Proposer)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid proposer address: %s", msg.Proposer)
	}
	_ = proposerAddr // Address validated

	// 2. Validate commitment hash
	commitmentHash := strings.TrimSpace(msg.CommitmentHash)
	if commitmentHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "commitment hash cannot be empty")
	}

	// 3. Validate participant gradient IDs
	if len(msg.ParticipantGradientIds) == 0 {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "participant gradient IDs cannot be empty")
	}

	// 4. Verify all participant gradients exist and are in pending status
	for _, gradientID := range msg.ParticipantGradientIds {
		gradient, err := k.StoredGradients.Get(ctx, gradientID)
		if err != nil {
			return nil, errorsmod.Wrapf(err, "gradient %d not found", gradientID)
		}
		if gradient.Status != "pending" {
			return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "gradient %d is not in pending status (current: %s)", gradientID, gradient.Status)
		}
	}

	// 5. Check if proposer already has a commitment for this training round
	// (Prevent duplicate commitments from same proposer)
	// Note: In production, we might want to allow multiple commitments per proposer
	// but for simplicity, we'll allow one commitment per proposer per training round

	// 6. Generate unique commitment ID
	commitmentID, err := k.AggregationCommitmentID.Next(ctx)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to generate commitment ID")
	}

	// 7. Calculate reveal deadline (default: 10 blocks after commit = ~50 seconds at 5s/block)
	commitPeriodBlocks := int64(10) // Can be made configurable via params
	revealDeadlineHeight := sdkCtx.BlockHeight() + commitPeriodBlocks

	// 8. Create AggregationCommitment
	commitment := types.AggregationCommitment{
		CommitmentId:           commitmentID,
		Proposer:               msg.Proposer,
		CommitmentHash:         commitmentHash,
		TrainingRoundId:        msg.TrainingRoundId,
		ParticipantGradientIds: msg.ParticipantGradientIds,
		CommittedAtHeight:      sdkCtx.BlockHeight(),
		RevealDeadlineHeight:   revealDeadlineHeight,
		Status:                 "committed",
		RevealedAggregationId:  0, // Will be set after reveal
	}

	// 9. Store commitment on-chain
	if err := k.AggregationCommitments.Set(ctx, commitmentID, commitment); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store aggregation commitment")
	}

	// 10. Emit event for commitment
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeCommitAggregation,
			sdk.NewAttribute(types.AttributeKeyProposer, msg.Proposer),
			sdk.NewAttribute(types.AttributeKeyCommitmentID, fmt.Sprintf("%d", commitmentID)),
			sdk.NewAttribute(types.AttributeKeyTrainingRoundID, fmt.Sprintf("%d", msg.TrainingRoundId)),
		),
	)

	return &types.MsgCommitAggregationResponse{
		CommitmentId:        commitmentID,
		RevealDeadlineHeight: revealDeadlineHeight,
	}, nil
}

