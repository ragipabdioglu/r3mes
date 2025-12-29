package keeper

import (
	"context"
	"strings"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// ChallengeAggregation handles MsgChallengeAggregation
// Initiates a challenge to an aggregation for dispute resolution
func (k msgServer) ChallengeAggregation(ctx context.Context, msg *types.MsgChallengeAggregation) (*types.MsgChallengeAggregationResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate challenger address
	challengerAddr, err := k.addressCodec.StringToBytes(msg.Challenger)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid challenger address: %s", msg.Challenger)
	}
	_ = challengerAddr // Address validated

	// 2. Verify aggregation exists
	aggregation, err := k.AggregationRecords.Get(ctx, msg.AggregationId)
	if err != nil {
		return nil, errorsmod.Wrapf(err, "aggregation %d not found", msg.AggregationId)
	}

	// 3. Verify aggregation is in challengeable status
	if aggregation.Status != "pending" && aggregation.Status != "finalized" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "aggregation %d cannot be challenged (status: %s)", msg.AggregationId, aggregation.Status)
	}

	// 4. Validate challenge reason
	reason := strings.TrimSpace(msg.Reason)
	if reason == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "challenge reason cannot be empty")
	}

	// 5. Validate evidence IPFS hash
	evidenceHash := strings.TrimSpace(msg.EvidenceIpfsHash)
	if evidenceHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "evidence IPFS hash cannot be empty")
	}
	if !strings.HasPrefix(evidenceHash, "Qm") && !strings.HasPrefix(evidenceHash, "bafy") && !strings.HasPrefix(evidenceHash, "bafk") {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", evidenceHash)
	}

	// 6. OPTIMISTIC VERIFICATION: Layer 2 - High-Stakes Challenge
	// Require bond from challenger (10x base reward)
	bondAmount := sdk.NewCoins()
	for _, coin := range msg.BondAmount {
		bondAmount = bondAmount.Add(coin)
	}

	// Get actual gradient hash from stored gradients or aggregation metadata
	// Try to get gradient hash from aggregation metadata if available
	expectedHash := aggregation.AggregatedGradientIpfsHash // Default: use IPFS hash as fallback
	
	// If aggregation has gradient IDs, retrieve the stored gradients and use their gradient hash
	// For now, aggregation stores IPFS hash, but we can compute expected gradient hash from stored gradients
	// In a production system, aggregation would store the computed gradient hash
	// For now, we use the IPFS hash as the identifier (the verification layer will download and verify)
	// This is acceptable because Layer 2 verification downloads and verifies the actual gradient data

	// Use Layer 2 verification (high-stakes challenge with bond)
	challengeID, err := k.VerifyGradientLayer2(
		sdkCtx,
		msg.Challenger,
		evidenceHash, // This should be the disputed gradient hash
		expectedHash,
		msg.AggregationId,
		bondAmount,
	)
	if err != nil {
		return nil, errorsmod.Wrap(err, "Layer 2 verification failed")
	}

	// 7. Update aggregation status to "challenged"
	aggregation.Status = "challenged"
	if err := k.AggregationRecords.Set(ctx, msg.AggregationId, aggregation); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update aggregation status")
	}

	// 8. Emit event for aggregation challenge
	sdkCtx.EventManager().EmitEvent(
		types.NewEventAggregationChallenged(
			msg.Challenger,
			challengeID,
			msg.AggregationId,
		),
	)

	return &types.MsgChallengeAggregationResponse{
		ChallengeId: challengeID,
	}, nil
}

