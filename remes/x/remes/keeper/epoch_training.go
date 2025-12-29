package keeper

import (
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// EpochInterval is the number of blocks between epoch updates (default: 100 blocks)
const EpochInterval = int64(100)

// TrainingEpoch represents an epoch in the training system
type TrainingEpoch struct {
	EpochID          uint64
	TrainingRoundID uint64
	StartHeight      int64
	EndHeight        int64
	Status           string // "collecting", "aggregating", "finalized"
	ShardAssignments map[string]uint64 // miner_address -> shard_id
}

// ProcessEpochInterval processes epoch interval updates (called in EndBlocker)
// Every 100 blocks, triggers model update if epoch is complete
func (k Keeper) ProcessEpochInterval(ctx sdk.Context) error {
	currentHeight := ctx.BlockHeight()

	// Check if we're at an epoch boundary (every 100 blocks)
	if currentHeight%EpochInterval != 0 {
		return nil // Not an epoch boundary
	}

	// Get current training round (simplified - in production, track active rounds)
	// For now, we'll process all pending aggregations that should trigger model updates
	aggregations, err := k.getAllAggregations(ctx)
	if err != nil {
		return fmt.Errorf("failed to get aggregations: %w", err)
	}

	// Find the best aggregation for this epoch (highest training round, most participants)
	bestAggregation := k.selectBestAggregationForEpoch(aggregations, currentHeight)
	if bestAggregation == nil {
		return nil // No aggregation ready for epoch update
	}

	// Update global model state with epoch aggregation
	if err := k.UpdateGlobalModelStateIfNeeded(ctx, *bestAggregation); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to update global model state for epoch: %v", err))
		return nil // Don't fail epoch processing
	}

	// Emit epoch update event
	ctx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeEpochUpdate,
			sdk.NewAttribute("epoch_height", fmt.Sprintf("%d", currentHeight)),
			sdk.NewAttribute(types.AttributeKeyAggregationID, fmt.Sprintf("%d", bestAggregation.AggregationId)),
			sdk.NewAttribute(types.AttributeKeyTrainingRoundID, fmt.Sprintf("%d", bestAggregation.TrainingRoundId)),
		),
	)

	return nil
}

// selectBestAggregationForEpoch selects the best aggregation for epoch update
func (k Keeper) selectBestAggregationForEpoch(
	aggregations []types.AggregationRecord,
	currentHeight int64,
) *types.AggregationRecord {
	if len(aggregations) == 0 {
		return nil
	}

	var bestAggregation *types.AggregationRecord
	maxParticipants := uint64(0)
	highestRound := uint64(0)

	for i := range aggregations {
		agg := &aggregations[i]

		// Only consider finalized aggregations
		if agg.Status != "finalized" {
			continue
		}

		// Prefer aggregations from recent blocks (within last epoch interval)
		if currentHeight-agg.SubmittedAtHeight > EpochInterval {
			continue // Too old
		}

		participantCount := uint64(len(agg.ParticipantGradientIds))

		// Select aggregation with highest training round, then most participants
		if agg.TrainingRoundId > highestRound ||
			(agg.TrainingRoundId == highestRound && participantCount > maxParticipants) {
			highestRound = agg.TrainingRoundId
			maxParticipants = participantCount
			bestAggregation = agg
		}
	}

	return bestAggregation
}

// GetEpochInfo returns information about the current epoch
func (k Keeper) GetEpochInfo(ctx sdk.Context) (uint64, int64, int64) {
	currentHeight := ctx.BlockHeight()
	epochNumber := uint64(currentHeight / EpochInterval)
	epochStartHeight := (currentHeight / EpochInterval) * EpochInterval
	epochEndHeight := epochStartHeight + EpochInterval

	return epochNumber, epochStartHeight, epochEndHeight
}

// IsEpochBoundary checks if current block is an epoch boundary
func (k Keeper) IsEpochBoundary(ctx sdk.Context) bool {
	return ctx.BlockHeight()%EpochInterval == 0
}

// GetStableShardAssignment returns stable shard assignment for a miner in a training round
// Shard assignment is deterministic: (wallet_address + block_hash + round_id) % total_shards
func (k Keeper) GetStableShardAssignment(
	ctx sdk.Context,
	minerAddress string,
	trainingRoundID uint64,
	totalShards uint64,
) (uint64, error) {
	// Use existing CalculateShardID function
	return k.CalculateShardID(ctx, minerAddress, trainingRoundID, totalShards)
}

