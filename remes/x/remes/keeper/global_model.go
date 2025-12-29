package keeper

import (
	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// UpdateGlobalModelStateIfNeeded updates the global model state if the given aggregation
// should become the new global model.
// Strategy:
// 1. If no global model exists, use this aggregation
// 2. If this aggregation has a higher training round ID, use it
// 3. If same training round ID but this aggregation is newer (higher aggregation ID), use it
// 4. Otherwise, keep existing global model
func (k Keeper) UpdateGlobalModelStateIfNeeded(ctx sdk.Context, aggregation types.AggregationRecord) error {
	currentState, err := k.GlobalModelState.Get(ctx)
	if err != nil {
		// No global model exists yet, create initial state
		newState := types.GlobalModelState{
			ModelIpfsHash:      aggregation.AggregatedGradientIpfsHash,
			ModelVersion:       aggregation.ModelVersion,
			LastUpdatedHeight:  ctx.BlockHeight(),
			LastUpdatedTime:    ctx.BlockTime(),
			TrainingRoundId:    aggregation.TrainingRoundId,
			LastAggregationId:  aggregation.AggregationId,
		}
		return k.GlobalModelState.Set(ctx, newState)
	}

	// Check if this aggregation should update the global model
	shouldUpdate := false

	// Case 1: Higher training round ID
	if aggregation.TrainingRoundId > k.getTrainingRoundFromState(currentState) {
		shouldUpdate = true
	} else if aggregation.TrainingRoundId == k.getTrainingRoundFromState(currentState) {
		// Case 2: Same training round, but this aggregation is newer (higher aggregation ID)
		// This means it's a more recent aggregation for the same round
		// We'll update if this aggregation ID is higher (assuming sequential IDs)
		// Note: This is a simple heuristic - in production, you might want to track
		// which aggregation ID was used for the current global model
		if aggregation.AggregationId > k.getLastAggregationIDFromState(currentState) {
			shouldUpdate = true
		}
	}

	if shouldUpdate {
		// Update global model state
		newState := types.GlobalModelState{
			ModelIpfsHash:      aggregation.AggregatedGradientIpfsHash,
			ModelVersion:       aggregation.ModelVersion,
			LastUpdatedHeight:  ctx.BlockHeight(),
			LastUpdatedTime:    ctx.BlockTime(),
			TrainingRoundId:    aggregation.TrainingRoundId,
			LastAggregationId:  aggregation.AggregationId,
		}
		return k.GlobalModelState.Set(ctx, newState)
	}

	return nil
}

// getTrainingRoundFromState extracts training round ID from global model state
func (k Keeper) getTrainingRoundFromState(state types.GlobalModelState) uint64 {
	return state.TrainingRoundId
}

// getLastAggregationIDFromState gets the last aggregation ID that updated the global model
func (k Keeper) getLastAggregationIDFromState(state types.GlobalModelState) uint64 {
	return state.LastAggregationId
}

// UpdateGlobalModelStateFromFinalizedAggregation updates global model state when an aggregation is finalized
// This is called from EndBlocker when aggregations are finalized
func (k Keeper) UpdateGlobalModelStateFromFinalizedAggregation(ctx sdk.Context, aggregationID uint64) error {
	aggregation, err := k.AggregationRecords.Get(ctx, aggregationID)
	if err != nil {
		return errorsmod.Wrapf(err, "aggregation %d not found", aggregationID)
	}

	// Only update if aggregation is finalized
	if aggregation.Status != "finalized" {
		return nil
	}

	// Update global model state
	return k.UpdateGlobalModelStateIfNeeded(ctx, aggregation)
}

