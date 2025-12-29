package keeper

import (
	"context"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// AddAggregationToDeadlineIndex adds an aggregation ID to the deadline index
// This allows O(1) lookup of aggregations that need to be finalized at a specific block height
func (k Keeper) AddAggregationToDeadlineIndex(ctx context.Context, deadlineHeight int64, aggregationID uint64) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get existing aggregations for this deadline height
	existingList, err := k.PendingAggregationsByDeadline.Get(ctx, deadlineHeight)
	if err != nil {
		// Key doesn't exist yet, create new list
		existingList = types.AggregationIDList{
			AggregationIds: []uint64{},
		}
	}

	// Check if aggregation ID already exists (shouldn't happen, but be safe)
	for _, id := range existingList.AggregationIds {
		if id == aggregationID {
			// Already in index, skip
			return nil
		}
	}

	// Add aggregation ID to list
	existingList.AggregationIds = append(existingList.AggregationIds, aggregationID)

	// Update index
	if err := k.PendingAggregationsByDeadline.Set(ctx, deadlineHeight, existingList); err != nil {
		return errorsmod.Wrapf(err, "failed to add aggregation %d to deadline index at height %d", aggregationID, deadlineHeight)
	}

	sdkCtx.Logger().Debug("Added aggregation to deadline index", "aggregation_id", aggregationID, "deadline_height", deadlineHeight)
	return nil
}

// RemoveAggregationFromDeadlineIndex removes an aggregation ID from the deadline index
func (k Keeper) RemoveAggregationFromDeadlineIndex(ctx context.Context, deadlineHeight int64, aggregationID uint64) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get existing aggregations for this deadline height
	existingList, err := k.PendingAggregationsByDeadline.Get(ctx, deadlineHeight)
	if err != nil {
		// Key doesn't exist, nothing to remove
		return nil
	}

	// Remove aggregation ID from list
	newAggregations := make([]uint64, 0, len(existingList.AggregationIds))
	for _, id := range existingList.AggregationIds {
		if id != aggregationID {
			newAggregations = append(newAggregations, id)
		}
	}

	// Update or remove index entry
	if len(newAggregations) == 0 {
		// No more aggregations for this deadline, remove the index entry
		if err := k.PendingAggregationsByDeadline.Remove(ctx, deadlineHeight); err != nil {
			return errorsmod.Wrapf(err, "failed to remove deadline index entry at height %d", deadlineHeight)
		}
	} else {
		// Update with remaining aggregations
		updatedList := types.AggregationIDList{
			AggregationIds: newAggregations,
		}
		if err := k.PendingAggregationsByDeadline.Set(ctx, deadlineHeight, updatedList); err != nil {
			return errorsmod.Wrapf(err, "failed to update deadline index at height %d", deadlineHeight)
		}
	}

	sdkCtx.Logger().Debug("Removed aggregation from deadline index", "aggregation_id", aggregationID, "deadline_height", deadlineHeight)
	return nil
}

// getAggregationsByDeadline gets all aggregation IDs that need to be finalized at a specific block height
func (k Keeper) getAggregationsByDeadline(ctx context.Context, deadlineHeight int64) ([]uint64, error) {
	list, err := k.PendingAggregationsByDeadline.Get(ctx, deadlineHeight)
	if err != nil {
		// Key doesn't exist, return empty list
		return []uint64{}, nil
	}
	return list.AggregationIds, nil
}

