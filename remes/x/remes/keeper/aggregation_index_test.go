package keeper_test

import (
	"testing"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"
)

// TestAddAggregationToDeadlineIndex tests adding aggregation to deadline index
func TestAddAggregationToDeadlineIndex(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)

	deadlineHeight := int64(100)
	aggregationID := uint64(1)

	// Add aggregation to index
	err := f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID)
	require.NoError(t, err)

	// Verify aggregation was added
	ids, err := f.keeper.PendingAggregationsByDeadline.Get(sdkCtx, deadlineHeight)
	require.NoError(t, err)
	require.NotNil(t, ids)
	require.Contains(t, ids.AggregationIds, aggregationID)
}

// TestRemoveAggregationFromDeadlineIndex tests removing aggregation from deadline index
func TestRemoveAggregationFromDeadlineIndex(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)

	deadlineHeight := int64(100)
	aggregationID := uint64(1)

	// Add aggregation to index
	err := f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID)
	require.NoError(t, err)

	// Remove aggregation from index
	err = f.keeper.RemoveAggregationFromDeadlineIndex(sdkCtx, deadlineHeight, aggregationID)
	require.NoError(t, err)

	// Verify aggregation was removed
	ids, err := f.keeper.PendingAggregationsByDeadline.Get(sdkCtx, deadlineHeight)
	if err != nil {
		// If error, it means entry was removed (which is OK)
		return
	}
	require.NotContains(t, ids.AggregationIds, aggregationID)
}

// TestMultipleAggregationsSameDeadline tests multiple aggregations with same deadline
func TestMultipleAggregationsSameDeadline(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)

	deadlineHeight := int64(100)
	aggregationID1 := uint64(1)
	aggregationID2 := uint64(2)
	aggregationID3 := uint64(3)

	// Add multiple aggregations
	err := f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID1)
	require.NoError(t, err)

	err = f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID2)
	require.NoError(t, err)

	err = f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID3)
	require.NoError(t, err)

	// Verify all aggregations are in index
	ids, err := f.keeper.PendingAggregationsByDeadline.Get(sdkCtx, deadlineHeight)
	require.NoError(t, err)
	require.Contains(t, ids.GetAggregationIds(), aggregationID1)
	require.Contains(t, ids.GetAggregationIds(), aggregationID2)
	require.Contains(t, ids.GetAggregationIds(), aggregationID3)
	require.Equal(t, 3, len(ids.GetAggregationIds()))
}

// TestRemoveNonExistentAggregation tests removing non-existent aggregation
func TestRemoveNonExistentAggregation(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)

	deadlineHeight := int64(100)
	aggregationID := uint64(999) // Non-existent

	// Try to remove non-existent aggregation (should not error)
	err := f.keeper.RemoveAggregationFromDeadlineIndex(sdkCtx, deadlineHeight, aggregationID)
	// Should not error, just return
	require.NoError(t, err)
}

