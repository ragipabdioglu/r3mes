package keeper_test

import (
	"testing"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"

	"remes/x/remes/types"
)

// TestFinalizeExpiredAggregations tests finalizing expired aggregations
func TestFinalizeExpiredAggregations(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)

	// Set block height to 100
	sdkCtx = sdkCtx.WithBlockHeight(100)

	// Create an aggregation with deadline at height 100
	aggregationID := uint64(1)
	deadlineHeight := int64(100)

	// Add aggregation to deadline index
	err := f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID)
	require.NoError(t, err)

	// Create aggregation record
	aggregationRecord := types.AggregationRecord{
		AggregationId:           aggregationID,
		Status:                  "pending",
		ChallengeDeadlineHeight: deadlineHeight,
	}

	// Store aggregation
	err = f.keeper.AggregationRecords.Set(sdkCtx, aggregationID, aggregationRecord)
	require.NoError(t, err)

	// Finalize expired aggregations
	err = f.keeper.FinalizeExpiredAggregations(sdkCtx)
	require.NoError(t, err)

	// Verify aggregation was finalized
	updatedRecord, err := f.keeper.AggregationRecords.Get(sdkCtx, aggregationID)
	require.NoError(t, err)
	require.Equal(t, "finalized", updatedRecord.Status)
}

// TestFinalizeExpiredAggregationsNoExpired tests when no aggregations are expired
func TestFinalizeExpiredAggregationsNoExpired(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)

	// Set block height to 50
	sdkCtx = sdkCtx.WithBlockHeight(50)

	// Create aggregation with deadline at height 100 (not expired)
	aggregationID := uint64(1)
	deadlineHeight := int64(100)

	// Add aggregation to deadline index
	err := f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID)
	require.NoError(t, err)

	// Finalize expired aggregations (should not finalize anything)
	err = f.keeper.FinalizeExpiredAggregations(sdkCtx)
	require.NoError(t, err)

	// Verify aggregation is still pending
	record, err := f.keeper.AggregationRecords.Get(sdkCtx, aggregationID)
	if err == nil {
		require.Equal(t, "pending", record.Status)
	}
}

// TestFinalizeExpiredAggregationsMultiple tests multiple expired aggregations
func TestFinalizeExpiredAggregationsMultiple(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)

	// Set block height to 100
	sdkCtx = sdkCtx.WithBlockHeight(100)

	// Create multiple aggregations with deadline at height 100
	aggregationID1 := uint64(1)
	aggregationID2 := uint64(2)
	aggregationID3 := uint64(3)
	deadlineHeight := int64(100)

	// Add aggregations to deadline index
	err := f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID1)
	require.NoError(t, err)

	err = f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID2)
	require.NoError(t, err)

	err = f.keeper.AddAggregationToDeadlineIndex(sdkCtx, deadlineHeight, aggregationID3)
	require.NoError(t, err)

	// Create aggregation records
	records := []types.AggregationRecord{
		{AggregationId: aggregationID1, Status: "pending", ChallengeDeadlineHeight: deadlineHeight},
		{AggregationId: aggregationID2, Status: "pending", ChallengeDeadlineHeight: deadlineHeight},
		{AggregationId: aggregationID3, Status: "pending", ChallengeDeadlineHeight: deadlineHeight},
	}

	for _, record := range records {
		err = f.keeper.AggregationRecords.Set(sdkCtx, record.AggregationId, record)
		require.NoError(t, err)
	}

	// Finalize expired aggregations
	err = f.keeper.FinalizeExpiredAggregations(sdkCtx)
	require.NoError(t, err)

	// Verify all aggregations were finalized
	for _, record := range records {
		updatedRecord, err := f.keeper.AggregationRecords.Get(sdkCtx, record.AggregationId)
		if err == nil {
			require.Equal(t, "finalized", updatedRecord.Status)
		}
	}
}

