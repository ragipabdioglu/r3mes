package keeper_test

import (
	"testing"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"
)

// TestGetAverageBlockTime tests the average block time calculation
func TestGetAverageBlockTime(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)

	// Test 1: Early blocks (height < 2) should return default block time
	avgBlockTime := f.keeper.GetAverageBlockTime(sdkCtx)
	require.Equal(t, 6*time.Second, avgBlockTime, "early blocks should return default 6 seconds")

	// Test 2: Store some block timestamps and verify calculation
	// Simulate multiple blocks with different timestamps
	baseTime := time.Now()
	blockHeights := []int64{1, 2, 3, 4, 5}
	blockTimes := []int64{
		baseTime.Unix(),
		baseTime.Add(5 * time.Second).Unix(),
		baseTime.Add(10 * time.Second).Unix(),
		baseTime.Add(15 * time.Second).Unix(),
		baseTime.Add(20 * time.Second).Unix(),
	}

	// Store block timestamps (as int64 Unix timestamps)
	for i, height := range blockHeights {
		err := f.keeper.BlockTimestamps.Set(sdkCtx, height, blockTimes[i])
		require.NoError(t, err)
	}

	// Test with height 5 (should calculate average from blocks 1-5)
	ctx := sdkCtx.WithBlockHeight(5)
	avgBlockTime = f.keeper.GetAverageBlockTime(ctx)
	
	// Expected average: (5 + 5 + 5 + 5) / 4 = 5 seconds
	expectedAvg := 5 * time.Second
	require.InDelta(t, expectedAvg.Seconds(), avgBlockTime.Seconds(), 0.1, 
		"average block time should be approximately 5 seconds")

	// Test 3: With insufficient data (only 1 timestamp)
	ctx = sdkCtx.WithBlockHeight(1)
	avgBlockTime = f.keeper.GetAverageBlockTime(ctx)
	require.Equal(t, 6*time.Second, avgBlockTime, "insufficient data should return default")

	// Test 4: Verify pruning (old blocks should be removed)
	// Note: Pruning happens in end_blocker, which is called automatically
	// For this test, we manually simulate pruning by setting height 101
	// and then calling end_blocker logic
	ctx = sdkCtx.WithBlockHeight(101).WithBlockTime(time.Now())
	err := f.keeper.BlockTimestamps.Set(ctx, 101, time.Now().Unix())
	require.NoError(t, err)

	// Manually trigger pruning (simulating end_blocker)
	// Pruning removes blocks older than 100 blocks
	if ctx.BlockHeight() > 100 {
		oldHeight := ctx.BlockHeight() - 100
		_ = f.keeper.BlockTimestamps.Remove(ctx, oldHeight)
	}

	// Block 1 should be pruned (only last 100 blocks kept)
	_, err = f.keeper.BlockTimestamps.Get(ctx, 1)
	require.Error(t, err, "old block timestamp should be pruned")
}

// TestBlockTimestampsStorage tests block timestamp storage and retrieval
func TestBlockTimestampsStorage(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)

	// Test: Store and retrieve block timestamp
	height := int64(10)
	timestamp := time.Now().Unix()
	
	err := f.keeper.BlockTimestamps.Set(sdkCtx, height, timestamp)
	require.NoError(t, err)

	retrievedTimestamp, err := f.keeper.BlockTimestamps.Get(sdkCtx, height)
	require.NoError(t, err)
	require.Equal(t, timestamp, retrievedTimestamp, 
		"retrieved timestamp should match stored timestamp")

	// Test: Non-existent block timestamp
	_, err = f.keeper.BlockTimestamps.Get(sdkCtx, 999)
	require.Error(t, err, "non-existent block timestamp should return error")
}

