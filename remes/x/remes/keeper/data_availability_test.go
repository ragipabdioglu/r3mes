package keeper_test

import (
	"testing"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"
)

func TestCalculateDATimeout(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	avgBlockTime := 5 * time.Second

	// Test: Low network load (<= 50%)
	timeout := f.keeper.CalculateDATimeout(ctx, 0.3, avgBlockTime)
	require.Equal(t, int64(3), timeout, "low load should use base timeout (3 blocks)")

	timeout = f.keeper.CalculateDATimeout(ctx, 0.5, avgBlockTime)
	require.Equal(t, int64(3), timeout, "50% load should use base timeout (3 blocks)")

	// Test: Medium network load (> 50%, <= 80%)
	timeout = f.keeper.CalculateDATimeout(ctx, 0.6, avgBlockTime)
	require.Equal(t, int64(4), timeout, "medium load should use 4 blocks")

	timeout = f.keeper.CalculateDATimeout(ctx, 0.8, avgBlockTime)
	require.Equal(t, int64(4), timeout, "80% load should use 4 blocks")

	// Test: High network load (> 80%)
	timeout = f.keeper.CalculateDATimeout(ctx, 0.9, avgBlockTime)
	require.Equal(t, int64(5), timeout, "high load should use 5 blocks")

	timeout = f.keeper.CalculateDATimeout(ctx, 1.0, avgBlockTime)
	require.Equal(t, int64(5), timeout, "maximum load should use 5 blocks")
}

func TestGetNetworkLoad(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	// Test: Empty network (no challenges, no gradients)
	load, err := f.keeper.GetNetworkLoad(ctx)
	require.NoError(t, err)
	require.GreaterOrEqual(t, load, 0.0)
	require.LessOrEqual(t, load, 1.0)
	require.Equal(t, 0.0, load, "empty network should have 0% load")
}

func TestCalculateChallengeDeadline(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	deadline, err := f.keeper.CalculateChallengeDeadline(ctx)
	require.NoError(t, err)
	require.Greater(t, deadline, ctx.BlockHeight(), "deadline should be in the future")

	// Deadline should be at least 3 blocks ahead (base timeout)
	expectedMinDeadline := ctx.BlockHeight() + 3
	require.GreaterOrEqual(t, deadline, expectedMinDeadline, "deadline should be at least 3 blocks ahead")
}

func TestIsChallengeExpired(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	currentHeight := ctx.BlockHeight()

	// Test: Future deadline (not expired)
	futureDeadline := currentHeight + 10
	require.False(t, f.keeper.IsChallengeExpired(ctx, futureDeadline), "future deadline should not be expired")

	// Test: Past deadline (expired)
	pastDeadline := currentHeight - 5
	require.True(t, f.keeper.IsChallengeExpired(ctx, pastDeadline), "past deadline should be expired")

	// Test: Current block (not expired, deadline is exclusive)
	currentDeadline := currentHeight
	require.False(t, f.keeper.IsChallengeExpired(ctx, currentDeadline), "current block deadline should not be expired")
}

