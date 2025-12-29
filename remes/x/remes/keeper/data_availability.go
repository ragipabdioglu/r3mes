package keeper

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// Default parameters for automatic Data Availability (DA) sampling.
// These can be made configurable via Params in the future if needed.
const (
	// daSampleRate is the fraction of active pinning commitments that will be
	// challenged on average when sampling runs. For example, 0.1 = ~10%.
	daSampleRate = 0.10

	// daMaxChallengesPerBlock is a hard cap on how many automatic challenges
	// can be issued in a single sampling run to avoid overloading the network.
	daMaxChallengesPerBlock = 20
)

// CalculateDATimeout calculates dynamic timeout for data availability challenges
// based on network load and average block time
// Formula:
//   - Base timeout: 3 blocks (minimum)
//   - Network load > 80%: +2 blocks (5 blocks total)
//   - Network load > 50%: +1 block (4 blocks total)
//   - Network load <= 50%: base timeout (3 blocks)
func (k Keeper) CalculateDATimeout(
	ctx sdk.Context,
	networkLoad float64,
	avgBlockTime time.Duration,
) int64 {
	baseTimeout := int64(3) // 3 blocks minimum

	// Adjust based on network load
	if networkLoad > 0.8 {
		// High network load: increase timeout to 5 blocks
		return baseTimeout + 2
	} else if networkLoad > 0.5 {
		// Medium network load: increase timeout to 4 blocks
		return baseTimeout + 1
	}

	// Low network load: use base timeout (3 blocks)
	return baseTimeout
}

// GetNetworkLoad calculates current network load based on pending challenges and transactions
// Returns a value between 0.0 (no load) and 1.0 (maximum load)
func (k Keeper) GetNetworkLoad(ctx sdk.Context) (float64, error) {
	// Count pending challenges
	pendingChallenges := 0
	challengeIter, err := k.ChallengeRecords.Iterate(ctx, nil)
	if err != nil {
		return 0.0, fmt.Errorf("failed to iterate challenges: %w", err)
	}
	defer challengeIter.Close()

	for ; challengeIter.Valid(); challengeIter.Next() {
		challenge, err := challengeIter.Value()
		if err != nil {
			continue
		}
		// Count challenges that are still pending
		if challenge.Status == "pending" || challenge.Status == "challenged" {
			pendingChallenges++
		}
	}

	// Count pending gradients (recent submissions)
	recentGradients := 0
	gradientIter, err := k.StoredGradients.Iterate(ctx, nil)
	if err != nil {
		return 0.0, fmt.Errorf("failed to iterate gradients: %w", err)
	}
	defer gradientIter.Close()

	currentHeight := ctx.BlockHeight()
	for ; gradientIter.Valid(); gradientIter.Next() {
		gradient, err := gradientIter.Value()
		if err != nil {
			continue
		}
		// Count gradients submitted in last 10 blocks (recent activity)
		if currentHeight-gradient.SubmittedAtHeight < 10 {
			recentGradients++
		}
	}

	// Calculate network load
	// Normalize based on expected capacity
	// Assume:
	// - 10 pending challenges = 50% load
	// - 50 recent gradients = 50% load
	// - Combined max = 100% load
	challengeLoad := float64(pendingChallenges) / 20.0 // 20 challenges = 100% load
	if challengeLoad > 1.0 {
		challengeLoad = 1.0
	}

	gradientLoad := float64(recentGradients) / 100.0 // 100 gradients = 100% load
	if gradientLoad > 1.0 {
		gradientLoad = 1.0
	}

	// Combined load (weighted average)
	networkLoad := (challengeLoad*0.6 + gradientLoad*0.4)
	if networkLoad > 1.0 {
		networkLoad = 1.0
	}

	return networkLoad, nil
}

// GetAverageBlockTime calculates average block time from recent blocks
// Returns average block time in seconds
// Uses stored block timestamps to calculate average over recent blocks
func (k Keeper) GetAverageBlockTime(ctx sdk.Context) time.Duration {
	// Default block time if calculation fails
	defaultBlockTime := 6 * time.Second

	currentHeight := ctx.BlockHeight()
	
	// If we're at genesis or very early blocks, use default
	if currentHeight < 2 {
		return defaultBlockTime
	}
	
	// Calculate average block time from stored timestamps
	// Use last 100 blocks (or fewer if not available)
	var timestamps []int64
	var totalDiff int64
	var count int64
	
	// Collect timestamps from recent blocks (up to 100 blocks)
	maxBlocks := int64(100)
	startHeight := currentHeight - maxBlocks
	if startHeight < 1 {
		startHeight = 1
	}
	
	for height := startHeight; height < currentHeight; height++ {
		timestamp, err := k.BlockTimestamps.Get(ctx, height)
		if err != nil {
			// Skip if timestamp not found (may happen for very old blocks)
			continue
		}
		timestamps = append(timestamps, timestamp)
	}
	
	// Calculate average time difference between consecutive blocks
	if len(timestamps) < 2 {
		// Not enough data, use default
		return defaultBlockTime
	}
	
	for i := 1; i < len(timestamps); i++ {
		diff := timestamps[i] - timestamps[i-1]
		if diff > 0 && diff < 60 { // Sanity check: block time should be between 0 and 60 seconds
			totalDiff += diff
			count++
		}
	}
	
	if count == 0 {
		return defaultBlockTime
	}
	
	avgSeconds := float64(totalDiff) / float64(count)
	return time.Duration(avgSeconds) * time.Second
}

// CalculateChallengeDeadline calculates the deadline block height for a challenge
// based on dynamic timeout calculation
func (k Keeper) CalculateChallengeDeadline(ctx sdk.Context) (int64, error) {
	// Get current network load
	networkLoad, err := k.GetNetworkLoad(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to get network load: %w", err)
	}

	// Get average block time
	avgBlockTime := k.GetAverageBlockTime(ctx)

	// Calculate timeout in blocks
	timeoutBlocks := k.CalculateDATimeout(ctx, networkLoad, avgBlockTime)

	// Calculate deadline block height
	currentHeight := ctx.BlockHeight()
	deadlineHeight := currentHeight + timeoutBlocks

	return deadlineHeight, nil
}

// IsChallengeExpired checks if a challenge has expired based on its deadline
func (k Keeper) IsChallengeExpired(ctx sdk.Context, challengeDeadlineHeight int64) bool {
	currentHeight := ctx.BlockHeight()
	return currentHeight > challengeDeadlineHeight
}

// shouldSamplePinningForDA deterministically decides whether a given
// (node_address, ipfs_hash) pair should be challenged for data availability
// at the current block, based on a fixed sampling rate.
//
// This uses a deterministic hash of (node_address | ipfs_hash | block_height)
// so that all validators reach the same decision without extra coordination.
func shouldSamplePinningForDA(
	ctx sdk.Context,
	nodeAddress string,
	ipfsHash string,
	sampleRate float64,
) bool {
	if sampleRate <= 0 {
		return false
	}
	if sampleRate >= 1 {
		return true
	}

	key := fmt.Sprintf("%s|%s|%d", nodeAddress, ipfsHash, ctx.BlockHeight())
	hash := sha256.Sum256([]byte(key))
	// Use first 8 bytes as a uint64 and normalize to [0,1).
	randUint := binary.BigEndian.Uint64(hash[:8])
	const maxUint64 = ^uint64(0)
	r := float64(randUint) / float64(maxUint64)
	return r < sampleRate
}

// SampleDataAvailabilityChallenges automatically issues data availability
// challenges for a random subset of active pinning commitments.
//
// This provides probabilistic guarantees that pinned content remains available,
// without requiring explicit MsgChallengePinning transactions from users.
//
// High-level algorithm:
//   1. Compute current network load and dynamic timeout parameters.
//   2. Iterate over active PinningIncentive records.
//   3. For each active record that has not been challenged yet,
//      use a deterministic hash to decide sampling with probability daSampleRate.
//   4. For sampled records, create DataAvailabilityChallenge entries with
//      dynamic response deadlines.
//   5. Cap the total number of new challenges per block to daMaxChallengesPerBlock.
func (k Keeper) SampleDataAvailabilityChallenges(ctx sdk.Context) error {
	sdkCtx := ctx

	// Compute network load and average block time once.
	networkLoad, err := k.GetNetworkLoad(sdkCtx)
	if err != nil {
		// Fallback to default if network load calculation fails.
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to get network load for DA sampling: %v", err))
		networkLoad = 0.0
	}
	avgBlockTime := k.GetAverageBlockTime(sdkCtx)

	// Calculate timeout in blocks using the same logic as manual challenges.
	timeoutBlocks := k.CalculateDATimeout(sdkCtx, networkLoad, avgBlockTime)

	created := 0

	err = k.PinningIncentives.Walk(ctx, nil, func(key string, pinningIncentive types.PinningIncentive) (stop bool, err error) {
		// Respect per-block cap.
		if created >= daMaxChallengesPerBlock {
			return true, nil
		}

		// Only process active pinning commitments.
		if pinningIncentive.Status != "active" {
			return false, nil
		}

		// Avoid repeatedly auto-challenging the same commitment:
		// if it has already been challenged at least once, skip.
		if pinningIncentive.ChallengeCount > 0 {
			return false, nil
		}

		// Deterministic sampling decision.
		if !shouldSamplePinningForDA(sdkCtx, pinningIncentive.NodeAddress, pinningIncentive.IpfsHash, daSampleRate) {
			return false, nil
		}

		// Generate challenge ID.
		challengeID, err := k.DataAvailabilityChallengeID.Next(ctx)
		if err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to generate DA challenge ID: %v", err))
			return false, nil
		}

		// Calculate response deadline based on dynamic timeout.
		responseDeadline := sdkCtx.BlockTime().Add(time.Duration(timeoutBlocks) * avgBlockTime)

		challenge := types.DataAvailabilityChallenge{
			ChallengeId:      challengeID,
			Challenger:       "auto-da-sampler", // system-generated challenge
			NodeAddress:      pinningIncentive.NodeAddress,
			IpfsHash:         pinningIncentive.IpfsHash,
			ChallengeTime:    sdkCtx.BlockTime(),
			ResponseDeadline: responseDeadline,
			Status:           "pending",
		}

		// Store challenge.
		if err := k.DataAvailabilityChallenges.Set(ctx, challengeID, challenge); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to store DA challenge: %v", err))
			return false, nil
		}

		// Update pinning incentive challenge count.
		pinningIncentive.ChallengeCount++
		if err := k.PinningIncentives.Set(ctx, key, pinningIncentive); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to update pinning incentive for DA challenge: %v", err))
			return false, nil
		}

		// Emit event for observability.
		sdkCtx.EventManager().EmitEvent(
			sdk.NewEvent(
				types.EventTypeChallengePinning,
				sdk.NewAttribute(types.AttributeKeyChallengeID, fmt.Sprintf("%d", challengeID)),
				sdk.NewAttribute(types.AttributeKeyChallenger, "auto-da-sampler"),
				sdk.NewAttribute(types.AttributeKeyNodeAddress, pinningIncentive.NodeAddress),
				sdk.NewAttribute(types.AttributeKeyIPFSHash, pinningIncentive.IpfsHash),
			),
		)

		created++
		return false, nil
	})

	if err != nil {
		return fmt.Errorf("failed to walk pinning incentives for DA sampling: %w", err)
	}

	if created > 0 {
		sdkCtx.Logger().Info("Automatic DA sampling issued challenges",
			"count", created,
			"block_height", sdkCtx.BlockHeight(),
			"sample_rate", daSampleRate,
		)
	}

	return nil
}


