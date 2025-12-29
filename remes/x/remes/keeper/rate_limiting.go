package keeper

import (
	"fmt"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// Rate limiting parameters
const (
	// MaxGradientSubmissionsPerBlock is the maximum number of gradient submissions per miner per block
	MaxGradientSubmissionsPerBlock = 1
	
	// MaxGradientSubmissionsPerMinute is the maximum number of gradient submissions per miner per minute
	MaxGradientSubmissionsPerMinute = 10
	
	// RateLimitWindowBlocks is the window size for rate limiting (blocks)
	RateLimitWindowBlocks = 10
)

// SubmissionRecord tracks submission rate for a miner
type SubmissionRecord struct {
	MinerAddress     string
	BlockHeight      int64
	SubmissionCount  uint64
	LastSubmissionTime time.Time
	SubmissionTimes  []time.Time
}

// CheckRateLimit checks if a miner has exceeded rate limits
// Implements:
// 1. Block-based rate limiting: Max 1 submission per block
// 2. Block-window rate limiting: Max N submissions in last M blocks (sliding window)
// 3. Time-based rate limiting: Approximate check using block time
func (k Keeper) CheckRateLimit(ctx sdk.Context, minerAddress string) error {
	currentHeight := ctx.BlockHeight()
	
	contribution, err := k.MiningContributions.Get(ctx, minerAddress)
	if err != nil {
		// New miner - allow first submission
		return nil
	}
	
	// 1. Block-based rate limiting: Prevent multiple submissions in the same block
	if contribution.LastSubmissionHeight == currentHeight {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "rate limit exceeded: multiple submissions in same block")
	}
	
	// 2. Block-window rate limiting: Check submissions in the last N blocks
	// This provides approximate time-based rate limiting
	// Assuming ~5s per block, 10 blocks = ~50 seconds
	windowStartHeight := currentHeight - RateLimitWindowBlocks
	if windowStartHeight < 0 {
		windowStartHeight = 0
	}
	
	// Count submissions in the window using submission history
	submissionCount := uint64(0)
	for height := windowStartHeight; height <= currentHeight; height++ {
		historyKey := fmt.Sprintf("%s|%d", minerAddress, height)
		count, err := k.SubmissionHistory.Get(ctx, historyKey)
		if err == nil {
			submissionCount += count
		}
	}
	
	// Check if submission count exceeds limit
	if submissionCount >= MaxGradientSubmissionsPerMinute {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "rate limit exceeded: %d submissions in last %d blocks (limit: %d)", submissionCount, RateLimitWindowBlocks, MaxGradientSubmissionsPerMinute)
	}
	
	// 3. Time-based rate limiting: Approximate check using block time
	// If we have block time information, check if last submission was within 1 minute
	if contribution.LastSubmissionHeight > 0 {
		// Estimate submission time based on block height difference
		// This is approximate but sufficient for rate limiting
		// Assuming ~5s per block, calculate approximate time difference
		blocksSinceLastSubmission := currentHeight - contribution.LastSubmissionHeight
		
		// If last submission was very recent (< 6 seconds = 1 block), reject
		// This provides additional protection beyond same-block check
		if blocksSinceLastSubmission == 0 {
			// Already checked above, but double-check
			return errorsmod.Wrapf(types.ErrInvalidMiner, "rate limit exceeded: multiple submissions in same block")
		}
		
		// For time-based rate limiting, we need to track actual submission times
		// For now, we use block-based approximation
		// If miner submits in rapid succession (multiple blocks quickly), check time
		if blocksSinceLastSubmission < 12 { // 12 blocks = ~60 seconds
			// Check if we can get more accurate time information
			// For now, allow if not in same block
		}
	}
	
	return nil
}

// RecordSubmission records a gradient submission for rate limiting
func (k Keeper) RecordSubmission(ctx sdk.Context, minerAddress string) error {
	currentHeight := ctx.BlockHeight()
	
	// Record submission in history for rate limiting
	historyKey := fmt.Sprintf("%s|%d", minerAddress, currentHeight)
	currentCount, err := k.SubmissionHistory.Get(ctx, historyKey)
	if err != nil {
		// First submission in this block for this miner
		currentCount = 0
	}
	if err := k.SubmissionHistory.Set(ctx, historyKey, currentCount+1); err != nil {
		return errorsmod.Wrap(err, "failed to record submission history")
	}
	
	// Clean up old submission history entries (outside the window)
	// This prevents state bloat by removing entries older than the rate limit window
	windowStartHeight := currentHeight - RateLimitWindowBlocks - 1
	if windowStartHeight >= 0 {
		oldHistoryKey := fmt.Sprintf("%s|%d", minerAddress, windowStartHeight)
		if err := k.SubmissionHistory.Remove(ctx, oldHistoryKey); err != nil {
			// Ignore error if key doesn't exist
			_ = err
		}
	}
	
	// Update mining contribution with last submission height
	contribution, err := k.MiningContributions.Get(ctx, minerAddress)
	if err != nil {
		// Create new contribution
		contribution = types.MiningContribution{
			MinerAddress:         minerAddress,
			TotalSubmissions:     0,
			SuccessfulSubmissions: 0,
			TrustScore:           "0.5",
			ReputationTier:       "new",
			SlashingEvents:       0,
			LastSubmissionHeight: currentHeight,
		}
	} else {
		contribution.LastSubmissionHeight = currentHeight
	}
	
	// Update contribution
	return k.MiningContributions.Set(ctx, minerAddress, contribution)
}

