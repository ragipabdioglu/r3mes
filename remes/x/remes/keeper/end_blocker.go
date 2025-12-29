package keeper

import (
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// FinalizeExpiredAggregations finalizes aggregations whose challenge period has expired.
// This is called at the end of each block.
func (k Keeper) FinalizeExpiredAggregations(ctx sdk.Context) error {
	// Finalize dataset proposals that have passed voting period
	if err := k.FinalizeDatasetProposals(ctx); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to finalize dataset proposals: %v", err))
	}

	// Process migration window completion
	if err := k.ProcessMigrationWindow(ctx); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to process migration window: %v", err))
	}

	// Check for window boundaries and trigger aggregation if needed
	if err := k.ProcessWindowBoundaries(ctx); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to process window boundaries: %v", err))
	}

	// Sync serving node model versions periodically (every 10 blocks)
	if ctx.BlockHeight()%10 == 0 {
		if err := k.SyncServingNodeModelVersions(ctx); err != nil {
			ctx.Logger().Error(fmt.Sprintf("Failed to sync serving node model versions: %v", err))
		}
	}

	// Detect network partitions periodically (every 20 blocks)
	if ctx.BlockHeight()%20 == 0 {
		if err := k.DetectPartitions(ctx); err != nil {
			ctx.Logger().Error(fmt.Sprintf("Failed to detect partitions: %v", err))
		}
	}

	// Optimize network scaling (every block)
	if err := k.OptimizeNetworkScaling(ctx); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to optimize network scaling: %v", err))
	}

	// Sample automatic data availability challenges for pinned content
	// periodically to provide probabilistic DA guarantees without requiring
	// explicit MsgChallengePinning transactions from users.
	if ctx.BlockHeight()%30 == 0 {
		if err := k.SampleDataAvailabilityChallenges(ctx); err != nil {
			ctx.Logger().Error(fmt.Sprintf("Failed to sample data availability challenges: %v", err))
		}
	}

	// Process pinning rewards (every block)
	if err := k.ProcessPinningRewards(ctx); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to process pinning rewards: %v", err))
	}

	// Process treasury buy-back & burn (every block, checks interval internally)
	if err := k.ProcessTreasuryBuyBack(ctx); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to process treasury buy-back: %v", err))
	}

	// Monitor convergence (every 10 blocks)
	if ctx.BlockHeight()%10 == 0 {
		if err := k.MonitorConvergence(ctx); err != nil {
			ctx.Logger().Error(fmt.Sprintf("Failed to monitor convergence: %v", err))
		}
	}

	// Prune genesis vault periodically (every 1000 blocks)
	// Removes old/unused entries while maintaining minimum vault size
	if ctx.BlockHeight()%1000 == 0 {
		expirationThreshold := int64(5000)  // 5000 blocks = ~1 day (assuming 6s block time)
		minVaultSize := uint64(1000)        // Minimum 1000 entries to maintain
		if err := k.PruneVault(ctx, expirationThreshold, minVaultSize); err != nil {
			ctx.Logger().Error(fmt.Sprintf("Failed to prune genesis vault: %v", err))
		}
	}

	currentHeight := ctx.BlockHeight()
	
	// Store block timestamp for average block time calculation
	// Only keep last 100 blocks to limit state growth
	blockTimestamp := ctx.BlockTime().Unix()
	if err := k.BlockTimestamps.Set(ctx, currentHeight, blockTimestamp); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to store block timestamp: %v", err))
	}
	
	// Clean up old block timestamps (keep only last 100 blocks)
	if currentHeight > 100 {
		oldHeight := currentHeight - 100
		if err := k.BlockTimestamps.Remove(ctx, oldHeight); err != nil {
			// Ignore error if key doesn't exist
			ctx.Logger().Debug(fmt.Sprintf("Failed to remove old block timestamp (may not exist): %v", err))
		}
	}
	
	// Finalize expired pending aggregations using deadline index for O(1) lookup
	// This is more efficient than iterating all aggregations
	finalizedCount := 0
	
	// Get aggregations that expire at current height (or earlier)
	// Note: We check current height exactly, but also check a small range to catch any missed ones
	expiredAggregations, err := k.getAggregationsByDeadline(ctx, currentHeight)
	if err != nil {
		ctx.Logger().Error("Failed to get aggregations by deadline", "height", currentHeight, "error", err)
		// Fallback to iteration if index lookup fails
		expiredAggregations = []uint64{}
	}
	
	// Also check a few blocks before current height (in case index wasn't updated)
	// This is a safety mechanism for edge cases
	for checkHeight := currentHeight - 5; checkHeight < currentHeight; checkHeight++ {
		if checkHeight < 1 {
			continue
		}
		additionalAggregations, err := k.getAggregationsByDeadline(ctx, checkHeight)
		if err == nil {
			// Merge with existing list (avoid duplicates)
			for _, aggID := range additionalAggregations {
				found := false
				for _, existingID := range expiredAggregations {
					if existingID == aggID {
						found = true
						break
					}
				}
				if !found {
					expiredAggregations = append(expiredAggregations, aggID)
				}
			}
		}
	}
	
	// Finalize each expired aggregation
	for _, aggregationID := range expiredAggregations {
		aggregation, err := k.AggregationRecords.Get(ctx, aggregationID)
		if err != nil {
			// Aggregation not found (may have been finalized already or removed)
			// Remove from index
			_ = k.RemoveAggregationFromDeadlineIndex(ctx, currentHeight, aggregationID)
			continue
		}
		
		// Verify aggregation is still pending and deadline has expired
		if aggregation.Status != "pending" {
			// Already finalized or challenged, remove from index
			_ = k.RemoveAggregationFromDeadlineIndex(ctx, aggregation.ChallengeDeadlineHeight, aggregationID)
			continue
		}
		
		if aggregation.ChallengeDeadlineHeight > 0 && currentHeight < aggregation.ChallengeDeadlineHeight {
			// Deadline not yet reached, skip
			continue
		}
		
		// Challenge period expired, finalize the aggregation
		aggregation.Status = "finalized"
		if err := k.AggregationRecords.Set(ctx, aggregation.AggregationId, aggregation); err != nil {
			ctx.Logger().Error("Failed to finalize aggregation", "aggregation_id", aggregation.AggregationId, "error", err)
			continue // Skip this aggregation, try next
		}
		
		// Remove from deadline index
		if err := k.RemoveAggregationFromDeadlineIndex(ctx, aggregation.ChallengeDeadlineHeight, aggregationID); err != nil {
			ctx.Logger().Warn("Failed to remove aggregation from deadline index", "aggregation_id", aggregationID, "error", err)
			// Don't fail - index cleanup is not critical
		}
		
		// Update global model state if this finalized aggregation should become the new global model
		if err := k.UpdateGlobalModelStateIfNeeded(ctx, aggregation); err != nil {
			ctx.Logger().Error("Failed to update global model state", "aggregation_id", aggregation.AggregationId, "error", err)
			// Don't fail - model update is not critical for finalization
		}
		
		// Update trust scores for miners whose gradients were accepted into this aggregation
		// This rewards miners for contributing to a finalized aggregation
		for _, gradientID := range aggregation.ParticipantGradientIds {
			gradient, err := k.StoredGradients.Get(ctx, gradientID)
			if err != nil {
				ctx.Logger().Error("Failed to get gradient for trust score update", "gradient_id", gradientID, "error", err)
				continue
			}
			
			// Update trust score for accepted gradient
			if err := k.UpdateTrustScore(ctx, gradient.Miner, "accepted"); err != nil {
				ctx.Logger().Error("Failed to update miner trust score", "miner", gradient.Miner, "error", err)
				// Don't fail - trust score update is not critical
			}
		}
		
		// Emit event
		ctx.EventManager().EmitEvent(
			types.NewEventAggregationFinalized(
				aggregation.Proposer,
				aggregation.AggregationId,
				aggregation.TrainingRoundId,
			),
		)
		
		finalizedCount++
	}
	
	// Note: Index-based lookup is now primary method
	// This provides O(1) lookup instead of O(n) iteration, significantly improving performance
	// when there are many aggregations
	
	if finalizedCount > 0 {
		ctx.Logger().Info("Finalized expired aggregations", "count", finalizedCount)
	}
	
	return nil
}

// getAllAggregations gets all aggregation records
// NOTE: This function is kept for backward compatibility but is no longer used in FinalizeExpiredAggregations
// Direct iteration with filtering is now used instead for better performance
func (k Keeper) getAllAggregations(ctx sdk.Context) ([]types.AggregationRecord, error) {
	aggregations := []types.AggregationRecord{}
	
	// Iterate through aggregation records
	err := k.AggregationRecords.Walk(ctx, nil, func(key uint64, value types.AggregationRecord) (stop bool, err error) {
		aggregations = append(aggregations, value)
		return false, nil
	})
	
	if err != nil {
		return nil, err
	}
	
	return aggregations, nil
}

