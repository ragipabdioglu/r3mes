package keeper

import (
	"context"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// QueryVaultStats queries genesis vault statistics
func (k Keeper) QueryVaultStats(ctx context.Context, req *types.QueryVaultStatsRequest) (*types.QueryVaultStatsResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get total entries count by iterating
	totalEntries := uint64(0)
	err := k.GenesisVault.Walk(ctx, nil, func(key uint64, entry types.GenesisVaultEntry) (stop bool, err error) {
		totalEntries++
		return false, nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to count vault entries: %w", err)
	}

	if totalEntries == 0 {
		return &types.QueryVaultStatsResponse{
			TotalEntries:           0,
			TotalUsageCount:        0,
			AverageUsagePerEntry:   0,
			OldestEntryHeight:      0,
			NewestEntryHeight:      0,
		}, nil
	}

	// Iterate through entries to calculate statistics
	var totalUsageCount uint64 = 0
	var oldestEntryHeight int64 = sdkCtx.BlockHeight()
	var newestEntryHeight int64 = 0

	err = k.GenesisVault.Walk(ctx, nil, func(key uint64, entry types.GenesisVaultEntry) (stop bool, err error) {
		totalUsageCount += entry.UsageCount
		
		if entry.CreatedHeight < oldestEntryHeight {
			oldestEntryHeight = entry.CreatedHeight
		}
		if entry.CreatedHeight > newestEntryHeight {
			newestEntryHeight = entry.CreatedHeight
		}
		
		return false, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to iterate vault entries: %w", err)
	}

	// Calculate average usage per entry
	averageUsagePerEntry := uint64(0)
	if totalEntries > 0 {
		averageUsagePerEntry = totalUsageCount / totalEntries
	}

	return &types.QueryVaultStatsResponse{
		TotalEntries:          totalEntries,
		TotalUsageCount:       totalUsageCount,
		AverageUsagePerEntry:  averageUsagePerEntry,
		OldestEntryHeight:     uint64(oldestEntryHeight),
		NewestEntryHeight:     uint64(newestEntryHeight),
	}, nil
}

// QueryMinerFraudScore queries miner's fraud score and trap statistics
func (k Keeper) QueryMinerFraudScore(ctx context.Context, req *types.QueryMinerFraudScoreRequest) (*types.QueryMinerFraudScoreResponse, error) {
	if req.Miner == "" {
		return nil, fmt.Errorf("miner address cannot be empty")
	}

	// Get miner's contribution record
	contribution, err := k.MiningContributions.Get(ctx, req.Miner)
	if err != nil {
		// If miner doesn't have a contribution record, return zero values
		return &types.QueryMinerFraudScoreResponse{
			Miner:       req.Miner,
			TrapsCaught: 0,
			TrapsFailed: 0,
			FraudScore:  "0.000000",
		}, nil
	}

	// Get fraud score (default to 0.0 if not set)
	fraudScore := "0.000000"
	if contribution.FraudScore != "" {
		fraudScore = contribution.FraudScore
	}

	return &types.QueryMinerFraudScoreResponse{
		Miner:       req.Miner,
		TrapsCaught: contribution.TrapsCaught,
		TrapsFailed: contribution.TrapsFailed,
		FraudScore:  fraudScore,
	}, nil
}

