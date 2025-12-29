package keeper

import (
	"context"
	"fmt"

	sdkmath "cosmossdk.io/math"
	"github.com/cosmos/cosmos-sdk/types/query"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"remes/x/remes/types"
)

// QueryMiners queries all miners with pagination (for dashboard API)
func (k Keeper) QueryMiners(ctx context.Context, req *types.QueryMinersRequest) (*types.QueryMinersResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Pagination parameters
	// Default limit reduced from 100 to 50 for better performance
	// Max limit reduced from 1000 to 500 to prevent DoS
	limit := uint64(50) // Default limit
	offset := uint64(0)
	if req.Pagination != nil {
		if req.Pagination.Limit > 0 {
			// Cap limit at 500 to prevent DoS
			if req.Pagination.Limit > 500 {
				limit = 500
			} else {
				limit = req.Pagination.Limit
			}
		}
		if req.Pagination.Offset > 0 {
			offset = req.Pagination.Offset
		}
	}

	var miners []types.MinerInfo
	var total uint64
	currentOffset := uint64(0)
	itemsCollected := uint64(0)
	hasMore := false

	// Pre-compute gradient counts per miner (optimization: single pass instead of nested Walk)
	// This reduces complexity from O(n*m) to O(n+m) where n=miners, m=gradients
	gradientCounts := make(map[string]uint64)
	_ = k.StoredGradients.Walk(sdkCtx, nil, func(gradKey uint64, gradValue types.StoredGradient) (stop bool, err error) {
		gradientCounts[gradValue.Miner]++
		return false, nil
	})

	// Walk through all mining contributions
	err := k.MiningContributions.Walk(sdkCtx, nil, func(key string, value types.MiningContribution) (stop bool, err error) {
		total++

		// Skip items before offset
		if currentOffset < offset {
			currentOffset++
			return false, nil
		}

		// Collect items up to limit
		if itemsCollected < limit {
			// Parse trust score
			trustScore, parseErr := sdkmath.LegacyNewDecFromStr(value.TrustScore)
			if parseErr != nil {
				trustScore = sdkmath.LegacyZeroDec()
			}

			// Determine status based on trust score
			status := "inactive"
			if trustScore.GT(sdkmath.LegacyZeroDec()) {
				status = "active"
			}

			// Get gradient count from pre-computed map (O(1) lookup instead of O(m) Walk)
			totalGradients := gradientCounts[key]

			minerInfo := types.MinerInfo{
				Address:                key,
				Status:                 status,
				TotalSubmissions:        value.TotalSubmissions,
				SuccessfulSubmissions:   value.SuccessfulSubmissions,
				TrustScore:             value.TrustScore,
				ReputationTier:         value.ReputationTier,
				SlashingEvents:         value.SlashingEvents,
				LastSubmissionHeight:   value.LastSubmissionHeight,
				TotalGradients:         totalGradients,
			}

			miners = append(miners, minerInfo)
			itemsCollected++
		} else {
			hasMore = true
			return true, nil
		}

		return false, nil
	})

	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	// Build pagination response
	var pageResponse *query.PageResponse
	if req.Pagination != nil && req.Pagination.CountTotal {
		pageResponse = &query.PageResponse{
			Total: total,
		}
		if hasMore {
			pageResponse.NextKey = []byte("has_more")
		}
	} else if hasMore {
		pageResponse = &query.PageResponse{
			NextKey: []byte("has_more"),
		}
	}

	return &types.QueryMinersResponse{
		Miners:     miners,
		Total:      total,
		Pagination: pageResponse,
	}, nil
}

// QueryStatistics queries network statistics (for dashboard API)
func (k Keeper) QueryStatistics(ctx context.Context, req *types.QueryStatisticsRequest) (*types.QueryStatisticsResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get network metrics
	metrics, err := k.GetNetworkMetrics(sdkCtx)
	if err != nil {
		// If GetNetworkMetrics fails, use zero values
		metrics = NetworkMetrics{
			PendingGradients:     0,
			PendingAggregations:  0,
			AverageGradientSize: 0,
			LastUpdated:          0,
		}
	}

	// Count total miners and active miners (optimized: single pass)
	totalMiners := uint64(0)
	activeMiners := uint64(0)
	_ = k.MiningContributions.Walk(sdkCtx, nil, func(key string, value types.MiningContribution) (stop bool, err error) {
		totalMiners++
		trustScore, parseErr := sdkmath.LegacyNewDecFromStr(value.TrustScore)
		if parseErr == nil && trustScore.GT(sdkmath.LegacyZeroDec()) {
			activeMiners++
		}
		return false, nil
	})

	// Count total gradients (optimized: single pass)
	totalGradients := uint64(0)
	_ = k.StoredGradients.Walk(sdkCtx, nil, func(key uint64, value types.StoredGradient) (stop bool, err error) {
		totalGradients++
		return false, nil
	})

	// Count total aggregations (optimized: single pass)
	totalAggregations := uint64(0)
	_ = k.AggregationRecords.Walk(sdkCtx, nil, func(key uint64, value types.AggregationRecord) (stop bool, err error) {
		totalAggregations++
		return false, nil
	})

	return &types.QueryStatisticsResponse{
		TotalMiners:          totalMiners,
		ActiveMiners:         activeMiners,
		TotalGradients:       totalGradients,
		TotalAggregations:    totalAggregations,
		PendingGradients:     metrics.PendingGradients,
		PendingAggregations:  metrics.PendingAggregations,
		AverageGradientSize:  metrics.AverageGradientSize,
		LastUpdated:          metrics.LastUpdated,
	}, nil
}

// QueryBlocks queries recent blocks with pagination (for dashboard API)
func (k Keeper) QueryBlocks(ctx context.Context, req *types.QueryBlocksRequest) (*types.QueryBlocksResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Pagination parameters
	_ = req.Limit // Limit will be used when historical block support is added
	offset := req.Offset

	// Get current block height
	currentHeight := sdkCtx.BlockHeight()
	if currentHeight < 0 {
		return &types.QueryBlocksResponse{
			Blocks: []types.BlockInfo{},
			Total:  0,
		}, nil
	}

	// Collect blocks (from most recent to oldest)
	var blocks []types.BlockInfo

	// For now, we can only get current block info from SDK context
	// Historical blocks would require block store access or Tendermint RPC
	// This is a simplified implementation that returns current block info when offset == 0
	if offset == 0 {
		// Return current block
		blockHash := sdkCtx.HeaderHash()
		blockHashHex := fmt.Sprintf("%x", blockHash)
		
		// Count transactions in current block (simplified - would need to query block store)
		// For now, we can't easily get tx count from SDK context without block store
		txCount := uint64(0)
		// Note: Actual tx count would require block store access or parsing block data
		
		blocks = append(blocks, types.BlockInfo{
			Height:    currentHeight,
			Hash:      blockHashHex,
			Timestamp: sdkCtx.BlockTime().Unix(),
			TxCount:   txCount,
		})
	}
	
	// Limit results to requested limit
	if uint64(len(blocks)) > req.Limit {
		blocks = blocks[:req.Limit]
	}

	// Note: For full historical block support, we would need:
	// 1. Block store access (requires keeper to have block store reference)
	// 2. Or Tendermint RPC client to query historical blocks
	// This implementation provides current block info as a starting point
	// Total is 1 if we have current block, 0 otherwise
	total := uint64(len(blocks))

	return &types.QueryBlocksResponse{
		Blocks: blocks,
		Total:  total,
	}, nil
}

// QueryBlock queries a specific block by height (for dashboard API)
func (k Keeper) QueryBlock(ctx context.Context, req *types.QueryBlockRequest) (*types.QueryBlockResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get current block height
	currentHeight := sdkCtx.BlockHeight()

	// If height is 0 or negative, return current block
	height := req.Height
	if height <= 0 {
		height = currentHeight
	}

	// Validate height
	if height < 1 {
		return nil, status.Error(codes.InvalidArgument, "invalid block height")
	}

	// If requesting current block, use SDK context
	if height == currentHeight {
		blockHash := sdkCtx.HeaderHash()
		blockHashHex := fmt.Sprintf("%x", blockHash)
		
		// Count transactions (simplified - would need block store access)
		txCount := uint64(0)
		// Note: Actual tx count would require block store access
		
		return &types.QueryBlockResponse{
			Block: types.BlockInfo{
				Height:    height,
				Hash:      blockHashHex,
				Timestamp: sdkCtx.BlockTime().Unix(),
				TxCount:   txCount,
			},
		}, nil
	}

	// For historical blocks, we would need block store access or Tendermint RPC client
	// This is a limitation - we can only query current block from SDK context
	// 
	// Production options:
	// 1. Add block store reference to keeper (requires keeper constructor changes)
	// 2. Use Tendermint RPC client to query historical blocks (recommended for query handlers)
	// 3. Store block metadata in state (BlockTimestamps collection stores last 100 blocks)
	//
	// Current workaround: Use Tendermint RPC endpoint (e.g., /block?height=N) for historical queries
	// This query handler is primarily for current block info via gRPC
	return nil, status.Error(codes.Unimplemented, "historical block queries require block store access or Tendermint RPC - currently only current block is supported via SDK context. Use Tendermint RPC endpoint for historical blocks")
}

// QueryActivePool queries the currently active task pool ID
func (k Keeper) QueryActivePool(ctx context.Context, req *types.QueryActivePoolRequest) (*types.QueryActivePoolResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)
	currentHeight := sdkCtx.BlockHeight()

	// Find the most recent active pool (highest pool_id with status="active" and not expired)
	var activePoolID uint64 = 0
	var foundActivePool bool = false
	var highestPoolID uint64 = 0

	// Iterate through all pools to find active one
	err := k.TaskPools.Walk(ctx, nil, func(key uint64, pool types.TaskPool) (stop bool, err error) {
		// Track highest pool ID seen
		if key > highestPoolID {
			highestPoolID = key
		}

		// Check if pool is active and not expired
		if pool.Status == "active" && int64(currentHeight) < pool.ExpiryHeight {
			// Found an active pool - use the one with highest pool_id
			if key > activePoolID {
				activePoolID = key
				foundActivePool = true
			}
		}
		return false, nil // Continue iteration
	})

	if err != nil {
		return nil, status.Error(codes.Internal, fmt.Sprintf("failed to query task pools: %v", err))
	}

	return &types.QueryActivePoolResponse{
		PoolId:        activePoolID,
		HasActivePool: foundActivePool,
	}, nil
}

// QueryAvailableChunks queries available chunks from a task pool
func (k Keeper) QueryAvailableChunks(ctx context.Context, req *types.QueryAvailableChunksRequest) (*types.QueryAvailableChunksResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	// Validate limit (max 1000 to prevent DoS)
	limit := req.Limit
	if limit == 0 {
		limit = 100 // Default limit
	}
	if limit > 1000 {
		limit = 1000 // Cap at 1000
	}

	// Get available chunks using keeper method (includes trap mixing and sanitization)
	chunks, err := k.GetAvailableChunksForMiner(ctx, req.PoolId, limit)
	if err != nil {
		return nil, status.Error(codes.Internal, fmt.Sprintf("failed to get available chunks: %v", err))
	}

	// Get total available chunks count from pool
	pool, err := k.GetTaskPool(ctx, req.PoolId)
	if err != nil {
		return nil, status.Error(codes.NotFound, fmt.Sprintf("task pool %d not found", req.PoolId))
	}

	// Count available chunks (status="available" and not assigned)
	totalAvailable := uint64(0)
	for _, chunk := range pool.AvailableChunks {
		if chunk.Status == "available" && chunk.AssignedMiner == "" {
			totalAvailable++
		}
	}

	return &types.QueryAvailableChunksResponse{
		Chunks:         chunks,
		TotalAvailable: totalAvailable,
	}, nil
}

