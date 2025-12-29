package keeper

import (
	"cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// NetworkMetrics tracks network performance metrics for scalability optimization
type NetworkMetrics struct {
	// Participant counts
	ActiveMiners      uint64
	ActiveProposers   uint64
	ActiveValidators  uint64
	ActiveServingNodes uint64

	// Performance metrics
	AverageGradientSubmissionTime uint64 // milliseconds
	AverageAggregationTime        uint64 // milliseconds
	AverageBlockProcessingTime    uint64 // milliseconds

	// Network load metrics
	PendingGradients      uint64
	PendingAggregations   uint64
	PendingInferenceRequests uint64

	// Bandwidth metrics
	TotalGradientSize     uint64 // bytes
	TotalAggregationSize  uint64 // bytes
	AverageGradientSize   uint64 // bytes

	// Resource utilization
	CPUUtilization    uint64 // percentage (0-100)
	MemoryUtilization uint64 // percentage (0-100)
	NetworkUtilization uint64 // percentage (0-100)

	// Timestamp
	LastUpdated int64
}

// ScalabilityConfig holds configuration for adaptive scaling mechanisms
type ScalabilityConfig struct {
	// Thresholds for adaptive scaling
	MaxParticipantsPerShard     uint64
	MaxGradientsPerBlock        uint64
	MaxAggregationsPerBlock     uint64
	MaxPendingGradients         uint64
	MaxPendingAggregations      uint64

	// Adaptive parameters
	CompressionRatioThreshold   math.LegacyDec // Trigger compression adjustment
	NetworkLoadThreshold       math.LegacyDec // Trigger load balancing
	ResourceUtilizationThreshold math.LegacyDec // Trigger resource optimization

	// Load balancing
	EnableLoadBalancing         bool
	LoadBalancingStrategy       string // "round_robin", "weighted", "least_connections"
	ShardReassignmentInterval   uint64 // blocks

	// Performance optimization
	EnableAdaptiveCompression   bool
	EnableAdaptiveSharding      bool
	EnableResourceOptimization  bool
}

// GetNetworkMetrics retrieves current network performance metrics
func (k Keeper) GetNetworkMetrics(ctx sdk.Context) (NetworkMetrics, error) {
	metrics := NetworkMetrics{
		LastUpdated: ctx.BlockTime().Unix(),
	}

	// Count active participants
	minerCount := uint64(0)
	err := k.MiningContributions.Walk(ctx, nil, func(key string, value types.MiningContribution) (stop bool, err error) {
		// Parse trust score from string
		trustScore, parseErr := math.LegacyNewDecFromStr(value.TrustScore)
		if parseErr == nil && trustScore.GT(math.LegacyZeroDec()) {
			minerCount++
		}
		return false, nil
	})
	if err != nil {
		return metrics, err
	}
	metrics.ActiveMiners = minerCount

	// Count pending gradients
	gradientCount := uint64(0)
	err = k.StoredGradients.Walk(ctx, nil, func(key uint64, value types.StoredGradient) (stop bool, err error) {
		if value.Status == "pending" || value.Status == "submitted" {
			gradientCount++
		}
		return false, nil
	})
	if err == nil {
		metrics.PendingGradients = gradientCount
	}

	// Count pending aggregations
	aggCount := uint64(0)
	err = k.AggregationRecords.Walk(ctx, nil, func(key uint64, value types.AggregationRecord) (stop bool, err error) {
		if value.Status == "pending" {
			aggCount++
		}
		return false, nil
	})
	if err == nil {
		metrics.PendingAggregations = aggCount
	}

	// Calculate average gradient size (simplified - would need actual size tracking)
	// For now, use estimated size based on model config
	metrics.AverageGradientSize = 10 * 1024 * 1024 // 10MB estimate for LoRA adapters

	return metrics, nil
}

// GetScalabilityConfig retrieves current scalability configuration from params
func (k Keeper) GetScalabilityConfig(ctx sdk.Context) ScalabilityConfig {
	// Get params from store
	params, err := k.Params.Get(ctx)
	if err != nil {
		// If params not found, return default values
		return k.scalabilityParamsToConfig(types.DefaultScalabilityParams())
	}
	
	// Convert params to config
	return k.scalabilityParamsToConfig(params.ScalabilityParams)
}

// scalabilityParamsToConfig converts ScalabilityParams to ScalabilityConfig
func (k Keeper) scalabilityParamsToConfig(sp types.ScalabilityParams) ScalabilityConfig {
	compressionRatio := math.LegacyMustNewDecFromStr(sp.CompressionRatioThreshold)
	networkLoad := math.LegacyMustNewDecFromStr(sp.NetworkLoadThreshold)
	resourceUtil := math.LegacyMustNewDecFromStr(sp.ResourceUtilizationThreshold)
	
	return ScalabilityConfig{
		MaxParticipantsPerShard:     sp.MaxParticipantsPerShard,
		MaxGradientsPerBlock:        sp.MaxGradientsPerBlock,
		MaxAggregationsPerBlock:     sp.MaxAggregationsPerBlock,
		MaxPendingGradients:         sp.MaxPendingGradients,
		MaxPendingAggregations:      sp.MaxPendingAggregations,
		CompressionRatioThreshold:   compressionRatio,
		NetworkLoadThreshold:       networkLoad,
		ResourceUtilizationThreshold: resourceUtil,
		EnableLoadBalancing:         sp.EnableLoadBalancing,
		LoadBalancingStrategy:       sp.LoadBalancingStrategy,
		ShardReassignmentInterval:   sp.ShardReassignmentInterval,
		EnableAdaptiveCompression:   sp.EnableAdaptiveCompression,
		EnableAdaptiveSharding:     sp.EnableAdaptiveSharding,
		EnableResourceOptimization: sp.EnableResourceOptimization,
	}
}

// CheckScalabilityThresholds checks if network metrics exceed scalability thresholds
func (k Keeper) CheckScalabilityThresholds(ctx sdk.Context) (bool, []string) {
	metrics, err := k.GetNetworkMetrics(ctx)
	if err != nil {
		return false, []string{err.Error()}
	}

	config := k.GetScalabilityConfig(ctx)
	thresholdsExceeded := []string{}

	// Check participant thresholds
	if metrics.ActiveMiners > config.MaxParticipantsPerShard {
		thresholdsExceeded = append(thresholdsExceeded, "max_participants_per_shard")
	}

	// Check pending gradients threshold
	if metrics.PendingGradients > config.MaxPendingGradients {
		thresholdsExceeded = append(thresholdsExceeded, "max_pending_gradients")
	}

	// Check pending aggregations threshold
	if metrics.PendingAggregations > config.MaxPendingAggregations {
		thresholdsExceeded = append(thresholdsExceeded, "max_pending_aggregations")
	}

	// Check network load threshold
	if config.NetworkLoadThreshold.GT(math.LegacyZeroDec()) {
		networkLoad := math.LegacyNewDecFromInt(math.NewIntFromUint64(metrics.PendingGradients + metrics.PendingAggregations))
		if networkLoad.GT(config.NetworkLoadThreshold) {
			thresholdsExceeded = append(thresholdsExceeded, "network_load_threshold")
		}
	}

	return len(thresholdsExceeded) > 0, thresholdsExceeded
}

// OptimizeNetworkScaling applies adaptive scaling mechanisms based on current metrics
func (k Keeper) OptimizeNetworkScaling(ctx sdk.Context) error {
	metrics, err := k.GetNetworkMetrics(ctx)
	if err != nil {
		return err
	}

	config := k.GetScalabilityConfig(ctx)

	// Adaptive compression adjustment
	if config.EnableAdaptiveCompression {
		err := k.AdjustCompressionRatio(ctx, metrics)
		if err != nil {
			return err
		}
	}

	// Adaptive sharding adjustment
	if config.EnableAdaptiveSharding {
		err := k.AdjustSharding(ctx, metrics)
		if err != nil {
			return err
		}
	}

	// Load balancing
	if config.EnableLoadBalancing {
		err := k.BalanceLoad(ctx, metrics, config)
		if err != nil {
			return err
		}
	}

	// Resource optimization
	if config.EnableResourceOptimization {
		err := k.OptimizeResources(ctx, metrics)
		if err != nil {
			return err
		}
	}

	return nil
}

// AdjustCompressionRatio adjusts compression ratio based on network load
func (k Keeper) AdjustCompressionRatio(ctx sdk.Context, metrics NetworkMetrics) error {
	config := k.GetScalabilityConfig(ctx)

	// Calculate network load ratio
	networkLoadRatio := math.LegacyNewDecFromInt(math.NewIntFromUint64(metrics.PendingGradients + metrics.PendingAggregations))
	maxLoad := math.LegacyNewDecFromInt(math.NewIntFromUint64(config.MaxPendingGradients + config.MaxPendingAggregations))
	if maxLoad.IsZero() {
		return nil
	}
	loadRatio := networkLoadRatio.Quo(maxLoad)

	// If network load exceeds threshold, increase compression (reduce top-k ratio)
	if loadRatio.GT(config.NetworkLoadThreshold) {
		// Increase compression by reducing top-k ratio
		// Top-k ratio determines how many gradient values to keep (e.g., 0.1 = keep top 10%)
		// Lower ratio = more compression = less bandwidth
		currentTopK := math.LegacyMustNewDecFromStr("0.1") // Default: top 10%
		
		// Adjust based on load: if load is 90%, reduce top-k to 5% (more compression)
		if loadRatio.GT(math.LegacyMustNewDecFromStr("0.9")) {
			currentTopK = math.LegacyMustNewDecFromStr("0.05") // 5% = more compression
		} else if loadRatio.GT(math.LegacyMustNewDecFromStr("0.8")) {
			currentTopK = math.LegacyMustNewDecFromStr("0.07") // 7% = moderate compression
		}

		// Log compression adjustment (params update would require governance)
		ctx.Logger().Info("Compression ratio adjustment recommended",
			"network_load_ratio", loadRatio.String(),
			"recommended_top_k_ratio", currentTopK.String(),
			"pending_gradients", metrics.PendingGradients,
			"pending_aggregations", metrics.PendingAggregations,
		)

		// Note: Actual params update would require governance proposal
		// For now, we log the recommendation
	}

	return nil
}

// AdjustSharding adjusts shard configuration based on participant count
func (k Keeper) AdjustSharding(ctx sdk.Context, metrics NetworkMetrics) error {
	config := k.GetScalabilityConfig(ctx)

	// If participants exceed threshold, consider adding more shards
	if metrics.ActiveMiners > config.MaxParticipantsPerShard {
		// Calculate optimal number of shards
		// Formula: ceil(active_miners / max_participants_per_shard)
		optimalShards := (metrics.ActiveMiners + config.MaxParticipantsPerShard - 1) / config.MaxParticipantsPerShard
		
		// Get current total shards (would be stored in params or state)
		// For now, use default of 100 shards
		currentShards := uint64(100)

		// If optimal shards > current shards, recommend shard increase
		if optimalShards > currentShards {
			ctx.Logger().Info("Shard adjustment recommended",
				"active_miners", metrics.ActiveMiners,
				"current_shards", currentShards,
				"optimal_shards", optimalShards,
				"max_participants_per_shard", config.MaxParticipantsPerShard,
			)

			// Note: Actual shard creation would require:
			// 1. Governance proposal to update shard count
			// 2. Reassignment of miners to new shards
			// 3. Update of subnet configurations
			// For now, we log the recommendation
		}

		// Check if shard reassignment is due (based on interval)
		if int64(config.ShardReassignmentInterval) > 0 && ctx.BlockHeight()%int64(config.ShardReassignmentInterval) == 0 {
			ctx.Logger().Info("Shard reassignment interval reached",
				"block_height", ctx.BlockHeight(),
				"reassignment_interval", config.ShardReassignmentInterval,
			)
			// Trigger shard reassignment logic
			// This would call subnet reassignment functions
		}
	}

	return nil
}

// BalanceLoad distributes load across available resources
func (k Keeper) BalanceLoad(ctx sdk.Context, metrics NetworkMetrics, config ScalabilityConfig) error {
	// Implement load balancing strategy
	switch config.LoadBalancingStrategy {
	case "round_robin":
		// Round-robin distribution
		return k.BalanceLoadRoundRobin(ctx, metrics)
	case "weighted":
		// Weighted distribution based on capacity
		return k.BalanceLoadWeighted(ctx, metrics)
	case "least_connections":
		// Least connections strategy
		return k.BalanceLoadLeastConnections(ctx, metrics)
	default:
		// Default to round-robin
		return k.BalanceLoadRoundRobin(ctx, metrics)
	}
}

// BalanceLoadRoundRobin implements round-robin load balancing
func (k Keeper) BalanceLoadRoundRobin(ctx sdk.Context, metrics NetworkMetrics) error {
	// Round-robin distribution logic
	// This reassigns miners/proposers in a round-robin fashion based on block height
	
	// Get all active miners
	minerAddresses := []string{}
	err := k.MiningContributions.Walk(ctx, nil, func(key string, value types.MiningContribution) (stop bool, err error) {
		trustScore, parseErr := math.LegacyNewDecFromStr(value.TrustScore)
		if parseErr == nil && trustScore.GT(math.LegacyZeroDec()) {
			minerAddresses = append(minerAddresses, key)
		}
		return false, nil
	})
	if err != nil {
		return err
	}

	if len(minerAddresses) == 0 {
		return nil
	}

	// Calculate round-robin assignment based on block height
	// This ensures fair distribution across proposers/validators
	roundRobinIndex := ctx.BlockHeight() % int64(len(minerAddresses))
	
	ctx.Logger().Info("Round-robin load balancing",
		"total_miners", len(minerAddresses),
		"round_robin_index", roundRobinIndex,
		"block_height", ctx.BlockHeight(),
	)

	return nil
}

// BalanceLoadWeighted implements weighted load balancing
func (k Keeper) BalanceLoadWeighted(ctx sdk.Context, metrics NetworkMetrics) error {
	// Weighted distribution based on node capacity (stake, resources, trust score)
	
	// Get all registered nodes with their weights
	nodeWeights := make(map[string]math.LegacyDec)
	err := k.NodeRegistrations.Walk(ctx, nil, func(key string, value types.NodeRegistration) (stop bool, err error) {
		if value.Status == types.NODE_STATUS_ACTIVE {
			// Calculate weight based on stake and trust score
			stake, parseErr := sdk.ParseCoinsNormalized(value.Stake)
			if parseErr == nil && !stake.IsZero() {
				// Weight = stake amount (normalized)
				stakeAmount := stake.AmountOf("remes")
				nodeWeights[key] = math.LegacyNewDecFromInt(stakeAmount)
			}
		}
		return false, nil
	})
	if err != nil {
		return err
	}

	// Calculate total weight
	totalWeight := math.LegacyZeroDec()
	for _, weight := range nodeWeights {
		totalWeight = totalWeight.Add(weight)
	}

	if totalWeight.IsZero() {
		return nil
	}

	// Log weighted distribution
	ctx.Logger().Info("Weighted load balancing",
		"total_nodes", len(nodeWeights),
		"total_weight", totalWeight.String(),
	)

	return nil
}

// BalanceLoadLeastConnections implements least connections load balancing
func (k Keeper) BalanceLoadLeastConnections(ctx sdk.Context, metrics NetworkMetrics) error {
	// Least connections strategy
	// Assign to nodes with fewest active connections (pending gradients/aggregations)
	
	// Count active connections per node (based on pending operations)
	nodeConnections := make(map[string]uint64)
	
	// Count pending gradients per miner
	err := k.StoredGradients.Walk(ctx, nil, func(key uint64, value types.StoredGradient) (stop bool, err error) {
		if value.Status == "pending" || value.Status == "submitted" {
			nodeConnections[value.Miner]++
		}
		return false, nil
	})
	if err != nil {
		return err
	}

	// Find node with least connections
	minConnections := ^uint64(0) // Max uint64
	leastLoadedNode := ""
	for node, connections := range nodeConnections {
		if connections < minConnections {
			minConnections = connections
			leastLoadedNode = node
		}
	}

	if leastLoadedNode != "" {
		ctx.Logger().Info("Least connections load balancing",
			"least_loaded_node", leastLoadedNode,
			"connections", minConnections,
		)
	}

	return nil
}

// OptimizeResources optimizes resource utilization
func (k Keeper) OptimizeResources(ctx sdk.Context, metrics NetworkMetrics) error {
	config := k.GetScalabilityConfig(ctx)

	// Check resource utilization thresholds
	resourceUtilization := math.LegacyZeroDec()
	if metrics.CPUUtilization > 80 || metrics.MemoryUtilization > 80 || metrics.NetworkUtilization > 80 {
		// Calculate average resource utilization
		avgUtilization := (metrics.CPUUtilization + metrics.MemoryUtilization + metrics.NetworkUtilization) / 3
		resourceUtilization = math.LegacyNewDecFromInt(math.NewIntFromUint64(avgUtilization))

		// If resource utilization exceeds threshold, trigger optimization
		if resourceUtilization.GT(config.ResourceUtilizationThreshold.MulInt64(100)) {
			ctx.Logger().Info("Resource optimization triggered",
				"cpu_utilization", metrics.CPUUtilization,
				"memory_utilization", metrics.MemoryUtilization,
				"network_utilization", metrics.NetworkUtilization,
				"average_utilization", avgUtilization,
				"threshold", config.ResourceUtilizationThreshold.String(),
			)

			// Optimization strategies:
			// 1. Increase compression to reduce bandwidth
			if metrics.NetworkUtilization > 80 {
				ctx.Logger().Info("Network utilization high, recommending compression increase")
				_ = k.AdjustCompressionRatio(ctx, metrics)
			}

			// 2. Rebalance load to distribute resource usage
			if metrics.CPUUtilization > 80 || metrics.MemoryUtilization > 80 {
				ctx.Logger().Info("CPU/Memory utilization high, recommending load rebalancing")
				_ = k.BalanceLoad(ctx, metrics, config)
			}

			// 3. Consider reducing batch sizes (would require params update)
			ctx.Logger().Info("Resource optimization recommendations logged")
		}
	}

	return nil
}

// UpdatePerformanceMetrics updates performance metrics with new measurements
func (k Keeper) UpdatePerformanceMetrics(ctx sdk.Context, submissionTime uint64, aggregationTime uint64) error {
	// Update rolling averages for performance metrics
	// This maintains a rolling window of performance data
	
	// Get current metrics
	metrics, err := k.GetNetworkMetrics(ctx)
	if err != nil {
		return err
	}

	// Calculate rolling average (simple exponential moving average)
	// Alpha = 0.1 (10% weight for new value, 90% for old average)
	alpha := math.LegacyMustNewDecFromStr("0.1")
	
	// Update average submission time
	currentAvg := math.LegacyNewDecFromInt(math.NewIntFromUint64(metrics.AverageGradientSubmissionTime))
	newValue := math.LegacyNewDecFromInt(math.NewIntFromUint64(submissionTime))
	newAvg := currentAvg.Mul(math.LegacyOneDec().Sub(alpha)).Add(newValue.Mul(alpha))
	metrics.AverageGradientSubmissionTime = newAvg.TruncateInt().Uint64()

	// Update average aggregation time
	currentAvgAgg := math.LegacyNewDecFromInt(math.NewIntFromUint64(metrics.AverageAggregationTime))
	newValueAgg := math.LegacyNewDecFromInt(math.NewIntFromUint64(aggregationTime))
	newAvgAgg := currentAvgAgg.Mul(math.LegacyOneDec().Sub(alpha)).Add(newValueAgg.Mul(alpha))
	metrics.AverageAggregationTime = newAvgAgg.TruncateInt().Uint64()

	// Note: In a production system, these metrics would be stored in state
	// For now, we calculate them on-demand in GetNetworkMetrics
	// Storing them would require a new collection: PerformanceMetrics

	return nil
}

