package keeper

import (
	"encoding/json"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
)

// StateInspector provides state inspection capabilities for debugging
type StateInspector struct {
	keeper *Keeper
	config *DebugConfig
	enabled bool
}

// NewStateInspector creates a new state inspector
func NewStateInspector(keeper *Keeper, config *DebugConfig) *StateInspector {
	return &StateInspector{
		keeper:  keeper,
		config:  config,
		enabled: config != nil && config.Enabled && config.StateInspection && config.IsBlockchainEnabled(),
	}
}

// CollectionStats represents statistics for a collection
type CollectionStats struct {
	CollectionName string `json:"collection_name"`
	EntryCount     int64  `json:"entry_count"`
	KeyPrefix      string `json:"key_prefix,omitempty"`
}

// StateDump represents a complete state dump
type StateDump struct {
	BlockHeight      int64                       `json:"block_height"`
	BlockTime        string                      `json:"block_time"`
	Params           interface{}                 `json:"params,omitempty"`
	CollectionStats  []CollectionStats           `json:"collection_stats"`
	CacheStats       map[string]interface{}      `json:"cache_stats,omitempty"`
}

// GetCollectionStats gets statistics for all collections
func (si *StateInspector) GetCollectionStats(ctx sdk.Context) ([]CollectionStats, error) {
	if !si.enabled {
		return nil, fmt.Errorf("state inspection is disabled")
	}

	stats := []CollectionStats{}

	// Count entries in major collections
	// Note: This is a simplified implementation - full implementation would iterate through all collections

	// GlobalModelState (singleton)
	_, err := si.keeper.GlobalModelState.Get(ctx)
	if err == nil {
		stats = append(stats, CollectionStats{
			CollectionName: "GlobalModelState",
			EntryCount:     1,
		})
	}

	// StoredGradients count (simplified - would need iteration)
	stats = append(stats, CollectionStats{
		CollectionName: "StoredGradients",
		EntryCount:     -1, // -1 indicates "unknown" (would require full iteration)
	})

	// MiningContributions count (simplified)
	stats = append(stats, CollectionStats{
		CollectionName: "MiningContributions",
		EntryCount:     -1,
	})

	return stats, nil
}

// GetStateDump creates a complete state dump
func (si *StateInspector) GetStateDump(ctx sdk.Context) (*StateDump, error) {
	if !si.enabled {
		return nil, fmt.Errorf("state inspection is disabled")
	}

	collectionStats, err := si.GetCollectionStats(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get collection stats: %w", err)
	}

	// Get params
	var params interface{}
	paramsObj, err := si.keeper.Params.Get(ctx)
	if err == nil {
		// Convert params to JSON-serializable format
		paramsBytes, _ := json.Marshal(paramsObj)
		json.Unmarshal(paramsBytes, &params)
	}

	// Get cache stats
	cacheStats := map[string]interface{}{}
	if si.keeper.gradientCache != nil {
		cacheStats["gradient_cache"] = map[string]interface{}{
			"enabled": true,
			// Additional cache stats would go here
		}
	}

	return &StateDump{
		BlockHeight:     ctx.BlockHeight(),
		BlockTime:       ctx.BlockTime().Format("2006-01-02T15:04:05Z07:00"),
		Params:          params,
		CollectionStats: collectionStats,
		CacheStats:      cacheStats,
	}, nil
}

// GetParamsDump returns current parameters
func (si *StateInspector) GetParamsDump(ctx sdk.Context) (interface{}, error) {
	if !si.enabled {
		return nil, fmt.Errorf("state inspection is disabled")
	}

	params, err := si.keeper.Params.Get(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get params: %w", err)
	}

	// Convert to JSON-serializable format
	var result interface{}
	paramsBytes, err := json.Marshal(params)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal params: %w", err)
	}
	
	if err := json.Unmarshal(paramsBytes, &result); err != nil {
		return nil, fmt.Errorf("failed to unmarshal params: %w", err)
	}

	return result, nil
}

// GetCacheStats returns cache statistics
func (si *StateInspector) GetCacheStats(ctx sdk.Context) (map[string]interface{}, error) {
	if !si.enabled {
		return nil, fmt.Errorf("state inspection is disabled")
	}

	stats := map[string]interface{}{}

	if si.keeper.gradientCache != nil {
		stats["gradient_cache"] = map[string]interface{}{
			"enabled": true,
			// Additional cache stats would be added here
		}
	} else {
		stats["gradient_cache"] = map[string]interface{}{
			"enabled": false,
		}
	}

	return stats, nil
}

// GetDebugStateInspector returns the state inspector for the keeper
func (k Keeper) GetDebugStateInspector() *StateInspector {
	if k.debugConfig == nil || !k.debugConfig.Enabled || !k.debugConfig.StateInspection || !k.debugConfig.IsBlockchainEnabled() {
		return nil
	}

	return NewStateInspector(&k, k.debugConfig)
}
