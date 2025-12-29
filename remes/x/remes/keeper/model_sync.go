package keeper

import (
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// SyncServingNodeModelVersions synchronizes model versions for all serving nodes
// This is called when a new model version is activated
func (k Keeper) SyncServingNodeModelVersions(ctx sdk.Context) error {
	// Get active model version
	activeVersions, err := k.ActiveModelVersions.Get(ctx)
	if err != nil {
		// No active versions - nothing to sync
		return nil
	}

	// Get the latest active model version
	var latestModelVersion *types.ModelVersion
	var latestModelRegistry *types.ModelRegistry
	for _, versionNumber := range activeVersions.VersionNumbers {
		version, err := k.ModelVersions.Get(ctx, versionNumber)
		if err != nil {
			continue
		}

		// Get model registry (config)
		modelRegistry, err := k.ModelRegistries.Get(ctx, version.ModelId)
		if err != nil {
			continue
		}

		// Track latest version
		if latestModelVersion == nil || version.VersionNumber > latestModelVersion.VersionNumber {
			latestModelVersion = &version
			latestModelRegistry = &modelRegistry
		}
	}

	if latestModelVersion == nil || latestModelRegistry == nil {
		// No active model versions
		return nil
	}

	// Iterate through all serving node statuses
	iter, err := k.ServingNodeStatuses.Iterate(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to iterate serving nodes: %w", err)
	}
	defer iter.Close()

	syncedCount := 0
	for ; iter.Valid(); iter.Next() {
		servingNodeAddr, err := iter.Key()
		if err != nil {
			continue
		}

		status, err := iter.Value()
		if err != nil {
			continue
		}

		// Check if serving node needs model update
		needsUpdate := false
		modelVersionStr := fmt.Sprintf("v%d", latestModelVersion.VersionNumber)
		if status.ModelVersion != modelVersionStr {
			needsUpdate = true
		}

		// Check if model IPFS hash is outdated
		if status.ModelIpfsHash != latestModelVersion.ModelHash {
			needsUpdate = true
		}

		if needsUpdate {
			// Update serving node status to indicate sync needed
			status.ModelVersion = modelVersionStr
			status.ModelIpfsHash = latestModelVersion.ModelHash
			status.LastHeartbeat = ctx.BlockTime()

			// Mark as unavailable until model is downloaded
			status.IsAvailable = false

			if err := k.ServingNodeStatuses.Set(ctx, servingNodeAddr, status); err != nil {
				ctx.Logger().Error(fmt.Sprintf("Failed to update serving node %s: %v", servingNodeAddr, err))
				continue
			}

			// Emit event for model sync notification
			ctx.EventManager().EmitEvent(
				sdk.NewEvent(
					types.EventTypeModelVersionSync,
					sdk.NewAttribute(types.AttributeKeyServingNode, servingNodeAddr),
				sdk.NewAttribute(types.AttributeKeyModelVersion, modelVersionStr),
				sdk.NewAttribute(types.AttributeKeyModelIPFSHash, latestModelVersion.ModelHash),
					sdk.NewAttribute(types.AttributeKeySyncRequired, "true"),
				),
			)

			syncedCount++
		}
	}

	if syncedCount > 0 {
		ctx.Logger().Info(fmt.Sprintf("Synchronized model versions for %d serving nodes", syncedCount))
	}

	return nil
}

// NotifyServingNodesOfModelUpdate notifies serving nodes of a new model version
// This is called when a model version is activated
func (k Keeper) NotifyServingNodesOfModelUpdate(
	ctx sdk.Context,
	modelVersionNumber uint64,
) error {
	// Get model version
	modelVersion, err := k.ModelVersions.Get(ctx, modelVersionNumber)
	if err != nil {
		return fmt.Errorf("model version not found: %w", err)
	}

	// Sync all serving nodes
	if err := k.SyncServingNodeModelVersions(ctx); err != nil {
		return fmt.Errorf("failed to sync serving nodes: %w", err)
	}

	// Emit global event for model update
	modelVersionStr := fmt.Sprintf("v%d", modelVersionNumber)
	ctx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeModelVersionActivated,
			sdk.NewAttribute(types.AttributeKeyModelVersionID, fmt.Sprintf("%d", modelVersionNumber)),
			sdk.NewAttribute(types.AttributeKeyModelVersion, modelVersionStr),
			sdk.NewAttribute(types.AttributeKeyModelIPFSHash, modelVersion.ModelHash),
			sdk.NewAttribute(types.AttributeKeySyncRequired, "true"),
		),
	)

	return nil
}

// VerifyServingNodeModelSync verifies that a serving node has synced to the latest model
func (k Keeper) VerifyServingNodeModelSync(
	ctx sdk.Context,
	servingNodeAddr string,
) (bool, error) {
	// Get serving node status
	status, err := k.ServingNodeStatuses.Get(ctx, servingNodeAddr)
	if err != nil {
		return false, fmt.Errorf("serving node not found: %w", err)
	}

	// Get current global model state
	globalState, err := k.GlobalModelState.Get(ctx)
	if err != nil {
		// No global state - consider synced
		return true, nil
	}

	// Get active model versions
	activeVersions, err := k.ActiveModelVersions.Get(ctx)
	if err != nil {
		// No active versions - consider synced
		return true, nil
	}

	// Get latest model version
	var latestModelVersion *types.ModelVersion
	for _, versionNumber := range activeVersions.VersionNumbers {
		version, err := k.ModelVersions.Get(ctx, versionNumber)
		if err != nil {
			continue
		}

		if latestModelVersion == nil || version.VersionNumber > latestModelVersion.VersionNumber {
			latestModelVersion = &version
		}
	}

	if latestModelVersion == nil {
		// No active versions - consider synced
		return true, nil
	}

	// Verify model version matches
	modelVersionStr := fmt.Sprintf("v%d", latestModelVersion.VersionNumber)
	if status.ModelVersion != modelVersionStr {
		return false, nil
	}

	// Verify model IPFS hash matches
	if status.ModelIpfsHash != latestModelVersion.ModelHash {
		return false, nil
	}

	// Verify model version matches global state
	if status.ModelVersion != globalState.ModelVersion {
		return false, nil
	}

	return true, nil
}

