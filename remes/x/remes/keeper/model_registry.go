package keeper

import (
	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// RegisterModel registers a new model configuration (governance-only)
// This should be called from a governance proposal handler
func (k Keeper) RegisterModel(
	ctx sdk.Context,
	config types.ModelConfig,
	proposalID uint64,
) (uint64, error) {
	// Generate model ID
	modelID, err := k.ModelID.Next(ctx)
	if err != nil {
		return 0, errorsmod.Wrap(err, "failed to generate model ID")
	}

	// Create registry entry
	registry := types.ModelRegistry{
		ModelId:         modelID,
		Config:          config,
		ApprovedAtHeight: ctx.BlockHeight(),
		ApprovedBy:      proposalID,
		IsActive:        true,
		CreatedAt:      ctx.BlockTime(),
	}

	// Store
	if err := k.ModelRegistries.Set(ctx, modelID, registry); err != nil {
		return 0, errorsmod.Wrap(err, "failed to store model registry")
	}

	return modelID, nil
}

// GetModelConfig retrieves a model configuration by ID
func (k Keeper) GetModelConfig(ctx sdk.Context, modelID uint64) (types.ModelRegistry, error) {
	registry, err := k.ModelRegistries.Get(ctx, modelID)
	if err != nil {
		return types.ModelRegistry{}, errorsmod.Wrapf(err, "model %d not found", modelID)
	}
	return registry, nil
}

// GetActiveModelConfig retrieves the active model configuration
// Returns model_id=1 (default BitNet) if no active model is specified
func (k Keeper) GetActiveModelConfig(ctx sdk.Context) (types.ModelRegistry, error) {
	// Try to find an active model
	var activeModel *types.ModelRegistry
	err := k.ModelRegistries.Walk(ctx, nil, func(key uint64, value types.ModelRegistry) (stop bool, err error) {
		if value.IsActive {
			activeModel = &value
			return true, nil // Stop iteration
		}
		return false, nil
	})

	if err != nil {
		return types.ModelRegistry{}, errorsmod.Wrap(err, "failed to iterate model registries")
	}

	// If no active model found, default to model_id=1 (BitNet)
	if activeModel == nil {
		registry, err := k.ModelRegistries.Get(ctx, 1)
		if err != nil {
			return types.ModelRegistry{}, errorsmod.Wrap(err, "no active model found and default model (ID=1) not found")
		}
		return registry, nil
	}

	return *activeModel, nil
}

// ActivateModel activates a model for training
func (k Keeper) ActivateModel(ctx sdk.Context, modelID uint64) error {
	registry, err := k.ModelRegistries.Get(ctx, modelID)
	if err != nil {
		return errorsmod.Wrapf(err, "model %d not found", modelID)
	}

	registry.IsActive = true
	return k.ModelRegistries.Set(ctx, modelID, registry)
}

// DeactivateModel deactivates a model
func (k Keeper) DeactivateModel(ctx sdk.Context, modelID uint64) error {
	registry, err := k.ModelRegistries.Get(ctx, modelID)
	if err != nil {
		return errorsmod.Wrapf(err, "model %d not found", modelID)
	}

	registry.IsActive = false
	return k.ModelRegistries.Set(ctx, modelID, registry)
}

// ValidateModelConfig validates that a model_config_id exists and is active
func (k Keeper) ValidateModelConfig(ctx sdk.Context, modelConfigID uint64) error {
	// If model_config_id is 0, it's legacy BitNet (backward compatibility)
	if modelConfigID == 0 {
		// Check if default model (ID=1) exists
		_, err := k.ModelRegistries.Get(ctx, 1)
		if err != nil {
			return errorsmod.Wrap(err, "legacy model (ID=0) requires default model (ID=1) to exist")
		}
		return nil
	}

	registry, err := k.ModelRegistries.Get(ctx, modelConfigID)
	if err != nil {
		return errorsmod.Wrapf(err, "model config %d not found", modelConfigID)
	}

	if !registry.IsActive {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "model config %d is not active", modelConfigID)
	}

	return nil
}

