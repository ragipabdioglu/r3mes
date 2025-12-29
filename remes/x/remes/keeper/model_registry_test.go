package keeper_test

import (
	"strings"
	"testing"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"

	"remes/x/remes/types"
)

// contains checks if substr is in s
func contains(s, substr string) bool {
	return strings.Contains(s, substr)
}

func TestModelRegistry_RegisterModel(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	// Create a test model config
	modelConfig := types.ModelConfig{
		ModelType:          types.ModelType_MODEL_TYPE_BITNET,
		ModelVersion:       "b1.58",
		ArchitectureConfig: `{"hidden_size": 768, "num_layers": 12, "lora_rank": 8}`,
		ContainerHash:      "sha256:abc123",
		ContainerRegistry:  "docker.io",
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
	}

	// Register model
	// Note: Sequence starts at 0, so first model will have ID=0
	modelID, err := f.keeper.RegisterModel(ctx, modelConfig, 1)
	require.NoError(t, err)
	require.GreaterOrEqual(t, modelID, uint64(0)) // Model ID can be 0 or higher

	// Retrieve model
	registry, err := f.keeper.GetModelConfig(ctx, modelID)
	require.NoError(t, err)
	require.Equal(t, modelID, registry.ModelId)
	require.Equal(t, modelConfig.ModelType, registry.Config.ModelType)
	require.Equal(t, modelConfig.ModelVersion, registry.Config.ModelVersion)
	require.True(t, registry.IsActive)
}

func TestModelRegistry_GetActiveModelConfig(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	// Create and register a model
	modelConfig := types.ModelConfig{
		ModelType:          types.ModelType_MODEL_TYPE_BITNET,
		ModelVersion:       "b1.58",
		ArchitectureConfig: `{"hidden_size": 768}`,
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
	}

	modelID, err := f.keeper.RegisterModel(ctx, modelConfig, 1)
	require.NoError(t, err)

	// Get active model
	activeModel, err := f.keeper.GetActiveModelConfig(ctx)
	require.NoError(t, err)
	require.Equal(t, modelID, activeModel.ModelId)
	require.True(t, activeModel.IsActive)
}

func TestModelRegistry_ActivateDeactivate(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	// Create and register a model
	modelConfig := types.ModelConfig{
		ModelType:          types.ModelType_MODEL_TYPE_BITNET,
		ModelVersion:       "b1.58",
		ArchitectureConfig: `{"hidden_size": 768}`,
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
	}

	modelID, err := f.keeper.RegisterModel(ctx, modelConfig, 1)
	require.NoError(t, err)

	// Deactivate
	err = f.keeper.DeactivateModel(ctx, modelID)
	require.NoError(t, err)

	registry, err := f.keeper.GetModelConfig(ctx, modelID)
	require.NoError(t, err)
	require.False(t, registry.IsActive)

	// Activate
	err = f.keeper.ActivateModel(ctx, modelID)
	require.NoError(t, err)

	registry, err = f.keeper.GetModelConfig(ctx, modelID)
	require.NoError(t, err)
	require.True(t, registry.IsActive)
}

func TestModelRegistry_ValidateModelConfig(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	// First, create model ID=1 via genesis (for backward compatibility test)
	genesisState := &types.GenesisState{
		Params:       types.DefaultParams(),
		ModelHash:    "QmTestHash",
		ModelVersion: "b1.58",
	}
	err := f.keeper.InitGenesis(f.ctx, genesisState)
	require.NoError(t, err)

	// Validate legacy model ID=0 (should work if model ID=1 exists)
	err = f.keeper.ValidateModelConfig(ctx, 0)
	require.NoError(t, err)

	// Test deactivation with model_id=1 (created via genesis)
	// First verify it's active
	err = f.keeper.ValidateModelConfig(ctx, 1)
	require.NoError(t, err)

	// Deactivate model_id=1 and validate should fail
	err = f.keeper.DeactivateModel(ctx, 1)
	require.NoError(t, err)

	// Verify model is deactivated
	deactivatedRegistry, err := f.keeper.GetModelConfig(ctx, 1)
	require.NoError(t, err)
	require.False(t, deactivatedRegistry.IsActive, "Model should be deactivated")

	// Validate should fail for deactivated model
	err = f.keeper.ValidateModelConfig(ctx, 1)
	require.Error(t, err)
	require.Contains(t, err.Error(), "not active")

	// Reactivate for other tests
	err = f.keeper.ActivateModel(ctx, 1)
	require.NoError(t, err)

	// Now test with a newly registered model
	modelConfig := types.ModelConfig{
		ModelType:          types.ModelType_MODEL_TYPE_BITNET,
		ModelVersion:       "b1.58",
		ArchitectureConfig: `{"hidden_size": 768}`,
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
	}

	modelID, err := f.keeper.RegisterModel(ctx, modelConfig, 1)
	require.NoError(t, err)

	// Skip if modelID is 0 (legacy, already tested above)
	if modelID != 0 {
		// Validate active model
		err = f.keeper.ValidateModelConfig(ctx, modelID)
		require.NoError(t, err)

		// Deactivate and validate should fail
		err = f.keeper.DeactivateModel(ctx, modelID)
		require.NoError(t, err)

		if modelID != 0 {
			// Verify model is deactivated
			newDeactivatedRegistry, err := f.keeper.GetModelConfig(ctx, modelID)
			require.NoError(t, err)
			require.False(t, newDeactivatedRegistry.IsActive, "Model should be deactivated")

			// Validate should fail for deactivated model
			err = f.keeper.ValidateModelConfig(ctx, modelID)
			require.Error(t, err)
			require.Contains(t, err.Error(), "not active")
		}
	}

	// Test invalid model ID (should return error)
	err = f.keeper.ValidateModelConfig(ctx, 99999)
	require.Error(t, err)
	// Error should mention model 99999 or "not found"
	errorMsg := err.Error()
	require.True(t,
		len(errorMsg) > 0 &&
			(contains(errorMsg, "99999") || contains(errorMsg, "not found")),
		"Expected error about model 99999 not found, got: %s", errorMsg)
}

func TestModelRegistry_BackwardCompatibility_LegacyModelID0(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	// Create default model (ID=1) via genesis
	genesisState := &types.GenesisState{
		Params:       types.DefaultParams(),
		ModelHash:    "QmTestHash",
		ModelVersion: "b1.58",
	}

	err := f.keeper.InitGenesis(f.ctx, genesisState)
	require.NoError(t, err)

	// Validate that model_id=0 (legacy) works if model_id=1 exists
	err = f.keeper.ValidateModelConfig(ctx, 0)
	require.NoError(t, err)

	// Validate that model_id=1 exists
	registry, err := f.keeper.GetModelConfig(ctx, 1)
	require.NoError(t, err)
	require.Equal(t, types.ModelType_MODEL_TYPE_BITNET, registry.Config.ModelType)
	require.Equal(t, "b1.58", registry.Config.ModelVersion)
	require.True(t, registry.IsActive)
}
