package keeper

import (
	"context"
	"errors"
	"time"

	"cosmossdk.io/collections"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// InitGenesis initializes the module's state from a provided genesis state.
func (k Keeper) InitGenesis(ctx context.Context, genState *types.GenesisState) error {
	// Set params
	if err := k.Params.Set(ctx, genState.Params); err != nil {
		return err
	}

	// Initialize global model state if model hash is provided
	if genState.ModelHash != "" {
		globalModelState := types.GlobalModelState{
			ModelIpfsHash:      genState.ModelHash,
			ModelVersion:       genState.ModelVersion,
			LastUpdatedHeight:  0, // Genesis block
			LastUpdatedTime:    time.Now(),
			TrainingRoundId:    0, // Genesis - no training round yet
			LastAggregationId:  0, // Genesis - no aggregation yet
		}
		if err := k.GlobalModelState.Set(ctx, globalModelState); err != nil {
			return err
		}
	}

	// Initialize stored gradients from genesis
	for _, sg := range genState.StoredGradientList {
		if err := k.StoredGradients.Set(ctx, sg.Id, sg); err != nil {
			return err
		}
	}

	// Initialize model registry from genesis
	// If model_registry_list is provided, use it
	if len(genState.ModelRegistryList) > 0 {
		for _, modelRegistry := range genState.ModelRegistryList {
			if err := k.ModelRegistries.Set(ctx, modelRegistry.ModelId, modelRegistry); err != nil {
				return err
			}
		}
	} else if genState.ModelHash != "" {
		// Backward compatibility: Auto-create Model ID 1 from legacy fields
		// NOTE: In InitGenesis, ensure backward compatibility by auto-creating Model ID 1 if legacy fields exist.
		sdkCtx := sdk.UnwrapSDKContext(ctx)
		
		// Create ModelConfig from legacy fields
		modelConfig := types.ModelConfig{
			ModelType:        types.ModelType_MODEL_TYPE_BITNET,
			ModelVersion:     genState.ModelVersion,
			ArchitectureConfig: `{"hidden_size": 768, "num_layers": 12, "lora_rank": 8}`,
			ContainerHash:    "", // Will be set by governance later
			ContainerRegistry: "docker.io",
			ExecutionEnvironmentId: 0, // Default
			EncryptionType:   types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
			EncryptedWeightsIpfsHash: "",
			DecryptionKeyHolder: "",
		}

		// Create ModelRegistry with model_id = 1
		modelRegistry := types.ModelRegistry{
			ModelId:         1,
			Config:          modelConfig,
			ApprovedAtHeight: 0, // Genesis
			ApprovedBy:      0, // Genesis (no proposal)
			IsActive:        true,
			CreatedAt:      sdkCtx.BlockTime(),
		}

		// Store model registry
		if err := k.ModelRegistries.Set(ctx, 1, modelRegistry); err != nil {
			return err
		}

		// Set ModelID counter to 2 (next model will be ID 2)
		// Note: We can't directly set Sequence, but we can ensure it starts at 2
		// by checking if model_id=1 exists and initializing accordingly
		// The Sequence will auto-increment from its current state
	}

	// Initialize model version 1 (genesis model)
	// Create default model version for genesis
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	genesisVersion := types.ModelVersion{
		VersionNumber:       1,
		ModelId:             1,
		ModelHash:           genState.ModelHash,
		IpfsPath:            "v1/",
		Architecture:        "BitNet",
		CompatibilityInfo:   `{"compatible_with": [1], "breaking_changes": false}`,
		Status:              "active",
		ActivationHeight:    0,
		MigrationWindowStart: 0,
		MigrationWindowEnd:   0,
		GovernanceProposalId: 0,
		CreatedAt:           sdkCtx.BlockTime(),
	}

	if err := k.ModelVersions.Set(ctx, 1, genesisVersion); err != nil {
		return err
	}

	// Initialize active model versions
	activeVersions := types.ActiveModelVersions{
		VersionNumbers:        []uint64{1},
		MigrationWindowActive: false,
		PrimaryVersion:        1,
	}

	if err := k.ActiveModelVersions.Set(ctx, activeVersions); err != nil {
		return err
	}

	// Initialize treasury if not exists
	_, treasuryErr := k.Treasury.Get(ctx)
	if treasuryErr != nil {
		// Treasury doesn't exist, create default treasury
		defaultTreasury := types.Treasury{
			Balance:          "0remes",
			BuyBackThreshold: "1000000remes",
			BurnFraction:     "0.5", // 50%
			BuyBackInterval:  100,   // Every 100 blocks
			TotalBurned:      "0remes",
		}
		if setErr := k.Treasury.Set(ctx, defaultTreasury); setErr != nil {
			return setErr
		}
	}

	// Initialize genesis vault with trap job entries
	if len(genState.GenesisVaultList) > 0 {
		if err := k.InitializeGenesisVault(ctx, genState.GenesisVaultList); err != nil {
			return err
		}
		sdkCtx := sdk.UnwrapSDKContext(ctx)
		sdkCtx.Logger().Info("Initialized genesis vault from genesis state", "entries", len(genState.GenesisVaultList))
	}

	return nil
}

// ExportGenesis returns the module's exported genesis.
func (k Keeper) ExportGenesis(ctx context.Context) (*types.GenesisState, error) {
	genesis := types.DefaultGenesis()

	// Export params
	params, err := k.Params.Get(ctx)
	if err != nil && !errors.Is(err, collections.ErrNotFound) {
		return nil, err
	}
	if !errors.Is(err, collections.ErrNotFound) {
		genesis.Params = params
	}

	// Export global model state
	globalModelState, err := k.GlobalModelState.Get(ctx)
	if err != nil && !errors.Is(err, collections.ErrNotFound) {
		return nil, err
	}
	if !errors.Is(err, collections.ErrNotFound) {
		genesis.ModelHash = globalModelState.ModelIpfsHash
		genesis.ModelVersion = globalModelState.ModelVersion
	}

	// Export stored gradients
	err = k.StoredGradients.Walk(ctx, nil, func(key uint64, value types.StoredGradient) (stop bool, err error) {
		genesis.StoredGradientList = append(genesis.StoredGradientList, value)
		return false, nil
	})
	if err != nil {
		return nil, err
	}

	// Export model registry
	err = k.ModelRegistries.Walk(ctx, nil, func(key uint64, value types.ModelRegistry) (stop bool, err error) {
		genesis.ModelRegistryList = append(genesis.ModelRegistryList, value)
		return false, nil
	})
	if err != nil {
		return nil, err
	}

	// Export genesis vault entries
	err = k.GenesisVault.Walk(ctx, nil, func(key uint64, value types.GenesisVaultEntry) (stop bool, err error) {
		genesis.GenesisVaultList = append(genesis.GenesisVaultList, value)
		return false, nil
	})
	if err != nil {
		return nil, err
	}

	return genesis, nil
}
