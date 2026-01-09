package keeper

import (
	"context"
	"fmt"

	"cosmossdk.io/collections"
	"cosmossdk.io/core/address"
	corestore "cosmossdk.io/core/store"
	"github.com/cosmos/cosmos-sdk/codec"
	sdk "github.com/cosmos/cosmos-sdk/types"
	capabilitykeeper "github.com/cosmos/ibc-go/modules/capability/keeper"

	"remes/x/remes/keeper/core"
	"remes/x/remes/keeper/dataset"
	"remes/x/remes/keeper/economics"
	"remes/x/remes/keeper/infra"
	"remes/x/remes/keeper/model"
	"remes/x/remes/keeper/node"
	"remes/x/remes/keeper/security"
	"remes/x/remes/keeper/training"
	"remes/x/remes/types"
)

// Keeper is the main keeper that orchestrates all domain-specific keepers
// This follows the composition pattern instead of inheritance
type Keeper struct {
	// Core keeper handles basic functionality
	core *core.CoreKeeper

	// Domain-specific keepers
	model     ModelKeeper
	training  TrainingKeeper
	dataset   DatasetKeeper
	node      NodeKeeper
	economics EconomicsKeeper
	security  SecurityKeeper
	infra     InfraKeeper

	// Schema for collections
	Schema collections.Schema
}

// NewKeeper creates a new Keeper with domain-specific sub-keepers
func NewKeeper(
	storeService corestore.KVStoreService,
	cdc codec.Codec,
	addressCodec address.Codec,
	authority []byte,
	bankKeeper types.BankKeeper,
	authKeeper types.AuthKeeper,
	ipfsAPIURL string,
	capabilityKeeper *capabilitykeeper.Keeper,
	scopedKeeper capabilitykeeper.ScopedKeeper,
) (Keeper, error) {
	if _, err := addressCodec.BytesToString(authority); err != nil {
		return Keeper{}, fmt.Errorf("invalid authority address %s: %w", authority, err)
	}

	// SECURITY: Validate production security requirements
	if err := ValidateProductionSecurity(ipfsAPIURL); err != nil {
		return Keeper{}, fmt.Errorf("security validation failed: %w", err)
	}

	// Create core keeper first
	coreKeeper, err := core.NewCoreKeeper(
		storeService,
		cdc,
		addressCodec,
		authority,
		bankKeeper,
		authKeeper,
		capabilityKeeper,
		scopedKeeper,
	)
	if err != nil {
		return Keeper{}, fmt.Errorf("failed to create core keeper: %w", err)
	}

	// Create infrastructure keeper
	infraKeeper, err := infra.NewInfraKeeper(
		storeService,
		cdc,
		ipfsAPIURL,
	)
	if err != nil {
		return Keeper{}, fmt.Errorf("failed to create infra keeper: %w", err)
	}

	// Create domain-specific keepers
	modelKeeper, err := model.NewModelKeeper(storeService, cdc, coreKeeper)
	if err != nil {
		return Keeper{}, fmt.Errorf("failed to create model keeper: %w", err)
	}

	trainingKeeper, err := training.NewTrainingKeeper(storeService, cdc, coreKeeper, infraKeeper)
	if err != nil {
		return Keeper{}, fmt.Errorf("failed to create training keeper: %w", err)
	}

	datasetKeeper, err := dataset.NewDatasetKeeper(storeService, cdc, coreKeeper)
	if err != nil {
		return Keeper{}, fmt.Errorf("failed to create dataset keeper: %w", err)
	}

	nodeKeeper, err := node.NewNodeKeeper(storeService, cdc, coreKeeper, bankKeeper)
	if err != nil {
		return Keeper{}, fmt.Errorf("failed to create node keeper: %w", err)
	}

	economicsKeeper, err := economics.NewEconomicsKeeper(storeService, cdc, coreKeeper, bankKeeper)
	if err != nil {
		return Keeper{}, fmt.Errorf("failed to create economics keeper: %w", err)
	}

	securityKeeper, err := security.NewSecurityKeeper(storeService, cdc, coreKeeper, authKeeper)
	if err != nil {
		return Keeper{}, fmt.Errorf("failed to create security keeper: %w", err)
	}

	// Build schema from core keeper
	schema := coreKeeper.GetSchema()

	return Keeper{
		core:      coreKeeper,
		model:     modelKeeper,
		training:  trainingKeeper,
		dataset:   datasetKeeper,
		node:      nodeKeeper,
		economics: economicsKeeper,
		security:  securityKeeper,
		infra:     infraKeeper,
		Schema:    schema,
	}, nil
}

// GetAuthority returns the module's authority.
func (k Keeper) GetAuthority() []byte {
	return k.core.GetAuthority()
}

// GetParams returns the current module parameters.
func (k Keeper) GetParams(ctx context.Context) (types.Params, error) {
	return k.core.GetParams(ctx)
}

// SetParams sets the module parameters.
func (k Keeper) SetParams(ctx context.Context, params types.Params) error {
	return k.core.SetParams(ctx, params)
}

// GetCoreKeeper returns the core keeper for direct access when needed
func (k Keeper) GetCoreKeeper() *core.CoreKeeper {
	return k.core
}

// GetModelKeeper returns the model keeper
func (k Keeper) GetModelKeeper() ModelKeeper {
	return k.model
}

// GetTrainingKeeper returns the training keeper
func (k Keeper) GetTrainingKeeper() TrainingKeeper {
	return k.training
}

// GetDatasetKeeper returns the dataset keeper
func (k Keeper) GetDatasetKeeper() DatasetKeeper {
	return k.dataset
}

// GetNodeKeeper returns the node keeper
func (k Keeper) GetNodeKeeper() NodeKeeper {
	return k.node
}

// GetEconomicsKeeper returns the economics keeper
func (k Keeper) GetEconomicsKeeper() EconomicsKeeper {
	return k.economics
}

// GetSecurityKeeper returns the security keeper
func (k Keeper) GetSecurityKeeper() SecurityKeeper {
	return k.security
}

// GetInfraKeeper returns the infrastructure keeper
func (k Keeper) GetInfraKeeper() InfraKeeper {
	return k.infra
}

// Legacy methods for backward compatibility
// These delegate to the appropriate domain-specific keepers

// Model-related methods
func (k Keeper) RegisterModel(ctx context.Context, model types.ModelRegistry) error {
	return k.model.RegisterModel(ctx, model)
}

func (k Keeper) GetModel(ctx context.Context, modelID uint64) (types.ModelRegistry, error) {
	return k.model.GetModel(ctx, modelID)
}

func (k Keeper) UpdateGlobalModelState(ctx context.Context, state types.GlobalModelState) error {
	return k.model.UpdateGlobalModelState(ctx, state)
}

// Training-related methods
func (k Keeper) SubmitGradient(ctx context.Context, gradient types.StoredGradient) error {
	return k.training.SubmitGradient(ctx, gradient)
}

func (k Keeper) AggregateGradients(ctx context.Context, gradients []types.StoredGradient) error {
	return k.training.AggregateGradients(ctx, gradients)
}

func (k Keeper) GetTrainingWindow(ctx context.Context, windowID uint64) (types.TrainingWindow, error) {
	return k.training.GetTrainingWindow(ctx, windowID)
}

// Dataset-related methods
func (k Keeper) ProposeDataset(ctx context.Context, proposal types.DatasetProposal) error {
	return k.dataset.ProposeDataset(ctx, proposal)
}

func (k Keeper) VoteOnDataset(ctx context.Context, vote types.DatasetVote) error {
	return k.dataset.VoteOnDataset(ctx, vote)
}

func (k Keeper) GetDatasetProposal(ctx context.Context, proposalID uint64) (types.DatasetProposal, error) {
	return k.dataset.GetDatasetProposal(ctx, proposalID)
}

// Node-related methods
func (k Keeper) RegisterNode(ctx context.Context, node types.NodeRegistration) error {
	return k.node.RegisterNode(ctx, node)
}

func (k Keeper) GetNode(ctx context.Context, address string) (types.NodeRegistration, error) {
	return k.node.GetNode(ctx, address)
}

func (k Keeper) UpdateNodeStatus(ctx context.Context, address string, status types.NodeStatus) error {
	return k.node.UpdateNodeStatus(ctx, address, status)
}

// Economics-related methods
func (k Keeper) CalculateRewards(ctx context.Context, contributions []types.MiningContribution) ([]economics.Reward, error) {
	return k.economics.CalculateRewards(ctx, contributions)
}

func (k Keeper) DistributeRewards(ctx context.Context, rewards []economics.Reward) error {
	return k.economics.DistributeRewards(ctx, rewards)
}

func (k Keeper) GetTreasury(ctx context.Context) (types.Treasury, error) {
	return k.economics.GetTreasury(ctx)
}

// Security-related methods
func (k Keeper) VerifySignature(ctx context.Context, address string, message []byte, signature []byte) error {
	return k.security.VerifySignature(ctx, address, message, signature)
}

func (k Keeper) ValidateNonce(ctx context.Context, address string, nonce uint64) error {
	return k.security.ValidateNonce(ctx, address, nonce)
}

func (k Keeper) DetectFraud(ctx context.Context, submission security.GradientSubmission) (bool, error) {
	return k.security.DetectFraud(ctx, submission)
}

// Infrastructure-related methods
func (k Keeper) VerifyIPFSContent(ctx context.Context, hash string) (bool, error) {
	return k.infra.VerifyIPFSContent(ctx, hash)
}

func (k Keeper) CacheGradient(ctx context.Context, hash string, data []byte) error {
	return k.infra.CacheGradient(ctx, hash, data)
}

func (k Keeper) GetCachedGradient(ctx context.Context, hash string) ([]byte, error) {
	return k.infra.GetCachedGradient(ctx, hash)
}

// ProcessTreasuryBuyBack delegates to economics keeper
func (k Keeper) ProcessTreasuryBuyBack(ctx context.Context) error {
	return k.economics.ProcessTreasuryBuyBack(ctx)
}

// InitGenesis initializes the module's state from a genesis state
func (k Keeper) InitGenesis(ctx context.Context, genState *types.GenesisState) error {
	// Set params
	if err := k.core.SetParams(ctx, genState.Params); err != nil {
		return fmt.Errorf("failed to set params: %w", err)
	}

	// Initialize stored gradients
	for _, gradient := range genState.StoredGradientList {
		if err := k.training.SubmitGradient(ctx, gradient); err != nil {
			return fmt.Errorf("failed to store gradient: %w", err)
		}
	}

	return nil
}

// ExportGenesis exports the module's state to a genesis state
func (k Keeper) ExportGenesis(ctx context.Context) (*types.GenesisState, error) {
	params, err := k.core.GetParams(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get params: %w", err)
	}

	return &types.GenesisState{
		Params:             params,
		StoredGradientList: []types.StoredGradient{},
	}, nil
}

// FinalizeExpiredAggregations finalizes aggregations whose challenge period has expired
func (k Keeper) FinalizeExpiredAggregations(ctx context.Context) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	currentTime := sdkCtx.BlockTime().Unix()

	// Get pending aggregations by deadline
	// Iterate through deadlines that have passed
	iter, err := k.training.(*training.TrainingKeeper).PendingAggregationsByDeadline.Iterate(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to iterate pending aggregations: %w", err)
	}
	defer iter.Close()

	var toFinalize []struct {
		deadline      int64
		aggregationID uint64
	}

	for ; iter.Valid(); iter.Next() {
		deadline, err := iter.Key()
		if err != nil {
			continue
		}

		// Skip if deadline hasn't passed
		if deadline > currentTime {
			continue
		}

		list, err := iter.Value()
		if err != nil {
			continue
		}

		for _, aggID := range list.AggregationIds {
			toFinalize = append(toFinalize, struct {
				deadline      int64
				aggregationID uint64
			}{deadline, aggID})
		}
	}

	// Finalize each aggregation
	for _, item := range toFinalize {
		if err := k.finalizeAggregation(ctx, item.aggregationID); err != nil {
			sdkCtx.Logger().Error("failed to finalize aggregation",
				"aggregation_id", item.aggregationID,
				"error", err)
			continue
		}

		// Remove from pending list
		if err := k.training.(*training.TrainingKeeper).RemovePendingAggregation(ctx, item.deadline, item.aggregationID); err != nil {
			sdkCtx.Logger().Error("failed to remove pending aggregation",
				"aggregation_id", item.aggregationID,
				"error", err)
		}
	}

	return nil
}

// finalizeAggregation finalizes a single aggregation
func (k Keeper) finalizeAggregation(ctx context.Context, aggregationID uint64) error {
	aggregation, err := k.training.(*training.TrainingKeeper).GetAggregation(ctx, aggregationID)
	if err != nil {
		return fmt.Errorf("failed to get aggregation: %w", err)
	}

	// Update status to finalized
	aggregation.Status = "finalized"

	// Store updated aggregation
	if err := k.training.(*training.TrainingKeeper).AggregationRecords.Set(ctx, aggregationID, aggregation); err != nil {
		return fmt.Errorf("failed to update aggregation: %w", err)
	}

	return nil
}
