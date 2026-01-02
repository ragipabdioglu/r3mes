package model

import (
	"context"
	"fmt"

	"cosmossdk.io/collections"
	corestore "cosmossdk.io/core/store"
	"github.com/cosmos/cosmos-sdk/codec"

	"remes/x/remes/keeper/core"
	"remes/x/remes/types"
)

// ModelKeeper handles model management functionality
type ModelKeeper struct {
	core *core.CoreKeeper

	// Model-related collections
	GlobalModelState       collections.Item[types.GlobalModelState]
	ModelRegistries        collections.Map[uint64, types.ModelRegistry]
	ModelID                collections.Sequence
	ModelVersions          collections.Map[uint64, types.ModelVersion]
	ModelUpgradeProposals  collections.Map[uint64, types.ModelUpgradeProposal]
	ModelUpgradeProposalID collections.Sequence
	ModelUpgradeVotes      collections.Map[uint64, types.ModelUpgradeVote]
	ModelUpgradeVoteID     collections.Sequence
	ActiveModelVersions    collections.Item[types.ActiveModelVersions]
}

// NewModelKeeper creates a new model keeper
func NewModelKeeper(
	storeService corestore.KVStoreService,
	cdc codec.Codec,
	coreKeeper *core.CoreKeeper,
) (*ModelKeeper, error) {
	sb := collections.NewSchemaBuilder(storeService)

	k := &ModelKeeper{
		core: coreKeeper,

		GlobalModelState:       collections.NewItem(sb, types.GlobalModelStateKey, "global_model_state", codec.CollValue[types.GlobalModelState](cdc)),
		ModelRegistries:        collections.NewMap(sb, types.ModelRegistryKey, "model_registries", collections.Uint64Key, codec.CollValue[types.ModelRegistry](cdc)),
		ModelID:                collections.NewSequence(sb, types.ModelIDKey, "model_id"),
		ModelVersions:          collections.NewMap(sb, types.ModelVersionKey, "model_versions", collections.Uint64Key, codec.CollValue[types.ModelVersion](cdc)),
		ModelUpgradeProposals:  collections.NewMap(sb, types.ModelUpgradeProposalKey, "model_upgrade_proposals", collections.Uint64Key, codec.CollValue[types.ModelUpgradeProposal](cdc)),
		ModelUpgradeProposalID: collections.NewSequence(sb, types.ModelUpgradeProposalIDKey, "model_upgrade_proposal_id"),
		ModelUpgradeVotes:      collections.NewMap(sb, types.ModelUpgradeVoteKey, "model_upgrade_votes", collections.Uint64Key, codec.CollValue[types.ModelUpgradeVote](cdc)),
		ModelUpgradeVoteID:     collections.NewSequence(sb, types.ModelUpgradeVoteIDKey, "model_upgrade_vote_id"),
		ActiveModelVersions:    collections.NewItem(sb, types.ActiveModelVersionsKey, "active_model_versions", codec.CollValue[types.ActiveModelVersions](cdc)),
	}

	// Build schema (not used directly but validates collections)
	_, err := sb.Build()
	if err != nil {
		return nil, fmt.Errorf("failed to build model keeper schema: %w", err)
	}

	return k, nil
}

// RegisterModel registers a new model
func (k *ModelKeeper) RegisterModel(ctx context.Context, model types.ModelRegistry) error {
	// Generate new model ID
	modelID, err := k.ModelID.Next(ctx)
	if err != nil {
		return fmt.Errorf("failed to generate model ID: %w", err)
	}

	// Set model ID
	model.ModelId = modelID

	// Store model
	if err := k.ModelRegistries.Set(ctx, modelID, model); err != nil {
		return fmt.Errorf("failed to store model: %w", err)
	}

	return nil
}

// GetModel retrieves a model by ID
func (k *ModelKeeper) GetModel(ctx context.Context, modelID uint64) (types.ModelRegistry, error) {
	model, err := k.ModelRegistries.Get(ctx, modelID)
	if err != nil {
		return types.ModelRegistry{}, fmt.Errorf("failed to get model %d: %w", modelID, err)
	}
	return model, nil
}

// UpdateModel updates an existing model
func (k *ModelKeeper) UpdateModel(ctx context.Context, modelID uint64, updates types.ModelRegistry) error {
	// Verify model exists
	_, err := k.GetModel(ctx, modelID)
	if err != nil {
		return err
	}

	// Ensure ID consistency
	updates.ModelId = modelID

	// Update model
	if err := k.ModelRegistries.Set(ctx, modelID, updates); err != nil {
		return fmt.Errorf("failed to update model %d: %w", modelID, err)
	}

	return nil
}

// GetGlobalModelState retrieves the global model state
func (k *ModelKeeper) GetGlobalModelState(ctx context.Context) (types.GlobalModelState, error) {
	state, err := k.GlobalModelState.Get(ctx)
	if err != nil {
		return types.GlobalModelState{}, fmt.Errorf("failed to get global model state: %w", err)
	}
	return state, nil
}

// UpdateGlobalModelState updates the global model state
func (k *ModelKeeper) UpdateGlobalModelState(ctx context.Context, state types.GlobalModelState) error {
	if err := k.GlobalModelState.Set(ctx, state); err != nil {
		return fmt.Errorf("failed to update global model state: %w", err)
	}
	return nil
}

// CreateModelVersion creates a new model version
func (k *ModelKeeper) CreateModelVersion(ctx context.Context, version types.ModelVersion) error {
	if err := k.ModelVersions.Set(ctx, version.VersionNumber, version); err != nil {
		return fmt.Errorf("failed to create model version: %w", err)
	}
	return nil
}

// GetModelVersion retrieves a model version
func (k *ModelKeeper) GetModelVersion(ctx context.Context, versionNumber uint64) (types.ModelVersion, error) {
	version, err := k.ModelVersions.Get(ctx, versionNumber)
	if err != nil {
		return types.ModelVersion{}, fmt.Errorf("failed to get model version %d: %w", versionNumber, err)
	}
	return version, nil
}

// ProposeModelUpgrade creates a model upgrade proposal
func (k *ModelKeeper) ProposeModelUpgrade(ctx context.Context, proposal types.ModelUpgradeProposal) error {
	// Generate new proposal ID
	proposalID, err := k.ModelUpgradeProposalID.Next(ctx)
	if err != nil {
		return fmt.Errorf("failed to generate proposal ID: %w", err)
	}

	// Set proposal ID
	proposal.ProposalId = proposalID

	// Store proposal
	if err := k.ModelUpgradeProposals.Set(ctx, proposalID, proposal); err != nil {
		return fmt.Errorf("failed to store model upgrade proposal: %w", err)
	}

	return nil
}

// GetModelUpgradeProposal retrieves a model upgrade proposal
func (k *ModelKeeper) GetModelUpgradeProposal(ctx context.Context, proposalID uint64) (types.ModelUpgradeProposal, error) {
	proposal, err := k.ModelUpgradeProposals.Get(ctx, proposalID)
	if err != nil {
		return types.ModelUpgradeProposal{}, fmt.Errorf("failed to get model upgrade proposal %d: %w", proposalID, err)
	}
	return proposal, nil
}

// VoteOnModelUpgrade records a vote on a model upgrade proposal
func (k *ModelKeeper) VoteOnModelUpgrade(ctx context.Context, vote types.ModelUpgradeVote) error {
	// Generate new vote ID
	voteID, err := k.ModelUpgradeVoteID.Next(ctx)
	if err != nil {
		return fmt.Errorf("failed to generate vote ID: %w", err)
	}

	// Set vote ID
	vote.VoteId = voteID

	// Store vote
	if err := k.ModelUpgradeVotes.Set(ctx, voteID, vote); err != nil {
		return fmt.Errorf("failed to store model upgrade vote: %w", err)
	}

	return nil
}

// GetActiveModelVersions retrieves active model versions
func (k *ModelKeeper) GetActiveModelVersions(ctx context.Context) (types.ActiveModelVersions, error) {
	versions, err := k.ActiveModelVersions.Get(ctx)
	if err != nil {
		return types.ActiveModelVersions{}, fmt.Errorf("failed to get active model versions: %w", err)
	}
	return versions, nil
}

// UpdateActiveModelVersions updates active model versions
func (k *ModelKeeper) UpdateActiveModelVersions(ctx context.Context, versions types.ActiveModelVersions) error {
	if err := k.ActiveModelVersions.Set(ctx, versions); err != nil {
		return fmt.Errorf("failed to update active model versions: %w", err)
	}
	return nil
}

// ListModels returns all registered models
func (k *ModelKeeper) ListModels(ctx context.Context) ([]types.ModelRegistry, error) {
	var models []types.ModelRegistry

	err := k.ModelRegistries.Walk(ctx, nil, func(key uint64, value types.ModelRegistry) (stop bool, err error) {
		models = append(models, value)
		return false, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to list models: %w", err)
	}

	return models, nil
}

// GetModelCount returns the total number of registered models
func (k *ModelKeeper) GetModelCount(ctx context.Context) (uint64, error) {
	return k.ModelID.Peek(ctx)
}
