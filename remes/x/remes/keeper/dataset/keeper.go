package dataset

import (
	"context"
	"fmt"

	"cosmossdk.io/collections"
	corestore "cosmossdk.io/core/store"
	"github.com/cosmos/cosmos-sdk/codec"

	"remes/x/remes/keeper/core"
	"remes/x/remes/types"
)

// DatasetKeeper handles dataset management functionality
type DatasetKeeper struct {
	core *core.CoreKeeper

	// Dataset-related collections
	DatasetProposals  collections.Map[uint64, types.DatasetProposal]
	DatasetProposalID collections.Sequence
	DatasetVotes      collections.Map[uint64, types.DatasetVote]
	DatasetVoteID     collections.Sequence
	ApprovedDatasets  collections.Map[uint64, types.ApprovedDataset]
	ApprovedDatasetID collections.Sequence
}

// NewDatasetKeeper creates a new dataset keeper
func NewDatasetKeeper(
	storeService corestore.KVStoreService,
	cdc codec.Codec,
	coreKeeper *core.CoreKeeper,
) (*DatasetKeeper, error) {
	sb := collections.NewSchemaBuilder(storeService)

	k := &DatasetKeeper{
		core: coreKeeper,

		DatasetProposals:  collections.NewMap(sb, types.DatasetProposalKey, "dataset_proposals", collections.Uint64Key, codec.CollValue[types.DatasetProposal](cdc)),
		DatasetProposalID: collections.NewSequence(sb, types.DatasetProposalIDKey, "dataset_proposal_id"),
		DatasetVotes:      collections.NewMap(sb, types.DatasetVoteKey, "dataset_votes", collections.Uint64Key, codec.CollValue[types.DatasetVote](cdc)),
		DatasetVoteID:     collections.NewSequence(sb, types.DatasetVoteIDKey, "dataset_vote_id"),
		ApprovedDatasets:  collections.NewMap(sb, types.ApprovedDatasetKey, "approved_datasets", collections.Uint64Key, codec.CollValue[types.ApprovedDataset](cdc)),
		ApprovedDatasetID: collections.NewSequence(sb, types.ApprovedDatasetIDKey, "approved_dataset_id"),
	}

	// Build schema (not used directly but validates collections)
	_, err := sb.Build()
	if err != nil {
		return nil, fmt.Errorf("failed to build dataset keeper schema: %w", err)
	}

	return k, nil
}

// ProposeDataset creates a new dataset proposal
func (k *DatasetKeeper) ProposeDataset(ctx context.Context, proposal types.DatasetProposal) error {
	// Generate new proposal ID
	proposalID, err := k.DatasetProposalID.Next(ctx)
	if err != nil {
		return fmt.Errorf("failed to generate proposal ID: %w", err)
	}

	// Set proposal ID
	proposal.ProposalId = proposalID

	// Store proposal
	if err := k.DatasetProposals.Set(ctx, proposalID, proposal); err != nil {
		return fmt.Errorf("failed to store dataset proposal: %w", err)
	}

	return nil
}

// GetDatasetProposal retrieves a dataset proposal by ID
func (k *DatasetKeeper) GetDatasetProposal(ctx context.Context, proposalID uint64) (types.DatasetProposal, error) {
	proposal, err := k.DatasetProposals.Get(ctx, proposalID)
	if err != nil {
		return types.DatasetProposal{}, fmt.Errorf("failed to get dataset proposal %d: %w", proposalID, err)
	}
	return proposal, nil
}

// UpdateDatasetProposal updates an existing dataset proposal
func (k *DatasetKeeper) UpdateDatasetProposal(ctx context.Context, proposalID uint64, updates types.DatasetProposal) error {
	// Verify proposal exists
	_, err := k.GetDatasetProposal(ctx, proposalID)
	if err != nil {
		return err
	}

	// Ensure ID consistency
	updates.ProposalId = proposalID

	// Update proposal
	if err := k.DatasetProposals.Set(ctx, proposalID, updates); err != nil {
		return fmt.Errorf("failed to update dataset proposal %d: %w", proposalID, err)
	}

	return nil
}

// VoteOnDataset records a vote on a dataset proposal
func (k *DatasetKeeper) VoteOnDataset(ctx context.Context, vote types.DatasetVote) error {
	// Verify proposal exists
	_, err := k.GetDatasetProposal(ctx, vote.ProposalId)
	if err != nil {
		return fmt.Errorf("proposal does not exist: %w", err)
	}

	// Generate new vote ID
	voteID, err := k.DatasetVoteID.Next(ctx)
	if err != nil {
		return fmt.Errorf("failed to generate vote ID: %w", err)
	}

	// Set vote ID
	vote.VoteId = voteID

	// Store vote
	if err := k.DatasetVotes.Set(ctx, voteID, vote); err != nil {
		return fmt.Errorf("failed to store dataset vote: %w", err)
	}

	return nil
}

// GetDatasetVote retrieves a dataset vote by ID
func (k *DatasetKeeper) GetDatasetVote(ctx context.Context, voteID uint64) (types.DatasetVote, error) {
	vote, err := k.DatasetVotes.Get(ctx, voteID)
	if err != nil {
		return types.DatasetVote{}, fmt.Errorf("failed to get dataset vote %d: %w", voteID, err)
	}
	return vote, nil
}

// ApproveDataset approves a dataset proposal
func (k *DatasetKeeper) ApproveDataset(ctx context.Context, proposalID uint64) error {
	// Get the proposal
	proposal, err := k.GetDatasetProposal(ctx, proposalID)
	if err != nil {
		return err
	}

	// Generate new approved dataset ID
	approvedID, err := k.ApprovedDatasetID.Next(ctx)
	if err != nil {
		return fmt.Errorf("failed to generate approved dataset ID: %w", err)
	}

	// Create approved dataset from proposal
	approvedDataset := types.ApprovedDataset{
		DatasetId:       approvedID,
		DatasetIpfsHash: proposal.DatasetIpfsHash,
		Metadata:        proposal.Metadata,
		Proposer:        proposal.Proposer,
		Status:          "approved",
		// Add other fields as needed
	}

	// Store approved dataset
	if err := k.ApprovedDatasets.Set(ctx, approvedID, approvedDataset); err != nil {
		return fmt.Errorf("failed to store approved dataset: %w", err)
	}

	// Update proposal status
	proposal.Status = "approved"
	if err := k.DatasetProposals.Set(ctx, proposalID, proposal); err != nil {
		return fmt.Errorf("failed to update proposal status: %w", err)
	}

	return nil
}

// GetApprovedDataset retrieves an approved dataset by ID
func (k *DatasetKeeper) GetApprovedDataset(ctx context.Context, datasetID uint64) (types.ApprovedDataset, error) {
	dataset, err := k.ApprovedDatasets.Get(ctx, datasetID)
	if err != nil {
		return types.ApprovedDataset{}, fmt.Errorf("failed to get approved dataset %d: %w", datasetID, err)
	}
	return dataset, nil
}

// ListDatasetProposals returns all dataset proposals
func (k *DatasetKeeper) ListDatasetProposals(ctx context.Context) ([]types.DatasetProposal, error) {
	var proposals []types.DatasetProposal

	err := k.DatasetProposals.Walk(ctx, nil, func(key uint64, value types.DatasetProposal) (stop bool, err error) {
		proposals = append(proposals, value)
		return false, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to list dataset proposals: %w", err)
	}

	return proposals, nil
}

// ListApprovedDatasets returns all approved datasets
func (k *DatasetKeeper) ListApprovedDatasets(ctx context.Context) ([]types.ApprovedDataset, error) {
	var datasets []types.ApprovedDataset

	err := k.ApprovedDatasets.Walk(ctx, nil, func(key uint64, value types.ApprovedDataset) (stop bool, err error) {
		datasets = append(datasets, value)
		return false, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to list approved datasets: %w", err)
	}

	return datasets, nil
}

// GetVotesForProposal returns all votes for a specific proposal
func (k *DatasetKeeper) GetVotesForProposal(ctx context.Context, proposalID uint64) ([]types.DatasetVote, error) {
	var votes []types.DatasetVote

	err := k.DatasetVotes.Walk(ctx, nil, func(key uint64, value types.DatasetVote) (stop bool, err error) {
		if value.ProposalId == proposalID {
			votes = append(votes, value)
		}
		return false, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to get votes for proposal %d: %w", proposalID, err)
	}

	return votes, nil
}

// GetProposalCount returns the total number of dataset proposals
func (k *DatasetKeeper) GetProposalCount(ctx context.Context) (uint64, error) {
	return k.DatasetProposalID.Peek(ctx)
}

// GetApprovedDatasetCount returns the total number of approved datasets
func (k *DatasetKeeper) GetApprovedDatasetCount(ctx context.Context) (uint64, error) {
	return k.ApprovedDatasetID.Peek(ctx)
}
