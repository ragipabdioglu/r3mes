package keeper

import (
	"context"

	"remes/x/remes/types"
)

// CoreKeeperInterface defines the interface for core keeper functionality
type CoreKeeperInterface interface {
	GetAuthority() []byte
	GetParams(ctx context.Context) (types.Params, error)
	SetParams(ctx context.Context, params types.Params) error
	GetBankKeeper() types.BankKeeper
	GetAuthKeeper() types.AuthKeeper
}

// ModelKeeperInterface defines the interface for model management
type ModelKeeperInterface interface {
	RegisterModel(ctx context.Context, model types.ModelRegistry) error
	GetModel(ctx context.Context, modelID uint64) (types.ModelRegistry, error)
	GetGlobalModelState(ctx context.Context) (types.GlobalModelState, error)
	UpdateGlobalModelState(ctx context.Context, state types.GlobalModelState) error
	CreateModelUpgradeProposal(ctx context.Context, proposal types.ModelUpgradeProposal) (uint64, error)
	GetModelUpgradeProposal(ctx context.Context, proposalID uint64) (types.ModelUpgradeProposal, error)
}

// TrainingKeeperInterface defines the interface for training and gradient management
type TrainingKeeperInterface interface {
	SubmitGradient(ctx context.Context, gradient types.StoredGradient) (uint64, error)
	GetGradient(ctx context.Context, gradientID uint64) (types.StoredGradient, error)
	CreateAggregation(ctx context.Context, aggregation types.AggregationRecord) (uint64, error)
	GetAggregation(ctx context.Context, aggregationID uint64) (types.AggregationRecord, error)
	GetAggregationsByDeadline(ctx context.Context, deadline int64) ([]uint64, error)
	RemoveAggregationFromDeadlineIndex(ctx context.Context, deadline int64, aggregationID uint64) error
}

// NodeKeeperInterface defines the interface for node management
type NodeKeeperInterface interface {
	RegisterNode(ctx context.Context, registration types.NodeRegistration) error
	GetNode(ctx context.Context, nodeAddress string) (types.NodeRegistration, error)
	UpdateNodeStatus(ctx context.Context, nodeAddress string, status types.ServingNodeStatus) error
	GetNodeStatus(ctx context.Context, nodeAddress string) (types.ServingNodeStatus, error)
	IsAuthorizedValidator(ctx context.Context, address string) (bool, error)
	AuthorizeValidator(ctx context.Context, address string) error
	IsAuthorizedProposer(ctx context.Context, address string) (bool, error)
	AuthorizeProposer(ctx context.Context, address string) error
	GetMiningContribution(ctx context.Context, minerAddress string) (types.MiningContribution, error)
	UpdateMiningContribution(ctx context.Context, contribution types.MiningContribution) error
}

// DatasetKeeperInterface defines the interface for dataset management
type DatasetKeeperInterface interface {
	CreateDatasetProposal(ctx context.Context, proposal types.DatasetProposal) (uint64, error)
	GetDatasetProposal(ctx context.Context, proposalID uint64) (types.DatasetProposal, error)
	VoteOnDataset(ctx context.Context, vote types.DatasetVote) (uint64, error)
	GetDatasetVote(ctx context.Context, voteID uint64) (types.DatasetVote, error)
	ApproveDataset(ctx context.Context, dataset types.ApprovedDataset) (uint64, error)
	GetApprovedDataset(ctx context.Context, datasetID uint64) (types.ApprovedDataset, error)
}

// EconomicsKeeperInterface defines the interface for economic incentives
type EconomicsKeeperInterface interface {
	GetTreasury(ctx context.Context) (types.Treasury, error)
	UpdateTreasury(ctx context.Context, treasury types.Treasury) error
	CalculateReward(ctx context.Context, minerAddress string, gradientID uint64) (string, error)
	DistributeReward(ctx context.Context, minerAddress string, amount string) error
}

// SecurityKeeperInterface defines the interface for security and validation
type SecurityKeeperInterface interface {
	CreateTrapJob(ctx context.Context, trapJob types.TrapJob) error
	GetTrapJob(ctx context.Context, trapJobID string) (types.TrapJob, error)
	CreateChallenge(ctx context.Context, challenge types.ChallengeRecord) (uint64, error)
	GetChallenge(ctx context.Context, challengeID uint64) (types.ChallengeRecord, error)
	SlashValidator(ctx context.Context, validatorAddress string, penalty string) error
}

// InfraKeeperInterface defines the interface for infrastructure components
type InfraKeeperInterface interface {
	VerifyIPFSContent(ctx context.Context, ipfsHash string) error
	CacheGradient(ctx context.Context, gradientID uint64, data []byte) error
	GetCachedGradient(ctx context.Context, gradientID uint64) ([]byte, error)
}
