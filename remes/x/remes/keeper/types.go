package keeper

import (
	"context"

	"remes/x/remes/keeper/economics"
	"remes/x/remes/keeper/security"
	"remes/x/remes/types"
)

// Interface definitions for domain-specific keepers
// These interfaces define the contract that each keeper must implement

// ModelKeeper interface defines model management operations
type ModelKeeper interface {
	RegisterModel(ctx context.Context, model types.ModelRegistry) error
	GetModel(ctx context.Context, modelID uint64) (types.ModelRegistry, error)
	UpdateModel(ctx context.Context, modelID uint64, updates types.ModelRegistry) error
	GetGlobalModelState(ctx context.Context) (types.GlobalModelState, error)
	UpdateGlobalModelState(ctx context.Context, state types.GlobalModelState) error
	CreateModelVersion(ctx context.Context, version types.ModelVersion) error
	GetModelVersion(ctx context.Context, versionNumber uint64) (types.ModelVersion, error)
	ProposeModelUpgrade(ctx context.Context, proposal types.ModelUpgradeProposal) error
	GetModelUpgradeProposal(ctx context.Context, proposalID uint64) (types.ModelUpgradeProposal, error)
	VoteOnModelUpgrade(ctx context.Context, vote types.ModelUpgradeVote) error
	GetActiveModelVersions(ctx context.Context) (types.ActiveModelVersions, error)
	UpdateActiveModelVersions(ctx context.Context, versions types.ActiveModelVersions) error
	ListModels(ctx context.Context) ([]types.ModelRegistry, error)
	GetModelCount(ctx context.Context) (uint64, error)
}

// TrainingKeeper interface defines training and gradient operations
type TrainingKeeper interface {
	SubmitGradient(ctx context.Context, gradient types.StoredGradient) error
	GetGradient(ctx context.Context, gradientID uint64) (types.StoredGradient, error)
	AggregateGradients(ctx context.Context, gradients []types.StoredGradient) error
	GetAggregation(ctx context.Context, aggregationID uint64) (types.AggregationRecord, error)
	CreateTrainingWindow(ctx context.Context, window types.TrainingWindow) error
	GetTrainingWindow(ctx context.Context, windowID uint64) (types.TrainingWindow, error)
	GetMiningContribution(ctx context.Context, minerAddress string) (types.MiningContribution, error)
	AddPendingAggregation(ctx context.Context, deadline int64, aggregationID uint64) error
	GetPendingAggregationsByDeadline(ctx context.Context, deadline int64) ([]uint64, error)
	RemovePendingAggregation(ctx context.Context, deadline int64, aggregationID uint64) error
	RecordConvergenceMetrics(ctx context.Context, metrics types.ConvergenceMetrics) error
	GetConvergenceMetrics(ctx context.Context, trainingRoundID uint64) (types.ConvergenceMetrics, error)
}

// DatasetKeeper interface defines dataset management operations
type DatasetKeeper interface {
	ProposeDataset(ctx context.Context, proposal types.DatasetProposal) error
	GetDatasetProposal(ctx context.Context, proposalID uint64) (types.DatasetProposal, error)
	UpdateDatasetProposal(ctx context.Context, proposalID uint64, updates types.DatasetProposal) error
	VoteOnDataset(ctx context.Context, vote types.DatasetVote) error
	GetDatasetVote(ctx context.Context, voteID uint64) (types.DatasetVote, error)
	ApproveDataset(ctx context.Context, proposalID uint64) error
	GetApprovedDataset(ctx context.Context, datasetID uint64) (types.ApprovedDataset, error)
	ListDatasetProposals(ctx context.Context) ([]types.DatasetProposal, error)
	ListApprovedDatasets(ctx context.Context) ([]types.ApprovedDataset, error)
	GetVotesForProposal(ctx context.Context, proposalID uint64) ([]types.DatasetVote, error)
	GetProposalCount(ctx context.Context) (uint64, error)
	GetApprovedDatasetCount(ctx context.Context) (uint64, error)
}

// NodeKeeper interface defines node management operations
type NodeKeeper interface {
	RegisterNode(ctx context.Context, node types.NodeRegistration) error
	GetNode(ctx context.Context, address string) (types.NodeRegistration, error)
	UpdateNode(ctx context.Context, address string, updates types.NodeRegistration) error
	UpdateNodeStatus(ctx context.Context, address string, status types.NodeStatus) error
	DeregisterNode(ctx context.Context, address string) error
	ListNodes(ctx context.Context) ([]types.NodeRegistration, error)
	UpdateServingNodeStatus(ctx context.Context, address string, status types.ServingNodeStatus) error
	GetServingNodeStatus(ctx context.Context, address string) (types.ServingNodeStatus, error)
	CreateInferenceRequest(ctx context.Context, request types.InferenceRequest) error
	GetInferenceRequest(ctx context.Context, requestID string) (types.InferenceRequest, error)
	AuthorizeValidator(ctx context.Context, address string) error
	IsValidatorAuthorized(ctx context.Context, address string) (bool, error)
	AuthorizeProposer(ctx context.Context, address string) error
	IsProposerAuthorized(ctx context.Context, address string) (bool, error)
	GetNodeCount(ctx context.Context) (int, error)
}

// EconomicsKeeper interface defines economic operations
type EconomicsKeeper interface {
	CalculateRewards(ctx context.Context, contributions []types.MiningContribution) ([]economics.Reward, error)
	DistributeRewards(ctx context.Context, rewards []economics.Reward) error
	GetTreasury(ctx context.Context) (types.Treasury, error)
	UpdateTreasury(ctx context.Context, treasury types.Treasury) error
	ProcessTreasuryBuyBack(ctx context.Context) error
	CalculateStakingRewards(ctx context.Context, stakingInfo []types.StakingInfo) ([]economics.Reward, error)
	SlashValidator(ctx context.Context, validatorAddr string, slashAmount string) error
	GetEconomicParameters(ctx context.Context) (types.EconomicParams, error)
	UpdateEconomicParameters(ctx context.Context, economicParams types.EconomicParams) error
	GetTotalSupply(ctx context.Context, denom string) (string, error)
	GetInflationRate(ctx context.Context) (string, error)
}

// SecurityKeeper interface defines security operations
type SecurityKeeper interface {
	VerifySignature(ctx context.Context, address string, message []byte, signature []byte) error
	ValidateNonce(ctx context.Context, address string, nonce uint64) error
	DetectFraud(ctx context.Context, submission security.GradientSubmission) (bool, error)
	CreateChallenge(ctx context.Context, challenge types.ChallengeRecord) error
	GetChallenge(ctx context.Context, challengeID uint64) (types.ChallengeRecord, error)
	CreateTrapJob(ctx context.Context, trapJob types.TrapJob) error
	GetTrapJob(ctx context.Context, jobID string) (types.TrapJob, error)
	CreateDataAvailabilityChallenge(ctx context.Context, challenge types.DataAvailabilityChallenge) error
	GetDataAvailabilityChallenge(ctx context.Context, challengeID uint64) (types.DataAvailabilityChallenge, error)
	AddGenesisVaultEntry(ctx context.Context, entry types.GenesisVaultEntry) error
	GetGenesisVaultEntry(ctx context.Context, entryID uint64) (types.GenesisVaultEntry, error)
	CreateTaskPool(ctx context.Context, taskPool types.TaskPool) error
	GetTaskPool(ctx context.Context, poolID uint64) (types.TaskPool, error)
	ValidateExecutionEnvironment(ctx context.Context, env types.ExecutionEnvironment) error
	MonitorSecurityMetrics(ctx context.Context) error
}

// InfraKeeper interface defines infrastructure operations
type InfraKeeper interface {
	VerifyIPFSContent(ctx context.Context, hash string) (bool, error)
	CacheGradient(ctx context.Context, hash string, data []byte) error
	GetCachedGradient(ctx context.Context, hash string) ([]byte, error)
}

// Reward represents a reward to be distributed (defined here to avoid circular imports)
type Reward struct {
	Recipient string
	Amount    string
	Reason    string
}

// GradientSubmission represents a gradient submission for fraud detection
type GradientSubmission struct {
	Miner     string
	IPFSHash  string
	Signature []byte
	Nonce     uint64
}
