package keeper

import (
	"fmt"
	"time"

	"cosmossdk.io/collections"
	"cosmossdk.io/core/address"
	corestore "cosmossdk.io/core/store"
	"github.com/cosmos/cosmos-sdk/codec"

	"remes/x/remes/types"
)

type Keeper struct {
	storeService corestore.KVStoreService
	cdc          codec.Codec
	addressCodec address.Codec
	// Address capable of executing a MsgUpdateParams message.
	// Typically, this should be the x/gov module account.
	authority []byte

	// Bank keeper for token minting and distribution
	bankKeeper types.BankKeeper

	// Auth keeper for account and signature verification
	authKeeper types.AuthKeeper

	// IPFS manager for passive content retrieval (validation only)
	ipfsManager *IPFSManager

	// Gradient cache for avoiding repeated IPFS downloads
	gradientCache *GradientCache

	// Debug configuration (loaded from environment variables)
	debugConfig *DebugConfig

	Schema collections.Schema

	// Params stores module parameters
	Params collections.Item[types.Params]

	// GlobalModelState stores the current global model state (singleton)
	GlobalModelState collections.Item[types.GlobalModelState]

	// StoredGradients stores gradient submissions by ID
	StoredGradients collections.Map[uint64, types.StoredGradient]

	// StoredGradientID is a counter for the next stored gradient ID
	StoredGradientID collections.Sequence

	// AggregationRecords stores aggregation records by ID
	AggregationRecords collections.Map[uint64, types.AggregationRecord]

	// AggregationID is a counter for the next aggregation ID
	AggregationID collections.Sequence

	// AggregationCommitments stores aggregation commitments by ID (commit-reveal scheme)
	AggregationCommitments collections.Map[uint64, types.AggregationCommitment]

	// AggregationCommitmentID is a counter for the next commitment ID
	AggregationCommitmentID collections.Sequence

	// MiningContributions stores miner contributions by miner address
	MiningContributions collections.Map[string, types.MiningContribution]

	// ChallengeRecords stores challenge records by ID
	ChallengeRecords collections.Map[uint64, types.ChallengeRecord]

	// ChallengeID is a counter for the next challenge ID
	ChallengeID collections.Sequence

	// DatasetProposals stores dataset proposals by ID
	DatasetProposals collections.Map[uint64, types.DatasetProposal]

	// DatasetProposalID is a counter for the next dataset proposal ID
	DatasetProposalID collections.Sequence

	// DatasetVotes stores individual votes on dataset proposals (by vote_id)
	DatasetVotes collections.Map[uint64, types.DatasetVote]

	// DatasetVoteID is a counter for the next dataset vote ID
	DatasetVoteID collections.Sequence

	// ApprovedDatasets stores approved datasets by ID
	ApprovedDatasets collections.Map[uint64, types.ApprovedDataset]

	// ApprovedDatasetID is a counter for the next approved dataset ID
	ApprovedDatasetID collections.Sequence

	// NodeRegistrations stores node registrations by node address
	NodeRegistrations collections.Map[string, types.NodeRegistration]

	// UsedNonces tracks used nonces per miner (key: "miner_address|nonce", value: true)
	UsedNonces collections.Map[string, bool]

	// NonceWindows tracks nonce windows per miner (key: "nonce_window:miner_address", value: "minNonce|maxNonce")
	NonceWindows collections.Map[string, string]

	// SubmissionHistory tracks submission rate for rate limiting (key: "miner_address|block_height", value: count)
	SubmissionHistory collections.Map[string, uint64]

	// PinningIncentives stores pinning commitments (key: "node_address|ipfs_hash", value: PinningIncentive)
	PinningIncentives collections.Map[string, types.PinningIncentive]

	// DataAvailabilityChallenges stores challenges (key: challenge_id, value: DataAvailabilityChallenge)
	DataAvailabilityChallenges collections.Map[uint64, types.DataAvailabilityChallenge]

	// DataAvailabilityChallengeID is a counter for the next challenge ID
	DataAvailabilityChallengeID collections.Sequence

	// InferenceRequests stores inference requests (key: request_id, value: InferenceRequest)
	InferenceRequests collections.Map[string, types.InferenceRequest]

	// ServingNodeStatuses stores serving node statuses (key: node_address, value: ServingNodeStatus)
	ServingNodeStatuses collections.Map[string, types.ServingNodeStatus]

	// TrapJobs stores trap jobs (key: trap_job_id, value: TrapJob)
	TrapJobs collections.Map[string, types.TrapJob]

	// TrapJobAppeals stores trap job appeals (key: appeal_id, value: TrapJobAppeal)
	TrapJobAppeals collections.Map[string, types.TrapJobAppeal]

	// SlashingAppeals stores slashing appeals (key: appeal_id, value: SlashingAppeal)
	SlashingAppeals collections.Map[string, types.SlashingAppeal]

	// ValidatorVerificationRecords stores validator verification records (key: validator_address, value: ValidatorVerificationRecord)
	ValidatorVerificationRecords collections.Map[string, types.ValidatorVerificationRecord]

	// ProposerCensorshipRecords stores proposer censorship records (key: proposer_address, value: ProposerCensorshipRecord)
	ProposerCensorshipRecords collections.Map[string, types.ProposerCensorshipRecord]

	// MentorRelationships stores mentor relationships (key: "mentor_address|mentee_address", value: MentorRelationship)
	MentorRelationships collections.Map[string, types.MentorRelationship]

	// ExecutionEnvironmentProposals stores execution environment proposals (key: proposal_id, value: ExecutionEnvironmentProposal)
	ExecutionEnvironmentProposals collections.Map[uint64, types.ExecutionEnvironmentProposal]

	// ExecutionEnvironmentProposalID is a counter for the next execution environment proposal ID
	ExecutionEnvironmentProposalID collections.Sequence

	// ApprovedExecutionEnvironments stores approved execution environments (key: environment_id, value: ExecutionEnvironment)
	ApprovedExecutionEnvironments collections.Map[string, types.ExecutionEnvironment]

	// ApprovedExecutionEnvironmentID is a counter for the next approved execution environment ID
	ApprovedExecutionEnvironmentID collections.Sequence

	// ParticipantSyncStates stores participant synchronization states (key: participant_address, value: ParticipantSyncState)
	ParticipantSyncStates collections.Map[string, types.ParticipantSyncState]

	// ModelRegistries stores approved model configurations (key: model_id, value: ModelRegistry)
	ModelRegistries collections.Map[uint64, types.ModelRegistry]

	// ModelID is a counter for the next model ID
	ModelID collections.Sequence

	// Treasury stores the treasury state (singleton)
	Treasury collections.Item[types.Treasury]

	// ModelVersions stores model versions (key: version_number, value: ModelVersion)
	ModelVersions collections.Map[uint64, types.ModelVersion]

	// ModelUpgradeProposals stores model upgrade proposals (key: proposal_id, value: ModelUpgradeProposal)
	ModelUpgradeProposals collections.Map[uint64, types.ModelUpgradeProposal]

	// ModelUpgradeProposalID is a counter for the next proposal ID
	ModelUpgradeProposalID collections.Sequence

	// ModelUpgradeVotes stores individual votes on model upgrade proposals (by vote_id)
	ModelUpgradeVotes collections.Map[uint64, types.ModelUpgradeVote]

	// ModelUpgradeVoteID is a counter for the next model upgrade vote ID
	ModelUpgradeVoteID collections.Sequence

	// ActiveModelVersions stores active model versions (singleton)
	ActiveModelVersions collections.Item[types.ActiveModelVersions]

	// SubnetConfigs stores subnet configurations (key: subnet_id, value: SubnetConfig)
	SubnetConfigs collections.Map[uint64, types.SubnetConfig]

	// ActivationTransmissions stores activation transmissions (key: transmission_id, value: ActivationTransmission)
	ActivationTransmissions collections.Map[uint64, types.ActivationTransmission]

	// ActivationTransmissionID is a counter for the next transmission ID
	ActivationTransmissionID collections.Sequence

	// SubnetTrainingWorkflows stores subnet training workflows (key: workflow_id, value: SubnetTrainingWorkflow)
	SubnetTrainingWorkflows collections.Map[uint64, types.SubnetTrainingWorkflow]

	// SubnetTrainingWorkflowID is a counter for the next workflow ID
	SubnetTrainingWorkflowID collections.Sequence

	// TrainingWindows stores training windows (key: window_id, value: TrainingWindow)
	TrainingWindows collections.Map[uint64, types.TrainingWindow]

	// AsyncGradientSubmissions stores async gradient submissions (key: submission_id, value: AsyncGradientSubmission)
	AsyncGradientSubmissions collections.Map[uint64, types.AsyncGradientSubmission]

	// AsyncGradientSubmissionID is a counter for the next submission ID
	AsyncGradientSubmissionID collections.Sequence

	// LazyAggregations stores lazy aggregations (key: aggregation_id, value: LazyAggregation)
	LazyAggregations collections.Map[uint64, types.LazyAggregation]

	// LazyAggregationID is a counter for the next lazy aggregation ID
	LazyAggregationID collections.Sequence

	// BlockTimestamps stores block timestamps (key: block_height, value: timestamp_unix)
	// Used for calculating average block time
	// Only stores last 100 blocks to limit state growth
	BlockTimestamps collections.Map[int64, int64]

	// TaskPools stores task pools (key: pool_id, value: TaskPool)
	TaskPools collections.Map[uint64, types.TaskPool]

	// TaskPoolID is a counter for the next task pool ID
	TaskPoolID collections.Sequence

	// ConvergenceMetrics stores convergence metrics (key: training_round_id, value: ConvergenceMetrics)
	ConvergenceMetrics collections.Map[uint64, types.ConvergenceMetrics]

	// GenesisVault stores genesis vault entries for trap jobs (key: entry_id, value: GenesisVaultEntry)
	GenesisVault collections.Map[uint64, types.GenesisVaultEntry]

	// GenesisVaultCounter is a counter for the next genesis vault entry ID
	GenesisVaultCounter collections.Sequence

	// PendingAggregationsByDeadline stores pending aggregations indexed by challenge deadline height
	// Key: challenge_deadline_height (int64), Value: AggregationIDList (list of aggregation IDs)
	// This index allows O(1) lookup of aggregations that need to be finalized at a specific block height
	PendingAggregationsByDeadline collections.Map[int64, types.AggregationIDList]

	// AuthorizedValidatorAddresses stores authorized validator addresses (whitelist)
	// Key: node_address (string), Value: true (bool)
	// Only addresses in this whitelist can register as validators
	AuthorizedValidatorAddresses collections.Map[string, bool]

	// AuthorizedProposerAddresses stores authorized proposer addresses (whitelist)
	// Key: node_address (string), Value: true (bool)
	// Only addresses in this whitelist (or validators) can register as proposers
	AuthorizedProposerAddresses collections.Map[string, bool]
}

func NewKeeper(
	storeService corestore.KVStoreService,
	cdc codec.Codec,
	addressCodec address.Codec,
	authority []byte,
	bankKeeper types.BankKeeper,
	authKeeper types.AuthKeeper,
	ipfsAPIURL string,
) Keeper {
	if _, err := addressCodec.BytesToString(authority); err != nil {
		panic(fmt.Sprintf("invalid authority address %s: %s", authority, err))
	}

	// SECURITY: Validate production security requirements
	// This will panic if security requirements are not met in production
	if err := ValidateProductionSecurity(ipfsAPIURL); err != nil {
		panic(fmt.Sprintf("SECURITY VALIDATION FAILED: %v", err))
	}

	// Load debug configuration
	debugConfig, err := LoadDebugConfig()
	if err != nil {
		// Log error but don't panic (debug config is optional)
		// In production, debug config validation happens separately
		debugConfig = &DebugConfig{
			Enabled:    false,
			Level:      DebugLevelStandard,
			Components: make(map[string]bool),
		}
	}

	sb := collections.NewSchemaBuilder(storeService)

	// Create IPFS manager (required in production, validated above)
	var ipfsManager *IPFSManager
	if ipfsAPIURL != "" {
		ipfsManager = NewIPFSManager(ipfsAPIURL)
	} else {
		// This should not happen in production (caught by ValidateProductionSecurity)
		// But we keep this for backward compatibility in test environments
		// where ValidateProductionSecurity might be bypassed
		ipfsManager = nil
	}

	k := Keeper{
		storeService:  storeService,
		cdc:           cdc,
		addressCodec:  addressCodec,
		authority:     authority,
		bankKeeper:    bankKeeper,
		authKeeper:    authKeeper,
		ipfsManager:   ipfsManager,
		gradientCache: NewGradientCache(1 * time.Hour), // 1 hour TTL

		Params: collections.NewItem(sb, types.ParamsKey, "params", codec.CollValue[types.Params](cdc)),

		GlobalModelState: collections.NewItem(sb, types.GlobalModelStateKey, "global_model_state", codec.CollValue[types.GlobalModelState](cdc)),

		StoredGradients: collections.NewMap(sb, types.StoredGradientKey, "stored_gradients", collections.Uint64Key, codec.CollValue[types.StoredGradient](cdc)),

		StoredGradientID: collections.NewSequence(sb, types.StoredGradientIDKey, "stored_gradient_id"),

		AggregationRecords: collections.NewMap(sb, types.AggregationRecordKey, "aggregation_records", collections.Uint64Key, codec.CollValue[types.AggregationRecord](cdc)),

		AggregationID: collections.NewSequence(sb, types.AggregationIDKey, "aggregation_id"),

		AggregationCommitments: collections.NewMap(sb, types.AggregationCommitmentKey, "aggregation_commitments", collections.Uint64Key, codec.CollValue[types.AggregationCommitment](cdc)),

		AggregationCommitmentID: collections.NewSequence(sb, types.AggregationCommitmentIDKey, "aggregation_commitment_id"),

		MiningContributions: collections.NewMap(sb, types.MiningContributionKey, "mining_contributions", collections.StringKey, codec.CollValue[types.MiningContribution](cdc)),

		ChallengeRecords: collections.NewMap(sb, types.ChallengeRecordKey, "challenge_records", collections.Uint64Key, codec.CollValue[types.ChallengeRecord](cdc)),

		ChallengeID: collections.NewSequence(sb, types.ChallengeIDKey, "challenge_id"),

		DatasetProposals: collections.NewMap(sb, types.DatasetProposalKey, "dataset_proposals", collections.Uint64Key, codec.CollValue[types.DatasetProposal](cdc)),

		DatasetProposalID: collections.NewSequence(sb, types.DatasetProposalIDKey, "dataset_proposal_id"),

		DatasetVotes: collections.NewMap(sb, types.DatasetVoteKey, "dataset_votes", collections.Uint64Key, codec.CollValue[types.DatasetVote](cdc)),

		DatasetVoteID: collections.NewSequence(sb, types.DatasetVoteIDKey, "dataset_vote_id"),

		ApprovedDatasets: collections.NewMap(sb, types.ApprovedDatasetKey, "approved_datasets", collections.Uint64Key, codec.CollValue[types.ApprovedDataset](cdc)),

		ApprovedDatasetID: collections.NewSequence(sb, types.ApprovedDatasetIDKey, "approved_dataset_id"),

		NodeRegistrations: collections.NewMap(sb, types.NodeRegistrationKey, "node_registrations", collections.StringKey, codec.CollValue[types.NodeRegistration](cdc)),

		UsedNonces: collections.NewMap(sb, types.UsedNonceKey, "used_nonces", collections.StringKey, collections.BoolValue),

		NonceWindows: collections.NewMap(sb, types.NonceWindowKey, "nonce_windows", collections.StringKey, collections.StringValue),

		SubmissionHistory: collections.NewMap(sb, types.SubmissionHistoryKey, "submission_history", collections.StringKey, collections.Uint64Value),

		PinningIncentives: collections.NewMap(sb, types.PinningIncentiveKey, "pinning_incentives", collections.StringKey, codec.CollValue[types.PinningIncentive](cdc)),

		DataAvailabilityChallenges: collections.NewMap(sb, types.DataAvailabilityChallengeKey, "data_availability_challenges", collections.Uint64Key, codec.CollValue[types.DataAvailabilityChallenge](cdc)),

		DataAvailabilityChallengeID: collections.NewSequence(sb, types.DataAvailabilityChallengeIDKey, "data_availability_challenge_id"),

		InferenceRequests: collections.NewMap(sb, types.InferenceRequestKey, "inference_requests", collections.StringKey, codec.CollValue[types.InferenceRequest](cdc)),

		ServingNodeStatuses: collections.NewMap(sb, types.ServingNodeStatusKey, "serving_node_statuses", collections.StringKey, codec.CollValue[types.ServingNodeStatus](cdc)),

		TrapJobs: collections.NewMap(sb, types.TrapJobKey, "trap_jobs", collections.StringKey, codec.CollValue[types.TrapJob](cdc)),

		TrapJobAppeals: collections.NewMap(sb, types.TrapJobAppealKey, "trap_job_appeals", collections.StringKey, codec.CollValue[types.TrapJobAppeal](cdc)),

		SlashingAppeals: collections.NewMap(sb, types.SlashingAppealKey, "slashing_appeals", collections.StringKey, codec.CollValue[types.SlashingAppeal](cdc)),

		ValidatorVerificationRecords: collections.NewMap(sb, types.ValidatorVerificationRecordKey, "validator_verification_records", collections.StringKey, codec.CollValue[types.ValidatorVerificationRecord](cdc)),

		ProposerCensorshipRecords: collections.NewMap(sb, types.ProposerCensorshipRecordKey, "proposer_censorship_records", collections.StringKey, codec.CollValue[types.ProposerCensorshipRecord](cdc)),

		MentorRelationships: collections.NewMap(sb, types.MentorRelationshipKey, "mentor_relationships", collections.StringKey, codec.CollValue[types.MentorRelationship](cdc)),

		ExecutionEnvironmentProposals: collections.NewMap(sb, types.ExecutionEnvironmentProposalKey, "execution_environment_proposals", collections.Uint64Key, codec.CollValue[types.ExecutionEnvironmentProposal](cdc)),

		ExecutionEnvironmentProposalID: collections.NewSequence(sb, types.ExecutionEnvironmentProposalIDKey, "execution_environment_proposal_id"),

		ApprovedExecutionEnvironments: collections.NewMap(sb, types.ApprovedExecutionEnvironmentKey, "approved_execution_environments", collections.StringKey, codec.CollValue[types.ExecutionEnvironment](cdc)),

		ApprovedExecutionEnvironmentID: collections.NewSequence(sb, types.ApprovedExecutionEnvironmentIDKey, "approved_execution_environment_id"),

		ParticipantSyncStates: collections.NewMap(sb, types.ParticipantSyncStateKey, "participant_sync_states", collections.StringKey, codec.CollValue[types.ParticipantSyncState](cdc)),

		ModelRegistries: collections.NewMap(sb, types.ModelRegistryKey, "model_registries", collections.Uint64Key, codec.CollValue[types.ModelRegistry](cdc)),

		ModelID: collections.NewSequence(sb, types.ModelIDKey, "model_id"),

		Treasury: collections.NewItem(sb, types.TreasuryKey, "treasury", codec.CollValue[types.Treasury](cdc)),

		ModelVersions: collections.NewMap(sb, types.ModelVersionKey, "model_versions", collections.Uint64Key, codec.CollValue[types.ModelVersion](cdc)),

		ModelUpgradeProposals: collections.NewMap(sb, types.ModelUpgradeProposalKey, "model_upgrade_proposals", collections.Uint64Key, codec.CollValue[types.ModelUpgradeProposal](cdc)),

		ModelUpgradeProposalID: collections.NewSequence(sb, types.ModelUpgradeProposalIDKey, "model_upgrade_proposal_id"),

		ModelUpgradeVotes: collections.NewMap(sb, types.ModelUpgradeVoteKey, "model_upgrade_votes", collections.Uint64Key, codec.CollValue[types.ModelUpgradeVote](cdc)),

		ModelUpgradeVoteID: collections.NewSequence(sb, types.ModelUpgradeVoteIDKey, "model_upgrade_vote_id"),

		ActiveModelVersions: collections.NewItem(sb, types.ActiveModelVersionsKey, "active_model_versions", codec.CollValue[types.ActiveModelVersions](cdc)),

		SubnetConfigs: collections.NewMap(sb, types.SubnetConfigKey, "subnet_configs", collections.Uint64Key, codec.CollValue[types.SubnetConfig](cdc)),

		ActivationTransmissions: collections.NewMap(sb, types.ActivationTransmissionKey, "activation_transmissions", collections.Uint64Key, codec.CollValue[types.ActivationTransmission](cdc)),

		ActivationTransmissionID: collections.NewSequence(sb, types.ActivationTransmissionIDKey, "activation_transmission_id"),

		SubnetTrainingWorkflows: collections.NewMap(sb, types.SubnetTrainingWorkflowKey, "subnet_training_workflows", collections.Uint64Key, codec.CollValue[types.SubnetTrainingWorkflow](cdc)),

		SubnetTrainingWorkflowID: collections.NewSequence(sb, types.SubnetTrainingWorkflowIDKey, "subnet_training_workflow_id"),

		TrainingWindows: collections.NewMap(sb, types.TrainingWindowKey, "training_windows", collections.Uint64Key, codec.CollValue[types.TrainingWindow](cdc)),

		AsyncGradientSubmissions: collections.NewMap(sb, types.AsyncGradientSubmissionKey, "async_gradient_submissions", collections.Uint64Key, codec.CollValue[types.AsyncGradientSubmission](cdc)),

		AsyncGradientSubmissionID: collections.NewSequence(sb, types.AsyncGradientSubmissionIDKey, "async_gradient_submission_id"),

		LazyAggregations: collections.NewMap(sb, types.LazyAggregationKey, "lazy_aggregations", collections.Uint64Key, codec.CollValue[types.LazyAggregation](cdc)),

		LazyAggregationID: collections.NewSequence(sb, types.LazyAggregationIDKey, "lazy_aggregation_id"),

		BlockTimestamps: collections.NewMap(sb, types.BlockTimestampsKey, "block_timestamps", collections.Int64Key, collections.Int64Value),

		TaskPools: collections.NewMap(sb, types.TaskPoolKey, "task_pools", collections.Uint64Key, codec.CollValue[types.TaskPool](cdc)),

		TaskPoolID: collections.NewSequence(sb, types.TaskPoolIDKey, "task_pool_id"),

		ConvergenceMetrics: collections.NewMap(sb, types.ConvergenceMetricsKey, "convergence_metrics", collections.Uint64Key, codec.CollValue[types.ConvergenceMetrics](cdc)),

		GenesisVault: collections.NewMap(sb, types.GenesisVaultKey, "genesis_vault", collections.Uint64Key, codec.CollValue[types.GenesisVaultEntry](cdc)),

		GenesisVaultCounter: collections.NewSequence(sb, types.GenesisVaultCounterKey, "genesis_vault_counter"),

		PendingAggregationsByDeadline: collections.NewMap(sb, types.PendingAggregationsByDeadlineKey, "pending_aggregations_by_deadline", collections.Int64Key, codec.CollValue[types.AggregationIDList](cdc)),

		AuthorizedValidatorAddresses: collections.NewMap(sb, types.AuthorizedValidatorAddressKey, "authorized_validator_addresses", collections.StringKey, collections.BoolValue),

		AuthorizedProposerAddresses: collections.NewMap(sb, types.AuthorizedProposerAddressKey, "authorized_proposer_addresses", collections.StringKey, collections.BoolValue),

		debugConfig: debugConfig,
	}

	schema, err := sb.Build()
	if err != nil {
		panic(err)
	}
	k.Schema = schema

	return k
}

// GetAuthority returns the module's authority.
func (k Keeper) GetAuthority() []byte {
	return k.authority
}
