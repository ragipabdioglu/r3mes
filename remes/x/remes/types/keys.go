package types

import "cosmossdk.io/collections"

const (
	// ModuleName defines the module name
	ModuleName = "remes"

	// StoreKey defines the primary module store key
	StoreKey = ModuleName

	// GovModuleName duplicates the gov module's name to avoid a dependency with x/gov.
	// It should be synced with the gov module's name if it is ever changed.
	// See: https://github.com/cosmos/cosmos-sdk/blob/v0.52.0-beta.2/x/gov/types/keys.go#L9
	GovModuleName = "gov"
)

// Key prefixes for state storage
var (
	// ParamsKey is the prefix to retrieve all Params
	ParamsKey = collections.NewPrefix("p_remes")

	// GlobalModelStateKey is the key for the global model state (singleton)
	GlobalModelStateKey = collections.NewPrefix("gms_")

	// StoredGradientKey is the prefix for stored gradients (by ID)
	StoredGradientKey = collections.NewPrefix("sg_")

	// StoredGradientIDKey is the key for the next stored gradient ID (counter)
	StoredGradientIDKey = collections.NewPrefix("sgid_")

	// AggregationRecordKey is the prefix for aggregation records (by ID)
	AggregationRecordKey = collections.NewPrefix("ar_")

	// AggregationIDKey is the key for the next aggregation ID (counter)
	AggregationIDKey = collections.NewPrefix("arid_")

	// AggregationCommitmentKey is the prefix for aggregation commitments (by ID)
	AggregationCommitmentKey = collections.NewPrefix("ac_")

	// AggregationCommitmentIDKey is the key for the next commitment ID (counter)
	AggregationCommitmentIDKey = collections.NewPrefix("acid_")

	// MiningContributionKey is the prefix for mining contributions (by miner address)
	MiningContributionKey = collections.NewPrefix("mc_")

	// ChallengeRecordKey is the prefix for challenge records (by ID)
	ChallengeRecordKey = collections.NewPrefix("cr_")

	// ChallengeIDKey is the key for the next challenge ID (counter)
	ChallengeIDKey = collections.NewPrefix("crid_")

	// DatasetProposalKey is the prefix for dataset proposals (by ID)
	DatasetProposalKey = collections.NewPrefix("dp_")

	// DatasetProposalIDKey is the key for the next dataset proposal ID (counter)
	DatasetProposalIDKey = collections.NewPrefix("dpid_")

	// DatasetVoteKey is the prefix for dataset votes (by vote_id)
	DatasetVoteKey = collections.NewPrefix("dv_")

	// DatasetVoteIDKey is the key for the next dataset vote ID (counter)
	DatasetVoteIDKey = collections.NewPrefix("dvid_")

	// ApprovedDatasetKey is the prefix for approved datasets (by ID)
	ApprovedDatasetKey = collections.NewPrefix("ad_")

	// ApprovedDatasetIDKey is the key for the next approved dataset ID (counter)
	ApprovedDatasetIDKey = collections.NewPrefix("adid_")

	// NodeRegistrationKey is the prefix for node registrations (by node address)
	NodeRegistrationKey = collections.NewPrefix("nr_")

	// UsedNonceKey is the prefix for tracking used nonces (by miner address + nonce)
	UsedNonceKey = collections.NewPrefix("un_")

	// NonceWindowKey is the prefix for tracking nonce windows (by miner address)
	NonceWindowKey = collections.NewPrefix("nw_")

	// SubmissionHistoryKey is the prefix for submission history (by miner_address + block_height)
	// Used for rate limiting: tracks submissions in a sliding window
	SubmissionHistoryKey = collections.NewPrefix("sh_")

	// PinningIncentiveKey is the prefix for pinning incentives (by node_address + ipfs_hash)
	PinningIncentiveKey = collections.NewPrefix("pi_")

	// DataAvailabilityChallengeKey is the prefix for data availability challenges (by challenge ID)
	DataAvailabilityChallengeKey = collections.NewPrefix("dac_")

	// DataAvailabilityChallengeIDKey is the key for the next challenge ID (counter)
	DataAvailabilityChallengeIDKey = collections.NewPrefix("dacid_")

	// InferenceRequestKey is the prefix for inference requests (by request_id)
	InferenceRequestKey = collections.NewPrefix("ir_")

	// ServingNodeStatusKey is the prefix for serving node status (by node_address)
	ServingNodeStatusKey = collections.NewPrefix("sns_")

	// TrapJobKey is the prefix for trap jobs (by trap_job_id)
	TrapJobKey = collections.NewPrefix("tj_")

	// TrapJobAppealKey is the prefix for trap job appeals (by appeal_id)
	TrapJobAppealKey = collections.NewPrefix("tja_")

	// SlashingAppealKey is the prefix for slashing appeals (by appeal_id)
	SlashingAppealKey = collections.NewPrefix("sa_")

	// ValidatorVerificationRecordKey is the prefix for validator verification records (by validator_address)
	ValidatorVerificationRecordKey = collections.NewPrefix("vvr_")

	// ProposerCensorshipRecordKey is the prefix for proposer censorship records (by proposer_address)
	ProposerCensorshipRecordKey = collections.NewPrefix("pcr_")

	// MentorRelationshipKey is the prefix for mentor relationships (by mentor_address + mentee_address)
	MentorRelationshipKey = collections.NewPrefix("mr_")

	// ExecutionEnvironmentProposalKey is the prefix for execution environment proposals (by proposal_id)
	ExecutionEnvironmentProposalKey = collections.NewPrefix("eep_")

	// ExecutionEnvironmentProposalIDKey is the key for the next execution environment proposal ID (counter)
	ExecutionEnvironmentProposalIDKey = collections.NewPrefix("eepid_")

	// ApprovedExecutionEnvironmentKey is the prefix for approved execution environments (by environment_id)
	ApprovedExecutionEnvironmentKey = collections.NewPrefix("aee_")

	// ApprovedExecutionEnvironmentIDKey is the key for the next approved execution environment ID (counter)
	ApprovedExecutionEnvironmentIDKey = collections.NewPrefix("aeeid_")

	// ParticipantSyncStateKey is the prefix for participant sync states (by participant_address)
	ParticipantSyncStateKey = collections.NewPrefix("pss_")

	// ModelRegistryKey is the prefix for model registries (by model_id)
	ModelRegistryKey = collections.NewPrefix("modreg_")

	// ModelIDKey is the key for the next model ID (counter)
	ModelIDKey = collections.NewPrefix("modid_")

	// TreasuryKey is the key for the treasury (singleton)
	TreasuryKey = collections.NewPrefix("treasury_")

	// ModelVersionKey is the prefix for model versions (by version_number)
	ModelVersionKey = collections.NewPrefix("modver_")

	// ModelUpgradeProposalKey is the prefix for model upgrade proposals (by proposal_id)
	ModelUpgradeProposalKey = collections.NewPrefix("modup_")

	// ModelUpgradeProposalIDKey is the key for the next proposal ID (counter)
	ModelUpgradeProposalIDKey = collections.NewPrefix("modupid_")

	// ModelUpgradeVoteKey is the prefix for model upgrade votes (by vote_id)
	ModelUpgradeVoteKey = collections.NewPrefix("modupvote_")

	// ModelUpgradeVoteIDKey is the key for the next model upgrade vote ID (counter)
	ModelUpgradeVoteIDKey = collections.NewPrefix("modupvoteid_")

	// ActiveModelVersionsKey is the key for active model versions (singleton)
	ActiveModelVersionsKey = collections.NewPrefix("amv_")

	// SubnetConfigKey is the prefix for subnet configurations (by subnet_id)
	SubnetConfigKey = collections.NewPrefix("subnet_")

	// ActivationTransmissionKey is the prefix for activation transmissions (by transmission_id)
	ActivationTransmissionKey = collections.NewPrefix("actrans_")

	// ActivationTransmissionIDKey is the key for the next transmission ID (counter)
	ActivationTransmissionIDKey = collections.NewPrefix("actransid_")

	// SubnetTrainingWorkflowKey is the prefix for subnet training workflows (by workflow_id)
	SubnetTrainingWorkflowKey = collections.NewPrefix("stworkflow_")

	// SubnetTrainingWorkflowIDKey is the key for the next workflow ID (counter)
	SubnetTrainingWorkflowIDKey = collections.NewPrefix("stworkflowid_")

	// TrainingWindowKey is the prefix for training windows (by window_id)
	TrainingWindowKey = collections.NewPrefix("twindow_")

	// AsyncGradientSubmissionKey is the prefix for async gradient submissions (by submission_id)
	AsyncGradientSubmissionKey = collections.NewPrefix("agsub_")

	// AsyncGradientSubmissionIDKey is the key for the next submission ID (counter)
	AsyncGradientSubmissionIDKey = collections.NewPrefix("agsubid_")

	// LazyAggregationKey is the prefix for lazy aggregations (by aggregation_id)
	LazyAggregationKey = collections.NewPrefix("lazyagg_")

	// LazyAggregationIDKey is the key for the next lazy aggregation ID (counter)
	LazyAggregationIDKey = collections.NewPrefix("lazyaggid_")

	// PendingAggregationsByDeadlineKey is the prefix for pending aggregations index by challenge deadline height
	// Key: challenge_deadline_height (int64), Value: []uint64 (aggregation IDs)
	PendingAggregationsByDeadlineKey = collections.NewPrefix("pagg_deadline_")

	// BlockTimestampsKey is the prefix for block timestamps (by block height)
	// Used for calculating average block time
	BlockTimestampsKey = collections.NewPrefix("bt_")

	// TaskPoolKey is the prefix for task pools (by pool_id)
	TaskPoolKey = collections.NewPrefix("tp_")

	// ConvergenceMetricsKey is the prefix for convergence metrics (by training_round_id)
	ConvergenceMetricsKey = collections.NewPrefix("cm_")

	// TaskPoolIDKey is the key for the next task pool ID (counter)
	TaskPoolIDKey = collections.NewPrefix("tpid_")

	// GenesisVaultKey is the prefix for genesis vault entries (by entry_id)
	GenesisVaultKey = collections.NewPrefix("gv_")

	// GenesisVaultCounterKey is the key for the next genesis vault entry ID (counter)
	GenesisVaultCounterKey = collections.NewPrefix("gvid_")

	// AuthorizedValidatorAddressKey is the prefix for authorized validator addresses (whitelist)
	AuthorizedValidatorAddressKey = collections.NewPrefix("ava_")

	// AuthorizedProposerAddressKey is the prefix for authorized proposer addresses (whitelist)
	AuthorizedProposerAddressKey = collections.NewPrefix("apa_")
)
