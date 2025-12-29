package types

import (
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
)

// Event types for the remes module
const (
	EventTypeGradientSubmitted = "gradient_submitted"
	EventTypeAggregationSubmitted = "aggregation_submitted"
	EventTypeAggregationChallenged = "aggregation_challenged"
	EventTypeAggregationFinalized = "aggregation_finalized"
	EventTypeRegisterModel = "register_model"
	EventTypeActivateModel = "activate_model"
	EventTypeTreasuryBuyBack = "treasury_buy_back"
	EventTypeEpochUpdate = "epoch_update"
	EventTypeCreateSubnet = "create_subnet"
	EventTypeSubmitSubnetActivation = "submit_subnet_activation"
	EventTypeAssignMinerToSubnet = "assign_miner_to_subnet"
	EventTypeCreateTrainingWindow = "create_training_window"
	EventTypeSubmitAsyncGradient = "submit_async_gradient"
	EventTypeSubmitLazyAggregation = "submit_lazy_aggregation"
	EventTypeWindowBoundary = "window_boundary"
	EventTypeClaimTask = "claim_task"
	EventTypeCompleteTask = "complete_task"
	EventTypeCommitAggregation = "commit_aggregation"
	EventTypeRevealAggregation = "reveal_aggregation"
	EventTypeModelVersionSync = "model_version_sync"
	EventTypeModelVersionActivated = "model_version_activated"
	EventTypePartitionDetected = "partition_detected"
	EventTypePartitionRecovered = "partition_recovered"
	EventTypeMinerInferenceReward = "miner_inference_reward"
)

// Event attribute keys
const (
	AttributeKeyMiner           = "miner"
	AttributeKeyGradientID      = "gradient_id"
	AttributeKeyIPFSHash        = "ipfs_hash"
	AttributeKeyModelVersion    = "model_version"
	AttributeKeyTrainingRoundID = "training_round_id"
	AttributeKeyAggregationID   = "aggregation_id"
	AttributeKeyProposer        = "proposer"
	AttributeKeyChallengeID     = "challenge_id"
	AttributeKeyChallenger      = "challenger"
	
	// Dataset governance events
	EventTypeProposeDataset     = "propose_dataset"
	EventTypeVoteDataset        = "vote_dataset"
	EventTypeMarkDatasetOfficial = "mark_dataset_official"
	EventTypeRemoveDataset       = "remove_dataset"
	AttributeKeyProposalID       = "proposal_id"
	AttributeKeyDatasetID         = "dataset_id"
	AttributeKeyDatasetIPFSHash  = "dataset_ipfs_hash"
	AttributeKeyVoter             = "voter"
	AttributeKeyVoteOption       = "vote_option"
	AttributeKeyAuthority         = "authority"
	AttributeKeyReason            = "reason"
	
	// Node registration events
	EventTypeRegisterNode         = "register_node"
	EventTypeUpdateNodeRegistration = "update_node_registration"
	EventTypeSubmitResourceUsage  = "submit_resource_usage"
	EventTypeResourceViolationWarning = "resource_violation_warning"
	EventTypeResourceViolationSlash = "resource_violation_slash"
	EventTypeResourceViolationFullSlash = "resource_violation_full_slash"
	EventTypeNodeSlash                  = "node_slash"
	EventTypeMinerSlash                 = "miner_slash"
	EventTypeCommitPinning              = "commit_pinning"
	EventTypeChallengePinning           = "challenge_pinning"
	EventTypeRespondToChallenge         = "respond_to_challenge"
	EventTypeSubmitCPUVerification      = "submit_cpu_verification"
	EventTypeResolveChallenge           = "resolve_challenge"
	EventTypeSubmitRandomVerifierResult = "submit_random_verifier_result"
	EventTypeRequestInference           = "request_inference"
	EventTypeSubmitInferenceResult      = "submit_inference_result"
	EventTypeUpdateServingNodeStatus    = "update_serving_node_status"
	EventTypeCreateTrapJob              = "create_trap_job"
	EventTypeSubmitTrapJobResult        = "submit_trap_job_result"
	EventTypeAppealTrapJobSlashing      = "appeal_trap_job_slashing"
	EventTypeReportLazyValidation       = "report_lazy_validation"
	EventTypeReportFalseVerdict         = "report_false_verdict"
	EventTypeReportProposerCensorship   = "report_proposer_censorship"
	EventTypeAppealSlashing             = "appeal_slashing"
	EventTypeRegisterMentorRelationship = "register_mentor_relationship"
	EventTypeFraudDetectionBounty        = "fraud_detection_bounty"
	EventTypeContinuousValidatorIncentive = "continuous_validator_incentive"
	AttributeKeyValidator               = "validator"
	AttributeKeyFraudDetected           = "fraud_detected"
	AttributeKeyRequestID               = "request_id"
	AttributeKeyRequester               = "requester"
	AttributeKeyServingNode             = "serving_node"
	AttributeKeyResultIPFSHash          = "result_ipfs_hash"
	AttributeKeyLatencyMs               = "latency_ms"
	AttributeKeyIsAvailable             = "is_available"
	AttributeKeyTrapJobID               = "trap_job_id"
	AttributeKeyTargetMiner              = "target_miner"
	AttributeKeyCreator                  = "creator"
	AttributeKeyAppealID                  = "appeal_id"
	AttributeKeyAppellant                = "appellant"
	AttributeKeyReporter                 = "reporter"
	AttributeKeyMentor                   = "mentor"
	AttributeKeyMentee                   = "mentee"
	AttributeKeyDetector                 = "detector"
	AttributeKeyBountyAmount             = "bounty_amount"
	AttributeKeyRewardAmount             = "reward_amount"
	AttributeKeyNodeAddress             = "node_address"
	AttributeKeyNodeType                = "node_type"
	AttributeKeyRole                    = "role"
	AttributeKeyIsValid                 = "is_valid"
	AttributeKeySlashAmount             = "slash_amount"
	AttributeKeySlashFraction           = "slash_fraction"
	AttributeKeyStake                   = "stake"
	AttributeKeyModelID                 = "model_id"
	AttributeKeyModelType               = "model_type"
	AttributeKeyBurnAmount              = "burn_amount"
	AttributeKeyTotalBurned             = "total_burned"
	AttributeKeyTreasuryBalance         = "treasury_balance"
	AttributeKeySubnetID                = "subnet_id"
	AttributeKeyWindowID                 = "window_id"
	AttributeKeyLayerRange               = "layer_range"
	AttributeKeyNextSubnetID             = "next_subnet_id"
	AttributeKeyActivationHash           = "activation_hash"
	AttributeKeyTransmissionID           = "transmission_id"
	AttributeKeyStartHeight              = "start_height"
	AttributeKeyEndHeight                = "end_height"
	AttributeKeyAggregatorNode           = "aggregator_node"
	AttributeKeySubmissionID             = "submission_id"
	AttributeKeyGradientCount            = "gradient_count"
	AttributeKeyModelVersionID          = "model_version_id"
	AttributeKeyModelIPFSHash            = "model_ipfs_hash"
	AttributeKeySyncRequired            = "sync_required"
	AttributeKeyParticipantAddress      = "participant_address"
	AttributeKeyPartitionHeight         = "partition_height"
	AttributeKeyPartitionDuration       = "partition_duration"
	AttributeKeyRecoveryType            = "recovery_type"
	AttributeKeySyncLagBlocks           = "sync_lag_blocks"
	AttributeKeyParticipantType         = "participant_type"
	AttributeKeyCommitmentID            = "commitment_id"
	AttributeKeyPoolID                  = "pool_id"
	AttributeKeyChunkID                 = "chunk_id"
	AttributeKeyGradientHash            = "gradient_hash"
)

// NewEventGradientSubmitted creates a new gradient submitted event
func NewEventGradientSubmitted(
	miner string,
	gradientID uint64,
	ipfsHash string,
	modelVersion string,
	trainingRoundID uint64,
) sdk.Event {
	return sdk.NewEvent(
		EventTypeGradientSubmitted,
		sdk.NewAttribute(AttributeKeyMiner, miner),
		sdk.NewAttribute(AttributeKeyGradientID, fmt.Sprintf("%d", gradientID)),
		sdk.NewAttribute(AttributeKeyIPFSHash, ipfsHash),
		sdk.NewAttribute(AttributeKeyModelVersion, modelVersion),
		sdk.NewAttribute(AttributeKeyTrainingRoundID, fmt.Sprintf("%d", trainingRoundID)),
	)
}

// NewEventAggregationSubmitted creates a new aggregation submitted event
func NewEventAggregationSubmitted(
	proposer string,
	aggregationID uint64,
	trainingRoundID uint64,
) sdk.Event {
	return sdk.NewEvent(
		EventTypeAggregationSubmitted,
		sdk.NewAttribute(AttributeKeyProposer, proposer),
		sdk.NewAttribute(AttributeKeyAggregationID, fmt.Sprintf("%d", aggregationID)),
		sdk.NewAttribute(AttributeKeyTrainingRoundID, fmt.Sprintf("%d", trainingRoundID)),
	)
}

// NewEventAggregationChallenged creates a new aggregation challenged event
func NewEventAggregationChallenged(
	challenger string,
	challengeID uint64,
	aggregationID uint64,
) sdk.Event {
	return sdk.NewEvent(
		EventTypeAggregationChallenged,
		sdk.NewAttribute(AttributeKeyChallenger, challenger),
		sdk.NewAttribute(AttributeKeyChallengeID, fmt.Sprintf("%d", challengeID)),
		sdk.NewAttribute(AttributeKeyAggregationID, fmt.Sprintf("%d", aggregationID)),
	)
}

// NewEventAggregationFinalized creates a new aggregation finalized event
func NewEventAggregationFinalized(
	proposer string,
	aggregationID uint64,
	trainingRoundID uint64,
) sdk.Event {
	return sdk.NewEvent(
		EventTypeAggregationFinalized,
		sdk.NewAttribute(AttributeKeyProposer, proposer),
		sdk.NewAttribute(AttributeKeyAggregationID, fmt.Sprintf("%d", aggregationID)),
		sdk.NewAttribute(AttributeKeyTrainingRoundID, fmt.Sprintf("%d", trainingRoundID)),
	)
}

