package training

import (
	"context"
	"fmt"

	"cosmossdk.io/collections"
	corestore "cosmossdk.io/core/store"
	"github.com/cosmos/cosmos-sdk/codec"

	"remes/x/remes/keeper/core"
	"remes/x/remes/keeper/infra"
	"remes/x/remes/types"
)

// TrainingKeeper handles training and gradient management functionality
type TrainingKeeper struct {
	core  *core.CoreKeeper
	infra *infra.InfraKeeper

	// Training-related collections
	StoredGradients               collections.Map[uint64, types.StoredGradient]
	StoredGradientID              collections.Sequence
	AggregationRecords            collections.Map[uint64, types.AggregationRecord]
	AggregationID                 collections.Sequence
	AggregationCommitments        collections.Map[uint64, types.AggregationCommitment]
	AggregationCommitmentID       collections.Sequence
	MiningContributions           collections.Map[string, types.MiningContribution]
	TrainingWindows               collections.Map[uint64, types.TrainingWindow]
	AsyncGradientSubmissions      collections.Map[uint64, types.AsyncGradientSubmission]
	AsyncGradientSubmissionID     collections.Sequence
	LazyAggregations              collections.Map[uint64, types.LazyAggregation]
	LazyAggregationID             collections.Sequence
	ConvergenceMetrics            collections.Map[uint64, types.ConvergenceMetrics]
	PendingAggregationsByDeadline collections.Map[int64, types.AggregationIDList]
	SubnetConfigs                 collections.Map[uint64, types.SubnetConfig]
	ActivationTransmissions       collections.Map[uint64, types.ActivationTransmission]
	ActivationTransmissionID      collections.Sequence
	SubnetTrainingWorkflows       collections.Map[uint64, types.SubnetTrainingWorkflow]
	SubnetTrainingWorkflowID      collections.Sequence
}

// NewTrainingKeeper creates a new training keeper
func NewTrainingKeeper(
	storeService corestore.KVStoreService,
	cdc codec.Codec,
	coreKeeper *core.CoreKeeper,
	infraKeeper *infra.InfraKeeper,
) (*TrainingKeeper, error) {
	sb := collections.NewSchemaBuilder(storeService)

	k := &TrainingKeeper{
		core:  coreKeeper,
		infra: infraKeeper,

		StoredGradients:               collections.NewMap(sb, types.StoredGradientKey, "stored_gradients", collections.Uint64Key, codec.CollValue[types.StoredGradient](cdc)),
		StoredGradientID:              collections.NewSequence(sb, types.StoredGradientIDKey, "stored_gradient_id"),
		AggregationRecords:            collections.NewMap(sb, types.AggregationRecordKey, "aggregation_records", collections.Uint64Key, codec.CollValue[types.AggregationRecord](cdc)),
		AggregationID:                 collections.NewSequence(sb, types.AggregationIDKey, "aggregation_id"),
		AggregationCommitments:        collections.NewMap(sb, types.AggregationCommitmentKey, "aggregation_commitments", collections.Uint64Key, codec.CollValue[types.AggregationCommitment](cdc)),
		AggregationCommitmentID:       collections.NewSequence(sb, types.AggregationCommitmentIDKey, "aggregation_commitment_id"),
		MiningContributions:           collections.NewMap(sb, types.MiningContributionKey, "mining_contributions", collections.StringKey, codec.CollValue[types.MiningContribution](cdc)),
		TrainingWindows:               collections.NewMap(sb, types.TrainingWindowKey, "training_windows", collections.Uint64Key, codec.CollValue[types.TrainingWindow](cdc)),
		AsyncGradientSubmissions:      collections.NewMap(sb, types.AsyncGradientSubmissionKey, "async_gradient_submissions", collections.Uint64Key, codec.CollValue[types.AsyncGradientSubmission](cdc)),
		AsyncGradientSubmissionID:     collections.NewSequence(sb, types.AsyncGradientSubmissionIDKey, "async_gradient_submission_id"),
		LazyAggregations:              collections.NewMap(sb, types.LazyAggregationKey, "lazy_aggregations", collections.Uint64Key, codec.CollValue[types.LazyAggregation](cdc)),
		LazyAggregationID:             collections.NewSequence(sb, types.LazyAggregationIDKey, "lazy_aggregation_id"),
		ConvergenceMetrics:            collections.NewMap(sb, types.ConvergenceMetricsKey, "convergence_metrics", collections.Uint64Key, codec.CollValue[types.ConvergenceMetrics](cdc)),
		PendingAggregationsByDeadline: collections.NewMap(sb, types.PendingAggregationsByDeadlineKey, "pending_aggregations_by_deadline", collections.Int64Key, codec.CollValue[types.AggregationIDList](cdc)),
		SubnetConfigs:                 collections.NewMap(sb, types.SubnetConfigKey, "subnet_configs", collections.Uint64Key, codec.CollValue[types.SubnetConfig](cdc)),
		ActivationTransmissions:       collections.NewMap(sb, types.ActivationTransmissionKey, "activation_transmissions", collections.Uint64Key, codec.CollValue[types.ActivationTransmission](cdc)),
		ActivationTransmissionID:      collections.NewSequence(sb, types.ActivationTransmissionIDKey, "activation_transmission_id"),
		SubnetTrainingWorkflows:       collections.NewMap(sb, types.SubnetTrainingWorkflowKey, "subnet_training_workflows", collections.Uint64Key, codec.CollValue[types.SubnetTrainingWorkflow](cdc)),
		SubnetTrainingWorkflowID:      collections.NewSequence(sb, types.SubnetTrainingWorkflowIDKey, "subnet_training_workflow_id"),
	}

	// Build schema (not used directly but validates collections)
	_, err := sb.Build()
	if err != nil {
		return nil, fmt.Errorf("failed to build training keeper schema: %w", err)
	}

	return k, nil
}

// SubmitGradient submits a gradient for training
func (k *TrainingKeeper) SubmitGradient(ctx context.Context, gradient types.StoredGradient) error {
	// Generate new gradient ID
	gradientID, err := k.StoredGradientID.Next(ctx)
	if err != nil {
		return fmt.Errorf("failed to generate gradient ID: %w", err)
	}

	// Set gradient ID
	gradient.Id = gradientID

	// Verify IPFS content exists (if infra keeper is available)
	if k.infra != nil {
		exists, err := k.infra.VerifyIPFSContent(ctx, gradient.IpfsHash)
		if err != nil {
			return fmt.Errorf("failed to verify IPFS content: %w", err)
		}
		if !exists {
			return fmt.Errorf("IPFS content does not exist: %s", gradient.IpfsHash)
		}
	}

	// Store gradient
	if err := k.StoredGradients.Set(ctx, gradientID, gradient); err != nil {
		return fmt.Errorf("failed to store gradient: %w", err)
	}

	// Update mining contribution
	if err := k.updateMiningContribution(ctx, gradient.Miner, gradientID); err != nil {
		return fmt.Errorf("failed to update mining contribution: %w", err)
	}

	return nil
}

// GetGradient retrieves a gradient by ID
func (k *TrainingKeeper) GetGradient(ctx context.Context, gradientID uint64) (types.StoredGradient, error) {
	gradient, err := k.StoredGradients.Get(ctx, gradientID)
	if err != nil {
		return types.StoredGradient{}, fmt.Errorf("failed to get gradient %d: %w", gradientID, err)
	}
	return gradient, nil
}

// AggregateGradients performs gradient aggregation
func (k *TrainingKeeper) AggregateGradients(ctx context.Context, gradients []types.StoredGradient) error {
	// Generate new aggregation ID
	aggregationID, err := k.AggregationID.Next(ctx)
	if err != nil {
		return fmt.Errorf("failed to generate aggregation ID: %w", err)
	}

	// Extract gradient IDs
	var gradientIDs []uint64
	for _, gradient := range gradients {
		gradientIDs = append(gradientIDs, gradient.Id)
	}

	// Create aggregation record
	aggregation := types.AggregationRecord{
		AggregationId:          aggregationID,
		ParticipantGradientIds: gradientIDs,
		Status:                 "pending",
		// Add other required fields based on types.AggregationRecord
	}

	// Store aggregation
	if err := k.AggregationRecords.Set(ctx, aggregationID, aggregation); err != nil {
		return fmt.Errorf("failed to store aggregation: %w", err)
	}

	return nil
}

// GetAggregation retrieves an aggregation by ID
func (k *TrainingKeeper) GetAggregation(ctx context.Context, aggregationID uint64) (types.AggregationRecord, error) {
	aggregation, err := k.AggregationRecords.Get(ctx, aggregationID)
	if err != nil {
		return types.AggregationRecord{}, fmt.Errorf("failed to get aggregation %d: %w", aggregationID, err)
	}
	return aggregation, nil
}

// CreateTrainingWindow creates a new training window
func (k *TrainingKeeper) CreateTrainingWindow(ctx context.Context, window types.TrainingWindow) error {
	if err := k.TrainingWindows.Set(ctx, window.WindowId, window); err != nil {
		return fmt.Errorf("failed to create training window: %w", err)
	}
	return nil
}

// GetTrainingWindow retrieves a training window by ID
func (k *TrainingKeeper) GetTrainingWindow(ctx context.Context, windowID uint64) (types.TrainingWindow, error) {
	window, err := k.TrainingWindows.Get(ctx, windowID)
	if err != nil {
		return types.TrainingWindow{}, fmt.Errorf("failed to get training window %d: %w", windowID, err)
	}
	return window, nil
}

// GetMiningContribution retrieves mining contribution for a miner
func (k *TrainingKeeper) GetMiningContribution(ctx context.Context, minerAddress string) (types.MiningContribution, error) {
	contribution, err := k.MiningContributions.Get(ctx, minerAddress)
	if err != nil {
		return types.MiningContribution{}, fmt.Errorf("failed to get mining contribution for %s: %w", minerAddress, err)
	}
	return contribution, nil
}

// updateMiningContribution updates mining contribution for a miner
func (k *TrainingKeeper) updateMiningContribution(ctx context.Context, minerAddress string, gradientID uint64) error {
	// Get existing contribution or create new one
	contribution, err := k.MiningContributions.Get(ctx, minerAddress)
	if err != nil {
		// Create new contribution if not found
		contribution = types.MiningContribution{
			MinerAddress: minerAddress,
			// Initialize other fields as needed
		}
	}

	// Update contribution (add logic to update fields based on gradient submission)
	// This is a placeholder - implement actual contribution calculation
	contribution.MinerAddress = minerAddress

	// Store updated contribution
	if err := k.MiningContributions.Set(ctx, minerAddress, contribution); err != nil {
		return fmt.Errorf("failed to update mining contribution: %w", err)
	}

	return nil
}

// AddPendingAggregation adds an aggregation to the pending list by deadline
func (k *TrainingKeeper) AddPendingAggregation(ctx context.Context, deadline int64, aggregationID uint64) error {
	// Get existing list or create new one
	list, err := k.PendingAggregationsByDeadline.Get(ctx, deadline)
	if err != nil {
		// Create new list if not found
		list = types.AggregationIDList{
			AggregationIds: []uint64{aggregationID},
		}
	} else {
		// Add to existing list
		list.AggregationIds = append(list.AggregationIds, aggregationID)
	}

	// Store updated list
	return k.PendingAggregationsByDeadline.Set(ctx, deadline, list)
}

// GetPendingAggregationsByDeadline retrieves aggregations pending at a specific deadline
func (k *TrainingKeeper) GetPendingAggregationsByDeadline(ctx context.Context, deadline int64) ([]uint64, error) {
	list, err := k.PendingAggregationsByDeadline.Get(ctx, deadline)
	if err != nil {
		return nil, fmt.Errorf("failed to get pending aggregations for deadline %d: %w", deadline, err)
	}
	return list.AggregationIds, nil
}

// RemovePendingAggregation removes an aggregation from the pending list
func (k *TrainingKeeper) RemovePendingAggregation(ctx context.Context, deadline int64, aggregationID uint64) error {
	list, err := k.PendingAggregationsByDeadline.Get(ctx, deadline)
	if err != nil {
		return fmt.Errorf("failed to get pending aggregations: %w", err)
	}

	// Remove aggregation ID from list
	var newList []uint64
	for _, id := range list.AggregationIds {
		if id != aggregationID {
			newList = append(newList, id)
		}
	}

	// Update list
	list.AggregationIds = newList

	// Store updated list (or remove if empty)
	if len(newList) == 0 {
		return k.PendingAggregationsByDeadline.Remove(ctx, deadline)
	}
	return k.PendingAggregationsByDeadline.Set(ctx, deadline, list)
}

// RecordConvergenceMetrics records convergence metrics for a training round
func (k *TrainingKeeper) RecordConvergenceMetrics(ctx context.Context, metrics types.ConvergenceMetrics) error {
	if err := k.ConvergenceMetrics.Set(ctx, metrics.TrainingRoundId, metrics); err != nil {
		return fmt.Errorf("failed to record convergence metrics: %w", err)
	}
	return nil
}

// GetConvergenceMetrics retrieves convergence metrics for a training round
func (k *TrainingKeeper) GetConvergenceMetrics(ctx context.Context, trainingRoundID uint64) (types.ConvergenceMetrics, error) {
	metrics, err := k.ConvergenceMetrics.Get(ctx, trainingRoundID)
	if err != nil {
		return types.ConvergenceMetrics{}, fmt.Errorf("failed to get convergence metrics for round %d: %w", trainingRoundID, err)
	}
	return metrics, nil
}
