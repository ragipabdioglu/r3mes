package keeper

import (
	"encoding/binary"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

const (
	// WindowDuration is the fixed duration of a training window in blocks (100 blocks ~8-10 minutes)
	WindowDuration int64 = 100
)

// CreateTrainingWindow creates a new training window
func (k Keeper) CreateTrainingWindow(
	ctx sdk.Context,
	windowID uint64,
	startHeight int64,
	aggregatorNode string,
) error {
	// Validate start height
	if startHeight < 0 {
		return fmt.Errorf("start_height must be non-negative")
	}

	// Calculate end height
	endHeight := startHeight + WindowDuration

	// Check if window already exists
	_, err := k.TrainingWindows.Get(ctx, windowID)
	if err == nil {
		return fmt.Errorf("window %d already exists", windowID)
	}

	// Derive global seed from block hash (use window ID as training round ID)
	globalSeedUint64, err := k.DeriveGlobalSeed(ctx, windowID)
	if err != nil {
		return fmt.Errorf("failed to derive global seed: %w", err)
	}

	// Convert uint64 seed to []byte
	globalSeed := make([]byte, 8)
	binary.BigEndian.PutUint64(globalSeed, globalSeedUint64)

	// Create training window
	window := types.TrainingWindow{
		WindowId:       windowID,
		StartHeight:    startHeight,
		EndHeight:      endHeight,
		Status:         "collecting",
		GradientHashes: []string{},
		AggregatorNode: aggregatorNode,
		AggregationHash: "",
		ModelUpdateHash: "",
		GlobalSeed:     globalSeed,
		CreatedAt:      ctx.BlockTime(),
		FinalizedAt:   nil,
	}

	// Store window
	if err := k.TrainingWindows.Set(ctx, windowID, window); err != nil {
		return fmt.Errorf("failed to store training window: %w", err)
	}

	return nil
}

// GetTrainingWindow retrieves a training window
func (k Keeper) GetTrainingWindow(ctx sdk.Context, windowID uint64) (types.TrainingWindow, error) {
	window, err := k.TrainingWindows.Get(ctx, windowID)
	if err != nil {
		return types.TrainingWindow{}, fmt.Errorf("window %d not found: %w", windowID, err)
	}
	return window, nil
}

// SubmitAsyncGradient submits an asynchronous gradient (non-blocking)
func (k Keeper) SubmitAsyncGradient(
	ctx sdk.Context,
	windowID uint64,
	minerAddress string,
	gradientHash string,
	subnetID uint64,
	layerRange types.LayerRange,
) (uint64, error) {
	// Get training window
	window, err := k.GetTrainingWindow(ctx, windowID)
	if err != nil {
		return 0, err
	}

	// Validate window status
	if window.Status != "collecting" {
		return 0, fmt.Errorf("window %d is not in collecting status (current: %s)", windowID, window.Status)
	}

	// Validate window is still open
	if ctx.BlockHeight() > window.EndHeight {
		return 0, fmt.Errorf("window %d has already closed (end_height: %d, current: %d)", windowID, window.EndHeight, ctx.BlockHeight())
	}

	// Generate submission ID
	submissionID, err := k.AsyncGradientSubmissionID.Next(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to generate submission ID: %w", err)
	}

	// Create async gradient submission
	submission := types.AsyncGradientSubmission{
		SubmissionId:   submissionID,
		WindowId:       windowID,
		MinerAddress:   minerAddress,
		GradientHash:   gradientHash,
		SubnetId:       subnetID,
		LayerRange:     layerRange,
		Timestamp:      ctx.BlockTime(),
		CreditRecord:   true, // This is a credit record, not immediate update
	}

	// Store submission
	if err := k.AsyncGradientSubmissions.Set(ctx, submissionID, submission); err != nil {
		return 0, fmt.Errorf("failed to store async gradient submission: %w", err)
	}

	// Add gradient hash to window
	window.GradientHashes = append(window.GradientHashes, gradientHash)

	// Update window
	if err := k.TrainingWindows.Set(ctx, windowID, window); err != nil {
		return 0, fmt.Errorf("failed to update training window: %w", err)
	}

	return submissionID, nil
}

// SubmitLazyAggregation submits lazy aggregation at window boundary
func (k Keeper) SubmitLazyAggregation(
	ctx sdk.Context,
	windowID uint64,
	aggregatorNode string,
	collectedGradients []string,
	aggregationMethod string,
	resultHash string,
	merkleRoot []byte,
) (uint64, error) {
	// Get training window
	window, err := k.GetTrainingWindow(ctx, windowID)
	if err != nil {
		return 0, err
	}

	// Validate window status
	if window.Status != "collecting" && window.Status != "aggregating" {
		return 0, fmt.Errorf("window %d is not in collecting or aggregating status (current: %s)", windowID, window.Status)
	}

	// Validate aggregator node
	if window.AggregatorNode != aggregatorNode {
		return 0, fmt.Errorf("aggregator node mismatch: expected %s, got %s", window.AggregatorNode, aggregatorNode)
	}

	// Generate aggregation ID
	aggregationID, err := k.LazyAggregationID.Next(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to generate aggregation ID: %w", err)
	}

	// Create lazy aggregation
	aggregation := types.LazyAggregation{
		AggregationId:      aggregationID,
		WindowId:           windowID,
		AggregatorNode:     aggregatorNode,
		CollectedGradients: collectedGradients,
		AggregationMethod:  aggregationMethod,
		ResultHash:         resultHash,
		MerkleRoot:         merkleRoot,
		SubmissionHeight:   ctx.BlockHeight(),
		CreatedAt:          ctx.BlockTime(),
	}

	// Store aggregation
	if err := k.LazyAggregations.Set(ctx, aggregationID, aggregation); err != nil {
		return 0, fmt.Errorf("failed to store lazy aggregation: %w", err)
	}

	// Update window
	window.Status = "finalized"
	window.AggregationHash = resultHash
	finalizedAt := ctx.BlockTime()
	window.FinalizedAt = &finalizedAt

	if err := k.TrainingWindows.Set(ctx, windowID, window); err != nil {
		return 0, fmt.Errorf("failed to update training window: %w", err)
	}

	return aggregationID, nil
}

// CheckWindowBoundary checks if current block is at a window boundary
func (k Keeper) CheckWindowBoundary(ctx sdk.Context) (bool, uint64, error) {
	// Iterate through all windows to find active ones
	iter, err := k.TrainingWindows.Iterate(ctx, nil)
	if err != nil {
		return false, 0, fmt.Errorf("failed to iterate windows: %w", err)
	}
	defer iter.Close()

	currentHeight := ctx.BlockHeight()

	for ; iter.Valid(); iter.Next() {
		window, err := iter.Value()
		if err != nil {
			continue
		}

		// Check if we're at the end of a collecting window
		if window.Status == "collecting" && currentHeight >= window.EndHeight {
			return true, window.WindowId, nil
		}
	}

	return false, 0, nil
}

// GetActiveWindows returns all active (collecting) windows
func (k Keeper) GetActiveWindows(ctx sdk.Context) ([]types.TrainingWindow, error) {
	windows := make([]types.TrainingWindow, 0)
	iter, err := k.TrainingWindows.Iterate(ctx, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to iterate windows: %w", err)
	}
	defer iter.Close()

	for ; iter.Valid(); iter.Next() {
		window, err := iter.Value()
		if err != nil {
			continue
		}

		if window.Status == "collecting" {
			windows = append(windows, window)
		}
	}

	return windows, nil
}

// ProcessWindowBoundaries processes window boundaries and transitions windows to aggregating status
func (k Keeper) ProcessWindowBoundaries(ctx sdk.Context) error {
	currentHeight := ctx.BlockHeight()

	// Iterate through all windows
	iter, err := k.TrainingWindows.Iterate(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to iterate windows: %w", err)
	}
	defer iter.Close()

	for ; iter.Valid(); iter.Next() {
		window, err := iter.Value()
		if err != nil {
			continue
		}

		// Check if we're at or past the end of a collecting window
		if window.Status == "collecting" && currentHeight >= window.EndHeight {
			// Transition window to aggregating status
			window.Status = "aggregating"
			if err := k.TrainingWindows.Set(ctx, window.WindowId, window); err != nil {
				ctx.Logger().Error(fmt.Sprintf("Failed to update window %d status: %v", window.WindowId, err))
				continue
			}

			// Emit event for window boundary
			ctx.EventManager().EmitEvent(
				sdk.NewEvent(
					types.EventTypeWindowBoundary,
					sdk.NewAttribute(types.AttributeKeyWindowID, fmt.Sprintf("%d", window.WindowId)),
					sdk.NewAttribute(types.AttributeKeyEndHeight, fmt.Sprintf("%d", window.EndHeight)),
					sdk.NewAttribute(types.AttributeKeyAggregatorNode, window.AggregatorNode),
					sdk.NewAttribute(types.AttributeKeyGradientCount, fmt.Sprintf("%d", len(window.GradientHashes))),
				),
			)

			ctx.Logger().Info(fmt.Sprintf("Window %d reached boundary, ready for aggregation", window.WindowId))
		}
	}

	return nil
}

