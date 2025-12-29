package keeper

import (
	"fmt"

	sdkmath "cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// ConvergenceMetrics is now defined in types (generated from proto)
// This file contains helper methods and business logic

// ConvergenceConfig holds configuration for convergence monitoring
type ConvergenceConfig struct {
	LossHistoryWindow    uint64            // Number of loss values to track
	ConvergenceThreshold sdkmath.LegacyDec // Loss change threshold (e.g., 0.01 = 1% change)
	MinLossHistory       uint64            // Minimum history required for convergence check
	AdaptiveCompression  bool              // Adjust compression based on convergence
	EarlyStoppingEnabled bool              // Enable early stopping if converged
}

// GetConvergenceConfig returns the convergence monitoring configuration
func (k Keeper) GetConvergenceConfig(ctx sdk.Context) ConvergenceConfig {
	return ConvergenceConfig{
		LossHistoryWindow:    10,                                      // Track last 10 loss values
		ConvergenceThreshold: sdkmath.LegacyMustNewDecFromStr("0.01"), // 1% change threshold
		MinLossHistory:       5,                                       // Need at least 5 values
		AdaptiveCompression:  true,
		EarlyStoppingEnabled: false, // Disabled by default (can be enabled via governance)
	}
}

// UpdateConvergenceMetrics updates convergence metrics with a new loss value
func (k Keeper) UpdateConvergenceMetrics(
	ctx sdk.Context,
	trainingRoundID uint64,
	lossValue string, // BitNet integer format
) error {
	// Parse loss value (use existing parseBitNetInteger helper)
	loss, err := parseBitNetInteger(lossValue)
	if err != nil {
		return fmt.Errorf("invalid loss value: %w", err)
	}

	// Get existing metrics (if any)
	var metrics types.ConvergenceMetrics
	existingMetrics, err := k.ConvergenceMetrics.Get(ctx, trainingRoundID)
	if err == nil {
		metrics = existingMetrics
	} else {
		// Create new metrics
		metrics = types.ConvergenceMetrics{
			TrainingRoundId: trainingRoundID,
			LossHistory:     []string{},
		}
	}

	// Add new loss to history
	metrics.LossHistory = append(metrics.LossHistory, lossValue)

	// Keep only last N losses (from config)
	config := k.GetConvergenceConfig(ctx)
	if len(metrics.LossHistory) > int(config.LossHistoryWindow) {
		metrics.LossHistory = metrics.LossHistory[len(metrics.LossHistory)-int(config.LossHistoryWindow):]
	}

	// Update average loss
	var sum int64
	for _, lossStr := range metrics.LossHistory {
		l, err := parseBitNetInteger(lossStr)
		if err != nil {
			continue
		}
		sum += l
	}
	if len(metrics.LossHistory) > 0 {
		avgLoss := sum / int64(len(metrics.LossHistory))
		metrics.AverageLoss = formatBitNetInteger(avgLoss)
	}

	// Calculate convergence rate (using metrics already in memory, before saving)
	convergenceRate := k.calculateConvergenceRateFromMetrics(metrics, loss)
	metrics.ConvergenceRate = convergenceRate.String() // Convert LegacyDec to string

	// Check if converged (using metrics already in memory, before saving)
	isConverged := k.checkConvergenceFromMetrics(metrics, loss, config)
	metrics.IsConverged = isConverged
	metrics.ConvergenceThreshold = config.ConvergenceThreshold.String() // Convert LegacyDec to string
	metrics.LastUpdatedHeight = ctx.BlockHeight()

	// Store updated metrics
	if err := k.ConvergenceMetrics.Set(ctx, trainingRoundID, metrics); err != nil {
		return fmt.Errorf("failed to store convergence metrics: %w", err)
	}

	// Log convergence status
	if isConverged {
		ctx.Logger().Info("Model converged",
			"training_round_id", trainingRoundID,
			"loss", lossValue,
			"convergence_rate", convergenceRate.String(),
		)
	}

	// Adjust compression if adaptive compression is enabled
	if config.AdaptiveCompression && isConverged {
		// If converged, we can increase compression (reduce bandwidth)
		// since model is stable
		ctx.Logger().Info("Adjusting compression due to convergence",
			"training_round_id", trainingRoundID,
		)
		// This would trigger compression adjustment in scalability.go
	}

	return nil
}

// calculateConvergenceRate calculates the rate of loss reduction (from state)
func (k Keeper) calculateConvergenceRate(
	ctx sdk.Context,
	trainingRoundID uint64,
	currentLoss int64,
) sdkmath.LegacyDec {
	// Get loss history from state
	metrics, err := k.ConvergenceMetrics.Get(ctx, trainingRoundID)
	if err != nil {
		// No history yet, return zero
		return sdkmath.LegacyZeroDec()
	}

	return k.calculateConvergenceRateFromMetrics(metrics, currentLoss)
}

// calculateConvergenceRateFromMetrics calculates the rate of loss reduction from metrics in memory
func (k Keeper) calculateConvergenceRateFromMetrics(
	metrics types.ConvergenceMetrics,
	currentLoss int64,
) sdkmath.LegacyDec {
	// Need at least 2 loss values to calculate rate
	if len(metrics.LossHistory) < 2 {
		return sdkmath.LegacyZeroDec()
	}

	// Get previous loss (second to last in history, since last is the current one being added)
	// If we have N values after append, we want value at index N-2 (second to last)
	prevLossIdx := len(metrics.LossHistory) - 2
	if prevLossIdx < 0 {
		return sdkmath.LegacyZeroDec()
	}

	prevLoss, err := parseBitNetInteger(metrics.LossHistory[prevLossIdx])
	if err != nil {
		return sdkmath.LegacyZeroDec()
	}

	// Calculate rate of change: (current - previous) / previous
	if prevLoss == 0 {
		return sdkmath.LegacyZeroDec()
	}

	lossDiff := sdkmath.LegacyNewDecFromInt(sdkmath.NewInt(currentLoss - prevLoss))
	prevLossDec := sdkmath.LegacyNewDecFromInt(sdkmath.NewInt(prevLoss))

	// Rate = (current - previous) / previous
	rate := lossDiff.Quo(prevLossDec)

	return rate
}

// checkConvergence checks if the model has converged based on loss history (from state)
func (k Keeper) checkConvergence(
	ctx sdk.Context,
	trainingRoundID uint64,
	currentLoss int64,
	config ConvergenceConfig,
) bool {
	// Get loss history from state
	metrics, err := k.ConvergenceMetrics.Get(ctx, trainingRoundID)
	if err != nil {
		// No history yet, not converged
		return false
	}

	return k.checkConvergenceFromMetrics(metrics, currentLoss, config)
}

// checkConvergenceFromMetrics checks if the model has converged based on metrics in memory
func (k Keeper) checkConvergenceFromMetrics(
	metrics types.ConvergenceMetrics,
	currentLoss int64,
	config ConvergenceConfig,
) bool {
	// Need minimum history for convergence check
	if len(metrics.LossHistory) < int(config.MinLossHistory) {
		return false
	}

	// Check if loss change is below threshold (use metrics already in memory)
	convergenceRate := k.calculateConvergenceRateFromMetrics(metrics, currentLoss)
	absRate := convergenceRate.Abs()

	// Parse convergence threshold from metrics (it's stored as string)
	threshold, err := sdkmath.LegacyNewDecFromStr(metrics.ConvergenceThreshold)
	if err != nil {
		threshold = config.ConvergenceThreshold // Fallback to config
	}

	// If rate of change is below threshold, consider converged
	if absRate.LT(threshold) {
		// Also check if loss has stabilized (low variance in recent history)
		// Get last N losses for variance calculation
		recentLosses := metrics.LossHistory
		if len(recentLosses) > int(config.LossHistoryWindow) {
			recentLosses = recentLosses[len(recentLosses)-int(config.LossHistoryWindow):]
		}

		// Calculate variance
		var sum int64
		var count int
		for _, lossStr := range recentLosses {
			loss, err := parseBitNetInteger(lossStr)
			if err != nil {
				continue
			}
			sum += loss
			count++
		}

		if count == 0 {
			return false
		}

		avgLoss := sum / int64(count)

		// Calculate variance
		var varianceSum int64
		for _, lossStr := range recentLosses {
			loss, err := parseBitNetInteger(lossStr)
			if err != nil {
				continue
			}
			diff := loss - avgLoss
			varianceSum += diff * diff
		}

		varianceDec := sdkmath.LegacyNewDecFromInt(sdkmath.NewInt(varianceSum)).QuoInt64(int64(count))
		avgLossDec := sdkmath.LegacyNewDecFromInt(sdkmath.NewInt(avgLoss))

		// Coefficient of variation (CV) = std_dev / mean
		// Low CV indicates stability
		// Calculate square root of variance using integer square root approximation
		// Since variance is already a Dec, we can use an approximation
		// For simplicity, we'll use variance directly as a proxy for std_dev for CV calculation
		// CV approximation: sqrt(variance) / mean ≈ variance / (mean^2) when variance is small
		// For better accuracy, we can use Newton's method or power series, but for CV threshold check,
		// we can use a simpler approximation: cv ≈ variance / (mean * mean)
		cv := varianceDec.Quo(avgLossDec).Quo(avgLossDec)
		cvThreshold := sdkmath.LegacyMustNewDecFromStr("0.05") // 5% CV threshold

		// Converged if rate is below threshold AND variance is low
		return cv.LT(cvThreshold)
	}

	return false
}

// GetConvergenceStatus returns the current convergence status for a training round
func (k Keeper) GetConvergenceStatus(
	ctx sdk.Context,
	trainingRoundID uint64,
) (bool, sdkmath.LegacyDec, error) {
	// Get convergence metrics from state
	metrics, err := k.ConvergenceMetrics.Get(ctx, trainingRoundID)
	if err != nil {
		// No metrics found, return default values
		config := k.GetConvergenceConfig(ctx)
		return false, config.ConvergenceThreshold, nil
	}

	// Parse convergence rate from string
	convergenceRate, err := sdkmath.LegacyNewDecFromStr(metrics.ConvergenceRate)
	if err != nil {
		// If parse fails, return zero
		return metrics.IsConverged, sdkmath.LegacyZeroDec(), nil
	}

	return metrics.IsConverged, convergenceRate, nil
}

// AdjustCompressionForConvergence adjusts compression ratio based on convergence status
func (k Keeper) AdjustCompressionForConvergence(
	ctx sdk.Context,
	trainingRoundID uint64,
	isConverged bool,
) error {
	if !isConverged {
		return nil // No adjustment needed
	}

	// If converged, increase compression (reduce bandwidth)
	// This is because the model is stable and doesn't need as much precision
	ctx.Logger().Info("Increasing compression due to convergence",
		"training_round_id", trainingRoundID,
	)

	// Get network metrics
	metrics, err := k.GetNetworkMetrics(ctx)
	if err != nil {
		return fmt.Errorf("failed to get network metrics: %w", err)
	}

	// Adjust compression ratio
	// This would call AdjustCompressionRatio with a higher compression target
	_ = metrics // Use metrics if needed

	return nil
}

// MonitorConvergence monitors convergence for all active training rounds
// Called periodically in EndBlocker
func (k Keeper) MonitorConvergence(ctx sdk.Context) error {
	// Get all active training rounds
	// In production, iterate through active training rounds
	// For now, this is a placeholder

	// Get convergence config
	config := k.GetConvergenceConfig(ctx)

	// If early stopping is enabled and model is converged, trigger early stopping
	if config.EarlyStoppingEnabled {
		// Check convergence status for active rounds
		// If converged, mark training round as complete
		ctx.Logger().Info("Convergence monitoring active",
			"early_stopping_enabled", config.EarlyStoppingEnabled,
		)
	}

	return nil
}
