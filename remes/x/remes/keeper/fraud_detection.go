package keeper

import (
	"context"
	"fmt"
	"strconv"

	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// DetectFraud detects fraud based on trap verification results
// Updates miner's fraud score and trap tracking statistics
func (k Keeper) DetectFraud(ctx context.Context, miner string, trapResult SimilarityResult) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get miner's contribution record
	contribution, err := k.MiningContributions.Get(ctx, miner)
	if err != nil {
		// If miner doesn't have a contribution record yet, create one
		contribution = types.MiningContribution{
			MinerAddress: miner,
		}
	}

	// Update trap tracking
	if trapResult.IsValid {
		contribution.TrapsCaught++
		sdkCtx.Logger().Info(fmt.Sprintf("Miner %s passed trap verification (similarity: %.6f)", miner, trapResult.SimilarityScore))
	} else {
		contribution.TrapsFailed++
		sdkCtx.Logger().Error(fmt.Sprintf("Miner %s failed trap verification: %s (similarity: %.6f)", miner, trapResult.Reason, trapResult.SimilarityScore))
	}

	// Calculate fraud score
	fraudScore := k.calculateFraudScore(contribution)
	contribution.FraudScore = fmt.Sprintf("%.6f", fraudScore)

	// Update miner contribution
	if err := k.MiningContributions.Set(ctx, miner, contribution); err != nil {
		return fmt.Errorf("failed to update miner contribution: %w", err)
	}

	// If fraud score exceeds threshold, trigger additional actions (slashing, etc.)
	// Level 3 slashing: 50% of staked tokens (severe penalty for fraud)
	if fraudScore > 0.5 {
		sdkCtx.Logger().Error(fmt.Sprintf("Miner %s fraud score exceeds threshold: %.6f, triggering slashing", miner, fraudScore))
		
		// Trigger Level 3 slashing (50% of stake)
		slashFraction := sdkmath.LegacyMustNewDecFromStr("0.5") // 50%
		reason := fmt.Sprintf("fraud_detection_high_score: score=%.6f", fraudScore)
		
		if err := k.SlashMiner(ctx, miner, slashFraction, reason); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to slash miner %s for fraud: %v", miner, err))
			// Continue even if slashing fails (log error but don't fail the fraud detection)
		} else {
			sdkCtx.Logger().Info(fmt.Sprintf("Successfully slashed miner %s: 50%% of stake (fraud score: %.6f)", miner, fraudScore))
		}
	}

	return nil
}

// calculateFraudScore calculates fraud score based on trap statistics
// Formula: traps_failed / (traps_caught + traps_failed + 1)
// Returns value between 0.0 (honest) and 1.0 (fraudulent)
func (k Keeper) calculateFraudScore(contribution types.MiningContribution) float64 {
	totalTraps := contribution.TrapsCaught + contribution.TrapsFailed
	if totalTraps == 0 {
		return 0.0 // No traps yet, neutral score
	}

	// Fraud score is the ratio of failed traps
	fraudScore := float64(contribution.TrapsFailed) / float64(totalTraps)

	// Clamp to [0.0, 1.0]
	if fraudScore > 1.0 {
		return 1.0
	}
	if fraudScore < 0.0 {
		return 0.0
	}

	return fraudScore
}

// GetMinerFraudScore retrieves miner's fraud score
func (k Keeper) GetMinerFraudScore(ctx context.Context, miner string) (float64, error) {
	contribution, err := k.MiningContributions.Get(ctx, miner)
	if err != nil {
		return 0.0, fmt.Errorf("miner contribution not found: %w", err)
	}

	if contribution.FraudScore == "" {
		return 0.0, nil
	}

	fraudScore, err := strconv.ParseFloat(contribution.FraudScore, 64)
	if err != nil {
		return 0.0, fmt.Errorf("failed to parse fraud score: %w", err)
	}

	return fraudScore, nil
}

