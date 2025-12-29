package keeper

import (
	"fmt"

	"cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// Trust score update parameters
const (
	// TrustScoreIncreaseOnFinalization is the trust score increase when a gradient is finalized
	TrustScoreIncreaseOnFinalization = 0.01 // +1% per finalized gradient
	
	// TrustScoreDecreaseOnFraud is the trust score decrease when fraud is detected
	TrustScoreDecreaseOnFraud = 0.10 // -10% per fraud detection
	
	// TrustScoreDecreaseOnChallenge is the trust score decrease when a gradient is challenged (even if not fraud)
	TrustScoreDecreaseOnChallenge = 0.02 // -2% per challenge
	
	// MinTrustScore is the minimum trust score (cannot go below this)
	MinTrustScore = 0.0
	
	// MaxTrustScore is the maximum trust score (cannot go above this)
	MaxTrustScore = 1.0
)

// UpdateTrustScore updates a miner's trust score based on gradient validation results
// This should be called when:
// 1. A gradient is finalized (challenge period expired without challenge)
// 2. A gradient challenge is resolved (fraud detected or not)
// 3. A gradient is accepted into an aggregation
func (k Keeper) UpdateTrustScore(
	ctx sdk.Context,
	minerAddress string,
	updateType string, // "finalized", "fraud_detected", "challenged", "accepted"
) error {
	contribution, err := k.MiningContributions.Get(ctx, minerAddress)
	if err != nil {
		// Miner not found - create new contribution with default trust score
		contribution = types.MiningContribution{
			MinerAddress:          minerAddress,
			TotalSubmissions:     0,
			SuccessfulSubmissions: 0,
			TrustScore:            "0.5",
			ReputationTier:        "new",
			SlashingEvents:        0,
		}
	}
	
	// Parse current trust score
	currentTrustScore, err := math.LegacyNewDecFromStr(contribution.TrustScore)
	if err != nil {
		// If parsing fails, use default
		currentTrustScore = math.LegacyNewDecWithPrec(5, 1) // 0.5
	}
	
	// Calculate new trust score based on update type
	var newTrustScore math.LegacyDec
	
	switch updateType {
	case "finalized":
		// Gradient finalized without challenge - increase trust score
		increase := math.LegacyNewDecWithPrec(1, 2) // 0.01
		newTrustScore = currentTrustScore.Add(increase)
		
	case "fraud_detected":
		// Fraud detected - significant decrease
		decrease := math.LegacyNewDecWithPrec(10, 2) // 0.10
		newTrustScore = currentTrustScore.Sub(decrease)
		
	case "challenged":
		// Gradient challenged (but not necessarily fraud) - small decrease
		decrease := math.LegacyNewDecWithPrec(2, 2) // 0.02
		newTrustScore = currentTrustScore.Sub(decrease)
		
	case "accepted":
		// Gradient accepted into aggregation - small increase
		increase := math.LegacyNewDecWithPrec(5, 3) // 0.005
		newTrustScore = currentTrustScore.Add(increase)
		
	default:
		return errorsmod.Wrapf(types.ErrInvalidMiner, "unknown trust score update type: %s", updateType)
	}
	
	// Clamp trust score to [MinTrustScore, MaxTrustScore]
	if newTrustScore.LT(math.LegacyZeroDec()) {
		newTrustScore = math.LegacyZeroDec()
	}
	if newTrustScore.GT(math.LegacyOneDec()) {
		newTrustScore = math.LegacyOneDec()
	}
	
	// Update contribution
	contribution.TrustScore = newTrustScore.String()
	
	// Update reputation tier based on trust score
	contribution.ReputationTier = k.calculateReputationTier(newTrustScore)
	
	// Save updated contribution
	return k.MiningContributions.Set(ctx, minerAddress, contribution)
}

// calculateReputationTier calculates reputation tier based on trust score
func (k Keeper) calculateReputationTier(trustScore math.LegacyDec) string {
	if trustScore.LT(math.LegacyNewDecWithPrec(3, 1)) { // < 0.3
		return "low"
	} else if trustScore.LT(math.LegacyNewDecWithPrec(6, 1)) { // < 0.6
		return "medium"
	} else if trustScore.LT(math.LegacyNewDecWithPrec(8, 1)) { // < 0.8
		return "high"
	} else {
		return "excellent"
	}
}

// UpdateTrustScoreOnGradientFinalized updates trust score when a gradient is finalized
// This is called from EndBlocker when challenge period expires
func (k Keeper) UpdateTrustScoreOnGradientFinalized(ctx sdk.Context, gradientID uint64) error {
	gradient, err := k.StoredGradients.Get(ctx, gradientID)
	if err != nil {
		return errorsmod.Wrapf(err, "gradient %d not found", gradientID)
	}
	
	// Only update if gradient was finalized without challenge
	if gradient.Status == "finalized" || gradient.Status == "aggregated" {
		return k.UpdateTrustScore(ctx, gradient.Miner, "finalized")
	}
	
	return nil
}

// UpdateTrustScoreOnChallengeResolution updates trust score when a challenge is resolved
// This function takes sdk.Context (not context.Context) for consistency with other keeper methods
func (k Keeper) UpdateTrustScoreOnChallengeResolution(
	ctx sdk.Context,
	challengeID uint64,
	fraudDetected bool,
) error {
	challenge, err := k.ChallengeRecords.Get(ctx, challengeID)
	if err != nil {
		return errorsmod.Wrapf(err, "challenge %d not found", challengeID)
	}
	
	// Get aggregation to find participant miners
	aggregation, err := k.AggregationRecords.Get(ctx, challenge.AggregationId)
	if err != nil {
		return errorsmod.Wrapf(err, "aggregation %d not found", challenge.AggregationId)
	}
	
	// Update trust score for all miners in the aggregation
	// If fraud detected, decrease trust score for all participants
	// If no fraud, small decrease for being challenged
	updateType := "challenged"
	if fraudDetected {
		updateType = "fraud_detected"
	}
	
	// Update trust score for proposer (most responsible)
	if err := k.UpdateTrustScore(ctx, aggregation.Proposer, updateType); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to update proposer trust score: %v", err))
	}
	
	// Optionally update trust score for participant miners
	// For now, we'll only update proposer's trust score
	// In production, you might want to update all participants
	
	return nil
}

