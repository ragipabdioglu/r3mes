package keeper

import (
	"context"
	"fmt"

	"cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// Fraud detection bounty parameters
const (
	// FraudDetectionBountyMultiplierMin is the minimum bounty multiplier (10x)
	FraudDetectionBountyMultiplierMin = 10
	
	// FraudDetectionBountyMultiplierMax is the maximum bounty multiplier (20x)
	FraudDetectionBountyMultiplierMax = 20
	
	// BaseValidatorIncentive is the base reward for validators per verification (even without fraud)
	BaseValidatorIncentive = 30 // R3MES tokens
	
	// ContinuousValidatorRewardInterval is the number of blocks between continuous validator rewards
	ContinuousValidatorRewardInterval = 100 // blocks
)

// CalculateFraudDetectionBounty calculates the bounty reward for fraud detection
// Formula: base_reward * multiplier (10x to 20x based on severity)
// Severity: 1.0 = critical fraud, 0.5 = minor fraud
func (k Keeper) CalculateFraudDetectionBounty(ctx context.Context, baseReward sdk.Coins, severity float64) (sdk.Coins, error) {
	// Clamp severity between 0.5 and 1.0
	if severity < 0.5 {
		severity = 0.5
	}
	if severity > 1.0 {
		severity = 1.0
	}
	
	// Calculate multiplier: 10x (minor) to 20x (critical)
	// Linear interpolation: multiplier = 10 + (severity - 0.5) * 20
	multiplier := 10.0 + (severity-0.5)*20.0
	
	// Calculate bounty
	bounty := sdk.NewCoins()
	for _, coin := range baseReward {
		multiplierDec := math.LegacyMustNewDecFromStr(fmt.Sprintf("%.6f", multiplier))
		bountyAmount := multiplierDec.MulInt(coin.Amount).TruncateInt()
		if !bountyAmount.IsZero() {
			bounty = bounty.Add(sdk.NewCoin(coin.Denom, bountyAmount))
		}
	}
	
	return bounty, nil
}

// DistributeFraudDetectionBounty distributes bounty reward to fraud detector
func (k Keeper) DistributeFraudDetectionBounty(
	ctx context.Context,
	detectorAddress string,
	baseReward sdk.Coins,
	severity float64,
	fraudType string,
) error {
	// Calculate bounty
	bounty, err := k.CalculateFraudDetectionBounty(ctx, baseReward, severity)
	if err != nil {
		return fmt.Errorf("failed to calculate fraud detection bounty: %w", err)
	}
	
	// Convert detector address to AccAddress
	detectorAddr, err := k.addressCodec.StringToBytes(detectorAddress)
	if err != nil {
		return fmt.Errorf("invalid detector address: %w", err)
	}
	
	// Mint coins to module account
	if err := k.bankKeeper.MintCoins(ctx, RemesModuleName, bounty); err != nil {
		return fmt.Errorf("failed to mint bounty coins: %w", err)
	}
	
	// Send coins from module to detector
	if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, RemesModuleName, detectorAddr, bounty); err != nil {
		return fmt.Errorf("failed to send bounty to detector: %w", err)
	}
	
	// Emit event
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeFraudDetectionBounty,
			sdk.NewAttribute(types.AttributeKeyDetector, detectorAddress),
			sdk.NewAttribute(types.AttributeKeyBountyAmount, bounty.String()),
			sdk.NewAttribute("fraud_type", fraudType),
			sdk.NewAttribute("severity", fmt.Sprintf("%.2f", severity)),
		),
	)
	
	return nil
}

// DistributeContinuousValidatorIncentive distributes continuous reward to validators
// Validators receive rewards even when no fraud is detected, incentivizing active participation
func (k Keeper) DistributeContinuousValidatorIncentive(
	ctx context.Context,
	validatorAddress string,
	verificationCount uint64,
) error {
	// Calculate reward: base_reward * verification_count
	baseReward := math.NewInt(BaseValidatorIncentive)
	totalReward := baseReward.Mul(math.NewIntFromUint64(verificationCount))
	
	// Create reward coins
	reward := sdk.NewCoins(sdk.NewCoin("remes", totalReward))
	
	if reward.IsZero() {
		return nil // No reward to distribute
	}
	
	// Convert validator address to AccAddress
	validatorAddr, err := k.addressCodec.StringToBytes(validatorAddress)
	if err != nil {
		return fmt.Errorf("invalid validator address: %w", err)
	}
	
	// Mint coins to module account
	if err := k.bankKeeper.MintCoins(ctx, RemesModuleName, reward); err != nil {
		return fmt.Errorf("failed to mint validator incentive coins: %w", err)
	}
	
	// Send coins from module to validator
	if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, RemesModuleName, validatorAddr, reward); err != nil {
		return fmt.Errorf("failed to send validator incentive: %w", err)
	}
	
	// Emit event
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeContinuousValidatorIncentive,
			sdk.NewAttribute(types.AttributeKeyValidator, validatorAddress),
			sdk.NewAttribute(types.AttributeKeyRewardAmount, reward.String()),
			sdk.NewAttribute("verification_count", fmt.Sprintf("%d", verificationCount)),
		),
	)
	
	return nil
}

// GetTransparentRewardFormula returns the transparent reward formula documentation
// This provides clear bounds and calculations for all reward types
func (k Keeper) GetTransparentRewardFormula() map[string]interface{} {
	return map[string]interface{}{
		"miner_reward": map[string]interface{}{
			"base": BaseMinerReward,
			"formula": "base_reward * (1.0 + trust_score)",
			"bounds": map[string]interface{}{
				"min": BaseMinerReward * 1,   // 100 tokens (trust_score = 0.0)
				"max": BaseMinerReward * 2,   // 200 tokens (trust_score = 1.0)
			},
		},
		"proposer_reward": map[string]interface{}{
			"base": BaseProposerReward,
			"formula": "base_reward + (participant_count * participant_bonus)",
			"bounds": map[string]interface{}{
				"min": BaseProposerReward,                    // 500 tokens (0 participants)
				"max": BaseProposerReward + (100 * ProposerParticipantBonus), // 1500 tokens (100 participants)
			},
		},
		"serving_reward": map[string]interface{}{
			"base": BaseServingReward,
			"formula": "base_reward * performance_score",
			"bounds": map[string]interface{}{
				"min": 0,                // 0 tokens (performance_score = 0.0)
				"max": BaseServingReward, // 50 tokens (performance_score = 1.0)
			},
		},
		"validator_reward": map[string]interface{}{
			"base": BaseValidatorReward,
			"formula": "base_reward + (verification_count * verification_bonus)",
			"bounds": map[string]interface{}{
				"min": BaseValidatorReward,                    // 30 tokens (0 verifications)
				"max": BaseValidatorReward + (100 * ValidatorVerificationBonus), // 530 tokens (100 verifications)
			},
		},
		"fraud_detection_bounty": map[string]interface{}{
			"formula": "base_reward * multiplier",
			"multiplier": map[string]interface{}{
				"min": FraudDetectionBountyMultiplierMin, // 10x
				"max": FraudDetectionBountyMultiplierMax, // 20x
			},
			"severity_range": map[string]interface{}{
				"min": 0.5, // Minor fraud
				"max": 1.0, // Critical fraud
			},
		},
		"continuous_validator_incentive": map[string]interface{}{
			"base": BaseValidatorIncentive,
			"formula": "base_reward * verification_count",
			"interval": ContinuousValidatorRewardInterval, // blocks
		},
	}
}

