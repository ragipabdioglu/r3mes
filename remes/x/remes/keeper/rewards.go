package keeper

import (
	"context"
	"fmt"

	"cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// Reward parameters (can be moved to params later)
const (
	// Base reward for miner per gradient submission
	BaseMinerReward = 100 // R3MES tokens

	// Base reward for proposer per aggregation
	BaseProposerReward = 500 // R3MES tokens

	// Bonus per participant in aggregation
	ProposerParticipantBonus = 10 // R3MES tokens per participant

	// Base reward for serving node per inference
	BaseServingReward = 50 // R3MES tokens

	// Base reward for validator per verification
	BaseValidatorReward = 30 // R3MES tokens

	// Bonus per verification for validator
	ValidatorVerificationBonus = 5 // R3MES tokens per verification

	// Module name for minting
	RemesModuleName = types.ModuleName

	// Maximum supply cap (1 billion R3MES tokens)
	// This prevents unlimited minting and ensures economic security
	MaxSupplyCap = 1_000_000_000

	// Denom for R3MES tokens
	TokenDenom = "remes"
)

// checkSupplyCap verifies that minting won't exceed the maximum supply cap
// Returns error if minting would exceed the cap
func (k Keeper) checkSupplyCap(ctx context.Context, mintAmount sdk.Coins) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get current total supply
	currentSupply := k.bankKeeper.GetSupply(ctx, TokenDenom)

	// Get mint amount for remes denom
	mintRemes := mintAmount.AmountOf(TokenDenom)

	// Calculate new total supply
	newSupply := currentSupply.Amount.Add(mintRemes)

	// Check against max supply cap
	maxSupply := math.NewInt(MaxSupplyCap)
	if newSupply.GT(maxSupply) {
		sdkCtx.Logger().Error(fmt.Sprintf(
			"Supply cap exceeded: current=%s, mint=%s, max=%s",
			currentSupply.Amount.String(),
			mintRemes.String(),
			maxSupply.String(),
		))
		return fmt.Errorf(
			"minting %s would exceed max supply cap of %s (current: %s)",
			mintRemes.String(),
			maxSupply.String(),
			currentSupply.Amount.String(),
		)
	}

	return nil
}

// mintCoinsWithSupplyCheck mints coins after verifying supply cap
func (k Keeper) mintCoinsWithSupplyCheck(ctx context.Context, moduleName string, coins sdk.Coins) error {
	// Check supply cap before minting
	if err := k.checkSupplyCap(ctx, coins); err != nil {
		return err
	}

	// Mint coins
	return k.bankKeeper.MintCoins(ctx, moduleName, coins)
}

// GetCurrentSupply returns the current total supply of R3MES tokens
func (k Keeper) GetCurrentSupply(ctx context.Context) math.Int {
	supply := k.bankKeeper.GetSupply(ctx, TokenDenom)
	return supply.Amount
}

// GetRemainingMintableSupply returns how many tokens can still be minted
func (k Keeper) GetRemainingMintableSupply(ctx context.Context) math.Int {
	currentSupply := k.GetCurrentSupply(ctx)
	maxSupply := math.NewInt(MaxSupplyCap)

	if currentSupply.GTE(maxSupply) {
		return math.ZeroInt()
	}

	return maxSupply.Sub(currentSupply)
}

// IsSupplyCapReached checks if the supply cap has been reached
func (k Keeper) IsSupplyCapReached(ctx context.Context) bool {
	return k.GetRemainingMintableSupply(ctx).IsZero()
}

// CalculateGradientQualityScore calculates a quality score for a gradient based on various factors
// Factors:
// - Trust score (0.0 to 1.0): Miner's historical performance
// - Validation status: Whether gradient passed validation
// - Consistency: Model version, training round, shard assignment correctness
// Returns: Quality score from 0.0 to 1.0
func (k Keeper) CalculateGradientQualityScore(ctx context.Context, minerAddress string, gradientID uint64) (math.LegacyDec, error) {
	// Base quality score
	qualityScore := math.LegacyNewDecWithPrec(5, 1) // 0.5 base

	// 1. Trust score component (0.0 to 0.4 weight)
	contribution, err := k.MiningContributions.Get(ctx, minerAddress)
	if err == nil {
		trustScore, err := math.LegacyNewDecFromStr(contribution.TrustScore)
		if err == nil {
			// Trust score contributes 0.0 to 0.4 to quality
			trustComponent := trustScore.Mul(math.LegacyNewDecWithPrec(4, 1)) // 0.0 to 0.4
			qualityScore = qualityScore.Add(trustComponent)
		}
	}

	// 2. Validation status component (0.0 to 0.1 weight)
	// Check if gradient has been validated (no challenges or passed challenges)
	gradient, err := k.StoredGradients.Get(ctx, gradientID)
	if err == nil {
		// If gradient status is "finalized" or "accepted", add validation bonus
		if gradient.Status == "finalized" || gradient.Status == "accepted" {
			qualityScore = qualityScore.Add(math.LegacyNewDecWithPrec(1, 1)) // +0.1
		}
		// If gradient has been challenged but not rejected, it's still valid
		if gradient.Status == "challenged" {
			qualityScore = qualityScore.Add(math.LegacyNewDecWithPrec(5, 2)) // +0.05
		}
	}

	// 3. Consistency component (0.0 to 0.1 weight)
	// Check if miner's submissions are consistent (same model version, training round)
	// This is a simplified check - in production, could check historical consistency
	successRate := math.LegacyZeroDec()
	if contribution.SuccessfulSubmissions > 0 && contribution.TotalSubmissions > 0 {
		successRate = math.LegacyNewDecFromInt(math.NewIntFromUint64(contribution.SuccessfulSubmissions)).
			Quo(math.LegacyNewDecFromInt(math.NewIntFromUint64(contribution.TotalSubmissions)))
	}
	// Success rate contributes 0.0 to 0.1
	consistencyComponent := successRate.Mul(math.LegacyNewDecWithPrec(1, 1)) // 0.0 to 0.1
	qualityScore = qualityScore.Add(consistencyComponent)

	// Cap quality score at 1.0
	if qualityScore.GT(math.LegacyOneDec()) {
		qualityScore = math.LegacyOneDec()
	}

	return qualityScore, nil
}

// CalculateMinerReward calculates the reward for a miner based on quality score
// Formula: base_reward * (1.0 + quality_score)
// Quality score ranges from 0.0 to 1.0, so multiplier ranges from 1.0 to 2.0
func (k Keeper) CalculateMinerReward(ctx context.Context, minerAddress string, gradientID uint64) (sdk.Coins, error) {
	// Calculate gradient quality score
	qualityScore, err := k.CalculateGradientQualityScore(ctx, minerAddress, gradientID)
	if err != nil {
		// Fallback to default quality score
		qualityScore = math.LegacyNewDecWithPrec(5, 1) // 0.5
	}

	// Calculate reward: base_reward * (1.0 + quality_score)
	// This gives range: base_reward * 1.0 to base_reward * 2.0
	baseReward := math.NewInt(BaseMinerReward)
	qualityMultiplier := math.LegacyOneDec().Add(qualityScore) // 1.0 to 2.0 range
	rewardAmount := qualityMultiplier.MulInt(baseReward).TruncateInt()

	// Create reward coins
	reward := sdk.NewCoins(sdk.NewCoin("remes", rewardAmount))

	return reward, nil
}

// CalculateProposerReward calculates the reward for a proposer based on aggregation work
// Formula: base_reward + (participant_count * bonus)
func (k Keeper) CalculateProposerReward(ctx context.Context, participantCount uint64) (sdk.Coins, error) {
	baseReward := math.NewInt(BaseProposerReward)
	participantBonus := math.NewInt(ProposerParticipantBonus).Mul(math.NewIntFromUint64(participantCount))

	totalReward := baseReward.Add(participantBonus)

	// Create reward coins
	reward := sdk.NewCoins(sdk.NewCoin("remes", totalReward))

	return reward, nil
}

// DistributeMinerReward distributes reward to a miner
func (k Keeper) DistributeMinerReward(ctx context.Context, minerAddress string, gradientID uint64) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Check if supply cap is reached
	if k.IsSupplyCapReached(ctx) {
		sdkCtx.Logger().Warn("Supply cap reached, cannot distribute miner reward")
		return fmt.Errorf("supply cap reached, no more tokens can be minted")
	}

	// Calculate reward
	reward, err := k.CalculateMinerReward(ctx, minerAddress, gradientID)
	if err != nil {
		return fmt.Errorf("failed to calculate miner reward: %w", err)
	}

	// Convert miner address to AccAddress
	minerAddr, err := k.addressCodec.StringToBytes(minerAddress)
	if err != nil {
		return fmt.Errorf("invalid miner address: %w", err)
	}

	// Mint coins to module account WITH SUPPLY CHECK
	if err := k.mintCoinsWithSupplyCheck(ctx, RemesModuleName, reward); err != nil {
		return fmt.Errorf("failed to mint coins: %w", err)
	}

	// Send coins from module to miner
	if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, RemesModuleName, minerAddr, reward); err != nil {
		return fmt.Errorf("failed to send coins to miner: %w", err)
	}

	sdkCtx.Logger().Info(fmt.Sprintf("Distributed miner reward: %s to %s", reward.String(), minerAddress))

	return nil
}

// DistributeProposerReward distributes reward to a proposer
func (k Keeper) DistributeProposerReward(ctx context.Context, proposerAddress string, participantCount uint64) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Check if supply cap is reached
	if k.IsSupplyCapReached(ctx) {
		sdkCtx.Logger().Warn("Supply cap reached, cannot distribute proposer reward")
		return fmt.Errorf("supply cap reached, no more tokens can be minted")
	}

	// Calculate reward
	reward, err := k.CalculateProposerReward(ctx, participantCount)
	if err != nil {
		return fmt.Errorf("failed to calculate proposer reward: %w", err)
	}

	// Convert proposer address to AccAddress
	proposerAddr, err := k.addressCodec.StringToBytes(proposerAddress)
	if err != nil {
		return fmt.Errorf("invalid proposer address: %w", err)
	}

	// Mint coins to module account WITH SUPPLY CHECK
	if err := k.mintCoinsWithSupplyCheck(ctx, RemesModuleName, reward); err != nil {
		return fmt.Errorf("failed to mint coins: %w", err)
	}

	// Send coins from module to proposer
	if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, RemesModuleName, proposerAddr, reward); err != nil {
		return fmt.Errorf("failed to send coins to proposer: %w", err)
	}

	sdkCtx.Logger().Info(fmt.Sprintf("Distributed proposer reward: %s to %s", reward.String(), proposerAddress))

	return nil
}

// CalculateServingReward calculates the reward for a serving node based on performance
// Formula: base_reward * performance_score (0.0 to 1.0)
func (k Keeper) CalculateServingReward(ctx context.Context, performanceScore float64) (sdk.Coins, error) {
	baseReward := math.NewInt(BaseServingReward)
	// Convert float64 to Dec using string conversion
	performanceStr := fmt.Sprintf("%.6f", performanceScore)
	performanceMultiplier, err := math.LegacyNewDecFromStr(performanceStr)
	if err != nil {
		performanceMultiplier = math.LegacyZeroDec()
	}
	rewardAmount := performanceMultiplier.MulInt(baseReward).TruncateInt()

	// Create reward coins
	reward := sdk.NewCoins(sdk.NewCoin("remes", rewardAmount))

	return reward, nil
}

// DistributeServingReward distributes reward to a serving node
func (k Keeper) DistributeServingReward(ctx context.Context, servingAddress string, performanceScore float64) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Check if supply cap is reached
	if k.IsSupplyCapReached(ctx) {
		sdkCtx.Logger().Warn("Supply cap reached, cannot distribute serving reward")
		return fmt.Errorf("supply cap reached, no more tokens can be minted")
	}

	// Calculate reward
	reward, err := k.CalculateServingReward(ctx, performanceScore)
	if err != nil {
		return fmt.Errorf("failed to calculate serving reward: %w", err)
	}

	// Convert serving address to AccAddress
	servingAddr, err := k.addressCodec.StringToBytes(servingAddress)
	if err != nil {
		return fmt.Errorf("invalid serving address: %w", err)
	}

	// Mint coins to module account WITH SUPPLY CHECK
	if err := k.mintCoinsWithSupplyCheck(ctx, RemesModuleName, reward); err != nil {
		return fmt.Errorf("failed to mint coins: %w", err)
	}

	// Send coins from module to serving node
	if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, RemesModuleName, servingAddr, reward); err != nil {
		return fmt.Errorf("failed to send coins to serving node: %w", err)
	}

	sdkCtx.Logger().Info(fmt.Sprintf("Distributed serving reward: %s to %s", reward.String(), servingAddress))

	return nil
}

// CalculateValidatorReward calculates the reward for a validator based on verification count
// Formula: base_reward + (verification_count * bonus)
func (k Keeper) CalculateValidatorReward(ctx context.Context, verificationCount uint64) (sdk.Coins, error) {
	baseReward := math.NewInt(BaseValidatorReward)
	verificationBonus := math.NewInt(ValidatorVerificationBonus).Mul(math.NewIntFromUint64(verificationCount))

	totalReward := baseReward.Add(verificationBonus)

	// Create reward coins
	reward := sdk.NewCoins(sdk.NewCoin("remes", totalReward))

	return reward, nil
}

// DistributeValidatorReward distributes reward to a validator
func (k Keeper) DistributeValidatorReward(ctx context.Context, validatorAddress string, verificationCount uint64) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Check if supply cap is reached
	if k.IsSupplyCapReached(ctx) {
		sdkCtx.Logger().Warn("Supply cap reached, cannot distribute validator reward")
		return fmt.Errorf("supply cap reached, no more tokens can be minted")
	}

	// Calculate reward
	reward, err := k.CalculateValidatorReward(ctx, verificationCount)
	if err != nil {
		return fmt.Errorf("failed to calculate validator reward: %w", err)
	}

	// Convert validator address to AccAddress
	validatorAddr, err := k.addressCodec.StringToBytes(validatorAddress)
	if err != nil {
		return fmt.Errorf("invalid validator address: %w", err)
	}

	// Mint coins to module account WITH SUPPLY CHECK
	if err := k.mintCoinsWithSupplyCheck(ctx, RemesModuleName, reward); err != nil {
		return fmt.Errorf("failed to mint coins: %w", err)
	}

	// Send coins from module to validator
	if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, RemesModuleName, validatorAddr, reward); err != nil {
		return fmt.Errorf("failed to send coins to validator: %w", err)
	}

	sdkCtx.Logger().Info(fmt.Sprintf("Distributed validator reward: %s to %s", reward.String(), validatorAddress))

	return nil
}
