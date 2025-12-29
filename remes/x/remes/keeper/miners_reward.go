package keeper

import (
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// DistributeMinersInferenceReward distributes inference fees to miners
// based on their mining contributions (proportional to trust score and submissions)
func (k Keeper) DistributeMinersInferenceReward(
	ctx sdk.Context,
	rewardPool sdk.Coins,
) error {
	if rewardPool.IsZero() {
		return nil
	}

	// Get all mining contributions
	totalWeight := sdkmath.ZeroInt()
	minerWeights := make(map[string]sdkmath.Int)

	// Iterate through all mining contributions
	iter, err := k.MiningContributions.Iterate(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to iterate mining contributions: %w", err)
	}
	defer iter.Close()

	for ; iter.Valid(); iter.Next() {
		minerAddr, err := iter.Key()
		if err != nil {
			continue
		}

		contribution, err := iter.Value()
		if err != nil {
			continue
		}

		// Calculate weight based on trust score and successful submissions
		// Weight = trust_score * successful_submissions
		trustScore, err := sdkmath.LegacyNewDecFromStr(contribution.TrustScore)
		if err != nil {
			// Skip if trust score is invalid
			continue
		}

		// Only count miners with positive trust score and successful submissions
		if trustScore.IsPositive() && contribution.SuccessfulSubmissions > 0 {
			weight := trustScore.MulInt(sdkmath.NewIntFromUint64(contribution.SuccessfulSubmissions)).TruncateInt()
			if weight.IsPositive() {
				minerWeights[minerAddr] = weight
				totalWeight = totalWeight.Add(weight)
			}
		}
	}

	if totalWeight.IsZero() {
		// No miners to reward - return pool to treasury
		return k.CollectInferenceRevenue(ctx, rewardPool)
	}

	// Distribute rewards proportionally
	distributedCount := 0
	for minerAddr, weight := range minerWeights {
		// Calculate miner's share
		minerReward := sdk.NewCoins()
		for _, coin := range rewardPool {
			// Share = (weight / totalWeight) * coin.Amount
			share := weight.Mul(coin.Amount).Quo(totalWeight)
			if share.IsPositive() {
				minerReward = minerReward.Add(sdk.NewCoin(coin.Denom, share))
			}
		}

		if !minerReward.IsZero() {
			// Parse miner address
			minerAddrBytes, err := k.addressCodec.StringToBytes(minerAddr)
			if err != nil {
				ctx.Logger().Error(fmt.Sprintf("Failed to parse miner address %s: %v", minerAddr, err))
				continue
			}

			// Send reward to miner
			if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, minerAddrBytes, minerReward); err != nil {
				ctx.Logger().Error(fmt.Sprintf("Failed to send miner reward to %s: %v", minerAddr, err))
				continue
			}

			distributedCount++

			// Get contribution for trust score (already retrieved above)
			contribution, _ := k.MiningContributions.Get(ctx, minerAddr)
			trustScoreStr := "0.0"
			if contribution.TrustScore != "" {
				trustScoreStr = contribution.TrustScore
			}

			// Emit event
			ctx.EventManager().EmitEvent(
				sdk.NewEvent(
					types.EventTypeMinerInferenceReward,
					sdk.NewAttribute(types.AttributeKeyMiner, minerAddr),
					sdk.NewAttribute(types.AttributeKeyRewardAmount, minerReward.String()),
					sdk.NewAttribute("trust_score", trustScoreStr),
				),
			)
		}
	}

	if distributedCount > 0 {
		ctx.Logger().Info(fmt.Sprintf("Distributed inference rewards to %d miners", distributedCount))
	}

	return nil
}

