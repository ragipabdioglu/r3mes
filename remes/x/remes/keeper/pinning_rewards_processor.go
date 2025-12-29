package keeper

import (
	"context"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// ProcessPinningRewards processes rewards for all active pinning commitments
// Called every block in end_blocker.go
func (k Keeper) ProcessPinningRewards(ctx context.Context) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	currentTime := sdkCtx.BlockTime()

	// Iterate over all pinning incentives
	err := k.PinningIncentives.Walk(ctx, nil, func(key string, pinningIncentive types.PinningIncentive) (stop bool, err error) {
		// Only process active commitments
		if pinningIncentive.Status != "active" {
			return false, nil
		}

		// Check if commitment has expired
		if pinningIncentive.CommitmentEnd != nil && currentTime.After(*pinningIncentive.CommitmentEnd) {
			// Commitment expired - mark as completed and refund stake
			pinningIncentive.Status = "completed"
			if err := k.PinningIncentives.Set(ctx, key, pinningIncentive); err != nil {
				sdkCtx.Logger().Error("Failed to update expired pinning incentive", "error", err, "key", key)
				return false, nil
			}

			// Refund stake
			stake, err := sdk.ParseCoinsNormalized(pinningIncentive.Stake)
			if err == nil && !stake.IsZero() {
				nodeAddr, err := sdk.AccAddressFromBech32(pinningIncentive.NodeAddress)
				if err == nil {
					if refundErr := k.bankKeeper.SendCoinsFromModuleToAccount(sdkCtx, types.ModuleName, nodeAddr, stake); refundErr != nil {
						sdkCtx.Logger().Error("Failed to refund stake for expired commitment", "error", refundErr, "key", key)
					}
				}
			}
			return false, nil
		}

		// Calculate reward for this block
		rewardRate, err := sdk.ParseCoinsNormalized(pinningIncentive.RewardRate)
		if err != nil || rewardRate.IsZero() {
			// Skip if reward rate is invalid
			return false, nil
		}

		// Check if node has responded to all challenges successfully
		// If there are pending challenges that haven't been responded to, skip reward
		hasFailedChallenge := false
		challengeIter, err := k.DataAvailabilityChallenges.Iterate(ctx, nil)
		if err == nil {
			defer challengeIter.Close()
			for ; challengeIter.Valid(); challengeIter.Next() {
				challenge, err := challengeIter.Value()
				if err != nil {
					continue
				}
				// Check if this challenge is for this pinning commitment
				if challenge.NodeAddress == pinningIncentive.NodeAddress && challenge.IpfsHash == pinningIncentive.IpfsHash {
					if challenge.Status == "failed" {
						// Node failed to respond - slash stake and mark as slashed
						hasFailedChallenge = true
						break
					}
					if challenge.Status == "pending" && currentTime.After(challenge.ResponseDeadline) {
						// Challenge expired without response - mark as failed
						challenge.Status = "failed"
						resolutionTime := currentTime
						challenge.ResolutionTime = &resolutionTime
						if err := k.DataAvailabilityChallenges.Set(ctx, challenge.ChallengeId, challenge); err != nil {
							sdkCtx.Logger().Error("Failed to update expired challenge", "error", err)
						}
						hasFailedChallenge = true
						break
					}
				}
			}
		}

		if hasFailedChallenge {
			// Slash stake and mark as slashed
			pinningIncentive.Status = "slashed"
			if err := k.PinningIncentives.Set(ctx, key, pinningIncentive); err != nil {
				sdkCtx.Logger().Error("Failed to update slashed pinning incentive", "error", err, "key", key)
				return false, nil
			}

			// Stake is already in module account, so slashing means it stays there
			// (burned or sent to treasury - can be configured)
			sdkCtx.Logger().Info("Pinning commitment slashed due to failed challenge",
				"node", pinningIncentive.NodeAddress,
				"ipfs_hash", pinningIncentive.IpfsHash,
			)
			return false, nil
		}

		// Distribute reward for this block
		nodeAddr, err := sdk.AccAddressFromBech32(pinningIncentive.NodeAddress)
		if err != nil {
			sdkCtx.Logger().Error("Invalid node address in pinning incentive", "error", err, "key", key)
			return false, nil
		}

		// Mint and send reward
		if err := k.bankKeeper.MintCoins(sdkCtx, types.ModuleName, rewardRate); err != nil {
			sdkCtx.Logger().Error("Failed to mint pinning reward", "error", err, "key", key)
			return false, nil
		}

		if err := k.bankKeeper.SendCoinsFromModuleToAccount(sdkCtx, types.ModuleName, nodeAddr, rewardRate); err != nil {
			sdkCtx.Logger().Error("Failed to send pinning reward", "error", err, "key", key)
			// Try to burn the minted coins if send failed
			_ = k.bankKeeper.BurnCoins(sdkCtx, types.ModuleName, rewardRate)
			return false, nil
		}

		// Update total rewards
		currentTotal, err := sdk.ParseCoinsNormalized(pinningIncentive.TotalRewards)
		if err != nil {
			currentTotal = sdk.NewCoins()
		}
		newTotal := currentTotal.Add(rewardRate...)
		pinningIncentive.TotalRewards = newTotal.String()

		// Update last verification time
		pinningIncentive.LastVerification = &currentTime

		if err := k.PinningIncentives.Set(ctx, key, pinningIncentive); err != nil {
			sdkCtx.Logger().Error("Failed to update pinning incentive rewards", "error", err, "key", key)
			return false, nil
		}

		return false, nil
	})

	if err != nil {
		return fmt.Errorf("failed to process pinning rewards: %w", err)
	}

	return nil
}

