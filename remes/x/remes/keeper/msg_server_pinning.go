package keeper

import (
	"context"
	"fmt"
	"strings"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"
	errorsmod "cosmossdk.io/errors"
	"remes/x/remes/types"
)

// CommitPinning handles MsgCommitPinning
// Allows a node to commit to pinning IPFS content with a stake
func (k msgServer) CommitPinning(ctx context.Context, msg *types.MsgCommitPinning) (*types.MsgCommitPinningResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate node address
	nodeAddr, err := k.addressCodec.StringToBytes(msg.NodeAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid node address: %s", msg.NodeAddress)
	}
	_ = nodeAddr // Address validated

	// 2. Validate IPFS hash
	ipfsHash := strings.TrimSpace(msg.IpfsHash)
	if ipfsHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "IPFS hash cannot be empty")
	}
	if !strings.HasPrefix(ipfsHash, "Qm") && !strings.HasPrefix(ipfsHash, "bafy") && !strings.HasPrefix(ipfsHash, "bafk") {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", ipfsHash)
	}

	// 3. Verify IPFS content exists
	if err := k.VerifyIPFSContentExists(sdkCtx, ipfsHash); err != nil {
		return nil, errorsmod.Wrap(err, "IPFS content verification failed")
	}

	// 4. Parse and validate stake amount
	stake, err := sdk.ParseCoinsNormalized(msg.Stake)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidRequest, "invalid stake amount: %s", msg.Stake)
	}
	if stake.IsZero() {
		return nil, errorsmod.Wrapf(types.ErrInvalidRequest, "stake amount cannot be zero")
	}

	// 5. Check if node has sufficient balance
	nodeAcc, err := sdk.AccAddressFromBech32(msg.NodeAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid node address: %v", err)
	}
	balance := k.bankKeeper.SpendableCoins(sdkCtx, nodeAcc)
	if !balance.IsAllGTE(stake) {
		return nil, errorsmod.Wrapf(
			types.ErrInsufficientStake,
			"insufficient balance: required %s, available %s",
			stake,
			balance,
		)
	}

	// 6. Check if pinning commitment already exists
	pinningKey := fmt.Sprintf("%s|%s", msg.NodeAddress, ipfsHash)
	existing, err := k.PinningIncentives.Get(ctx, pinningKey)
	if err == nil && existing.Status == "active" {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidRequest,
			"pinning commitment already exists for node %s and IPFS hash %s",
			msg.NodeAddress,
			ipfsHash,
		)
	}

	// 7. Calculate commitment end time (if duration specified)
	commitmentStart := sdkCtx.BlockTime()
	var commitmentEnd *time.Time
	if msg.CommitmentDurationBlocks > 0 {
		// Calculate end time based on block time (assume 6s per block)
		avgBlockTime := k.GetAverageBlockTime(sdkCtx)
		endTime := commitmentStart.Add(time.Duration(msg.CommitmentDurationBlocks) * avgBlockTime)
		commitmentEnd = &endTime
	}

	// 8. Calculate reward rate (base reward per block)
	// Reward formula: 0.1% of stake per block (can be made configurable via params)
	baseRewardRate := stake[0].Amount.Mul(sdkmath.NewInt(1)).Quo(sdkmath.NewInt(1000)) // 0.1% per block
	rewardRate := sdk.NewCoin(stake[0].Denom, baseRewardRate)

	// 9. Escrow stake to module account
	if err := k.bankKeeper.SendCoinsFromAccountToModule(sdkCtx, nodeAcc, types.ModuleName, stake); err != nil {
		return nil, errorsmod.Wrap(err, "failed to escrow stake")
	}

	// 10. Create pinning incentive record
	pinningIncentive := types.PinningIncentive{
		NodeAddress:      msg.NodeAddress,
		IpfsHash:         ipfsHash,
		Stake:            stake.String(),
		RewardRate:       rewardRate.String(),
		CommitmentStart:  commitmentStart,
		CommitmentEnd:    commitmentEnd,
		Status:           "active",
		ChallengeCount:   0,
		TotalRewards:     "0",
	}

	if err := k.PinningIncentives.Set(ctx, pinningKey, pinningIncentive); err != nil {
		// Refund stake on error
		_ = k.bankKeeper.SendCoinsFromModuleToAccount(sdkCtx, types.ModuleName, nodeAcc, stake)
		return nil, errorsmod.Wrap(err, "failed to store pinning incentive")
	}

	// 11. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeCommitPinning,
			sdk.NewAttribute(types.AttributeKeyNodeAddress, msg.NodeAddress),
			sdk.NewAttribute(types.AttributeKeyIPFSHash, ipfsHash),
			sdk.NewAttribute("stake", stake.String()),
			sdk.NewAttribute("reward_rate", rewardRate.String()),
		),
	)

	return &types.MsgCommitPinningResponse{
		PinningId: pinningKey,
	}, nil
}

// ChallengePinning handles MsgChallengePinning
// Allows validators to challenge a pinning commitment to verify data availability
func (k msgServer) ChallengePinning(ctx context.Context, msg *types.MsgChallengePinning) (*types.MsgChallengePinningResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate challenger address
	challengerAddr, err := k.addressCodec.StringToBytes(msg.Challenger)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid challenger address: %s", msg.Challenger)
	}
	_ = challengerAddr // Address validated

	// 2. Validate node address
	nodeAddr, err := k.addressCodec.StringToBytes(msg.NodeAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid node address: %s", msg.NodeAddress)
	}
	_ = nodeAddr // Address validated

	// 3. Validate IPFS hash
	ipfsHash := strings.TrimSpace(msg.IpfsHash)
	if ipfsHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "IPFS hash cannot be empty")
	}

	// 4. Verify pinning commitment exists
	pinningKey := fmt.Sprintf("%s|%s", msg.NodeAddress, ipfsHash)
	pinningIncentive, err := k.PinningIncentives.Get(ctx, pinningKey)
	if err != nil {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidRequest,
			"pinning commitment not found for node %s and IPFS hash %s",
			msg.NodeAddress,
			ipfsHash,
		)
	}

	// 5. Verify commitment is active
	if pinningIncentive.Status != "active" {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidRequest,
			"pinning commitment is not active (status: %s)",
			pinningIncentive.Status,
		)
	}

	// 6. Calculate response deadline based on dynamic timeout
	networkLoad, err := k.GetNetworkLoad(sdkCtx)
	if err != nil {
		networkLoad = 0.0
	}
	avgBlockTime := k.GetAverageBlockTime(sdkCtx)
	timeoutBlocks := k.CalculateDATimeout(sdkCtx, networkLoad, avgBlockTime)
	responseDeadline := sdkCtx.BlockTime().Add(time.Duration(timeoutBlocks) * avgBlockTime)

	// 8. Generate challenge ID
	challengeID, err := k.DataAvailabilityChallengeID.Next(ctx)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to generate challenge ID")
	}

	// 9. Create data availability challenge
	challenge := types.DataAvailabilityChallenge{
		ChallengeId:      challengeID,
		Challenger:       msg.Challenger,
		NodeAddress:      msg.NodeAddress,
		IpfsHash:         ipfsHash,
		ChallengeTime:    sdkCtx.BlockTime(),
		ResponseDeadline: responseDeadline,
		Status:           "pending",
	}

	if err := k.DataAvailabilityChallenges.Set(ctx, challengeID, challenge); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store challenge")
	}

	// 10. Update pinning incentive challenge count
	pinningIncentive.ChallengeCount++
	if err := k.PinningIncentives.Set(ctx, pinningKey, pinningIncentive); err != nil {
		sdkCtx.Logger().Error("Failed to update pinning incentive challenge count", "error", err)
	}

	// 11. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeChallengePinning,
			sdk.NewAttribute(types.AttributeKeyChallengeID, fmt.Sprintf("%d", challengeID)),
			sdk.NewAttribute(types.AttributeKeyChallenger, msg.Challenger),
			sdk.NewAttribute(types.AttributeKeyNodeAddress, msg.NodeAddress),
			sdk.NewAttribute(types.AttributeKeyIPFSHash, ipfsHash),
		),
	)

	return &types.MsgChallengePinningResponse{
		ChallengeId: challengeID,
	}, nil
}
