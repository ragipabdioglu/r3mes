package keeper

import (
	"context"
	"fmt"

	errorsmod "cosmossdk.io/errors"
	sdkmath "cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// SlashNode slashes a node's stake for violations
// FIXED: Ensures consistency between registration.Stake and actual balance
func (k Keeper) SlashNode(
	ctx context.Context,
	nodeAddress string,
	slashFraction sdkmath.LegacyDec, // e.g., 0.05 for 5%, 1.0 for 100%
	reason string,
) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get node registration
	registration, err := k.NodeRegistrations.Get(sdkCtx, nodeAddress)
	if err != nil {
		return errorsmod.Wrap(err, "node not registered")
	}

	// Parse registered stake
	registeredStake, err := sdk.ParseCoinsNormalized(registration.Stake)
	if err != nil {
		return errorsmod.Wrap(err, "invalid stake format")
	}

	if registeredStake.IsZero() {
		return errorsmod.Wrap(err, "node has no stake to slash")
	}

	// Convert node address to AccAddress
	nodeAddr, err := sdk.AccAddressFromBech32(nodeAddress)
	if err != nil {
		return errorsmod.Wrap(err, "invalid node address")
	}

	// Get node's actual balance
	actualBalance := k.bankKeeper.SpendableCoins(ctx, nodeAddr)

	// CRITICAL FIX: Synchronize stake with actual balance before slashing
	// If registered stake > actual balance, update registration first
	for _, coin := range registeredStake {
		actualAmount := actualBalance.AmountOf(coin.Denom)
		if coin.Amount.GT(actualAmount) {
			sdkCtx.Logger().Warn(fmt.Sprintf(
				"Node %s has stake mismatch: registered=%s, actual=%s. Syncing to actual balance.",
				nodeAddress, coin.String(), actualAmount.String(),
			))
		}
	}

	// Calculate slash amount based on MINIMUM of registered stake and actual balance
	// This prevents slashing more than what's actually available
	effectiveStake := sdk.NewCoins()
	for _, coin := range registeredStake {
		actualAmount := actualBalance.AmountOf(coin.Denom)
		effectiveAmount := sdkmath.MinInt(coin.Amount, actualAmount)
		if !effectiveAmount.IsZero() {
			effectiveStake = effectiveStake.Add(sdk.NewCoin(coin.Denom, effectiveAmount))
		}
	}

	if effectiveStake.IsZero() {
		// No effective stake to slash - update registration to reflect reality
		registration.Stake = sdk.NewCoins().String()
		registration.SlashingEvents++
		if err := k.NodeRegistrations.Set(sdkCtx, nodeAddress, registration); err != nil {
			return err
		}

		sdkCtx.Logger().Warn(fmt.Sprintf(
			"Node %s has no effective stake to slash (registered: %s, actual: %s)",
			nodeAddress, registeredStake.String(), actualBalance.String(),
		))
		return nil
	}

	// Calculate slash amount from effective stake
	slashAmount := sdk.NewCoins()
	for _, coin := range effectiveStake {
		slashValue := slashFraction.MulInt(coin.Amount).TruncateInt()
		if !slashValue.IsZero() {
			slashAmount = slashAmount.Add(sdk.NewCoin(coin.Denom, slashValue))
		}
	}

	if slashAmount.IsZero() {
		return nil // Nothing to slash
	}

	// Send coins from node account to module (for burning or distribution)
	if err := k.bankKeeper.SendCoinsFromAccountToModule(ctx, nodeAddr, types.ModuleName, slashAmount); err != nil {
		return errorsmod.Wrap(err, "failed to transfer slashed coins")
	}

	// Burn the slashed coins (or send to community pool - for now, burn)
	if err := k.bankKeeper.BurnCoins(ctx, types.ModuleName, slashAmount); err != nil {
		// If burning fails, coins remain in module account (can be recovered)
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to burn slashed coins: %v", err))
	}

	// CRITICAL FIX: Update registration stake to reflect actual remaining balance
	// Query balance AFTER slashing to ensure consistency
	newActualBalance := k.bankKeeper.SpendableCoins(ctx, nodeAddr)
	newStake := sdk.NewCoins()
	for _, coin := range registeredStake {
		newAmount := newActualBalance.AmountOf(coin.Denom)
		if !newAmount.IsZero() {
			newStake = newStake.Add(sdk.NewCoin(coin.Denom, newAmount))
		}
	}
	registration.Stake = newStake.String()

	// Update slashing events
	registration.SlashingEvents++

	if err := k.NodeRegistrations.Set(sdkCtx, nodeAddress, registration); err != nil {
		return err
	}

	// Emit slashing event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeNodeSlash,
			sdk.NewAttribute(types.AttributeKeyNodeAddress, nodeAddress),
			sdk.NewAttribute(types.AttributeKeySlashAmount, slashAmount.String()),
			sdk.NewAttribute(types.AttributeKeySlashFraction, slashFraction.String()),
			sdk.NewAttribute(types.AttributeKeyReason, reason),
		),
	)

	sdkCtx.Logger().Info(fmt.Sprintf("Node slashed: %s, amount: %s, reason: %s", nodeAddress, slashAmount.String(), reason))

	return nil
}

// SlashMiner slashes a miner's stake for gradient-related violations
func (k Keeper) SlashMiner(
	ctx context.Context,
	minerAddress string,
	slashFraction sdkmath.LegacyDec,
	reason string,
) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get miner contribution
	contribution, err := k.MiningContributions.Get(sdkCtx, minerAddress)
	if err != nil {
		// Miner not found, but still try to slash from balance
		return k.SlashNode(ctx, minerAddress, slashFraction, reason)
	}

	// Reset trust score to 0.0 on slashing
	contribution.TrustScore = "0.0"
	contribution.SlashingEvents++

	// Update contribution
	if err := k.MiningContributions.Set(sdkCtx, minerAddress, contribution); err != nil {
		return err
	}

	// Slash from node registration if exists
	_, err = k.NodeRegistrations.Get(sdkCtx, minerAddress)
	if err == nil {
		// Node is registered, slash from registration
		return k.SlashNode(ctx, minerAddress, slashFraction, reason)
	}

	// If not registered as node, slash from account balance directly
	minerAddr, err := sdk.AccAddressFromBech32(minerAddress)
	if err != nil {
		return errorsmod.Wrap(err, "invalid miner address")
	}

	balance := k.bankKeeper.SpendableCoins(ctx, minerAddr)
	if balance.IsZero() {
		return nil // Nothing to slash
	}

	// Calculate slash amount from balance
	slashAmount := sdk.NewCoins()
	for _, coin := range balance {
		slashValue := slashFraction.MulInt(coin.Amount).TruncateInt()
		if !slashValue.IsZero() {
			slashAmount = slashAmount.Add(sdk.NewCoin(coin.Denom, slashValue))
		}
	}

	if slashAmount.IsZero() {
		return nil
	}

	// Transfer and burn
	if err := k.bankKeeper.SendCoinsFromAccountToModule(ctx, minerAddr, types.ModuleName, slashAmount); err != nil {
		return errorsmod.Wrap(err, "failed to transfer slashed coins")
	}

	if err := k.bankKeeper.BurnCoins(ctx, types.ModuleName, slashAmount); err != nil {
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to burn slashed coins: %v", err))
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeMinerSlash,
			sdk.NewAttribute(types.AttributeKeyMiner, minerAddress),
			sdk.NewAttribute(types.AttributeKeySlashAmount, slashAmount.String()),
			sdk.NewAttribute(types.AttributeKeyReason, reason),
		),
	)

	return nil
}
