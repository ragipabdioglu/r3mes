package keeper

import (
	"fmt"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// CollectInferenceRevenue collects revenue from inference fees
// Called from serving module when inference is completed
// Treasury receives 20% of inference fees
func (k Keeper) CollectInferenceRevenue(
	ctx sdk.Context,
	totalFee sdk.Coins,
) error {
	// Calculate treasury share (20%)
	treasuryShare := sdk.NewCoins()
	for _, coin := range totalFee {
		treasuryAmount := coin.Amount.Mul(sdkmath.NewInt(20)).Quo(sdkmath.NewInt(100))
		if treasuryAmount.IsPositive() {
			treasuryShare = treasuryShare.Add(sdk.NewCoin(coin.Denom, treasuryAmount))
		}
	}

	if treasuryShare.IsZero() {
		return nil // No treasury share to collect
	}

	// Get treasury
	treasury, err := k.Treasury.Get(ctx)
	if err != nil {
		// Initialize treasury if not exists
		treasury = types.Treasury{
			Balance:           treasuryShare.String(),
			BuyBackThreshold:  "1000000remes", // Default: 1M tokens
			BurnFraction:      "1.0",          // Default: 100% burn
			LastBuyBackHeight: 0,
			BuyBackInterval:   10000, // Default: 10000 blocks (~14 hours at 5s/block)
			TotalBurned:       "0remes",
		}
	} else {
		// Add to existing balance
		currentBalance, err := sdk.ParseCoinsNormalized(treasury.Balance)
		if err != nil {
			return errorsmod.Wrap(err, "failed to parse treasury balance")
		}
		newBalance := currentBalance.Add(treasuryShare...)
		treasury.Balance = newBalance.String()
	}

	// Store treasury
	if err := k.Treasury.Set(ctx, treasury); err != nil {
		return errorsmod.Wrap(err, "failed to store treasury")
	}

	// Transfer coins to module account (treasury)
	// Get the sender account (this should be the serving module or fee collector)
	// In production, the serving module would call this function with the fee collector account
	// For now, we assume coins are already in the module account or will be transferred by the caller
	// If coins need to be transferred, use:
	// moduleAddr := sdk.AccAddress(k.GetAuthority()) // or use authtypes.NewModuleAddress(types.ModuleName)
	// if err := k.bankKeeper.SendCoinsFromAccountToModule(ctx, feeCollectorAddr, types.ModuleName, treasuryShare); err != nil {
	//     return errorsmod.Wrap(err, "failed to transfer coins to treasury module account")
	// }
	
	// Note: In practice, the coins should already be in the module account when this function is called
	// This function just tracks the treasury share in state. The actual coin transfer should happen
	// before calling this function (e.g., in the serving module's fee collection logic).

	return nil
}

// ProcessTreasuryBuyBack processes buy-back & burn in EndBlocker
func (k Keeper) ProcessTreasuryBuyBack(ctx sdk.Context) error {
	// Get treasury
	treasury, err := k.Treasury.Get(ctx)
	if err != nil {
		// Treasury not initialized yet - nothing to do
		return nil
	}

	// Check if enough time has passed since last buy-back
	currentHeight := ctx.BlockHeight()
	if treasury.LastBuyBackHeight > 0 {
		blocksSinceLastBuyBack := currentHeight - treasury.LastBuyBackHeight
		if blocksSinceLastBuyBack < treasury.BuyBackInterval {
			// Not enough time has passed
			return nil
		}
	}

	// Parse treasury balance
	balance, err := sdk.ParseCoinsNormalized(treasury.Balance)
	if err != nil {
		return errorsmod.Wrap(err, "failed to parse treasury balance")
	}

	if balance.IsZero() {
		// No balance to buy back
		return nil
	}

	// Parse buy-back threshold
	threshold, err := sdk.ParseCoinsNormalized(treasury.BuyBackThreshold)
	if err != nil {
		return errorsmod.Wrap(err, "failed to parse buy-back threshold")
	}

	// Check if balance exceeds threshold
	shouldBuyBack := false
	for _, coin := range balance {
		thresholdCoin := threshold.AmountOf(coin.Denom)
		if thresholdCoin.IsPositive() && coin.Amount.GTE(thresholdCoin) {
			shouldBuyBack = true
			break
		}
	}

	if !shouldBuyBack {
		// Balance below threshold
		return nil
	}

	// Parse burn fraction
	burnFraction, err := sdkmath.LegacyNewDecFromStr(treasury.BurnFraction)
	if err != nil {
		return errorsmod.Wrap(err, "failed to parse burn fraction")
	}

	// Calculate amount to burn
	burnAmount := sdk.NewCoins()
	for _, coin := range balance {
		// Use entire balance for buy-back (in production, this would be market buy)
		burnCoinAmount := burnFraction.MulInt(coin.Amount).TruncateInt()
		if burnCoinAmount.IsPositive() {
			burnAmount = burnAmount.Add(sdk.NewCoin(coin.Denom, burnCoinAmount))
		}
	}

	if burnAmount.IsZero() {
		return nil
	}

	// Burn tokens
	// In production, this would:
	// 1. Buy tokens from market using treasury balance
	// 2. Burn the purchased tokens
	// For now, we directly burn from treasury balance
	if err := k.bankKeeper.BurnCoins(ctx, types.ModuleName, burnAmount); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to burn treasury tokens: %v", err))
		// Don't fail - burning is not critical for block processing
		return nil
	}

	// Update treasury balance (subtract burned amount)
	newBalance := balance.Sub(burnAmount...)
	treasury.Balance = newBalance.String()
	treasury.LastBuyBackHeight = currentHeight

	// Update total burned
	totalBurned, err := sdk.ParseCoinsNormalized(treasury.TotalBurned)
	if err != nil {
		// Initialize if not set
		totalBurned = sdk.NewCoins()
	}
	totalBurned = totalBurned.Add(burnAmount...)
	treasury.TotalBurned = totalBurned.String()

	// Store updated treasury
	if err := k.Treasury.Set(ctx, treasury); err != nil {
		return errorsmod.Wrap(err, "failed to update treasury")
	}

	// Emit event
	ctx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeTreasuryBuyBack,
			sdk.NewAttribute(types.AttributeKeyBurnAmount, burnAmount.String()),
			sdk.NewAttribute(types.AttributeKeyTotalBurned, totalBurned.String()),
			sdk.NewAttribute(types.AttributeKeyTreasuryBalance, newBalance.String()),
		),
	)

	return nil
}

// GetTreasuryBalance returns the current treasury balance
func (k Keeper) GetTreasuryBalance(ctx sdk.Context) (sdk.Coins, error) {
	treasury, err := k.Treasury.Get(ctx)
	if err != nil {
		return sdk.NewCoins(), nil // Treasury not initialized
	}

	balance, err := sdk.ParseCoinsNormalized(treasury.Balance)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to parse treasury balance")
	}

	return balance, nil
}

