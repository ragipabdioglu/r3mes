package keeper

import (
	"context"
	"fmt"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// TrackPinnedContent tracks content that has been pinned by a node
func (k Keeper) TrackPinnedContent(ctx context.Context, pinner string, cid string, sizeBytes uint64) error {
	// Store pin information
	pin := types.PinInfo{
		Pinner:     pinner,
		Cid:        cid,
		SizeBytes:  sizeBytes,
		PinnedAt:   time.Now().Unix(),
		UnpinnedAt: 0, // Still pinned
	}

	store := k.storeService.OpenKVStore(ctx)
	key := []byte(fmt.Sprintf("pin:%s:%s", pinner, cid))
	
	// Serialize and store
	bz, err := k.cdc.Marshal(&pin)
	if err != nil {
		return err
	}

	return store.Set(key, bz)
}

// CalculatePinningRewards calculates rewards for pinning content
func (k Keeper) CalculatePinningRewards(ctx context.Context, pinner string, duration time.Duration) (sdk.Coin, error) {
	// Get all pins by this pinner
	pins, err := k.GetPinsByPinner(ctx, pinner)
	if err != nil {
		return sdk.Coin{}, err
	}

	// Calculate total size pinned
	var totalSizeBytes uint64
	for _, pin := range pins {
		// Only count pins that are still active (not unpinned)
		if pin.UnpinnedAt == 0 {
			totalSizeBytes += pin.SizeBytes
		}
	}

	// Reward formula: base reward per GB per day
	baseRewardPerGBPerDay := sdkmath.NewInt(1000000) // 1 REMES per GB per day (in uremes)
	gbPinned := sdkmath.NewIntFromUint64(totalSizeBytes / (1024 * 1024 * 1024))
	daysPinned := sdkmath.NewInt(int64(duration.Hours() / 24))
	
	reward := baseRewardPerGBPerDay.Mul(gbPinned).Mul(daysPinned)
	
	return sdk.NewCoin("uremes", reward), nil
}

// GetPinsByPinner gets all pins by a specific pinner
func (k Keeper) GetPinsByPinner(ctx context.Context, pinner string) ([]types.PinInfo, error) {
	store := k.storeService.OpenKVStore(ctx)
	
	var pins []types.PinInfo
	prefix := []byte(fmt.Sprintf("pin:%s:", pinner))
	
	// Calculate prefix end bytes (increment last byte, or append 0xFF if needed)
	prefixEnd := make([]byte, len(prefix))
	copy(prefixEnd, prefix)
	for i := len(prefixEnd) - 1; i >= 0; i-- {
		if prefixEnd[i] < 0xFF {
			prefixEnd[i]++
			prefixEnd = prefixEnd[:i+1]
			break
		}
	}
	if len(prefixEnd) == 0 || (len(prefixEnd) == len(prefix) && prefixEnd[len(prefixEnd)-1] == 0xFF) {
		// All bytes are 0xFF, use nil to iterate all keys with this prefix
		prefixEnd = nil
	}
	
	// Iterate over all pins for this pinner
	iterator, err := store.Iterator(prefix, prefixEnd)
	if err != nil {
		return nil, err
	}
	defer iterator.Close()

	for ; iterator.Valid(); iterator.Next() {
		var pin types.PinInfo
		if err := k.cdc.Unmarshal(iterator.Value(), &pin); err != nil {
			continue
		}
		pins = append(pins, pin)
	}

	return pins, nil
}

// DistributePinningRewards distributes rewards to pinners
func (k Keeper) DistributePinningRewards(ctx context.Context, pinner string) error {
	// Calculate rewards
	// In production, this would be called periodically (e.g., daily)
	duration := 24 * time.Hour // Example: 1 day
	reward, err := k.CalculatePinningRewards(ctx, pinner, duration)
	if err != nil {
		return err
	}

	// Distribute reward
	pinnerAddr, err := sdk.AccAddressFromBech32(pinner)
	if err != nil {
		return err
	}

	// Send reward from module account to pinner
	return k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, pinnerAddr, sdk.NewCoins(reward))
}
