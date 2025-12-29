package keeper

import (
	"context"
	"fmt"
	"math/rand"
	"sort"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// InitializeGenesisVault initializes the genesis vault with initial trap entries
// This should be called during genesis or by an off-chain script
// The 5000 initial trap entries should be created off-chain and loaded via genesis transaction
func (k Keeper) InitializeGenesisVault(ctx context.Context, entries []types.GenesisVaultEntry) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	
	// Load entries into vault
	for _, entry := range entries {
		if err := k.GenesisVault.Set(ctx, entry.EntryId, entry); err != nil {
			return errorsmod.Wrapf(err, "failed to initialize vault entry %d", entry.EntryId)
		}
	}
	
	sdkCtx.Logger().Info(fmt.Sprintf("Initialized genesis vault with %d entries", len(entries)))
	return nil
}

// AddToVault adds a new entry to the genesis vault (for proof of reuse)
func (k Keeper) AddToVault(ctx context.Context, entry types.GenesisVaultEntry) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	
	// Validate fingerprint JSON format
	if _, err := entry.GetFingerprint(); err != nil {
		return errorsmod.Wrap(err, "invalid fingerprint format")
	}
	
	// Encrypt entry if encryption is enabled (optional)
	if entry.Encrypted {
		encryptedEntry, err := k.EncryptVaultEntry(entry)
		if err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to encrypt vault entry: %v", err))
			// Continue without encryption if encryption fails
		} else {
			entry = encryptedEntry
		}
	}
	
	// Set entry
	if err := k.GenesisVault.Set(ctx, entry.EntryId, entry); err != nil {
		return errorsmod.Wrapf(err, "failed to add vault entry %d", entry.EntryId)
	}
	
	sdkCtx.Logger().Info(fmt.Sprintf("Added entry %d to genesis vault", entry.EntryId))
	return nil
}

// GetVaultEntry retrieves a vault entry by ID
// Automatically decrypts if entry is encrypted
func (k Keeper) GetVaultEntry(ctx context.Context, entryID uint64) (types.GenesisVaultEntry, error) {
	entry, err := k.GenesisVault.Get(ctx, entryID)
	if err != nil {
		return types.GenesisVaultEntry{}, errorsmod.Wrapf(err, "vault entry %d not found", entryID)
	}
	
	// Decrypt if encrypted
	if entry.Encrypted {
		decryptedEntry, err := k.DecryptVaultEntry(entry)
		if err != nil {
			return types.GenesisVaultEntry{}, errorsmod.Wrapf(err, "failed to decrypt vault entry %d", entryID)
		}
		entry = decryptedEntry
	}
	
	return entry, nil
}

// SelectRandomTraps selects random trap entries from the vault
// Uses weighted selection based on usage count (prefer entries with lower usage)
func (k Keeper) SelectRandomTraps(ctx context.Context, count uint64) ([]types.GenesisVaultEntry, error) {
	// Get all vault entries
	var allEntries []types.GenesisVaultEntry
	
	err := k.GenesisVault.Walk(ctx, nil, func(key uint64, value types.GenesisVaultEntry) (stop bool, err error) {
		allEntries = append(allEntries, value)
		return false, nil
	})
	
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to iterate vault entries")
	}
	
	if len(allEntries) == 0 {
		return []types.GenesisVaultEntry{}, nil
	}
	
	// Weight entries by inverse usage count (entries with lower usage are preferred)
	type weightedEntry struct {
		entry  types.GenesisVaultEntry
		weight float64
	}
	
	weightedEntries := make([]weightedEntry, len(allEntries))
	for i, entry := range allEntries {
		// Weight = 1 / (usage_count + 1) - lower usage = higher weight
		weight := 1.0 / float64(entry.UsageCount+1)
		weightedEntries[i] = weightedEntry{
			entry:  entry,
			weight: weight,
		}
	}
	
	// Sort by weight (descending)
	sort.Slice(weightedEntries, func(i, j int) bool {
		return weightedEntries[i].weight > weightedEntries[j].weight
	})
	
	// Select random entries using weighted selection
	selected := make([]types.GenesisVaultEntry, 0, count)
	rng := rand.New(rand.NewSource(sdk.UnwrapSDKContext(ctx).BlockTime().Unix()))
	
	// For simplicity, use reservoir sampling with weights
	// Select top N entries with some randomness
	for i := uint64(0); i < count && i < uint64(len(weightedEntries)); i++ {
		// Add some randomness: select from top 3*count entries
		selectionPoolSize := min(uint64(len(weightedEntries)), 3*count)
		if selectionPoolSize > i {
			idx := uint64(rng.Intn(int(selectionPoolSize-i))) + i
			selected = append(selected, weightedEntries[idx].entry)
			// Swap to avoid duplicates
			weightedEntries[i], weightedEntries[idx] = weightedEntries[idx], weightedEntries[i]
		} else {
			selected = append(selected, weightedEntries[i].entry)
		}
	}
	
	return selected, nil
}

// PruneVault removes old or unused entries from the vault
// Uses LRU policy: removes entries that haven't been used recently
func (k Keeper) PruneVault(ctx context.Context, expirationThreshold int64, minVaultSize uint64) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	currentHeight := sdkCtx.BlockHeight()
	
	// Get all entries
	var entriesToPrune []uint64
	
	err := k.GenesisVault.Walk(ctx, nil, func(key uint64, value types.GenesisVaultEntry) (stop bool, err error) {
		if value.IsExpired(currentHeight, expirationThreshold) {
			entriesToPrune = append(entriesToPrune, key)
		}
		return false, nil
	})
	
	if err != nil {
		return errorsmod.Wrap(err, "failed to iterate vault entries for pruning")
	}
	
	// Count total entries by iterating
	totalEntries := uint64(0)
	err = k.GenesisVault.Walk(ctx, nil, func(key uint64, entry types.GenesisVaultEntry) (stop bool, err error) {
		totalEntries++
		return false, nil
	})
	if err != nil {
		return errorsmod.Wrap(err, "failed to count vault entries")
	}
	
	// Only prune if we have more than minVaultSize entries
	if uint64(len(entriesToPrune)) > 0 && totalEntries > minVaultSize {
		// Calculate how many we can prune (keep at least minVaultSize)
		maxToPrune := totalEntries - minVaultSize
		if uint64(len(entriesToPrune)) > maxToPrune {
			// Sort by last_used_height (oldest first)
			type entryWithKey struct {
				key    uint64
				entry  types.GenesisVaultEntry
			}
			entriesWithKeys := make([]entryWithKey, 0, len(entriesToPrune))
			for _, key := range entriesToPrune {
				entry, _ := k.GenesisVault.Get(ctx, key)
				entriesWithKeys = append(entriesWithKeys, entryWithKey{
					key:   key,
					entry: entry,
				})
			}
			sort.Slice(entriesWithKeys, func(i, j int) bool {
				return entriesWithKeys[i].entry.LastUsedHeight < entriesWithKeys[j].entry.LastUsedHeight
			})
			// Take oldest entries up to maxToPrune
			entriesToPrune = make([]uint64, maxToPrune)
			for i := uint64(0); i < maxToPrune; i++ {
				entriesToPrune[i] = entriesWithKeys[i].key
			}
		}
		
		// Remove entries
		for _, key := range entriesToPrune {
			if err := k.GenesisVault.Remove(ctx, key); err != nil {
				sdkCtx.Logger().Error(fmt.Sprintf("Failed to prune vault entry %d: %v", key, err))
			}
		}
		
		sdkCtx.Logger().Info(fmt.Sprintf("Pruned %d entries from genesis vault", len(entriesToPrune)))
	}
	
	return nil
}

// Helper function for min
func min(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

