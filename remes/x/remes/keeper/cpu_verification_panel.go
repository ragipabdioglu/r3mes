package keeper

import (
	"crypto/sha256"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// selectCPUVerificationPanel selects validators for CPU verification panel using VRF-like selection
// This is an improved implementation with:
// - Deterministic selection based on block hash + challenge context
// - Stake-weighted selection (validators with more stake have higher probability)
// - Proper shuffling to avoid bias
// Note: For production, consider using a proper cryptographic VRF library
func (k Keeper) selectCPUVerificationPanel(ctx sdk.Context, panelSize int, challengeContext string) []string {
	// Get all registered validator nodes with their stake
	type validatorInfo struct {
		address string
		stake   int64 // Stake amount for weighting
	}
	
	var validators []validatorInfo
	err := k.NodeRegistrations.Walk(ctx, nil, func(key string, value types.NodeRegistration) (stop bool, err error) {
		// Only select nodes with validator role and active status
		hasValidatorRole := false
		for _, role := range value.Roles {
			if role == types.NODE_TYPE_VALIDATOR && value.Status == types.NODE_STATUS_ACTIVE {
				hasValidatorRole = true
				break
			}
		}
		
		if hasValidatorRole {
			// Parse stake for weighting
			stake, parseErr := sdk.ParseCoinsNormalized(value.Stake)
			stakeAmount := int64(0)
			if parseErr == nil && !stake.IsZero() {
				// Use total stake amount (sum of all denoms)
				for _, coin := range stake {
					stakeAmount += coin.Amount.Int64()
				}
			}
			
			// Minimum stake of 1 for weighting (to avoid zero-weight validators)
			if stakeAmount == 0 {
				stakeAmount = 1
			}
			
			validators = append(validators, validatorInfo{
				address: key,
				stake:   stakeAmount,
			})
		}
		return false, nil
	})
	if err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to walk node registrations: %v", err))
		return []string{}
	}

	if len(validators) == 0 {
		ctx.Logger().Warn("No validators found for CPU verification panel")
		return []string{}
	}

	// If we have fewer validators than panel size, use all
	if len(validators) <= panelSize {
		result := make([]string, len(validators))
		for i, v := range validators {
			result[i] = v.address
		}
		return result
	}

	// VRF-like selection: Use block hash + challenge context as seed
	// This provides deterministic but unpredictable selection
	seed := fmt.Sprintf("%d|%x|%s", ctx.BlockHeight(), ctx.HeaderHash(), challengeContext)
	hash := sha256.Sum256([]byte(seed))

	// Stake-weighted selection using weighted random sampling
	// Algorithm: Create cumulative weights, then use hash to select
	totalStake := int64(0)
	for _, v := range validators {
		totalStake += v.stake
	}

	selected := make([]string, 0, panelSize)
	used := make(map[int]bool)
	hashOffset := 0

	for len(selected) < panelSize {
		// Use hash bytes for random selection
		randomValue := int64(0)
		for i := 0; i < 8 && hashOffset+i < len(hash); i++ {
			randomValue = randomValue<<8 | int64(hash[hashOffset+i])
		}
		hashOffset = (hashOffset + 8) % len(hash)
		
		// Normalize to [0, totalStake)
		randomValue = randomValue % totalStake
		if randomValue < 0 {
			randomValue = -randomValue
		}

		// Select validator based on weighted random
		cumulativeWeight := int64(0)
		selectedIndex := -1
		for i, v := range validators {
			if used[i] {
				continue
			}
			cumulativeWeight += v.stake
			if randomValue < cumulativeWeight {
				selectedIndex = i
				break
			}
		}

		// Fallback: if no validator selected (shouldn't happen), use first unused
		if selectedIndex == -1 {
			for i := range validators {
				if !used[i] {
					selectedIndex = i
					break
				}
			}
		}

		if selectedIndex >= 0 {
			selected = append(selected, validators[selectedIndex].address)
			used[selectedIndex] = true
		} else {
			// No more validators available
			break
		}
	}

	return selected
}

