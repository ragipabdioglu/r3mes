package keeper

import (
	"crypto/sha256"
	"fmt"
	"sort"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// FederatedTrapJobPayload represents a multi-sig trap job payload
type FederatedTrapJobPayload struct {
	TrapJobID        string
	DataShardHash    string
	ExpectedGradient string // Encrypted
	ModelStateHash   string
	InjectionHeight  int64
	ExpiryHeight     int64
	Signers          []string
}

// CreateFederatedTrapJob creates a trap job signed by top 3 validators (multi-sig)
// Replaces single Protocol Oracle for improved security and decentralization
func (k Keeper) CreateFederatedTrapJob(
	ctx sdk.Context,
	trapJobID string,
	targetMiner string,
	dataShardHash string,
	expectedGradientHash string,
	deadlineBlocks uint64,
) (*types.TrapJob, error) {
	// 1. Select top 3 validators by stake
	topValidators := k.selectTopValidatorsByStake(ctx, 3)
	if len(topValidators) < 3 {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "insufficient validators for federated trap job (need 3, got %d)", len(topValidators))
	}

	// 2. In production, validators would sign off-chain and submit signatures
	// For now, we create the trap job structure with signer addresses
	// Signatures will be collected via MsgCreateTrapJob with multi-sig validation

	// 5. Create trap job record
	deadlineHeight := ctx.BlockHeight() + int64(deadlineBlocks)
	trapJob := types.TrapJob{
		TrapJobId:            trapJobID,
		TargetMiner:          targetMiner,
		DatasetIpfsHash:      dataShardHash,
		ExpectedGradientHash: expectedGradientHash,
		CreatedAtHeight:      ctx.BlockHeight(),
		DeadlineHeight:       deadlineHeight,
		Status:               "pending",
		FederatedSigners:     topValidators,
		FederatedSignatures:  nil, // Will be populated when validators submit signatures
		CreatedBy:            "federated_validators", // Multi-sig, not single oracle
	}

	return &trapJob, nil
}

// selectTopValidatorsByStake selects top N validators by stake
func (k Keeper) selectTopValidatorsByStake(ctx sdk.Context, n int) []string {
	type validatorStake struct {
		address string
		stake   sdkmath.Int
	}

	var validators []validatorStake

	// Collect all validators with stake
	_ = k.NodeRegistrations.Walk(ctx, nil, func(key string, value types.NodeRegistration) (stop bool, err error) {
		hasValidatorRole := false
		for _, role := range value.Roles {
			if role == types.NODE_TYPE_VALIDATOR && value.Status == types.NODE_STATUS_ACTIVE {
				hasValidatorRole = true
				break
			}
		}

		if hasValidatorRole {
			stake, parseErr := sdk.ParseCoinsNormalized(value.Stake)
			stakeAmount := sdkmath.ZeroInt()
			if parseErr == nil && !stake.IsZero() {
				for _, coin := range stake {
					stakeAmount = stakeAmount.Add(coin.Amount)
				}
			}

			validators = append(validators, validatorStake{
				address: key,
				stake:   stakeAmount,
			})
		}
		return false, nil
	})

	// Sort by stake (descending)
	sort.Slice(validators, func(i, j int) bool {
		return validators[i].stake.GT(validators[j].stake)
	})

	// Return top N
	result := make([]string, 0, n)
	for i := 0; i < n && i < len(validators); i++ {
		result = append(result, validators[i].address)
	}

	return result
}

// serializeTrapJobPayloadForVerification serializes trap job payload for signature verification
func (k Keeper) serializeTrapJobPayloadForVerification(
	trapJobID string,
	dataShardHash string,
	expectedGradient string,
	injectionHeight int64,
	expiryHeight int64,
	signers []string,
) []byte {
	// Create deterministic serialization
	serialized := fmt.Sprintf(
		"%s|%s|%s|%d|%d|%v",
		trapJobID,
		dataShardHash,
		expectedGradient,
		injectionHeight,
		expiryHeight,
		signers,
	)
	return []byte(serialized)
}

// verifyFederatedTrapJobSignature verifies multi-sig (2/3 threshold)
func (k Keeper) verifyFederatedTrapJobSignature(
	ctx sdk.Context,
	payloadBytes []byte,
	signers []string,
	signatures [][]byte,
) error {
	if len(signers) != 3 || len(signatures) != 3 {
		return errorsmod.Wrapf(types.ErrInvalidSignature, "federated trap job requires exactly 3 signers, got %d signers and %d signatures", len(signers), len(signatures))
	}

	// Verify at least 2/3 signatures
	validSignatures := 0
	for i, signer := range signers {
		if i >= len(signatures) {
			continue
		}

		signerAddr, err := k.addressCodec.StringToBytes(signer)
		if err != nil {
			continue
		}

		account := k.authKeeper.GetAccount(ctx, signerAddr)
		if account == nil {
			continue
		}

		pubKey := account.GetPubKey()
		if pubKey == nil {
			continue
		}

		// Hash payload for signature verification
		hash := sha256.Sum256(payloadBytes)
		if pubKey.VerifySignature(hash[:], signatures[i]) {
			validSignatures++
		}
	}

	// Require 2/3 threshold
	if validSignatures < 2 {
		return errorsmod.Wrapf(types.ErrInvalidSignature, "federated trap job requires at least 2/3 valid signatures, got %d/3", validSignatures)
	}

	return nil
}

