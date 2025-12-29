package keeper

import (
	"context"
	"crypto/sha256"
	"fmt"
	"os"

	errorsmod "cosmossdk.io/errors"
	sdkmath "cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// VerifyMessageSignature verifies a Secp256k1 signature for a message.
// This implements the same message hashing as the Python client.
func (k Keeper) VerifyMessageSignature(
	ctx sdk.Context,
	minerAddress string,
	messageHash []byte,
	signature []byte,
) error {
	// Decode miner address
	minerAcc, err := sdk.AccAddressFromBech32(minerAddress)
	if err != nil {
		return errorsmod.Wrapf(types.ErrInvalidMiner, fmt.Sprintf("invalid miner address: %v", err))
	}

	// Get account public key from auth keeper
	account := k.authKeeper.GetAccount(ctx, minerAcc)
	if account == nil {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "miner account not found")
	}

	pubKey := account.GetPubKey()
	if pubKey == nil {
		return errorsmod.Wrapf(types.ErrInvalidSignature, "miner account has no public key")
	}

	// Verify signature
	// Note: Cosmos SDK uses DER-encoded signatures
	// The Python client sends DER-encoded signatures
	if !pubKey.VerifySignature(messageHash, signature) {
		return errorsmod.Wrapf(types.ErrInvalidSignature, "signature verification failed")
	}

	return nil
}

// CreateMessageHash creates a deterministic message hash for signing.
// This must match the Python client's create_message_hash function.
func CreateMessageHash(
	chainID string,
	miner string,
	ipfsHash string,
	modelVersion string,
	trainingRoundID uint64,
	shardID uint64,
	gradientHash string,
	gpuArchitecture string,
	nonce uint64,
) []byte {
	// Create deterministic message string (must match Python format)
	messageStr := fmt.Sprintf(
		"%s|%s|%s|%s|%d|%d|%s|%s|%d",
		chainID,
		miner,
		ipfsHash,
		modelVersion,
		trainingRoundID,
		shardID,
		gradientHash,
		gpuArchitecture,
		nonce,
	)

	// Hash the message
	hash := sha256.Sum256([]byte(messageStr))
	return hash[:]
}

// VerifyNonce checks if a nonce has been used before (replay attack prevention).
// Uses a sliding window approach to prevent unbounded state growth.
func (k Keeper) VerifyNonce(ctx sdk.Context, minerAddress string, nonce uint64) error {
	// Basic validation: nonce must be > 0
	if nonce == 0 {
		return errorsmod.Wrapf(types.ErrInvalidNonce, "nonce cannot be zero")
	}

	// Get or create nonce window for this miner
	windowKey := fmt.Sprintf("nonce_window:%s", minerAddress)

	// Try to get existing window
	var minNonce, maxNonce uint64
	windowData, err := k.NonceWindows.Get(ctx, windowKey)
	if err != nil {
		// No window exists, create new one
		// Window size: 10000 nonces (configurable)
		minNonce = 1
		maxNonce = 10000
	} else {
		// Parse existing window
		fmt.Sscanf(windowData, "%d|%d", &minNonce, &maxNonce)
	}

	// Check if nonce is within valid window
	if nonce < minNonce {
		return errorsmod.Wrapf(types.ErrInvalidNonce, "nonce %d is below minimum window %d for miner %s", nonce, minNonce, minerAddress)
	}

	// If nonce exceeds max window, slide the window forward
	windowSize := uint64(10000) // Configurable window size
	if nonce > maxNonce {
		// Slide window: new min = nonce - windowSize + 1, new max = nonce + windowSize
		newMinNonce := nonce - windowSize + 1
		if newMinNonce < 1 {
			newMinNonce = 1
		}
		newMaxNonce := nonce + windowSize

		// Clean up old nonces that are now outside the window
		// This is done lazily - old nonces will be overwritten
		minNonce = newMinNonce
		maxNonce = newMaxNonce

		// Save updated window
		newWindowStr := fmt.Sprintf("%d|%d", minNonce, maxNonce)
		if err := k.NonceWindows.Set(ctx, windowKey, newWindowStr); err != nil {
			return errorsmod.Wrapf(err, "failed to update nonce window")
		}
	}

	// Create composite key: "miner_address|nonce"
	nonceKey := fmt.Sprintf("%s|%d", minerAddress, nonce)

	// Check if nonce has been used before
	used, err := k.UsedNonces.Get(ctx, nonceKey)
	if err == nil && used {
		return errorsmod.Wrapf(types.ErrInvalidNonce, "nonce %d has already been used by miner %s", nonce, minerAddress)
	}

	// Nonce is valid (not used before)
	return nil
}

// MarkNonceAsUsed marks a nonce as used for a miner (called after successful transaction)
func (k Keeper) MarkNonceAsUsed(ctx sdk.Context, minerAddress string, nonce uint64) error {
	// Create composite key: "miner_address|nonce"
	nonceKey := fmt.Sprintf("%s|%d", minerAddress, nonce)

	// Mark nonce as used
	if err := k.UsedNonces.Set(ctx, nonceKey, true); err != nil {
		return err
	}

	// Update nonce window if needed
	windowKey := fmt.Sprintf("nonce_window:%s", minerAddress)
	windowData, err := k.NonceWindows.Get(ctx, windowKey)

	var minNonce, maxNonce uint64
	windowSize := uint64(10000)

	if err != nil {
		// Create new window
		minNonce = 1
		maxNonce = windowSize
	} else {
		fmt.Sscanf(windowData, "%d|%d", &minNonce, &maxNonce)
	}

	// Extend window if nonce is near the edge
	if nonce > maxNonce-1000 {
		maxNonce = nonce + windowSize
		newWindowStr := fmt.Sprintf("%d|%d", minNonce, maxNonce)
		if err := k.NonceWindows.Set(ctx, windowKey, newWindowStr); err != nil {
			return err
		}
	}

	return nil
}

// CleanupOldNonces removes nonces that are outside the current window
// This should be called periodically (e.g., in EndBlocker) to prevent state bloat
func (k Keeper) CleanupOldNonces(ctx sdk.Context, minerAddress string) (int, error) {
	windowKey := fmt.Sprintf("nonce_window:%s", minerAddress)
	windowData, err := k.NonceWindows.Get(ctx, windowKey)
	if err != nil {
		return 0, nil // No window, nothing to clean
	}

	var minNonce, maxNonce uint64
	fmt.Sscanf(windowData, "%d|%d", &minNonce, &maxNonce)

	// Delete nonces below minNonce
	// Note: This is a simplified cleanup - in production, you'd want to
	// iterate through the nonces more efficiently
	cleaned := 0
	for i := uint64(1); i < minNonce && i < minNonce+1000; i++ {
		nonceKey := fmt.Sprintf("%s|%d", minerAddress, i)
		if err := k.UsedNonces.Remove(ctx, nonceKey); err == nil {
			cleaned++
		}
	}

	return cleaned, nil
}

// CheckStakingRequirement verifies that the miner has staked tokens.
// Miners must have a minimum stake to participate (prevents spam)
// FIXED: In production, only "remes" denom is accepted
func (k Keeper) CheckStakingRequirement(ctx sdk.Context, minerAddress string) error {
	// Determine if we're in production
	isProduction := os.Getenv("R3MES_ENV") == "production" || os.Getenv("R3MES_ENV") == "prod"

	// Minimum stake requirement (can be made configurable via params)
	minStake := sdkmath.NewInt(1000) // 1000 tokens

	// Convert miner address
	minerAcc, err := sdk.AccAddressFromBech32(minerAddress)
	if err != nil {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "invalid miner address: %v", err)
	}

	// First, check if miner is registered as a node (has stake)
	registration, err := k.NodeRegistrations.Get(ctx, minerAddress)
	if err == nil {
		// Node is registered, check stake
		stake, err := sdk.ParseCoinsNormalized(registration.Stake)
		if err == nil {
			if isProduction {
				// PRODUCTION: Only accept "remes" denom
				remesAmount := stake.AmountOf("remes")
				if remesAmount.GTE(minStake) {
					return nil // Stake requirement met
				}
			} else {
				// NON-PRODUCTION: Accept both "remes" and "stake" denoms for testing
				remesAmount := stake.AmountOf("remes")
				stakeAmount := stake.AmountOf("stake")
				if remesAmount.GTE(minStake) || stakeAmount.GTE(minStake) {
					return nil // Stake requirement met
				}
			}
		}
	}

	// If not registered as node, check bank balance
	balances := k.bankKeeper.SpendableCoins(ctx, minerAcc)

	if isProduction {
		// PRODUCTION: Only accept "remes" denom
		remesBalance := balances.AmountOf("remes")
		if remesBalance.GTE(minStake) {
			return nil // Balance requirement met
		}
		return errorsmod.Wrapf(types.ErrInsufficientStake,
			"miner must have at least 1000 remes staked (production mode). Current balance: %s remes",
			remesBalance.String())
	} else {
		// NON-PRODUCTION: Accept both denoms for testing
		remesBalance := balances.AmountOf("remes")
		stakeBalance := balances.AmountOf("stake")
		if remesBalance.GTE(minStake) || stakeBalance.GTE(minStake) {
			return nil // Balance requirement met
		}
		return errorsmod.Wrapf(types.ErrInsufficientStake,
			"miner must have at least 1000 remes or 1000 stake (testnet mode)")
	}
}

// VerifyIPFSContentExists verifies that the IPFS content exists.
func (k Keeper) VerifyIPFSContentExists(ctx sdk.Context, ipfsHash string) error {
	if k.ipfsManager == nil {
		// If IPFS manager is not configured, skip verification
		return nil
	}

	// Convert sdk.Context to context.Context for IPFS manager
	ipfsCtx := context.Background()
	// Verify content exists in IPFS
	exists, err := k.ipfsManager.VerifyContentExists(ipfsCtx, ipfsHash)
	if err != nil {
		return errorsmod.Wrapf(types.ErrInvalidIPFSHash, fmt.Sprintf("IPFS verification error: %v", err))
	}

	if !exists {
		return errorsmod.Wrapf(types.ErrInvalidIPFSHash, "IPFS content does not exist")
	}

	return nil
}
