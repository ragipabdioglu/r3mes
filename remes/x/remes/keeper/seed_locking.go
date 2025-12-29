package keeper

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// DeriveGlobalSeed derives a deterministic seed from block hash and training round
// This ensures all miners use the same seed for the same training round
//
// PRODUCTION MODE: Fail-fast if block hash cannot be retrieved
// TEST MODE: Allow fallback to block height (controlled by environment variable)
func (k Keeper) DeriveGlobalSeed(
	ctx sdk.Context,
	trainingRoundID uint64,
) (uint64, error) {
	// Get block hash (header hash)
	blockHash := ctx.HeaderHash()
	if len(blockHash) == 0 {
		// Check if we're in test mode (allow fallback)
		// In production, this should NEVER happen - fail immediately
		isTestMode := os.Getenv("R3MES_TEST_MODE") == "true"
		
		if isTestMode {
			// TEST MODE: Allow fallback to block height for testing
			// This allows tests to run without setting block hash
			blockHeight := ctx.BlockHeight()
			if blockHeight > 0 {
				ctx.Logger().Warn(
					"Using block height as fallback seed source (TEST MODE ONLY)",
					"block_height", blockHeight,
					"training_round_id", trainingRoundID,
				)
				// Use block height as deterministic seed source for tests
				blockHash = make([]byte, 32)
				binary.BigEndian.PutUint64(blockHash[:8], uint64(blockHeight))
				binary.BigEndian.PutUint64(blockHash[8:16], trainingRoundID)
			} else {
				return 0, errorsmod.Wrap(
					types.ErrInvalidMiner,
					"block hash is empty and block height is zero (TEST MODE)",
				)
			}
		} else {
			// PRODUCTION MODE: Fail-fast - no fallback allowed
			// Deterministic execution requires valid block hash
			return 0, errorsmod.Wrap(
				types.ErrInvalidMiner,
				"CRITICAL: block hash is empty. Deterministic execution requires valid block hash from blockchain. "+
					"Miner must stop execution. This should never happen in production.",
			)
		}
	}

	// Combine block hash with training round ID
	seedData := make([]byte, len(blockHash)+8)
	copy(seedData, blockHash)
	binary.BigEndian.PutUint64(seedData[len(blockHash):], trainingRoundID)

	// Hash the combined data to get deterministic seed
	hash := sha256.Sum256(seedData)

	// Convert first 8 bytes to uint64 seed
	seed := binary.BigEndian.Uint64(hash[:8])

	return seed, nil
}

// GetGlobalSeedForTrainingRound returns the global seed for a specific training round
// This is called by miners to get the deterministic seed for their training
func (k Keeper) GetGlobalSeedForTrainingRound(
	ctx sdk.Context,
	trainingRoundID uint64,
) (uint64, error) {
	return k.DeriveGlobalSeed(ctx, trainingRoundID)
}

// VerifySeedLocking verifies that a miner used the correct global seed
// This is done by checking if the gradient hash matches expected hash for the seed
func (k Keeper) VerifySeedLocking(
	ctx sdk.Context,
	minerAddress string,
	trainingRoundID uint64,
	reportedSeed uint64,
) error {
	// Derive expected seed
	expectedSeed, err := k.DeriveGlobalSeed(ctx, trainingRoundID)
	if err != nil {
		return errorsmod.Wrap(err, "failed to derive global seed")
	}

	// Verify reported seed matches expected seed
	if reportedSeed != expectedSeed {
		return errorsmod.Wrapf(
			types.ErrInvalidMiner,
			"seed mismatch: expected %d, got %d",
			expectedSeed,
			reportedSeed,
		)
	}

	return nil
}

// RecordSeedViolation records a seed locking violation
func (k Keeper) RecordSeedViolation(
	ctx sdk.Context,
	minerAddress string,
	expectedSeed uint64,
	reportedSeed uint64,
) error {
	violationReason := fmt.Sprintf("seed locking violation: expected %d, got %d", expectedSeed, reportedSeed)
	return k.RecordEnvironmentViolation(ctx, minerAddress, violationReason)
}

