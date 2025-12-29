package keeper

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"
)

// GenerateBlindingFactor generates a random blinding factor for trap job obfuscation
// The blinding factor is used to hide the expected gradient from miners
// until verification time
func (k Keeper) GenerateBlindingFactor(ctx sdk.Context, trapJobID string) (string, error) {
	// Generate 32-byte random blinding factor
	blindingBytes := make([]byte, 32)
	if _, err := rand.Read(blindingBytes); err != nil {
		return "", errorsmod.Wrap(err, "failed to generate blinding factor")
	}

	// Convert to hex string for storage
	blindingFactor := hex.EncodeToString(blindingBytes)

	ctx.Logger().Info("Generated blinding factor for trap job",
		"trap_job_id", trapJobID,
		"blinding_factor_length", len(blindingFactor),
	)

	return blindingFactor, nil
}

// ApplyBlindingFactor applies a blinding factor to an expected gradient hash
// This obfuscates the gradient so miners cannot detect trap jobs
// Formula: blinded_hash = hash(original_hash + blinding_factor)
func (k Keeper) ApplyBlindingFactor(
	expectedGradientHash string,
	blindingFactor string,
) string {
	// Combine expected gradient hash with blinding factor
	combined := fmt.Sprintf("%s:%s", expectedGradientHash, blindingFactor)
	
	// Hash the combination
	hash := sha256.Sum256([]byte(combined))
	
	// Return hex-encoded blinded hash
	return hex.EncodeToString(hash[:])
}

// RemoveBlindingFactor removes the blinding factor from a blinded gradient hash
// This is used during verification to recover the original expected gradient
// Note: This requires knowledge of the blinding factor
func (k Keeper) RemoveBlindingFactor(
	blindedHash string,
	blindingFactor string,
	originalHash string,
) (bool, error) {
	// Reconstruct the blinded hash
	expectedBlindedHash := k.ApplyBlindingFactor(originalHash, blindingFactor)
	
	// Compare with submitted blinded hash
	return blindedHash == expectedBlindedHash, nil
}

// VerifyBlindedTrapJob verifies a trap job result with blinding factor
// This ensures miners cannot detect trap jobs by comparing hashes
func (k Keeper) VerifyBlindedTrapJob(
	ctx sdk.Context,
	trapJobID string,
	submittedGradientHash string,
) (bool, error) {
	// Get trap job
	trapJob, err := k.TrapJobs.Get(ctx, trapJobID)
	if err != nil {
		return false, errorsmod.Wrap(err, "trap job not found")
	}

	// Check if blinding factor exists
	if trapJob.BlindingFactor == "" {
		// No blinding factor - use direct comparison (backward compatibility)
		return submittedGradientHash == trapJob.ExpectedGradientHash, nil
	}

	// Apply blinding factor to expected gradient hash
	blindedExpectedHash := k.ApplyBlindingFactor(
		trapJob.ExpectedGradientHash,
		trapJob.BlindingFactor,
	)

	// Apply blinding factor to submitted gradient hash
	blindedSubmittedHash := k.ApplyBlindingFactor(
		submittedGradientHash,
		trapJob.BlindingFactor,
	)

	// Compare blinded hashes
	// Miner cannot detect this is a trap job because they don't know the blinding factor
	if blindedSubmittedHash == blindedExpectedHash {
		ctx.Logger().Info("Trap job verification passed (blinded)",
			"trap_job_id", trapJobID,
		)
		return true, nil
	}

	ctx.Logger().Info("Trap job verification failed (blinded)",
		"trap_job_id", trapJobID,
		"expected_blinded", blindedExpectedHash[:16]+"...",
		"submitted_blinded", blindedSubmittedHash[:16]+"...",
	)

	return false, nil
}

// StoreBlindingFactor stores the blinding factor for a trap job
// This is called when creating a trap job
func (k Keeper) StoreBlindingFactor(
	ctx sdk.Context,
	trapJobID string,
	blindingFactor string,
) error {
	// Get trap job
	trapJob, err := k.TrapJobs.Get(ctx, trapJobID)
	if err != nil {
		return errorsmod.Wrap(err, "trap job not found")
	}

	// Store blinding factor
	trapJob.BlindingFactor = blindingFactor
	trapJob.IsBlinded = true

	// Update trap job
	if err := k.TrapJobs.Set(ctx, trapJobID, trapJob); err != nil {
		return errorsmod.Wrap(err, "failed to update trap job with blinding factor")
	}

	ctx.Logger().Info("Stored blinding factor for trap job",
		"trap_job_id", trapJobID,
	)

	return nil
}

// GetBlindedExpectedHash returns the blinded expected gradient hash for a trap job
// This is what miners see - they cannot detect it's a trap job
func (k Keeper) GetBlindedExpectedHash(
	ctx sdk.Context,
	trapJobID string,
) (string, error) {
	// Get trap job
	trapJob, err := k.TrapJobs.Get(ctx, trapJobID)
	if err != nil {
		return "", errorsmod.Wrap(err, "trap job not found")
	}

	// If no blinding factor, return original hash
	if trapJob.BlindingFactor == "" {
		return trapJob.ExpectedGradientHash, nil
	}

	// Return blinded hash
	return k.ApplyBlindingFactor(
		trapJob.ExpectedGradientHash,
		trapJob.BlindingFactor,
	), nil
}

