package keeper

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
)

// GPUArchitecture represents GPU architecture information
type GPUArchitecture struct {
	Architecture      string
	ComputeCapability string
	DeterministicMode bool
	CUDAVersion      string
}

// VerificationResult represents the result of gradient verification
type VerificationResult struct {
	Valid                   bool
	Reason                  string
	RequiresCPUVerification bool
	HashMatch               bool
	CrossArchitecture       bool
}

// VerifyGradientHash verifies gradient hash with GPU architecture awareness
func (k Keeper) VerifyGradientHash(
	ctx sdk.Context,
	minerHash string,
	validatorHash string,
	minerGPU string,
	validatorGPU string,
) VerificationResult {
	// Step 1: Check if same GPU architecture - direct hash comparison
	if minerGPU == validatorGPU {
		// Same architecture: Require exact hash match (bit-exact)
		if minerHash == validatorHash {
			return VerificationResult{
				Valid:     true,
				Reason:    "exact_hash_match_same_architecture",
				HashMatch: true,
			}
		}
		// Same architecture but hash mismatch - likely fraud or non-deterministic execution
		return VerificationResult{
			Valid:                   false,
			Reason:                  "hash_mismatch_same_architecture",
			RequiresCPUVerification: true, // MANDATORY CPU fallback
			HashMatch:               false,
		}
	}

	// Step 2: Different GPU architectures - MANDATORY CPU verification
	// Floating-point differences between architectures are expected
	// CPU Iron Sandbox provides bit-exact resolution
	return VerificationResult{
		Valid:                   false,
		Reason:                  "cross_architecture_verification_required",
		RequiresCPUVerification: true, // MANDATORY for cross-architecture disputes
		HashMatch:               false,
		CrossArchitecture:       true,
	}
}

// ComputeDeterministicHash computes deterministic hash of gradient data
func ComputeDeterministicHash(gradientData []byte) string {
	hash := sha256.Sum256(gradientData)
	return hex.EncodeToString(hash[:])
}

// VerifyExactHashMatch verifies exact hash match with no tolerance
func VerifyExactHashMatch(minerHash string, validatorHash string) bool {
	return minerHash == validatorHash
}

// ShouldCompareDirectly checks if GPU architectures should be compared directly
func ShouldCompareDirectly(minerGPU string, validatorGPU string) bool {
	return minerGPU == validatorGPU
}

// CreateCPUVerificationRequest creates a CPU verification request for dispute resolution
func CreateCPUVerificationRequest(
	challengeID string,
	disputedGradientIPFSHash string,
	expectedHash string,
	seed uint64,
) map[string]interface{} {
	return map[string]interface{}{
		"challenge_id":                  challengeID,
		"disputed_gradient":             disputedGradientIPFSHash,
		"execution_mode":                "CPU", // MUST be CPU for disputes
		"expected_result":               expectedHash,
		"seed":                          seed,
		"requires_deterministic_execution": true,
	}
}

// VerifyGradientWithArchitecture verifies gradient with GPU architecture awareness
func (k Keeper) VerifyGradientWithArchitecture(
	ctx sdk.Context,
	minerHash string,
	validatorHash string,
	minerGPUArchitecture string,
	validatorGPUArchitecture string,
) (VerificationResult, error) {
	// Get GPU architecture configs (if needed for whitelist checking)
	// For now, we'll do basic verification

	result := k.VerifyGradientHash(
		ctx,
		minerHash,
		validatorHash,
		minerGPUArchitecture,
		validatorGPUArchitecture,
	)

	// Log verification result
	if !result.Valid {
		ctx.Logger().Info(
			fmt.Sprintf(
				"Gradient verification failed: %s (miner: %s, validator: %s)",
				result.Reason,
				minerGPUArchitecture,
				validatorGPUArchitecture,
			),
		)
	}

	return result, nil
}

// InitiateCPUVerification initiates CPU Iron Sandbox verification for disputes
func (k Keeper) InitiateCPUVerification(
	ctx sdk.Context,
	challengeID string,
	disputedGradientIPFSHash string,
	expectedHash string,
	seed uint64,
) error {
	// Create CPU verification request
	verificationRequest := CreateCPUVerificationRequest(
		challengeID,
		disputedGradientIPFSHash,
		expectedHash,
		seed,
	)

	// Store verification request (could be stored in ChallengeRecord)
	ctx.Logger().Info(
		fmt.Sprintf(
			"CPU verification initiated for challenge %s: %v",
			challengeID,
			verificationRequest,
		),
	)

	// In production, this would:
	// 1. Select validator panel (3 random validators)
	// 2. Send verification request to validators
	// 3. Set verification deadline (50 blocks)
	// 4. Wait for consensus result

	return nil
}

