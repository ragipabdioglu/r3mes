package training

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"remes/x/remes/types"
)

// TrapJobVerdict represents the result of trap job verification
type TrapJobVerdict string

const (
	VerdictNormalJob  TrapJobVerdict = "normal_job"
	VerdictTrapPassed TrapJobVerdict = "trap_passed"
	VerdictTrapFailed TrapJobVerdict = "trap_failed"
	VerdictTimeout    TrapJobVerdict = "timeout"
)

// TrapJobVerificationResult holds the complete verification result
type TrapJobVerificationResult struct {
	Verdict           TrapJobVerdict
	TrapJobID         string
	MinerAddress      string
	ExpectedHash      string
	ActualHash        string
	FingerprintMatch  bool
	SlashAmount       string
	RewardAmount      string
	ErrorMessage      string
}

// VerifyTrapJobResult verifies if a gradient submission passes the trap job test
// This is the CRITICAL missing implementation
func (k *TrainingKeeper) VerifyTrapJobResult(
	ctx context.Context,
	gradient types.StoredGradient,
) (*TrapJobVerificationResult, error) {
	result := &TrapJobVerificationResult{
		MinerAddress: gradient.Miner,
		ActualHash:   gradient.GradientHash,
	}

	// 1. Check if this gradient is for a trap job
	trapJob, isTrap, err := k.getTrapJobForGradient(ctx, gradient)
	if err != nil {
		return nil, fmt.Errorf("failed to check trap job: %w", err)
	}

	if !isTrap {
		result.Verdict = VerdictNormalJob
		return result, nil
	}

	result.TrapJobID = trapJob.TrapJobId
	result.ExpectedHash = trapJob.ExpectedGradientHash

	// 2. Check deadline
	currentHeight := k.core.GetCurrentBlockHeight(ctx)
	if currentHeight > trapJob.DeadlineHeight {
		result.Verdict = VerdictTimeout
		result.SlashAmount = "25" // 25% slash for timeout
		result.ErrorMessage = fmt.Sprintf(
			"trap job deadline exceeded: current=%d, deadline=%d",
			currentHeight, trapJob.DeadlineHeight,
		)
		return result, nil
	}

	// 3. Compare gradient hashes
	if gradient.GradientHash != trapJob.ExpectedGradientHash {
		result.Verdict = VerdictTrapFailed
		result.SlashAmount = "50" // 50% slash for wrong gradient
		result.ErrorMessage = fmt.Sprintf(
			"gradient hash mismatch: expected=%s, actual=%s",
			trapJob.ExpectedGradientHash[:16]+"...",
			gradient.GradientHash[:16]+"...",
		)
		return result, nil
	}

	// 4. Trap job passed!
	// Note: Fingerprint verification removed - not in proto schema
	// Future: Add fingerprint fields to proto if needed for Top-K verification
	result.Verdict = VerdictTrapPassed
	result.RewardAmount = "10" // 10% bonus for passing trap job
	return result, nil
}

// getTrapJobForGradient checks if a gradient submission is for a trap job
func (k *TrainingKeeper) getTrapJobForGradient(
	ctx context.Context,
	gradient types.StoredGradient,
) (*types.TrapJob, bool, error) {
	// Query trap jobs targeting this miner
	// In production, this would query the TrapJobs collection
	
	// Check by training round ID (trap jobs have special round IDs)
	if k.isTrapJobRound(gradient.TrainingRoundId) {
		// Get trap job details
		trapJob, err := k.getTrapJobByRound(ctx, gradient.TrainingRoundId)
		if err != nil {
			return nil, false, err
		}
		
		// Verify this trap job targets this miner
		if trapJob.TargetMiner == gradient.Miner {
			return trapJob, true, nil
		}
	}

	return nil, false, nil
}

// isTrapJobRound checks if a training round ID corresponds to a trap job
// Trap jobs are injected at ~1% rate with special round ID patterns
func (k *TrainingKeeper) isTrapJobRound(roundID uint64) bool {
	// Trap job round IDs have a specific pattern
	// For example: high bit set, or specific modulo
	// This makes them indistinguishable to miners but identifiable to validators
	
	// Simple implementation: check if round ID is in trap job range
	// In production, use cryptographic commitment scheme
	return roundID >= 1000000000 && roundID < 2000000000
}

// getTrapJobByRound retrieves a trap job by its training round ID
func (k *TrainingKeeper) getTrapJobByRound(
	ctx context.Context,
	roundID uint64,
) (*types.TrapJob, error) {
	// Query from TrapJobs collection
	trapJob, err := k.TrapJobs.Get(ctx, roundID)
	if err != nil {
		return nil, fmt.Errorf("failed to get trap job for round %d: %w", roundID, err)
	}
	return &trapJob, nil
}

// verifyFingerprint compares gradient fingerprints
// Fingerprint = Top-K gradient values with their indices
// NOTE: Currently unused - fingerprint fields not in proto schema
// Kept for future implementation when fingerprint verification is added
func (k *TrainingKeeper) verifyFingerprint(actual, expected string) bool {
	// Parse JSON fingerprints and compare
	// Allow small tolerance for floating point differences
	
	// Simple string comparison for now
	// In production, parse and compare with tolerance
	return actual == expected
}

// computeGradientHash computes SHA256 hash of gradient data
// This should match the Python implementation in dora_trainer.py
func computeGradientHash(gradientData []byte) string {
	hash := sha256.Sum256(gradientData)
	return hex.EncodeToString(hash[:])
}

// ProcessTrapJobVerdict processes the verdict and applies economic consequences
func (k *TrainingKeeper) ProcessTrapJobVerdict(
	ctx context.Context,
	result *TrapJobVerificationResult,
) error {
	switch result.Verdict {
	case VerdictTrapPassed:
		// Reward miner for passing trap job
		// This proves they're doing honest computation
		return k.rewardTrapJobPass(ctx, result.MinerAddress, result.RewardAmount)

	case VerdictTrapFailed:
		// Slash miner for failing trap job
		// This indicates lazy or malicious behavior
		return k.slashTrapJobFailure(ctx, result.MinerAddress, result.SlashAmount, result.ErrorMessage)

	case VerdictTimeout:
		// Slash miner for not responding to trap job
		return k.slashTrapJobTimeout(ctx, result.MinerAddress, result.SlashAmount)

	case VerdictNormalJob:
		// Normal job, no special action needed
		return nil

	default:
		return fmt.Errorf("unknown trap job verdict: %s", result.Verdict)
	}
}

// rewardTrapJobPass rewards a miner for passing a trap job
func (k *TrainingKeeper) rewardTrapJobPass(ctx context.Context, miner string, amount string) error {
	// Log the reward
	k.core.Logger(ctx).Info(
		"trap job passed - rewarding miner",
		"miner", miner,
		"bonus_percent", amount,
	)
	
	// Call economics keeper to add bonus reward
	if k.economics != nil {
		return k.economics.AddTrapJobBonus(ctx, miner, amount)
	}
	
	return nil
}

// slashTrapJobFailure slashes a miner for failing a trap job
func (k *TrainingKeeper) slashTrapJobFailure(ctx context.Context, miner string, amount string, reason string) error {
	// Log the slash
	k.core.Logger(ctx).Warn(
		"trap job failed - slashing miner",
		"miner", miner,
		"slash_percent", amount,
		"reason", reason,
	)
	
	// Call economics keeper to slash stake
	if k.economics != nil {
		return k.economics.SlashForTrapJobFailure(ctx, miner, amount, reason)
	}
	
	return nil
}

// slashTrapJobTimeout slashes a miner for not responding to a trap job
func (k *TrainingKeeper) slashTrapJobTimeout(ctx context.Context, miner string, amount string) error {
	// Log the slash
	k.core.Logger(ctx).Warn(
		"trap job timeout - slashing miner",
		"miner", miner,
		"slash_percent", amount,
	)
	
	// Call economics keeper to slash stake
	if k.economics != nil {
		return k.economics.SlashForTrapJobTimeout(ctx, miner, amount)
	}
	
	return nil
}
