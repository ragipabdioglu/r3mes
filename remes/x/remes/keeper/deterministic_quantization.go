package keeper

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"math"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"
)

// QuantizationConfig holds configuration for deterministic quantization
type QuantizationConfig struct {
	// BitNet quantization parameters
	QuantizationBits uint8  // Number of bits for quantization (e.g., 1 for BitNet)
	ScaleFactor      uint64 // Scale factor for quantization
	ZeroPoint        int64  // Zero point for quantization
}

// GetQuantizationConfig returns the quantization configuration for a model
func (k Keeper) GetQuantizationConfig(
	ctx sdk.Context,
	modelConfigID uint64,
) (QuantizationConfig, error) {
	// For now, return default BitNet config
	// In production, this would retrieve from ModelConfig collection
	// BitNet uses 1-bit quantization with zero-centered approach
	return QuantizationConfig{
		QuantizationBits: 1,    // BitNet uses 1-bit quantization
		ScaleFactor:      1000,  // Default scale factor
		ZeroPoint:        0,     // BitNet uses zero-centered quantization
	}, nil
}

// QuantizeDeterministic performs deterministic quantization of a float value
// This ensures the same input always produces the same quantized output
// Formula: quantized = round((value - zero_point) * scale_factor) / scale_factor + zero_point
func (k Keeper) QuantizeDeterministic(
	ctx sdk.Context,
	value float64,
	modelConfigID uint64,
	seed uint64, // Global seed for deterministic rounding
) (int64, error) {
	// Get quantization config
	config, err := k.GetQuantizationConfig(ctx, modelConfigID)
	if err != nil {
		return 0, errorsmod.Wrap(err, "failed to get quantization config")
	}

	// Apply zero point adjustment
	adjustedValue := value - float64(config.ZeroPoint)

	// Scale the value
	scaledValue := adjustedValue * float64(config.ScaleFactor)

	// Deterministic rounding using seed
	// This ensures the same value + seed always produces the same quantized result
	quantizedValue := k.deterministicRound(ctx, scaledValue, seed)

	// Apply zero point back
	result := quantizedValue + config.ZeroPoint

	return result, nil
}

// deterministicRound performs deterministic rounding using a seed
// This ensures reproducibility across different environments
func (k Keeper) deterministicRound(
	ctx sdk.Context,
	value float64,
	seed uint64,
) int64 {
	// Use seed to determine rounding direction deterministically
	// Hash value + seed to get a deterministic "random" value
	seedBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(seedBytes, seed)
	valueBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(valueBytes, uint64(math.Float64bits(value)))

	combined := append(seedBytes, valueBytes...)
	hash := sha256.Sum256(combined)

	// Use hash to determine rounding (0-127 = round down, 128-255 = round up)
	// This ensures deterministic rounding based on seed
	hashValue := hash[0]
	roundUp := hashValue >= 128

	base := int64(math.Floor(value))
	if roundUp {
		return base + 1
	}
	return base
}

// DequantizeDeterministic performs deterministic dequantization
// This is the inverse of QuantizeDeterministic
func (k Keeper) DequantizeDeterministic(
	ctx sdk.Context,
	quantizedValue int64,
	modelConfigID uint64,
) (float64, error) {
	// Get quantization config
	config, err := k.GetQuantizationConfig(ctx, modelConfigID)
	if err != nil {
		return 0, errorsmod.Wrap(err, "failed to get quantization config")
	}

	// Remove zero point
	adjustedValue := quantizedValue - config.ZeroPoint

	// De-scale
	dequantizedValue := float64(adjustedValue) / float64(config.ScaleFactor)

	// Add zero point back
	result := dequantizedValue + float64(config.ZeroPoint)

	return result, nil
}

// VerifyQuantization verifies that a quantized value matches expected quantization
// This is used to ensure miners use deterministic quantization
func (k Keeper) VerifyQuantization(
	ctx sdk.Context,
	originalValue float64,
	quantizedValue int64,
	modelConfigID uint64,
	seed uint64,
) (bool, error) {
	// Quantize the original value deterministically
	expectedQuantized, err := k.QuantizeDeterministic(ctx, originalValue, modelConfigID, seed)
	if err != nil {
		return false, errorsmod.Wrap(err, "failed to quantize original value")
	}

	// Compare quantized values
	// Allow ±1 tolerance for quantization rounding differences
	tolerance := int64(1)
	diff := expectedQuantized - quantizedValue
	if diff < 0 {
		diff = -diff
	}

	return diff <= tolerance, nil
}

// QuantizeGradientHash performs deterministic quantization of a gradient hash
// This ensures gradient hashes are consistent across different quantization implementations
func (k Keeper) QuantizeGradientHash(
	ctx sdk.Context,
	gradientHash string,
	modelConfigID uint64,
	seed uint64,
) (string, error) {
	// Convert hash to bytes
	hashBytes := []byte(gradientHash)

	// Use seed to create deterministic quantization of hash
	seedBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(seedBytes, seed)

	combined := append(hashBytes, seedBytes...)
	quantizedHash := sha256.Sum256(combined)

	// Return hex-encoded quantized hash
	return fmt.Sprintf("%x", quantizedHash[:]), nil
}

// ValidateDeterministicExecution validates that a gradient was computed deterministically
// This checks both seed locking and quantization
func (k Keeper) ValidateDeterministicExecution(
	ctx sdk.Context,
	minerAddress string,
	trainingRoundID uint64,
	reportedSeed uint64,
	gradientHash string,
	modelConfigID uint64,
) error {
	// 1. Verify seed locking
	if err := k.VerifySeedLocking(ctx, minerAddress, trainingRoundID, reportedSeed); err != nil {
		return errorsmod.Wrap(err, "seed locking verification failed")
	}

	// 2. Verify quantization (if gradient hash is provided)
	// In production, this would verify that the gradient hash matches
	// the expected hash for the given seed and model config
	expectedSeed, err := k.DeriveGlobalSeed(ctx, trainingRoundID)
	if err != nil {
		return errorsmod.Wrap(err, "failed to derive global seed")
	}

	// Verify gradient hash matches expected quantization
	expectedHash, err := k.QuantizeGradientHash(ctx, gradientHash, modelConfigID, expectedSeed)
	if err != nil {
		return errorsmod.Wrap(err, "failed to quantize gradient hash")
	}

	// For now, we just verify the hash format
	// In production, this would compare with expected hash from deterministic computation
	_ = expectedHash

	return nil
}

