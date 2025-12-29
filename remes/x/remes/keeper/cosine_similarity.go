package keeper

import (
	"fmt"
	"math"

	"remes/x/remes/types"
)

// CalculateCosineSimilarityWithMasking calculates cosine similarity using the masking method
// This is the REQUIRED method for tolerant verification
//
// Algorithm:
// 1. Use vault's ExpectedFingerprint.Indices to extract values from miner's full gradient
// 2. Compare vault's ExpectedFingerprint.Values with miner's masked vector
// 3. Calculate cosine similarity between these two vectors
//
// Why masking method?
// - Miner's Top-K list may differ due to hardware noise (different GPU)
// - We align miner's results to vault's reference system (indices)
// - Miner's Top-K list is NOT used - only vault's indices matter
// Exported for testing
func CalculateCosineSimilarityWithMasking(
	expectedFingerprint types.GradientFingerprint,
	minerFullGradient []float64,
) (float64, error) {
	// Validate inputs
	if len(expectedFingerprint.Indices) == 0 {
		return 0.0, fmt.Errorf("expected fingerprint indices array is empty")
	}
	if len(expectedFingerprint.Values) == 0 {
		return 0.0, fmt.Errorf("expected fingerprint values array is empty")
	}
	if len(expectedFingerprint.Indices) != len(expectedFingerprint.Values) {
		return 0.0, fmt.Errorf("expected fingerprint indices and values arrays have different lengths: %d vs %d", len(expectedFingerprint.Indices), len(expectedFingerprint.Values))
	}
	
	// Step 1: Extract masked vector from miner's full gradient using vault indices
	vaultIndices := expectedFingerprint.Indices  // e.g., [5, 100, 999]
	vaultValues := expectedFingerprint.Values    // e.g., [0.9, 0.5, 0.3]
	
	minerVectorMasked := make([]float64, len(vaultIndices))
	for i, idx := range vaultIndices {
		if idx < 0 || idx >= len(minerFullGradient) {
			return 0.0, fmt.Errorf("index %d out of bounds for miner gradient (size: %d)", idx, len(minerFullGradient))
		}
		minerVectorMasked[i] = minerFullGradient[idx]  // Extract value at vault's index
	}
	
	// Step 2: Calculate cosine similarity between vault values and miner masked vector
	similarity := calculateCosineSimilarity(vaultValues, minerVectorMasked)
	
	return similarity, nil
}

// calculateCosineSimilarity calculates cosine similarity between two vectors
// Formula: cos(θ) = (A · B) / (||A|| * ||B||)
func calculateCosineSimilarity(vec1, vec2 []float64) float64 {
	if len(vec1) != len(vec2) {
		return 0.0
	}
	
	// Dot product: A · B
	dotProduct := 0.0
	for i := range vec1 {
		dotProduct += vec1[i] * vec2[i]
	}
	
	// Magnitude (L2 norm) of vec1: ||A||
	mag1 := 0.0
	for i := range vec1 {
		mag1 += vec1[i] * vec1[i]
	}
	mag1 = math.Sqrt(mag1)
	
	// Magnitude (L2 norm) of vec2: ||B||
	mag2 := 0.0
	for i := range vec2 {
		mag2 += vec2[i] * vec2[i]
	}
	mag2 = math.Sqrt(mag2)
	
	// Handle zero vectors
	if mag1 == 0.0 || mag2 == 0.0 {
		return 0.0
	}
	
	// Cosine similarity: (A · B) / (||A|| * ||B||)
	similarity := dotProduct / (mag1 * mag2)
	
	// Clamp to [-1, 1] range (should already be in range, but ensure it)
	if similarity > 1.0 {
		return 1.0
	}
	if similarity < -1.0 {
		return -1.0
	}
	
	return similarity
}

