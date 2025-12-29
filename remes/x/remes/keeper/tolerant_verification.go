package keeper

import (
	"context"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
)

// SimilarityResult represents the result of gradient verification
type SimilarityResult struct {
	SimilarityScore      float64 // 0.0-1.0
	IsValid              bool
	Reason               string // "exact_hash_match" | "tolerant_cosine_match_masked" | "low_similarity_fraud"
	GPUArchitectureMatch bool
}

// VerifyGradientTolerant performs tolerant verification of a miner's gradient
// Uses two-step verification:
// 1. Exact hash match (same GPU, bit-exact verification)
// 2. Tolerant match using masking method (cosine similarity with vault indices)
func (k Keeper) VerifyGradientTolerant(
	ctx context.Context,
	minerGradientHash string,
	minerGradientIPFSHash string,
	expectedGradientHash string,
	expectedFingerprintJSON string,
	minerGPU string,
	expectedGPU string,
) (SimilarityResult, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	
	// Step 1: Exact Hash Match (Same GPU)
	if minerGPU == expectedGPU && minerGradientHash == expectedGradientHash {
		sdkCtx.Logger().Info(fmt.Sprintf("Exact hash match: miner %s, GPU %s", minerGradientHash[:16], minerGPU))
		return SimilarityResult{
			SimilarityScore:      1.0,
			IsValid:              true,
			Reason:               "exact_hash_match",
			GPUArchitectureMatch: true,
		}, nil
	}
	
	// Step 2: Tolerant Match (Cosine Similarity with Masking)
	// Download miner's full gradient from IPFS
	minerFullGradient, err := k.ipfsManager.RetrieveGradientTensor(ctx, minerGradientIPFSHash)
	if err != nil {
		return SimilarityResult{
			SimilarityScore: 0.0,
			IsValid:         false,
			Reason:          fmt.Sprintf("failed to retrieve miner gradient from IPFS: %v", err),
		}, fmt.Errorf("failed to retrieve miner gradient: %w", err)
	}
	
	// Parse expected fingerprint
	expectedFingerprint, err := ParseFingerprint(expectedFingerprintJSON)
	if err != nil {
		return SimilarityResult{
			SimilarityScore: 0.0,
			IsValid:         false,
			Reason:          fmt.Sprintf("failed to parse expected fingerprint: %v", err),
		}, fmt.Errorf("failed to parse expected fingerprint: %w", err)
	}
	
	// Calculate cosine similarity using masking method
	similarity, err := CalculateCosineSimilarityWithMasking(expectedFingerprint, minerFullGradient)
	if err != nil {
		return SimilarityResult{
			SimilarityScore: 0.0,
			IsValid:         false,
			Reason:          fmt.Sprintf("failed to calculate cosine similarity: %v", err),
		}, fmt.Errorf("failed to calculate cosine similarity: %w", err)
	}
	
	// Apply threshold based on GPU architecture match
	threshold := getSimilarityThreshold(minerGPU == expectedGPU)
	
	sdkCtx.Logger().Info(fmt.Sprintf("Cosine similarity: %.6f (threshold: %.6f), miner GPU: %s, expected GPU: %s", similarity, threshold, minerGPU, expectedGPU))
	
	isValid := similarity >= threshold
	reason := "tolerant_cosine_match_masked"
	if !isValid {
		reason = "low_similarity_fraud"
	}
	
	return SimilarityResult{
		SimilarityScore:      similarity,
		IsValid:              isValid,
		Reason:               reason,
		GPUArchitectureMatch: minerGPU == expectedGPU,
	}, nil
}

// getSimilarityThreshold returns the similarity threshold based on GPU architecture match
func getSimilarityThreshold(sameGPU bool) float64 {
	if sameGPU {
		return 0.999 // Higher threshold for same GPU
	}
	return 0.95 // Lower threshold for different GPU (allows hardware differences)
}

// RetrieveGradientTensor retrieves and deserializes gradient tensor from IPFS
// Uses caching to avoid repeated downloads
func (k Keeper) RetrieveGradientTensor(ctx context.Context, ipfsHash string) ([]float64, error) {
	// Check cache first
	if cached, found := k.gradientCache.Get(ipfsHash); found {
		return cached, nil
	}
	
	// Retrieve raw data from IPFS
	data, err := k.ipfsManager.RetrieveContent(ctx, ipfsHash)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve gradient from IPFS: %w", err)
	}
	
	// Deserialize gradient tensor
	gradient, err := DeserializeGradientTensor(data)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize gradient tensor: %w", err)
	}
	
	// Cache the gradient
	k.gradientCache.Set(ipfsHash, gradient)
	
	return gradient, nil
}

