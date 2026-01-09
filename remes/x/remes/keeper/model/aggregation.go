package model

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math/big"
	"sort"

	"remes/x/remes/types"
)

// KRİTİK EKSİKLİK #5 ÇÖZÜMÜ: Gradient Aggregation

// AggregationMethod defines the aggregation algorithm
type AggregationMethod string

const (
	// WeightedAverage uses trust-weighted averaging
	WeightedAverage AggregationMethod = "weighted_average"
	// TrimmedMean removes outliers before averaging (Byzantine-robust)
	TrimmedMean AggregationMethod = "trimmed_mean"
	// Median uses coordinate-wise median (Byzantine-robust)
	Median AggregationMethod = "median"
)

// AggregationConfig holds configuration for gradient aggregation
type AggregationConfig struct {
	Method            AggregationMethod
	ByzantineThreshold float64 // Fraction of gradients to trim (for TrimmedMean)
	MinGradients      int     // Minimum gradients required for aggregation
	MaxGradients      int     // Maximum gradients to include
}

// DefaultAggregationConfig returns default aggregation configuration
func DefaultAggregationConfig() AggregationConfig {
	return AggregationConfig{
		Method:            TrimmedMean,
		ByzantineThreshold: 0.2, // Trim 20% from each end
		MinGradients:      3,
		MaxGradients:      100,
	}
}

// GradientInfo holds gradient metadata for aggregation
type GradientInfo struct {
	GradientID   uint64
	MinerAddress string
	IPFSHash     string
	GradientHash string
	TrustScore   *big.Int // Miner's trust score (scaled by 1e18)
	Loss         *big.Int // Claimed loss (scaled by 1e18)
}

// AggregationResult holds the result of gradient aggregation
type AggregationResult struct {
	Success           bool
	AggregatedHash    string   // IPFS hash of aggregated gradient
	MerkleRoot        string   // Merkle root of included gradients
	IncludedGradients []uint64 // IDs of gradients included
	ExcludedGradients []uint64 // IDs of gradients excluded (outliers)
	TotalWeight       *big.Int // Total weight used in aggregation
	Error             string
}

// AggregateGradients performs gradient aggregation for a training round
func (k *ModelKeeper) AggregateGradients(
	ctx context.Context,
	trainingRoundID uint64,
	gradientIDs []uint64,
	config AggregationConfig,
) (*AggregationResult, error) {
	// Validate input
	if len(gradientIDs) < config.MinGradients {
		return &AggregationResult{
			Success: false,
			Error:   fmt.Sprintf("insufficient gradients: got %d, need %d", len(gradientIDs), config.MinGradients),
		}, nil
	}

	// Limit gradients if too many
	if len(gradientIDs) > config.MaxGradients {
		gradientIDs = gradientIDs[:config.MaxGradients]
	}

	// Collect gradient info
	gradients := make([]GradientInfo, 0, len(gradientIDs))
	for _, gid := range gradientIDs {
		info, err := k.getGradientInfo(ctx, gid)
		if err != nil {
			continue // Skip invalid gradients
		}
		gradients = append(gradients, info)
	}

	if len(gradients) < config.MinGradients {
		return &AggregationResult{
			Success: false,
			Error:   fmt.Sprintf("insufficient valid gradients: got %d, need %d", len(gradients), config.MinGradients),
		}, nil
	}

	// Apply aggregation method
	var included, excluded []uint64
	var weights []*big.Int

	switch config.Method {
	case TrimmedMean:
		included, excluded, weights = k.trimmedMeanSelection(gradients, config.ByzantineThreshold)
	case Median:
		included, excluded, weights = k.medianSelection(gradients)
	default: // WeightedAverage
		included, excluded, weights = k.weightedAverageSelection(gradients)
	}

	if len(included) == 0 {
		return &AggregationResult{
			Success:           false,
			ExcludedGradients: excluded,
			Error:             "no gradients passed selection",
		}, nil
	}

	// Compute Merkle root of included gradients
	merkleRoot := k.computeMerkleRoot(gradients, included)

	// Compute total weight
	totalWeight := big.NewInt(0)
	for _, w := range weights {
		totalWeight.Add(totalWeight, w)
	}

	// Note: Actual gradient aggregation (tensor math) happens off-chain
	// On-chain we only track which gradients are included and their weights
	// The aggregated gradient IPFS hash will be submitted by the proposer

	return &AggregationResult{
		Success:           true,
		MerkleRoot:        merkleRoot,
		IncludedGradients: included,
		ExcludedGradients: excluded,
		TotalWeight:       totalWeight,
	}, nil
}

// getGradientInfo retrieves gradient information from storage
func (k *ModelKeeper) getGradientInfo(ctx context.Context, gradientID uint64) (GradientInfo, error) {
	// This would query the stored gradient from the gradient keeper
	// For now, return a placeholder
	return GradientInfo{
		GradientID:   gradientID,
		TrustScore:   big.NewInt(1e18), // Default trust score
		Loss:         big.NewInt(0),
	}, nil
}

// trimmedMeanSelection selects gradients using trimmed mean (Byzantine-robust)
func (k *ModelKeeper) trimmedMeanSelection(
	gradients []GradientInfo,
	trimRatio float64,
) (included, excluded []uint64, weights []*big.Int) {
	n := len(gradients)
	trimCount := int(float64(n) * trimRatio)

	// Sort by loss (ascending) - trim highest and lowest loss
	sorted := make([]GradientInfo, len(gradients))
	copy(sorted, gradients)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Loss.Cmp(sorted[j].Loss) < 0
	})

	// Exclude trimmed gradients
	for i := 0; i < trimCount && i < len(sorted); i++ {
		excluded = append(excluded, sorted[i].GradientID)
	}
	for i := n - trimCount; i < n && i >= 0; i++ {
		excluded = append(excluded, sorted[i].GradientID)
	}

	// Include middle gradients with trust-weighted weights
	for i := trimCount; i < n-trimCount && i < len(sorted); i++ {
		included = append(included, sorted[i].GradientID)
		weights = append(weights, sorted[i].TrustScore)
	}

	return
}

// medianSelection selects gradients using median (most Byzantine-robust)
func (k *ModelKeeper) medianSelection(
	gradients []GradientInfo,
) (included, excluded []uint64, weights []*big.Int) {
	n := len(gradients)

	// Sort by loss
	sorted := make([]GradientInfo, len(gradients))
	copy(sorted, gradients)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Loss.Cmp(sorted[j].Loss) < 0
	})

	// Select median gradient(s)
	if n%2 == 1 {
		// Odd: single median
		medianIdx := n / 2
		included = append(included, sorted[medianIdx].GradientID)
		weights = append(weights, big.NewInt(1e18))

		// Exclude others
		for i, g := range sorted {
			if i != medianIdx {
				excluded = append(excluded, g.GradientID)
			}
		}
	} else {
		// Even: two middle values
		midLow := n/2 - 1
		midHigh := n / 2
		included = append(included, sorted[midLow].GradientID, sorted[midHigh].GradientID)
		weights = append(weights, big.NewInt(5e17), big.NewInt(5e17)) // Equal weight

		for i, g := range sorted {
			if i != midLow && i != midHigh {
				excluded = append(excluded, g.GradientID)
			}
		}
	}

	return
}

// weightedAverageSelection includes all gradients with trust-based weights
func (k *ModelKeeper) weightedAverageSelection(
	gradients []GradientInfo,
) (included, excluded []uint64, weights []*big.Int) {
	for _, g := range gradients {
		included = append(included, g.GradientID)
		weights = append(weights, g.TrustScore)
	}
	return
}

// computeMerkleRoot computes Merkle root of gradient hashes
func (k *ModelKeeper) computeMerkleRoot(gradients []GradientInfo, includedIDs []uint64) string {
	// Create map for quick lookup
	includedSet := make(map[uint64]bool)
	for _, id := range includedIDs {
		includedSet[id] = true
	}

	// Collect hashes of included gradients
	var hashes [][]byte
	for _, g := range gradients {
		if includedSet[g.GradientID] {
			hashBytes, _ := hex.DecodeString(g.GradientHash)
			if len(hashBytes) == 0 {
				// Use gradient ID as fallback
				hashBytes = []byte(fmt.Sprintf("%d", g.GradientID))
			}
			hashes = append(hashes, hashBytes)
		}
	}

	if len(hashes) == 0 {
		return ""
	}

	// Build Merkle tree
	for len(hashes) > 1 {
		var nextLevel [][]byte
		for i := 0; i < len(hashes); i += 2 {
			var combined []byte
			combined = append(combined, hashes[i]...)
			if i+1 < len(hashes) {
				combined = append(combined, hashes[i+1]...)
			} else {
				combined = append(combined, hashes[i]...) // Duplicate last if odd
			}
			hash := sha256.Sum256(combined)
			nextLevel = append(nextLevel, hash[:])
		}
		hashes = nextLevel
	}

	return hex.EncodeToString(hashes[0])
}

// ValidateAggregation validates an aggregation result
func (k *ModelKeeper) ValidateAggregation(
	ctx context.Context,
	result *AggregationResult,
	submittedMerkleRoot string,
) error {
	if result.MerkleRoot != submittedMerkleRoot {
		return fmt.Errorf("merkle root mismatch: expected %s, got %s", result.MerkleRoot, submittedMerkleRoot)
	}
	return nil
}

// FinalizeAggregation finalizes the aggregation and updates model state
func (k *ModelKeeper) FinalizeAggregation(
	ctx context.Context,
	trainingRoundID uint64,
	aggregatedIPFSHash string,
	result *AggregationResult,
) error {
	// Update global model state with new aggregated gradient
	state, err := k.GetGlobalModelState(ctx)
	if err != nil {
		// Initialize if not exists
		state = types.GlobalModelState{}
	}

	state.TrainingRoundId = trainingRoundID
	state.ModelIpfsHash = aggregatedIPFSHash
	// Note: Actual model update would happen after aggregated gradient is applied

	return k.UpdateGlobalModelState(ctx, state)
}
