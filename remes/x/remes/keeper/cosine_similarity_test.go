package keeper_test

import (
	"math"
	"testing"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

func TestCalculateCosineSimilarityWithMasking(t *testing.T) {
	// Test case: Vault indices [0, 2, 4] with values [1.0, 0.5, 0.3]
	// Miner gradient: [1.0, 0.2, 0.5, 0.1, 0.3, 0.4]
	// Expected: Extract miner values at indices [0, 2, 4] = [1.0, 0.5, 0.3]
	// Cosine similarity should be high (close to 1.0)

	expectedFingerprint := types.GradientFingerprint{
		TopK:    3,
		Indices: []int{0, 2, 4}, // Vault indices
		Values:  []float64{1.0, 0.5, 0.3}, // Vault values
		Shape:   []int64{6},
	}

	minerFullGradient := []float64{1.0, 0.2, 0.5, 0.1, 0.3, 0.4}

	similarity, err := keeper.CalculateCosineSimilarityWithMasking(expectedFingerprint, minerFullGradient)
	if err != nil {
		t.Fatalf("Failed to calculate cosine similarity: %v", err)
	}

	// Expected similarity should be close to 1.0 (same values at same indices)
	if similarity < 0.99 {
		t.Errorf("Expected similarity >= 0.99, got %.6f", similarity)
	}

	// Test case: Different values (lower similarity)
	expectedFingerprint2 := types.GradientFingerprint{
		TopK:    3,
		Indices: []int{0, 2, 4},
		Values:  []float64{1.0, 0.5, 0.3},
		Shape:   []int64{6},
	}

	minerFullGradient2 := []float64{0.1, 0.2, 0.1, 0.1, 0.1, 0.4} // Different values

	similarity2, err := keeper.CalculateCosineSimilarityWithMasking(expectedFingerprint2, minerFullGradient2)
	if err != nil {
		t.Fatalf("Failed to calculate cosine similarity: %v", err)
	}

	// Expected similarity should be low (different values)
	if similarity2 > 0.5 {
		t.Errorf("Expected similarity < 0.5 for different values, got %.6f", similarity2)
	}
}

func TestCalculateCosineSimilarityWithMasking_IndexOutOfBounds(t *testing.T) {
	expectedFingerprint := types.GradientFingerprint{
		TopK:    3,
		Indices: []int{0, 2, 10}, // Index 10 out of bounds
		Values:  []float64{1.0, 0.5, 0.3},
		Shape:   []int64{6},
	}

	minerFullGradient := []float64{1.0, 0.2, 0.5, 0.1, 0.3, 0.4}

	_, err := keeper.CalculateCosineSimilarityWithMasking(expectedFingerprint, minerFullGradient)
	if err == nil {
		t.Error("Expected error for index out of bounds")
	}
}

func TestCalculateCosineSimilarityWithMasking_EmptyIndices(t *testing.T) {
	expectedFingerprint := types.GradientFingerprint{
		TopK:    0,
		Indices: []int{}, // Empty indices
		Values:  []float64{},
		Shape:   []int64{6},
	}

	minerFullGradient := []float64{1.0, 0.2, 0.5}

	_, err := keeper.CalculateCosineSimilarityWithMasking(expectedFingerprint, minerFullGradient)
	if err == nil {
		t.Error("Expected error for empty indices array")
	}
}

func TestCalculateCosineSimilarity(t *testing.T) {
	// Test cosine similarity calculation
	vec1 := []float64{1.0, 0.0, 0.0}
	vec2 := []float64{1.0, 0.0, 0.0}

	// Note: calculateCosineSimilarity is not exported, so we test via CalculateCosineSimilarityWithMasking
	// For direct testing, we'd need to export it or test indirectly
	expectedFingerprint := types.GradientFingerprint{
		TopK:    3,
		Indices: []int{0, 1, 2},
		Values:  vec1,
		Shape:   []int64{3},
	}
	similarity, _ := keeper.CalculateCosineSimilarityWithMasking(expectedFingerprint, vec2)
	if math.Abs(similarity-1.0) > 0.001 {
		t.Errorf("Expected similarity=1.0 for identical vectors, got %.6f", similarity)
	}

	// Test orthogonal vectors (should be 0.0)
	vec3 := []float64{1.0, 0.0, 0.0}
	vec4 := []float64{0.0, 1.0, 0.0}

	expectedFingerprint2 := types.GradientFingerprint{
		TopK:    2,
		Indices: []int{0, 1},
		Values:  vec3,
		Shape:   []int64{2},
	}
	similarity2, _ := keeper.CalculateCosineSimilarityWithMasking(expectedFingerprint2, vec4)
	if math.Abs(similarity2-0.0) > 0.001 {
		t.Errorf("Expected similarity=0.0 for orthogonal vectors, got %.6f", similarity2)
	}
}

