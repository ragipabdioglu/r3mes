package keeper

import (
	"math"
	"testing"

	"remes/x/remes/types"
)

func TestExtractTopKFingerprint(t *testing.T) {
	// Create test gradient tensor
	gradient := []float64{0.1, 0.5, 0.3, 0.9, 0.2, 0.8, 0.4, 0.6, 0.7, 0.05}
	k := 3

	// Extract Top-K fingerprint
	fingerprint, err := ExtractTopKFingerprint(gradient, k)
	if err != nil {
		t.Fatalf("Failed to extract fingerprint: %v", err)
	}

	// Verify Top-K count
	if fingerprint.TopK != k {
		t.Errorf("Expected TopK=%d, got %d", k, fingerprint.TopK)
	}

	// Verify indices and values arrays have same length (REQUIRED)
	if len(fingerprint.Indices) != len(fingerprint.Values) {
		t.Errorf("Indices and Values arrays must have same length: indices=%d, values=%d", len(fingerprint.Indices), len(fingerprint.Values))
	}

	// Verify indices and values are ZORUNLU (not empty)
	if len(fingerprint.Indices) == 0 {
		t.Error("Indices array is empty (ZORUNLU)")
	}
	if len(fingerprint.Values) == 0 {
		t.Error("Values array is empty (ZORUNLU)")
	}

	// Verify Top-K values are correct (sorted by absolute value)
	// Expected: indices [3, 5, 8] with values [0.9, 0.8, 0.7] (largest absolute values)
	expectedIndices := []int{3, 5, 8} // Positions of 0.9, 0.8, 0.7
	expectedValues := []float64{0.9, 0.8, 0.7}

	// Check that indices match expected positions
	for i, idx := range fingerprint.Indices {
		if idx != expectedIndices[i] {
			t.Errorf("Index mismatch at position %d: expected %d, got %d", i, expectedIndices[i], idx)
		}
		// Verify value at that index matches
		if math.Abs(fingerprint.Values[i]-expectedValues[i]) > 0.001 {
			t.Errorf("Value mismatch at position %d: expected %.3f, got %.3f", i, expectedValues[i], fingerprint.Values[i])
		}
	}
}

func TestParseFingerprint(t *testing.T) {
	// Valid fingerprint JSON
	validJSON := `{"top_k":3,"indices":[3,5,8],"values":[0.9,0.8,0.7],"shape":[10]}`
	
	fingerprint, err := ParseFingerprint(validJSON)
	if err != nil {
		t.Fatalf("Failed to parse valid fingerprint: %v", err)
	}

	// Verify structure
	if fingerprint.TopK != 3 {
		t.Errorf("Expected TopK=3, got %d", fingerprint.TopK)
	}
	if len(fingerprint.Indices) != 3 || len(fingerprint.Values) != 3 {
		t.Errorf("Expected 3 indices and values, got %d indices, %d values", len(fingerprint.Indices), len(fingerprint.Values))
	}

	// Invalid fingerprint (missing indices)
	invalidJSON1 := `{"top_k":3,"values":[0.9,0.8,0.7],"shape":[10]}`
	_, err = ParseFingerprint(invalidJSON1)
	if err == nil {
		t.Error("Expected error for missing indices array")
	}

	// Invalid fingerprint (indices and values length mismatch)
	invalidJSON2 := `{"top_k":3,"indices":[3,5],"values":[0.9,0.8,0.7],"shape":[10]}`
	_, err = ParseFingerprint(invalidJSON2)
	if err == nil {
		t.Error("Expected error for indices/values length mismatch")
	}
}

func TestSerializeFingerprint(t *testing.T) {
	fingerprint := types.GradientFingerprint{
		TopK:    3,
		Indices: []int{3, 5, 8},
		Values:  []float64{0.9, 0.8, 0.7},
		Shape:   []int64{10},
	}

	jsonStr, err := SerializeFingerprint(fingerprint)
	if err != nil {
		t.Fatalf("Failed to serialize fingerprint: %v", err)
	}

	// Parse back and verify
	parsed, err := ParseFingerprint(jsonStr)
	if err != nil {
		t.Fatalf("Failed to parse serialized fingerprint: %v", err)
	}

	if parsed.TopK != fingerprint.TopK {
		t.Errorf("TopK mismatch: expected %d, got %d", fingerprint.TopK, parsed.TopK)
	}
	if len(parsed.Indices) != len(fingerprint.Indices) {
		t.Errorf("Indices length mismatch: expected %d, got %d", len(fingerprint.Indices), len(parsed.Indices))
	}
}

