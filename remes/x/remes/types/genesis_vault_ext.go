package types

import (
	"encoding/json"
	"fmt"
)

// GradientFingerprint represents the Top-K fingerprint structure
// Both Indices and Values are REQUIRED for cosine similarity calculation using masking method
type GradientFingerprint struct {
	TopK    int       `json:"top_k"`
	Indices []int     `json:"indices"` // REQUIRED - positions in the gradient tensor
	Values  []float64 `json:"values"`  // REQUIRED - values at those positions (same order as indices)
	Shape   []int64   `json:"shape"`   // Original tensor shape
}

// GetFingerprint parses the expected_fingerprint JSON string and returns GradientFingerprint
func (e *GenesisVaultEntry) GetFingerprint() (GradientFingerprint, error) {
	if e.ExpectedFingerprint == "" {
		return GradientFingerprint{}, fmt.Errorf("expected_fingerprint is empty")
	}

	var fp GradientFingerprint
	err := json.Unmarshal([]byte(e.ExpectedFingerprint), &fp)
	if err != nil {
		return GradientFingerprint{}, fmt.Errorf("failed to parse fingerprint JSON: %w", err)
	}

	// Validate that indices and values exist and have same length
	if len(fp.Indices) == 0 {
		return GradientFingerprint{}, fmt.Errorf("fingerprint indices array is empty")
	}
	if len(fp.Values) == 0 {
		return GradientFingerprint{}, fmt.Errorf("fingerprint values array is empty")
	}
	if len(fp.Indices) != len(fp.Values) {
		return GradientFingerprint{}, fmt.Errorf("fingerprint indices and values arrays have different lengths: %d vs %d", len(fp.Indices), len(fp.Values))
	}

	return fp, nil
}

// IsExpired checks if the vault entry should be pruned
// An entry is considered expired if it hasn't been used for a long time
func (e *GenesisVaultEntry) IsExpired(currentHeight int64, expirationThreshold int64) bool {
	if e.UsageCount == 0 {
		// Never used entries can be expired if created long ago
		return (currentHeight - e.CreatedHeight) > expirationThreshold
	}
	// Used entries expire if not used recently
	return (currentHeight - e.LastUsedHeight) > expirationThreshold
}

// IncrementUsage increments the usage count and updates last used height
func (e *GenesisVaultEntry) IncrementUsage(currentHeight int64) {
	e.UsageCount++
	e.LastUsedHeight = currentHeight
}
