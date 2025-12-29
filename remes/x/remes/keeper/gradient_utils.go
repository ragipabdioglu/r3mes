package keeper

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"sort"

	"remes/x/remes/types"
)

// DeserializeGradientTensor deserializes gradient tensor from IPFS data
// Supports both protobuf and pickle+gzip formats
// Returns flattened gradient as float64 array
func DeserializeGradientTensor(ipfsData []byte) ([]float64, error) {
	// Try protobuf format first (preferred)
	if gradient, err := deserializeProtobufGradient(ipfsData); err == nil {
		return gradient, nil
	}

	// Try pickle+gzip format (fallback)
	if gradient, err := deserializePickleGradient(ipfsData); err == nil {
		return gradient, nil
	}

	return nil, fmt.Errorf("unsupported gradient format: neither protobuf nor pickle+gzip")
}

// deserializeProtobufGradient deserializes gradient from protobuf format
// Uses Python helper script to deserialize protobuf data
// In the future, this could be optimized by generating Go protobuf code from gradient.proto
func deserializeProtobufGradient(data []byte) ([]float64, error) {
	// Get project root from environment variable or use default relative path
	projectRoot := os.Getenv("R3MES_ROOT")
	if projectRoot == "" {
		// Default: assume we're in remes/ directory, go up to R3MES root
		projectRoot = "../.."
	}

	scriptPath := filepath.Join(projectRoot, "miner-engine", "utils", "deserialize_gradient.py")

	// Resolve absolute path
	absScriptPath, err := filepath.Abs(scriptPath)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve script path: %w", err)
	}

	// Check if script exists
	if _, err := os.Stat(absScriptPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("Python deserialization script not found at %s. Set R3MES_ROOT environment variable to project root", absScriptPath)
	}

	// Execute Python script with protobuf format
	cmd := exec.Command("python3", absScriptPath, "protobuf")
	cmd.Stdin = bytes.NewReader(data)

	output, err := cmd.Output()
	if err != nil {
		// Try to get stderr for better error message
		if exitError, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("python protobuf deserialization failed: %s, stderr: %s", err.Error(), string(exitError.Stderr))
		}
		return nil, fmt.Errorf("python protobuf deserialization failed: %w", err)
	}

	// Parse JSON output (array of floats)
	var result []float64
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON output from Python script: %w", err)
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("deserialized gradient is empty")
	}

	return result, nil
}

// deserializePickleGradient deserializes gradient from pickle+gzip format
// Uses Python helper script to deserialize pickle+gzip data
func deserializePickleGradient(data []byte) ([]float64, error) {
	// Get project root from environment variable or use default relative path
	// R3MES_ROOT can be set to point to the project root directory
	projectRoot := os.Getenv("R3MES_ROOT")
	if projectRoot == "" {
		// Default: assume we're in remes/ directory, go up to R3MES root
		projectRoot = "../.."
	}

	scriptPath := filepath.Join(projectRoot, "miner-engine", "utils", "deserialize_gradient.py")

	// Resolve absolute path
	absScriptPath, err := filepath.Abs(scriptPath)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve script path: %w", err)
	}

	// Check if script exists
	if _, err := os.Stat(absScriptPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("Python deserialization script not found at %s. Set R3MES_ROOT environment variable to project root", absScriptPath)
	}

	// Execute Python script with pickle format
	cmd := exec.Command("python3", absScriptPath, "pickle")
	cmd.Stdin = bytes.NewReader(data)

	output, err := cmd.Output()
	if err != nil {
		// Try to get stderr for better error message
		if exitError, ok := err.(*exec.ExitError); ok {
			return nil, fmt.Errorf("python deserialization failed: %s, stderr: %s", err.Error(), string(exitError.Stderr))
		}
		return nil, fmt.Errorf("python deserialization failed: %w", err)
	}

	// Parse JSON output (array of floats)
	var result []float64
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, fmt.Errorf("failed to parse JSON output from Python script: %w", err)
	}

	if len(result) == 0 {
		return nil, fmt.Errorf("deserialized gradient is empty")
	}

	return result, nil
}

// ExtractTopKFingerprint extracts Top-K fingerprint from gradient tensor
// Returns GradientFingerprint with indices and values (both REQUIRED, same order)
// Exported for testing
func ExtractTopKFingerprint(gradientTensor []float64, k int) (types.GradientFingerprint, error) {
	if len(gradientTensor) == 0 {
		return types.GradientFingerprint{}, fmt.Errorf("gradient tensor is empty")
	}

	if k <= 0 {
		return types.GradientFingerprint{}, fmt.Errorf("k must be positive, got %d", k)
	}

	// Cap k to tensor size
	if k > len(gradientTensor) {
		k = len(gradientTensor)
	}

	// Create index-value pairs with absolute values for sorting
	type indexedValue struct {
		index    int
		value    float64
		absValue float64
	}

	indexedValues := make([]indexedValue, len(gradientTensor))
	for i, val := range gradientTensor {
		indexedValues[i] = indexedValue{
			index:    i,
			value:    val,
			absValue: math.Abs(val),
		}
	}

	// Sort by absolute value (descending)
	sort.Slice(indexedValues, func(i, j int) bool {
		return indexedValues[i].absValue > indexedValues[j].absValue
	})

	// Extract Top-K
	topK := indexedValues[:k]

	// Separate indices and values (REQUIRED - same order)
	indices := make([]int, k)
	values := make([]float64, k)
	for i, entry := range topK {
		indices[i] = entry.index // REQUIRED - position in original tensor
		values[i] = entry.value  // REQUIRED - value at that position (same order as indices)
	}

	// Create fingerprint (shape is not critical for cosine similarity, but useful for debugging)
	fingerprint := types.GradientFingerprint{
		TopK:    k,
		Indices: indices,                             // REQUIRED
		Values:  values,                              // REQUIRED - same order as indices
		Shape:   []int64{int64(len(gradientTensor))}, // Simplified shape (1D)
	}

	return fingerprint, nil
}

// ParseFingerprint parses fingerprint JSON string and validates structure
// Exported for testing
func ParseFingerprint(jsonStr string) (types.GradientFingerprint, error) {
	var fp types.GradientFingerprint

	if err := json.Unmarshal([]byte(jsonStr), &fp); err != nil {
		return types.GradientFingerprint{}, fmt.Errorf("failed to parse fingerprint JSON: %w", err)
	}

	// Validate that indices and values exist and have same length (REQUIRED)
	if len(fp.Indices) == 0 {
		return types.GradientFingerprint{}, fmt.Errorf("fingerprint indices array is empty")
	}
	if len(fp.Values) == 0 {
		return types.GradientFingerprint{}, fmt.Errorf("fingerprint values array is empty")
	}
	if len(fp.Indices) != len(fp.Values) {
		return types.GradientFingerprint{}, fmt.Errorf("fingerprint indices and values arrays have different lengths: %d vs %d", len(fp.Indices), len(fp.Values))
	}

	return fp, nil
}

// SerializeFingerprint serializes fingerprint to JSON string
func SerializeFingerprint(fp types.GradientFingerprint) (string, error) {
	data, err := json.Marshal(fp)
	if err != nil {
		return "", fmt.Errorf("failed to serialize fingerprint to JSON: %w", err)
	}
	return string(data), nil
}
