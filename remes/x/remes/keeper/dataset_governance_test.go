package keeper

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"remes/x/remes/types"
)

func TestVerifyDatasetIntegrity_FailClosed(t *testing.T) {
	// Test that VerifyDatasetIntegrity fails when IPFS manager is not configured
	// This is a security-critical test: fail-closed behavior
	
	k := Keeper{} // Keeper without IPFS manager
	
	ctx := context.Background()
	datasetIPFSHash := "QmTest123"
	expectedChecksum := "abc123"
	
	// Should fail (not return true) when IPFS manager is nil
	// In test mode, it should return false with error
	os.Setenv("R3MES_TEST_MODE", "true")
	defer os.Unsetenv("R3MES_TEST_MODE")
	
	valid, err := k.VerifyDatasetIntegrity(ctx, datasetIPFSHash, expectedChecksum)
	
	require.False(t, valid, "Should return false when IPFS manager is not configured")
	require.Error(t, err, "Should return error when IPFS manager is not configured")
	require.Contains(t, err.Error(), "IPFS manager not configured")
}

func TestVerifyDatasetIntegrity_EmptyHash(t *testing.T) {
	k := Keeper{}
	ctx := context.Background()
	
	// Empty IPFS hash should fail
	valid, err := k.VerifyDatasetIntegrity(ctx, "", "abc123")
	require.False(t, valid)
	require.Error(t, err)
	require.Contains(t, err.Error(), "dataset IPFS hash cannot be empty")
}

func TestVerifyDatasetIntegrity_EmptyChecksum(t *testing.T) {
	k := Keeper{}
	ctx := context.Background()
	
	// Empty checksum should fail
	valid, err := k.VerifyDatasetIntegrity(ctx, "QmTest123", "")
	require.False(t, valid)
	require.Error(t, err)
	require.Contains(t, err.Error(), "expected checksum cannot be empty")
}

func TestCalculateDatasetChecksum(t *testing.T) {
	// Test checksum calculation
	testData := []byte("test dataset content")
	checksum := CalculateDatasetChecksum(testData)
	
	require.NotEmpty(t, checksum)
	require.Len(t, checksum, 64) // SHA256 hex string length
	
	// Same data should produce same checksum
	checksum2 := CalculateDatasetChecksum(testData)
	require.Equal(t, checksum, checksum2)
	
	// Different data should produce different checksum
	testData2 := []byte("different content")
	checksum3 := CalculateDatasetChecksum(testData2)
	require.NotEqual(t, checksum, checksum3)
}

func TestValidateDatasetMetadata(t *testing.T) {
	k := Keeper{}
	
	// Valid metadata
	validMetadata := types.DatasetMetadata{
		Name:        "Test Dataset",
		Description: "Test description",
		SizeBytes:   1000,
		NumSamples:  100,
		Checksum:    "abc123",
	}
	err := k.ValidateDatasetMetadata(validMetadata)
	require.NoError(t, err)
	
	// Invalid: empty name
	invalidMetadata := validMetadata
	invalidMetadata.Name = ""
	err = k.ValidateDatasetMetadata(invalidMetadata)
	require.Error(t, err)
	require.Contains(t, err.Error(), "dataset name cannot be empty")
	
	// Invalid: empty description
	invalidMetadata = validMetadata
	invalidMetadata.Description = ""
	err = k.ValidateDatasetMetadata(invalidMetadata)
	require.Error(t, err)
	require.Contains(t, err.Error(), "dataset description cannot be empty")
	
	// Invalid: zero size
	invalidMetadata = validMetadata
	invalidMetadata.SizeBytes = 0
	err = k.ValidateDatasetMetadata(invalidMetadata)
	require.Error(t, err)
	require.Contains(t, err.Error(), "dataset size cannot be zero")
	
	// Invalid: zero samples
	invalidMetadata = validMetadata
	invalidMetadata.NumSamples = 0
	err = k.ValidateDatasetMetadata(invalidMetadata)
	require.Error(t, err)
	require.Contains(t, err.Error(), "must have at least one sample")
	
	// Invalid: empty checksum
	invalidMetadata = validMetadata
	invalidMetadata.Checksum = ""
	err = k.ValidateDatasetMetadata(invalidMetadata)
	require.Error(t, err)
	require.Contains(t, err.Error(), "dataset checksum is required")
}

