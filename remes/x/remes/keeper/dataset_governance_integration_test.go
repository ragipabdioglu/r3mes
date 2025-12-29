//go:build integration
// +build integration

package keeper_test

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

// TestDatasetIntegrity_IPFSIntegration tests dataset integrity verification
// with a real IPFS node (if available).
// This test requires:
//   - IPFS node running locally (default: /ip4/127.0.0.1/tcp/5001)
//   - R3MES_TEST_MODE=true (to allow test execution)
//
// To run this test:
//   go test -tags=integration -v ./x/remes/keeper -run TestDatasetIntegrity_IPFSIntegration
func TestDatasetIntegrity_IPFSIntegration(t *testing.T) {
	// Skip if not running integration tests
	if os.Getenv("RUN_INTEGRATION_TESTS") != "true" {
		t.Skip("Skipping integration test. Set RUN_INTEGRATION_TESTS=true to run.")
	}

	// Set test mode
	os.Setenv("R3MES_TEST_MODE", "true")
	defer os.Unsetenv("R3MES_TEST_MODE")

	// Get IPFS API URL from environment or use default
	ipfsAPIURL := os.Getenv("IPFS_API_URL")
	if ipfsAPIURL == "" {
		ipfsAPIURL = "/ip4/127.0.0.1/tcp/5001"
	}

	// Initialize keeper with IPFS manager
	f := initFixture(t)
	
	// Create IPFS manager
	ipfsManager, err := keeper.NewIPFSManager(ipfsAPIURL)
	if err != nil {
		t.Skipf("IPFS node not available at %s: %v. Skipping integration test.", ipfsAPIURL, err)
	}

	// Check if IPFS is actually available
	if !ipfsManager.IsAvailable() {
		t.Skipf("IPFS node not available at %s. Skipping integration test.", ipfsAPIURL)
	}

	// Set IPFS manager in keeper
	f.keeper.IpfsManager = ipfsManager

	// Create a test dataset
	testDatasetContent := []byte(`{
		"version": "1.0",
		"samples": [
			{"id": 1, "text": "Sample 1", "label": "A"},
			{"id": 2, "text": "Sample 2", "label": "B"},
			{"id": 3, "text": "Sample 3", "label": "C"}
		]
	}`)

	// Upload to IPFS (this would normally be done by the dataset proposer)
	// For integration test, we'll simulate by checking if we can retrieve content
	ctx := context.Background()

	// Calculate expected checksum
	expectedChecksum := keeper.CalculateDatasetChecksum(testDatasetContent)

	// Note: In a real integration test, you would:
	// 1. Upload testDatasetContent to IPFS using ipfsManager
	// 2. Get the IPFS hash
	// 3. Verify the integrity using VerifyDatasetIntegrity
	//
	// For now, we'll test that the IPFS manager can connect and is functional
	t.Run("IPFS_Manager_Available", func(t *testing.T) {
		require.True(t, ipfsManager.IsAvailable(), "IPFS manager should be available")
	})

	t.Run("Checksum_Calculation", func(t *testing.T) {
		checksum := keeper.CalculateDatasetChecksum(testDatasetContent)
		require.NotEmpty(t, checksum, "Checksum should not be empty")
		require.Len(t, checksum, 64, "SHA256 checksum should be 64 characters (hex)")

		// Same content should produce same checksum
		checksum2 := keeper.CalculateDatasetChecksum(testDatasetContent)
		require.Equal(t, checksum, checksum2, "Same content should produce same checksum")

		// Different content should produce different checksum
		differentContent := []byte("different content")
		checksum3 := keeper.CalculateDatasetChecksum(differentContent)
		require.NotEqual(t, checksum, checksum3, "Different content should produce different checksum")
	})

	t.Run("VerifyDatasetIntegrity_WithIPFS", func(t *testing.T) {
		// This test requires actual IPFS content
		// For a complete integration test, you would:
		// 1. Upload content to IPFS
		// 2. Get the IPFS hash
		// 3. Call VerifyDatasetIntegrity with the hash and checksum
		//
		// Example (if you have a test IPFS hash):
		// testIPFSHash := "QmTestHash..." // Replace with actual IPFS hash
		// valid, err := f.keeper.VerifyDatasetIntegrity(ctx, testIPFSHash, expectedChecksum)
		// require.NoError(t, err, "Verification should succeed")
		// require.True(t, valid, "Dataset integrity should be valid")

		// For now, we'll just verify that the function doesn't panic
		// and returns an error for invalid hash (which is expected)
		invalidHash := "invalid_hash"
		valid, err := f.keeper.VerifyDatasetIntegrity(ctx, invalidHash, expectedChecksum)
		require.False(t, valid, "Invalid hash should return false")
		require.Error(t, err, "Invalid hash should return error")
	})

	t.Run("IPFS_Manager_RetrieveContent", func(t *testing.T) {
		// Test that IPFS manager can attempt to retrieve content
		// (This will fail if content doesn't exist, but should not panic)
		nonExistentHash := "QmNonExistentHash123456789"
		content, err := ipfsManager.RetrieveContent(ctx, nonExistentHash)
		
		// Either content is nil (not found) or error is returned
		// Both are acceptable behaviors
		if err != nil {
			// Error is expected for non-existent content
			require.Contains(t, err.Error(), "not found", "Error should indicate content not found")
		} else {
			// If no error, content should be nil
			require.Nil(t, content, "Non-existent content should return nil")
		}
	})
}

// TestDatasetGovernance_ProductionMode tests that dataset governance
// enforces fail-closed behavior in production mode
func TestDatasetGovernance_ProductionMode(t *testing.T) {
	// Set production mode (no R3MES_TEST_MODE)
	os.Unsetenv("R3MES_TEST_MODE")
	defer os.Setenv("R3MES_TEST_MODE", "true")

	f := initFixture(t)
	
	// Keeper without IPFS manager (simulating production misconfiguration)
	// Note: initFixture creates keeper with empty IPFS URL, so ipfsManager is nil

	ctx := context.Background()
	datasetIPFSHash := "QmTest123"
	expectedChecksum := "abc123"

	// In production mode, should fail (not return true) when IPFS manager is nil
	valid, err := f.keeper.VerifyDatasetIntegrity(ctx, datasetIPFSHash, expectedChecksum)

	require.False(t, valid, "Should return false when IPFS manager is not configured in production")
	require.Error(t, err, "Should return error when IPFS manager is not configured in production")
	require.Contains(t, err.Error(), "IPFS manager not configured - dataset verification required in production")
}

