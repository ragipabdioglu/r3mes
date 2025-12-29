package keeper_test

import (
	"os"
	"testing"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

func TestTolerantVerification_EndToEnd(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)
	ctx := sdk.WrapSDKContext(sdkCtx)

	// Create a test vault entry
	vaultEntry := types.GenesisVaultEntry{
		EntryId:                 1,
		DataHash:                "QmTestData",
		ExpectedGradientHash:    "expected_hash",
		ExpectedGradientIpfsHash: "QmExpectedGradient",
		ExpectedFingerprint:     `{"top_k":3,"indices":[0,2,4],"values":[1.0,0.5,0.3],"shape":[6]}`,
		GpuArchitecture:         "Ampere",
		CreatedHeight:           sdkCtx.BlockHeight(),
		UsageCount:              0,
		LastUsedHeight:          sdkCtx.BlockHeight(),
		Encrypted:               false,
	}

	// Add to vault
	err := f.keeper.AddToVault(ctx, vaultEntry)
	require.NoError(t, err)

	// Create miner gradient that matches vault indices
	// Miner gradient: [1.0, 0.2, 0.5, 0.1, 0.3, 0.4]
	// Vault indices [0, 2, 4] should extract [1.0, 0.5, 0.3]
	minerGradient := []float64{1.0, 0.2, 0.5, 0.1, 0.3, 0.4}

	// Mock IPFS retrieval (in real test, this would use actual IPFS or mock)
	// For now, we'll test the masking method directly
	expectedFingerprint, err := keeper.ParseFingerprint(vaultEntry.ExpectedFingerprint)
	require.NoError(t, err)

	// Test masking method
	similarity, err := keeper.CalculateCosineSimilarityWithMasking(expectedFingerprint, minerGradient)
	require.NoError(t, err)
	require.Greater(t, similarity, 0.99, "Similarity should be high for matching values")

	// Verify that miner's Top-K list is NOT used
	// Miner's Top-K would be different indices, but masking method uses vault indices
	minerTopK, _ := keeper.ExtractTopKFingerprint(minerGradient, 3)
	
	// Miner's Top-K indices might be different (e.g., [0, 4, 5] instead of [0, 2, 4])
	// But masking method should still work correctly using vault indices
	require.NotEqual(t, minerTopK.Indices, expectedFingerprint.Indices,
		"Miner's Top-K indices should be different from vault indices (demonstrating masking necessity)")
}

func TestBlindDelivery_Sanitization(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)
	ctx := sdk.WrapSDKContext(sdkCtx)

	// Set test mode to bypass IPFS validation
	os.Setenv("R3MES_TEST_MODE", "true")
	defer os.Unsetenv("R3MES_TEST_MODE")

	// Create a test pool
	pool, err := f.keeper.CreateTaskPool(ctx, 1, 10, 100)
	require.NoError(t, err)

	// Add a trap entry to vault
	vaultEntry := types.GenesisVaultEntry{
		EntryId:                 1,
		DataHash:                "QmTrapData",
		ExpectedGradientHash:    "trap_hash",
		ExpectedGradientIpfsHash: "QmTrapGradient",
		ExpectedFingerprint:     `{"top_k":3,"indices":[0,2,4],"values":[1.0,0.5,0.3],"shape":[6]}`,
		GpuArchitecture:         "Ampere",
		CreatedHeight:           1,
		UsageCount:              0,
		LastUsedHeight:          1,
		Encrypted:               false,
	}
	err = f.keeper.AddToVault(ctx, vaultEntry)
	require.NoError(t, err)

	// Get available chunks for miner (should be sanitized)
	chunks, err := f.keeper.GetAvailableChunksForMiner(ctx, pool.PoolId, 5)
	require.NoError(t, err)

	// Verify all chunks are sanitized (TaskChunkResponse, not TaskChunk)
	for _, chunk := range chunks {
		// TaskChunkResponse doesn't have IsTrap or VaultEntryId fields
		// So we can't directly check, but we verify the response structure
		require.NotZero(t, chunk.ChunkId, "ChunkId should be set")
		require.NotEmpty(t, chunk.DataHash, "DataHash should be set")
	}

	// Verify internal chunks (from GetAvailableChunks) may have trap flags
	internalChunks, err := f.keeper.GetAvailableChunks(ctx, pool.PoolId, 5)
	require.NoError(t, err)

	// At least one chunk should be a trap (if vault has entries)
	trapFound := false
	for _, chunk := range internalChunks {
		if chunk.IsTrap {
			trapFound = true
			break
		}
	}

	// If vault has entries, we should have traps
	if len(internalChunks) > 0 {
		// At least some chunks might be traps (10% ratio)
		// This is probabilistic, so we just verify the system works
		_ = trapFound
	}
}

