package keeper_test

import (
	"fmt"
	"testing"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

// TestPerformance_GradientSubmissionUnderLoad simulates a burst of gradient submissions
// to detect obvious performance regressions (panic, OOM, extreme latency).
//
// Not a formal benchmark, but a sanity check under moderate load.
func TestPerformance_GradientSubmissionUnderLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping performance test in short mode")
	}

	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)
	ctx := sdk.WrapSDKContext(sdkCtx)
	msgServer := keeper.NewMsgServerImpl(f.keeper)

	// 1. Initialize minimal genesis state
	genesisState := &types.GenesisState{
		Params:       types.DefaultParams(),
		ModelHash:    "QmPerfTestModelHash",
		ModelVersion: "b1.58",
	}
	err := f.keeper.InitGenesis(f.ctx, genesisState)
	require.NoError(t, err)

	// Initialize sequences manually for tests
	if err := f.keeper.StoredGradientID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize StoredGradientID sequence: %v", err)
	}

	// 2. Register a simple model (needed for SubmitGradient)
	modelConfig := types.ModelConfig{
		ModelType:          types.ModelType_MODEL_TYPE_BITNET,
		ModelVersion:       "b1.58",
		ArchitectureConfig: `{"hidden_size": 768, "num_layers": 2, "lora_rank": 4}`,
		ContainerHash:      "sha256:perf123",
		ContainerRegistry:  "docker.io",
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
	}

	modelID, err := f.keeper.RegisterModel(sdkCtx, modelConfig, 1)
	require.NoError(t, err)

	// 3. Simulate N gradient submissions
	const submissions = 200
	const totalShards = uint64(100)

	start := time.Now()

	for i := 0; i < submissions; i++ {
		minerAddr, err := f.addressCodec.BytesToString([]byte(fmt.Sprintf("miner-%06d", i)))
		require.NoError(t, err)

		ipfsHash := fmt.Sprintf("QmPerfTestHash%06d", i)
		gradientHash := fmt.Sprintf("perf_gradient_hash_%06d", i)

		// Calculate deterministic shard ID using same logic as production
		expectedShardID, err := f.keeper.CalculateShardID(sdkCtx, minerAddr, uint64(1), totalShards)
		require.NoError(t, err)

		msg := &types.MsgSubmitGradient{
			Miner:           minerAddr,
			IpfsHash:        ipfsHash,
			ModelVersion:    "b1.58",
			TrainingRoundId: uint64(1),
			ShardId:         expectedShardID,
			GradientHash:    gradientHash,
			ModelConfigId:   modelID,
			GpuArchitecture: "Ampere", // Valid architecture for whitelist validation
			// Minimal fields to pass validation; signature/nonce/IPFS are relaxed in tests via mocks.
			Nonce:            uint64(i + 1),
			Signature:        []byte("mock_signature"),
			ContainerHash:    "sha256:test123", // Required for container validation
			ContainerSignature: []byte("mock_container_sig"), // Required for container validation
		}

		_, err = msgServer.SubmitGradient(ctx, msg)
		require.NoError(t, err, "submission %d failed", i)
	}

	duration := time.Since(start)
	t.Logf("Submitted %d gradients in %s", submissions, duration)
}


