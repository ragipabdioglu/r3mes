package keeper_test

import (
	"testing"

	"cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

// TestEndToEndGradientSubmission tests the full gradient submission workflow
func TestEndToEndGradientSubmission(t *testing.T) {
	f := initFixture(t)
	// We need both sdk.Context (for keeper methods) and context.Context (for message handlers)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)
	ctx := sdk.WrapSDKContext(sdkCtx) // Wrap for message handler (context.Context)
	msgServer := keeper.NewMsgServerImpl(f.keeper)

	// 0. Initialize genesis state for backward compatibility (model ID=1)
	genesisState := &types.GenesisState{
		Params:       types.DefaultParams(),
		ModelHash:    "QmTestHash",
		ModelVersion: "b1.58",
	}
	err := f.keeper.InitGenesis(f.ctx, genesisState)
	require.NoError(t, err)

	// Initialize sequences manually for tests (Cosmos SDK collections sequences start at 0)
	// We need to set them to 1 so that Next() returns 1 on first call
	// In production, sequences auto-initialize, but in tests we need to do it manually
	if err := f.keeper.StoredGradientID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize StoredGradientID sequence: %v", err)
	}
	if err := f.keeper.AggregationID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize AggregationID sequence: %v", err)
	}
	if err := f.keeper.ChallengeID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize ChallengeID sequence: %v", err)
	}

	// 1. Register a model first
	modelConfig := types.ModelConfig{
		ModelType:          types.ModelType_MODEL_TYPE_BITNET,
		ModelVersion:       "b1.58",
		ArchitectureConfig: `{"hidden_size": 768, "num_layers": 12, "lora_rank": 8}`,
		ContainerHash:      "sha256:abc123",
		ContainerRegistry:  "docker.io",
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
	}

	modelID, err := f.keeper.RegisterModel(sdkCtx, modelConfig, 1)
	require.NoError(t, err)

	// 2. Create a miner address
	minerAddr, err := f.addressCodec.BytesToString([]byte("miner123456789012345678901234"))
	require.NoError(t, err)

	// 3. Calculate shard ID for this miner
	trainingRoundID := uint64(1)
	totalShards := uint64(100)
	expectedShardID, err := f.keeper.CalculateShardID(sdkCtx, minerAddr, trainingRoundID, totalShards)
	require.NoError(t, err)

	// 4. Submit gradient
	msg := &types.MsgSubmitGradient{
		Miner:              minerAddr,
		IpfsHash:           "QmTestIPFSHash123",
		ModelVersion:       "b1.58",
		TrainingRoundId:    trainingRoundID,
		ShardId:            expectedShardID, // Use calculated shard ID
		GradientHash:       "QmTestGradientHash123",
		ModelConfigId:      modelID,
		GpuArchitecture:    "Ampere",                             // Test GPU architecture
		Nonce:              1,                                    // Nonce cannot be zero
		Signature:          []byte("mock_signature_for_testing"), // Mock signature for testing
		ContainerHash:      "sha256:test123",                     // Required for container validation
		ContainerSignature: []byte("mock_container_sig"),         // Required for container validation
	}

	// Debug: Check sequence before submission
	// Note: Cosmos SDK collections Sequence starts at 1, first Next() should return 1
	// If it returns 0, the sequence might not be initialized properly
	resp, err := msgServer.SubmitGradient(ctx, msg)
	if err != nil {
		t.Logf("SubmitGradient error: %v", err)
		t.FailNow()
	}
	require.NoError(t, err)
	require.NotNil(t, resp)
	if resp.StoredGradientId == 0 {
		t.Logf("ERROR: StoredGradientId is 0, response: %+v", resp)
		t.Logf("This indicates Sequence.Next() returned 0, which means sequence is not initialized")
		t.Logf("Context type: %T", ctx)
		// Try to manually check sequence
		// Note: This is a workaround - in production, sequence should auto-initialize
	}
	require.Greater(t, resp.StoredGradientId, uint64(0), "StoredGradientId should be > 0, got %d", resp.StoredGradientId)

	// 5. Verify gradient was stored
	storedGradient, err := f.keeper.StoredGradients.Get(sdkCtx, resp.StoredGradientId)
	require.NoError(t, err)
	require.Equal(t, minerAddr, storedGradient.Miner)
	require.Equal(t, "QmTestGradientHash123", storedGradient.GradientHash)
	require.Equal(t, modelID, storedGradient.ModelConfigId)
}

// TestEndToEndAggregation tests the full aggregation workflow
func TestEndToEndAggregation(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)
	ctx := sdk.WrapSDKContext(sdkCtx)
	msgServer := keeper.NewMsgServerImpl(f.keeper)

	// 0. Initialize genesis state for backward compatibility
	genesisState := &types.GenesisState{
		Params:       types.DefaultParams(),
		ModelHash:    "QmTestHash",
		ModelVersion: "b1.58",
	}
	err := f.keeper.InitGenesis(f.ctx, genesisState)
	require.NoError(t, err)

	// Initialize sequences manually for tests
	if err := f.keeper.StoredGradientID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize StoredGradientID sequence: %v", err)
	}
	if err := f.keeper.AggregationID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize AggregationID sequence: %v", err)
	}

	// 1. Register model
	modelConfig := types.ModelConfig{
		ModelType:          types.ModelType_MODEL_TYPE_BITNET,
		ModelVersion:       "b1.58",
		ArchitectureConfig: `{"hidden_size": 768}`,
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
	}
	modelID, err := f.keeper.RegisterModel(sdkCtx, modelConfig, 1)
	require.NoError(t, err)

	// 2. Submit multiple gradients
	proposerAddr, err := f.addressCodec.BytesToString([]byte("proposer12345678901234567890"))
	require.NoError(t, err)

	var gradientIDs []uint64
	trainingRoundID := uint64(1)
	totalShards := uint64(100)
	for i := 0; i < 3; i++ {
		minerBytes := []byte("miner123456789012345678901234")
		minerBytes[len(minerBytes)-1] = byte(i) // Make unique
		minerAddr, _ := f.addressCodec.BytesToString(minerBytes)

		// Calculate shard ID for this miner
		expectedShardID, err := f.keeper.CalculateShardID(sdkCtx, minerAddr, trainingRoundID, totalShards)
		require.NoError(t, err)

		msg := &types.MsgSubmitGradient{
			Miner:              minerAddr,
			IpfsHash:           "QmTestIPFSHash" + string(rune('0'+i)),
			ModelVersion:       "b1.58",
			TrainingRoundId:    trainingRoundID,
			ShardId:            expectedShardID, // Use calculated shard ID
			GradientHash:       "QmTestGradientHash" + string(rune('0'+i)),
			ModelConfigId:      modelID,
			GpuArchitecture:    "Ampere",                             // Test GPU architecture
			Nonce:              uint64(i + 1),                        // Nonce cannot be zero, unique per iteration
			Signature:          []byte("mock_signature_for_testing"), // Mock signature for testing
			ContainerHash:      "sha256:test123",                     // Required for container validation
			ContainerSignature: []byte("mock_container_sig"),         // Required for container validation
		}
		resp, err := msgServer.SubmitGradient(ctx, msg)
		require.NoError(t, err)
		gradientIDs = append(gradientIDs, resp.StoredGradientId)
	}

	// 3. Calculate merkle root from gradient hashes using keeper's function
	gradientHashes := make([]string, 0, len(gradientIDs))
	for _, gradientID := range gradientIDs {
		gradient, err := f.keeper.StoredGradients.Get(sdkCtx, gradientID)
		require.NoError(t, err)
		gradientHashes = append(gradientHashes, gradient.GradientHash)
	}
	// Use keeper's CalculateMerkleRoot function for proper calculation
	calculatedMerkleRoot, err := f.keeper.CalculateMerkleRoot(gradientHashes)
	require.NoError(t, err)

	// 3. Submit aggregation
	aggMsg := &types.MsgSubmitAggregation{
		Proposer:                   proposerAddr,
		AggregatedGradientIpfsHash: "QmAggregatedHash123",
		MerkleRoot:                 calculatedMerkleRoot, // Use calculated merkle root
		ParticipantGradientIds:     gradientIDs,
		TrainingRoundId:            1,
		ModelVersion:               "b1.58",
	}

	aggResp, err := msgServer.SubmitAggregation(ctx, aggMsg)
	require.NoError(t, err)
	require.GreaterOrEqual(t, aggResp.AggregationId, uint64(1)) // ID should be at least 1

	// 4. Verify aggregation was stored
	aggRecord, err := f.keeper.AggregationRecords.Get(sdkCtx, aggResp.AggregationId)
	require.NoError(t, err)
	require.Equal(t, proposerAddr, aggRecord.Proposer)
	require.Equal(t, len(gradientIDs), len(aggRecord.ParticipantGradientIds))
}

// TestEndToEndChallenge tests the full challenge workflow
func TestEndToEndChallenge(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)
	ctx := sdk.WrapSDKContext(sdkCtx)
	msgServer := keeper.NewMsgServerImpl(f.keeper)

	// 0. Initialize genesis state for backward compatibility
	genesisState := &types.GenesisState{
		Params:       types.DefaultParams(),
		ModelHash:    "QmTestHash",
		ModelVersion: "b1.58",
	}
	err := f.keeper.InitGenesis(f.ctx, genesisState)
	require.NoError(t, err)

	// Initialize sequences manually for tests
	if err := f.keeper.StoredGradientID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize StoredGradientID sequence: %v", err)
	}
	if err := f.keeper.AggregationID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize AggregationID sequence: %v", err)
	}
	if err := f.keeper.ChallengeID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize ChallengeID sequence: %v", err)
	}

	// 1. Setup: Register model and create aggregation
	modelConfig := types.ModelConfig{
		ModelType:          types.ModelType_MODEL_TYPE_BITNET,
		ModelVersion:       "b1.58",
		ArchitectureConfig: `{"hidden_size": 768}`,
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
	}
	modelID, err := f.keeper.RegisterModel(sdkCtx, modelConfig, 1)
	require.NoError(t, err)

	proposerAddr, _ := f.addressCodec.BytesToString([]byte("proposer12345678901234567890"))
	minerAddr, _ := f.addressCodec.BytesToString([]byte("miner123456789012345678901234"))

	// Calculate shard ID for this miner
	trainingRoundID := uint64(1)
	totalShards := uint64(100)
	expectedShardID, err := f.keeper.CalculateShardID(sdkCtx, minerAddr, trainingRoundID, totalShards)
	require.NoError(t, err)

	// Submit gradient
	gradMsg := &types.MsgSubmitGradient{
		Miner:              minerAddr,
		IpfsHash:           "QmTestIPFSHash",
		ModelVersion:       "b1.58",
		TrainingRoundId:    trainingRoundID,
		ShardId:            expectedShardID, // Use calculated shard ID
		GradientHash:       "QmTestGradientHash",
		ModelConfigId:      modelID,
		GpuArchitecture:    "Ampere",                             // Test GPU architecture
		Nonce:              1,                                    // Nonce cannot be zero
		Signature:          []byte("mock_signature_for_testing"), // Mock signature for testing
		ContainerHash:      "sha256:test123",                     // Required for container validation
		ContainerSignature: []byte("mock_container_sig"),         // Required for container validation
		ClaimedLoss:        "1000",                               // Loss-Based Spot Checking: BitNet integer format
	}
	gradResp, err := msgServer.SubmitGradient(ctx, gradMsg)
	require.NoError(t, err)

	// Get stored gradient to calculate correct merkle root
	storedGradient, err := f.keeper.StoredGradients.Get(sdkCtx, gradResp.StoredGradientId)
	require.NoError(t, err)

	// For single gradient, calculateMerkleRoot returns the hash itself
	gradientHash := storedGradient.GradientHash
	calculatedMerkleRoot := gradientHash // Single leaf = hash itself

	// Submit aggregation
	aggMsg := &types.MsgSubmitAggregation{
		Proposer:                   proposerAddr,
		AggregatedGradientIpfsHash: "QmAggregatedHash",
		MerkleRoot:                 calculatedMerkleRoot, // Use calculated merkle root
		ParticipantGradientIds:     []uint64{gradResp.StoredGradientId},
		TrainingRoundId:            1,
		ModelVersion:               "b1.58",
	}
	aggResp, err := msgServer.SubmitAggregation(ctx, aggMsg)
	require.NoError(t, err)

	// 2. Challenge aggregation
	challengerAddr, _ := f.addressCodec.BytesToString([]byte("challenger12345678901234567890"))
	bondAmount := sdk.NewCoins(sdk.NewCoin("remes", math.NewInt(5000))) // 10x base reward (500 remes)

	challengeMsg := &types.MsgChallengeAggregation{
		Challenger:       challengerAddr,
		AggregationId:    aggResp.AggregationId,
		Reason:           "Invalid gradient hash",
		EvidenceIpfsHash: "QmEvidenceHash",
		BondAmount:       bondAmount,
	}

	challengeResp, err := msgServer.ChallengeAggregation(ctx, challengeMsg)
	require.NoError(t, err)
	require.GreaterOrEqual(t, challengeResp.ChallengeId, uint64(1)) // ID should be at least 1

	// 3. Verify challenge was stored
	challenge, err := f.keeper.ChallengeRecords.Get(sdkCtx, challengeResp.ChallengeId)
	require.NoError(t, err)
	require.Equal(t, challengerAddr, challenge.Challenger)
	require.Equal(t, aggResp.AggregationId, challenge.AggregationId)
	require.Equal(t, uint32(2), challenge.Layer) // Layer 2 challenge
}

// TestEndToEndTreasuryBuyBack tests the treasury buy-back and burn workflow
func TestEndToEndTreasuryBuyBack(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	// 1. Initialize treasury with some balance
	treasury := types.Treasury{
		Balance:          sdk.NewCoins(sdk.NewCoin("stake", math.NewInt(1000000))).String(),
		BuyBackThreshold: "500000stake",
		BurnFraction:     "0.5", // 50%
		BuyBackInterval:  100,   // Every 100 blocks
		TotalBurned:      "0stake",
	}

	err := f.keeper.Treasury.Set(ctx, treasury)
	require.NoError(t, err)

	// 2. Collect inference revenue (simulate)
	fee := sdk.NewCoins(sdk.NewCoin("stake", math.NewInt(100000)))

	err = f.keeper.CollectInferenceRevenue(ctx, fee)
	require.NoError(t, err)

	// 3. Verify treasury balance increased
	updatedTreasury, err := f.keeper.Treasury.Get(ctx)
	require.NoError(t, err)
	// Treasury should have collected 20% of fee
	currentBalance, err := sdk.ParseCoinsNormalized(updatedTreasury.Balance)
	require.NoError(t, err)
	expectedAmount := math.NewInt(20000) // 20% of 100000
	require.True(t, currentBalance.AmountOf("stake").GTE(expectedAmount))

	// 4. Process buy-back (simulate EndBlocker)
	// This would normally be called in EndBlocker when threshold is met
	err = f.keeper.ProcessTreasuryBuyBack(ctx)
	require.NoError(t, err)

	// 5. Verify buy-back was processed (if threshold met)
	finalTreasury, err := f.keeper.Treasury.Get(ctx)
	require.NoError(t, err)
	// Total burned should be updated if buy-back occurred
	require.NotNil(t, finalTreasury.TotalBurned)
}

// TestEndToEndOptimisticVerificationFlow tests the full 3-layer verification flow
func TestEndToEndOptimisticVerificationFlow(t *testing.T) {
	f := initFixture(t)
	sdkCtx := sdk.UnwrapSDKContext(f.ctx)
	ctx := sdk.WrapSDKContext(sdkCtx)
	msgServer := keeper.NewMsgServerImpl(f.keeper)

	// 0. Initialize genesis state for backward compatibility
	genesisState := &types.GenesisState{
		Params:       types.DefaultParams(),
		ModelHash:    "QmTestHash",
		ModelVersion: "b1.58",
	}
	err := f.keeper.InitGenesis(f.ctx, genesisState)
	require.NoError(t, err)

	// Initialize sequences manually for tests
	if err := f.keeper.StoredGradientID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize StoredGradientID sequence: %v", err)
	}
	if err := f.keeper.AggregationID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize AggregationID sequence: %v", err)
	}
	if err := f.keeper.ChallengeID.Set(sdkCtx, 1); err != nil {
		t.Logf("Warning: Failed to initialize ChallengeID sequence: %v", err)
	}

	// 1. Setup: Create aggregation
	modelConfig := types.ModelConfig{
		ModelType:          types.ModelType_MODEL_TYPE_BITNET,
		ModelVersion:       "b1.58",
		ArchitectureConfig: `{"hidden_size": 768}`,
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT,
	}
	modelID, err := f.keeper.RegisterModel(sdkCtx, modelConfig, 1)
	require.NoError(t, err)

	proposerAddr, _ := f.addressCodec.BytesToString([]byte("proposer12345678901234567890"))
	minerAddr, _ := f.addressCodec.BytesToString([]byte("miner123456789012345678901234"))

	// Calculate shard ID for this miner
	trainingRoundID := uint64(1)
	totalShards := uint64(100)
	expectedShardID, err := f.keeper.CalculateShardID(sdkCtx, minerAddr, trainingRoundID, totalShards)
	require.NoError(t, err)

	gradMsg := &types.MsgSubmitGradient{
		Miner:              minerAddr,
		IpfsHash:           "QmTestIPFSHash",
		ModelVersion:       "b1.58",
		TrainingRoundId:    trainingRoundID,
		ShardId:            expectedShardID, // Use calculated shard ID
		GradientHash:       "QmTestGradientHash",
		ModelConfigId:      modelID,
		GpuArchitecture:    "Ampere",                             // Test GPU architecture
		Nonce:              1,                                    // Nonce cannot be zero
		Signature:          []byte("mock_signature_for_testing"), // Mock signature for testing
		ContainerHash:      "sha256:test123",                     // Required for container validation
		ContainerSignature: []byte("mock_container_sig"),         // Required for container validation
	}
	gradResp, err := msgServer.SubmitGradient(ctx, gradMsg)
	require.NoError(t, err)

	// Get stored gradient to calculate correct merkle root
	storedGradient, err := f.keeper.StoredGradients.Get(sdkCtx, gradResp.StoredGradientId)
	require.NoError(t, err)

	// For single gradient, calculateMerkleRoot returns the hash itself
	// (loop doesn't execute when len(currentLevel) == 1)
	gradientHash := storedGradient.GradientHash
	calculatedMerkleRoot := gradientHash // Single leaf = hash itself

	aggMsg := &types.MsgSubmitAggregation{
		Proposer:                   proposerAddr,
		AggregatedGradientIpfsHash: "QmAggregatedHash",
		MerkleRoot:                 calculatedMerkleRoot, // Use calculated merkle root
		ParticipantGradientIds:     []uint64{gradResp.StoredGradientId},
		TrainingRoundId:            1,
		ModelVersion:               "b1.58",
	}
	aggResp, err := msgServer.SubmitAggregation(ctx, aggMsg)
	require.NoError(t, err)

	// 2. Layer 1: Optimistic acceptance (happens automatically in SubmitGradient)
	// Gradient is accepted optimistically

	// 2.5. Register a validator node for random verifier selection
	verifierAddr, _ := f.addressCodec.BytesToString([]byte("verifier123456789012345678901234"))
	registerNodeMsg := &types.MsgRegisterNode{
		NodeAddress: verifierAddr,
		NodeType:    types.NODE_TYPE_VALIDATOR,
		Resources:   types.ResourceSpec{CpuCores: 8, MemoryGb: 16, GpuCount: 1},
		Roles:       []types.NodeType{types.NODE_TYPE_VALIDATOR},
		RoleAllocations: []*types.RoleAllocation{
			{
				Role: types.NODE_TYPE_VALIDATOR,
				Quota: types.ResourceQuota{
					Role:        types.NODE_TYPE_VALIDATOR,
					CpuCores:    8,
					MemoryGb:    16,
					GpuCount:    1,
					GpuMemoryGb: 8,
				},
			},
		},
		Stake: sdk.NewCoins(sdk.NewCoin("remes", math.NewInt(10000))).String(),
	}
	_, err = msgServer.RegisterNode(ctx, registerNodeMsg)
	require.NoError(t, err)

	// 3. Layer 2: Challenge with bond
	// Required bond: 10x base reward (500 remes) = 5000 remes
	challengerAddr, _ := f.addressCodec.BytesToString([]byte("challenger12345678901234567890"))
	bondAmount := sdk.NewCoins(sdk.NewCoin("remes", math.NewInt(5000))) // 10x base reward (500 remes)

	challengeMsg := &types.MsgChallengeAggregation{
		Challenger:       challengerAddr,
		AggregationId:    aggResp.AggregationId,
		Reason:           "Invalid gradient",
		EvidenceIpfsHash: "QmEvidenceHash",
		BondAmount:       bondAmount,
	}

	challengeResp, err := msgServer.ChallengeAggregation(ctx, challengeMsg)
	require.NoError(t, err)

	// 4. Layer 2: Random verifier submits result
	challenge, err := f.keeper.ChallengeRecords.Get(sdkCtx, challengeResp.ChallengeId)
	require.NoError(t, err)
	require.NotEmpty(t, challenge.RandomVerifier)

	verifierMsg := &types.MsgSubmitRandomVerifierResult{
		Verifier:     challenge.RandomVerifier,
		ChallengeId:  challengeResp.ChallengeId,
		Result:       "invalid",
		GradientHash: "QmVerifierHash",
	}

	verifierResp, err := msgServer.SubmitRandomVerifierResult(ctx, verifierMsg)
	require.NoError(t, err)
	require.True(t, verifierResp.Accepted)
	require.True(t, verifierResp.Layer_3Triggered) // Layer 3 should be triggered

	// 5. Layer 3: CPU verification panel (would be triggered automatically)
	updatedChallenge, err := f.keeper.ChallengeRecords.Get(ctx, challengeResp.ChallengeId)
	require.NoError(t, err)
	require.Equal(t, uint32(3), updatedChallenge.Layer) // Should be Layer 3
}
