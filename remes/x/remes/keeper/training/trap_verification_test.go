package training_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	keepertest "remes/testutil/keeper"
	"remes/x/remes/keeper/training"
	"remes/x/remes/types"
)

type TrapVerificationTestSuite struct {
	suite.Suite
	ctx    context.Context
	keeper *training.TrainingKeeper
}

func (suite *TrapVerificationTestSuite) SetupTest() {
	suite.ctx, suite.keeper = keepertest.TrainingKeeper(suite.T())
}

func TestTrapVerificationTestSuite(t *testing.T) {
	suite.Run(t, new(TrapVerificationTestSuite))
}

// TestVerifyTrapJobResult_NormalJob tests verification of a normal (non-trap) job
func (suite *TrapVerificationTestSuite) TestVerifyTrapJobResult_NormalJob() {
	ctx := suite.ctx
	keeper := suite.keeper

	gradient := types.StoredGradient{
		GradientId:      "gradient-1",
		Miner:           "remes1test",
		TrainingRoundId: "round-1",
		GradientHash:    "hash123",
		IpfsHash:        "QmTest",
	}

	result, err := keeper.VerifyTrapJobResult(ctx, gradient)

	require.NoError(suite.T(), err)
	require.NotNil(suite.T(), result)
	require.Equal(suite.T(), training.VerdictNormalJob, result.Verdict)
}

// TestVerifyTrapJobResult_TrapPassed tests successful trap job verification
func (suite *TrapVerificationTestSuite) TestVerifyTrapJobResult_TrapPassed() {
	ctx := suite.ctx
	keeper := suite.keeper

	expectedHash := "expected_hash_123"

	// Create trap job
	trapJob := types.TrapJob{
		TrapJobId:            "trap-1",
		TargetMiner:          "remes1test",
		DatasetIpfsHash:      "QmDataset",
		ExpectedGradientHash: expectedHash,
		CreatedAtHeight:      100,
		DeadlineHeight:       200,
	}

	err := keeper.TrapJobs.Set(ctx, trapJob.TrapJobId, trapJob)
	require.NoError(suite.T(), err)

	// Submit gradient with matching hash
	gradient := types.StoredGradient{
		GradientId:      "gradient-1",
		Miner:           "remes1test",
		TrainingRoundId: "trap-1",
		GradientHash:    expectedHash,
		IpfsHash:        "QmTest",
	}

	result, err := keeper.VerifyTrapJobResult(ctx, gradient)

	require.NoError(suite.T(), err)
	require.NotNil(suite.T(), result)
	require.Equal(suite.T(), training.VerdictTrapPassed, result.Verdict)
	require.Equal(suite.T(), "10", result.RewardAmount)
}

// TestVerifyTrapJobResult_TrapFailed tests failed trap job verification
func (suite *TrapVerificationTestSuite) TestVerifyTrapJobResult_TrapFailed() {
	ctx := suite.ctx
	keeper := suite.keeper

	expectedHash := "expected_hash_123"
	wrongHash := "wrong_hash_456"

	// Create trap job
	trapJob := types.TrapJob{
		TrapJobId:            "trap-2",
		TargetMiner:          "remes1test",
		DatasetIpfsHash:      "QmDataset",
		ExpectedGradientHash: expectedHash,
		CreatedAtHeight:      100,
		DeadlineHeight:       200,
	}

	err := keeper.TrapJobs.Set(ctx, trapJob.TrapJobId, trapJob)
	require.NoError(suite.T(), err)

	// Submit gradient with wrong hash
	gradient := types.StoredGradient{
		GradientId:      "gradient-2",
		Miner:           "remes1test",
		TrainingRoundId: "trap-2",
		GradientHash:    wrongHash,
		IpfsHash:        "QmTest",
	}

	result, err := keeper.VerifyTrapJobResult(ctx, gradient)

	require.NoError(suite.T(), err)
	require.NotNil(suite.T(), result)
	require.Equal(suite.T(), training.VerdictTrapFailed, result.Verdict)
	require.Equal(suite.T(), "50", result.SlashAmount)
}

// TestVerifyTrapJobResult_MultipleTraps tests multiple trap jobs
func (suite *TrapVerificationTestSuite) TestVerifyTrapJobResult_MultipleTraps() {
	ctx := suite.ctx
	keeper := suite.keeper

	// Create multiple trap jobs
	trapJobs := []types.TrapJob{
		{
			TrapJobId:            "trap-3",
			TargetMiner:          "remes1test",
			DatasetIpfsHash:      "QmDataset1",
			ExpectedGradientHash: "hash1",
			CreatedAtHeight:      100,
			DeadlineHeight:       200,
		},
		{
			TrapJobId:            "trap-4",
			TargetMiner:          "remes1test",
			DatasetIpfsHash:      "QmDataset2",
			ExpectedGradientHash: "hash2",
			CreatedAtHeight:      100,
			DeadlineHeight:       200,
		},
	}

	for _, trapJob := range trapJobs {
		err := keeper.TrapJobs.Set(ctx, trapJob.TrapJobId, trapJob)
		require.NoError(suite.T(), err)
	}

	// Test first trap - pass
	gradient1 := types.StoredGradient{
		GradientId:      "gradient-3",
		Miner:           "remes1test",
		TrainingRoundId: "trap-3",
		GradientHash:    "hash1",
		IpfsHash:        "QmTest1",
	}

	result1, err := keeper.VerifyTrapJobResult(ctx, gradient1)
	require.NoError(suite.T(), err)
	require.Equal(suite.T(), training.VerdictTrapPassed, result1.Verdict)

	// Test second trap - fail
	gradient2 := types.StoredGradient{
		GradientId:      "gradient-4",
		Miner:           "remes1test",
		TrainingRoundId: "trap-4",
		GradientHash:    "wrong_hash",
		IpfsHash:        "QmTest2",
	}

	result2, err := keeper.VerifyTrapJobResult(ctx, gradient2)
	require.NoError(suite.T(), err)
	require.Equal(suite.T(), training.VerdictTrapFailed, result2.Verdict)
}

// TestGetTrapJobForGradient tests trap job lookup
func (suite *TrapVerificationTestSuite) TestGetTrapJobForGradient() {
	ctx := suite.ctx
	keeper := suite.keeper

	// Create trap job
	trapJob := types.TrapJob{
		TrapJobId:            "trap-5",
		TargetMiner:          "remes1test",
		DatasetIpfsHash:      "QmDataset",
		ExpectedGradientHash: "hash123",
		CreatedAtHeight:      100,
		DeadlineHeight:       200,
	}

	err := keeper.TrapJobs.Set(ctx, trapJob.TrapJobId, trapJob)
	require.NoError(suite.T(), err)

	// Test with trap job gradient
	gradient := types.StoredGradient{
		GradientId:      "gradient-5",
		Miner:           "remes1test",
		TrainingRoundId: "trap-5",
		GradientHash:    "hash123",
		IpfsHash:        "QmTest",
	}

	foundTrapJob, isTrap, err := keeper.GetTrapJobForGradient(ctx, gradient)
	require.NoError(suite.T(), err)
	require.True(suite.T(), isTrap)
	require.NotNil(suite.T(), foundTrapJob)
	require.Equal(suite.T(), trapJob.TrapJobId, foundTrapJob.TrapJobId)

	// Test with normal gradient
	normalGradient := types.StoredGradient{
		GradientId:      "gradient-6",
		Miner:           "remes1test",
		TrainingRoundId: "normal-round",
		GradientHash:    "hash456",
		IpfsHash:        "QmTest2",
	}

	_, isTrap, err = keeper.GetTrapJobForGradient(ctx, normalGradient)
	require.NoError(suite.T(), err)
	require.False(suite.T(), isTrap)
}

// TestTrapJobIntegration tests full trap job workflow
func (suite *TrapVerificationTestSuite) TestTrapJobIntegration() {
	ctx := suite.ctx
	keeper := suite.keeper

	// 1. Create trap job
	trapJob := types.TrapJob{
		TrapJobId:            "trap-integration",
		TargetMiner:          "remes1miner",
		DatasetIpfsHash:      "QmDataset",
		ExpectedGradientHash: "correct_hash",
		CreatedAtHeight:      100,
		DeadlineHeight:       200,
	}

	err := keeper.TrapJobs.Set(ctx, trapJob.TrapJobId, trapJob)
	require.NoError(suite.T(), err)

	// 2. Miner submits correct gradient
	correctGradient := types.StoredGradient{
		GradientId:      "gradient-correct",
		Miner:           "remes1miner",
		TrainingRoundId: "trap-integration",
		GradientHash:    "correct_hash",
		IpfsHash:        "QmCorrect",
	}

	result, err := keeper.VerifyTrapJobResult(ctx, correctGradient)
	require.NoError(suite.T(), err)
	require.Equal(suite.T(), training.VerdictTrapPassed, result.Verdict)
	require.Equal(suite.T(), "10", result.RewardAmount)

	// 3. Another miner submits wrong gradient
	wrongGradient := types.StoredGradient{
		GradientId:      "gradient-wrong",
		Miner:           "remes1badminer",
		TrainingRoundId: "trap-integration",
		GradientHash:    "wrong_hash",
		IpfsHash:        "QmWrong",
	}

	result, err = keeper.VerifyTrapJobResult(ctx, wrongGradient)
	require.NoError(suite.T(), err)
	require.Equal(suite.T(), training.VerdictTrapFailed, result.Verdict)
	require.Equal(suite.T(), "50", result.SlashAmount)
}

// BenchmarkVerifyTrapJobResult benchmarks trap job verification
func BenchmarkVerifyTrapJobResult(b *testing.B) {
	ctx, keeper := keepertest.TrainingKeeper(b)

	// Setup trap job
	trapJob := types.TrapJob{
		TrapJobId:            "trap-bench",
		TargetMiner:          "remes1test",
		DatasetIpfsHash:      "QmDataset",
		ExpectedGradientHash: "hash123",
		CreatedAtHeight:      100,
		DeadlineHeight:       200,
	}

	_ = keeper.TrapJobs.Set(ctx, trapJob.TrapJobId, trapJob)

	gradient := types.StoredGradient{
		GradientId:      "gradient-bench",
		Miner:           "remes1test",
		TrainingRoundId: "trap-bench",
		GradientHash:    "hash123",
		IpfsHash:        "QmTest",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = keeper.VerifyTrapJobResult(ctx, gradient)
	}
}
