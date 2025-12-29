package keeper_test

import (
	"testing"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

func TestSubmitRandomVerifierResult(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)
	msgServer := keeper.NewMsgServerImpl(f.keeper)

	// Create valid test addresses
	challengerAddr, err := f.addressCodec.BytesToString([]byte("challenger12345678901234567890"))
	require.NoError(t, err)
	
	verifierAddr, err := f.addressCodec.BytesToString([]byte("verifier123456789012345678901"))
	require.NoError(t, err)
	
	wrongVerifierAddr, err := f.addressCodec.BytesToString([]byte("wrongverifier1234567890123"))
	require.NoError(t, err)

	// Create a test challenge first
	challengeID, err := f.keeper.ChallengeID.Next(ctx)
	require.NoError(t, err)
	
	challenge := types.ChallengeRecord{
		ChallengeId:         challengeID,
		AggregationId:       1,
		Challenger:          challengerAddr,
		Status:              "pending",
		Layer:               2,
		RandomVerifier:      verifierAddr,
		RandomVerifierResult: "pending",
	}
	
	// Store challenge
	err = f.keeper.ChallengeRecords.Set(ctx, challengeID, challenge)
	require.NoError(t, err)

	// Test: Submit valid result
	msg := &types.MsgSubmitRandomVerifierResult{
		Verifier:     verifierAddr,
		ChallengeId:  challengeID,
		Result:       "invalid",
		GradientHash: "test_hash",
	}

	resp, err := msgServer.SubmitRandomVerifierResult(ctx, msg)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.True(t, resp.Accepted)

	// Verify challenge was updated
	updatedChallenge, err := f.keeper.ChallengeRecords.Get(ctx, challengeID)
	require.NoError(t, err)
	require.Equal(t, "invalid", updatedChallenge.RandomVerifierResult)

	// Test: Invalid verifier address (create new challenge for this test)
	challengeID2, err := f.keeper.ChallengeID.Next(ctx)
	require.NoError(t, err)
	
	challenge2 := types.ChallengeRecord{
		ChallengeId:         challengeID2,
		AggregationId:       1,
		Challenger:          challengerAddr,
		Status:              "pending",
		Layer:               2,
		RandomVerifier:      verifierAddr,
		RandomVerifierResult: "pending",
	}
	err = f.keeper.ChallengeRecords.Set(ctx, challengeID2, challenge2)
	require.NoError(t, err)
	
	msg2 := &types.MsgSubmitRandomVerifierResult{
		Verifier:    wrongVerifierAddr,
		ChallengeId: challengeID2,
		Result:      "valid",
	}
	
	_, err = msgServer.SubmitRandomVerifierResult(ctx, msg2)
	require.Error(t, err)
	require.Contains(t, err.Error(), "does not match selected random verifier")

	// Test: Invalid challenge ID (create new challenge for this test)
	challengeID3, err := f.keeper.ChallengeID.Next(ctx)
	require.NoError(t, err)
	// Don't store this challenge, so it doesn't exist
	
	msg3 := &types.MsgSubmitRandomVerifierResult{
		Verifier:    verifierAddr,
		ChallengeId: challengeID3 + 1000, // Non-existent ID
		Result:      "valid",
	}
	
	_, err = msgServer.SubmitRandomVerifierResult(ctx, msg3)
	require.Error(t, err)
	require.Contains(t, err.Error(), "challenge not found")
}

func TestSubmitRandomVerifierResultLayer3Trigger(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)
	msgServer := keeper.NewMsgServerImpl(f.keeper)

	// Create valid test addresses
	challengerAddr, err := f.addressCodec.BytesToString([]byte("challenger12345678901234567890"))
	require.NoError(t, err)
	
	verifierAddr, err := f.addressCodec.BytesToString([]byte("verifier123456789012345678901"))
	require.NoError(t, err)

	// Create Layer 2 challenge
	challengeID, err := f.keeper.ChallengeID.Next(ctx)
	require.NoError(t, err)
	
	challenge := types.ChallengeRecord{
		ChallengeId:         challengeID,
		AggregationId:       1,
		Challenger:          challengerAddr,
		Status:              "pending",
		Layer:               2,
		RandomVerifier:      verifierAddr,
		RandomVerifierResult: "pending",
	}
	
	err = f.keeper.ChallengeRecords.Set(ctx, challengeID, challenge)
	require.NoError(t, err)

	// Submit "invalid" result - should trigger Layer 3
	msg := &types.MsgSubmitRandomVerifierResult{
		Verifier:     verifierAddr,
		ChallengeId:  challengeID,
		Result:       "invalid",
		GradientHash: "test_hash",
	}

	resp, err := msgServer.SubmitRandomVerifierResult(ctx, msg)
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.True(t, resp.Accepted)
	// Layer 3 trigger logic is tested in optimistic_verification_test.go
}
