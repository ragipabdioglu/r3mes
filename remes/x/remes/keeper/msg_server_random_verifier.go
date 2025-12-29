package keeper

import (
	"context"
	"fmt"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// SubmitRandomVerifierResult handles MsgSubmitRandomVerifierResult
// Random GPU verifier submits their verification result for Layer 2 challenge
func (k msgServer) SubmitRandomVerifierResult(ctx context.Context, msg *types.MsgSubmitRandomVerifierResult) (*types.MsgSubmitRandomVerifierResultResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate verifier address
	verifierAddr, err := k.addressCodec.StringToBytes(msg.Verifier)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid verifier address: %s", msg.Verifier)
	}
	_ = verifierAddr

	// 2. Get challenge record
	challenge, err := k.ChallengeRecords.Get(ctx, msg.ChallengeId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "challenge not found")
	}

	// 3. Verify this is a Layer 2 challenge
	if challenge.Layer != 2 {
		return nil, errorsmod.Wrapf(types.ErrInvalidChallenge, "random verifier result only valid for Layer 2 challenges")
	}

	// 4. Verify the verifier matches the selected random verifier
	if challenge.RandomVerifier != msg.Verifier {
		return nil, errorsmod.Wrapf(types.ErrInvalidChallenge, "verifier address does not match selected random verifier")
	}

	// 5. Validate result value
	if msg.Result != "valid" && msg.Result != "invalid" && msg.Result != "pending" {
		return nil, errorsmod.Wrapf(types.ErrInvalidChallenge, "invalid result value, must be 'valid', 'invalid', or 'pending'")
	}

	// 6. Update challenge record with verifier result
	challenge.RandomVerifierResult = msg.Result
	if msg.GradientHash != "" {
		// Store verifier's computed gradient hash for comparison
		// This can be used for further verification if needed
	}

	// 6.5. LOSS-BASED SPOT CHECKING: Store loss verification results if provided
	if msg.VerifierLoss != "" {
		challenge.VerifierLoss = msg.VerifierLoss
		challenge.LossMatch = msg.LossMatch
		challenge.DataBatchSeed = msg.DataBatchSeed
		challenge.LossTolerance = "1" // Default BitNet tolerance: ±1 integer unit
	}

	// 7. Store updated challenge
	if err := k.ChallengeRecords.Set(ctx, msg.ChallengeId, challenge); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update challenge record")
	}

	// 8. Distribute validator reward for verification work
	// Validator gets reward for performing loss-based spot checking
	verificationCount := uint64(1) // One verification performed
	if err := k.DistributeValidatorReward(ctx, msg.Verifier, verificationCount); err != nil {
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to distribute validator reward: %v", err))
		// Don't fail the transaction, just log the error
	}

	// 9. Check if Layer 3 should be triggered
	layer3Triggered := false
	if msg.Result == "invalid" {
		// Verifier agrees with challenger - trigger Layer 3 CPU verification
		if err := k.VerifyGradientLayer3(sdkCtx, msg.ChallengeId); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to trigger Layer 3 verification: %v", err))
			// Don't fail the transaction, just log the error
		} else {
			layer3Triggered = true
		}
	} else if msg.Result == "valid" {
		// Verifier disagrees with challenger - challenge rejected, return bond
		if err := k.returnChallengeBond(sdkCtx, msg.ChallengeId, false); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to return challenge bond: %v", err))
		}
	}

	// 10. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeSubmitRandomVerifierResult,
			sdk.NewAttribute(types.AttributeKeyChallengeID, fmt.Sprintf("%d", msg.ChallengeId)),
			sdk.NewAttribute(types.AttributeKeyValidator, msg.Verifier),
			sdk.NewAttribute("result", msg.Result),
			sdk.NewAttribute("layer_3_triggered", fmt.Sprintf("%t", layer3Triggered)),
		),
	)

	return &types.MsgSubmitRandomVerifierResultResponse{
		Accepted:        true,
		Layer_3Triggered: layer3Triggered,
	}, nil
}

