package keeper

import (
	"context"
	"fmt"
	"strings"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// SubmitCPUVerification handles MsgSubmitCPUVerification
// Validators submit CPU verification results for challenges
func (k msgServer) SubmitCPUVerification(ctx context.Context, msg *types.MsgSubmitCPUVerification) (*types.MsgSubmitCPUVerificationResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate validator address
	validatorAddr, err := k.addressCodec.StringToBytes(msg.Validator)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid validator address: %s", msg.Validator)
	}
	_ = validatorAddr

	// 2. Get challenge
	challenge, err := k.ChallengeRecords.Get(ctx, msg.ChallengeId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "challenge not found")
	}

	// 3. Check if challenge is in valid status for CPU verification
	// Layer 2 challenges: status should be "pending" or "layer2_pending"
	// Layer 3 challenges: status should be "cpu_verification_pending"
	validStatuses := map[string]bool{
		"pending":                true,
		"layer2_pending":          true,
		"cpu_verification_pending": true,
	}
	if !validStatuses[challenge.Status] {
		return nil, errorsmod.Wrapf(err, "challenge is not in valid status for CPU verification: %s", challenge.Status)
	}

	// 4. Verify validator is in CPU verification panel
	isInPanel := false
	for _, panelValidator := range challenge.CpuVerificationPanel {
		if panelValidator == msg.Validator {
			isInPanel = true
			break
		}
	}
	if !isInPanel {
		return nil, errorsmod.Wrap(err, "validator is not in CPU verification panel")
	}

	// 5. Check if validator already submitted verification
	for _, result := range challenge.CpuVerificationResults {
		if result.ValidatorAddress == msg.Validator {
			return nil, errorsmod.Wrap(err, "validator already submitted verification")
		}
	}

	// 6. Create CPU verification result
	verificationTime := sdkCtx.BlockTime()
	verificationResult := types.CPUVerificationResult{
		ValidatorAddress: msg.Validator,
		IsValid:          msg.IsValid,
		ComputedHash:     strings.TrimSpace(msg.ComputedHash),
		ExpectedHash:     strings.TrimSpace(msg.ExpectedHash),
		VerificationTime: verificationTime,
	}

	// 7. Add result to challenge
	challenge.CpuVerificationResults = append(challenge.CpuVerificationResults, &verificationResult)

	// 7.5. Check if Layer 2 random verifier result is available
	// If Layer 2 and random verifier agrees with challenger, trigger Layer 3
	if challenge.Layer == 2 && challenge.RandomVerifierResult == "invalid" {
		// Random verifier agrees - trigger Layer 3
		if err := k.VerifyGradientLayer3(sdkCtx, msg.ChallengeId); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to trigger Layer 3: %v", err))
			// Continue - Layer 3 trigger failure shouldn't block CPU verification submission
		}
		// Re-fetch challenge after Layer 3 update
		challenge, err = k.ChallengeRecords.Get(ctx, msg.ChallengeId)
		if err != nil {
			return nil, errorsmod.Wrap(err, "failed to re-fetch challenge after Layer 3 trigger")
		}
	}

	// 8. Check if we have 2/3 consensus for Layer 3 (auto-resolve)
	if challenge.Layer == 3 {
		panelSize := len(challenge.CpuVerificationPanel)
		if panelSize > 0 {
			validCount := 0
			invalidCount := 0
			for _, result := range challenge.CpuVerificationResults {
				if result.IsValid {
					validCount++
				} else {
					invalidCount++
				}
			}
			consensusThreshold := (panelSize * 2) / 3
			if invalidCount >= consensusThreshold || validCount >= consensusThreshold {
				// Auto-resolve: Update challenge status
				if invalidCount >= consensusThreshold {
					challenge.Status = "cpu_verification_complete_fraud"
					challenge.FraudDetected = true
				} else {
					challenge.Status = "cpu_verification_complete_valid"
					challenge.FraudDetected = false
				}
			}
		}
	}

	// 9. Update challenge
	if err := k.ChallengeRecords.Set(ctx, msg.ChallengeId, challenge); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update challenge")
	}

	// 10. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeSubmitCPUVerification,
			sdk.NewAttribute(types.AttributeKeyChallengeID, fmt.Sprintf("%d", msg.ChallengeId)),
			sdk.NewAttribute(types.AttributeKeyValidator, msg.Validator),
			sdk.NewAttribute(types.AttributeKeyIsValid, fmt.Sprintf("%t", msg.IsValid)),
			sdk.NewAttribute("layer", fmt.Sprintf("%d", challenge.Layer)),
		),
	)

	return &types.MsgSubmitCPUVerificationResponse{
		Accepted: true,
	}, nil
}

// ResolveChallenge handles MsgResolveChallenge
// Resolves a challenge based on CPU verification panel consensus
func (k msgServer) ResolveChallenge(ctx context.Context, msg *types.MsgResolveChallenge) (*types.MsgResolveChallengeResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Get challenge first
	challenge, err := k.ChallengeRecords.Get(ctx, msg.ChallengeId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "challenge not found")
	}

	// 2. Verify resolver is module authority or a validator in the CPU verification panel
	resolverAddr, err := k.addressCodec.StringToBytes(msg.Resolver)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid resolver address: %s", msg.Resolver)
	}

	// Check if resolver is module authority or a validator in the CPU verification panel
	authority := sdk.AccAddress(k.GetAuthority()).String()
	isAuthority := msg.Resolver == authority
	
	// Check if resolver is in CPU verification panel
	isInPanel := false
	if !isAuthority {
		for _, panelValidator := range challenge.CpuVerificationPanel {
			if panelValidator == msg.Resolver {
				isInPanel = true
				break
			}
		}
	}
	
	if !isAuthority && !isInPanel {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "resolver %s is not authorized to resolve challenges (must be authority or in CPU verification panel)", msg.Resolver)
	}
	_ = resolverAddr
	if err != nil {
		return nil, errorsmod.Wrap(err, "challenge not found")
	}

	// 3. Check if challenge is ready for resolution
	// Layer 3 challenges should be in "cpu_verification_complete_*" status
	// Layer 2 challenges can be resolved directly
	readyForResolution := challenge.Status == "cpu_verification_complete_fraud" ||
		challenge.Status == "cpu_verification_complete_valid" ||
		challenge.Status == "layer2_pending" ||
		(challenge.Layer == 2 && challenge.RandomVerifierResult != "pending")

	if !readyForResolution && challenge.Layer != 3 {
		return nil, errorsmod.Wrapf(err, "challenge is not ready for resolution (status: %s, layer: %d)", challenge.Status, challenge.Layer)
	}

	// 4. Get aggregation
	aggregation, err := k.AggregationRecords.Get(ctx, challenge.AggregationId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "aggregation not found")
	}

	// 5. Check CPU verification results consensus
	// Require at least 2/3 of panel to agree
	panelSize := len(challenge.CpuVerificationPanel)
	if panelSize == 0 {
		return nil, errorsmod.Wrap(err, "CPU verification panel is empty")
	}

	validCount := 0
	invalidCount := 0
	for _, result := range challenge.CpuVerificationResults {
		if result.IsValid {
			validCount++
		} else {
			invalidCount++
		}
	}

	// 6. Determine resolution based on consensus or Layer 3 result
	// Layer 3: Use CPU verification panel consensus
	// Layer 2: Use random verifier result
	fraudDetected := false

	if challenge.Layer == 3 {
		// Layer 3: Use CPU verification panel consensus
		consensusThreshold := (panelSize * 2) / 3
		if invalidCount >= consensusThreshold {
			fraudDetected = true
		} else if validCount >= consensusThreshold {
			fraudDetected = false
		} else {
			// No consensus yet - wait for more verifications
			return nil, errorsmod.Wrap(err, "insufficient consensus for Layer 3 resolution")
		}
	} else if challenge.Layer == 2 {
		// Layer 2: Use random verifier result
		if challenge.RandomVerifierResult == "invalid" {
			// Random verifier agrees with challenger - trigger Layer 3
			if err := k.VerifyGradientLayer3(sdkCtx, msg.ChallengeId); err != nil {
				return nil, errorsmod.Wrap(err, "failed to trigger Layer 3")
			}
			// Re-fetch challenge
			challenge, err = k.ChallengeRecords.Get(ctx, msg.ChallengeId)
			if err != nil {
				return nil, errorsmod.Wrap(err, "failed to re-fetch challenge after Layer 3 trigger")
			}
			// Layer 3 will resolve later
			return &types.MsgResolveChallengeResponse{
				Resolved: false, // Layer 3 pending
			}, nil
		} else {
			// Random verifier disagrees - challenge rejected
			fraudDetected = false
		}
	}

	// 7. Update challenge status
	challenge.Status = "resolved"
	challenge.FraudDetected = fraudDetected
	challenge.ResolutionHeight = sdkCtx.BlockHeight()
	if err := k.ChallengeRecords.Set(ctx, msg.ChallengeId, challenge); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update challenge")
	}

	// 7.5. Handle bond distribution (Layer 2 bond return)
	if challenge.Layer >= 2 && challenge.BondAmount != "" {
		if err := k.returnChallengeBond(sdkCtx, msg.ChallengeId, fraudDetected); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to return challenge bond: %v", err))
			// Don't fail - bond distribution failure shouldn't block resolution
		}
	}

	// 8. If fraud detected, slash proposer and revert aggregation
	if fraudDetected {
		// Slash proposer (50% of stake)
		slashFraction := sdkmath.LegacyMustNewDecFromStr("0.50") // 50%
		if err := k.SlashNode(ctx, aggregation.Proposer, slashFraction, "fraud_detected_aggregation"); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to slash proposer: %v", err))
		}
		
		// Update trust score for fraud detection
		if err := k.UpdateTrustScoreOnChallengeResolution(sdkCtx, challenge.ChallengeId, true); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to update trust score on fraud detection: %v", err))
			// Don't fail - trust score update is not critical
		}
		
		// Distribute fraud detection bounty to challenger (10-20x base reward)
		baseReward := sdk.NewCoins(sdk.NewCoin("remes", sdkmath.NewInt(500))) // BaseProposerReward
		severity := 1.0 // Critical fraud (aggregation fraud)
		if err := k.DistributeFraudDetectionBounty(ctx, challenge.Challenger, baseReward, severity, "aggregation_fraud"); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to distribute fraud detection bounty: %v", err))
		}

		// Revert aggregation status
		aggregation.Status = "reverted"
		if err := k.AggregationRecords.Set(ctx, challenge.AggregationId, aggregation); err != nil {
			return nil, errorsmod.Wrap(err, "failed to revert aggregation")
		}

		// Revert participant gradients to pending
		for _, gradientID := range aggregation.ParticipantGradientIds {
			gradient, err := k.StoredGradients.Get(ctx, gradientID)
			if err != nil {
				continue
			}
			gradient.Status = "pending"
			if err := k.StoredGradients.Set(ctx, gradientID, gradient); err != nil {
				sdkCtx.Logger().Error(fmt.Sprintf("Failed to revert gradient %d: %v", gradientID, err))
			}
		}
	} else {
		// No fraud detected, but still update trust score (small decrease for being challenged)
		if err := k.UpdateTrustScoreOnChallengeResolution(sdkCtx, challenge.ChallengeId, false); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to update trust score on challenge resolution: %v", err))
			// Don't fail - trust score update is not critical
		}
		
		// Challenge rejected - finalize aggregation
		aggregation.Status = "finalized"
		if err := k.AggregationRecords.Set(ctx, challenge.AggregationId, aggregation); err != nil {
			return nil, errorsmod.Wrap(err, "failed to finalize aggregation")
		}
	}

	// 9. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeResolveChallenge,
			sdk.NewAttribute(types.AttributeKeyChallengeID, fmt.Sprintf("%d", msg.ChallengeId)),
			sdk.NewAttribute(types.AttributeKeyFraudDetected, fmt.Sprintf("%t", fraudDetected)),
			sdk.NewAttribute(types.AttributeKeyReason, msg.ResolutionReason),
		),
	)

	return &types.MsgResolveChallengeResponse{
		Resolved: true,
	}, nil
}

