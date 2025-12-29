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

// ReportLazyValidation handles MsgReportLazyValidation
// Reports lazy validation by a validator (20% slash)
func (k msgServer) ReportLazyValidation(ctx context.Context, msg *types.MsgReportLazyValidation) (*types.MsgReportLazyValidationResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate reporter address
	reporterAddr, err := k.addressCodec.StringToBytes(msg.Reporter)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid reporter address: %s", msg.Reporter)
	}
	_ = reporterAddr

	// 2. Validate validator address
	_, err = k.addressCodec.StringToBytes(msg.ValidatorAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid validator address: %s", msg.ValidatorAddress)
	}

	// 3. Verify validator is registered and has validator role
	registration, err := k.NodeRegistrations.Get(sdkCtx, msg.ValidatorAddress)
	if err != nil {
		return nil, errorsmod.Wrap(err, "validator not registered")
	}

	hasValidatorRole := false
	for _, role := range registration.Roles {
		if role == types.NODE_TYPE_VALIDATOR {
			hasValidatorRole = true
			break
		}
	}
	if !hasValidatorRole {
		return nil, errorsmod.Wrap(err, "node does not have validator role")
	}

	// 4. Validate evidence IPFS hash
	evidenceHash := strings.TrimSpace(msg.EvidenceIpfsHash)
	if evidenceHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "evidence IPFS hash cannot be empty")
	}

	// 5. Get or create validator verification record
	verificationRecord, err := k.ValidatorVerificationRecords.Get(ctx, msg.ValidatorAddress)
	if err != nil {
		// Create new record
		verificationRecord = types.ValidatorVerificationRecord{
			ValidatorAddress:        msg.ValidatorAddress,
			TotalVerifications:      0,
			SuccessfulVerifications: 0,
			FalseVerdicts:           0,
			LazyValidationCount:     0,
			LastVerificationHeight:  sdkCtx.BlockHeight(),
		}
	}

	// 6. Increment lazy validation count
	verificationRecord.LazyValidationCount++

	// 7. Check if threshold reached (e.g., 3 lazy validations = slash)
	lazyThreshold := uint64(3)
	if verificationRecord.LazyValidationCount >= lazyThreshold {
		// Slash validator (20% for lazy validation)
		slashFraction := sdkmath.LegacyMustNewDecFromStr("0.20") // 20%
		if err := k.SlashNode(ctx, msg.ValidatorAddress, slashFraction, "lazy_validation"); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to slash validator: %v", err))
		}
		// Reset lazy validation count after slashing
		verificationRecord.LazyValidationCount = 0
	}

	// 8. Update verification record
	if err := k.ValidatorVerificationRecords.Set(ctx, msg.ValidatorAddress, verificationRecord); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update verification record")
	}

	// 9. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeReportLazyValidation,
			sdk.NewAttribute(types.AttributeKeyValidator, msg.ValidatorAddress),
			sdk.NewAttribute(types.AttributeKeyReporter, msg.Reporter),
			sdk.NewAttribute("lazy_count", fmt.Sprintf("%d", verificationRecord.LazyValidationCount)),
		),
	)

	return &types.MsgReportLazyValidationResponse{
		Reported: true,
	}, nil
}

// ReportFalseVerdict handles MsgReportFalseVerdict
// Reports false verdict by a validator (50% slash)
func (k msgServer) ReportFalseVerdict(ctx context.Context, msg *types.MsgReportFalseVerdict) (*types.MsgReportFalseVerdictResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate reporter address
	reporterAddr, err := k.addressCodec.StringToBytes(msg.Reporter)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid reporter address: %s", msg.Reporter)
	}
	_ = reporterAddr

	// 2. Validate validator address
	validatorAddr, err := k.addressCodec.StringToBytes(msg.ValidatorAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid validator address: %s", msg.ValidatorAddress)
	}
	_ = validatorAddr

	// 3. Verify challenge exists
	challenge, err := k.ChallengeRecords.Get(ctx, msg.ChallengeId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "challenge not found")
	}

	// 4. Verify validator was in CPU verification panel
	isInPanel := false
	for _, panelValidator := range challenge.CpuVerificationPanel {
		if panelValidator == msg.ValidatorAddress {
			isInPanel = true
			break
		}
	}
	if !isInPanel {
		return nil, errorsmod.Wrap(err, "validator was not in CPU verification panel")
	}

	// 5. Get validator verification record
	verificationRecord, err := k.ValidatorVerificationRecords.Get(ctx, msg.ValidatorAddress)
	if err != nil {
		verificationRecord = types.ValidatorVerificationRecord{
			ValidatorAddress:        msg.ValidatorAddress,
			TotalVerifications:      0,
			SuccessfulVerifications: 0,
			FalseVerdicts:           0,
			LazyValidationCount:     0,
			LastVerificationHeight:  sdkCtx.BlockHeight(),
		}
	}

	// 6. Increment false verdict count
	verificationRecord.FalseVerdicts++

	// 7. Slash validator (50% for false verdict - malicious validation)
	slashFraction := sdkmath.LegacyMustNewDecFromStr("0.50") // 50%
	if err := k.SlashNode(ctx, msg.ValidatorAddress, slashFraction, "false_verdict"); err != nil {
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to slash validator: %v", err))
	}
	
	// Distribute fraud detection bounty to reporter (10-20x base reward)
	baseReward := sdk.NewCoins(sdk.NewCoin("remes", sdkmath.NewInt(30))) // BaseValidatorReward
	severity := 0.9 // High severity (malicious validation)
	if err := k.DistributeFraudDetectionBounty(ctx, msg.Reporter, baseReward, severity, "false_verdict"); err != nil {
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to distribute fraud detection bounty: %v", err))
	}

	// 8. Update verification record
	if err := k.ValidatorVerificationRecords.Set(ctx, msg.ValidatorAddress, verificationRecord); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update verification record")
	}

	// 9. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeReportFalseVerdict,
			sdk.NewAttribute(types.AttributeKeyValidator, msg.ValidatorAddress),
			sdk.NewAttribute(types.AttributeKeyReporter, msg.Reporter),
			sdk.NewAttribute(types.AttributeKeyChallengeID, fmt.Sprintf("%d", msg.ChallengeId)),
		),
	)

	return &types.MsgReportFalseVerdictResponse{
		Reported: true,
	}, nil
}

// ReportProposerCensorship handles MsgReportProposerCensorship
// Reports censorship by a proposer (10% slash)
func (k msgServer) ReportProposerCensorship(ctx context.Context, msg *types.MsgReportProposerCensorship) (*types.MsgReportProposerCensorshipResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate reporter address
	reporterAddr, err := k.addressCodec.StringToBytes(msg.Reporter)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid reporter address: %s", msg.Reporter)
	}
	_ = reporterAddr

	// 2. Validate proposer address
	proposerAddr, err := k.addressCodec.StringToBytes(msg.ProposerAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid proposer address: %s", msg.ProposerAddress)
	}
	_ = proposerAddr

	// 3. Verify aggregation exists
	aggregation, err := k.AggregationRecords.Get(ctx, msg.AggregationId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "aggregation not found")
	}

	// 4. Verify proposer matches
	if aggregation.Proposer != msg.ProposerAddress {
		return nil, errorsmod.Wrap(err, "proposer does not match aggregation")
	}

	// 5. Verify excluded gradients are valid and should have been included
	validExcludedCount := uint64(0)
	for _, gradientID := range msg.ExcludedGradientIds {
		gradient, err := k.StoredGradients.Get(ctx, gradientID)
		if err != nil {
			continue
		}
		// Check if gradient is valid and should have been included
		if gradient.Status == "pending" && gradient.ModelVersion == aggregation.ModelVersion {
			validExcludedCount++
		}
	}

	if validExcludedCount == 0 {
		return nil, errorsmod.Wrap(err, "no valid excluded gradients found")
	}

	// 6. Get or create proposer censorship record
	censorshipRecord, err := k.ProposerCensorshipRecords.Get(ctx, msg.ProposerAddress)
	if err != nil {
		censorshipRecord = types.ProposerCensorshipRecord{
			ProposerAddress:        msg.ProposerAddress,
			TotalAggregations:      0,
			CensoredGradients:      0,
			LastAggregationHeight:  sdkCtx.BlockHeight(),
		}
	}

	// 7. Update censorship record
	censorshipRecord.CensoredGradients += validExcludedCount

	// 8. Check if threshold reached (e.g., 5 censored gradients = slash)
	censorshipThreshold := uint64(5)
	if censorshipRecord.CensoredGradients >= censorshipThreshold {
		// Slash proposer (10% for censorship)
		slashFraction := sdkmath.LegacyMustNewDecFromStr("0.10") // 10%
		if err := k.SlashNode(ctx, msg.ProposerAddress, slashFraction, "proposer_censorship"); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to slash proposer: %v", err))
		}
		// Reset censored gradients count after slashing
		censorshipRecord.CensoredGradients = 0
	}

	// 9. Update censorship record
	if err := k.ProposerCensorshipRecords.Set(ctx, msg.ProposerAddress, censorshipRecord); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update censorship record")
	}

	// 10. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeReportProposerCensorship,
			sdk.NewAttribute(types.AttributeKeyProposer, msg.ProposerAddress),
			sdk.NewAttribute(types.AttributeKeyReporter, msg.Reporter),
			sdk.NewAttribute(types.AttributeKeyAggregationID, fmt.Sprintf("%d", msg.AggregationId)),
		),
	)

	return &types.MsgReportProposerCensorshipResponse{
		Reported: true,
	}, nil
}

// AppealSlashing handles MsgAppealSlashing
// Appeals a slashing decision with CPU verification
func (k msgServer) AppealSlashing(ctx context.Context, msg *types.MsgAppealSlashing) (*types.MsgAppealSlashingResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate appellant address
	appellantAddr, err := k.addressCodec.StringToBytes(msg.Appellant)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid appellant address: %s", msg.Appellant)
	}
	_ = appellantAddr

	// 2. Validate evidence IPFS hash
	evidenceHash := strings.TrimSpace(msg.EvidenceIpfsHash)
	if evidenceHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "evidence IPFS hash cannot be empty")
	}

	// 3. Generate unique appeal ID
	appealID := fmt.Sprintf("%s|%s|%d", msg.Appellant, msg.SlashingReason, sdkCtx.BlockHeight())

	// 4. Create appeal
	appeal := types.SlashingAppeal{
		AppealId:         appealID,
		SlashedNode:      msg.Appellant,
		SlashingReason:   strings.TrimSpace(msg.SlashingReason),
		AppealReason:     strings.TrimSpace(msg.AppealReason),
		EvidenceIpfsHash: evidenceHash,
		CreatedAtHeight:  sdkCtx.BlockHeight(),
		Status:           "pending",
	}

	// 5. Store appeal
	if err := k.SlashingAppeals.Set(ctx, appealID, appeal); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store appeal")
	}

	// 6. Trigger CPU Iron Sandbox verification for appeal
	// Select CPU verification panel for appeal verification
	cpuPanel := k.selectCPUVerificationPanel(sdkCtx, 3, fmt.Sprintf("appeal_%s", appealID))
	if len(cpuPanel) > 0 {
		// Create CPU verification request for appeal
		// The panel will verify the evidence using CPU mode for bit-exact verification
		// Note: CPU panel selection is logged, actual verification happens via SubmitCPUVerification
		appeal.Status = "verifying"
		
		// Update appeal status
		if err := k.SlashingAppeals.Set(ctx, appealID, appeal); err != nil {
			return nil, errorsmod.Wrap(err, "failed to update appeal status")
		}
		
		// Emit event for CPU panel selection (using existing appeal event)
		sdkCtx.EventManager().EmitEvent(
			sdk.NewEvent(
				types.EventTypeAppealSlashing,
				sdk.NewAttribute(types.AttributeKeyAppealID, appealID),
				sdk.NewAttribute("status", "verifying"),
				sdk.NewAttribute("panel_size", fmt.Sprintf("%d", len(cpuPanel))),
			),
		)
	} else {
		// No validators available, keep status as pending
		sdkCtx.Logger().Warn("No CPU verification panel available for appeal")
	}

	// 7. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeAppealSlashing,
			sdk.NewAttribute(types.AttributeKeyAppealID, appealID),
			sdk.NewAttribute(types.AttributeKeyAppellant, msg.Appellant),
			sdk.NewAttribute(types.AttributeKeyReason, msg.SlashingReason),
		),
	)

	return &types.MsgAppealSlashingResponse{
		AppealId: appealID,
	}, nil
}

// RegisterMentorRelationship handles MsgRegisterMentorRelationship
// Registers a mentor-mentee relationship for reduced staking requirements
func (k msgServer) RegisterMentorRelationship(ctx context.Context, msg *types.MsgRegisterMentorRelationship) (*types.MsgRegisterMentorRelationshipResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate mentor address
	mentorAddr, err := k.addressCodec.StringToBytes(msg.Mentor)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid mentor address: %s", msg.Mentor)
	}
	_ = mentorAddr

	// 2. Validate mentee address
	menteeAddr, err := k.addressCodec.StringToBytes(msg.Mentee)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid mentee address: %s", msg.Mentee)
	}
	_ = menteeAddr

	// 3. Verify mentor is registered and has good reputation
	mentorRegistration, err := k.NodeRegistrations.Get(sdkCtx, msg.Mentor)
	if err != nil {
		return nil, errorsmod.Wrap(err, "mentor not registered")
	}

	// Check if mentor has good reputation (low slashing events)
	if mentorRegistration.SlashingEvents > 2 {
		return nil, errorsmod.Wrap(err, "mentor has too many slashing events")
	}

	// 4. Validate reduced stake percentage
	reducedStake, err := sdkmath.LegacyNewDecFromStr(msg.ReducedStakePercentage)
	if err != nil {
		return nil, errorsmod.Wrap(err, "invalid reduced stake percentage")
	}

	// Must be between 0.1 (10%) and 0.9 (90%)
	if reducedStake.LT(sdkmath.LegacyMustNewDecFromStr("0.1")) || reducedStake.GT(sdkmath.LegacyMustNewDecFromStr("0.9")) {
		return nil, errorsmod.Wrap(err, "reduced stake percentage must be between 0.1 and 0.9")
	}

	// 5. Generate relationship key
	relationshipKey := fmt.Sprintf("%s|%s", msg.Mentor, msg.Mentee)

	// 6. Check if relationship already exists
	_, err = k.MentorRelationships.Get(ctx, relationshipKey)
	if err == nil {
		return nil, errorsmod.Wrap(err, "mentor relationship already exists")
	}

	// 7. Create mentor relationship
	relationship := types.MentorRelationship{
		MentorAddress:          msg.Mentor,
		MenteeAddress:          msg.Mentee,
		ReducedStakePercentage: msg.ReducedStakePercentage,
		CreatedAtHeight:        sdkCtx.BlockHeight(),
		Status:                 "active",
	}

	// 8. Store relationship
	if err := k.MentorRelationships.Set(ctx, relationshipKey, relationship); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store mentor relationship")
	}

	// 9. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeRegisterMentorRelationship,
			sdk.NewAttribute(types.AttributeKeyMentor, msg.Mentor),
			sdk.NewAttribute(types.AttributeKeyMentee, msg.Mentee),
			sdk.NewAttribute("reduced_stake", msg.ReducedStakePercentage),
		),
	)

	return &types.MsgRegisterMentorRelationshipResponse{
		Registered: true,
	}, nil
}

