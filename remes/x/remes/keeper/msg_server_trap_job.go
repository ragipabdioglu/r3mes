package keeper

import (
	"context"
	"crypto/rand"
	"fmt"
	"strings"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// CreateTrapJob handles MsgCreateTrapJob
// Creates a trap job for lazy mining detection
func (k msgServer) CreateTrapJob(ctx context.Context, msg *types.MsgCreateTrapJob) (*types.MsgCreateTrapJobResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Verify creator is module authority or validator
	creatorAddr, err := k.addressCodec.StringToBytes(msg.Creator)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid creator address: %s", msg.Creator)
	}

	// Check if creator is module authority
	authority := sdk.AccAddress(k.GetAuthority()).String()
	isAuthority := msg.Creator == authority

	// If not authority, check if creator is a validator
	if !isAuthority {
		registration, err := k.NodeRegistrations.Get(sdkCtx, msg.Creator)
		if err != nil {
			return nil, errorsmod.Wrap(err, "creator is not authorized to create trap jobs")
		}
		hasValidatorRole := false
		for _, role := range registration.Roles {
			if role == types.NODE_TYPE_VALIDATOR {
				hasValidatorRole = true
				break
			}
		}
		if !hasValidatorRole {
			return nil, errorsmod.Wrap(err, "creator is not authorized to create trap jobs")
		}
	}
	_ = creatorAddr

	// 2. Validate target miner address
	targetMinerAddr, err := k.addressCodec.StringToBytes(msg.TargetMiner)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid target miner address: %s", msg.TargetMiner)
	}
	_ = targetMinerAddr

	// 3. Validate dataset IPFS hash
	datasetHash := strings.TrimSpace(msg.DatasetIpfsHash)
	if datasetHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "dataset IPFS hash cannot be empty")
	}
	if !strings.HasPrefix(datasetHash, "Qm") && !strings.HasPrefix(datasetHash, "bafy") && !strings.HasPrefix(datasetHash, "bafk") {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", datasetHash)
	}

	// 4. Validate expected gradient hash
	expectedHash := strings.TrimSpace(msg.ExpectedGradientHash)
	if expectedHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "expected gradient hash cannot be empty")
	}

	// 5. FEDERATED TRAP JOBS: Use federated structure (multi-sig from top 3 validators)
	// Generate unique trap job ID
	trapJobID := fmt.Sprintf("%s|%s|%d|%x", msg.Creator, msg.TargetMiner, sdkCtx.BlockHeight(), sdkCtx.HeaderHash()[:8])

	// 6. Create federated trap job (replaces single Protocol Oracle)
	federatedTrapJob, err := k.Keeper.CreateFederatedTrapJob(
		sdkCtx,
		trapJobID,
		msg.TargetMiner,
		datasetHash,
		expectedHash,
		msg.DeadlineBlocks,
	)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to create federated trap job")
	}

	// 7. Verify multi-sig if signatures provided (2/3 threshold)
	if len(msg.FederatedSignatures) > 0 {
		// Create payload for signature verification
		payloadBytes := k.Keeper.serializeTrapJobPayloadForVerification(
			trapJobID,
			datasetHash,
			expectedHash,
			sdkCtx.BlockHeight(),
			sdkCtx.BlockHeight()+int64(msg.DeadlineBlocks),
			federatedTrapJob.FederatedSigners,
		)

		// Verify multi-sig
		if err := k.Keeper.verifyFederatedTrapJobSignature(
			sdkCtx,
			payloadBytes,
			federatedTrapJob.FederatedSigners,
			msg.FederatedSignatures,
		); err != nil {
			return nil, errorsmod.Wrap(err, "federated trap job signature verification failed")
		}

		// Store signatures
		federatedTrapJob.FederatedSignatures = msg.FederatedSignatures
	}

	// 8. Generate blinding factor for trap job obfuscation
	// This prevents miners from detecting trap jobs by comparing hashes
	blindingFactor, err := k.Keeper.GenerateBlindingFactor(sdkCtx, trapJobID)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to generate blinding factor")
	}

	// Store blinding factor in trap job
	federatedTrapJob.BlindingFactor = blindingFactor
	federatedTrapJob.IsBlinded = true

	// 9. Generate obfuscation seed if obfuscation is enabled
	if msg.ObfuscateWithDummyData {
		seedBytes := make([]byte, 8)
		if _, err := rand.Read(seedBytes); err != nil {
			return nil, errorsmod.Wrap(err, "failed to generate obfuscation seed")
		}
		obfuscationSeed := uint64(seedBytes[0])<<56 | uint64(seedBytes[1])<<48 | uint64(seedBytes[2])<<40 | uint64(seedBytes[3])<<32 |
			uint64(seedBytes[4])<<24 | uint64(seedBytes[5])<<16 | uint64(seedBytes[6])<<8 | uint64(seedBytes[7])
		federatedTrapJob.ObfuscationSeed = obfuscationSeed
		federatedTrapJob.IsObfuscated = true
	}

	// 10. Use federated trap job
	trapJob := *federatedTrapJob

	// 9. Store trap job
	if err := k.TrapJobs.Set(ctx, trapJobID, trapJob); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store trap job")
	}

	// 10. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeCreateTrapJob,
			sdk.NewAttribute(types.AttributeKeyTrapJobID, trapJobID),
			sdk.NewAttribute(types.AttributeKeyTargetMiner, msg.TargetMiner),
			sdk.NewAttribute(types.AttributeKeyCreator, msg.Creator),
		),
	)

	return &types.MsgCreateTrapJobResponse{
		TrapJobId: trapJobID,
	}, nil
}

// SubmitTrapJobResult handles MsgSubmitTrapJobResult
// Miner submits result for a trap job
func (k msgServer) SubmitTrapJobResult(ctx context.Context, msg *types.MsgSubmitTrapJobResult) (*types.MsgSubmitTrapJobResultResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate miner address
	minerAddr, err := k.addressCodec.StringToBytes(msg.Miner)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid miner address: %s", msg.Miner)
	}
	_ = minerAddr

	// 2. Get trap job
	trapJob, err := k.TrapJobs.Get(ctx, msg.TrapJobId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "trap job not found")
	}

	// 3. Verify miner matches target
	if trapJob.TargetMiner != msg.Miner {
		return nil, errorsmod.Wrap(err, "miner does not match trap job target")
	}

	// 4. Check if trap job is still pending
	if trapJob.Status != "pending" {
		return nil, errorsmod.Wrap(err, "trap job is not pending")
	}

	// 5. Check if deadline has passed
	if sdkCtx.BlockHeight() > trapJob.DeadlineHeight {
		// Deadline passed - slash miner for lazy mining (100% slash)
		slashFraction := sdkmath.LegacyMustNewDecFromStr("1.0") // 100%
		if err := k.SlashMiner(ctx, msg.Miner, slashFraction, "trap_job_deadline_exceeded"); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to slash miner: %v", err))
		}
		trapJob.Status = "failed"
		trapJob.VerificationResult = false
		if err := k.TrapJobs.Set(ctx, msg.TrapJobId, trapJob); err != nil {
			return nil, errorsmod.Wrap(err, "failed to update trap job")
		}
		return &types.MsgSubmitTrapJobResultResponse{
			Verified: false,
		}, nil
	}

	// 6. Validate gradient IPFS hash
	gradientHash := strings.TrimSpace(msg.GradientIpfsHash)
	if gradientHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "gradient IPFS hash cannot be empty")
	}

	// 7. Validate gradient hash
	submittedHash := strings.TrimSpace(msg.GradientHash)
	if submittedHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "gradient hash cannot be empty")
	}

	// 8. Verify gradient hash matches expected using blinding factor
	// Use blinded verification to prevent miners from detecting trap jobs
	verified, err := k.Keeper.VerifyBlindedTrapJob(sdkCtx, msg.TrapJobId, submittedHash)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to verify blinded trap job")
	}

	// 9. Update trap job
	trapJob.Status = "verified"
	trapJob.SubmittedGradientHash = submittedHash
	trapJob.VerificationResult = verified
	if err := k.TrapJobs.Set(ctx, msg.TrapJobId, trapJob); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update trap job")
	}

	// 10. If verification failed, slash miner (100% slash for lazy mining)
	if !verified {
		slashFraction := sdkmath.LegacyMustNewDecFromStr("1.0") // 100%
		if err := k.SlashMiner(ctx, msg.Miner, slashFraction, "trap_job_verification_failed"); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to slash miner: %v", err))
		}
		trapJob.Status = "failed"
		if err := k.TrapJobs.Set(ctx, msg.TrapJobId, trapJob); err != nil {
			return nil, errorsmod.Wrap(err, "failed to update trap job")
		}
	}

	// 11. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeSubmitTrapJobResult,
			sdk.NewAttribute(types.AttributeKeyTrapJobID, msg.TrapJobId),
			sdk.NewAttribute(types.AttributeKeyMiner, msg.Miner),
			sdk.NewAttribute(types.AttributeKeyIsValid, fmt.Sprintf("%t", verified)),
		),
	)

	return &types.MsgSubmitTrapJobResultResponse{
		Verified: verified,
	}, nil
}

// AppealTrapJobSlashing handles MsgAppealTrapJobSlashing
// Miner appeals a trap job slashing decision
func (k msgServer) AppealTrapJobSlashing(ctx context.Context, msg *types.MsgAppealTrapJobSlashing) (*types.MsgAppealTrapJobSlashingResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate appellant address
	appellantAddr, err := k.addressCodec.StringToBytes(msg.Appellant)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid appellant address: %s", msg.Appellant)
	}
	_ = appellantAddr

	// 2. Get trap job
	trapJob, err := k.TrapJobs.Get(ctx, msg.TrapJobId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "trap job not found")
	}

	// 3. Verify appellant matches target miner
	if trapJob.TargetMiner != msg.Appellant {
		return nil, errorsmod.Wrap(err, "appellant does not match trap job target")
	}

	// 4. Check if trap job was failed (can only appeal failed trap jobs)
	if trapJob.Status != "failed" {
		return nil, errorsmod.Wrap(err, "trap job is not failed, cannot appeal")
	}

	// 5. Validate evidence IPFS hash
	evidenceHash := strings.TrimSpace(msg.EvidenceIpfsHash)
	if evidenceHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "evidence IPFS hash cannot be empty")
	}

	// 6. Generate unique appeal ID
	appealID := fmt.Sprintf("%s|%s|%d", msg.TrapJobId, msg.Appellant, sdkCtx.BlockHeight())

	// 7. Create appeal
	appeal := types.TrapJobAppeal{
		AppealId:         appealID,
		TrapJobId:        msg.TrapJobId,
		Appellant:        msg.Appellant,
		Reason:           strings.TrimSpace(msg.Reason),
		EvidenceIpfsHash: evidenceHash,
		CreatedAtHeight:  sdkCtx.BlockHeight(),
		Status:           "pending",
	}

	// 8. Store appeal
	if err := k.TrapJobAppeals.Set(ctx, appealID, appeal); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store appeal")
	}

	// 9. CPU Iron Sandbox verification for appeal
	// Select CPU verification panel for appeal verification
	challengeContext := fmt.Sprintf("trap_job_appeal_%s_%s", msg.TrapJobId, msg.Appellant)
	cpuPanel := k.selectCPUVerificationPanel(sdkCtx, 3, challengeContext)
	if len(cpuPanel) > 0 {
		// Create CPU verification request for appeal
		// The panel will verify the evidence using CPU mode for bit-exact verification
		// Note: CPU panel selection is logged, actual verification happens via SubmitCPUVerification
		appeal.Status = "verifying"
		
		// Update appeal status
		if err := k.TrapJobAppeals.Set(ctx, appealID, appeal); err != nil {
			return nil, errorsmod.Wrap(err, "failed to update appeal status")
		}
		
		// Emit event for CPU panel selection
		sdkCtx.EventManager().EmitEvent(
			sdk.NewEvent(
				types.EventTypeAppealTrapJobSlashing,
				sdk.NewAttribute(types.AttributeKeyAppealID, appealID),
				sdk.NewAttribute("status", "verifying"),
				sdk.NewAttribute("panel_size", fmt.Sprintf("%d", len(cpuPanel))),
			),
		)
		
		sdkCtx.Logger().Info(fmt.Sprintf("CPU verification panel selected for trap job appeal: appeal_id=%s, panel_size=%d", appealID, len(cpuPanel)))
	} else {
		// No validators available, keep status as pending
		sdkCtx.Logger().Warn(fmt.Sprintf("No CPU verification panel available for trap job appeal: appeal_id=%s", appealID))
	}

	// 10. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeAppealTrapJobSlashing,
			sdk.NewAttribute(types.AttributeKeyAppealID, appealID),
			sdk.NewAttribute(types.AttributeKeyTrapJobID, msg.TrapJobId),
			sdk.NewAttribute(types.AttributeKeyAppellant, msg.Appellant),
		),
	)

	return &types.MsgAppealTrapJobSlashingResponse{
		AppealId: appealID,
	}, nil
}

