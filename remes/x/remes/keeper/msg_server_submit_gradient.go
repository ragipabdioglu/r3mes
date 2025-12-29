package keeper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// SubmitGradient handles MsgSubmitGradient
// IMPORTANT: Handler receives only IPFS hash + metadata (NO gradient data)
// Python miner uploads gradient data directly to IPFS before sending this message
// Memory Efficient: Go node never holds gradient data in memory during submission
// IPFS retrieval happens only on-demand for validation (passive role)
func (k msgServer) SubmitGradient(ctx context.Context, msg *types.MsgSubmitGradient) (*types.MsgSubmitGradientResponse, error) {
	// Convert context to SDK context for block height access
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Debug logging and profiling
	debugLogger := k.Keeper.GetDebugLogger()
	profiler := k.Keeper.GetDebugProfiler()
	traceCollector := k.Keeper.GetTraceCollector()
	var traceEntry *TraceEntry
	if traceCollector != nil {
		traceID := GenerateTraceID()
		traceEntry = traceCollector.StartTrace(traceID, "submit_gradient", sdkCtx)
	}

	var endProfiler func()
	if profiler != nil {
		endProfiler = profiler.StartTimer("SubmitGradient")
		defer endProfiler()
	}

	if debugLogger != nil {
		gradientHashShort := msg.GradientHash
		if len(gradientHashShort) > 16 {
			gradientHashShort = gradientHashShort[:16]
		}
		ipfsHashShort := msg.IpfsHash
		if len(ipfsHashShort) > 16 {
			ipfsHashShort = ipfsHashShort[:16]
		}
		debugLogger.Debug(sdkCtx, "Gradient submission started",
			"miner", msg.Miner,
			"gradient_hash", gradientHashShort,
			"ipfs_hash", ipfsHashShort,
			"training_round_id", msg.TrainingRoundId,
		)
	}

	// 1. Validate miner address
	minerAddr, err := k.addressCodec.StringToBytes(msg.Miner)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid miner address: %s", msg.Miner)
	}
	_ = minerAddr // Address validated

	// 2. Validate IPFS hash format (basic validation - should start with Qm or bafy for IPFS)
	ipfsHash := strings.TrimSpace(msg.IpfsHash)
	if ipfsHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "IPFS hash cannot be empty")
	}
	// Basic IPFS hash validation (CID v0 starts with Qm, CID v1 starts with bafy)
	if !strings.HasPrefix(ipfsHash, "Qm") && !strings.HasPrefix(ipfsHash, "bafy") && !strings.HasPrefix(ipfsHash, "bafk") {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", ipfsHash)
	}

	// 3. Validate gradient hash
	gradientHash := strings.TrimSpace(msg.GradientHash)
	if gradientHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "gradient hash cannot be empty")
	}

	// 4. Validate model version (backward compatibility - required if model_config_id is 0)
	// If model_config_id is 0, model_version must be provided for legacy compatibility
	// Also validate that the model version is active and compatible
	if msg.ModelConfigId == 0 && msg.ModelVersion == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidModelVersion, "model version cannot be empty when model_config_id is 0 (legacy mode)")
	}

	// 4.5. Validate model_config_id (if provided)
	if msg.ModelConfigId > 0 {
		if err := k.Keeper.ValidateModelConfig(sdkCtx, msg.ModelConfigId); err != nil {
			return nil, errorsmod.Wrap(err, "invalid model_config_id")
		}
	} else {
		// Legacy mode: model_config_id = 0, validate that default model (ID=1) exists
		if err := k.Keeper.ValidateModelConfig(sdkCtx, 0); err != nil {
			return nil, errorsmod.Wrap(err, "legacy model requires default model (ID=1) to exist")
		}
	}

	// 5. Check rate limiting
	if err := k.CheckRateLimit(sdkCtx, msg.Miner); err != nil {
		return nil, err
	}

	// 6. Validate GPU architecture (check against whitelist)
	gpuArch := strings.TrimSpace(msg.GpuArchitecture)
	if err := k.Keeper.ValidateGPUArchitecture(sdkCtx, gpuArch); err != nil {
		return nil, err
	}

	// 6. Validate nonce (replay attack prevention)
	if err := k.VerifyNonce(sdkCtx, msg.Miner, msg.Nonce); err != nil {
		return nil, err
	}

	// 6.5. Verify shard assignment (deterministic verification)
	// Shard assignment must be correct to ensure deterministic data distribution
	totalShards := uint64(100) // Default: 100 shards (can be made configurable)
	isValidShard, err := k.VerifyShardAssignment(sdkCtx, msg.Miner, msg.ShardId, msg.TrainingRoundId, totalShards)
	if err != nil {
		return nil, errorsmod.Wrap(err, "shard verification failed")
	}
	if !isValidShard {
		expectedShardID, calcErr := k.CalculateShardID(sdkCtx, msg.Miner, msg.TrainingRoundId, totalShards)
		if calcErr != nil {
			return nil, errorsmod.Wrapf(
				types.ErrInvalidGradientHash,
				"shard assignment mismatch: claimed %d, but could not calculate expected shard",
				msg.ShardId,
			)
		}
		return nil, errorsmod.Wrapf(
			types.ErrInvalidGradientHash,
			"shard assignment mismatch: claimed %d, expected %d",
			msg.ShardId,
			expectedShardID,
		)
	}

	// 6.5. Validate chunk size (Fixed Chunk / Variable Speed Protocol)
	// Protocol mandates exactly 2048 tokens per chunk
	// Validate token_count if provided (metadata-level validation without IPFS retrieval)
	if msg.TokenCount > 0 && msg.TokenCount != 2048 {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidGradientHash,
			"invalid token count: expected 2048 (protocol standard), got %d",
			msg.TokenCount,
		)
	}

	// 6.6. Verify Proof of Replication (PoRep) if provided
	// PoRep proves that the miner actually stores the gradient data, not just the hash
	if msg.PorepProofIpfsHash != "" {
		porepValid, err := k.VerifyPoRepFromIPFS(ctx, msg.PorepProofIpfsHash, msg.Miner, msg.IpfsHash)
		if err != nil {
			return nil, errorsmod.Wrap(err, "PoRep verification failed")
		}
		if !porepValid {
			return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "PoRep verification failed: invalid proof")
		}
	} else {
		// PoRep is optional for now, but can be made mandatory via params
		// For backward compatibility, we allow submissions without PoRep
		sdkCtx.Logger().Info(fmt.Sprintf("Gradient submission from %s without PoRep proof", msg.Miner))
	}

	// Mark nonce as used (after successful validation, before storing gradient)
	// This prevents replay attacks
	if err := k.MarkNonceAsUsed(sdkCtx, msg.Miner, msg.Nonce); err != nil {
		return nil, errorsmod.Wrap(err, "failed to mark nonce as used")
	}

	// 7. Create message hash for signature verification
	chainID := sdkCtx.ChainID()
	messageHash := CreateMessageHash(
		chainID,
		msg.Miner,
		msg.IpfsHash,
		msg.ModelVersion,
		msg.TrainingRoundId,
		msg.ShardId,
		msg.GradientHash,
		msg.GpuArchitecture,
		msg.Nonce,
	)

	// 8. Verify signature
	// Signature is required for security. Transaction signature alone is not sufficient
	// because it doesn't prove the miner signed the specific message content.
	if len(msg.Signature) == 0 {
		return nil, errorsmod.Wrapf(types.ErrInvalidSignature, "signature cannot be empty: message signature is required")
	}

	// Verify message signature
	if err := k.VerifyMessageSignature(sdkCtx, msg.Miner, messageHash, msg.Signature); err != nil {
		return nil, errorsmod.Wrap(err, "signature verification failed")
	}

	// 9. Verify IPFS content exists
	if err := k.VerifyIPFSContentExists(sdkCtx, msg.IpfsHash); err != nil {
		return nil, err
	}

	// 10. Verify proof-of-work (anti-spam mechanism)
	if msg.ProofOfWorkNonce > 0 {
		powDifficulty := uint8(4) // 4 leading zeros
		if err := k.VerifyProofOfWork(sdkCtx, messageHash, msg.ProofOfWorkNonce, powDifficulty); err != nil {
			return nil, errorsmod.Wrap(err, "proof-of-work verification failed")
		}
	}

	// 11. Check staking requirement
	if err := k.CheckStakingRequirement(sdkCtx, msg.Miner); err != nil {
		return nil, err
	}

	// 12. Record submission for rate limiting
	if err := k.RecordSubmission(sdkCtx, msg.Miner); err != nil {
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to record submission: %v", err))
	}

	// 12.5. OPTIMISTIC VERIFICATION: Layer 1 - GPU-to-GPU Verification (Optimistic Acceptance)
	// Note: expectedHash is not available at submission time (calculated by validators later)
	// For now, we optimistically accept all submissions (Layer 1 fast path)
	// Hash verification happens later during aggregation or challenge
	// This enables fast path (~95% of cases) with no slashing on acceptance
	_ = gradientHash // Gradient hash stored for later verification

	// 13. Generate unique stored gradient ID
	// Note: Sequence.Next() accepts both context.Context and sdk.Context
	// Using ctx (context.Context) for consistency with other message handlers
	gradientID, err := k.StoredGradientID.Next(ctx)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to generate gradient ID")
	}

	// 14. Derive global seed if not provided
	globalSeed := msg.GlobalSeed
	if globalSeed == 0 {
		derivedSeed, err := k.DeriveGlobalSeed(sdkCtx, msg.TrainingRoundId)
		if err != nil {
			return nil, errorsmod.Wrap(err, "failed to derive global seed")
		}
		globalSeed = derivedSeed
	}

	// 14.5. Validate deterministic execution (seed locking + quantization)
	// This ensures the gradient was computed deterministically
	if err := k.ValidateDeterministicExecution(
		sdkCtx,
		msg.Miner,
		msg.TrainingRoundId,
		globalSeed,
		gradientHash,
		msg.ModelConfigId,
	); err != nil {
		return nil, errorsmod.Wrap(err, "deterministic execution validation failed")
	}

	// 15. Create StoredGradient record
	storedGradient := types.StoredGradient{
		Id:                 gradientID,
		Miner:              msg.Miner,
		IpfsHash:           ipfsHash,
		ModelVersion:       msg.ModelVersion, // DEPRECATED: kept for backward compatibility
		TrainingRoundId:    msg.TrainingRoundId,
		ShardId:            msg.ShardId,
		GradientHash:       gradientHash,
		GpuArchitecture:    gpuArch,
		SubmittedAtHeight:  sdkCtx.BlockHeight(),
		Status:             "pending",                                      // Initial status
		ModelConfigId:      msg.ModelConfigId,                              // New: model-agnostic reference
		EncryptionType:     types.EncryptionType_ENCRYPTION_TYPE_PLAINTEXT, // Default: plaintext
		ContainerHash:      msg.ContainerHash,
		ContainerSignature: msg.ContainerSignature,
		GlobalSeed:         globalSeed,
		ClaimedLoss:        msg.ClaimedLoss,        // Loss-Based Spot Checking: miner's claimed loss
		PorepProofIpfsHash: msg.PorepProofIpfsHash, // Proof of Replication: proves miner stores actual data
	}

	// 15. Store gradient metadata on-chain
	if err := k.StoredGradients.Set(ctx, gradientID, storedGradient); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store gradient")
	}

	// 16. Update or create mining contribution record
	contribution, err := k.MiningContributions.Get(ctx, msg.Miner)
	if err != nil {
		// Create new contribution record
		contribution = types.MiningContribution{
			MinerAddress:          msg.Miner,
			TotalSubmissions:      1,
			SuccessfulSubmissions: 1,
			TrustScore:            "0.5", // Initial trust score for new miners
			ReputationTier:        "new",
			SlashingEvents:        0,
			LastSubmissionHeight:  sdkCtx.BlockHeight(),
		}
	} else {
		// Update existing contribution
		contribution.TotalSubmissions++
		contribution.SuccessfulSubmissions++
		contribution.LastSubmissionHeight = sdkCtx.BlockHeight()
	}
	if err := k.MiningContributions.Set(ctx, msg.Miner, contribution); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update mining contribution")
	}

	// 17. Update convergence metrics with claimed loss
	if msg.ClaimedLoss != "" {
		if err := k.UpdateConvergenceMetrics(sdkCtx, msg.TrainingRoundId, msg.ClaimedLoss); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to update convergence metrics: %v", err))
			// Don't fail the transaction, just log the error
		}
	}

	// 18. Update participant sync state
	if err := k.UpdateParticipantSyncState(sdkCtx, msg.Miner, msg.ModelVersion, "miner"); err != nil {
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to update participant sync state: %v", err))
	}

	// 18. Distribute miner reward
	// Reward distribution is critical for economic incentives
	// If it fails, the transaction should fail to ensure consistency
	if err := k.DistributeMinerReward(ctx, msg.Miner, gradientID); err != nil {
		return nil, errorsmod.Wrap(err, "failed to distribute miner reward")
	}

	// 19. Emit event for gradient submission
	sdkCtx.EventManager().EmitEvent(
		types.NewEventGradientSubmitted(
			msg.Miner,
			gradientID,
			ipfsHash,
			msg.ModelVersion,
			msg.TrainingRoundId,
		),
	)

	// 20. Calculate transaction hash from transaction bytes
	// SDK context provides transaction bytes via TxBytes() method
	txHash := ""
	if txBytes := sdkCtx.TxBytes(); len(txBytes) > 0 {
		hash := sha256.Sum256(txBytes)
		txHash = hex.EncodeToString(hash[:])
	} else {
		// Fallback: calculate deterministic hash if TxBytes not available
		// This should not happen in normal operation, but provides backward compatibility
		hashInput := fmt.Sprintf(
			"%d|%s|%d|%d|%s|%s",
			sdkCtx.BlockHeight(),
			msg.Miner,
			gradientID,
			sdkCtx.BlockTime().Unix(),
			msg.IpfsHash,
			msg.GradientHash,
		)
		hash := sha256.Sum256([]byte(hashInput))
		txHash = hex.EncodeToString(hash[:])
	}

	// Debug logging - completion
	if debugLogger != nil {
		txHashShort := txHash
		if len(txHashShort) > 16 {
			txHashShort = txHashShort[:16]
		}
		debugLogger.Info(sdkCtx, "Gradient submission completed successfully",
			"miner", msg.Miner,
			"gradient_id", gradientID,
			"tx_hash", txHashShort,
		)
	}

	// End trace
	if traceEntry != nil && traceCollector != nil {
		traceCollector.EndTrace(traceEntry, nil)
	}

	return &types.MsgSubmitGradientResponse{
		StoredGradientId: gradientID,
		TxHash:           txHash,
	}, nil
}
