package keeper

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// VerifyGradientLayer1 performs optimistic GPU-to-GPU verification
// This is the fast path - no slashing on acceptance
// Returns: (isValid, shouldChallenge)
func (k Keeper) VerifyGradientLayer1(
	ctx sdk.Context,
	minerAddress string,
	gradientHash string,
	expectedHash string,
	minerGPU string,
	validatorGPU string,
) (bool, bool, error) {
	// Same architecture: Direct hash comparison
	if minerGPU == validatorGPU {
		if gradientHash == expectedHash {
			// Optimistic acceptance - no slashing
			return true, false, nil
		}
		// Hash mismatch - allow but mark for potential challenge
		return false, true, nil
	}

	// Different architectures - skip to Layer 2 (no direct comparison)
	// This prevents false positives from GPU architecture differences
	return false, true, nil
}

// VerifyGradientLayer2 performs high-stakes challenge verification
// Requires bond from challenger, selects random GPU verifier
// Returns: (challengeID, error)
func (k Keeper) VerifyGradientLayer2(
	ctx sdk.Context,
	challengerAddress string,
	gradientHash string,
	expectedHash string,
	aggregationID uint64,
	bondAmount sdk.Coins,
) (uint64, error) {
	// Check bond requirement (10x base reward)
	baseReward := sdk.NewCoins(sdk.NewCoin("remes", sdkmath.NewInt(500))) // Base reward: 500 remes
	requiredBond := sdk.NewCoins()
	for _, coin := range baseReward {
		requiredBond = requiredBond.Add(sdk.NewCoin(coin.Denom, coin.Amount.Mul(sdkmath.NewInt(10))))
	}

	if !bondAmount.IsAllGTE(requiredBond) {
		return 0, errorsmod.Wrapf(
			types.ErrInsufficientStake,
			"challenge bond insufficient: required %s, got %s",
			requiredBond,
			bondAmount,
		)
	}

	// Escrow bond
	challengerAddr, err := k.addressCodec.StringToBytes(challengerAddress)
	if err != nil {
		return 0, errorsmod.Wrap(err, "invalid challenger address")
	}

	// Send bond to module account (escrow)
	if err := k.bankKeeper.SendCoinsFromAccountToModule(ctx, challengerAddr, types.ModuleName, bondAmount); err != nil {
		return 0, errorsmod.Wrap(err, "failed to escrow challenge bond")
	}

	// Generate challenge ID
	challengeID, err := k.ChallengeID.Next(ctx)
	if err != nil {
		// Refund bond on error
		_ = k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, challengerAddr, bondAmount)
		return 0, errorsmod.Wrap(err, "failed to generate challenge ID")
	}

	// Select random GPU verifier (VRF-based, using block hash)
	randomVerifier := k.selectRandomGPUVerifier(ctx, challengeID, aggregationID)

	// Create challenge record
	challengeRecord := types.ChallengeRecord{
		ChallengeId:          challengeID,
		Challenger:           challengerAddress,
		AggregationId:        aggregationID,
		Reason:               "hash_mismatch_layer1",
		EvidenceIpfsHash:     gradientHash, // Store gradient hash as evidence
		Status:               "pending",
		CreatedAtHeight:      ctx.BlockHeight(),
		Layer:                2, // Layer 2
		BondAmount:           bondAmount.String(),
		RandomVerifier:       randomVerifier,
		RandomVerifierResult: "pending",
	}

	// Store challenge record
	if err := k.ChallengeRecords.Set(ctx, challengeID, challengeRecord); err != nil {
		// Refund bond on error
		_ = k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, challengerAddr, bondAmount)
		return 0, errorsmod.Wrap(err, "failed to store challenge record")
	}

	return challengeID, nil
}

// VerifyGradientLayer3 performs CPU Iron Sandbox verification
// Only triggered if Layer 2 consensus supports challenge
func (k Keeper) VerifyGradientLayer3(
	ctx sdk.Context,
	challengeID uint64,
) error {
	challenge, err := k.ChallengeRecords.Get(ctx, challengeID)
	if err != nil {
		return errorsmod.Wrap(err, "challenge not found")
	}

	// Check if Layer 2 consensus supports challenge
	if challenge.Layer != 2 {
		return errorsmod.Wrapf(types.ErrInvalidChallenge, "Layer 3 can only be triggered after Layer 2")
	}

	// Check if random verifier agrees with challenger
	if challenge.RandomVerifierResult != "invalid" {
		// Verifier disagrees - challenge rejected, return bond to challenger
		return k.returnChallengeBond(ctx, challengeID, false)
	}

	// Verifier agrees - trigger CPU Iron Sandbox
	// Select 3-validator panel (VRF-based)
	challengeContext := fmt.Sprintf("layer3_challenge_%d", challengeID)
	panel := k.selectCPUVerificationPanel(ctx, 3, challengeContext)

	// Update challenge to Layer 3
	challenge.Layer = 3
	challenge.CpuVerificationPanel = panel
	challenge.Status = "cpu_verification_pending"

	return k.ChallengeRecords.Set(ctx, challengeID, challenge)
}

// selectRandomGPUVerifier selects a random GPU verifier using block hash for pseudo-random selection
// Uses VRF-like selection based on block hash (deterministic but unpredictable)
func (k Keeper) selectRandomGPUVerifier(ctx sdk.Context, challengeID uint64, aggregationID uint64) string {
	// Get all registered validator nodes with GPU capability
	type validatorInfo struct {
		address string
		stake   int64
	}

	var validators []validatorInfo
	_ = k.NodeRegistrations.Walk(ctx, nil, func(key string, value types.NodeRegistration) (stop bool, err error) {
		// Only select nodes with validator role and active status
		hasValidatorRole := false
		for _, role := range value.Roles {
			if role == types.NODE_TYPE_VALIDATOR && value.Status == types.NODE_STATUS_ACTIVE {
				hasValidatorRole = true
				break
			}
		}

		if hasValidatorRole {
			// Parse stake for weighting
			stake, parseErr := sdk.ParseCoinsNormalized(value.Stake)
			stakeAmount := int64(0)
			if parseErr == nil && !stake.IsZero() {
				for _, coin := range stake {
					stakeAmount += coin.Amount.Int64()
				}
			}

			if stakeAmount == 0 {
				stakeAmount = 1
			}

			validators = append(validators, validatorInfo{
				address: key,
				stake:   stakeAmount,
			})
		}
		return false, nil
	})

	if len(validators) == 0 {
		return ""
	}

	// Use block hash for pseudo-random selection (as per refactoring_plan.md note 3)
	blockHash := ctx.HeaderHash()
	seed := append(blockHash, make([]byte, 8)...)
	binary.BigEndian.PutUint64(seed[len(blockHash):], challengeID)
	seed = append(seed, make([]byte, 8)...)
	binary.BigEndian.PutUint64(seed[len(seed)-8:], aggregationID)

	hash := sha256.Sum256(seed)

	// Stake-weighted selection
	totalStake := int64(0)
	for _, v := range validators {
		totalStake += v.stake
	}

	if totalStake == 0 {
		// Fallback: select first validator
		return validators[0].address
	}

	// Use hash bytes for random selection
	randomValue := int64(0)
	for i := 0; i < 8 && i < len(hash); i++ {
		randomValue = randomValue<<8 | int64(hash[i])
	}
	if randomValue < 0 {
		randomValue = -randomValue
	}
	randomValue = randomValue % totalStake

	// Select validator based on weighted random
	cumulativeWeight := int64(0)
	for _, v := range validators {
		cumulativeWeight += v.stake
		if randomValue < cumulativeWeight {
			return v.address
		}
	}

	// Fallback: return first validator
	return validators[0].address
}

// returnChallengeBond returns the challenge bond to challenger or distributes it
// If challengeValid is true: challenger gets bond back + fraud detection bounty
// If challengeValid is false: challenger loses bond (distributed to miner + validator)
func (k Keeper) returnChallengeBond(ctx sdk.Context, challengeID uint64, challengeValid bool) error {
	challenge, err := k.ChallengeRecords.Get(ctx, challengeID)
	if err != nil {
		return errorsmod.Wrap(err, "challenge not found")
	}

	bondAmount, err := sdk.ParseCoinsNormalized(challenge.BondAmount)
	if err != nil {
		return errorsmod.Wrap(err, "failed to parse bond amount")
	}

	challengerAddr, err := k.addressCodec.StringToBytes(challenge.Challenger)
	if err != nil {
		return errorsmod.Wrap(err, "invalid challenger address")
	}

	if challengeValid {
		// Challenge valid: Return bond + fraud detection bounty (10x base reward)
		baseReward := sdk.NewCoins(sdk.NewCoin("remes", sdkmath.NewInt(500)))
		bounty := sdk.NewCoins()
		for _, coin := range baseReward {
			bounty = bounty.Add(sdk.NewCoin(coin.Denom, coin.Amount.Mul(sdkmath.NewInt(10))))
		}

		totalReturn := bondAmount.Add(bounty...)
		if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, challengerAddr, totalReturn); err != nil {
			return errorsmod.Wrap(err, "failed to return bond and bounty")
		}
	} else {
		// Challenge invalid: Distribute bond to miner and validator
		// Get aggregation to find miner
		aggregation, err := k.AggregationRecords.Get(ctx, challenge.AggregationId)
		if err == nil {
			// Split bond: 50% to miner (proposer), 50% to random verifier
			halfBond := sdk.NewCoins()
			for _, coin := range bondAmount {
				halfAmount := coin.Amount.Quo(sdkmath.NewInt(2))
				halfBond = halfBond.Add(sdk.NewCoin(coin.Denom, halfAmount))
			}

			// Send to proposer (miner)
			if aggregation.Proposer != "" {
				proposerAddr, err := k.addressCodec.StringToBytes(aggregation.Proposer)
				if err == nil {
					_ = k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, proposerAddr, halfBond)
				}
			}

			// Send to random verifier
			if challenge.RandomVerifier != "" {
				verifierAddr, err := k.addressCodec.StringToBytes(challenge.RandomVerifier)
				if err == nil {
					_ = k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, verifierAddr, halfBond)
				}
			}
		}
	}

	return nil
}

// PerformLossBasedSpotCheck performs Loss-Based Spot Checking for Layer 2 verification
// Instead of expensive full training re-runs, validators perform forward pass (inference)
// on a deterministic random batch and compare the calculated loss with miner's claimed loss
func (k Keeper) PerformLossBasedSpotCheck(
	ctx sdk.Context,
	challengeID uint64,
	minerWeightsIPFS string,
	minerClaimedLoss string,
	dataBatchSeed uint64,
	modelConfigID uint64,
) (string, bool, error) {
	// 1. Validate inputs
	if minerWeightsIPFS == "" {
		return "", false, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "miner weights IPFS hash cannot be empty")
	}
	if minerClaimedLoss == "" {
		return "", false, errorsmod.Wrapf(types.ErrInvalidGradientHash, "miner claimed loss cannot be empty")
	}

	// 2. Parse miner's claimed loss (BitNet integer format)
	// BitNet uses integer representation, so we parse as int64
	minerLossInt, err := parseBitNetInteger(minerClaimedLoss)
	if err != nil {
		return "", false, errorsmod.Wrap(err, "failed to parse miner claimed loss")
	}

	// 3. Download miner's weights from IPFS (off-chain operation)
	// In production, this would be done by the verifier node off-chain
	// For now, we just validate the IPFS hash format
	if !isValidIPFSHash(minerWeightsIPFS) {
		return "", false, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid miner weights IPFS hash format")
	}

	// 4. Select deterministic random batch using VRF seed
	// The same seed will always produce the same batch selection
	// This ensures reproducibility across verifiers
	selectedBatch := k.selectDeterministicBatch(ctx, dataBatchSeed, modelConfigID)

	// 5. Get dataset IPFS hash (optional - verification service may use default dataset)
	// For now, use empty string - verification service will use its configured dataset
	// In the future, this could be retrieved from model configuration or approved datasets
	datasetIPFS := ""

	// 6. Run forward pass (inference) - NOT training
	// This is ~100x cheaper than full training
	// In production, this is done by the Python inference service via gRPC
	verifierLossInt, err := k.runForwardPass(ctx, minerWeightsIPFS, selectedBatch, modelConfigID, datasetIPFS)
	if err != nil {
		return "", false, errorsmod.Wrap(err, "failed to run forward pass")
	}

	// 6. Compare losses with BitNet integer tolerance
	// BitNet tolerance: ±1 integer unit (due to quantization)
	tolerance := int64(1)
	lossDiff := absInt64(minerLossInt - verifierLossInt)
	lossMatch := lossDiff <= tolerance

	// 7. Convert verifier loss back to string (BitNet integer format)
	verifierLoss := fmt.Sprintf("%d", verifierLossInt)

	return verifierLoss, lossMatch, nil
}

// parseBitNetInteger parses a BitNet integer string to int64
func parseBitNetInteger(lossStr string) (int64, error) {
	// BitNet uses integer representation for loss
	// Parse as int64
	var loss int64
	_, err := fmt.Sscanf(lossStr, "%d", &loss)
	if err != nil {
		return 0, fmt.Errorf("invalid BitNet integer format: %w", err)
	}
	return loss, nil
}

// formatBitNetInteger formats an int64 loss value as BitNet integer string
func formatBitNetInteger(loss int64) string {
	return fmt.Sprintf("%d", loss)
}

// isValidIPFSHash validates IPFS hash format
func isValidIPFSHash(hash string) bool {
	// IPFS hashes can be CIDv0 (Qm...) or CIDv1 (bafy...)
	return (len(hash) >= 10 && (hash[:2] == "Qm" || hash[:4] == "bafy" || hash[:4] == "bafk"))
}

// selectDeterministicBatch selects a deterministic random batch using VRF seed
// The same seed will always produce the same batch selection
func (k Keeper) selectDeterministicBatch(ctx sdk.Context, seed uint64, modelConfigID uint64) string {
	// Use block hash + seed + model config ID for deterministic selection
	blockHash := ctx.HeaderHash()
	seedBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(seedBytes, seed)
	modelBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(modelBytes, modelConfigID)

	combined := append(blockHash, seedBytes...)
	combined = append(combined, modelBytes...)
	hash := sha256.Sum256(combined)

	// Return batch identifier (in production, this would be a batch ID or hash)
	return fmt.Sprintf("batch_%x", hash[:16])
}

// runForwardPass runs a real forward pass (inference) via Python inference service
// This replaces the previous simulation with actual model inference
// In production mode, simulation fallback is disabled - service must be available
func (k Keeper) runForwardPass(ctx sdk.Context, weightsIPFS string, batchID string, modelConfigID uint64, datasetIPFS string) (int64, error) {
	// Check if we're in test mode
	isTestMode := os.Getenv("R3MES_TEST_MODE") == "true"

	// Try to use verification service (Python inference engine)
	client, err := NewVerificationClient()
	if err != nil {
		if isTestMode {
			// In test mode, allow simulation fallback
			ctx.Logger().Warn("Verification service not available, using simulation fallback (TEST MODE)", "error", err)
			return k.runForwardPassSimulation(ctx, weightsIPFS, batchID, modelConfigID)
		}
		// PRODUCTION MODE: Fail-fast - verification service must be available
		return 0, fmt.Errorf("verification service not available - required in production: %w", err)
	}
	defer client.Close()

	// Check if service is healthy
	healthy, err := client.HealthCheck(ctx)
	if err != nil || !healthy {
		if isTestMode {
			// In test mode, allow simulation fallback
			ctx.Logger().Warn("Verification service health check failed, using simulation fallback (TEST MODE)", "error", err)
			return k.runForwardPassSimulation(ctx, weightsIPFS, batchID, modelConfigID)
		}
		// PRODUCTION MODE: Fail-fast - service must be healthy
		return 0, fmt.Errorf("verification service health check failed - required in production: %w", err)
	}

	// Call Python inference service
	lossInt, err := client.RunForwardPass(
		ctx,
		weightsIPFS,
		batchID,
		modelConfigID,
		datasetIPFS,
		300, // 5 minute timeout
	)
	if err != nil {
		if isTestMode {
			// In test mode, allow simulation fallback
			ctx.Logger().Warn("Verification service call failed, using simulation fallback (TEST MODE)", "error", err)
			return k.runForwardPassSimulation(ctx, weightsIPFS, batchID, modelConfigID)
		}
		// PRODUCTION MODE: Fail-fast - service call must succeed
		return 0, fmt.Errorf("verification service call failed - required in production: %w", err)
	}

	ctx.Logger().Info("Forward pass completed via verification service", "loss_int", lossInt)
	return lossInt, nil
}

// runForwardPassSimulation is a fallback simulation method for TEST MODE ONLY
// This is only used when R3MES_TEST_MODE=true and verification service is unavailable
// In production, verification service must be available (fail-fast)
// NOTE: Kept for testing purposes only - production code should never use this
func (k Keeper) runForwardPassSimulation(ctx sdk.Context, weightsIPFS string, batchID string, modelConfigID uint64) (int64, error) {
	// TEST MODE ONLY: This is a simulation fallback for testing
	// In production, use runForwardPass which calls the real Python inference service
	ctx.Logger().Warn("Using simulation method for forward pass (TEST MODE ONLY)")

	// For now, we simulate by using a deterministic hash-based approach
	// This ensures the same inputs produce the same output
	seed := fmt.Sprintf("%s_%s_%d", weightsIPFS, batchID, modelConfigID)
	hash := sha256.Sum256([]byte(seed))

	// Convert hash to int64 (simulated loss)
	lossInt := int64(0)
	for i := 0; i < 8 && i < len(hash); i++ {
		lossInt = lossInt<<8 | int64(hash[i])
	}
	if lossInt < 0 {
		lossInt = -lossInt
	}

	// Normalize to reasonable loss range (0-10000 for BitNet integer)
	lossInt = lossInt % 10000

	return lossInt, nil
}

// absInt64 returns absolute value of int64
func absInt64(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}

