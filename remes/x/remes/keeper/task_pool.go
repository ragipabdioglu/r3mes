package keeper

import (
	"context"
	"fmt"
	"math/rand"

	errorsmod "cosmossdk.io/errors"
	"cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// CreateTaskPool creates a new task pool with multiple chunks
// Each block can have 50 chunks, total pool size = chunksPerBlock * windowBlocks
func (k Keeper) CreateTaskPool(ctx context.Context, windowID uint64, chunksPerBlock uint64, windowBlocks int64) (types.TaskPool, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get next pool ID
	poolID, err := k.TaskPoolID.Next(ctx)
	if err != nil {
		return types.TaskPool{}, errorsmod.Wrap(err, "failed to get next pool ID")
	}

	// Calculate total chunks
	totalChunks := chunksPerBlock * uint64(windowBlocks)

	// Create pool
	pool := types.TaskPool{
		PoolId:          poolID,
		WindowId:        windowID,
		TotalChunks:     totalChunks,
		AvailableChunks: make([]types.TaskChunk, 0, totalChunks),
		CompletedChunks: make([]string, 0),
		Status:          "active",
		CreatedHeight:   sdkCtx.BlockHeight(),
		ExpiryHeight:    sdkCtx.BlockHeight() + windowBlocks,
	}

	// Generate chunks with deterministic shard assignment
	for i := uint64(0); i < totalChunks; i++ {
		chunk := types.TaskChunk{
			ChunkId:       i,
			DataHash:      k.GenerateChunkDataHash(windowID, i),
			ShardId:       i % 100, // 100 shards
			AssignedMiner: "",
			Status:        "available",
			ClaimedAt:     0,
			CompletedAt:   0,
			GradientHash:  "",
		}
		pool.AvailableChunks = append(pool.AvailableChunks, chunk)
	}

	// Validate pool
	if err := pool.Validate(); err != nil {
		return types.TaskPool{}, errorsmod.Wrap(err, "invalid task pool")
	}

	// Store pool
	if err := k.TaskPools.Set(ctx, poolID, pool); err != nil {
		return types.TaskPool{}, errorsmod.Wrap(err, "failed to store task pool")
	}

	return pool, nil
}

// GetTaskPool retrieves a task pool by ID
func (k Keeper) GetTaskPool(ctx context.Context, poolID uint64) (types.TaskPool, error) {
	pool, err := k.TaskPools.Get(ctx, poolID)
	if err != nil {
		return types.TaskPool{}, errorsmod.Wrapf(err, "task pool %d not found", poolID)
	}
	return pool, nil
}

// ClaimTask claims an available chunk from the task pool
func (k Keeper) ClaimTask(ctx context.Context, miner string, poolID uint64, chunkID uint64) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Check if miner is registered as a node (optional but recommended)
	// This ensures accountability and proper resource tracking
	// Note: Registration is not strictly required - miners can claim tasks with sufficient balance
	// but registration is recommended for better tracking and resource management
	_, err := k.NodeRegistrations.Get(sdkCtx, miner)
	if err != nil {
		// Miner is not registered - log warning but allow if they have sufficient stake/balance
		// Check staking requirement as fallback (checks bank balance if not registered)
		if stakeErr := k.CheckStakingRequirement(sdkCtx, miner); stakeErr != nil {
			return errorsmod.Wrapf(
				stakeErr,
				"miner %s must be registered as a node or have sufficient stake/balance to claim tasks. "+
					"Register with RegisterNode message or ensure sufficient balance",
				miner,
			)
		}
		// Has sufficient balance, allow but log warning
		sdkCtx.Logger().Info(fmt.Sprintf(
			"WARNING: Miner %s claiming task without node registration. "+
				"Registration is recommended for better tracking and resource management",
			miner,
		))
	}

	// Get pool
	pool, err := k.GetTaskPool(ctx, poolID)
	if err != nil {
		return err
	}

	// Check if pool is active
	if pool.Status != "active" {
		return errorsmod.Wrapf(types.ErrInvalidRequest, "task pool %d is not active (status: %s)", poolID, pool.Status)
	}

	// Check if pool has expired
	if sdkCtx.BlockHeight() >= pool.ExpiryHeight {
		pool.Status = "closed"
		if err := k.TaskPools.Set(ctx, poolID, pool); err != nil {
			return errorsmod.Wrap(err, "failed to update pool status")
		}
		return errorsmod.Wrapf(types.ErrInvalidRequest, "task pool %d has expired", poolID)
	}

	// Find and claim chunk
	found := false
	for i := range pool.AvailableChunks {
		if pool.AvailableChunks[i].ChunkId == chunkID {
			if !pool.AvailableChunks[i].IsAvailable() {
				return errorsmod.Wrapf(types.ErrInvalidRequest, "chunk %d is not available (status: %s)", chunkID, pool.AvailableChunks[i].Status)
			}

			// Claim chunk
			pool.AvailableChunks[i].Status = "in_progress"
			pool.AvailableChunks[i].AssignedMiner = miner
			pool.AvailableChunks[i].ClaimedAt = sdkCtx.BlockTime().Unix()
			found = true
			break
		}
	}

	if !found {
		return errorsmod.Wrapf(types.ErrInvalidRequest, "chunk %d not found in pool %d", chunkID, poolID)
	}

	// Update pool
	if err := k.TaskPools.Set(ctx, poolID, pool); err != nil {
		return errorsmod.Wrap(err, "failed to update task pool")
	}

	return nil
}

// CompleteTask marks a chunk as completed and records the gradient hash
// If chunk is a trap, performs tolerant verification
// If chunk is a real job, adds to vault (proof of reuse)
func (k Keeper) CompleteTask(ctx context.Context, miner string, poolID uint64, chunkID uint64, gradientHash string, gradientIPFSHash string, minerGPU string) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get pool
	pool, err := k.GetTaskPool(ctx, poolID)
	if err != nil {
		return err
	}

	// Find chunk
	var chunk *types.TaskChunk
	found := false
	for i := range pool.AvailableChunks {
		if pool.AvailableChunks[i].ChunkId == chunkID && pool.AvailableChunks[i].AssignedMiner == miner {
			if pool.AvailableChunks[i].Status != "in_progress" {
				return errorsmod.Wrapf(types.ErrInvalidRequest, "chunk %d is not in progress (status: %s)", chunkID, pool.AvailableChunks[i].Status)
			}
			chunk = &pool.AvailableChunks[i]
			found = true
			break
		}
	}

	if !found {
		return errorsmod.Wrapf(types.ErrInvalidRequest, "chunk %d not found or not assigned to miner %s", chunkID, miner)
	}

	// If trap, perform tolerant verification
	if chunk.IsTrap {
		vaultEntryID := chunk.GetVaultEntryID()
		vaultEntry, err := k.GetVaultEntry(ctx, vaultEntryID)
		if err != nil {
			return errorsmod.Wrapf(err, "failed to get vault entry %d for trap verification", vaultEntryID)
		}

		// Perform tolerant verification
		result, err := k.VerifyGradientTolerant(
			ctx,
			gradientHash,
			gradientIPFSHash,
			vaultEntry.ExpectedGradientHash,
			vaultEntry.ExpectedFingerprint,
			minerGPU,
			vaultEntry.GpuArchitecture,
		)
		if err != nil {
			return errorsmod.Wrap(err, "failed to verify trap gradient")
		}

		if !result.IsValid {
			// Fraud detected - update fraud score
			if err := k.DetectFraud(ctx, miner, result); err != nil {
				sdkCtx.Logger().Error(fmt.Sprintf("Failed to update fraud score: %v", err))
			}
			sdkCtx.Logger().Error(fmt.Sprintf("Trap verification failed for miner %s, chunk %d: %s (similarity: %.6f)", miner, chunkID, result.Reason, result.SimilarityScore))
			return errorsmod.Wrapf(types.ErrInvalidRequest, "trap verification failed: %s (similarity: %.6f)", result.Reason, result.SimilarityScore)
		}

		// Trap passed - update fraud score (positive signal)
		if err := k.DetectFraud(ctx, miner, result); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to update fraud score: %v", err))
		}
		sdkCtx.Logger().Info(fmt.Sprintf("Trap verification passed for miner %s, chunk %d (similarity: %.6f)", miner, chunkID, result.SimilarityScore))
	}

	// Complete chunk
	chunk.Status = "completed"
	chunk.GradientHash = gradientHash
	chunk.CompletedAt = sdkCtx.BlockTime().Unix()

	// Add to completed chunks list
	pool.CompletedChunks = append(pool.CompletedChunks, gradientHash)

	// Update pool
	if err := k.TaskPools.Set(ctx, poolID, pool); err != nil {
		return errorsmod.Wrap(err, "failed to update task pool")
	}

	// If real job (not trap), add to vault (proof of reuse)
	if !chunk.IsTrap {
		if err := k.addToVaultIfValid(ctx, *chunk, gradientHash, gradientIPFSHash, minerGPU); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to add real job to vault: %v", err))
			// Don't fail the task completion if vault addition fails
		}
	}

	// Distribute reward per completed task (not per block)
	if err := k.DistributeTaskReward(ctx, miner, chunkID); err != nil {
		return errorsmod.Wrap(err, "failed to distribute task reward")
	}

	// Debug logging - completion
	if debugLogger := k.GetDebugLogger(); debugLogger != nil {
		debugLogger.Info(sdkCtx, "Task completed successfully",
			"miner", miner,
			"pool_id", poolID,
			"chunk_id", chunkID,
			"is_trap", chunk.IsTrap,
		)
	}

	return nil
}

// addToVaultIfValid adds a verified real job to the genesis vault (proof of reuse)
func (k Keeper) addToVaultIfValid(ctx context.Context, chunk types.TaskChunk, gradientHash string, gradientIPFSHash string, minerGPU string) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Verify this is a real job (not a trap)
	if chunk.IsTrap {
		return fmt.Errorf("cannot add trap to vault")
	}

	// Retrieve gradient tensor from IPFS
	fullGradient, err := k.RetrieveGradientTensor(ctx, gradientIPFSHash)
	if err != nil {
		return fmt.Errorf("failed to retrieve gradient for vault: %w", err)
	}

	// Extract Top-K fingerprint (indices + values REQUIRED)
	fingerprint, err := ExtractTopKFingerprint(fullGradient, 100)
	if err != nil {
		return fmt.Errorf("failed to extract fingerprint: %w", err)
	}

	// Serialize fingerprint to JSON
	fingerprintJSON, err := SerializeFingerprint(fingerprint)
	if err != nil {
		return fmt.Errorf("failed to serialize fingerprint: %w", err)
	}

	// Get next vault entry ID
	entryID, err := k.GenesisVaultCounter.Next(ctx)
	if err != nil {
		return fmt.Errorf("failed to get next vault entry ID: %w", err)
	}

	// Create vault entry
	vaultEntry := types.GenesisVaultEntry{
		EntryId:                  entryID,
		DataHash:                 chunk.DataHash,
		ExpectedGradientHash:     gradientHash,
		ExpectedGradientIpfsHash: gradientIPFSHash,
		ExpectedFingerprint:      fingerprintJSON,
		GpuArchitecture:          minerGPU,
		CreatedHeight:            sdkCtx.BlockHeight(),
		UsageCount:               0,
		LastUsedHeight:           sdkCtx.BlockHeight(),
		Encrypted:                false,
	}

	// Add to vault
	if err := k.AddToVault(ctx, vaultEntry); err != nil {
		return fmt.Errorf("failed to add entry to vault: %w", err)
	}

	sdkCtx.Logger().Info(fmt.Sprintf("Added real job to vault: entry_id=%d, data_hash=%s", entryID, chunk.DataHash))
	return nil
}

// GetAvailableChunks returns available chunks from a task pool with blind delivery
// Mixes 90% real jobs with 10% traps from genesis vault, then shuffles randomly
// Returns internal TaskChunk objects (for keeper use)
func (k Keeper) GetAvailableChunks(ctx context.Context, poolID uint64, limit uint64) ([]types.TaskChunk, error) {
	pool, err := k.GetTaskPool(ctx, poolID)
	if err != nil {
		return nil, err
	}

	// Get real jobs (90% of limit)
	realJobCount := uint64(float64(limit) * 0.9)
	realJobs := make([]types.TaskChunk, 0, realJobCount)
	for _, chunk := range pool.AvailableChunks {
		if chunk.IsAvailable() && !chunk.IsTrap {
			realJobs = append(realJobs, chunk)
			if uint64(len(realJobs)) >= realJobCount {
				break
			}
		}
	}

	// Get traps (10% of limit) from genesis vault
	trapCount := limit - uint64(len(realJobs))
	trapChunks, err := k.selectRandomTraps(ctx, trapCount)
	if err != nil {
		sdk.UnwrapSDKContext(ctx).Logger().Error(fmt.Sprintf("Failed to select traps: %v", err))
		// Continue without traps if vault is empty
		trapChunks = []types.TaskChunk{}
	}

	// Combine real jobs and traps
	allChunks := append(realJobs, trapChunks...)

	// Shuffle using Fisher-Yates algorithm
	shuffledChunks := shuffleChunks(allChunks)

	return shuffledChunks, nil
}

// GetAvailableChunksForMiner returns sanitized chunks for miners (trap flags removed)
// This is the method that should be called from query handlers
func (k Keeper) GetAvailableChunksForMiner(ctx context.Context, poolID uint64, limit uint64) ([]types.TaskChunkResponse, error) {
	// Get internal chunks (may include traps)
	chunks, err := k.GetAvailableChunks(ctx, poolID, limit)
	if err != nil {
		return nil, err
	}

	// Sanitize for miner (remove is_trap and vault_entry_id)
	sanitized := make([]types.TaskChunkResponse, 0, len(chunks))
	for i := range chunks {
		response := chunks[i].ToMinerResponse()
		sanitized = append(sanitized, *response)
	}

	return sanitized, nil
}

// selectRandomTraps selects random trap entries from genesis vault and converts them to TaskChunks
func (k Keeper) selectRandomTraps(ctx context.Context, count uint64) ([]types.TaskChunk, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Select random trap entries from vault
	vaultEntries, err := k.SelectRandomTraps(ctx, count)
	if err != nil {
		return nil, err
	}

	// Convert vault entries to TaskChunks
	trapChunks := make([]types.TaskChunk, 0, len(vaultEntries))
	for _, entry := range vaultEntries {
		// Generate a unique chunk ID for trap (use negative pool ID * 10000 + entry_id as unique identifier)
		// In practice, this would be handled more carefully
		trapChunkID := entry.EntryId + 1000000 // Offset to avoid collisions with real chunks

		chunk := types.TaskChunk{
			ChunkId:       trapChunkID,
			DataHash:      entry.DataHash,
			ShardId:       0, // Traps don't have shard assignment
			AssignedMiner: "",
			Status:        "available",
			ClaimedAt:     0,
			CompletedAt:   0,
			GradientHash:  "",
			IsTrap:        true,          // Mark as trap (internal only)
			VaultEntryId:  entry.EntryId, // Reference to vault entry
		}
		trapChunks = append(trapChunks, chunk)
	}

	// Increment usage tracking for all selected entries
	for _, entry := range vaultEntries {
		entry.IncrementUsage(sdkCtx.BlockHeight())
		if err := k.GenesisVault.Set(ctx, entry.EntryId, entry); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to update vault entry usage for entry %d: %v", entry.EntryId, err))
		}
	}

	return trapChunks, nil
}

// shuffleChunks shuffles chunks using Fisher-Yates algorithm
func shuffleChunks(chunks []types.TaskChunk) []types.TaskChunk {
	shuffled := make([]types.TaskChunk, len(chunks))
	copy(shuffled, chunks)

	rng := rand.New(rand.NewSource(rand.Int63()))
	for i := len(shuffled) - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	}

	return shuffled
}

// GenerateChunkDataHash generates a deterministic hash for chunk data
func (k Keeper) GenerateChunkDataHash(windowID uint64, chunkID uint64) string {
	// Deterministic hash generation based on window ID and chunk ID
	// In production, this would use actual dataset shard data
	hashInput := fmt.Sprintf("window_%d_chunk_%d", windowID, chunkID)
	return fmt.Sprintf("Qm%s", hashInput[:44]) // Simulated IPFS hash format
}

// DistributeTaskReward distributes reward for a completed task
func (k Keeper) DistributeTaskReward(ctx context.Context, miner string, chunkID uint64) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Calculate reward per task (simplified - in production would use quality multiplier)
	// Default: 0.1 REMES per completed task
	baseReward := sdk.NewCoin("uremes", math.NewInt(100000)) // 0.1 REMES (6 decimals)

	// Mint and distribute tokens
	minerAddr, err := k.addressCodec.StringToBytes(miner)
	if err != nil {
		return errorsmod.Wrapf(err, "invalid miner address: %s", miner)
	}

	if err := k.bankKeeper.MintCoins(sdkCtx, types.ModuleName, sdk.NewCoins(baseReward)); err != nil {
		return errorsmod.Wrap(err, "failed to mint coins")
	}

	if err := k.bankKeeper.SendCoinsFromModuleToAccount(sdkCtx, types.ModuleName, minerAddr, sdk.NewCoins(baseReward)); err != nil {
		return errorsmod.Wrap(err, "failed to send coins to miner")
	}

	return nil
}
