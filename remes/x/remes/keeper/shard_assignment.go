package keeper

import (
	"crypto/sha256"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
)

// CalculateShardID calculates a deterministic shard ID for a miner in a training round.
// Formula: (wallet_address + block_hash + round_id) % total_shards
// This ensures:
// - Same miner gets same shard within a round (stable)
// - Different miners get different shards (distributed)
// - Different rounds can have different assignments (flexible)
func (k Keeper) CalculateShardID(
	ctx sdk.Context,
	minerAddress string,
	trainingRoundID uint64,
	totalShards uint64,
) (uint64, error) {
	if totalShards == 0 {
		return 0, fmt.Errorf("total_shards cannot be zero")
	}

	// Get block hash (or block height as fallback)
	blockHash := ctx.HeaderHash()
	if len(blockHash) == 0 {
		// Fallback: use block height as deterministic seed
		blockHash = []byte(fmt.Sprintf("block_%d", ctx.BlockHeight()))
	}

	// Create deterministic input: miner_address + block_hash + round_id
	input := fmt.Sprintf("%s|%x|%d", minerAddress, blockHash, trainingRoundID)

	// Hash the input
	hash := sha256.Sum256([]byte(input))

	// Convert first 8 bytes to uint64
	var shardID uint64
	for i := 0; i < 8 && i < len(hash); i++ {
		shardID = shardID<<8 | uint64(hash[i])
	}

	// Modulo to get shard ID in range [0, totalShards)
	shardID = shardID % totalShards

	return shardID, nil
}

// VerifyShardAssignment verifies that a miner's shard assignment is correct.
func (k Keeper) VerifyShardAssignment(
	ctx sdk.Context,
	minerAddress string,
	claimedShardID uint64,
	trainingRoundID uint64,
	totalShards uint64,
) (bool, error) {
	expectedShardID, err := k.CalculateShardID(ctx, minerAddress, trainingRoundID, totalShards)
	if err != nil {
		return false, err
	}

	return claimedShardID == expectedShardID, nil
}

