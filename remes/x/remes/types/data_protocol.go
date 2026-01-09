package types

import (
	"fmt"

	errorsmod "cosmossdk.io/errors"
)

// Fixed Chunk / Variable Speed Protocol Constants
const (
	// FIXED_CHUNK_SIZE_TOKENS is the protocol-mandated chunk size (never changes)
	// All miners receive chunks of exactly this size
	FIXED_CHUNK_SIZE_TOKENS = 2048
)

// ChunkData represents a fixed-size data chunk in the protocol
type ChunkData struct {
	ChunkID    uint64 `json:"chunk_id"`
	DataHash   string `json:"data_hash"`   // IPFS hash of chunk data
	TokenCount uint64 `json:"token_count"` // Must be exactly 2048
	ShardID    uint64 `json:"shard_id"`
	WindowID   uint64 `json:"window_id"`
}

// ValidateChunkSize validates that a chunk matches the protocol-mandated size
func ValidateChunkSize(chunk ChunkData) error {
	if chunk.TokenCount != FIXED_CHUNK_SIZE_TOKENS {
		return errorsmod.Wrapf(
			ErrInvalidChunkSize,
			"chunk must be exactly %d tokens, got %d",
			FIXED_CHUNK_SIZE_TOKENS,
			chunk.TokenCount,
		)
	}
	return nil
}

// ValidateChunkData validates the entire chunk data structure
func (cd *ChunkData) Validate() error {
	if cd.DataHash == "" {
		return fmt.Errorf("data_hash cannot be empty")
	}
	if err := ValidateChunkSize(*cd); err != nil {
		return err
	}
	return nil
}
