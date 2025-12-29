package types

import (
	"fmt"
)

// TaskChunkResponse is defined in task_pool.pb.go (generated from proto)
// This file only contains helper methods for TaskChunk and TaskPool

// Validate validates the task pool structure
func (tp *TaskPool) Validate() error {
	if tp.PoolId == 0 {
		return fmt.Errorf("pool_id cannot be zero")
	}
	if tp.TotalChunks == 0 {
		return fmt.Errorf("total_chunks cannot be zero")
	}
	if tp.Status != "active" && tp.Status != "aggregating" && tp.Status != "closed" {
		return fmt.Errorf("invalid status: %s", tp.Status)
	}
	if tp.ExpiryHeight <= tp.CreatedHeight {
		return fmt.Errorf("expiry_height must be greater than created_height")
	}
	return nil
}

// Validate validates the task chunk structure
func (tc *TaskChunk) Validate() error {
	if tc.DataHash == "" {
		return fmt.Errorf("data_hash cannot be empty")
	}
	if tc.Status != "available" && tc.Status != "in_progress" && tc.Status != "completed" {
		return fmt.Errorf("invalid status: %s", tc.Status)
	}
	return nil
}

// IsAvailable checks if the chunk is available for claiming
func (tc *TaskChunk) IsAvailable() bool {
	return tc.Status == "available"
}

// IsCompleted checks if the chunk is completed
func (tc *TaskChunk) IsCompleted() bool {
	return tc.Status == "completed"
}

// IsTrapChunk checks if this chunk is a trap job (internal use only)
func (tc *TaskChunk) IsTrapChunk() bool {
	return tc.IsTrap
}

// GetVaultEntryID returns the vault entry ID if this is a trap (internal use only)
func (tc *TaskChunk) GetVaultEntryID() uint64 {
	return tc.VaultEntryId
}

// ToMinerResponse converts TaskChunk to TaskChunkResponse by removing sensitive fields
func (tc *TaskChunk) ToMinerResponse() *TaskChunkResponse {
	return &TaskChunkResponse{
		ChunkId:  tc.ChunkId,
		DataHash: tc.DataHash,
		ShardId:  tc.ShardId,
	}
}
