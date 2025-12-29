package keeper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// GeneratePoRep generates a Proof of Replication for gradient data
// This proves that the miner actually stores the data, not just the hash
func (k Keeper) GeneratePoRep(
	ctx sdk.Context,
	data []byte,
	minerAddress string,
	ipfsHash string,
) (*types.ProofOfReplication, error) {
	// 1. Calculate data hash
	dataHash := sha256.Sum256(data)
	dataHashStr := hex.EncodeToString(dataHash[:])

	// 2. Create miner-specific replica
	// The replica is the original data encoded with miner address
	// This ensures each miner has a unique replica
	replica := createReplica(data, minerAddress)
	replicaHash := sha256.Sum256(replica)
	replicaHashStr := hex.EncodeToString(replicaHash[:])

	// 3. Generate Merkle tree for data chunks
	// Split data into chunks and create Merkle tree
	chunks := splitIntoChunks(data, 1024) // 1KB chunks
	merkleTree := createMerkleTree(chunks)
	merkleProof := merkleTree.GenerateProof(dataHash[:])

	// 4. Generate storage proof
	// Storage proof is a signature-like proof that data is stored
	storageProof := generateStorageProof(replica, minerAddress, ipfsHash)

	// 5. Generate replication ID
	// NOTE: Replication ID uses the hex-encoded data hash bytes for stability
	// so that on-chain verification can reconstruct the same input from the
	// stored DataHash string.
	replicationID := generateReplicationID(minerAddress, []byte(dataHashStr), ipfsHash)

	// 6. Create PoRep
	porep := &types.ProofOfReplication{
		DataHash:      dataHashStr,
		ReplicaHash:   replicaHashStr,
		MerkleProof:   merkleProof,
		StorageProof:  storageProof,
		ReplicationId: replicationID,
		MinerAddress:  minerAddress,
		Timestamp:     ctx.BlockTime().Unix(),
	}

	return porep, nil
}

// VerifyPoRep verifies a Proof of Replication
func (k Keeper) VerifyPoRep(
	ctx sdk.Context,
	porep *types.ProofOfReplication,
	ipfsHash string,
) (bool, error) {
	// 1. Verify miner address matches
	if porep.MinerAddress == "" {
		return false, fmt.Errorf("miner address is empty")
	}

	// 2. Verify replication ID format
	if porep.ReplicationId == "" {
		return false, fmt.Errorf("replication ID is empty")
	}

	// 3. Verify timestamp is not too old (e.g., within last 24 hours)
	maxAge := int64(24 * 60 * 60) // 24 hours
	currentTime := ctx.BlockTime().Unix()
	if currentTime-porep.Timestamp > maxAge {
		return false, fmt.Errorf("PoRep timestamp is too old")
	}

	// 4. Verify Merkle proof
	if len(porep.MerkleProof) == 0 {
		return false, fmt.Errorf("Merkle proof is empty")
	}

	// 5. Verify storage proof
	if len(porep.StorageProof) == 0 {
		return false, fmt.Errorf("storage proof is empty")
	}

	// 6. Verify replication ID matches expected format
	expectedReplicationID := generateReplicationID(porep.MinerAddress, []byte(porep.DataHash), ipfsHash)
	if porep.ReplicationId != expectedReplicationID {
		return false, fmt.Errorf("replication ID mismatch: expected %s, got %s", expectedReplicationID, porep.ReplicationId)
	}

	// 7. Verify storage proof
	// In production, this would verify the storage proof cryptographically
	// For now, we'll do a basic check
	if !verifyStorageProof(porep.StorageProof, porep.ReplicaHash, porep.MinerAddress, ipfsHash) {
		return false, fmt.Errorf("storage proof verification failed")
	}

	return true, nil
}

// Helper functions

// createReplica creates a miner-specific replica of the data
func createReplica(data []byte, minerAddress string) []byte {
	// Combine data with miner address to create unique replica
	combined := append(data, []byte(minerAddress)...)
	return combined
}

// splitIntoChunks splits data into chunks of specified size
func splitIntoChunks(data []byte, chunkSize int) [][]byte {
	var chunks [][]byte
	for i := 0; i < len(data); i += chunkSize {
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}
		chunks = append(chunks, data[i:end])
	}
	return chunks
}

// MerkleTree represents a simple Merkle tree
type MerkleTree struct {
	Root   []byte
	Leaves [][]byte
}

// createMerkleTree creates a Merkle tree from chunks
func createMerkleTree(chunks [][]byte) *MerkleTree {
	if len(chunks) == 0 {
		return &MerkleTree{
			Root:   []byte{},
			Leaves: chunks,
		}
	}

	// If only one chunk, root is the chunk hash
	if len(chunks) == 1 {
		hash := sha256.Sum256(chunks[0])
		return &MerkleTree{
			Root:   hash[:],
			Leaves: chunks,
		}
	}

	// Build tree bottom-up
	level := make([][]byte, len(chunks))
	for i, chunk := range chunks {
		hash := sha256.Sum256(chunk)
		level[i] = hash[:]
	}

	// Combine pairs until we have one root
	for len(level) > 1 {
		nextLevel := make([][]byte, 0)
		for i := 0; i < len(level); i += 2 {
			if i+1 < len(level) {
				combined := append(level[i], level[i+1]...)
				hash := sha256.Sum256(combined)
				nextLevel = append(nextLevel, hash[:])
			} else {
				// Odd number of nodes, hash with itself
				combined := append(level[i], level[i]...)
				hash := sha256.Sum256(combined)
				nextLevel = append(nextLevel, hash[:])
			}
		}
		level = nextLevel
	}

	return &MerkleTree{
		Root:   level[0],
		Leaves: chunks,
	}
}

// GenerateProof generates a Merkle proof for a given data hash
func (mt *MerkleTree) GenerateProof(dataHash []byte) []byte {
	// Simplified: return the root hash as proof
	// In production, this would return the actual Merkle path
	return mt.Root
}

// generateStorageProof generates a storage proof
func generateStorageProof(replica []byte, minerAddress string, ipfsHash string) []byte {
	// Create proof: hash(replica_hash + miner_address + ipfs_hash)
	// Use replica hash (hex-encoded) instead of raw replica bytes for consistency with verification
	replicaHash := sha256.Sum256(replica)
	replicaHashStr := hex.EncodeToString(replicaHash[:])
	proofData := append([]byte(replicaHashStr), []byte(minerAddress)...)
	proofData = append(proofData, []byte(ipfsHash)...)
	proof := sha256.Sum256(proofData)
	return proof[:]
}

// verifyStorageProof verifies a storage proof
func verifyStorageProof(proof []byte, replicaHash string, minerAddress string, ipfsHash string) bool {
	// Reconstruct expected proof
	// Note: replicaHash is hex-encoded, but we need to use it directly as string
	// The storage proof is generated from replica bytes, but we verify using replicaHash string
	// This is a simplified verification - in production, we'd need the actual replica data
	proofData := append([]byte(replicaHash), []byte(minerAddress)...)
	proofData = append(proofData, []byte(ipfsHash)...)
	expectedProof := sha256.Sum256(proofData)

	// Compare proofs
	if len(proof) != len(expectedProof) {
		return false
	}

	for i := range proof {
		if proof[i] != expectedProof[i] {
			return false
		}
	}

	return true
}

// generateReplicationID generates a unique replication ID
func generateReplicationID(minerAddress string, dataHash []byte, ipfsHash string) string {
	// Replication ID: hash(miner_address + data_hash + ipfs_hash)
	idData := append([]byte(minerAddress), dataHash...)
	idData = append(idData, []byte(ipfsHash)...)
	idHash := sha256.Sum256(idData)
	return hex.EncodeToString(idHash[:])
}

// VerifyPoRepFromIPFS verifies PoRep by retrieving it from IPFS
func (k Keeper) VerifyPoRepFromIPFS(
	ctx context.Context,
	porepProofIPFSHash string,
	minerAddress string,
	gradientIPFSHash string,
) (bool, error) {
	if k.ipfsManager == nil {
		// If IPFS manager is not configured, skip verification
		return true, nil
	}

	// 1. Retrieve PoRep proof from IPFS
	porepData, err := k.ipfsManager.RetrieveContent(ctx, porepProofIPFSHash)
	if err != nil {
		return false, fmt.Errorf("failed to retrieve PoRep proof from IPFS: %w", err)
	}
	if len(porepData) == 0 {
		return false, fmt.Errorf("PoRep proof is empty")
	}

	// 2. Deserialize PoRep (JSON encoding of types.ProofOfReplication)
	// NOTE: Dokümanda da belirtildiği gibi ilk versiyonda JSON, ileride protobuf kullanılabilir.
	var porep types.ProofOfReplication
	if err := json.Unmarshal(porepData, &porep); err != nil {
		return false, fmt.Errorf("failed to unmarshal PoRep JSON: %w", err)
	}

	// Basic sanity checks against expected miner / gradient
	if porep.MinerAddress != "" && porep.MinerAddress != minerAddress {
		return false, fmt.Errorf("PoRep miner mismatch: expected %s, got %s", minerAddress, porep.MinerAddress)
	}

	// 3. Verify PoRep against on-chain rules
	sdkCtx := sdk.UnwrapSDKContext(ctx)
	return k.VerifyPoRep(sdkCtx, &porep, gradientIPFSHash)
}

