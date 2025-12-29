package keeper

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

// CalculateMerkleRoot calculates the Merkle root hash from a list of gradient hashes.
// This matches the Python implementation in miner-engine/core/merkle.py
// Algorithm:
// 1. Start with leaves (gradient hashes)
// 2. Build tree level by level, pairing nodes
// 3. For odd number of nodes, duplicate the last node
// 4. Root is the final single hash
func (k Keeper) CalculateMerkleRoot(gradientHashes []string) (string, error) {
	if len(gradientHashes) == 0 {
		return "", fmt.Errorf("cannot create Merkle tree with empty leaves")
	}

	// Start with leaves
	currentLevel := make([]string, len(gradientHashes))
	copy(currentLevel, gradientHashes)

	// Build tree level by level
	for len(currentLevel) > 1 {
		nextLevel := make([]string, 0, (len(currentLevel)+1)/2)

		for i := 0; i < len(currentLevel); i += 2 {
			if i+1 < len(currentLevel) {
				// Pair two nodes
				combined := hashPair(currentLevel[i], currentLevel[i+1])
				nextLevel = append(nextLevel, combined)
			} else {
				// Odd number, duplicate last node
				combined := hashPair(currentLevel[i], currentLevel[i])
				nextLevel = append(nextLevel, combined)
			}
		}

		currentLevel = nextLevel
	}

	// Root is the final single hash
	return currentLevel[0], nil
}

// hashPair hashes a pair of nodes.
// Matches Python: hashlib.sha256(f"{left}:{right}".encode()).hexdigest()
func hashPair(left, right string) string {
	input := fmt.Sprintf("%s:%s", left, right)
	hash := sha256.Sum256([]byte(input))
	return hex.EncodeToString(hash[:])
}

// verifyMerkleRoot verifies that a Merkle root matches the calculated root from gradient hashes.
func (k Keeper) verifyMerkleRoot(gradientHashes []string, claimedRoot string) (bool, error) {
	if claimedRoot == "" {
		return false, fmt.Errorf("claimed Merkle root cannot be empty")
	}

	calculatedRoot, err := k.CalculateMerkleRoot(gradientHashes)
	if err != nil {
		return false, err
	}

	return calculatedRoot == claimedRoot, nil
}
