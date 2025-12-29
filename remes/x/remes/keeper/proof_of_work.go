package keeper

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// Proof-of-work parameters
const (
	// ProofOfWorkDifficulty is the number of leading zeros required in hash
	// Higher difficulty = more computation required
	ProofOfWorkDifficulty = 4 // 4 leading zeros = ~16 attempts on average
	
	// ProofOfWorkMaxAttempts is the maximum number of attempts allowed
	ProofOfWorkMaxAttempts = 10000
)

// VerifyProofOfWork verifies that a proof-of-work is valid
// PoW: Find nonce such that SHA256(message + nonce) has N leading zeros
func (k Keeper) VerifyProofOfWork(
	ctx sdk.Context,
	messageHash []byte,
	nonce uint64,
	difficulty uint8,
) error {
	// Create proof-of-work input: message_hash + nonce
	nonceBytes := make([]byte, 8)
	binary.BigEndian.PutUint64(nonceBytes, nonce)
	
	powInput := append(messageHash, nonceBytes...)
	
	// Compute hash
	hash := sha256.Sum256(powInput)
	
	// Check leading zeros
	requiredZeros := int(difficulty)
	zeroBytes := requiredZeros / 8
	zeroBits := requiredZeros % 8
	
	// Check full zero bytes
	for i := 0; i < zeroBytes; i++ {
		if hash[i] != 0 {
			return errorsmod.Wrapf(types.ErrInvalidSignature, "proof-of-work invalid: insufficient leading zeros")
		}
	}
	
	// Check partial zero byte
	if zeroBits > 0 {
		mask := byte(0xFF) << (8 - zeroBits)
		if hash[zeroBytes]&mask != 0 {
			return errorsmod.Wrapf(types.ErrInvalidSignature, "proof-of-work invalid: insufficient leading zeros")
		}
	}
	
	return nil
}

// CalculateProofOfWork calculates proof-of-work for a message
// This is used by Python miner to find a valid nonce
func CalculateProofOfWork(messageHash []byte, difficulty uint8) (uint64, error) {
	nonceBytes := make([]byte, 8)
	
	for nonce := uint64(0); nonce < ProofOfWorkMaxAttempts; nonce++ {
		binary.BigEndian.PutUint64(nonceBytes, nonce)
		
		powInput := append(messageHash, nonceBytes...)
		hash := sha256.Sum256(powInput)
		
		// Check leading zeros
		requiredZeros := int(difficulty)
		zeroBytes := requiredZeros / 8
		zeroBits := requiredZeros % 8
		
		valid := true
		
		// Check full zero bytes
		for i := 0; i < zeroBytes; i++ {
			if hash[i] != 0 {
				valid = false
				break
			}
		}
		
		if !valid {
			continue
		}
		
		// Check partial zero byte
		if zeroBits > 0 {
			mask := byte(0xFF) << (8 - zeroBits)
			if hash[zeroBytes]&mask != 0 {
				valid = false
			}
		}
		
		if valid {
			return nonce, nil
		}
	}
	
	return 0, fmt.Errorf("proof-of-work not found within %d attempts", ProofOfWorkMaxAttempts)
}

