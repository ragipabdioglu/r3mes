package keeper

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"os"

	errorsmod "cosmossdk.io/errors"
	"remes/x/remes/types"
)

// VaultEncryptionKey is the environment variable name for vault encryption key
const VaultEncryptionKeyEnv = "R3MES_VAULT_ENCRYPTION_KEY"

// getVaultEncryptionKey retrieves the vault encryption key from environment variable
// Returns 32-byte key for AES-256
func getVaultEncryptionKey() ([]byte, error) {
	keyHex := os.Getenv(VaultEncryptionKeyEnv)
	if keyHex == "" {
		return nil, fmt.Errorf("vault encryption key not set (env: %s)", VaultEncryptionKeyEnv)
	}

	key, err := hex.DecodeString(keyHex)
	if err != nil {
		return nil, fmt.Errorf("invalid encryption key format: %w", err)
	}

	if len(key) != 32 {
		return nil, fmt.Errorf("encryption key must be 32 bytes (AES-256), got %d bytes", len(key))
	}

	return key, nil
}

// EncryptVaultEntry encrypts a vault entry's sensitive fields
// Uses AES-256-GCM for authenticated encryption
func (k Keeper) EncryptVaultEntry(entry types.GenesisVaultEntry) (types.GenesisVaultEntry, error) {
	// If already encrypted or encryption disabled, return as-is
	if entry.Encrypted {
		return entry, nil
	}

	// Get encryption key
	key, err := getVaultEncryptionKey()
	if err != nil {
		// If key not set, return entry without encryption (optional feature)
		return entry, nil
	}

	// Create AES cipher
	block, err := aes.NewCipher(key)
	if err != nil {
		return types.GenesisVaultEntry{}, errorsmod.Wrap(err, "failed to create AES cipher")
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return types.GenesisVaultEntry{}, errorsmod.Wrap(err, "failed to create GCM mode")
	}

	// Encrypt expected fingerprint (most sensitive field)
	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return types.GenesisVaultEntry{}, errorsmod.Wrap(err, "failed to generate nonce")
	}

	plaintext := []byte(entry.ExpectedFingerprint)
	ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)

	// Store encrypted data as hex string
	entry.ExpectedFingerprint = hex.EncodeToString(ciphertext)
	entry.Encrypted = true

	return entry, nil
}

// DecryptVaultEntry decrypts a vault entry's sensitive fields
func (k Keeper) DecryptVaultEntry(entry types.GenesisVaultEntry) (types.GenesisVaultEntry, error) {
	// If not encrypted, return as-is
	if !entry.Encrypted {
		return entry, nil
	}

	// Get encryption key
	key, err := getVaultEncryptionKey()
	if err != nil {
		return types.GenesisVaultEntry{}, errorsmod.Wrap(err, "encryption key not available for decryption")
	}

	// Create AES cipher
	block, err := aes.NewCipher(key)
	if err != nil {
		return types.GenesisVaultEntry{}, errorsmod.Wrap(err, "failed to create AES cipher")
	}

	// Create GCM mode
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return types.GenesisVaultEntry{}, errorsmod.Wrap(err, "failed to create GCM mode")
	}

	// Decode hex string
	ciphertext, err := hex.DecodeString(entry.ExpectedFingerprint)
	if err != nil {
		return types.GenesisVaultEntry{}, errorsmod.Wrap(err, "failed to decode encrypted fingerprint")
	}

	// Extract nonce (first gcm.NonceSize() bytes)
	nonceSize := gcm.NonceSize()
	if len(ciphertext) < nonceSize {
		return types.GenesisVaultEntry{}, fmt.Errorf("ciphertext too short")
	}

	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]

	// Decrypt
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return types.GenesisVaultEntry{}, errorsmod.Wrap(err, "failed to decrypt fingerprint")
	}

	entry.ExpectedFingerprint = string(plaintext)
	entry.Encrypted = false

	return entry, nil
}

