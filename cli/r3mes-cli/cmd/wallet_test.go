package cmd

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
)

func TestGenerateCosmosAddress(t *testing.T) {
	testCases := []struct {
		name       string
		privateKey string
		wantPrefix string
	}{
		{
			name:       "valid private key 1",
			privateKey: "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
			wantPrefix: "remes1",
		},
		{
			name:       "valid private key 2",
			privateKey: "abcdef1234567890abcdef1234567890abcdef1234567890abcdef12345678",
			wantPrefix: "remes1",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			keyBytes, err := hex.DecodeString(tc.privateKey)
			if err != nil {
				t.Fatalf("Failed to decode private key: %v", err)
			}

			address, err := GenerateCosmosAddress(keyBytes)
			if err != nil {
				t.Fatalf("Failed to generate address: %v", err)
			}

			if len(address) < len(tc.wantPrefix) || address[:len(tc.wantPrefix)] != tc.wantPrefix {
				t.Errorf("Address should start with %s, got: %s", tc.wantPrefix, address)
			}
		})
	}
}

func TestGenerateCosmosAddressInvalid(t *testing.T) {
	// Empty key should fail
	_, err := GenerateCosmosAddress([]byte{})
	if err == nil {
		t.Error("Expected error for empty private key")
	}
}

func TestSaveAndLoadWallet(t *testing.T) {
	tempDir := t.TempDir()
	config := &Config{
		WalletPath: tempDir,
	}

	wallet := Wallet{
		Address:   "remes1testaddress123456789",
		CreatedAt: "2026-01-02T00:00:00Z",
		Salt:      "testsalt",
	}

	// Save wallet
	err := SaveWallet(wallet, "testuser", config)
	if err != nil {
		t.Fatalf("Failed to save wallet: %v", err)
	}

	// Verify file exists
	walletPath := filepath.Join(tempDir, "testuser.json")
	if _, err := os.Stat(walletPath); os.IsNotExist(err) {
		t.Error("Wallet file was not created")
	}

	// Load wallet
	config.WalletPath = tempDir
	// Change default wallet name for test
	err = SaveWallet(wallet, "default", config)
	if err != nil {
		t.Fatalf("Failed to save default wallet: %v", err)
	}

	loadedWallet, err := LoadDefaultWallet(config)
	if err != nil {
		t.Fatalf("Failed to load wallet: %v", err)
	}

	if loadedWallet.Address != wallet.Address {
		t.Errorf("Address mismatch. Expected: %s, Got: %s", wallet.Address, loadedWallet.Address)
	}
}

func TestEncryptDecrypt(t *testing.T) {
	password := "testpassword123"
	plaintext := "sensitive data to encrypt"
	salt := generateSalt()

	// Encrypt
	encrypted, err := encryptData(plaintext, password, salt)
	if err != nil {
		t.Fatalf("Failed to encrypt: %v", err)
	}

	if encrypted == plaintext {
		t.Error("Encrypted data should not equal plaintext")
	}

	// Decrypt
	decrypted, err := decryptData(encrypted, password, salt)
	if err != nil {
		t.Fatalf("Failed to decrypt: %v", err)
	}

	if decrypted != plaintext {
		t.Errorf("Decrypted data mismatch. Expected: %s, Got: %s", plaintext, decrypted)
	}
}

func TestDecryptWithWrongPassword(t *testing.T) {
	password := "correctpassword"
	wrongPassword := "wrongpassword"
	plaintext := "sensitive data"
	salt := generateSalt()

	encrypted, err := encryptData(plaintext, password, salt)
	if err != nil {
		t.Fatalf("Failed to encrypt: %v", err)
	}

	_, err = decryptData(encrypted, wrongPassword, salt)
	if err == nil {
		t.Error("Decryption should fail with wrong password")
	}
}

func TestSaltGeneration(t *testing.T) {
	salt1 := generateSalt()
	salt2 := generateSalt()

	if len(salt1) != 32 {
		t.Errorf("Salt should be 32 bytes, got %d", len(salt1))
	}

	if len(salt2) != 32 {
		t.Errorf("Salt should be 32 bytes, got %d", len(salt2))
	}

	// Salts should be unique
	if string(salt1) == string(salt2) {
		t.Error("Generated salts should be different")
	}
}

func BenchmarkGenerateCosmosAddress(b *testing.B) {
	keyBytes := make([]byte, 32)
	for i := range keyBytes {
		keyBytes[i] = byte(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = GenerateCosmosAddress(keyBytes)
	}
}

func BenchmarkEncryptData(b *testing.B) {
	password := "benchmarkpassword"
	data := "benchmark data to encrypt"
	salt := generateSalt()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = encryptData(data, password, salt)
	}
}
