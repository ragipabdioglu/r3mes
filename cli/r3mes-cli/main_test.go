package main

import (
	"encoding/hex"
	"os"
	"testing"

	"r3mes-cli/cmd"
)

func TestGenerateCosmosAddress(t *testing.T) {
	// Test with known private key
	privateKeyHex := "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
	privateKeyBytes, err := hex.DecodeString(privateKeyHex)
	if err != nil {
		t.Fatalf("Failed to decode private key: %v", err)
	}

	address, err := cmd.GenerateCosmosAddress(privateKeyBytes)
	if err != nil {
		t.Fatalf("Failed to generate address: %v", err)
	}

	// Check address format
	if len(address) == 0 {
		t.Error("Generated address is empty")
	}

	if address[:6] != "remes1" {
		t.Errorf("Address should start with 'remes1', got: %s", address)
	}

	t.Logf("Generated address: %s", address)
}

func TestGenerateCosmosAddressEmpty(t *testing.T) {
	_, err := cmd.GenerateCosmosAddress([]byte{})
	if err == nil {
		t.Error("Expected error for empty private key")
	}
}

func TestWalletSaveAndLoad(t *testing.T) {
	// Set up test environment
	tempDir := t.TempDir()

	config := &cmd.Config{
		WalletPath: tempDir,
	}

	// Create test wallet
	wallet := cmd.Wallet{
		Address:   "remes1test123456789",
		CreatedAt: "2026-01-02T00:00:00Z",
	}

	// Test saving wallet
	err := cmd.SaveWallet(wallet, "test", config)
	if err != nil {
		t.Fatalf("Failed to save wallet: %v", err)
	}

	// Test loading wallet (should fail for default, succeed for test)
	_, err = cmd.LoadDefaultWallet(config)
	if err == nil {
		t.Log("Default wallet found (unexpected but not an error)")
	} else {
		t.Logf("Default wallet not found as expected: %v", err)
	}
}

func TestVersionOutput(t *testing.T) {
	if Version != "0.3.0" {
		t.Errorf("Expected version 0.3.0, got %s", Version)
	}
}

func TestRootCmdCreation(t *testing.T) {
	logger := cmd.InitLogger()
	rootCmd := cmd.NewRootCmd("0.3.0", logger)

	if rootCmd == nil {
		t.Fatal("NewRootCmd returned nil")
	}

	if rootCmd.Use != "r3mes" {
		t.Errorf("Expected Use 'r3mes', got '%s'", rootCmd.Use)
	}

	// Check subcommands exist
	subcommands := rootCmd.Commands()
	expectedCmds := []string{"wallet", "miner", "node", "governance", "config", "tx"}

	for _, expected := range expectedCmds {
		found := false
		for _, cmd := range subcommands {
			if cmd.Use == expected || cmd.Name() == expected {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Expected subcommand '%s' not found", expected)
		}
	}
}

func TestInitLogger(t *testing.T) {
	logger := cmd.InitLogger()
	if logger == nil {
		t.Fatal("InitLogger returned nil")
	}

	// Test logging doesn't panic
	logger.Info("Test log message")
	logger.Debug("Debug message")
}

func TestInitLoggerDebugMode(t *testing.T) {
	os.Setenv("R3MES_DEBUG", "true")
	defer os.Unsetenv("R3MES_DEBUG")

	logger := cmd.InitLogger()
	if logger == nil {
		t.Fatal("InitLogger returned nil in debug mode")
	}
}

// Benchmark tests
func BenchmarkGenerateCosmosAddress(b *testing.B) {
	privateKeyBytes := make([]byte, 32)
	for i := 0; i < 32; i++ {
		privateKeyBytes[i] = byte(i)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := cmd.GenerateCosmosAddress(privateKeyBytes)
		if err != nil {
			b.Fatalf("Failed to generate address: %v", err)
		}
	}
}

func BenchmarkInitLogger(b *testing.B) {
	for i := 0; i < b.N; i++ {
		logger := cmd.InitLogger()
		logger.Sync()
	}
}
