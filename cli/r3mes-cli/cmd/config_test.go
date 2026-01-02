package cmd

import (
	"os"
	"testing"

	"github.com/spf13/viper"
)

func TestContainsLocalhost(t *testing.T) {
	testCases := []struct {
		url      string
		expected bool
	}{
		{"http://localhost:8080", true},
		{"https://localhost:443", true},
		{"http://127.0.0.1:9090", true},
		{"https://127.0.0.1", true},
		{"http://0.0.0.0:3000", true},
		{"http://[::1]:8080", true},
		{"https://api.r3mes.network", false},
		{"http://192.168.1.100:8080", false},
		{"https://rpc.r3mes.network:26657", false},
		{"http://10.0.0.1:9090", false},
	}

	for _, tc := range testCases {
		t.Run(tc.url, func(t *testing.T) {
			result := containsLocalhost(tc.url)
			if result != tc.expected {
				t.Errorf("containsLocalhost(%s) = %v, expected %v", tc.url, result, tc.expected)
			}
		})
	}
}

func TestMaskIfEmpty(t *testing.T) {
	testCases := []struct {
		input    string
		expected string
	}{
		{"", "(not set)"},
		{"value", "value"},
		{"http://localhost:8080", "http://localhost:8080"},
	}

	for _, tc := range testCases {
		t.Run(tc.input, func(t *testing.T) {
			result := maskIfEmpty(tc.input)
			if result != tc.expected {
				t.Errorf("maskIfEmpty(%s) = %v, expected %v", tc.input, result, tc.expected)
			}
		})
	}
}

func TestGetKeys(t *testing.T) {
	m := map[string]bool{
		"key1": true,
		"key2": true,
		"key3": true,
	}

	keys := getKeys(m)
	if len(keys) != 3 {
		t.Errorf("Expected 3 keys, got %d", len(keys))
	}
}

func TestConfigStruct(t *testing.T) {
	config := &Config{
		RPCEndpoint:  "http://test-rpc:26657",
		GRPCEndpoint: "test-grpc:9090",
		ChainID:      "test-chain",
		WalletPath:   "/tmp/wallets",
		MinerPort:    "9999",
		JSONOutput:   true,
		Verbose:      false,
	}

	if config.RPCEndpoint != "http://test-rpc:26657" {
		t.Errorf("Expected RPC endpoint 'http://test-rpc:26657', got '%s'", config.RPCEndpoint)
	}

	if config.GRPCEndpoint != "test-grpc:9090" {
		t.Errorf("Expected gRPC endpoint 'test-grpc:9090', got '%s'", config.GRPCEndpoint)
	}

	if config.ChainID != "test-chain" {
		t.Errorf("Expected chain ID 'test-chain', got '%s'", config.ChainID)
	}

	if config.MinerPort != "9999" {
		t.Errorf("Expected miner port '9999', got '%s'", config.MinerPort)
	}

	if !config.JSONOutput {
		t.Error("Expected JSONOutput to be true")
	}
}

func TestViperEnvBinding(t *testing.T) {
	// Reset viper for clean test
	viper.Reset()

	// Set test env vars
	os.Setenv("R3MES_RPC_ENDPOINT", "http://test-rpc:26657")
	os.Setenv("R3MES_CHAIN_ID", "test-chain")
	defer func() {
		os.Unsetenv("R3MES_RPC_ENDPOINT")
		os.Unsetenv("R3MES_CHAIN_ID")
	}()

	// Configure viper
	viper.SetEnvPrefix("R3MES")
	viper.AutomaticEnv()

	// Test reading env vars through viper
	rpcEndpoint := viper.GetString("RPC_ENDPOINT")
	if rpcEndpoint != "http://test-rpc:26657" {
		t.Errorf("Expected RPC endpoint 'http://test-rpc:26657', got '%s'", rpcEndpoint)
	}

	chainID := viper.GetString("CHAIN_ID")
	if chainID != "test-chain" {
		t.Errorf("Expected chain ID 'test-chain', got '%s'", chainID)
	}
}

func TestValidateConfig(t *testing.T) {
	// Test production validation
	os.Setenv("R3MES_ENV", "production")
	defer os.Unsetenv("R3MES_ENV")

	// Should fail with localhost
	config := &Config{
		RPCEndpoint:  "http://localhost:26657",
		GRPCEndpoint: "localhost:9090",
	}

	err := validateConfig(config)
	if err == nil {
		t.Error("Expected error for localhost in production, got nil")
	}

	// Should pass with non-localhost
	config = &Config{
		RPCEndpoint:  "https://rpc.r3mes.network:26657",
		GRPCEndpoint: "grpc.r3mes.network:9090",
	}

	err = validateConfig(config)
	if err != nil {
		t.Errorf("Expected no error for non-localhost, got: %v", err)
	}
}

func TestValidateConfigDevelopment(t *testing.T) {
	// Test development mode (should allow localhost)
	os.Setenv("R3MES_ENV", "development")
	defer os.Unsetenv("R3MES_ENV")

	config := &Config{
		RPCEndpoint:  "http://localhost:26657",
		GRPCEndpoint: "localhost:9090",
	}

	err := validateConfig(config)
	if err != nil {
		t.Errorf("Expected no error in development mode, got: %v", err)
	}
}
