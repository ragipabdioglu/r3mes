package cmd

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
)

// Config holds CLI configuration
type Config struct {
	RPCEndpoint  string `json:"rpc_endpoint" mapstructure:"rpc_endpoint"`
	GRPCEndpoint string `json:"grpc_endpoint" mapstructure:"grpc_endpoint"`
	ChainID      string `json:"chain_id" mapstructure:"chain_id"`
	WalletPath   string `json:"wallet_path" mapstructure:"wallet_path"`
	MinerPort    string `json:"miner_port" mapstructure:"miner_port"`
	JSONOutput   bool   `json:"json_output" mapstructure:"json_output"`
	Verbose      bool   `json:"verbose" mapstructure:"verbose"`
}

func newConfigCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "config",
		Short: "Configuration management",
		Long:  "View and manage R3MES CLI configuration settings.",
	}

	cmd.AddCommand(newConfigShowCmd())
	cmd.AddCommand(newConfigSetCmd())
	cmd.AddCommand(newConfigInitCmd())

	return cmd
}

func newConfigShowCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "show",
		Short: "Show current configuration",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			config := GetConfig()
			if config.JSONOutput {
				data, _ := json.MarshalIndent(config, "", "  ")
				fmt.Println(string(data))
				return nil
			}

			fmt.Println("Current Configuration:")
			fmt.Printf("  RPC Endpoint:  %s\n", maskIfEmpty(config.RPCEndpoint))
			fmt.Printf("  gRPC Endpoint: %s\n", maskIfEmpty(config.GRPCEndpoint))
			fmt.Printf("  Chain ID:      %s\n", config.ChainID)
			fmt.Printf("  Wallet Path:   %s\n", config.WalletPath)
			fmt.Printf("  Miner Port:    %s\n", config.MinerPort)

			if configFile := viper.ConfigFileUsed(); configFile != "" {
				fmt.Printf("\nConfig file: %s\n", configFile)
			}

			return nil
		},
	}
}

func newConfigSetCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "set <key> <value>",
		Short: "Set a configuration value",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			key := args[0]
			value := args[1]

			validKeys := map[string]bool{
				"rpc_endpoint":  true,
				"grpc_endpoint": true,
				"chain_id":      true,
				"wallet_path":   true,
				"miner_port":    true,
			}

			if !validKeys[key] {
				return fmt.Errorf("invalid config key: %s. Valid keys: %v", key, getKeys(validKeys))
			}

			viper.Set(key, value)

			// Ensure config directory exists
			home, _ := os.UserHomeDir()
			configPath := filepath.Join(home, ".r3mes", "config.yaml")

			if err := viper.WriteConfigAs(configPath); err != nil {
				return fmt.Errorf("failed to write config: %w", err)
			}

			logger.Info("Configuration updated",
				zap.String("key", key),
				zap.String("value", value),
				zap.String("file", configPath))

			fmt.Printf("✅ Set %s = %s\n", key, value)
			return nil
		},
	}
}

func newConfigInitCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "init",
		Short: "Initialize configuration file with defaults",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
			}

			home, err := os.UserHomeDir()
			if err != nil {
				return fmt.Errorf("failed to get home directory: %w", err)
			}

			configDir := filepath.Join(home, ".r3mes")
			if err := os.MkdirAll(configDir, 0700); err != nil {
				return fmt.Errorf("failed to create config directory: %w", err)
			}

			configPath := filepath.Join(configDir, "config.yaml")

			// Check if config already exists
			if _, err := os.Stat(configPath); err == nil {
				fmt.Printf("Config file already exists: %s\n", configPath)
				fmt.Print("Overwrite? [y/N]: ")
				var response string
				fmt.Scanln(&response)
				if strings.ToLower(response) != "y" {
					fmt.Println("Aborted.")
					return nil
				}
			}

			// Set defaults
			viper.Set("rpc_endpoint", "https://rpc.r3mes.network:26657")
			viper.Set("grpc_endpoint", "https://grpc.r3mes.network:9090")
			viper.Set("chain_id", "remes-mainnet")
			viper.Set("wallet_path", filepath.Join(configDir, "wallets"))
			viper.Set("miner_port", "8080")

			if err := viper.WriteConfigAs(configPath); err != nil {
				return fmt.Errorf("failed to write config: %w", err)
			}

			fmt.Printf("✅ Configuration initialized: %s\n", configPath)
			fmt.Println("\nEdit the file to customize your settings.")
			return nil
		},
	}
}

func maskIfEmpty(s string) string {
	if s == "" {
		return "(not set)"
	}
	return s
}

func getKeys(m map[string]bool) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	return keys
}

func containsLocalhost(url string) bool {
	lower := strings.ToLower(url)
	localhostIndicators := []string{"localhost", "127.0.0.1", "::1", "0.0.0.0"}
	for _, indicator := range localhostIndicators {
		if strings.Contains(lower, indicator) {
			return true
		}
	}
	return false
}
