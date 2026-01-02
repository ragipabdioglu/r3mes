package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

var (
	cfgFile string
	logger  *zap.Logger
	cfg     *Config
)

// NewRootCmd creates the root command with all subcommands
func NewRootCmd(version string, log *zap.Logger) *cobra.Command {
	logger = log

	rootCmd := &cobra.Command{
		Use:   "r3mes",
		Short: "R3MES CLI - Command-line interface for R3MES Network",
		Long: `R3MES CLI provides a comprehensive command-line interface for interacting
with the R3MES decentralized AI training network.

Features:
  - Wallet management (create, import, export, balance)
  - Miner operations (start, stop, status, stats)
  - Node management (start, stop, status, sync)
  - Governance participation (vote, proposals)
  - Transaction signing and broadcasting`,
		Version: version,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			return initConfig()
		},
		SilenceUsage:  true,
		SilenceErrors: true,
	}

	// Persistent flags (available to all commands)
	rootCmd.PersistentFlags().StringVar(&cfgFile, "config", "", "config file (default: $HOME/.r3mes/config.yaml)")
	rootCmd.PersistentFlags().String("rpc-endpoint", "", "RPC endpoint URL")
	rootCmd.PersistentFlags().String("grpc-endpoint", "", "gRPC endpoint URL")
	rootCmd.PersistentFlags().String("chain-id", "remes-test", "Chain ID")
	rootCmd.PersistentFlags().String("wallet-path", "", "Wallet storage path")
	rootCmd.PersistentFlags().Bool("json", false, "Output in JSON format")
	rootCmd.PersistentFlags().Bool("verbose", false, "Enable verbose logging")

	// Bind flags to viper
	viper.BindPFlag("rpc_endpoint", rootCmd.PersistentFlags().Lookup("rpc-endpoint"))
	viper.BindPFlag("grpc_endpoint", rootCmd.PersistentFlags().Lookup("grpc-endpoint"))
	viper.BindPFlag("chain_id", rootCmd.PersistentFlags().Lookup("chain-id"))
	viper.BindPFlag("wallet_path", rootCmd.PersistentFlags().Lookup("wallet-path"))
	viper.BindPFlag("json_output", rootCmd.PersistentFlags().Lookup("json"))
	viper.BindPFlag("verbose", rootCmd.PersistentFlags().Lookup("verbose"))

	// Add subcommands
	rootCmd.AddCommand(newWalletCmd())
	rootCmd.AddCommand(newMinerCmd())
	rootCmd.AddCommand(newNodeCmd())
	rootCmd.AddCommand(newGovernanceCmd())
	rootCmd.AddCommand(newConfigCmd())
	rootCmd.AddCommand(newTxCmd())

	return rootCmd
}

// initConfig reads in config file and ENV variables
func initConfig() error {
	if cfgFile != "" {
		viper.SetConfigFile(cfgFile)
	} else {
		home, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("failed to get home directory: %w", err)
		}

		configDir := filepath.Join(home, ".r3mes")
		if err := os.MkdirAll(configDir, 0700); err != nil {
			return fmt.Errorf("failed to create config directory: %w", err)
		}

		viper.AddConfigPath(configDir)
		viper.SetConfigName("config")
		viper.SetConfigType("yaml")
	}

	// Environment variables
	viper.SetEnvPrefix("R3MES")
	viper.SetEnvKeyReplacer(strings.NewReplacer("-", "_"))
	viper.AutomaticEnv()

	// Read config file (ignore if not found)
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return fmt.Errorf("failed to read config: %w", err)
		}
	}

	// Build config struct
	cfg = &Config{
		RPCEndpoint:  viper.GetString("rpc_endpoint"),
		GRPCEndpoint: viper.GetString("grpc_endpoint"),
		ChainID:      viper.GetString("chain_id"),
		WalletPath:   viper.GetString("wallet_path"),
		MinerPort:    viper.GetString("miner_port"),
		JSONOutput:   viper.GetBool("json_output"),
		Verbose:      viper.GetBool("verbose"),
	}

	// Set defaults
	if cfg.ChainID == "" {
		cfg.ChainID = "remes-test"
	}
	if cfg.WalletPath == "" {
		home, _ := os.UserHomeDir()
		cfg.WalletPath = filepath.Join(home, ".r3mes", "wallets")
	}
	if cfg.MinerPort == "" {
		cfg.MinerPort = "8080"
	}

	// Validate production config
	if err := validateConfig(cfg); err != nil {
		return err
	}

	return nil
}

func validateConfig(config *Config) error {
	env := strings.ToLower(os.Getenv("R3MES_ENV"))
	if env == "production" || env == "prod" {
		if containsLocalhost(config.RPCEndpoint) {
			return fmt.Errorf("RPC endpoint cannot use localhost in production")
		}
		if containsLocalhost(config.GRPCEndpoint) {
			return fmt.Errorf("gRPC endpoint cannot use localhost in production")
		}
	}
	return nil
}

// InitLogger creates a new zap logger
func InitLogger() *zap.Logger {
	config := zap.NewProductionConfig()
	config.EncoderConfig.TimeKey = "timestamp"
	config.EncoderConfig.EncodeTime = zapcore.ISO8601TimeEncoder

	if os.Getenv("R3MES_DEBUG") == "true" || viper.GetBool("verbose") {
		config.Level = zap.NewAtomicLevelAt(zap.DebugLevel)
	}

	log, err := config.Build()
	if err != nil {
		// Fallback to basic logger
		log, _ = zap.NewProduction()
	}

	return log
}

// GetConfig returns the current configuration
func GetConfig() *Config {
	return cfg
}

// GetLogger returns the logger instance
func GetLogger() *zap.Logger {
	return logger
}
