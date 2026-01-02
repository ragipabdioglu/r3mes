package main

import (
	"context"
	"os"
	"os/signal"
	"syscall"

	"r3mes-cli/cmd"

	"go.uber.org/zap"
)

const Version = "0.3.0"

func main() {
	// Initialize logger
	logger := cmd.InitLogger()
	defer logger.Sync()

	// Create context with cancellation for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigChan
		logger.Info("Received shutdown signal", zap.String("signal", sig.String()))
		cancel()
	}()

	// Execute root command with context
	rootCmd := cmd.NewRootCmd(Version, logger)
	if err := rootCmd.ExecuteContext(ctx); err != nil {
		logger.Error("Command execution failed", zap.Error(err))
		os.Exit(1)
	}
}
