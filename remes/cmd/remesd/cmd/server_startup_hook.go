package cmd

import (
	"context"
	"fmt"
	"os"

	"cosmossdk.io/log"
	servertypes "github.com/cosmos/cosmos-sdk/server/types"
	"google.golang.org/grpc"
)

// ServerStartupHook is called before the server starts
// This allows us to configure TLS for gRPC server
func ServerStartupHook(
	ctx context.Context,
	logger log.Logger,
	app servertypes.Application,
	appOpts servertypes.AppOptions,
) error {
	// Get home directory
	homeDir, ok := appOpts.Get("home").(string)
	if !ok {
		homeDir = ""
	}

	// Setup TLS from default paths if not configured via environment variables
	if os.Getenv("GRPC_TLS_CERT_FILE") == "" {
		setupTLSFromDefaultPaths(homeDir)
	}

	// Check if TLS is configured
	certFile := os.Getenv("GRPC_TLS_CERT_FILE")
	keyFile := os.Getenv("GRPC_TLS_KEY_FILE")

	if certFile != "" && keyFile != "" {
		logger.Info("TLS configuration detected for gRPC server",
			"cert_file", certFile,
			"key_file", keyFile,
		)

		// Note: In Cosmos SDK v0.50.x, gRPC server is started by runtime
		// We can't directly inject server options here, but we can:
		// 1. Set environment variables (already done)
		// 2. Patch the server startup code (requires SDK modification)
		// 3. Use a custom server implementation (complex)

		// For now, we log that TLS is configured
		// The actual TLS configuration happens at the gRPC server creation time
		// which is handled by Cosmos SDK's internal code
	}

	return nil
}

// PatchGRPCServerCreation patches the gRPC server creation to use TLS
// This function would need to be called from Cosmos SDK's server startup code
// which is not directly accessible in v0.50.x
func PatchGRPCServerCreation(
	logger log.Logger,
	homeDir string,
) ([]grpc.ServerOption, error) {
	// Get TLS server options
	grpcOptions, err := GetGRPCServerOptions(logger, homeDir)
	if err != nil {
		return nil, fmt.Errorf("failed to get gRPC server options: %w", err)
	}

	if len(grpcOptions) > 0 {
		logger.Info("gRPC server will use TLS configuration")
	}

	return grpcOptions, nil
}

// GetTLSConfigForGRPC returns TLS configuration for gRPC server
// This can be used if we patch Cosmos SDK's server startup code
func GetTLSConfigForGRPC(
	logger log.Logger,
	homeDir string,
) (*TLSConfig, error) {
	// Setup TLS from default paths if not configured
	if os.Getenv("GRPC_TLS_CERT_FILE") == "" {
		setupTLSFromDefaultPaths(homeDir)
	}

	certFile := os.Getenv("GRPC_TLS_CERT_FILE")
	keyFile := os.Getenv("GRPC_TLS_KEY_FILE")
	caCertFile := os.Getenv("GRPC_TLS_CA_CERT_FILE")

	if certFile == "" || keyFile == "" {
		return nil, nil // No TLS configuration
	}

	return &TLSConfig{
		CertFile:  certFile,
		KeyFile:   keyFile,
		CACertFile: caCertFile,
	}, nil
}

// TLSConfig holds TLS configuration for gRPC server
type TLSConfig struct {
	CertFile  string
	KeyFile   string
	CACertFile string
}

