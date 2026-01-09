package cmd

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"
	"path/filepath"

	"cosmossdk.io/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// configureGRPCTLS configures TLS for gRPC server if certificates are provided
func configureGRPCTLS(
	logger log.Logger,
) ([]grpc.ServerOption, error) {
	// Check for TLS certificate files via environment variables
	certFile := os.Getenv("GRPC_TLS_CERT_FILE")
	keyFile := os.Getenv("GRPC_TLS_KEY_FILE")
	caCertFile := os.Getenv("GRPC_TLS_CA_CERT_FILE")

	// If no TLS configuration, use insecure connection
	if certFile == "" || keyFile == "" {
		logger.Info("gRPC TLS is not configured, using insecure connection")
		return nil, nil
	}

	// Validate certificate files exist
	if _, err := os.Stat(certFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("TLS certificate file not found: %s", certFile)
	}
	if _, err := os.Stat(keyFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("TLS key file not found: %s", keyFile)
	}

	// Load server certificate
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load server certificate: %w", err)
	}

	// Create TLS config
	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
		},
	}

	// If CA certificate is provided, enable mutual authentication
	if caCertFile != "" {
		if _, err := os.Stat(caCertFile); os.IsNotExist(err) {
			return nil, fmt.Errorf("TLS CA certificate file not found: %s", caCertFile)
		}

		caCert, err := os.ReadFile(caCertFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read CA certificate: %w", err)
		}

		caCertPool := x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to parse CA certificate")
		}

		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
		tlsConfig.ClientCAs = caCertPool

		logger.Info("gRPC TLS with mutual authentication enabled",
			"cert", certFile,
			"key", keyFile,
			"ca_cert", caCertFile,
		)
	} else {
		logger.Info("gRPC TLS enabled (server authentication only)",
			"cert", certFile,
			"key", keyFile,
		)
	}

	// Create gRPC server option with TLS credentials
	creds := credentials.NewTLS(tlsConfig)
	return []grpc.ServerOption{grpc.Creds(creds)}, nil
}

// setupTLSFromDefaultPaths checks for certificates in default locations
// and sets environment variables if found
func setupTLSFromDefaultPaths(homeDir string) {
	// Default certificate paths relative to node home directory
	defaultCertPath := filepath.Join(homeDir, "certs", "server-cert.pem")
	defaultKeyPath := filepath.Join(homeDir, "certs", "server-key.pem")
	defaultCACertPath := filepath.Join(homeDir, "certs", "ca-cert.pem")

	// Check if default certificate files exist and set environment variables
	if _, err := os.Stat(defaultCertPath); err == nil {
		if _, err := os.Stat(defaultKeyPath); err == nil {
			os.Setenv("GRPC_TLS_CERT_FILE", defaultCertPath)
			os.Setenv("GRPC_TLS_KEY_FILE", defaultKeyPath)

			// Check for CA certificate
			if _, err := os.Stat(defaultCACertPath); err == nil {
				os.Setenv("GRPC_TLS_CA_CERT_FILE", defaultCACertPath)
			}
		}
	}
}

// GetGRPCServerOptions returns gRPC server options with TLS if configured
func GetGRPCServerOptions(
	logger log.Logger,
	homeDir string,
) ([]grpc.ServerOption, error) {
	// Setup TLS from default paths if not configured via environment variables
	if os.Getenv("GRPC_TLS_CERT_FILE") == "" {
		setupTLSFromDefaultPaths(homeDir)
	}

	// Configure gRPC TLS
	return configureGRPCTLS(logger)
}

// setupTLSForGRPCServer is called from PostSetup hook to verify and log TLS configuration
func setupTLSForGRPCServer(logger log.Logger, rootDir string) error {
	// Setup TLS from default paths if not configured via environment variables
	if os.Getenv("GRPC_TLS_CERT_FILE") == "" {
		setupTLSFromDefaultPaths(rootDir)
	}

	// Check if TLS is configured
	certFile := os.Getenv("GRPC_TLS_CERT_FILE")
	keyFile := os.Getenv("GRPC_TLS_KEY_FILE")
	caCertFile := os.Getenv("GRPC_TLS_CA_CERT_FILE")

	if certFile != "" && keyFile != "" {
		logger.Info("gRPC TLS configuration detected",
			"cert_file", certFile,
			"key_file", keyFile,
			"ca_cert_file", caCertFile,
			"mtls_enabled", caCertFile != "",
		)

		// Verify certificate files exist
		if _, err := os.Stat(certFile); os.IsNotExist(err) {
			logger.Error("TLS certificate file not found", "file", certFile)
			return fmt.Errorf("TLS certificate file not found: %s", certFile)
		}
		if _, err := os.Stat(keyFile); os.IsNotExist(err) {
			logger.Error("TLS key file not found", "file", keyFile)
			return fmt.Errorf("TLS key file not found: %s", keyFile)
		}
		if caCertFile != "" {
			if _, err := os.Stat(caCertFile); os.IsNotExist(err) {
				logger.Error("TLS CA certificate file not found", "file", caCertFile)
				return fmt.Errorf("TLS CA certificate file not found: %s", caCertFile)
			}
		}

		logger.Info("gRPC TLS configuration verified successfully")
	} else {
		logger.Info("gRPC TLS is not configured, using insecure connection")
	}

	// Note: In Cosmos SDK v0.50.x, gRPC server is started by runtime
	// and doesn't directly read TLS configuration from environment variables.
	// The actual TLS configuration would need to be injected at the gRPC server
	// creation time, which requires SDK modification or a custom server implementation.
	// For now, we verify and log the configuration here.

	return nil
}

