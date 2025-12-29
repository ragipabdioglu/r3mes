package keeper

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"os"
)

// LoadTLSConfig loads TLS configuration for gRPC server with mutual authentication
func LoadTLSConfig(
	certFile string,
	keyFile string,
	caFile string,
) (*tls.Config, error) {
	// Load server certificate
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load server certificate: %w", err)
	}

	// Load CA certificate for client verification
	caCert, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load CA certificate: %w", err)
	}

	caCertPool := x509.NewCertPool()
	if !caCertPool.AppendCertsFromPEM(caCert) {
		return nil, fmt.Errorf("failed to parse CA certificate")
	}

	// Create TLS config with mutual authentication
	config := &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientAuth:   tls.RequireAndVerifyClientCert, // Require client certificate
		ClientCAs:    caCertPool,
		MinVersion:   tls.VersionTLS12,
		// Cipher suites for security
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
		},
	}

	return config, nil
}

// LoadTLSClientConfig loads TLS configuration for gRPC client with mutual authentication
func LoadTLSClientConfig(
	certFile string,
	keyFile string,
	caFile string,
	serverName string,
) (*tls.Config, error) {
	// Load client certificate
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load client certificate: %w", err)
	}

	// Load CA certificate for server verification
	caCert, err := os.ReadFile(caFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load CA certificate: %w", err)
	}

	caCertPool := x509.NewCertPool()
	if !caCertPool.AppendCertsFromPEM(caCert) {
		return nil, fmt.Errorf("failed to parse CA certificate")
	}

	// Create TLS config with mutual authentication
	config := &tls.Config{
		Certificates: []tls.Certificate{cert},
		RootCAs:      caCertPool,
		ServerName:   serverName, // Server name for verification
		MinVersion:   tls.VersionTLS12,
		// Cipher suites for security
		CipherSuites: []uint16{
			tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305,
			tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
			tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305,
		},
	}

	return config, nil
}

// VerifyCertificate verifies a client certificate against allowed miner addresses
// This ensures only registered miners can connect
func VerifyCertificate(
	cert *x509.Certificate,
	allowedAddresses []string,
) (bool, error) {
	// Extract miner address from certificate subject
	// In production, this would be stored in certificate's Subject Alternative Name (SAN)
	// For now, we'll use Common Name (CN) as a simple approach
	minerAddress := cert.Subject.CommonName

	// Check if miner address is in allowed list
	for _, addr := range allowedAddresses {
		if addr == minerAddress {
			return true, nil
		}
	}

	return false, fmt.Errorf("certificate miner address %s not in allowed list", minerAddress)
}

