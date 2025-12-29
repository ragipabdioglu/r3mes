package keeper

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"
)

// IPFSManager handles IPFS operations for Go node (passive role).
// Go node only retrieves content from IPFS for validation, never stores.
type IPFSManager struct {
	apiURL   string
	timeout  time.Duration
	client   *http.Client
}

// NewIPFSManager creates a new IPFS manager for passive retrieval.
func NewIPFSManager(apiURL string) *IPFSManager {
	return &IPFSManager{
		apiURL:  apiURL,
		timeout: 30 * time.Second,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// RetrieveContent retrieves content from IPFS by hash (passive role).
// This is only called when validation is required, not during normal submission.
func (im *IPFSManager) RetrieveContent(ctx context.Context, ipfsHash string) ([]byte, error) {
	// Construct IPFS API URL
	url := fmt.Sprintf("%s/api/v0/cat?arg=%s", im.apiURL, ipfsHash)
	
	// Create request with context
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	
	// Execute request
	resp, err := im.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve from IPFS: %w", err)
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("IPFS API returned status %d", resp.StatusCode)
	}
	
	// Read content
	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read IPFS content: %w", err)
	}
	
	return content, nil
}

// VerifyContentExists checks if content exists in IPFS without downloading.
func (im *IPFSManager) VerifyContentExists(ctx context.Context, ipfsHash string) (bool, error) {
	// Use IPFS stat API to check existence
	url := fmt.Sprintf("%s/api/v0/object/stat?arg=%s", im.apiURL, ipfsHash)
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return false, fmt.Errorf("failed to create request: %w", err)
	}
	
	resp, err := im.client.Do(req)
	if err != nil {
		return false, fmt.Errorf("failed to check IPFS content: %w", err)
	}
	defer resp.Body.Close()
	
	return resp.StatusCode == http.StatusOK, nil
}

// IsAvailable checks if IPFS daemon is available.
func (im *IPFSManager) IsAvailable() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	// Try to get IPFS version as health check
	url := fmt.Sprintf("%s/api/v0/version", im.apiURL)
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return false
	}
	
	resp, err := im.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	
	return resp.StatusCode == http.StatusOK
}

// RetrieveGradientTensor retrieves gradient tensor from IPFS and deserializes it
// This is a wrapper that uses RetrieveContent and DeserializeGradientTensor
func (im *IPFSManager) RetrieveGradientTensor(ctx context.Context, ipfsHash string) ([]float64, error) {
	// Retrieve raw data from IPFS
	data, err := im.RetrieveContent(ctx, ipfsHash)
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve gradient from IPFS: %w", err)
	}
	
	// Deserialize gradient tensor
	gradient, err := DeserializeGradientTensor(data)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize gradient tensor: %w", err)
	}
	
	return gradient, nil
}

