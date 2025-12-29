package keeper

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	verificationgrpc "remes/x/remes/types/remes/remes/v1"
	verificationpb "remes/x/remes/types/remes/remes/v1"
)

// VerificationClient handles gRPC communication with Python inference service
type VerificationClient struct {
	conn   *grpc.ClientConn
	client verificationgrpc.VerificationServiceClient
	addr   string
}

// NewVerificationClient creates a new verification client
// Address can be set via R3MES_VERIFICATION_SERVICE_ADDR environment variable
// In production, R3MES_VERIFICATION_SERVICE_ADDR must be set (no localhost fallback)
func NewVerificationClient() (*VerificationClient, error) {
	addr := os.Getenv("R3MES_VERIFICATION_SERVICE_ADDR")
	if addr == "" {
		// In production, do not allow localhost default
		if os.Getenv("R3MES_ENV") == "production" {
			return nil, fmt.Errorf("R3MES_VERIFICATION_SERVICE_ADDR must be set in production (cannot use localhost default)")
		}
		addr = "localhost:50051"
	} else if os.Getenv("R3MES_ENV") == "production" {
		// Validate that production doesn't use localhost
		if strings.Contains(addr, "localhost") || strings.Contains(addr, "127.0.0.1") {
			return nil, fmt.Errorf("R3MES_VERIFICATION_SERVICE_ADDR cannot use localhost in production: %s", addr)
		}
	}

	// Use insecure credentials for local development
	// In production, use proper TLS credentials
	conn, err := grpc.Dial(
		addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithTimeout(10*time.Second),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to verification service at %s: %w", addr, err)
	}

	client := verificationgrpc.NewVerificationServiceClient(conn)

	return &VerificationClient{
		conn:   conn,
		client: client,
		addr:   addr,
	}, nil
}

// Close closes the gRPC connection
func (vc *VerificationClient) Close() error {
	if vc.conn != nil {
		return vc.conn.Close()
	}
	return nil
}

// RunForwardPass calls the Python inference service to run a forward pass
func (vc *VerificationClient) RunForwardPass(
	ctx context.Context,
	weightsIPFS string,
	batchID string,
	modelConfigID uint64,
	datasetIPFS string,
	timeoutSeconds uint32,
) (int64, error) {
	if timeoutSeconds == 0 {
		timeoutSeconds = 300 // Default 5 minutes
	}

	// Create context with timeout
	requestCtx, cancel := context.WithTimeout(ctx, time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	req := &verificationpb.RunForwardPassRequest{
		WeightsIpfsHash: weightsIPFS,
		BatchId:         batchID,
		ModelConfigId:   modelConfigID,
		DatasetIpfsHash: datasetIPFS,
		TimeoutSeconds:  timeoutSeconds,
	}

	resp, err := vc.client.RunForwardPass(requestCtx, req)
	if err != nil {
		return 0, fmt.Errorf("verification service error: %w", err)
	}

	if !resp.Success {
		return 0, fmt.Errorf("verification failed: %s", resp.ErrorMessage)
	}

	return resp.LossInt, nil
}

// HealthCheck checks if the verification service is healthy
func (vc *VerificationClient) HealthCheck(ctx context.Context) (bool, error) {
	req := &verificationpb.HealthCheckRequest{}
	resp, err := vc.client.HealthCheck(ctx, req)
	if err != nil {
		return false, fmt.Errorf("health check failed: %w", err)
	}

	return resp.Healthy, nil
}
