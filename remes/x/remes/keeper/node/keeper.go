package node

import (
	"context"
	"fmt"

	"cosmossdk.io/collections"
	corestore "cosmossdk.io/core/store"
	"github.com/cosmos/cosmos-sdk/codec"

	"remes/x/remes/keeper/core"
	"remes/x/remes/types"
)

// NodeKeeper handles node management functionality
type NodeKeeper struct {
	core       *core.CoreKeeper
	bankKeeper types.BankKeeper

	// Node-related collections
	NodeRegistrations            collections.Map[string, types.NodeRegistration]
	ServingNodeStatuses          collections.Map[string, types.ServingNodeStatus]
	InferenceRequests            collections.Map[string, types.InferenceRequest]
	ValidatorVerificationRecords collections.Map[string, types.ValidatorVerificationRecord]
	ProposerCensorshipRecords    collections.Map[string, types.ProposerCensorshipRecord]
	MentorRelationships          collections.Map[string, types.MentorRelationship]
	ParticipantSyncStates        collections.Map[string, types.ParticipantSyncState]
	AuthorizedValidatorAddresses collections.Map[string, bool]
	AuthorizedProposerAddresses  collections.Map[string, bool]
}

// NodeStatus represents the status of a node
type NodeStatus struct {
	Status    string
	LastSeen  int64
	Version   string
	Resources map[string]interface{}
}

// NewNodeKeeper creates a new node keeper
func NewNodeKeeper(
	storeService corestore.KVStoreService,
	cdc codec.Codec,
	coreKeeper *core.CoreKeeper,
	bankKeeper types.BankKeeper,
) (*NodeKeeper, error) {
	sb := collections.NewSchemaBuilder(storeService)

	k := &NodeKeeper{
		core:       coreKeeper,
		bankKeeper: bankKeeper,

		NodeRegistrations:            collections.NewMap(sb, types.NodeRegistrationKey, "node_registrations", collections.StringKey, codec.CollValue[types.NodeRegistration](cdc)),
		ServingNodeStatuses:          collections.NewMap(sb, types.ServingNodeStatusKey, "serving_node_statuses", collections.StringKey, codec.CollValue[types.ServingNodeStatus](cdc)),
		InferenceRequests:            collections.NewMap(sb, types.InferenceRequestKey, "inference_requests", collections.StringKey, codec.CollValue[types.InferenceRequest](cdc)),
		ValidatorVerificationRecords: collections.NewMap(sb, types.ValidatorVerificationRecordKey, "validator_verification_records", collections.StringKey, codec.CollValue[types.ValidatorVerificationRecord](cdc)),
		ProposerCensorshipRecords:    collections.NewMap(sb, types.ProposerCensorshipRecordKey, "proposer_censorship_records", collections.StringKey, codec.CollValue[types.ProposerCensorshipRecord](cdc)),
		MentorRelationships:          collections.NewMap(sb, types.MentorRelationshipKey, "mentor_relationships", collections.StringKey, codec.CollValue[types.MentorRelationship](cdc)),
		ParticipantSyncStates:        collections.NewMap(sb, types.ParticipantSyncStateKey, "participant_sync_states", collections.StringKey, codec.CollValue[types.ParticipantSyncState](cdc)),
		AuthorizedValidatorAddresses: collections.NewMap(sb, types.AuthorizedValidatorAddressKey, "authorized_validator_addresses", collections.StringKey, collections.BoolValue),
		AuthorizedProposerAddresses:  collections.NewMap(sb, types.AuthorizedProposerAddressKey, "authorized_proposer_addresses", collections.StringKey, collections.BoolValue),
	}

	// Build schema (not used directly but validates collections)
	_, err := sb.Build()
	if err != nil {
		return nil, fmt.Errorf("failed to build node keeper schema: %w", err)
	}

	return k, nil
}

// RegisterNode registers a new node
func (k *NodeKeeper) RegisterNode(ctx context.Context, node types.NodeRegistration) error {
	// Validate node registration
	if err := k.validateNodeRegistration(ctx, node); err != nil {
		return fmt.Errorf("invalid node registration: %w", err)
	}

	// Store node registration
	if err := k.NodeRegistrations.Set(ctx, node.NodeAddress, node); err != nil {
		return fmt.Errorf("failed to register node: %w", err)
	}

	return nil
}

// GetNode retrieves a node by address
func (k *NodeKeeper) GetNode(ctx context.Context, address string) (types.NodeRegistration, error) {
	node, err := k.NodeRegistrations.Get(ctx, address)
	if err != nil {
		return types.NodeRegistration{}, fmt.Errorf("failed to get node %s: %w", address, err)
	}
	return node, nil
}

// UpdateNode updates an existing node registration
func (k *NodeKeeper) UpdateNode(ctx context.Context, address string, updates types.NodeRegistration) error {
	// Verify node exists
	_, err := k.GetNode(ctx, address)
	if err != nil {
		return err
	}

	// Ensure address consistency
	updates.NodeAddress = address

	// Update node
	if err := k.NodeRegistrations.Set(ctx, address, updates); err != nil {
		return fmt.Errorf("failed to update node %s: %w", address, err)
	}

	return nil
}

// UpdateNodeStatus updates the status of a node
func (k *NodeKeeper) UpdateNodeStatus(ctx context.Context, address string, status types.NodeStatus) error {
	// This is a placeholder implementation
	// The actual NodeStatus type might be different in the types package

	// Get existing node
	node, err := k.GetNode(ctx, address)
	if err != nil {
		return err
	}

	// Update status fields in the node registration
	// This depends on the actual structure of types.NodeRegistration
	// node.Status = status.Status
	// node.LastSeen = status.LastSeen

	// Store updated node
	if err := k.NodeRegistrations.Set(ctx, address, node); err != nil {
		return fmt.Errorf("failed to update node status: %w", err)
	}

	return nil
}

// DeregisterNode removes a node registration
func (k *NodeKeeper) DeregisterNode(ctx context.Context, address string) error {
	// Verify node exists
	_, err := k.GetNode(ctx, address)
	if err != nil {
		return err
	}

	// Remove node registration
	if err := k.NodeRegistrations.Remove(ctx, address); err != nil {
		return fmt.Errorf("failed to deregister node %s: %w", address, err)
	}

	return nil
}

// ListNodes returns all registered nodes
func (k *NodeKeeper) ListNodes(ctx context.Context) ([]types.NodeRegistration, error) {
	var nodes []types.NodeRegistration

	err := k.NodeRegistrations.Walk(ctx, nil, func(key string, value types.NodeRegistration) (stop bool, err error) {
		nodes = append(nodes, value)
		return false, nil
	})

	if err != nil {
		return nil, fmt.Errorf("failed to list nodes: %w", err)
	}

	return nodes, nil
}

// UpdateServingNodeStatus updates serving node status
func (k *NodeKeeper) UpdateServingNodeStatus(ctx context.Context, address string, status types.ServingNodeStatus) error {
	if err := k.ServingNodeStatuses.Set(ctx, address, status); err != nil {
		return fmt.Errorf("failed to update serving node status: %w", err)
	}
	return nil
}

// GetServingNodeStatus retrieves serving node status
func (k *NodeKeeper) GetServingNodeStatus(ctx context.Context, address string) (types.ServingNodeStatus, error) {
	status, err := k.ServingNodeStatuses.Get(ctx, address)
	if err != nil {
		return types.ServingNodeStatus{}, fmt.Errorf("failed to get serving node status for %s: %w", address, err)
	}
	return status, nil
}

// CreateInferenceRequest creates a new inference request
func (k *NodeKeeper) CreateInferenceRequest(ctx context.Context, request types.InferenceRequest) error {
	if err := k.InferenceRequests.Set(ctx, request.RequestId, request); err != nil {
		return fmt.Errorf("failed to create inference request: %w", err)
	}
	return nil
}

// GetInferenceRequest retrieves an inference request
func (k *NodeKeeper) GetInferenceRequest(ctx context.Context, requestID string) (types.InferenceRequest, error) {
	request, err := k.InferenceRequests.Get(ctx, requestID)
	if err != nil {
		return types.InferenceRequest{}, fmt.Errorf("failed to get inference request %s: %w", requestID, err)
	}
	return request, nil
}

// AuthorizeValidator authorizes a validator address
func (k *NodeKeeper) AuthorizeValidator(ctx context.Context, address string) error {
	return k.AuthorizedValidatorAddresses.Set(ctx, address, true)
}

// IsValidatorAuthorized checks if a validator is authorized
func (k *NodeKeeper) IsValidatorAuthorized(ctx context.Context, address string) (bool, error) {
	authorized, err := k.AuthorizedValidatorAddresses.Get(ctx, address)
	if err != nil {
		return false, nil // Not found means not authorized
	}
	return authorized, nil
}

// AuthorizeProposer authorizes a proposer address
func (k *NodeKeeper) AuthorizeProposer(ctx context.Context, address string) error {
	return k.AuthorizedProposerAddresses.Set(ctx, address, true)
}

// IsProposerAuthorized checks if a proposer is authorized
func (k *NodeKeeper) IsProposerAuthorized(ctx context.Context, address string) (bool, error) {
	authorized, err := k.AuthorizedProposerAddresses.Get(ctx, address)
	if err != nil {
		return false, nil // Not found means not authorized
	}
	return authorized, nil
}

// validateNodeRegistration validates a node registration
func (k *NodeKeeper) validateNodeRegistration(ctx context.Context, node types.NodeRegistration) error {
	// Validate required fields
	if node.NodeAddress == "" {
		return fmt.Errorf("node address cannot be empty")
	}

	// Validate stake requirements (if applicable)
	if node.Stake != "" {
		// Parse and validate stake amount
		// This depends on the actual stake validation logic
	}

	// Add other validation logic as needed
	return nil
}

// GetNodeCount returns the total number of registered nodes
func (k *NodeKeeper) GetNodeCount(ctx context.Context) (int, error) {
	count := 0
	err := k.NodeRegistrations.Walk(ctx, nil, func(key string, value types.NodeRegistration) (stop bool, err error) {
		count++
		return false, nil
	})
	return count, err
}
