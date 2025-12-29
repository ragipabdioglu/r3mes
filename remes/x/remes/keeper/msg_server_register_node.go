package keeper

import (
	"context"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"
	sdkmath "cosmossdk.io/math"

	"remes/x/remes/types"
)

// RegisterNode handles MsgRegisterNode
func (k msgServer) RegisterNode(ctx context.Context, msg *types.MsgRegisterNode) (*types.MsgRegisterNodeResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate node address
	_, err := sdk.AccAddressFromBech32(msg.NodeAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid node address")
	}

	// Check if node is already registered
	_, err = k.NodeRegistrations.Get(sdkCtx, msg.NodeAddress)
	if err == nil {
		return nil, errorsmod.Wrap(err, "node already registered")
	}

	// Validate roles
	if len(msg.Roles) == 0 {
		return nil, errorsmod.Wrap(err, "at least one role is required")
	}

	// Validate role allocations match roles
	if len(msg.RoleAllocations) != len(msg.Roles) {
		return nil, errorsmod.Wrap(err, "role allocations must match roles")
	}

	// Validate stake
	stakeCoins, err := sdk.ParseCoinsNormalized(msg.Stake)
	if err != nil {
		return nil, errorsmod.Wrap(err, "invalid stake")
	}

	// Role-specific access control and minimum stake requirements
	for _, role := range msg.Roles {
		if role == types.NODE_TYPE_VALIDATOR {
			// Validator role: Check authorization (whitelist)
			authorized, err := k.AuthorizedValidatorAddresses.Get(sdkCtx, msg.NodeAddress)
			if err != nil || !authorized {
				// Check if address is module authority (for genesis validators)
				authority := sdk.AccAddress(k.GetAuthority()).String()
				if msg.NodeAddress != authority {
					return nil, errorsmod.Wrapf(
						types.ErrUnauthorized,
						"validator role requires authorization (whitelist or governance approval). Address: %s",
						msg.NodeAddress,
					)
				}
			}
			
			// Validator requires higher minimum stake: 100,000 REMES
			minValidatorStake := sdkmath.NewInt(100000)
			if stakeCoins.AmountOf("remes").LT(minValidatorStake) {
				return nil, errorsmod.Wrapf(
					types.ErrInsufficientStake,
					"validator role requires minimum 100000remes stake, got %s",
					msg.Stake,
				)
			}
		}
		
		if role == types.NODE_TYPE_PROPOSER {
			// Proposer role: Check if authorized OR if node has validator role
			authorized, err := k.AuthorizedProposerAddresses.Get(sdkCtx, msg.NodeAddress)
			if err != nil || !authorized {
				// Check if node also has validator role (validators can be proposers)
				hasValidatorRole := false
				for _, r := range msg.Roles {
					if r == types.NODE_TYPE_VALIDATOR {
						hasValidatorRole = true
						break
					}
				}
				if !hasValidatorRole {
					return nil, errorsmod.Wrapf(
						types.ErrUnauthorized,
						"proposer role requires validator role or authorization (whitelist). Address: %s",
						msg.NodeAddress,
					)
				}
			}
			
			// Proposer requires minimum stake: 50,000 REMES
			minProposerStake := sdkmath.NewInt(50000)
			if stakeCoins.AmountOf("remes").LT(minProposerStake) {
				return nil, errorsmod.Wrapf(
					types.ErrInsufficientStake,
					"proposer role requires minimum 50000remes stake, got %s",
					msg.Stake,
				)
			}
		}
	}

	// Validate resource proof IPFS hash
	if msg.ResourceProofIpfsHash != "" {
		if err := k.VerifyIPFSContentExists(sdkCtx, msg.ResourceProofIpfsHash); err != nil {
			return nil, err
		}
	}

	// Create node registration
	registration := types.NodeRegistration{
		NodeAddress:          msg.NodeAddress,
		NodeType:             msg.NodeType,
		Resources:            msg.Resources,
		Stake:                msg.Stake,
		Status:               types.NODE_STATUS_ACTIVE,
		Roles:                msg.Roles,
		RoleAllocations:       msg.RoleAllocations,
		RegisteredAtHeight:   sdkCtx.BlockHeight(),
		RegisteredAtTime:     sdkCtx.BlockTime(),
		ResourceProofIpfsHash: msg.ResourceProofIpfsHash,
		LastHeartbeatHeight:   sdkCtx.BlockHeight(),
		SlashingEvents:        0,
	}

	// Store registration
	if err := k.NodeRegistrations.Set(sdkCtx, msg.NodeAddress, registration); err != nil {
		return nil, err
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeRegisterNode,
			sdk.NewAttribute(types.AttributeKeyNodeAddress, msg.NodeAddress),
			sdk.NewAttribute(types.AttributeKeyNodeType, msg.NodeType.String()),
		),
	)

	return &types.MsgRegisterNodeResponse{
		// Use block height as registration ID (unique per registration since each node can only register once)
		// This provides a deterministic and unique identifier without requiring a separate counter
		RegistrationId: uint64(sdkCtx.BlockHeight()),
	}, nil
}

// UpdateNodeRegistration handles MsgUpdateNodeRegistration
func (k msgServer) UpdateNodeRegistration(ctx context.Context, msg *types.MsgUpdateNodeRegistration) (*types.MsgUpdateNodeRegistrationResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate node address
	_, err := sdk.AccAddressFromBech32(msg.NodeAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid node address")
	}

	// Get existing registration
	registration, err := k.NodeRegistrations.Get(sdkCtx, msg.NodeAddress)
	if err != nil {
		return nil, errorsmod.Wrap(err, "node not registered")
	}

	// Update resources if provided
	if msg.Resources != nil && msg.Resources.CpuCores > 0 {
		registration.Resources = *msg.Resources
	}

	// Update roles if provided
	if len(msg.Roles) > 0 {
		// Validate role-specific access control for new roles
		for _, role := range msg.Roles {
			if role == types.NODE_TYPE_VALIDATOR {
				// Check if already has validator role (no need to re-authorize)
				hasExistingValidatorRole := false
				for _, existingRole := range registration.Roles {
					if existingRole == types.NODE_TYPE_VALIDATOR {
						hasExistingValidatorRole = true
						break
					}
				}
				
				// Only check authorization if this is a new validator role
				if !hasExistingValidatorRole {
					authorized, err := k.AuthorizedValidatorAddresses.Get(sdkCtx, msg.NodeAddress)
					if err != nil || !authorized {
						authority := sdk.AccAddress(k.GetAuthority()).String()
						if msg.NodeAddress != authority {
							return nil, errorsmod.Wrapf(
								types.ErrUnauthorized,
								"validator role requires authorization (whitelist or governance approval). Address: %s",
								msg.NodeAddress,
							)
						}
					}
				}
			}
			
			if role == types.NODE_TYPE_PROPOSER {
				// Check if already has proposer role
				hasExistingProposerRole := false
				for _, existingRole := range registration.Roles {
					if existingRole == types.NODE_TYPE_PROPOSER {
						hasExistingProposerRole = true
						break
					}
				}
				
				// Only check authorization if this is a new proposer role
				if !hasExistingProposerRole {
					authorized, err := k.AuthorizedProposerAddresses.Get(sdkCtx, msg.NodeAddress)
					if err != nil || !authorized {
						// Check if node also has validator role
						hasValidatorRole := false
						for _, r := range msg.Roles {
							if r == types.NODE_TYPE_VALIDATOR {
								hasValidatorRole = true
								break
							}
						}
						// Also check existing roles
						if !hasValidatorRole {
							for _, r := range registration.Roles {
								if r == types.NODE_TYPE_VALIDATOR {
									hasValidatorRole = true
									break
								}
							}
						}
						if !hasValidatorRole {
							return nil, errorsmod.Wrapf(
								types.ErrUnauthorized,
								"proposer role requires validator role or authorization (whitelist). Address: %s",
								msg.NodeAddress,
							)
						}
					}
				}
			}
		}
		
		registration.Roles = msg.Roles
	}

	// Update role allocations if provided
	if len(msg.RoleAllocations) > 0 {
		registration.RoleAllocations = msg.RoleAllocations
	}

	// Update resource proof if provided
	if msg.ResourceProofIpfsHash != "" {
		if err := k.VerifyIPFSContentExists(sdkCtx, msg.ResourceProofIpfsHash); err != nil {
			return nil, err
		}
		registration.ResourceProofIpfsHash = msg.ResourceProofIpfsHash
	}

	// Store updated registration
	if err := k.NodeRegistrations.Set(sdkCtx, msg.NodeAddress, registration); err != nil {
		return nil, err
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeUpdateNodeRegistration,
			sdk.NewAttribute(types.AttributeKeyNodeAddress, msg.NodeAddress),
		),
	)

	return &types.MsgUpdateNodeRegistrationResponse{}, nil
}

