package types

import (
	"fmt"

	errorsmod "cosmossdk.io/errors"
	sdkerrors "github.com/cosmos/cosmos-sdk/types/errors"
)

// R3MES-specific error codes (string constants for error messages)
const (
	// Keeper errors (1000-1999)
	ErrInvalidGradientHashCode      = "invalid_gradient_hash"
	ErrGradientNotFoundCode         = "gradient_not_found"
	ErrInvalidAggregationCode       = "invalid_aggregation"
	ErrAggregationNotFoundCode      = "aggregation_not_found"
	ErrInvalidChallengeCode         = "invalid_challenge"
	ErrChallengeNotFoundCode        = "challenge_not_found"
	ErrInvalidMinerAddressCode      = "invalid_miner_address"
	ErrMinerNotFoundCode            = "miner_not_found"
	ErrInsufficientStakeCode        = "insufficient_stake"
	ErrInvalidModelConfigCode       = "invalid_model_config"
	ErrModelNotFoundCode            = "model_not_found"
	ErrInvalidDatasetProposalCode   = "invalid_dataset_proposal"
	ErrDatasetNotFoundCode          = "dataset_not_found"
	ErrInvalidNodeRegistrationCode  = "invalid_node_registration"
	ErrNodeNotFoundCode             = "node_not_found"
	ErrInvalidIPFSHashCode          = "invalid_ipfs_hash"
	ErrIPFSConnectionFailedCode     = "ipfs_connection_failed"
	ErrInvalidContainerHashCode     = "invalid_container_hash"
	ErrInvalidSeedCode              = "invalid_seed"
	ErrInvalidVerificationCode      = "invalid_verification"
	ErrVerificationFailedCode       = "verification_failed"
	ErrInvalidTaskPoolCode          = "invalid_task_pool"
	ErrTaskPoolNotFoundCode         = "task_pool_not_found"
	ErrInvalidChunkCode             = "invalid_chunk"
	ErrChunkNotFoundCode            = "chunk_not_found"
	ErrInvalidShardCode             = "invalid_shard"
	ErrShardNotFoundCode            = "shard_not_found"
	ErrInvalidWindowCode            = "invalid_window"
	ErrWindowNotFoundCode           = "window_not_found"
	ErrInvalidSubnetCode            = "invalid_subnet"
	ErrSubnetNotFoundCode           = "subnet_not_found"
	ErrInvalidAggregationIndexCode  = "invalid_aggregation_index"
	ErrAggregationIndexNotFoundCode = "aggregation_index_not_found"

	// Validation errors (2000-2999)
	ErrInvalidInputCode       = "invalid_input"
	ErrInvalidAddressCode     = "invalid_address"
	ErrInvalidAmountCode      = "invalid_amount"
	ErrInvalidParameterCode   = "invalid_parameter"
	ErrInvalidStateCode       = "invalid_state"
	ErrInvalidTransactionCode = "invalid_transaction"
	ErrInvalidSignatureCode   = "invalid_signature"
	ErrInvalidNonceCode       = "invalid_nonce"
	ErrInvalidTimestampCode   = "invalid_timestamp"
	ErrInvalidHeightCode      = "invalid_height"

	// Configuration errors (3000-3999)
	ErrMissingEnvironmentVariableCode = "missing_environment_variable"
	ErrInvalidConfigurationCode       = "invalid_configuration"
	ErrProductionConfigErrorCode      = "production_config_error"
	ErrLocalhostNotAllowedCode        = "localhost_not_allowed"

	// Network errors (4000-4999)
	ErrNetworkConnectionFailedCode     = "network_connection_failed"
	ErrNetworkTimeoutCode              = "network_timeout"
	ErrNetworkRequestFailedCode        = "network_request_failed"
	ErrBlockchainConnectionFailedCode  = "blockchain_connection_failed"
	ErrBlockchainQueryFailedCode       = "blockchain_query_failed"
	ErrBlockchainTransactionFailedCode = "blockchain_transaction_failed"

	// Authentication/Authorization errors (5000-5999)
	ErrAuthenticationFailedCode = "authentication_failed"
	ErrAuthorizationFailedCode  = "authorization_failed"
	ErrInvalidAPIKeyCode        = "invalid_api_key"
	ErrMissingCredentialsCode   = "missing_credentials"
	ErrInvalidCredentialsCode   = "invalid_credentials"
)

// Error variables (for direct use in errorsmod.Wrap/Wrapf)
// These are registered errors that can be used directly in error wrapping
// Using errorsmod.Register which is the standard way to register module-specific errors
var (
	ErrInvalidMiner           = errorsmod.Register(ModuleName, 1001, "invalid miner")
	ErrInvalidNodeAddress     = errorsmod.Register(ModuleName, 1002, "invalid node address")
	ErrInvalidSigner          = errorsmod.Register(ModuleName, 1003, "invalid signer")
	ErrUnauthorized           = errorsmod.Register(ModuleName, 1017, "unauthorized: role requires authorization")
	ErrInvalidModel           = errorsmod.Register(ModuleName, 1004, "invalid model")
	ErrInvalidModelVersion    = errorsmod.Register(ModuleName, 1005, "invalid model version")
	ErrInvalidVoteOption      = errorsmod.Register(ModuleName, 1006, "invalid vote option")
	ErrInvalidRequest         = errorsmod.Register(ModuleName, 1007, "invalid request")
	ErrProposalNotFound       = errorsmod.Register(ModuleName, 1008, "proposal not found")
	ErrInvalidGPUArchitecture = errorsmod.Register(ModuleName, 1009, "invalid GPU architecture")
	ErrNotImplemented         = errorsmod.Register(ModuleName, 1010, "not implemented")
	ErrInvalidChunkSize       = errorsmod.Register(ModuleName, 1011, "invalid chunk size")
	ErrInvalidGradientHash    = errorsmod.Register(ModuleName, 1012, "invalid gradient hash")
	ErrInvalidIPFSHash        = errorsmod.Register(ModuleName, 1013, "invalid IPFS hash")
	ErrInvalidChallenge       = errorsmod.Register(ModuleName, 1014, "invalid challenge")
	ErrInvalidSignature       = errorsmod.Register(ModuleName, 1015, "invalid signature")
	ErrInvalidNonce           = errorsmod.Register(ModuleName, 1016, "invalid nonce")
	ErrInsufficientStake      = errorsmod.Register(ModuleName, 1021, "insufficient stake")
	ErrDatasetNotFound        = errorsmod.Register(ModuleName, 1018, "dataset not found")
	ErrInvalidParameter       = errorsmod.Register(ModuleName, 1019, "invalid parameter")
)

// Error constructors for common errors

// ErrInvalidGradientHashf returns an error for invalid gradient hash
func ErrInvalidGradientHashf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInvalidRequest,
		"invalid gradient hash: %s", msg,
	)
}

// ErrGradientNotFoundf returns an error when gradient is not found
func ErrGradientNotFoundf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrNotFound,
		"%s", msg,
	)
}

// ErrInvalidMinerAddressf returns an error for invalid miner address
func ErrInvalidMinerAddressf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInvalidAddress,
		"%s", msg,
	)
}

// ErrInsufficientStakef returns an error when stake is insufficient
func ErrInsufficientStakef(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInsufficientFunds,
		"%s", msg,
	)
}

// ErrInvalidIPFSHashf returns an error for invalid IPFS hash
func ErrInvalidIPFSHashf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInvalidRequest,
		"%s", msg,
	)
}

// ErrIPFSConnectionFailedf returns an error when IPFS connection fails
func ErrIPFSConnectionFailedf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrUnknownRequest,
		"%s", msg,
	)
}

// ErrInvalidConfigurationf returns an error for invalid configuration
func ErrInvalidConfigurationf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInvalidRequest,
		"%s", msg,
	)
}

// ErrProductionConfigErrorf returns an error for production configuration issues
func ErrProductionConfigErrorf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInvalidRequest,
		"%s", msg,
	)
}

// ErrMissingEnvironmentVariablef returns an error when environment variable is missing
func ErrMissingEnvironmentVariablef(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInvalidRequest,
		"%s", msg,
	)
}

// ErrLocalhostNotAllowedf returns an error when localhost is used in production
func ErrLocalhostNotAllowedf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInvalidRequest,
		"%s", msg,
	)
}

// ErrBlockchainConnectionFailedf returns an error when blockchain connection fails
func ErrBlockchainConnectionFailedf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrUnknownRequest,
		"%s", msg,
	)
}

// ErrBlockchainQueryFailedf returns an error when blockchain query fails
func ErrBlockchainQueryFailedf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrUnknownRequest,
		"%s", msg,
	)
}

// ErrInvalidInputf returns an error for invalid input
func ErrInvalidInputf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInvalidRequest,
		"%s", msg,
	)
}

// ErrInvalidAddressf returns an error for invalid address
func ErrInvalidAddressf(format string, args ...interface{}) error {
	msg := fmt.Sprintf(format, args...)
	return errorsmod.Wrapf(
		sdkerrors.ErrInvalidAddress,
		"%s", msg,
	)
}

// IBC-specific errors
var (
	ErrInvalidGradientID   = errorsmod.Register(ModuleName, 1022, "invalid gradient ID")
	ErrInvalidMinerAddress = errorsmod.Register(ModuleName, 1023, "invalid miner address")
	ErrInvalidSourceChain  = errorsmod.Register(ModuleName, 1024, "invalid source chain")
)
