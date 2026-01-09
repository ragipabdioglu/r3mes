package r3mes

import "errors"

// Common errors returned by the SDK.
var (
	// ErrUserNotFound is returned when a user is not found.
	ErrUserNotFound = errors.New("user not found")

	// ErrMinerNotFound is returned when a miner is not found.
	ErrMinerNotFound = errors.New("miner not found")

	// ErrNodeNotFound is returned when a node is not found.
	ErrNodeNotFound = errors.New("node not found")

	// ErrInsufficientCredits is returned when the user has insufficient credits.
	ErrInsufficientCredits = errors.New("insufficient credits")

	// ErrInvalidWallet is returned when the wallet address is invalid.
	ErrInvalidWallet = errors.New("invalid wallet address")

	// ErrTransactionFailed is returned when a transaction fails.
	ErrTransactionFailed = errors.New("transaction failed")

	// ErrConnectionFailed is returned when connection to the network fails.
	ErrConnectionFailed = errors.New("connection failed")

	// ErrTimeout is returned when an operation times out.
	ErrTimeout = errors.New("operation timed out")

	// ErrRateLimited is returned when rate limit is exceeded.
	ErrRateLimited = errors.New("rate limit exceeded")

	// ErrInvalidResponse is returned when the server returns an invalid response.
	ErrInvalidResponse = errors.New("invalid response from server")

	// ErrBlockNotFound is returned when a block is not found.
	ErrBlockNotFound = errors.New("block not found")

	// ErrTransactionNotFound is returned when a transaction is not found.
	ErrTransactionNotFound = errors.New("transaction not found")

	// ErrProposalNotFound is returned when a governance proposal is not found.
	ErrProposalNotFound = errors.New("proposal not found")

	// ErrValidatorNotFound is returned when a validator is not found.
	ErrValidatorNotFound = errors.New("validator not found")

	// ErrDelegationNotFound is returned when a delegation is not found.
	ErrDelegationNotFound = errors.New("delegation not found")
)

// R3MESError represents an error from the R3MES network.
type R3MESError struct {
	Code    string                 `json:"code"`
	Message string                 `json:"message"`
	Details map[string]interface{} `json:"details,omitempty"`
}

// Error implements the error interface.
func (e *R3MESError) Error() string {
	if e.Code != "" {
		return "[" + e.Code + "] " + e.Message
	}
	return e.Message
}

// NewR3MESError creates a new R3MESError.
func NewR3MESError(code, message string) *R3MESError {
	return &R3MESError{
		Code:    code,
		Message: message,
	}
}
