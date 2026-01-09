package keeper

import (
	"fmt"

	sdkerrors "cosmossdk.io/errors"
)

// Production-ready error handling for R3MES keeper
// Replaces panic-based error handling with proper error types

var (
	// ErrCodespace defines the module error codespace
	ErrCodespace = "remes"

	// Core errors (1000-1099)
	ErrInvalidAuthority     = sdkerrors.Register(ErrCodespace, 1000, "invalid authority")
	ErrInvalidParams        = sdkerrors.Register(ErrCodespace, 1001, "invalid parameters")
	ErrUnauthorized         = sdkerrors.Register(ErrCodespace, 1002, "unauthorized")
	ErrInvalidAddress       = sdkerrors.Register(ErrCodespace, 1003, "invalid address")
	ErrInvalidAmount        = sdkerrors.Register(ErrCodespace, 1004, "invalid amount")

	// Model errors (1100-1199)
	ErrModelNotFound        = sdkerrors.Register(ErrCodespace, 1100, "model not found")
	ErrModelAlreadyExists   = sdkerrors.Register(ErrCodespace, 1101, "model already exists")
	ErrInvalidModelID       = sdkerrors.Register(ErrCodespace, 1102, "invalid model ID")
	ErrModelNotActive       = sdkerrors.Register(ErrCodespace, 1103, "model not active")
	ErrInvalidModelVersion  = sdkerrors.Register(ErrCodespace, 1104, "invalid model version")

	// Training errors (1200-1299)
	ErrGradientNotFound     = sdkerrors.Register(ErrCodespace, 1200, "gradient not found")
	ErrInvalidGradient      = sdkerrors.Register(ErrCodespace, 1201, "invalid gradient")
	ErrInvalidIPFSHash      = sdkerrors.Register(ErrCodespace, 1202, "invalid IPFS hash")
	ErrAggregationNotFound  = sdkerrors.Register(ErrCodespace, 1203, "aggregation not found")
	ErrInvalidAggregation   = sdkerrors.Register(ErrCodespace, 1204, "invalid aggregation")
	ErrTrainingWindowClosed = sdkerrors.Register(ErrCodespace, 1205, "training window closed")
	ErrDuplicateGradient    = sdkerrors.Register(ErrCodespace, 1206, "duplicate gradient submission")

	// Dataset errors (1300-1399)
	ErrDatasetNotFound      = sdkerrors.Register(ErrCodespace, 1300, "dataset not found")
	ErrInvalidDataset       = sdkerrors.Register(ErrCodespace, 1301, "invalid dataset")
	ErrDatasetNotApproved   = sdkerrors.Register(ErrCodespace, 1302, "dataset not approved")
	ErrInvalidProposal      = sdkerrors.Register(ErrCodespace, 1303, "invalid proposal")
	ErrProposalNotFound     = sdkerrors.Register(ErrCodespace, 1304, "proposal not found")

	// Node errors (1400-1499)
	ErrNodeNotFound         = sdkerrors.Register(ErrCodespace, 1400, "node not found")
	ErrNodeAlreadyExists    = sdkerrors.Register(ErrCodespace, 1401, "node already exists")
	ErrInvalidNodeType      = sdkerrors.Register(ErrCodespace, 1402, "invalid node type")
	ErrNodeNotActive        = sdkerrors.Register(ErrCodespace, 1403, "node not active")
	ErrInsufficientStake    = sdkerrors.Register(ErrCodespace, 1404, "insufficient stake")
	ErrValidatorNotFound    = sdkerrors.Register(ErrCodespace, 1405, "validator not found")
	ErrProposerNotFound     = sdkerrors.Register(ErrCodespace, 1406, "proposer not found")

	// Economics errors (1500-1599)
	ErrInsufficientFunds    = sdkerrors.Register(ErrCodespace, 1500, "insufficient funds")
	ErrInvalidReward        = sdkerrors.Register(ErrCodespace, 1501, "invalid reward")
	ErrTreasuryNotFound     = sdkerrors.Register(ErrCodespace, 1502, "treasury not found")
	ErrInvalidTreasury      = sdkerrors.Register(ErrCodespace, 1503, "invalid treasury")

	// Security errors (1600-1699)
	ErrInvalidSignature     = sdkerrors.Register(ErrCodespace, 1600, "invalid signature")
	ErrInvalidNonce         = sdkerrors.Register(ErrCodespace, 1601, "invalid nonce")
	ErrFraudDetected        = sdkerrors.Register(ErrCodespace, 1602, "fraud detected")
	ErrChallengeNotFound    = sdkerrors.Register(ErrCodespace, 1603, "challenge not found")
	ErrInvalidChallenge     = sdkerrors.Register(ErrCodespace, 1604, "invalid challenge")
	ErrTrapJobNotFound      = sdkerrors.Register(ErrCodespace, 1605, "trap job not found")

	// Infrastructure errors (1700-1799)
	ErrIPFSConnectionFailed = sdkerrors.Register(ErrCodespace, 1700, "IPFS connection failed")
	ErrIPFSVerificationFailed = sdkerrors.Register(ErrCodespace, 1701, "IPFS verification failed")
	ErrCacheNotFound        = sdkerrors.Register(ErrCodespace, 1702, "cache not found")
	ErrCacheWriteFailed     = sdkerrors.Register(ErrCodespace, 1703, "cache write failed")
)

// WrapError wraps an error with additional context
func WrapError(err error, format string, args ...interface{}) error {
	if err == nil {
		return nil
	}
	msg := fmt.Sprintf(format, args...)
	return fmt.Errorf("%s: %w", msg, err)
}

// IsNotFoundError checks if error is a "not found" type error
func IsNotFoundError(err error) bool {
	if err == nil {
		return false
	}
	return sdkerrors.IsOf(err,
		ErrModelNotFound,
		ErrGradientNotFound,
		ErrAggregationNotFound,
		ErrDatasetNotFound,
		ErrNodeNotFound,
		ErrValidatorNotFound,
		ErrProposerNotFound,
		ErrTreasuryNotFound,
		ErrChallengeNotFound,
		ErrTrapJobNotFound,
		ErrCacheNotFound,
	)
}

// IsInvalidError checks if error is an "invalid" type error
func IsInvalidError(err error) bool {
	if err == nil {
		return false
	}
	return sdkerrors.IsOf(err,
		ErrInvalidParams,
		ErrInvalidAddress,
		ErrInvalidAmount,
		ErrInvalidModelID,
		ErrInvalidModelVersion,
		ErrInvalidGradient,
		ErrInvalidIPFSHash,
		ErrInvalidAggregation,
		ErrInvalidDataset,
		ErrInvalidProposal,
		ErrInvalidNodeType,
		ErrInvalidReward,
		ErrInvalidTreasury,
		ErrInvalidSignature,
		ErrInvalidNonce,
		ErrInvalidChallenge,
	)
}

// IsUnauthorizedError checks if error is an authorization error
func IsUnauthorizedError(err error) bool {
	if err == nil {
		return false
	}
	return sdkerrors.IsOf(err,
		ErrUnauthorized,
		ErrInvalidAuthority,
	)
}

// IsFraudError checks if error is a fraud-related error
func IsFraudError(err error) bool {
	if err == nil {
		return false
	}
	return sdkerrors.IsOf(err, ErrFraudDetected)
}
