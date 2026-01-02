package core

import (
	"context"
	"errors"
	"fmt"

	"cosmossdk.io/collections"
	"cosmossdk.io/core/address"
	corestore "cosmossdk.io/core/store"
	"github.com/cosmos/cosmos-sdk/codec"

	"remes/x/remes/types"
)

// CoreKeeper handles core blockchain functionality
type CoreKeeper struct {
	storeService corestore.KVStoreService
	cdc          codec.Codec
	addressCodec address.Codec
	authority    []byte

	// Bank keeper for token operations
	bankKeeper types.BankKeeper
	// Auth keeper for account verification
	authKeeper types.AuthKeeper

	Schema collections.Schema

	// Core collections
	Params            collections.Item[types.Params]
	BlockTimestamps   collections.Map[int64, int64]
	UsedNonces        collections.Map[string, bool]
	NonceWindows      collections.Map[string, string]
	SubmissionHistory collections.Map[string, uint64]
}

// NewCoreKeeper creates a new core keeper
func NewCoreKeeper(
	storeService corestore.KVStoreService,
	cdc codec.Codec,
	addressCodec address.Codec,
	authority []byte,
	bankKeeper types.BankKeeper,
	authKeeper types.AuthKeeper,
) (*CoreKeeper, error) {
	if _, err := addressCodec.BytesToString(authority); err != nil {
		return nil, fmt.Errorf("invalid authority address: %w", err)
	}

	sb := collections.NewSchemaBuilder(storeService)

	k := &CoreKeeper{
		storeService: storeService,
		cdc:          cdc,
		addressCodec: addressCodec,
		authority:    authority,
		bankKeeper:   bankKeeper,
		authKeeper:   authKeeper,

		Params:            collections.NewItem(sb, types.ParamsKey, "params", codec.CollValue[types.Params](cdc)),
		BlockTimestamps:   collections.NewMap(sb, types.BlockTimestampsKey, "block_timestamps", collections.Int64Key, collections.Int64Value),
		UsedNonces:        collections.NewMap(sb, types.UsedNonceKey, "used_nonces", collections.StringKey, collections.BoolValue),
		NonceWindows:      collections.NewMap(sb, types.NonceWindowKey, "nonce_windows", collections.StringKey, collections.StringValue),
		SubmissionHistory: collections.NewMap(sb, types.SubmissionHistoryKey, "submission_history", collections.StringKey, collections.Uint64Value),
	}

	schema, err := sb.Build()
	if err != nil {
		return nil, fmt.Errorf("failed to build core keeper schema: %w", err)
	}
	k.Schema = schema

	return k, nil
}

// GetAuthority returns the module's authority
func (k *CoreKeeper) GetAuthority() []byte {
	return k.authority
}

// GetParams returns the current module parameters
func (k *CoreKeeper) GetParams(ctx context.Context) (types.Params, error) {
	params, err := k.Params.Get(ctx)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return types.DefaultParams(), nil
		}
		return types.Params{}, err
	}
	return params, nil
}

// SetParams sets the module parameters
func (k *CoreKeeper) SetParams(ctx context.Context, params types.Params) error {
	if err := params.Validate(); err != nil {
		return fmt.Errorf("invalid params: %w", err)
	}
	return k.Params.Set(ctx, params)
}

// GetSchema returns the collections schema
func (k *CoreKeeper) GetSchema() collections.Schema {
	return k.Schema
}

// GetStoreService returns the store service
func (k *CoreKeeper) GetStoreService() corestore.KVStoreService {
	return k.storeService
}

// GetCodec returns the codec
func (k *CoreKeeper) GetCodec() codec.Codec {
	return k.cdc
}

// GetAddressCodec returns the address codec
func (k *CoreKeeper) GetAddressCodec() address.Codec {
	return k.addressCodec
}

// GetBankKeeper returns the bank keeper
func (k *CoreKeeper) GetBankKeeper() types.BankKeeper {
	return k.bankKeeper
}

// GetAuthKeeper returns the auth keeper
func (k *CoreKeeper) GetAuthKeeper() types.AuthKeeper {
	return k.authKeeper
}

// StoreBlockTimestamp stores a block timestamp
func (k *CoreKeeper) StoreBlockTimestamp(ctx context.Context, height int64, timestamp int64) error {
	return k.BlockTimestamps.Set(ctx, height, timestamp)
}

// GetBlockTimestamp gets a block timestamp
func (k *CoreKeeper) GetBlockTimestamp(ctx context.Context, height int64) (int64, error) {
	return k.BlockTimestamps.Get(ctx, height)
}

// CleanupOldBlockTimestamps removes old block timestamps (keep only last 100 blocks)
func (k *CoreKeeper) CleanupOldBlockTimestamps(ctx context.Context, currentHeight int64) error {
	if currentHeight <= 100 {
		return nil
	}

	oldHeight := currentHeight - 100
	return k.BlockTimestamps.Remove(ctx, oldHeight)
}

// Nonce management methods
func (k *CoreKeeper) IsNonceUsed(ctx context.Context, key string) (bool, error) {
	used, err := k.UsedNonces.Get(ctx, key)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return false, nil
		}
		return false, err
	}
	return used, nil
}

func (k *CoreKeeper) MarkNonceAsUsed(ctx context.Context, key string) error {
	return k.UsedNonces.Set(ctx, key, true)
}

func (k *CoreKeeper) GetNonceWindow(ctx context.Context, key string) (string, error) {
	return k.NonceWindows.Get(ctx, key)
}

func (k *CoreKeeper) SetNonceWindow(ctx context.Context, key string, window string) error {
	return k.NonceWindows.Set(ctx, key, window)
}

func (k *CoreKeeper) RemoveNonce(ctx context.Context, key string) error {
	return k.UsedNonces.Remove(ctx, key)
}

// Submission history methods
func (k *CoreKeeper) GetSubmissionCount(ctx context.Context, key string) (uint64, error) {
	count, err := k.SubmissionHistory.Get(ctx, key)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return 0, nil
		}
		return 0, err
	}
	return count, nil
}

func (k *CoreKeeper) IncrementSubmissionCount(ctx context.Context, key string) error {
	count, err := k.GetSubmissionCount(ctx, key)
	if err != nil {
		return err
	}
	return k.SubmissionHistory.Set(ctx, key, count+1)
}

func (k *CoreKeeper) SetSubmissionCount(ctx context.Context, key string, count uint64) error {
	return k.SubmissionHistory.Set(ctx, key, count)
}
