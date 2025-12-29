package keeper_test

import (
	"context"
	"os"
	"testing"

	"cosmossdk.io/core/address"
	"cosmossdk.io/math"
	storetypes "cosmossdk.io/store/types"
	addresscodec "github.com/cosmos/cosmos-sdk/codec/address"
	cryptotypes "github.com/cosmos/cosmos-sdk/crypto/types"
	"github.com/cosmos/cosmos-sdk/runtime"
	"github.com/cosmos/cosmos-sdk/testutil"
	sdk "github.com/cosmos/cosmos-sdk/types"
	moduletestutil "github.com/cosmos/cosmos-sdk/types/module/testutil"
	authtypes "github.com/cosmos/cosmos-sdk/x/auth/types"

	"remes/x/remes/keeper"
	module "remes/x/remes/module"
	"remes/x/remes/types"
)

// mockBankKeeper is a mock implementation of BankKeeper for testing
type mockBankKeeper struct{}

func (m mockBankKeeper) MintCoins(ctx context.Context, moduleName string, amt sdk.Coins) error {
	return nil
}

func (m mockBankKeeper) SendCoinsFromModuleToAccount(ctx context.Context, senderModule string, recipientAddr sdk.AccAddress, amt sdk.Coins) error {
	return nil
}

func (m mockBankKeeper) SendCoinsFromAccountToModule(ctx context.Context, senderAddr sdk.AccAddress, recipientModule string, amt sdk.Coins) error {
	return nil
}

func (m mockBankKeeper) BurnCoins(ctx context.Context, moduleName string, amt sdk.Coins) error {
	return nil
}

func (m mockBankKeeper) GetBalance(ctx context.Context, addr sdk.AccAddress, denom string) sdk.Coin {
	// Return sufficient balance for testing (meets staking requirement)
	// Minimum stake requirement is 1000 tokens
	if denom == "remes" || denom == "stake" {
		return sdk.NewCoin(denom, math.NewInt(10000)) // 10000 tokens for testing
	}
	return sdk.NewCoin(denom, math.ZeroInt())
}

func (m mockBankKeeper) SpendableCoins(ctx context.Context, addr sdk.AccAddress) sdk.Coins {
	// Return sufficient coins for testing
	return sdk.NewCoins(
		sdk.NewCoin("remes", math.NewInt(10000)),
		sdk.NewCoin("stake", math.NewInt(10000)),
	)
}

func (m mockBankKeeper) GetSupply(ctx context.Context, denom string) sdk.Coin {
	// Return mock supply for testing
	if denom == "remes" || denom == "stake" {
		return sdk.NewCoin(denom, math.NewInt(1000000)) // 1M tokens total supply
	}
	return sdk.NewCoin(denom, math.ZeroInt())
}

// mockAuthKeeper is a mock implementation of AuthKeeper for testing
type mockAuthKeeper struct {
	addressCodec address.Codec
}

func (m mockAuthKeeper) AddressCodec() address.Codec {
	return m.addressCodec
}

func (m mockAuthKeeper) GetAccount(ctx context.Context, addr sdk.AccAddress) sdk.AccountI {
	// Return a BaseAccount with a mock pubkey that always verifies signatures
	// Use authtypes.BaseAccount for proper interface implementation
	baseAcc := authtypes.NewBaseAccount(addr, nil, 0, 0)

	// Create a mock pubkey that always verifies
	mockPubKey := &alwaysVerifyPubKey{}
	baseAcc.SetPubKey(mockPubKey)

	return baseAcc
}

// alwaysVerifyPubKey is a mock public key that always verifies signatures
type alwaysVerifyPubKey struct {
	cryptotypes.PubKey
}

func (p *alwaysVerifyPubKey) Address() cryptotypes.Address {
	return cryptotypes.Address("mock_pubkey_address")
}

func (p *alwaysVerifyPubKey) Bytes() []byte {
	return []byte("mock_pubkey_bytes")
}

func (p *alwaysVerifyPubKey) VerifySignature(msg []byte, sig []byte) bool {
	// Always return true for testing
	return true
}

func (p *alwaysVerifyPubKey) Equals(other cryptotypes.PubKey) bool {
	return true
}

func (p *alwaysVerifyPubKey) Type() string {
	return "mock"
}

type fixture struct {
	ctx          context.Context
	keeper       keeper.Keeper
	addressCodec address.Codec
}

func initFixture(t *testing.T) *fixture {
	t.Helper()

	encCfg := moduletestutil.MakeTestEncodingConfig(module.AppModule{})
	addressCodec := addresscodec.NewBech32Codec(sdk.GetConfig().GetBech32AccountAddrPrefix())
	storeKey := storetypes.NewKVStoreKey(types.StoreKey)

	storeService := runtime.NewKVStoreService(storeKey)
	testCtx := testutil.DefaultContextWithDB(t, storeKey, storetypes.NewTransientStoreKey("transient_test"))
	// Set block height to 1 for tests (needed for DeriveGlobalSeed)
	sdkCtx := testCtx.Ctx.WithBlockHeight(1)
	ctx := sdkCtx

	authority := authtypes.NewModuleAddress(types.GovModuleName)

	bankKeeper := mockBankKeeper{}
	authKeeper := mockAuthKeeper{addressCodec: addressCodec}

	// For testing, set R3MES_TEST_MODE=true to bypass security validation
	// This allows tests to run without IPFS daemon
	os.Setenv("R3MES_TEST_MODE", "true")
	defer os.Unsetenv("R3MES_TEST_MODE")

	// For testing, use empty IPFS URL to skip IPFS verification
	// This allows tests to run without IPFS daemon
	k := keeper.NewKeeper(
		storeService,
		encCfg.Codec,
		addressCodec,
		authority,
		bankKeeper,
		authKeeper,
		"", // Empty IPFS URL - skips IPFS verification in tests (allowed in test mode)
	)

	// Initialize params
	if err := k.Params.Set(ctx, types.DefaultParams()); err != nil {
		t.Fatalf("failed to set params: %v", err)
	}

	return &fixture{
		ctx:          ctx,
		keeper:       k,
		addressCodec: addressCodec,
	}
}
