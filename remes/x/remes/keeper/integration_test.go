package keeper_test

import (
	"context"
	"testing"
	"time"

	"cosmossdk.io/core/address"
	"cosmossdk.io/log"
	"cosmossdk.io/math"
	"cosmossdk.io/store"
	"cosmossdk.io/store/metrics"
	storetypes "cosmossdk.io/store/types"
	tmproto "github.com/cometbft/cometbft/proto/tendermint/types"
	dbm "github.com/cosmos/cosmos-db"
	"github.com/cosmos/cosmos-sdk/codec"
	codectypes "github.com/cosmos/cosmos-sdk/codec/types"
	"github.com/cosmos/cosmos-sdk/runtime"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

// mockBankKeeper implements types.BankKeeper for testing
type mockBankKeeper struct{}

func (m mockBankKeeper) SendCoins(ctx context.Context, fromAddr, toAddr sdk.AccAddress, amt sdk.Coins) error {
	return nil
}

func (m mockBankKeeper) GetBalance(ctx context.Context, addr sdk.AccAddress, denom string) sdk.Coin {
	return sdk.NewCoin(denom, math.NewInt(1000000))
}

func (m mockBankKeeper) SpendableCoins(ctx context.Context, addr sdk.AccAddress) sdk.Coins {
	return sdk.NewCoins(sdk.NewCoin("stake", math.NewInt(1000000)))
}

func (m mockBankKeeper) SendCoinsFromModuleToAccount(ctx context.Context, senderModule string, recipientAddr sdk.AccAddress, amt sdk.Coins) error {
	return nil
}

func (m mockBankKeeper) SendCoinsFromAccountToModule(ctx context.Context, senderAddr sdk.AccAddress, recipientModule string, amt sdk.Coins) error {
	return nil
}

func (m mockBankKeeper) MintCoins(ctx context.Context, moduleName string, amt sdk.Coins) error {
	return nil
}

func (m mockBankKeeper) GetSupply(ctx context.Context, denom string) sdk.Coin {
	return sdk.NewCoin(denom, math.NewInt(1000000000))
}

func (m mockBankKeeper) BurnCoins(ctx context.Context, moduleName string, amt sdk.Coins) error {
	return nil
}

// mockAuthKeeper implements types.AuthKeeper for testing
type mockAuthKeeper struct{}

func (m mockAuthKeeper) GetAccount(ctx context.Context, addr sdk.AccAddress) sdk.AccountI {
	return nil
}

func (m mockAuthKeeper) AddressCodec() address.Codec {
	return mockAddressCodec{}
}

// mockAddressCodec implements address.Codec for testing
type mockAddressCodec struct{}

func (m mockAddressCodec) StringToBytes(text string) ([]byte, error) {
	return []byte(text), nil
}

func (m mockAddressCodec) BytesToString(bz []byte) (string, error) {
	return string(bz), nil
}

func setupKeeper(t *testing.T) (keeper.Keeper, context.Context) {
	// Create store key
	storeKey := storetypes.NewKVStoreKey(types.StoreKey)

	// Create in-memory store
	db := dbm.NewMemDB()
	stateStore := store.NewCommitMultiStore(db, log.NewNopLogger(), metrics.NewNoOpMetrics())
	stateStore.MountStoreWithDB(storeKey, storetypes.StoreTypeIAVL, db)
	require.NoError(t, stateStore.LoadLatestVersion())

	// Create store service
	storeService := runtime.NewKVStoreService(storeKey)

	// Create codec
	interfaceRegistry := codectypes.NewInterfaceRegistry()
	types.RegisterInterfaces(interfaceRegistry)
	cdc := codec.NewProtoCodec(interfaceRegistry)

	// Create address codec
	addressCodec := mockAddressCodec{}

	// Create mock keepers
	bankKeeper := mockBankKeeper{}
	authKeeper := mockAuthKeeper{}

	// Create authority
	authority := []byte("authority")

	// Create keeper
	k, err := keeper.NewKeeper(
		storeService,
		cdc,
		addressCodec,
		authority,
		bankKeeper,
		authKeeper,
		"http://localhost:5001", // IPFS URL
	)
	require.NoError(t, err)

	// Create context with proper multistore
	ctx := sdk.NewContext(stateStore, tmproto.Header{Time: time.Now()}, false, log.NewNopLogger())

	return k, ctx
}

func TestKeeperBasicFunctionality(t *testing.T) {
	k, ctx := setupKeeper(t)

	// Test params
	params := types.DefaultParams()
	err := k.SetParams(ctx, params)
	require.NoError(t, err)

	retrievedParams, err := k.GetParams(ctx)
	require.NoError(t, err)
	require.Equal(t, params, retrievedParams)
}

func TestModelKeeper(t *testing.T) {
	k, ctx := setupKeeper(t)

	// Test model registration
	model := types.ModelRegistry{
		ModelId: 0, // Will be set by keeper
		Config: types.ModelConfig{
			ModelType:          types.ModelType_MODEL_TYPE_BITNET,
			ModelVersion:       "v1.0.0",
			ArchitectureConfig: `{"hidden_size": 768}`,
			ContainerHash:      "test-hash",
			ContainerRegistry:  "docker.io",
		},
		IsActive:  true,
		CreatedAt: time.Now(),
	}

	err := k.GetModelKeeper().RegisterModel(ctx, model)
	require.NoError(t, err)

	// Test model retrieval - ID starts from 0 after sequence increment
	retrievedModel, err := k.GetModelKeeper().GetModel(ctx, 0)
	require.NoError(t, err)
	require.Equal(t, uint64(0), retrievedModel.ModelId)
	require.Equal(t, types.ModelType_MODEL_TYPE_BITNET, retrievedModel.Config.ModelType)
}

func TestTrainingKeeper(t *testing.T) {
	k, ctx := setupKeeper(t)

	// Test gradient submission
	gradient := types.StoredGradient{
		Miner:           "test-miner",
		IpfsHash:        "QmGradientHash",
		ModelVersion:    "v1.0.0",
		TrainingRoundId: 1,
		ShardId:         0,
		GradientHash:    "hash123",
		GpuArchitecture: "cuda",
	}

	err := k.GetTrainingKeeper().SubmitGradient(ctx, gradient)
	require.NoError(t, err)

	// Test gradient retrieval - ID starts from 0
	retrievedGradient, err := k.GetTrainingKeeper().GetGradient(ctx, 0)
	require.NoError(t, err)
	require.Equal(t, uint64(0), retrievedGradient.Id)
	require.Equal(t, "test-miner", retrievedGradient.Miner)
}

func TestDatasetKeeper(t *testing.T) {
	k, ctx := setupKeeper(t)

	// Test dataset proposal
	proposal := types.DatasetProposal{
		Proposer:        "test-proposer",
		DatasetIpfsHash: "QmDatasetHash",
		// Note: Description field may not exist in current proto definition
	}

	err := k.GetDatasetKeeper().ProposeDataset(ctx, proposal)
	require.NoError(t, err)

	// Test proposal retrieval - ID starts from 0
	retrievedProposal, err := k.GetDatasetKeeper().GetDatasetProposal(ctx, 0)
	require.NoError(t, err)
	require.Equal(t, uint64(0), retrievedProposal.ProposalId)
	require.Equal(t, "test-proposer", retrievedProposal.Proposer)
}

func TestNodeKeeper(t *testing.T) {
	k, ctx := setupKeeper(t)

	// Test node registration
	node := types.NodeRegistration{
		NodeAddress: "test-node-address",
		NodeType:    types.NODE_TYPE_MINING, // Use proper enum value
		Stake:       "1000000stake",
	}

	err := k.GetNodeKeeper().RegisterNode(ctx, node)
	require.NoError(t, err)

	// Test node retrieval
	retrievedNode, err := k.GetNodeKeeper().GetNode(ctx, "test-node-address")
	require.NoError(t, err)
	require.Equal(t, "test-node-address", retrievedNode.NodeAddress)
	require.Equal(t, types.NODE_TYPE_MINING, retrievedNode.NodeType)
}

func TestMsgServer(t *testing.T) {
	k, ctx := setupKeeper(t)

	// Create message server
	msgServer := keeper.NewMsgServerImpl(k)

	// Test UpdateParams
	updateParamsMsg := &types.MsgUpdateParams{
		Authority: string(k.GetAuthority()),
		Params:    types.DefaultParams(),
	}

	_, err := msgServer.UpdateParams(ctx, updateParamsMsg)
	require.NoError(t, err)
}

func TestQueryServer(t *testing.T) {
	k, ctx := setupKeeper(t)

	// Create query server
	queryServer := keeper.NewQueryServerImpl(k)

	// Test Params query
	paramsReq := &types.QueryParamsRequest{}

	_, err := queryServer.Params(ctx, paramsReq)
	require.NoError(t, err)
}
