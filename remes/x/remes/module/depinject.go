package remes

import (
	"fmt"
	"os"

	"cosmossdk.io/core/address"
	"cosmossdk.io/core/appmodule"
	"cosmossdk.io/core/store"
	"cosmossdk.io/depinject"
	"cosmossdk.io/depinject/appconfig"
	"github.com/cosmos/cosmos-sdk/codec"
	authtypes "github.com/cosmos/cosmos-sdk/x/auth/types"
	capabilitykeeper "github.com/cosmos/ibc-go/modules/capability/keeper"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

var _ depinject.OnePerModuleType = AppModule{}

// IsOnePerModuleType implements the depinject.OnePerModuleType interface.
func (AppModule) IsOnePerModuleType() {}

func init() {
	appconfig.Register(
		&types.Module{},
		appconfig.Provide(ProvideModule),
	)
}

type ModuleInputs struct {
	depinject.In

	Config       *types.Module
	StoreService store.KVStoreService
	Cdc          codec.Codec
	AddressCodec address.Codec

	AuthKeeper       types.AuthKeeper
	BankKeeper       types.BankKeeper
	CapabilityKeeper *capabilitykeeper.Keeper `optional:"true"`
}

type ModuleOutputs struct {
	depinject.Out

	RemesKeeper keeper.Keeper
	Module      appmodule.AppModule
}

func ProvideModule(in ModuleInputs) (ModuleOutputs, error) {
	// default to governance authority if not provided
	authority := authtypes.NewModuleAddress(types.GovModuleName)
	if in.Config.Authority != "" {
		authority = authtypes.NewModuleAddressOrBech32Address(in.Config.Authority)
	}

	// IPFS API URL priority:
	// 1. Environment variable (IPFS_API_URL)
	// 2. Module config (in.Config.IpfsApiUrl)
	// 3. Default (http://127.0.0.1:5001)
	ipfsAPIURL := os.Getenv("IPFS_API_URL")
	if ipfsAPIURL == "" {
		if in.Config.IpfsApiUrl != "" {
			ipfsAPIURL = in.Config.IpfsApiUrl
		} else {
			ipfsAPIURL = "http://127.0.0.1:5001"
		}
	}

	// Get scoped keeper for IBC (if capability keeper is available)
	var scopedKeeper capabilitykeeper.ScopedKeeper
	if in.CapabilityKeeper != nil {
		scopedKeeper = in.CapabilityKeeper.ScopeToModule(types.ModuleName)
	}

	k, err := keeper.NewKeeper(
		in.StoreService,
		in.Cdc,
		in.AddressCodec,
		authority,
		in.BankKeeper,
		in.AuthKeeper,
		ipfsAPIURL,
		in.CapabilityKeeper,
		scopedKeeper,
	)
	if err != nil {
		return ModuleOutputs{}, fmt.Errorf("failed to create remes keeper: %w", err)
	}

	m := NewAppModule(in.Cdc, k, in.AuthKeeper, in.BankKeeper)

	return ModuleOutputs{RemesKeeper: k, Module: m}, nil
}
