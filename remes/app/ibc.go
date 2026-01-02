package app

import (
	"fmt"

	"cosmossdk.io/core/appmodule"
	storetypes "cosmossdk.io/store/types"
	"github.com/cosmos/cosmos-sdk/codec"
	servertypes "github.com/cosmos/cosmos-sdk/server/types"
	authtypes "github.com/cosmos/cosmos-sdk/x/auth/types"
	govtypes "github.com/cosmos/cosmos-sdk/x/gov/types"

	// IBC imports
	capabilitykeeper "github.com/cosmos/ibc-go/modules/capability/keeper"
	capabilitytypes "github.com/cosmos/ibc-go/modules/capability/types"
	ica "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts"
	icacontroller "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/controller"
	icacontrollerkeeper "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/controller/keeper"
	icacontrollertypes "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/controller/types"
	icahost "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/host"
	icahostkeeper "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/host/keeper"
	icahosttypes "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/host/types"
	icatypes "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/types"
	ibctransfer "github.com/cosmos/ibc-go/v8/modules/apps/transfer"
	ibctransferkeeper "github.com/cosmos/ibc-go/v8/modules/apps/transfer/keeper"
	ibctransfertypes "github.com/cosmos/ibc-go/v8/modules/apps/transfer/types"
	ibc "github.com/cosmos/ibc-go/v8/modules/core"
	ibcclienttypes "github.com/cosmos/ibc-go/v8/modules/core/02-client/types"
	ibcconnectiontypes "github.com/cosmos/ibc-go/v8/modules/core/03-connection/types"
	porttypes "github.com/cosmos/ibc-go/v8/modules/core/05-port/types"
	ibcexported "github.com/cosmos/ibc-go/v8/modules/core/exported"
	ibckeeper "github.com/cosmos/ibc-go/v8/modules/core/keeper"
	ibctm "github.com/cosmos/ibc-go/v8/modules/light-clients/07-tendermint"
)

// IBC Store Keys
const (
	IBCStoreKey           = ibcexported.StoreKey
	IBCTransferStoreKey   = ibctransfertypes.StoreKey
	ICAControllerStoreKey = icacontrollertypes.StoreKey
	ICAHostStoreKey       = icahosttypes.StoreKey
	CapabilityStoreKey    = capabilitytypes.StoreKey
	CapabilityMemStoreKey = capabilitytypes.MemStoreKey
)

// GetIBCStoreKeys returns all IBC-related store keys
func GetIBCStoreKeys() []string {
	return []string{
		IBCStoreKey,
		IBCTransferStoreKey,
		ICAControllerStoreKey,
		ICAHostStoreKey,
		CapabilityStoreKey,
	}
}

// registerIBCModules registers IBC keepers and modules
func (app *App) registerIBCModules(_ servertypes.AppOptions) error {
	// Get store keys
	keys := storetypes.NewKVStoreKeys(
		IBCStoreKey,
		IBCTransferStoreKey,
		ICAControllerStoreKey,
		ICAHostStoreKey,
		CapabilityStoreKey,
	)
	memKeys := storetypes.NewMemoryStoreKeys(CapabilityMemStoreKey)

	// Mount store keys
	for _, key := range keys {
		app.MountKVStores(map[string]*storetypes.KVStoreKey{key.Name(): key})
	}
	for _, key := range memKeys {
		app.MountMemoryStores(map[string]*storetypes.MemoryStoreKey{key.Name(): key})
	}

	// Create Capability Keeper
	app.CapabilityKeeper = capabilitykeeper.NewKeeper(
		app.appCodec,
		keys[CapabilityStoreKey],
		memKeys[CapabilityMemStoreKey],
	)

	// Seal capability keeper after all scoped keepers are created
	defer app.CapabilityKeeper.Seal()

	// Create scoped keepers
	app.ScopedIBCKeeper = app.CapabilityKeeper.ScopeToModule(ibcexported.ModuleName)
	app.ScopedTransferKeeper = app.CapabilityKeeper.ScopeToModule(ibctransfertypes.ModuleName)
	app.ScopedICAControllerKeeper = app.CapabilityKeeper.ScopeToModule(icacontrollertypes.SubModuleName)
	app.ScopedICAHostKeeper = app.CapabilityKeeper.ScopeToModule(icahosttypes.SubModuleName)

	// Create IBC Keeper
	app.IBCKeeper = ibckeeper.NewKeeper(
		app.appCodec,
		keys[IBCStoreKey],
		app.GetSubspace(ibcexported.ModuleName),
		app.StakingKeeper,
		app.UpgradeKeeper,
		app.ScopedIBCKeeper,
		authtypes.NewModuleAddress(govtypes.ModuleName).String(),
	)

	// Create Transfer Keeper
	app.TransferKeeper = ibctransferkeeper.NewKeeper(
		app.appCodec,
		keys[IBCTransferStoreKey],
		app.GetSubspace(ibctransfertypes.ModuleName),
		app.IBCKeeper.ChannelKeeper,
		app.IBCKeeper.ChannelKeeper,
		app.IBCKeeper.PortKeeper,
		app.AuthKeeper,
		app.BankKeeper,
		app.ScopedTransferKeeper,
		authtypes.NewModuleAddress(govtypes.ModuleName).String(),
	)

	// Create ICA Controller Keeper
	app.ICAControllerKeeper = icacontrollerkeeper.NewKeeper(
		app.appCodec,
		keys[ICAControllerStoreKey],
		app.GetSubspace(icacontrollertypes.SubModuleName),
		app.IBCKeeper.ChannelKeeper,
		app.IBCKeeper.ChannelKeeper,
		app.IBCKeeper.PortKeeper,
		app.ScopedICAControllerKeeper,
		app.MsgServiceRouter(),
		authtypes.NewModuleAddress(govtypes.ModuleName).String(),
	)

	// Create ICA Host Keeper
	app.ICAHostKeeper = icahostkeeper.NewKeeper(
		app.appCodec,
		keys[ICAHostStoreKey],
		app.GetSubspace(icahosttypes.SubModuleName),
		app.IBCKeeper.ChannelKeeper,
		app.IBCKeeper.ChannelKeeper,
		app.IBCKeeper.PortKeeper,
		app.AuthKeeper,
		app.ScopedICAHostKeeper,
		app.MsgServiceRouter(),
		authtypes.NewModuleAddress(govtypes.ModuleName).String(),
	)

	// Create IBC Router
	ibcRouter := porttypes.NewRouter()

	// Add transfer route
	transferModule := ibctransfer.NewIBCModule(app.TransferKeeper)
	ibcRouter.AddRoute(ibctransfertypes.ModuleName, transferModule)

	// Add ICA routes
	icaControllerModule := icacontroller.NewIBCMiddleware(nil, app.ICAControllerKeeper)
	icaHostModule := icahost.NewIBCModule(app.ICAHostKeeper)
	ibcRouter.AddRoute(icacontrollertypes.SubModuleName, icaControllerModule)
	ibcRouter.AddRoute(icahosttypes.SubModuleName, icaHostModule)

	// Set IBC Router
	app.IBCKeeper.SetRouter(ibcRouter)

	return nil
}

// RegisterIBC registers IBC modules for client-side registration
// Since IBC modules don't support dependency injection, we need to manually register them
func RegisterIBC(cdc codec.Codec) map[string]appmodule.AppModule {
	return map[string]appmodule.AppModule{
		ibcexported.ModuleName:      ibc.AppModule{},
		ibctransfertypes.ModuleName: ibctransfer.AppModule{},
		icatypes.ModuleName:         ica.AppModule{},
		ibctm.ModuleName:            ibctm.AppModule{},
	}
}

// GetIBCKeeper returns the IBC keeper
// Used for supply with IBC keeper getter for the IBC modules with App Wiring
func (app *App) GetIBCKeeper() *ibckeeper.Keeper {
	return app.IBCKeeper
}

// GetTransferKeeper returns the IBC transfer keeper
func (app *App) GetTransferKeeper() ibctransferkeeper.Keeper {
	return app.TransferKeeper
}

// GetICAControllerKeeper returns the ICA controller keeper
func (app *App) GetICAControllerKeeper() icacontrollerkeeper.Keeper {
	return app.ICAControllerKeeper
}

// GetICAHostKeeper returns the ICA host keeper
func (app *App) GetICAHostKeeper() icahostkeeper.Keeper {
	return app.ICAHostKeeper
}

// GetCapabilityKeeper returns the capability keeper
func (app *App) GetCapabilityKeeper() *capabilitykeeper.Keeper {
	return app.CapabilityKeeper
}

// IBCModuleParams contains default IBC module parameters
type IBCModuleParams struct {
	// Client params
	AllowedClients []string
	// Connection params
	MaxExpectedTimePerBlock uint64
	// Transfer params
	SendEnabled    bool
	ReceiveEnabled bool
}

// DefaultIBCModuleParams returns default IBC module parameters
func DefaultIBCModuleParams() IBCModuleParams {
	return IBCModuleParams{
		AllowedClients:          []string{ibctm.ModuleName},
		MaxExpectedTimePerBlock: uint64(30_000_000_000), // 30 seconds in nanoseconds
		SendEnabled:             true,
		ReceiveEnabled:          true,
	}
}

// GetIBCClientParams returns IBC client parameters
func GetIBCClientParams() ibcclienttypes.Params {
	return ibcclienttypes.NewParams(ibctm.ModuleName)
}

// GetIBCConnectionParams returns IBC connection parameters
func GetIBCConnectionParams() ibcconnectiontypes.Params {
	return ibcconnectiontypes.NewParams(30_000_000_000) // 30 seconds
}

// GetIBCTransferParams returns IBC transfer parameters
func GetIBCTransferParams() ibctransfertypes.Params {
	return ibctransfertypes.NewParams(true, true)
}

// ValidateIBCConfig validates IBC configuration
func ValidateIBCConfig(app *App) error {
	if app.IBCKeeper == nil {
		return fmt.Errorf("IBC keeper is not initialized")
	}
	if app.CapabilityKeeper == nil {
		return fmt.Errorf("capability keeper is not initialized")
	}
	return nil
}
