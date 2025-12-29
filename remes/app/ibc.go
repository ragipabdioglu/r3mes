package app

import (
	"cosmossdk.io/core/appmodule"
	"github.com/cosmos/cosmos-sdk/codec"
	servertypes "github.com/cosmos/cosmos-sdk/server/types"
	// IBC imports temporarily disabled for compatibility
	// icamodule "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts"
	// icacontrollerkeeper "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/controller/keeper"
	// icahostkeeper "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/host/keeper"
	// ibctransferkeeper "github.com/cosmos/ibc-go/v8/modules/apps/transfer/keeper"
	// ibckeeper "github.com/cosmos/ibc-go/v8/modules/core/keeper"
)

// registerIBCModules register IBC keepers and non dependency inject modules.
// 
// STATUS: DISABLED - IBC module initialization temporarily disabled
// 
// REASON: IBC-go v8 requires different constructor signatures than Cosmos SDK v0.50.x provides.
// This needs to be fixed for full IBC support. For now, IBC functionality is disabled to allow the build to succeed.
// 
// RE-ENABLE PLAN: See remes/docs/IBC_MODULE_STATUS.md for detailed migration guide.
// 
// TODO: Re-enable IBC modules when IBC-go v8 compatibility with Cosmos SDK v0.50.x is resolved.
func (app *App) registerIBCModules(appOpts servertypes.AppOptions) error {
	// IBC modules disabled for now - needs IBC-go v8 compatibility fixes
	// See remes/docs/IBC_MODULE_STATUS.md for re-enable plan
	return nil
}

// RegisterIBC Since the IBC modules don't support dependency injection,
// we need to manually register the modules on the client side.
// This needs to be removed after IBC supports App Wiring.
// 
// STATUS: DISABLED - IBC modules temporarily disabled for IBC-go v8 compatibility
// 
// RE-ENABLE PLAN: See remes/docs/IBC_MODULE_STATUS.md for detailed migration guide.
// 
// TODO: Re-enable IBC modules when IBC-go v8 compatibility with Cosmos SDK v0.50.x is resolved.
func RegisterIBC(cdc codec.Codec) map[string]appmodule.AppModule {
	// IBC modules disabled for now
	// See remes/docs/IBC_MODULE_STATUS.md for re-enable plan
	return map[string]appmodule.AppModule{}
}

// GetIBCKeeper returns the IBC keeper.
// Used for supply with IBC keeper getter for the IBC modules with App Wiring.
// 
// STATUS: DISABLED - Returns nil until IBC-go v8 compatibility is fixed
// 
// RE-ENABLE PLAN: See remes/docs/IBC_MODULE_STATUS.md for detailed migration guide.
// 
// TODO: Re-enable when IBC-go v8 compatibility is fixed
func (app *App) GetIBCKeeper() interface{} {
	// return app.IBCKeeper
	// See remes/docs/IBC_MODULE_STATUS.md for re-enable plan
	return nil
}
