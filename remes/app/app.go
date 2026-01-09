package app

import (
	"fmt"
	"io"

	clienthelpers "cosmossdk.io/client/v2/helpers"
	"cosmossdk.io/core/appmodule"
	"cosmossdk.io/depinject"
	"cosmossdk.io/log"
	storetypes "cosmossdk.io/store/types"
	circuitkeeper "cosmossdk.io/x/circuit/keeper"
	upgradekeeper "cosmossdk.io/x/upgrade/keeper"

	abci "github.com/cometbft/cometbft/abci/types"
	dbm "github.com/cosmos/cosmos-db"
	"github.com/cosmos/cosmos-sdk/baseapp"
	"github.com/cosmos/cosmos-sdk/client"
	"github.com/cosmos/cosmos-sdk/codec"
	codectypes "github.com/cosmos/cosmos-sdk/codec/types"
	"github.com/cosmos/cosmos-sdk/runtime"
	"github.com/cosmos/cosmos-sdk/server"
	"github.com/cosmos/cosmos-sdk/server/api"
	"github.com/cosmos/cosmos-sdk/server/config"
	servertypes "github.com/cosmos/cosmos-sdk/server/types"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/cosmos/cosmos-sdk/types/module"
	"github.com/cosmos/cosmos-sdk/x/auth"
	authkeeper "github.com/cosmos/cosmos-sdk/x/auth/keeper"
	authsims "github.com/cosmos/cosmos-sdk/x/auth/simulation"
	authtypes "github.com/cosmos/cosmos-sdk/x/auth/types"
	authzkeeper "github.com/cosmos/cosmos-sdk/x/authz/keeper"
	bankkeeper "github.com/cosmos/cosmos-sdk/x/bank/keeper"
	consensuskeeper "github.com/cosmos/cosmos-sdk/x/consensus/keeper"
	distrkeeper "github.com/cosmos/cosmos-sdk/x/distribution/keeper"
	"github.com/cosmos/cosmos-sdk/x/genutil"
	genutiltypes "github.com/cosmos/cosmos-sdk/x/genutil/types"
	govkeeper "github.com/cosmos/cosmos-sdk/x/gov/keeper"
	mintkeeper "github.com/cosmos/cosmos-sdk/x/mint/keeper"
	paramskeeper "github.com/cosmos/cosmos-sdk/x/params/keeper"
	paramstypes "github.com/cosmos/cosmos-sdk/x/params/types"
	slashingkeeper "github.com/cosmos/cosmos-sdk/x/slashing/keeper"
	stakingkeeper "github.com/cosmos/cosmos-sdk/x/staking/keeper"

	// IBC imports
	capabilitykeeper "github.com/cosmos/ibc-go/modules/capability/keeper"
	icacontrollerkeeper "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/controller/keeper"
	icahostkeeper "github.com/cosmos/ibc-go/v8/modules/apps/27-interchain-accounts/host/keeper"
	ibctransferkeeper "github.com/cosmos/ibc-go/v8/modules/apps/transfer/keeper"
	ibckeeper "github.com/cosmos/ibc-go/v8/modules/core/keeper"

	"net/http"
	"remes/docs"
	remesmodulekeeper "remes/x/remes/keeper"
	remesibc "remes/x/remes/ibc"
	remestypes "remes/x/remes/types"

	"github.com/gorilla/mux"
)

const (
	// Name is the name of the application.
	Name = "remes"
	// AccountAddressPrefix is the prefix for accounts addresses.
	AccountAddressPrefix = "remes"
	// ChainCoinType is the coin type of the chain - using unique coin type for remes
	ChainCoinType = 9999
)

// DefaultNodeHome default home directories for the application daemon
var DefaultNodeHome string

var (
	_ runtime.AppI            = (*App)(nil)
	_ servertypes.Application = (*App)(nil)
)

// App extends an ABCI application, but with most of its parameters exported.
// They are exported for convenience in creating helper functions, as object
// capabilities aren't needed for testing.
type App struct {
	*runtime.App
	legacyAmino       *codec.LegacyAmino
	appCodec          codec.Codec
	txConfig          client.TxConfig
	interfaceRegistry codectypes.InterfaceRegistry
	config            *Config

	// keepers
	// only keepers required by the app are exposed
	// the list of all modules is available in the app_config
	AuthKeeper            authkeeper.AccountKeeper
	BankKeeper            bankkeeper.Keeper
	StakingKeeper         *stakingkeeper.Keeper
	SlashingKeeper        slashingkeeper.Keeper
	MintKeeper            mintkeeper.Keeper
	DistrKeeper           distrkeeper.Keeper
	GovKeeper             *govkeeper.Keeper
	UpgradeKeeper         *upgradekeeper.Keeper
	AuthzKeeper           authzkeeper.Keeper
	ConsensusParamsKeeper consensuskeeper.Keeper
	CircuitBreakerKeeper  circuitkeeper.Keeper
	ParamsKeeper          paramskeeper.Keeper

	// ibc keepers
	IBCKeeper           *ibckeeper.Keeper
	ICAControllerKeeper icacontrollerkeeper.Keeper
	ICAHostKeeper       icahostkeeper.Keeper
	TransferKeeper      ibctransferkeeper.Keeper

	// Capability keepers for IBC
	CapabilityKeeper          *capabilitykeeper.Keeper
	ScopedIBCKeeper           capabilitykeeper.ScopedKeeper
	ScopedTransferKeeper      capabilitykeeper.ScopedKeeper
	ScopedICAControllerKeeper capabilitykeeper.ScopedKeeper
	ScopedICAHostKeeper       capabilitykeeper.ScopedKeeper

	// simulation manager
	sm          *module.SimulationManager
	RemesKeeper remesmodulekeeper.Keeper
}

func init() {
	var err error
	clienthelpers.EnvPrefix = Name
	DefaultNodeHome, err = clienthelpers.GetNodeHomeDirectory("." + Name)
	if err != nil {
		panic(err)
	}
}

// AppConfig returns the default app config.
func AppConfig() depinject.Config {
	return depinject.Configs(
		appConfig,
		depinject.Supply(
			// supply custom module basics
			map[string]module.AppModuleBasic{
				genutiltypes.ModuleName: genutil.NewAppModuleBasic(genutiltypes.DefaultMessageValidator),
			},
		),
	)
}

// New returns a reference to an initialized App.
func New(
	logger log.Logger,
	db dbm.DB,
	traceStore io.Writer,
	loadLatest bool,
	appOpts servertypes.AppOptions,
	baseAppOptions ...func(*baseapp.BaseApp),
) *App {
	// Load configuration
	config := LoadConfig()

	var (
		app        = &App{config: config}
		appBuilder *runtime.AppBuilder

		// merge the AppConfig and other configuration in one config
		appConfig = depinject.Configs(
			AppConfig(),
			depinject.Supply(
				appOpts, // supply app options
				logger,  // supply logger

				// Supply with IBC keeper getter for the IBC modules with App Wiring.
				// The IBC Keeper cannot be passed because it has not been initiated yet.
				// Passing the getter, the app IBC Keeper will always be accessible.
				// This needs to be removed after IBC supports App Wiring.
				app.GetIBCKeeper,

				// here alternative options can be supplied to the DI container.
				// those options can be used f.e to override the default behavior of some modules.
				// for instance supplying a custom address codec for not using bech32 addresses.
				// read the depinject documentation and depinject module wiring for more information
				// on available options and how to use them.
			),
		)
	)

	var appModules map[string]appmodule.AppModule
	if err := depinject.Inject(appConfig,
		&appBuilder,
		&appModules,
		&app.appCodec,
		&app.legacyAmino,
		&app.txConfig,
		&app.interfaceRegistry,
		&app.AuthKeeper,
		&app.BankKeeper,
		&app.StakingKeeper,
		&app.SlashingKeeper,
		&app.MintKeeper,
		&app.DistrKeeper,
		&app.GovKeeper,
		&app.UpgradeKeeper,
		&app.AuthzKeeper,
		&app.ConsensusParamsKeeper,
		&app.CircuitBreakerKeeper,
		&app.ParamsKeeper,
		&app.RemesKeeper,
	); err != nil {
		logger.Error("Failed to inject dependencies", "error", err)
		panic(fmt.Errorf("dependency injection failed: %w", err))
	}

	// add to default baseapp options
	// enable optimistic execution
	baseAppOptions = append(baseAppOptions, baseapp.SetOptimisticExecution())

	// build app
	app.App = appBuilder.Build(db, traceStore, baseAppOptions...)

	// register legacy modules
	if err := app.registerIBCModules(appOpts); err != nil {
		logger.Error("Failed to register IBC modules", "error", err)
		panic(fmt.Errorf("IBC module registration failed: %w", err))
	}

	/****  Module Options ****/

	// create the simulation manager and define the order of the modules for deterministic simulations
	overrideModules := map[string]module.AppModuleSimulation{
		authtypes.ModuleName: auth.NewAppModule(app.appCodec, app.AuthKeeper, authsims.RandomGenesisAccounts, nil),
	}
	app.sm = module.NewSimulationManagerFromAppModules(app.ModuleManager.Modules, overrideModules)

	app.sm.RegisterStoreDecoders()

	// A custom InitChainer sets if extra pre-init-genesis logic is required.
	// This is necessary for manually registered modules that do not support app wiring.
	// Manually set the module version map as shown below.
	// The upgrade module will automatically handle de-duplication of the module version map.
	app.SetInitChainer(func(ctx sdk.Context, req *abci.RequestInitChain) (*abci.ResponseInitChain, error) {
		if err := app.UpgradeKeeper.SetModuleVersionMap(ctx, app.ModuleManager.GetVersionMap()); err != nil {
			return nil, err
		}
		return app.App.InitChainer(ctx, req)
	})

	if err := app.Load(loadLatest); err != nil {
		logger.Error("Failed to load app", "error", err)
		panic(fmt.Errorf("app loading failed: %w", err))
	}

	return app
}

// GetSubspace returns a param subspace for a given module name.
func (app *App) GetSubspace(moduleName string) paramstypes.Subspace {
	subspace, _ := app.ParamsKeeper.GetSubspace(moduleName)
	return subspace
}

// LegacyAmino returns App's amino codec.
func (app *App) LegacyAmino() *codec.LegacyAmino {
	return app.legacyAmino
}

// AppCodec returns App's app codec.
func (app *App) AppCodec() codec.Codec {
	return app.appCodec
}

// InterfaceRegistry returns App's InterfaceRegistry.
func (app *App) InterfaceRegistry() codectypes.InterfaceRegistry {
	return app.interfaceRegistry
}

// TxConfig returns App's TxConfig
func (app *App) TxConfig() client.TxConfig {
	return app.txConfig
}

// GetKey returns the KVStoreKey for the provided store key.
func (app *App) GetKey(storeKey string) *storetypes.KVStoreKey {
	kvStoreKey, ok := app.UnsafeFindStoreKey(storeKey).(*storetypes.KVStoreKey)
	if !ok {
		return nil
	}
	return kvStoreKey
}

// GetStoreKeys returns all store keys as a map.
func (app *App) GetStoreKeys() map[string]*storetypes.KVStoreKey {
	keys := make(map[string]*storetypes.KVStoreKey)

	// Add IBC store keys
	for _, keyName := range GetIBCStoreKeys() {
		if key := app.GetKey(keyName); key != nil {
			keys[keyName] = key
		}
	}

	return keys
}

// GetRemesKeeper returns the Remes keeper
func (app *App) GetRemesKeeper() *remesmodulekeeper.Keeper {
	return &app.RemesKeeper
}

// SimulationManager implements the SimulationApp interface
func (app *App) SimulationManager() *module.SimulationManager {
	return app.sm
}

// RegisterAPIRoutes registers all application module routes with the provided
// API server.
func (app *App) RegisterAPIRoutes(apiSvr *api.Server, apiConfig config.APIConfig) {
	app.App.RegisterAPIRoutes(apiSvr, apiConfig)
	// register swagger API in app.go so that other applications can override easily
	if err := server.RegisterSwaggerAPI(apiSvr.ClientCtx, apiSvr.Router, apiConfig.Swagger); err != nil {
		fmt.Printf("⚠️  Failed to register Swagger API: %v\n", err)
		// Don't panic in production, just log the error
		if !app.config.IsProduction() {
			panic(fmt.Errorf("swagger API registration failed: %w", err))
		}
	}

	// register app's OpenAPI routes.
	docs.RegisterOpenAPIService(Name, apiSvr.Router)

	// Register Web Dashboard API and WebSocket handlers
	// Router is already *mux.Router type
	app.registerDashboardRoutesOnMux(apiSvr.Router)
}

// Note: Cosmos SDK v0.50.x doesn't have RegisterGRPCGatewayRoutes in runtime.AppI
// gRPC Gateway routes are registered automatically by the SDK.
// TLS configuration for gRPC server is handled via environment variables
// and automatic certificate detection in cmd/remesd/cmd/tls_server.go

// registerDashboardRoutesOnMux registers dashboard API routes directly on mux.Router
func (app *App) registerDashboardRoutesOnMux(muxRouter *mux.Router) {
	// Check if dashboard is enabled in configuration
	if !app.config.EnableDashboard {
		fmt.Println("📊 Dashboard disabled by configuration")
		return
	}

	// Get remes keeper from app
	remesKeeper := app.GetRemesKeeper()
	if remesKeeper == nil {
		// Log error (in production, use proper logger)
		fmt.Println("⚠️  RemesKeeper not available, skipping dashboard routes")
		return // Keeper not available
	}

	// Create dashboard API
	dashboardAPI := remesmodulekeeper.NewDashboardAPI(*remesKeeper)

	// Dashboard API routes
	apiMux := http.NewServeMux()
	dashboardAPI.RegisterRoutes(apiMux)

	// Mount API routes
	muxRouter.PathPrefix("/api/dashboard/").Handler(http.StripPrefix("/api/dashboard", apiMux))

	// WebSocket endpoint - only if enabled
	if app.config.EnableWebSocket {
		muxRouter.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
			remesKeeper.HandleWebSocket(w, r)
		})
		fmt.Println("🔌 WebSocket endpoint registered: /ws")
	}

	// Log success (in production, use proper logger)
	fmt.Println("✅ Dashboard API routes registered: /api/dashboard/*")
}

// registerDashboardRoutes registers dashboard API routes and WebSocket handlers (fallback)
func (app *App) registerDashboardRoutes(router any) {
	// Get remes keeper from app
	remesKeeper := app.GetRemesKeeper()
	if remesKeeper == nil {
		return // Keeper not available
	}

	// Create dashboard API
	dashboardAPI := remesmodulekeeper.NewDashboardAPI(*remesKeeper)

	// Try multiple router types (Cosmos SDK v0.50 may use different router types)
	var muxRouter *mux.Router

	// Try direct cast first
	if r, ok := router.(*mux.Router); ok {
		muxRouter = r
	} else {
		// Try to get router via reflection or interface
		// Some Cosmos SDK versions wrap the router
		type Router interface {
			HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request)) *mux.Route
			PathPrefix(template string) *mux.Route
		}
		if r, ok := router.(Router); ok {
			// Use interface methods directly
			apiMux := http.NewServeMux()
			dashboardAPI.RegisterRoutes(apiMux)

			// Register via interface
			r.PathPrefix("/api/dashboard/").Handler(http.StripPrefix("/api/dashboard", apiMux))
			r.HandleFunc("/ws", func(w http.ResponseWriter, req *http.Request) {
				remesKeeper.HandleWebSocket(w, req)
			})
			return
		}

		// If all else fails, try to use http.ServeMux directly
		if httpMux, ok := router.(*http.ServeMux); ok {
			// Register directly on http.ServeMux
			dashboardAPI.RegisterRoutes(httpMux)
			httpMux.HandleFunc("/ws", func(w http.ResponseWriter, req *http.Request) {
				remesKeeper.HandleWebSocket(w, req)
			})
			return
		}

		// Log error if router type is unknown
		// Note: In production, you might want to use a logger here
		return
	}

	// Register routes using gorilla/mux router
	if muxRouter != nil {
		// WebSocket endpoint
		muxRouter.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
			remesKeeper.HandleWebSocket(w, r)
		})

		// Dashboard API routes
		apiMux := http.NewServeMux()
		dashboardAPI.RegisterRoutes(apiMux)

		// Mount API routes
		muxRouter.PathPrefix("/api/dashboard/").Handler(http.StripPrefix("/api/dashboard", apiMux))
	}
}

// GetMaccPerms returns a copy of the module account permissions
//
// NOTE: This is solely to be used for testing purposes.
func GetMaccPerms() map[string][]string {
	dup := make(map[string][]string)
	for _, perms := range moduleAccPerms {
		dup[perms.GetAccount()] = perms.GetPermissions()
	}

	return dup
}

// BlockedAddresses returns all the app's blocked account addresses.
func BlockedAddresses() map[string]bool {
	result := make(map[string]bool)

	if len(blockAccAddrs) > 0 {
		for _, addr := range blockAccAddrs {
			result[addr] = true
		}
	} else {
		for addr := range GetMaccPerms() {
			result[addr] = true
		}
	}

	return result
}


// registerRemesIBCModule registers R3MES IBC module for cross-chain gradient synchronization
// Note: Main IBC modules are registered in ibc.go
func (app *App) registerRemesIBCModule() error {
	// Get IBC keeper (should be initialized by depinject)
	ibcKeeper := app.GetIBCKeeper()
	if ibcKeeper == nil {
		return fmt.Errorf("IBC keeper not initialized")
	}

	// Create R3MES IBC module
	remesIBCModule := remesibc.NewIBCModule(app.RemesKeeper, app.appCodec)

	// Register R3MES IBC module with IBC router
	// Port ID format: "remes" for gradient synchronization
	portID := remestypes.IBCPortID
	
	// Get existing router and add R3MES route
	ibcRouter := ibcKeeper.Router
	if ibcRouter != nil {
		ibcRouter.AddRoute(portID, remesIBCModule)
	}

	app.BaseApp.Logger().Info("✅ R3MES IBC module registered",
		"port_id", portID,
		"version", remestypes.IBCVersion,
	)

	return nil
}
