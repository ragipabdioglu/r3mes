package keeper_test

import (
	"fmt"
	goruntime "runtime"
	"testing"
	"time"

	"cosmossdk.io/log"
	"cosmossdk.io/store"
	"cosmossdk.io/store/metrics"
	storetypes "cosmossdk.io/store/types"
	tmproto "github.com/cometbft/cometbft/proto/tendermint/types"
	dbm "github.com/cosmos/cosmos-db"
	"github.com/cosmos/cosmos-sdk/codec"
	codectypes "github.com/cosmos/cosmos-sdk/codec/types"
	sdkruntime "github.com/cosmos/cosmos-sdk/runtime"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

// BenchmarkKeeperCreation measures keeper initialization time
func BenchmarkKeeperCreation(b *testing.B) {
	for b.Loop() {
		storeKey := storetypes.NewKVStoreKey(types.StoreKey)
		db := dbm.NewMemDB()
		stateStore := store.NewCommitMultiStore(db, log.NewNopLogger(), metrics.NewNoOpMetrics())
		stateStore.MountStoreWithDB(storeKey, storetypes.StoreTypeIAVL, db)
		_ = stateStore.LoadLatestVersion()

		storeService := sdkruntime.NewKVStoreService(storeKey)
		interfaceRegistry := codectypes.NewInterfaceRegistry()
		cdc := codec.NewProtoCodec(interfaceRegistry)

		_, _ = keeper.NewKeeper(
			storeService,
			cdc,
			mockAddressCodec{},
			[]byte("authority"),
			mockBankKeeper{},
			mockAuthKeeper{},
			"http://localhost:5001",
		)
	}
}

// BenchmarkParamsSetGet measures params read/write performance
func BenchmarkParamsSetGet(b *testing.B) {
	k, ctx := setupBenchmarkKeeper(b)
	params := types.DefaultParams()

	b.ResetTimer()
	for b.Loop() {
		_ = k.SetParams(ctx, params)
		_, _ = k.GetParams(ctx)
	}
}

// BenchmarkModelRegistration measures model registration performance
func BenchmarkModelRegistration(b *testing.B) {
	k, ctx := setupBenchmarkKeeper(b)

	b.ResetTimer()
	i := 0
	for b.Loop() {
		model := types.ModelRegistry{
			Config: types.ModelConfig{
				ModelType:          types.ModelType_MODEL_TYPE_BITNET,
				ModelVersion:       fmt.Sprintf("v1.0.%d", i),
				ArchitectureConfig: `{"hidden_size": 768}`,
				ContainerHash:      fmt.Sprintf("hash-%d", i),
				ContainerRegistry:  "docker.io",
			},
			IsActive:  true,
			CreatedAt: time.Now(),
		}
		_ = k.GetModelKeeper().RegisterModel(ctx, model)
		i++
	}
}

// BenchmarkGradientSubmission measures gradient submission performance
func BenchmarkGradientSubmission(b *testing.B) {
	k, ctx := setupBenchmarkKeeper(b)

	b.ResetTimer()
	i := 0
	for b.Loop() {
		gradient := types.StoredGradient{
			Miner:           fmt.Sprintf("miner-%d", i),
			IpfsHash:        fmt.Sprintf("QmHash%d", i),
			ModelVersion:    "v1.0.0",
			TrainingRoundId: uint64(i),
			ShardId:         0,
			GradientHash:    fmt.Sprintf("gradient-hash-%d", i),
			GpuArchitecture: "cuda",
		}
		_ = k.GetTrainingKeeper().SubmitGradient(ctx, gradient)
		i++
	}
}

// BenchmarkNodeRegistration measures node registration performance
func BenchmarkNodeRegistration(b *testing.B) {
	k, ctx := setupBenchmarkKeeper(b)

	b.ResetTimer()
	i := 0
	for b.Loop() {
		node := types.NodeRegistration{
			NodeAddress: fmt.Sprintf("node-address-%d", i),
			NodeType:    types.NODE_TYPE_MINING,
			Stake:       "1000000stake",
		}
		_ = k.GetNodeKeeper().RegisterNode(ctx, node)
		i++
	}
}

// BenchmarkMemoryAllocation measures memory allocation patterns
func BenchmarkMemoryAllocation(b *testing.B) {
	b.ReportAllocs()

	k, ctx := setupBenchmarkKeeper(b)

	b.ResetTimer()
	for b.Loop() {
		_ = k.SetParams(ctx, types.DefaultParams())
		_, _ = k.GetParams(ctx)

		model := types.ModelRegistry{
			Config: types.ModelConfig{
				ModelType:    types.ModelType_MODEL_TYPE_BITNET,
				ModelVersion: "v1.0.0",
			},
		}
		_ = k.GetModelKeeper().RegisterModel(ctx, model)
	}
}

// setupBenchmarkKeeper creates a keeper for benchmarking
func setupBenchmarkKeeper(b *testing.B) (keeper.Keeper, sdk.Context) {
	storeKey := storetypes.NewKVStoreKey(types.StoreKey)
	db := dbm.NewMemDB()
	stateStore := store.NewCommitMultiStore(db, log.NewNopLogger(), metrics.NewNoOpMetrics())
	stateStore.MountStoreWithDB(storeKey, storetypes.StoreTypeIAVL, db)
	if err := stateStore.LoadLatestVersion(); err != nil {
		b.Fatal(err)
	}

	storeService := sdkruntime.NewKVStoreService(storeKey)
	interfaceRegistry := codectypes.NewInterfaceRegistry()
	types.RegisterInterfaces(interfaceRegistry)
	cdc := codec.NewProtoCodec(interfaceRegistry)

	k, err := keeper.NewKeeper(
		storeService,
		cdc,
		mockAddressCodec{},
		[]byte("authority"),
		mockBankKeeper{},
		mockAuthKeeper{},
		"http://localhost:5001",
	)
	if err != nil {
		b.Fatal(err)
	}

	ctx := sdk.NewContext(stateStore, tmproto.Header{Time: time.Now()}, false, log.NewNopLogger())
	return k, ctx
}

// TestMemoryFootprint measures memory usage of keeper components
func TestMemoryFootprint(t *testing.T) {
	goruntime.GC()
	var m1 goruntime.MemStats
	goruntime.ReadMemStats(&m1)

	k, _ := setupKeeper(t)

	goruntime.GC()
	var m2 goruntime.MemStats
	goruntime.ReadMemStats(&m2)

	keeperMemory := m2.Alloc - m1.Alloc

	t.Logf("=== Memory Footprint Analysis ===")
	t.Logf("Keeper Memory Usage: %d bytes (%.2f KB)", keeperMemory, float64(keeperMemory)/1024)
	t.Logf("Heap Alloc: %d bytes", m2.HeapAlloc)
	t.Logf("Heap Objects: %d", m2.HeapObjects)

	if k.GetCoreKeeper() == nil {
		t.Error("Core keeper should not be nil")
	}

	if keeperMemory > 10*1024*1024 {
		t.Errorf("Keeper memory usage too high: %d bytes", keeperMemory)
	}
}

// TestKeeperPerformanceMetrics runs comprehensive performance tests
func TestKeeperPerformanceMetrics(t *testing.T) {
	k, ctx := setupKeeper(t)

	t.Log("=== Keeper Performance Metrics ===")

	iterations := 1000

	start := time.Now()
	for range iterations {
		_ = k.SetParams(ctx, types.DefaultParams())
	}
	paramsSetDuration := time.Since(start)
	t.Logf("Params Set: %d ops in %v (%.2f ops/sec)",
		iterations, paramsSetDuration, float64(iterations)/paramsSetDuration.Seconds())

	start = time.Now()
	for range iterations {
		_, _ = k.GetParams(ctx)
	}
	paramsGetDuration := time.Since(start)
	t.Logf("Params Get: %d ops in %v (%.2f ops/sec)",
		iterations, paramsGetDuration, float64(iterations)/paramsGetDuration.Seconds())

	start = time.Now()
	for i := range iterations {
		model := types.ModelRegistry{
			Config: types.ModelConfig{
				ModelType:    types.ModelType_MODEL_TYPE_BITNET,
				ModelVersion: fmt.Sprintf("v%d", i),
			},
		}
		_ = k.GetModelKeeper().RegisterModel(ctx, model)
	}
	modelRegDuration := time.Since(start)
	t.Logf("Model Registration: %d ops in %v (%.2f ops/sec)",
		iterations, modelRegDuration, float64(iterations)/modelRegDuration.Seconds())

	start = time.Now()
	for i := range iterations {
		node := types.NodeRegistration{
			NodeAddress: fmt.Sprintf("node-%d", i),
			NodeType:    types.NODE_TYPE_MINING,
			Stake:       "1000stake",
		}
		_ = k.GetNodeKeeper().RegisterNode(ctx, node)
	}
	nodeRegDuration := time.Since(start)
	t.Logf("Node Registration: %d ops in %v (%.2f ops/sec)",
		iterations, nodeRegDuration, float64(iterations)/nodeRegDuration.Seconds())

	t.Log("=== Performance Summary ===")
	totalDuration := paramsSetDuration + paramsGetDuration + modelRegDuration + nodeRegDuration
	t.Logf("Total operations: %d", iterations*4)
	t.Logf("Total duration: %v", totalDuration)
	t.Logf("Average throughput: %.2f ops/sec", float64(iterations*4)/totalDuration.Seconds())
}
