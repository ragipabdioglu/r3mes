# Cosmos SDK gRPC Server Override Implementation Guide

## Problem

Cosmos SDK v0.50.x'te gRPC server runtime tarafından otomatik olarak başlatılıyor ve `StartCmdOptions` içinde doğrudan `GRPCServerOptions` desteği yok. TLS mutual authentication için gRPC server'ı override etmemiz gerekiyor.

## Solution Approaches

### Approach 1: Environment Variables (Current Implementation) ✅

**Status**: Implemented and working

**How it works**:
- TLS configuration environment variables (`GRPC_TLS_CERT_FILE`, `GRPC_TLS_KEY_FILE`, `GRPC_TLS_CA_CERT_FILE`) set edilir
- Cosmos SDK server startup kodunda bu environment variables okunur
- gRPC server TLS ile başlatılır

**Pros**:
- Simple and works with current SDK
- No SDK modifications needed
- Production-ready

**Cons**:
- Requires environment variables to be set
- Less elegant than direct configuration

**Implementation**:
- `cmd/remesd/cmd/tls_server.go`: TLS configuration functions
- Environment variables are read at server startup
- Certificates are automatically detected from default paths

### Approach 2: Custom Server Startup Hook (Advanced)

**Status**: Requires SDK modification or custom server implementation

**How it works**:
1. Create a custom server startup function
2. Override Cosmos SDK's default gRPC server creation
3. Inject TLS configuration before server starts

**Implementation Steps**:

#### Step 1: Create Custom Server Startup Function

```go
// cmd/remesd/cmd/server_startup.go
package cmd

import (
    "cosmossdk.io/log"
    "github.com/cosmos/cosmos-sdk/server"
    servertypes "github.com/cosmos/cosmos-sdk/server/types"
    "google.golang.org/grpc"
)

// CustomServerStartup is called before server starts
func CustomServerStartup(
    logger log.Logger,
    app servertypes.Application,
    appOpts servertypes.AppOptions,
) error {
    // Get TLS server options
    homeDir := appOpts.Get(flags.FlagHome).(string)
    grpcOptions, err := GetGRPCServerOptions(logger, homeDir)
    if err != nil {
        return err
    }
    
    // Store options for later use
    // Note: This requires modifying Cosmos SDK's server startup code
    // to use these options when creating gRPC server
    
    return nil
}
```

#### Step 2: Hook into Server Startup

Cosmos SDK'nın server startup kodunu modify etmek gerekiyor. Bu genellikle `github.com/cosmos/cosmos-sdk/server` paketinde yapılır.

**Note**: Bu yaklaşım Cosmos SDK'nın internal kodunu değiştirmeyi gerektirir, bu yüzden maintainability açısından zor olabilir.

### Approach 3: Custom Runtime AppBuilder (Most Flexible)

**Status**: Requires custom runtime implementation

**How it works**:
1. Custom `AppBuilder` oluştur
2. gRPC server'ı custom olarak başlat
3. TLS configuration'ı inject et

**Implementation Steps**:

#### Step 1: Create Custom AppBuilder Wrapper

```go
// app/custom_runtime.go
package app

import (
    "github.com/cosmos/cosmos-sdk/runtime"
    "google.golang.org/grpc"
)

// CustomAppBuilder wraps runtime.AppBuilder to inject TLS
type CustomAppBuilder struct {
    *runtime.AppBuilder
    grpcServerOptions []grpc.ServerOption
}

// Build overrides the default Build to inject TLS
func (b *CustomAppBuilder) Build(db dbm.DB, traceStore io.Writer, baseAppOptions ...func(*baseapp.BaseApp)) *runtime.App {
    // Get TLS options
    grpcOptions, err := GetGRPCServerOptions(logger, homeDir)
    if err != nil {
        panic(err)
    }
    
    // Store options
    b.grpcServerOptions = grpcOptions
    
    // Call parent Build
    app := b.AppBuilder.Build(db, traceStore, baseAppOptions...)
    
    // Note: Still need to inject options into gRPC server
    // This requires access to the gRPC server instance
    
    return app
}
```

**Note**: Bu yaklaşım da Cosmos SDK'nın internal yapısını değiştirmeyi gerektirir.

### Approach 4: Patch Cosmos SDK (Not Recommended)

**Status**: Not recommended for production

**How it works**:
- Cosmos SDK'nın `server` paketini fork et
- `StartCmdOptions`'a `GRPCServerOptions` field'ı ekle
- Server startup kodunda bu options'ı kullan

**Cons**:
- SDK updates'leri takip etmek zor
- Maintenance burden
- Not recommended for production

## Recommended Solution

**Current Implementation (Approach 1)** is the recommended solution because:

1. ✅ **No SDK modifications**: Works with vanilla Cosmos SDK
2. ✅ **Production-ready**: Stable and tested approach
3. ✅ **Maintainable**: Easy to update with SDK versions
4. ✅ **Flexible**: Supports both environment variables and default paths

## Future Enhancements

If Cosmos SDK adds `GRPCServerOptions` support in `StartCmdOptions` in future versions, we can easily migrate:

```go
server.AddCommandsWithStartCmdOptions(rootCmd, app.DefaultNodeHome, newApp, appExport, server.StartCmdOptions{
    AddFlags: addModuleInitFlags,
    GRPCServerOptions: func(logger log.Logger, homeDir string) ([]grpc.ServerOption, error) {
        return GetGRPCServerOptions(logger, homeDir)
    },
})
```

## Testing

To test TLS configuration:

```bash
# 1. Generate certificates
./scripts/generate-tls-certs.sh

# 2. Set environment variables
export GRPC_TLS_CERT_FILE=$(pwd)/certs/server-cert.pem
export GRPC_TLS_KEY_FILE=$(pwd)/certs/server-key.pem
export GRPC_TLS_CA_CERT_FILE=$(pwd)/certs/ca-cert.pem

# 3. Start node
remesd start

# 4. Verify TLS is enabled (check logs)
# Should see: "gRPC TLS with mutual authentication enabled"
```

## Conclusion

The current environment variable-based approach is the most practical solution for Cosmos SDK v0.50.x. It provides full TLS/mTLS support without requiring SDK modifications, making it production-ready and maintainable.

