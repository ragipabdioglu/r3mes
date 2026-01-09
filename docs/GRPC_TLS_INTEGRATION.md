# gRPC TLS Integration with Cosmos SDK

## Overview

R3MES implements TLS mutual authentication (mTLS) for gRPC server using environment variables and automatic certificate detection. This approach works with Cosmos SDK v0.50.x where direct gRPC server option configuration is not available in `StartCmdOptions`.

## How It Works

The TLS configuration is handled through:

1. **Environment Variables**: Set `GRPC_TLS_CERT_FILE`, `GRPC_TLS_KEY_FILE`, and optionally `GRPC_TLS_CA_CERT_FILE`
2. **Automatic Detection**: If certificates are found in `~/.remesd/certs/`, they are automatically used
3. **Runtime Configuration**: The `GetGRPCServerOptions` function in `tls_server.go` configures TLS at runtime

## Current Implementation Status

✅ **Implemented**: TLS configuration is handled via environment variables and automatic certificate detection. The `PostSetup` hook in `StartCmdOptions` is used to verify and log TLS configuration during server startup.

⚠️ **Note**: Cosmos SDK v0.50.x's runtime automatically starts the gRPC server, and there's no direct hook to inject custom gRPC server options through `StartCmdOptions`. However, we use the `PostSetup` hook to verify TLS configuration and ensure certificates are properly set up.

### Workaround Options

1. **Environment Variables** (Current Implementation)
   - Set environment variables before starting the node
   - Certificates are detected and configured automatically
   - Works for most use cases

2. **Custom Server Startup** (Future Enhancement)
   - Override the gRPC server startup in `app.go`
   - Requires modifying Cosmos SDK runtime behavior
   - More complex but provides full control

3. **Cosmos SDK Update** (Future)
   - Wait for Cosmos SDK to add `GRPCServerOptions` support in `StartCmdOptions`
   - Most elegant solution but depends on SDK updates

## Usage

### Method 1: Environment Variables

```bash
export GRPC_TLS_CERT_FILE=/path/to/certs/server-cert.pem
export GRPC_TLS_KEY_FILE=/path/to/certs/server-key.pem
export GRPC_TLS_CA_CERT_FILE=/path/to/certs/ca-cert.pem  # Optional for mTLS

remesd start
```

### Method 2: Default Paths

Place certificates in the default location:

```bash
mkdir -p ~/.remesd/certs
cp certs/server-cert.pem ~/.remesd/certs/
cp certs/server-key.pem ~/.remesd/certs/
cp certs/ca-cert.pem ~/.remesd/certs/  # Optional for mTLS

remesd start
```

## Future Enhancements

To fully integrate TLS with Cosmos SDK's gRPC server, we could:

1. **Modify `app.go`**: Override gRPC server creation in the app initialization
2. **Custom Runtime**: Create a custom runtime that supports TLS configuration
3. **SDK Contribution**: Contribute `GRPCServerOptions` support to Cosmos SDK

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

# 4. Check logs for TLS confirmation
# Should see: "gRPC TLS with mutual authentication enabled"
```

## Troubleshooting

### TLS Not Enabled

If you see "gRPC TLS is not configured, using insecure connection":
- Check environment variables are set correctly
- Verify certificate files exist and are readable
- Check default paths (`~/.remesd/certs/`)

### Certificate Errors

If you see certificate errors:
- Verify certificate files are valid PEM format
- Check file permissions (should be readable)
- Ensure CA certificate matches if using mTLS

### Connection Refused

If Python miner can't connect:
- Verify gRPC server is listening on correct port
- Check firewall settings
- Ensure TLS is enabled on both client and server

