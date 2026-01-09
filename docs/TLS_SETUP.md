# TLS Mutual Authentication (mTLS) Setup

R3MES uses TLS mutual authentication (mTLS) to secure communication between Python miners and Go nodes.

## Overview

mTLS ensures that:
- **Server Authentication**: Python miner verifies it's connecting to the legitimate Go node
- **Client Authentication**: Go node verifies the Python miner is authorized
- **Encrypted Communication**: All data is encrypted in transit

## Certificate Generation

### Prerequisites
- OpenSSL installed on your system

### Generate Certificates

Run the certificate generation script:

```bash
cd /home/rabdi/R3MES
./scripts/generate-tls-certs.sh
```

This will create the following files in `./certs/`:
- `ca-cert.pem` - CA certificate
- `ca-key.pem` - CA private key
- `server-cert.pem` - Go node server certificate
- `server-key.pem` - Go node server private key
- `client-cert.pem` - Python miner client certificate
- `client-key.pem` - Python miner client private key

## Go Node Configuration

### Using TLS in Go Node

The Go node automatically configures TLS for the gRPC server if certificate files are provided via environment variables or found in default locations. The TLS configuration is verified and logged during server startup using the `PostSetup` hook in `StartCmdOptions`.

#### Method 1: Environment Variables (Recommended)

Set the following environment variables before starting the node:

```bash
export GRPC_TLS_CERT_FILE=/path/to/certs/server-cert.pem
export GRPC_TLS_KEY_FILE=/path/to/certs/server-key.pem
export GRPC_TLS_CA_CERT_FILE=/path/to/certs/ca-cert.pem  # Optional, for mTLS
```

Then start the node:
```bash
remesd start
```

#### Method 2: Default Certificate Paths

If certificates are placed in the default location (`~/.remesd/certs/`), TLS will be automatically enabled:

```bash
mkdir -p ~/.remesd/certs
cp certs/server-cert.pem ~/.remesd/certs/
cp certs/server-key.pem ~/.remesd/certs/
cp certs/ca-cert.pem ~/.remesd/certs/  # Optional, for mTLS
```

The node will automatically detect and use these certificates.

#### Verification

When TLS is enabled, you should see a log message like:
```
gRPC TLS with mutual authentication enabled cert=... key=... ca_cert=...
```

If TLS is not configured, you'll see:
```
gRPC TLS is not configured, using insecure connection
```

## Python Miner Configuration

### Using TLS in Python Miner

When initializing the `BlockchainClient`, enable TLS:

```python
from bridge.blockchain_client import BlockchainClient

client = BlockchainClient(
    node_url="localhost:9090",
    private_key="your_private_key",
    chain_id="remes-test",
    use_tls=True,
    tls_cert_file="certs/client-cert.pem",
    tls_key_file="certs/client-key.pem",
    tls_ca_file="certs/ca-cert.pem",
    tls_server_name="localhost",  # Optional, defaults to hostname from node_url
)
```

### Development Mode

For development and testing, you can disable TLS:

```python
client = BlockchainClient(
    node_url="localhost:9090",
    use_tls=False,  # Insecure channel for development
)
```

## Security Considerations

1. **Certificate Storage**: Store certificates securely. Never commit private keys to version control.

2. **Certificate Rotation**: Regularly rotate certificates (default: 365 days).

3. **Production Deployment**: Always use TLS in production. Never use insecure channels.

4. **Certificate Validation**: The system validates:
   - Certificate chain (signed by trusted CA)
   - Certificate expiration
   - Server name matching
   - Client certificate authorization

## Troubleshooting

### Certificate Errors

If you encounter certificate errors:
1. Verify certificate files exist and are readable
2. Check certificate expiration dates
3. Ensure CA certificate matches between client and server
4. Verify server name matches certificate Subject Alternative Name (SAN)

### Connection Errors

If connections fail:
1. Check firewall settings
2. Verify gRPC server is listening on correct port
3. Ensure TLS is enabled on both client and server
4. Check certificate file paths are correct

## Testing

To test TLS connection:

```python
# Test with TLS
client = BlockchainClient(
    node_url="localhost:9090",
    use_tls=True,
    tls_cert_file="certs/client-cert.pem",
    tls_key_file="certs/client-key.pem",
    tls_ca_file="certs/ca-cert.pem",
)

# Try to connect
try:
    response = client.query_params()
    print("TLS connection successful!")
except Exception as e:
    print(f"TLS connection failed: {e}")
```

