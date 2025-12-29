#!/bin/bash
# Generate TLS certificates for mTLS between Python miner and Go node
# This script generates:
# - CA certificate and key
# - Server certificate and key (for Go node)
# - Client certificate and key (for Python miner)

set -e

CERT_DIR="${CERT_DIR:-./certs}"
CA_KEY="${CERT_DIR}/ca-key.pem"
CA_CERT="${CERT_DIR}/ca-cert.pem"
SERVER_KEY="${CERT_DIR}/server-key.pem"
SERVER_CERT="${CERT_DIR}/server-cert.pem"
SERVER_CSR="${CERT_DIR}/server.csr"
CLIENT_KEY="${CERT_DIR}/client-key.pem"
CLIENT_CERT="${CERT_DIR}/client-cert.pem"
CLIENT_CSR="${CERT_DIR}/client.csr"

# Create cert directory
mkdir -p "${CERT_DIR}"

echo "Generating TLS certificates for mTLS..."

# 1. Generate CA private key
echo "1. Generating CA private key..."
openssl genrsa -out "${CA_KEY}" 4096

# 2. Generate CA certificate
echo "2. Generating CA certificate..."
openssl req -new -x509 -days 365 -key "${CA_KEY}" -out "${CA_CERT}" \
    -subj "/C=US/ST=State/L=City/O=R3MES/CN=R3MES-CA"

# 3. Generate server private key
echo "3. Generating server private key..."
openssl genrsa -out "${SERVER_KEY}" 4096

# 4. Generate server certificate signing request
echo "4. Generating server certificate signing request..."
openssl req -new -key "${SERVER_KEY}" -out "${SERVER_CSR}" \
    -subj "/C=US/ST=State/L=City/O=R3MES/CN=remes-node"

# 5. Generate server certificate (signed by CA)
echo "5. Generating server certificate..."
openssl x509 -req -days 365 -in "${SERVER_CSR}" -CA "${CA_CERT}" -CAkey "${CA_KEY}" \
    -CAcreateserial -out "${SERVER_CERT}" \
    -extensions v3_req -extfile <(
        echo "[v3_req]"
        echo "subjectAltName = @alt_names"
        echo "[alt_names]"
        echo "DNS.1 = localhost"
        echo "IP.1 = 127.0.0.1"
    )

# 6. Generate client private key
echo "6. Generating client private key..."
openssl genrsa -out "${CLIENT_KEY}" 4096

# 7. Generate client certificate signing request
echo "7. Generating client certificate signing request..."
openssl req -new -key "${CLIENT_KEY}" -out "${CLIENT_CSR}" \
    -subj "/C=US/ST=State/L=City/O=R3MES/CN=python-miner"

# 8. Generate client certificate (signed by CA)
echo "8. Generating client certificate..."
openssl x509 -req -days 365 -in "${CLIENT_CSR}" -CA "${CA_CERT}" -CAkey "${CA_KEY}" \
    -CAcreateserial -out "${CLIENT_CERT}" \
    -extensions v3_req -extfile <(
        echo "[v3_req]"
        echo "subjectAltName = @alt_names"
        echo "[alt_names]"
        echo "DNS.1 = python-miner"
    )

# Clean up CSR files
rm -f "${SERVER_CSR}" "${CLIENT_CSR}"

echo ""
echo "TLS certificates generated successfully!"
echo ""
echo "Certificate files:"
echo "  CA Certificate: ${CA_CERT}"
echo "  CA Key: ${CA_KEY}"
echo "  Server Certificate: ${SERVER_CERT}"
echo "  Server Key: ${SERVER_KEY}"
echo "  Client Certificate: ${CLIENT_CERT}"
echo "  Client Key: ${CLIENT_KEY}"
echo ""
echo "To use these certificates:"
echo "  - Go node: Use ${SERVER_CERT} and ${SERVER_KEY} with CA ${CA_CERT}"
echo "  - Python miner: Use ${CLIENT_CERT} and ${CLIENT_KEY} with CA ${CA_CERT}"

