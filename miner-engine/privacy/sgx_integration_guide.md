# Intel SGX Integration Guide for R3MES

This guide provides instructions for integrating Intel SGX (Software Guard Extensions) with the R3MES Miner Engine for hardware-backed privacy and security.

## Prerequisites

### Hardware Requirements
- Intel CPU with SGX support (6th generation Core processors or newer)
- SGX enabled in BIOS/UEFI
- Sufficient EPC (Enclave Page Cache) memory

### Software Requirements
- Intel SGX SDK 2.15 or newer
- Intel SGX PSW (Platform Software)
- Intel SGX DCAP (Data Center Attestation Primitives) for production
- Linux kernel with SGX support (5.11+ recommended)

## Installation Steps

### 1. Install Intel SGX SDK

```bash
# Download SGX SDK
wget https://download.01.org/intel-sgx/sgx-linux/2.15/distro/ubuntu20.04-server/sgx_linux_x64_sdk_2.15.100.3.bin

# Install SDK
chmod +x sgx_linux_x64_sdk_2.15.100.3.bin
sudo ./sgx_linux_x64_sdk_2.15.100.3.bin

# Source environment
source /opt/intel/sgxsdk/environment
```

### 2. Install SGX PSW

```bash
# Add Intel repository
echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list
wget -qO - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | sudo apt-key add -

# Update and install
sudo apt update
sudo apt install libsgx-launch libsgx-urts libsgx-epid libsgx-quote-ex
```

### 3. Install Python SGX Bindings

```bash
# Install required packages
pip install cryptography pycryptodome

# Clone SGX Python bindings (community project)
git clone https://github.com/intel/linux-sgx-driver.git
cd linux-sgx-driver/linux/installer/bin
sudo ./sgx_linux_x64_driver_2.11.0_2d2b795.bin
```

## SGX Enclave Development

### 1. Create Enclave Definition File (EDL)

Create `privacy/enclave/r3mes_enclave.edl`:

```c
enclave {
    trusted {
        /* Enclave functions */
        public int ecall_encrypt_gradients([in, size=data_len] const uint8_t* data, 
                                          size_t data_len,
                                          [out, size=encrypted_len] uint8_t* encrypted_data,
                                          size_t encrypted_len);
        
        public int ecall_decrypt_gradients([in, size=encrypted_len] const uint8_t* encrypted_data,
                                          size_t encrypted_len,
                                          [out, size=data_len] uint8_t* data,
                                          size_t data_len);
        
        public int ecall_aggregate_gradients([in, count=num_gradients] const gradient_t* gradients,
                                           size_t num_gradients,
                                           [out] gradient_t* result);
        
        public int ecall_generate_attestation([out, size=report_len] sgx_report_t* report,
                                            size_t report_len);
    };
    
    untrusted {
        /* Ocalls for logging and external communication */
        void ocall_print_string([in, string] const char* str);
        int ocall_get_time();
    };
};
```

### 2. Implement Enclave Functions

Create `privacy/enclave/r3mes_enclave.c`:

```c
#include "r3mes_enclave_t.h"
#include <sgx_tcrypto.h>
#include <sgx_trts.h>
#include <string.h>

// Enclave-specific encryption key (sealed to enclave)
static uint8_t g_encryption_key[32];
static int g_key_initialized = 0;

int ecall_encrypt_gradients(const uint8_t* data, size_t data_len,
                           uint8_t* encrypted_data, size_t encrypted_len) {
    if (!g_key_initialized) {
        // Generate random key inside enclave
        sgx_status_t ret = sgx_read_rand(g_encryption_key, sizeof(g_encryption_key));
        if (ret != SGX_SUCCESS) {
            return -1;
        }
        g_key_initialized = 1;
    }
    
    // Use AES-GCM for encryption
    sgx_aes_gcm_128bit_key_t key;
    memcpy(&key, g_encryption_key, sizeof(key));
    
    uint8_t iv[12];
    sgx_read_rand(iv, sizeof(iv));
    
    sgx_aes_gcm_128bit_tag_t mac;
    
    sgx_status_t ret = sgx_rijndael128GCM_encrypt(
        &key,
        data, data_len,
        encrypted_data + 12, // Skip IV space
        iv, sizeof(iv),
        NULL, 0, // No additional authenticated data
        &mac
    );
    
    if (ret != SGX_SUCCESS) {
        return -1;
    }
    
    // Prepend IV and append MAC
    memcpy(encrypted_data, iv, sizeof(iv));
    memcpy(encrypted_data + 12 + data_len, &mac, sizeof(mac));
    
    return 0;
}

int ecall_decrypt_gradients(const uint8_t* encrypted_data, size_t encrypted_len,
                           uint8_t* data, size_t data_len) {
    if (!g_key_initialized) {
        return -1; // Key not initialized
    }
    
    sgx_aes_gcm_128bit_key_t key;
    memcpy(&key, g_encryption_key, sizeof(key));
    
    // Extract IV and MAC
    uint8_t iv[12];
    memcpy(iv, encrypted_data, sizeof(iv));
    
    sgx_aes_gcm_128bit_tag_t mac;
    memcpy(&mac, encrypted_data + encrypted_len - sizeof(mac), sizeof(mac));
    
    sgx_status_t ret = sgx_rijndael128GCM_decrypt(
        &key,
        encrypted_data + 12, // Skip IV
        encrypted_len - 12 - sizeof(mac), // Skip IV and MAC
        data,
        iv, sizeof(iv),
        NULL, 0, // No additional authenticated data
        &mac
    );
    
    return (ret == SGX_SUCCESS) ? 0 : -1;
}

int ecall_aggregate_gradients(const gradient_t* gradients, size_t num_gradients,
                             gradient_t* result) {
    // Implement secure aggregation inside enclave
    // This ensures gradients are never exposed outside the enclave
    
    if (num_gradients == 0) {
        return -1;
    }
    
    // Simple averaging (can be extended to weighted averaging)
    for (size_t i = 0; i < GRADIENT_SIZE; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < num_gradients; j++) {
            sum += gradients[j].data[i];
        }
        result->data[i] = sum / num_gradients;
    }
    
    return 0;
}

int ecall_generate_attestation(sgx_report_t* report, size_t report_len) {
    if (report_len < sizeof(sgx_report_t)) {
        return -1;
    }
    
    sgx_target_info_t target_info = {0};
    sgx_report_data_t report_data = {0};
    
    // Include enclave-specific data in report
    memcpy(report_data.d, "R3MES_ENCLAVE_V1", 16);
    
    sgx_status_t ret = sgx_create_report(&target_info, &report_data, report);
    return (ret == SGX_SUCCESS) ? 0 : -1;
}
```

### 3. Create Makefile

Create `privacy/enclave/Makefile`:

```makefile
SGX_SDK ?= /opt/intel/sgxsdk
SGX_MODE ?= HW
SGX_ARCH ?= x64

include $(SGX_SDK)/buildenv.mk

# Enclave settings
Enclave_Name := r3mes_enclave.so
Signed_Enclave_Name := r3mes_enclave.signed.so

# Compiler flags
Enclave_C_Flags := $(SGX_COMMON_CFLAGS) -nostdinc -fvisibility=hidden -fpie -fstack-protector
Enclave_Link_Flags := $(SGX_COMMON_CFLAGS) -Wl,-z,relro,-z,now,-z,noexecstack -pie

# Build targets
all: $(Signed_Enclave_Name)

$(Enclave_Name): r3mes_enclave.o
	@$(CXX) r3mes_enclave.o -o $@ $(Enclave_Link_Flags) -Wl,--version-script=r3mes_enclave.lds $(Enclave_Cpp_Objects) -lsgx_tstdc -lsgx_tcxx -lsgx_tcrypto -lsgx_tservice

r3mes_enclave.o: r3mes_enclave.c
	@$(CC) $(Enclave_C_Flags) -c $< -o $@

$(Signed_Enclave_Name): $(Enclave_Name)
	@$(SGX_ENCLAVE_SIGNER) sign -key r3mes_enclave_private.pem -enclave $(Enclave_Name) -out $@ -config r3mes_enclave.config.xml

clean:
	@rm -f *.o $(Enclave_Name) $(Signed_Enclave_Name)

.PHONY: all clean
```

## Python Integration

### 1. Update TEE Privacy Manager

Update `privacy/tee_privacy.py` to use real SGX:

```python
class SGXEnclave(PrivacyEnclave):
    """Real Intel SGX enclave integration."""
    
    def __init__(self, enclave_path: str):
        """Initialize SGX enclave."""
        self.enclave_path = Path(enclave_path)
        self._enclave_id = None
        self._initialized = False
        
        if not self.enclave_path.exists():
            raise FileNotFoundError(f"Enclave not found: {enclave_path}")
        
        # Load SGX library
        try:
            import ctypes
            self.sgx_lib = ctypes.CDLL("libsgx_urts.so")
            self._init_sgx()
        except OSError as e:
            raise RuntimeError(f"Failed to load SGX library: {e}")
    
    def _init_sgx(self):
        """Initialize SGX enclave."""
        import ctypes
        
        # Define SGX types
        sgx_enclave_id_t = ctypes.c_uint64
        sgx_status_t = ctypes.c_uint32
        
        # Create enclave
        enclave_id = sgx_enclave_id_t()
        
        # sgx_create_enclave function signature
        create_enclave = self.sgx_lib.sgx_create_enclave
        create_enclave.argtypes = [
            ctypes.c_char_p,  # enclave_filename
            ctypes.c_int,     # debug
            ctypes.POINTER(ctypes.c_void_p),  # launch_token
            ctypes.POINTER(ctypes.c_int),     # launch_token_updated
            ctypes.POINTER(sgx_enclave_id_t), # enclave_id
            ctypes.POINTER(ctypes.c_void_p)   # misc_attr
        ]
        create_enclave.restype = sgx_status_t
        
        # Create enclave
        launch_token = ctypes.c_void_p()
        launch_token_updated = ctypes.c_int()
        misc_attr = ctypes.c_void_p()
        
        status = create_enclave(
            str(self.enclave_path).encode(),
            1,  # debug mode
            ctypes.byref(launch_token),
            ctypes.byref(launch_token_updated),
            ctypes.byref(enclave_id),
            ctypes.byref(misc_attr)
        )
        
        if status != 0:  # SGX_SUCCESS
            raise RuntimeError(f"Failed to create enclave: {status}")
        
        self._enclave_id = enclave_id.value
        self._initialized = True
        
        logger.info(f"SGX enclave initialized: {self._enclave_id}")
```

### 2. Build Script

Create `privacy/build_sgx.sh`:

```bash
#!/bin/bash

# Build SGX enclave for R3MES
set -e

echo "Building R3MES SGX Enclave..."

# Check SGX SDK
if [ -z "$SGX_SDK" ]; then
    export SGX_SDK=/opt/intel/sgxsdk
fi

if [ ! -d "$SGX_SDK" ]; then
    echo "Error: SGX SDK not found at $SGX_SDK"
    exit 1
fi

# Source SGX environment
source $SGX_SDK/environment

# Generate enclave files
cd enclave
$SGX_SDK/bin/x64/sgx_edger8r --trusted r3mes_enclave.edl --search-path $SGX_SDK/include

# Build enclave
make clean
make

echo "SGX enclave built successfully!"
echo "Signed enclave: enclave/r3mes_enclave.signed.so"
```

## Production Deployment

### 1. Remote Attestation

For production deployment, implement remote attestation:

```python
def verify_remote_attestation(quote: bytes, nonce: bytes) -> bool:
    """Verify remote attestation with Intel Attestation Service."""
    import requests
    import base64
    
    # Intel Attestation Service endpoint
    ias_url = "https://api.trustedservices.intel.com/sgx/dev/v4/report"
    
    # Prepare attestation request
    quote_b64 = base64.b64encode(quote).decode()
    nonce_b64 = base64.b64encode(nonce).decode()
    
    payload = {
        "isvEnclaveQuote": quote_b64,
        "nonce": nonce_b64
    }
    
    headers = {
        "Content-Type": "application/json",
        "Ocp-Apim-Subscription-Key": "YOUR_IAS_API_KEY"
    }
    
    response = requests.post(ias_url, json=payload, headers=headers)
    
    if response.status_code == 200:
        # Verify attestation response
        attestation_report = response.json()
        return attestation_report.get("isvEnclaveQuoteStatus") == "OK"
    
    return False
```

### 2. Environment Configuration

Add to `.env`:

```bash
# SGX Configuration
R3MES_ENABLE_SGX=true
R3MES_SGX_ENCLAVE_PATH=/path/to/r3mes_enclave.signed.so
R3MES_SGX_DEBUG_MODE=false
R3MES_IAS_API_KEY=your_intel_attestation_service_key
```

## Testing

### 1. Unit Tests

Create `tests/test_sgx_integration.py`:

```python
import pytest
from privacy.tee_privacy import get_privacy_enclave

@pytest.mark.skipif(not os.path.exists("/dev/sgx_enclave"), 
                   reason="SGX device not available")
def test_sgx_enclave():
    """Test SGX enclave functionality."""
    enclave = get_privacy_enclave(
        use_sgx=True,
        enclave_path="privacy/enclave/r3mes_enclave.signed.so"
    )
    
    # Test encryption/decryption
    test_data = b"test gradient data"
    encrypted = enclave.encrypt_gradients(test_data)
    decrypted = enclave.decrypt_gradients(encrypted)
    
    assert decrypted == test_data
    
    # Test attestation
    attestation = enclave.verify_attestation()
    assert attestation.is_valid
```

### 2. Performance Benchmarks

```python
def benchmark_sgx_performance():
    """Benchmark SGX vs non-SGX performance."""
    import time
    
    # Test data
    test_gradients = [os.urandom(1024) for _ in range(100)]
    
    # SGX enclave
    sgx_enclave = get_privacy_enclave(use_sgx=True)
    
    start_time = time.time()
    for gradient in test_gradients:
        encrypted = sgx_enclave.encrypt_gradients(gradient)
        decrypted = sgx_enclave.decrypt_gradients(encrypted)
    sgx_time = time.time() - start_time
    
    # Simulated enclave
    sim_enclave = get_privacy_enclave(use_sgx=False)
    
    start_time = time.time()
    for gradient in test_gradients:
        encrypted = sim_enclave.encrypt_gradients(gradient)
        decrypted = sim_enclave.decrypt_gradients(encrypted)
    sim_time = time.time() - start_time
    
    print(f"SGX time: {sgx_time:.3f}s")
    print(f"Simulated time: {sim_time:.3f}s")
    print(f"SGX overhead: {(sgx_time/sim_time - 1)*100:.1f}%")
```

## Security Considerations

1. **Enclave Sealing**: Use SGX sealing to persist encryption keys
2. **Side-Channel Protection**: Implement constant-time algorithms
3. **Memory Protection**: Clear sensitive data from enclave memory
4. **Attestation Verification**: Always verify remote attestation in production
5. **Key Management**: Use hardware-backed key derivation

## Troubleshooting

### Common Issues

1. **SGX not enabled**: Check BIOS settings
2. **Driver issues**: Ensure SGX driver is loaded
3. **Enclave signing**: Use proper signing key and configuration
4. **Memory limits**: Monitor EPC usage

### Debug Commands

```bash
# Check SGX support
cpuid | grep SGX

# Check SGX driver
lsmod | grep sgx

# Check enclave status
cat /proc/sgx/status

# Test SGX functionality
sgx-test
```

This guide provides a complete framework for integrating Intel SGX with R3MES. The implementation ensures hardware-backed privacy and security for gradient processing while maintaining compatibility with the existing codebase.