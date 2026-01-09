# Security & Verification

R3MES implements a robust security model with three-layer optimistic verification, trap job detection, and comprehensive authentication mechanisms.

---

## Overview

The R3MES verification system is designed to balance speed and security through a layered approach:

| Layer | Purpose | Speed | Cost |
|-------|---------|-------|------|
| Layer 1 | GPU-to-GPU optimistic verification | ~1 block | Minimal |
| Layer 2 | Loss-based spot checking | ~5 blocks | Challenger bond required |
| Layer 3 | CPU Iron Sandbox arbitration | ~50 blocks | High (validator panel) |

---

## Three-Layer Verification System

### Layer 1: Optimistic Verification

The default verification path for all gradient submissions.

| Characteristic | Value |
|----------------|-------|
| Speed | ~1 block |
| Cost | Minimal (hash comparison) |
| Success Rate | ~95% of cases |
| Slashing | None on acceptance |

**Process:**
1. Miner submits gradient with hash
2. System compares hash with expected value
3. If match → Accept optimistically
4. If mismatch → Mark for potential challenge

### Layer 2: Loss-Based Spot Checking

Triggered when Layer 1 hash mismatch occurs and a challenger disputes.

| Characteristic | Value |
|----------------|-------|
| Speed | ~5 blocks |
| Challenger Bond | 10x base reward (5,000 REMES) |
| Verification Method | Forward pass inference |
| Cost Advantage | ~100x cheaper than full training |

**Key Innovation:** Instead of re-running expensive full training, validators perform a forward pass (inference) on a deterministic random batch to verify the miner's claimed loss.

**Process:**
1. Random verifier selected via VRF
2. Verifier downloads miner's weights from IPFS
3. Deterministic batch selected using VRF seed
4. Forward pass executed (NOT full training)
5. Loss compared within BitNet integer tolerance

### Layer 3: CPU Iron Sandbox

Final arbiter for disputed verifications.

| Characteristic | Value |
|----------------|-------|
| Speed | ~50 blocks |
| Panel Size | 3 validators (VRF selection) |
| Consensus | 2/3 agreement required |
| Execution | Mandatory CPU mode (bit-exact) |

**Trigger Conditions:**
- Layer 2 consensus supports the challenge
- Cross-architecture disputes requiring deterministic verification

---

## GPU Architecture Handling

Different GPU architectures (e.g., RTX 3090 Ampere vs RTX 4090 Ada) can produce microscopic floating-point differences in CUDA kernels.

### Architecture Detection

| Compute Capability | Architecture |
|-------------------|--------------|
| 6.0, 6.1 | Pascal |
| 7.0 | Volta |
| 7.5 | Turing |
| 8.0, 8.6 | Ampere |
| 8.9 | Ada |
| 9.0 | Blackwell |

### Verification Rules

| Scenario | Action |
|----------|--------|
| Same architecture, hash match | Accept (exact match) |
| Same architecture, hash mismatch | Likely fraud → CPU verification |
| Different architectures | Mandatory CPU verification |

---

## Trap Job Security

Trap jobs are pre-computed "Golden Vectors" used to detect lazy or malicious miners.

### Security Model

| Feature | Description |
|---------|-------------|
| Multi-Sig | Requires 2/3 signatures from top 3 validators |
| Pre-Computed | Golden Vectors computed offline, not on-demand |
| Encrypted | Expected gradients encrypted, only validators can decrypt |
| Blinded | Cryptographic blinding prevents miner identification |

### Trap Job Detection

Miners cannot distinguish trap jobs from real work (Panopticon effect):
- `is_trap` flag never exposed to miners
- Statistical characteristics match normal jobs
- Dummy metadata injection for obfuscation

### Verification Process

1. Decrypt expected answer (Golden Vector)
2. Unblind miner's result
3. Compare using cosine similarity
4. Threshold: similarity ≥ 0.95, norm difference ≤ 0.05
5. Failure → 100% slash (LAZY_MINING)

---

## Tolerant Verification

Handles hardware-induced gradient variations while catching cheaters.

### Masking Method

The masking method solves the "Top-K shift" problem caused by hardware differences:

| Approach | Problem | Solution |
|----------|---------|----------|
| Direct fingerprint comparison | Top-K indices may shift due to noise | Fails honest miners |
| Masking/Projection | Extract values at vault's indices | Hardware-agnostic comparison |

**Process:**
1. Get vault indices (e.g., [5, 100, 999])
2. Download miner's full gradient tensor
3. Extract miner's values at vault indices
4. Calculate cosine similarity between vault values and extracted values

### Similarity Thresholds

| Scenario | Threshold |
|----------|-----------|
| Same GPU | 0.999 |
| Different GPU | 0.95 |

---

## Authentication & Authorization

### Message Signing

All miner submissions require cryptographic signatures:
- ECDSA with secp256k1 curve
- SHA256 message hashing
- Nonce/challenge-response for replay protection

### Rate Limiting

| Limit Type | Value |
|------------|-------|
| Per Block | 1 submission per miner |
| Per Minute | 10 submissions per miner |
| Window | Sliding 10-block window |

**Features:**
- Submission history tracking per miner
- Automatic cleanup of old entries
- Multi-level protection (block + window)

### TLS Mutual Authentication

- mTLS for Python miner to Go node communication
- Client certificates required for gradient submission

---

## Security Testing

### Penetration Testing

| Test Category | Coverage |
|---------------|----------|
| SQL Injection | API endpoint protection |
| XSS Protection | Input sanitization |
| Rate Limiting | Request throttling |
| CORS Configuration | Origin validation |
| Authentication Bypass | Access control |
| API Key Security | Key validation |
| CSRF Protection | Token verification |
| Security Headers | HTTP header hardening |

### Economic Attack Testing

| Attack Type | Defense |
|-------------|---------|
| Sybil Attack | Same wallet detection |
| Gradient Manipulation | Malicious gradient rejection |
| Collusion Attack | Identical gradient detection |
| Nothing-at-Stake | Staking requirements |
| Long-Range Attack | Old block prevention |

---

## Security Checklist

| Category | Status |
|----------|--------|
| SQL injection protection | ✅ Parameterized queries |
| XSS protection | ✅ Input sanitization |
| CSRF protection | ✅ Origin validation |
| Rate limiting | ✅ Per IP/API key |
| CORS configuration | ✅ Restricted origins |
| Security headers | ✅ X-Content-Type-Options, X-Frame-Options |
| API key hashing | ✅ SHA256 |
| Input validation | ✅ Pydantic models |
| Authentication | ✅ Wallet signatures |
| Economic attack prevention | ✅ Staking, reputation |

---

## Next Steps

- [Governance System →](governance) - Protocol governance
- [Tokenomics →](tokenomics) - Economic model
- [API Reference →](api-reference) - Technical integration

---

**Need Help?** [Join Discord](https://discord.gg/r3mes) | [GitHub](https://github.com/r3mes-network/r3mes)
