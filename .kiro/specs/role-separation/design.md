# Design Document: R3MES Role Separation

## Overview

This design document describes the architectural changes needed to separate R3MES Network roles into independently deployable components. The primary goal is to enable the Backend API to run on standard VPS servers without GPU hardware, while maintaining full functionality for GPU-dependent roles (Miner, Serving Node).

### Current State

The Backend API (`backend/`) currently imports GPU-dependent libraries (torch, transformers, bitsandbytes) at startup, causing failures on GPU-less servers. All roles are loosely coupled but share dependencies inappropriately.

### Target State

```
┌─────────────────────────────────────────────────────────────────┐
│                    R3MES Network Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              INFRASTRUCTURE (No GPU Required)             │   │
│  │  ┌─────────────┐  ┌─────────┐  ┌───────┐  ┌─────────┐   │   │
│  │  │ Backend API │  │PostgreSQL│  │ Redis │  │  Nginx  │   │   │
│  │  │ (FastAPI)   │  │         │  │       │  │         │   │   │
│  │  └─────────────┘  └─────────┘  └───────┘  └─────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              │ WebSocket/REST                    │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                 PARTICIPANT ROLES                         │   │
│  │                                                           │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                │   │
│  │  │     MINER       │  │  SERVING NODE   │  GPU Required  │   │
│  │  │  (GPU + IPFS)   │  │  (GPU + API)    │                │   │
│  │  └─────────────────┘  └─────────────────┘                │   │
│  │                                                           │   │
│  │  ┌─────────────────┐  ┌─────────────────┐                │   │
│  │  │   VALIDATOR     │  │    PROPOSER     │  No GPU        │   │
│  │  │  (Blockchain)   │  │  (Aggregation)  │                │   │
│  │  └─────────────────┘  └─────────────────┘                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture

### Component Separation

| Component | Location | GPU Required | Deployment |
|-----------|----------|--------------|------------|
| Backend API | `backend/` | ❌ No | Docker/VPS |
| Miner Engine | `miner-engine/` | ✅ Yes | Desktop/Docker |
| Serving Node | `miner-engine/` (shared) | ✅ Yes | Docker/Bare metal |
| Validator | `remes/` (remesd binary) | ❌ No | Docker/VPS |
| Proposer | `remes/` (remesd binary) | ❌ No | Docker/VPS |

### Inference Mode Architecture

The Backend API will support three inference modes controlled by `R3MES_INFERENCE_MODE`:

```
┌─────────────────────────────────────────────────────────────┐
│                    Backend API Inference Modes               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  R3MES_INFERENCE_MODE=disabled (default)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • No AI libraries loaded                             │   │
│  │  • /chat endpoint returns 503 "Inference disabled"    │   │
│  │  • Minimal memory footprint                           │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  R3MES_INFERENCE_MODE=mock                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • No AI libraries loaded                             │   │
│  │  • /chat endpoint returns mock responses              │   │
│  │  • Useful for development/testing                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  R3MES_INFERENCE_MODE=remote                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • No AI libraries loaded                             │   │
│  │  • /chat endpoint proxies to Serving Nodes            │   │
│  │  • Load balancing across registered nodes             │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  R3MES_INFERENCE_MODE=local (requires GPU)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  • AI libraries loaded (torch, transformers)          │   │
│  │  • /chat endpoint runs inference locally              │   │
│  │  • Requires GPU hardware                              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### Backend API Component

```python
# backend/app/inference_mode.py

from enum import Enum
from typing import Optional
import os

class InferenceMode(Enum):
    DISABLED = "disabled"  # No inference, return 503
    MOCK = "mock"          # Return mock responses
    REMOTE = "remote"      # Proxy to serving nodes
    LOCAL = "local"        # Local GPU inference

def get_inference_mode() -> InferenceMode:
    """Get inference mode from environment variable."""
    mode = os.getenv("R3MES_INFERENCE_MODE", "disabled").lower()
    try:
        return InferenceMode(mode)
    except ValueError:
        return InferenceMode.DISABLED

def should_load_ai_libraries() -> bool:
    """Check if AI libraries should be loaded."""
    return get_inference_mode() == InferenceMode.LOCAL
```

### Lazy Loading Pattern

```python
# backend/app/model_manager.py (modified)

class AIModelManager:
    def __init__(self, base_model_path: str):
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self.base_model_path = base_model_path
    
    def _lazy_load(self):
        """Lazy load model only when needed."""
        if self._loaded:
            return
        
        from .inference_mode import should_load_ai_libraries
        if not should_load_ai_libraries():
            raise RuntimeError("AI libraries not loaded. Set R3MES_INFERENCE_MODE=local")
        
        # Import GPU libraries only when needed
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self._model = AutoModelForCausalLM.from_pretrained(self.base_model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model_path)
        self._loaded = True
```

### Serving Node Registry Interface

```python
# backend/app/serving_node_registry.py (interface)

class ServingNodeRegistry:
    async def register_node(self, node_id: str, endpoint: str, loras: List[str]) -> bool:
        """Register a serving node."""
        pass
    
    async def get_serving_nodes_for_lora(self, lora_name: str) -> List[ServingNode]:
        """Get available serving nodes for a specific LoRA."""
        pass
    
    async def proxy_inference(self, request: ChatRequest) -> AsyncGenerator[str, None]:
        """Proxy inference request to a serving node."""
        pass
```

### Role Installation Interfaces

```python
# Each role has a clear entry point

# Miner: miner-engine/r3mes/cli/main.py
# Command: r3mes-miner start

# Serving: miner-engine/r3mes/serving/main.py  
# Command: r3mes-serving start

# Validator: remes/cmd/remesd/main.go
# Command: remesd start --role validator

# Proposer: remes/cmd/remesd/main.go
# Command: remesd start --role proposer
```

## Data Models

### Environment Configuration

```python
# backend/app/env_config.py

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum

class InferenceMode(str, Enum):
    DISABLED = "disabled"
    MOCK = "mock"
    REMOTE = "remote"
    LOCAL = "local"

class BackendConfig(BaseModel):
    """Backend API configuration."""
    
    # Inference settings
    inference_mode: InferenceMode = Field(
        default=InferenceMode.DISABLED,
        description="Inference mode: disabled, mock, remote, or local"
    )
    
    # Database settings
    database_url: str = Field(..., description="PostgreSQL connection URL")
    
    # Redis settings
    redis_url: str = Field(default="redis://localhost:6379", description="Redis URL")
    
    # CORS settings
    cors_allowed_origins: str = Field(..., description="Comma-separated allowed origins")
    
    # Optional: Model settings (only for local mode)
    model_path: Optional[str] = Field(default=None, description="Path to model (local mode only)")
```

### Role Configuration

```python
# Stake requirements per role
ROLE_STAKE_REQUIREMENTS = {
    1: {"name": "Miner", "min_stake": 1000, "gpu_required": True},
    2: {"name": "Serving", "min_stake": 1000, "gpu_required": True},
    3: {"name": "Validator", "min_stake": 100000, "gpu_required": False},
    4: {"name": "Proposer", "min_stake": 50000, "gpu_required": False, "requires_validator": True},
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Inference Mode Behavior

*For any* valid inference mode setting (disabled, mock, remote), the Backend API SHALL start successfully without importing torch, transformers, bitsandbytes, or accelerate libraries.

**Validates: Requirements 1.1, 1.4, 1.5, 1.6**

### Property 2: GPU-less Graceful Degradation

*For any* inference request to the Backend API when inference mode is "disabled", the system SHALL return HTTP 503 with a JSON error message containing "inference" and "disabled" keywords.

**Validates: Requirements 1.2**

### Property 3: Requirements File GPU Independence

*For any* package listed in `requirements-api.txt`, the package name SHALL NOT be one of: torch, transformers, bitsandbytes, accelerate, peft, sentence-transformers.

**Validates: Requirements 1.3, 9.1, 9.5**

### Property 4: GPU Role Exit Behavior

*For any* GPU-dependent role (Miner, Serving Node) started on a system without GPU, the process SHALL exit with non-zero exit code and display an error message containing "GPU" or "CUDA".

**Validates: Requirements 3.4, 4.3**

### Property 5: Serving Node Registration Protocol

*For any* Serving Node that connects to the Backend API, the node SHALL send a registration message via WebSocket containing: node_id, endpoint, and available_loras list, followed by periodic heartbeat messages.

**Validates: Requirements 4.4, 4.5, 4.7**

### Property 6: Stake Requirement Enforcement

*For any* role registration attempt with stake amount less than the minimum required (Miner: 1000, Serving: 1000, Validator: 100000, Proposer: 50000), the blockchain SHALL reject the registration.

**Validates: Requirements 5.3, 6.2**

### Property 7: Proposer Validator Prerequisite

*For any* Proposer role registration attempt, the system SHALL verify the address already has Validator role, and reject if not.

**Validates: Requirements 6.3**

### Property 8: Environment Variable Validation

*For any* Backend API startup without required environment variables (DATABASE_URL, CORS_ALLOWED_ORIGINS in production), the system SHALL exit with non-zero code and display error message listing missing variables.

**Validates: Requirements 8.2, 8.3, 8.4**

### Property 9: Gradient Aggregation Correctness

*For any* set of valid gradients from multiple miners, the Proposer's aggregation function SHALL produce a result that is mathematically equivalent to the average of input gradients (within floating-point tolerance).

**Validates: Requirements 6.4**

## Error Handling

### Backend API Error Responses

| Scenario | HTTP Code | Error Message |
|----------|-----------|---------------|
| Inference disabled | 503 | `{"error": "Inference service disabled", "code": "INFERENCE_DISABLED"}` |
| No serving nodes available | 503 | `{"error": "No serving nodes available", "code": "NO_SERVING_NODES"}` |
| GPU not available (local mode) | 500 | `{"error": "GPU required for local inference", "code": "GPU_REQUIRED"}` |
| Missing env variable | 500 | `{"error": "Missing required environment variable: {var}", "code": "MISSING_ENV"}` |

### Role Startup Errors

| Role | Error Condition | Exit Code | Message |
|------|-----------------|-----------|---------|
| Miner | No GPU | 1 | "ERROR: GPU required for mining. No CUDA device found." |
| Serving | No GPU | 1 | "ERROR: GPU required for serving. No CUDA device found." |
| Validator | Insufficient stake | 1 | "ERROR: Minimum stake of 100,000 REMES required." |
| Proposer | No validator role | 1 | "ERROR: Validator role required before becoming Proposer." |

## Testing Strategy

### Unit Tests

Unit tests verify specific examples and edge cases:

1. **Inference mode parsing** - Test each mode string parses correctly
2. **Environment variable validation** - Test missing/invalid variables
3. **Error response format** - Test error JSON structure
4. **Stake calculation** - Test stake requirement checks

### Property-Based Tests

Property-based tests verify universal properties across all inputs using the `hypothesis` library for Python:

```python
# Test configuration
# Minimum 100 iterations per property test
# Tag format: Feature: role-separation, Property N: {property_text}
```

**Property Test Implementation:**

1. **Property 1 (Inference Mode)**: Generate random inference mode values, verify no GPU imports
2. **Property 3 (Requirements File)**: Parse requirements file, verify no GPU packages
3. **Property 4 (GPU Exit)**: Mock GPU detection, verify exit behavior
4. **Property 6 (Stake Requirements)**: Generate random stake amounts, verify rejection below minimum
5. **Property 8 (Env Validation)**: Generate random env var combinations, verify error messages

### Integration Tests

1. Backend API startup in each inference mode
2. Serving node registration and heartbeat flow
3. Docker compose infrastructure deployment
4. Role registration on blockchain

## File Structure Changes

```
backend/
├── requirements.txt          # Full requirements (backward compat)
├── requirements-api.txt      # API-only, no GPU libs (NEW)
├── requirements-inference.txt # GPU libs for local mode (NEW)
├── Dockerfile                # Current (will be renamed)
├── Dockerfile.api            # API-only image (NEW)
├── app/
│   ├── inference_mode.py     # Inference mode logic (NEW)
│   ├── main.py               # Modified for lazy loading
│   └── model_manager.py      # Modified for lazy loading

deploy/
├── docker-compose.production.yml      # Current
├── docker-compose.infrastructure.yml  # API + DB + Redis (NEW)

miner-engine/
├── Dockerfile.miner          # Miner image (NEW)
├── Dockerfile.serving        # Serving image (NEW)
├── docker-compose.miner.yml  # Miner deployment (NEW)

docs/
├── roles/
│   ├── miner-setup.md        # Miner installation guide (NEW)
│   ├── serving-setup.md      # Serving installation guide (NEW)
│   ├── validator-setup.md    # Validator installation guide (NEW)
│   └── proposer-setup.md     # Proposer installation guide (NEW)
```
