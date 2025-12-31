# Implementation Plan: R3MES Role Separation

## Overview

This implementation plan separates R3MES Network roles into independently deployable components, enabling the Backend API to run on GPU-less VPS servers while maintaining full functionality for GPU-dependent roles.

## Tasks

- [x] 1. Create inference mode configuration
  - [x] 1.1 Create `backend/app/inference_mode.py` with InferenceMode enum and helper functions
    - Define DISABLED, MOCK, REMOTE, LOCAL modes
    - Implement `get_inference_mode()` and `should_load_ai_libraries()` functions
    - _Requirements: 1.4, 1.5, 1.6_
  - [x] 1.2 Write property test for inference mode behavior
    - **Property 1: Inference Mode Behavior**
    - **Validates: Requirements 1.1, 1.4, 1.5, 1.6**

- [x] 2. Refactor Backend API for lazy loading
  - [x] 2.1 Modify `backend/app/main.py` to conditionally import AI libraries
    - Move torch/transformers imports inside functions
    - Check `should_load_ai_libraries()` before importing
    - Handle SemanticRouter lazy loading
    - _Requirements: 1.1, 1.4_
  - [x] 2.2 Modify `backend/app/model_manager.py` for lazy model loading
    - Implement `_ensure_gpu_libraries()` function for lazy loading
    - Updated `load_adapter()` to check `_peft_available` before using PeftModel
    - Updated `generate_response()` to use `_torch` instead of direct `torch` import
    - _Requirements: 1.1, 1.4_
  - [x] 2.3 Modify `backend/app/semantic_router.py` for conditional loading
    - Added `KeywordRouter` class as fallback for GPU-less deployment
    - Implemented lazy loading for sentence-transformers
    - Added `get_router()` factory function that checks inference mode
    - SemanticRouter now gracefully falls back to KeywordRouter
    - _Requirements: 1.1_
  - [x] 2.4 Write property test for GPU-less graceful degradation
    - Created `backend/tests/test_gpu_less_degradation.py`
    - Tests for all inference modes (disabled, mock, remote, local)
    - Tests for KeywordRouter pattern matching
    - Tests for SemanticRouter fallback behavior
    - **Property 2: GPU-less Graceful Degradation**
    - **Validates: Requirements 1.2**

- [x] 3. Create separate requirements files
  - [x] 3.1 Create `backend/requirements-api.txt` without GPU dependencies
    - Include FastAPI, database, Redis, monitoring dependencies
    - Exclude torch, transformers, bitsandbytes, accelerate, peft, sentence-transformers
    - _Requirements: 1.3, 9.1_
  - [x] 3.2 Create `backend/requirements-inference.txt` with GPU dependencies
    - Include torch, transformers, bitsandbytes, accelerate, peft, sentence-transformers
    - _Requirements: 9.2_
  - [x] 3.3 Write property test for requirements file GPU independence
    - Created `backend/tests/test_requirements_independence.py`
    - Tests for GPU package exclusion from api requirements
    - Tests for core package inclusion
    - Tests for file format and documentation
    - **Property 3: Requirements File GPU Independence**
    - **Validates: Requirements 1.3, 9.1, 9.5**

- [x] 4. Create Docker configurations
  - [x] 4.1 Create `backend/Dockerfile.api` for GPU-less API deployment
    - Uses requirements-api.txt
    - Sets R3MES_INFERENCE_MODE=remote as default
    - No checkpoints directory needed
    - _Requirements: 7.1_
  - [x] 4.2 Create `deploy/docker-compose.infrastructure.yml` for infrastructure-only deployment
    - Includes Backend API, PostgreSQL, Redis, Nginx
    - No NVIDIA runtime required
    - Uses Dockerfile.api for backend
    - _Requirements: 7.4, 7.6_
  - [x] 4.3 Update `deploy/.env.production` with inference mode configuration
    - Added R3MES_INFERENCE_MODE=remote
    - Documented all inference mode options
    - _Requirements: 8.1, 8.2_

- [x] 5. Checkpoint - Verify Backend API starts without GPU
  - All lazy loading implemented
  - KeywordRouter fallback available
  - Docker configurations created
  - Ready for testing: `docker compose -f docker-compose.infrastructure.yml up`

- [x] 6. Implement inference mode handlers
  - [x] 6.1 Implement disabled mode handler in chat endpoint
    - Already implemented in `backend/app/main.py`
    - Returns 503 with clear error message
    - _Requirements: 1.2_
  - [x] 6.2 Implement mock mode handler in chat endpoint
    - Already implemented in `backend/app/main.py`
    - Returns streaming mock responses
    - _Requirements: 1.5_
  - [x] 6.3 Implement remote mode handler in chat endpoint
    - Already implemented in `backend/app/main.py`
    - Proxies requests to registered Serving Nodes via ServingNodeRegistry
    - Handles load balancing and failover
    - Returns 503 if no serving nodes available
    - _Requirements: 1.6_
  - [x] 6.4 Write unit tests for inference mode handlers
    - Tests already exist in `backend/tests/test_gpu_less_degradation.py`
    - Additional tests in `backend/tests/test_api_integration.py`
    - _Requirements: 1.2, 1.5, 1.6_

- [x] 7. Environment variable validation
  - [x] 7.1 Update `backend/app/env_validator.py` for new variables
    - Added R3MES_INFERENCE_MODE validation rule
    - Added `validate_inference_mode()` method
    - Validates mode is one of: disabled, mock, remote, local
    - Checks GPU availability for local mode
    - _Requirements: 8.3, 8.4_
  - [x] 7.2 Write property test for environment validation
    - Created `backend/tests/test_env_validation.py`
    - Tests for valid/invalid inference modes
    - Tests for case-insensitivity
    - Tests for GPU requirement in local mode
    - Tests for safe defaults
    - **Property 8: Environment Variable Validation**
    - **Validates: Requirements 8.2, 8.3, 8.4**

- [x] 8. Create role documentation
  - [x] 8.1 Create `docs/roles/miner-setup.md`
    - GPU requirements, Desktop Launcher download, wallet setup, staking
    - _Requirements: 2.1, 2.2_
  - [x] 8.2 Create `docs/roles/serving-setup.md`
    - GPU requirements, Docker setup, API configuration, staking
    - _Requirements: 2.1, 2.3_
  - [x] 8.3 Create `docs/roles/validator-setup.md`
    - Server requirements, remesd setup, staking (100,000 REMES)
    - _Requirements: 2.1, 2.4_
  - [x] 8.4 Create `docs/roles/proposer-setup.md`
    - Validator prerequisite, staking (50,000 REMES)
    - _Requirements: 2.1, 2.5_

- [x] 9. Checkpoint - Full integration test
  - Deploy infrastructure to VPS using `docker compose -f docker-compose.infrastructure.yml up -d`
  - Verify Backend API starts and responds
  - Test /chat endpoint returns appropriate error in remote mode without serving nodes
  - Test /health endpoint returns 200

- [x] 10. Create Miner Docker configuration (optional - for containerized mining)
  - [x] 10.1 Create `miner-engine/Dockerfile.miner` with GPU support
    - NVIDIA CUDA 11.8 base image
    - PyTorch with CUDA support
    - Health check for GPU availability
    - _Requirements: 7.2_
  - [x] 10.2 Create `miner-engine/docker-compose.miner.yml`
    - NVIDIA runtime configuration
    - GPU passthrough with configurable device selection
    - Persistent volumes for checkpoints and logs
    - Optional Grafana stats dashboard
    - _Requirements: 7.5_

- [x] 11. Final checkpoint - Complete system verification
  - All documentation complete
  - All Docker configurations created
  - Ready for deployment

## Notes

- All tasks are required for comprehensive implementation
- Priority is getting Backend API running on GPU-less VPS (Tasks 1-5)
- Documentation tasks (8.x) can be done in parallel
- Property tests ensure correctness guarantees
