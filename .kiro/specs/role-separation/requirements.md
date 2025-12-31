# Requirements Document

## Introduction

This document defines the requirements for separating R3MES Network roles into distinct, independently deployable components. The R3MES Network has 4 participant roles (Miner, Serving, Validator, Proposer) and infrastructure services (Backend API, Web Dashboard). Currently, the Backend API incorrectly bundles GPU-dependent AI libraries, preventing deployment on standard VPS servers. This separation will enable:

1. Clear installation flows for each role
2. GPU-less deployment of infrastructure services
3. Independent scaling of each component
4. Reduced resource requirements per component

## Glossary

- **Backend_API**: The FastAPI service that provides REST/WebSocket endpoints for the web dashboard and network participants. Infrastructure service, NOT a network role.
- **Miner**: Network participant role that trains AI models using GPU hardware and submits gradients to the blockchain.
- **Serving_Node**: Network participant role that provides AI inference services using GPU hardware.
- **Validator**: Network participant role that validates blockchain transactions and maintains consensus. No GPU required.
- **Proposer**: Network participant role that aggregates gradients from miners and proposes updates. No GPU required.
- **Infrastructure_Service**: Supporting services (Backend API, PostgreSQL, Redis, Nginx) that enable the network but are not participant roles.
- **GPU_Dependency**: Python libraries requiring CUDA/GPU hardware (torch, transformers, bitsandbytes, accelerate).
- **Desktop_Launcher**: Tauri-based desktop application for easy role setup and management.
- **Miner_Engine**: Python package in `miner-engine/` directory that implements the Miner role.

## Requirements

### Requirement 1: Backend API GPU Independence

**User Story:** As a system administrator, I want to deploy the Backend API on a standard VPS server without GPU, so that I can run infrastructure services cost-effectively.

#### Acceptance Criteria

1. WHEN the Backend_API starts without GPU hardware, THE Backend_API SHALL start successfully without errors
2. WHEN GPU-dependent features are requested on a GPU-less deployment, THE Backend_API SHALL return a graceful error message indicating the feature is unavailable
3. THE Backend_API SHALL have a separate requirements file (`requirements-api.txt`) that excludes GPU dependencies
4. WHEN `R3MES_INFERENCE_MODE` is set to "disabled", THE Backend_API SHALL skip loading AI model libraries entirely
5. WHEN `R3MES_INFERENCE_MODE` is set to "mock", THE Backend_API SHALL use mock responses for inference endpoints
6. WHEN `R3MES_INFERENCE_MODE` is set to "remote", THE Backend_API SHALL proxy inference requests to registered Serving_Nodes

### Requirement 2: Role-Specific Installation Flows

**User Story:** As a network participant, I want clear installation instructions for my specific role, so that I can set up my node without confusion.

#### Acceptance Criteria

1. THE Documentation SHALL provide separate installation guides for each role (Miner, Serving, Validator, Proposer)
2. WHEN a user wants to become a Miner, THE Documentation SHALL specify: GPU requirements, Desktop_Launcher download, wallet setup, staking process
3. WHEN a user wants to become a Serving_Node, THE Documentation SHALL specify: GPU requirements, model download, API endpoint configuration, staking process
4. WHEN a user wants to become a Validator, THE Documentation SHALL specify: server requirements (no GPU), blockchain node setup, staking process (100,000 REMES minimum)
5. WHEN a user wants to become a Proposer, THE Documentation SHALL specify: server requirements (no GPU), validator prerequisite, staking process (50,000 REMES minimum)
6. THE Documentation SHALL clearly indicate which roles require GPU hardware and which do not

### Requirement 3: Miner Role Separation

**User Story:** As a miner, I want a standalone installation package, so that I can run mining operations independently of other components.

#### Acceptance Criteria

1. THE Miner_Engine SHALL be installable as a standalone Python package via pip
2. WHEN a user installs the Miner_Engine, THE system SHALL install only miner-specific dependencies
3. THE Miner_Engine SHALL include GPU detection and validation on startup
4. IF GPU is not available, THEN THE Miner_Engine SHALL display a clear error message and exit
5. THE Miner_Engine SHALL connect to the blockchain via gRPC for gradient submission
6. THE Miner_Engine SHALL connect to IPFS for gradient storage
7. THE Desktop_Launcher SHALL provide a GUI for Miner setup and monitoring

### Requirement 4: Serving Node Role Separation

**User Story:** As a serving node operator, I want to run inference services independently, so that I can provide AI services to the network.

#### Acceptance Criteria

1. THE Serving_Node component SHALL be deployable as a standalone service
2. WHEN a Serving_Node starts, THE system SHALL validate GPU availability
3. IF GPU is not available, THEN THE Serving_Node SHALL display a clear error message and exit
4. THE Serving_Node SHALL register itself with the Backend_API via WebSocket
5. THE Serving_Node SHALL report available LoRA adapters to the Backend_API
6. WHEN inference requests arrive, THE Serving_Node SHALL process them and return results
7. THE Serving_Node SHALL send heartbeat messages to maintain active status

### Requirement 5: Validator Role Separation

**User Story:** As a validator, I want to run a blockchain validation node without GPU requirements, so that I can participate in consensus cost-effectively.

#### Acceptance Criteria

1. THE Validator component SHALL run without GPU hardware
2. THE Validator SHALL be the `remesd` binary from the `remes/` directory
3. WHEN a user sets up a Validator, THE system SHALL require minimum stake of 100,000 REMES
4. THE Validator SHALL participate in CometBFT consensus
5. THE Validator SHALL validate gradient submissions from miners
6. THE Documentation SHALL provide systemd service configuration for Validator

### Requirement 6: Proposer Role Separation

**User Story:** As a proposer, I want to aggregate gradients without GPU requirements, so that I can participate in model updates cost-effectively.

#### Acceptance Criteria

1. THE Proposer component SHALL run without GPU hardware
2. WHEN a user sets up a Proposer, THE system SHALL require minimum stake of 50,000 REMES
3. WHEN a user sets up a Proposer, THE system SHALL verify the user has Validator role first
4. THE Proposer SHALL aggregate gradients from multiple miners
5. THE Proposer SHALL submit aggregated updates to the blockchain
6. THE Documentation SHALL provide systemd service configuration for Proposer

### Requirement 7: Docker Deployment Separation

**User Story:** As a DevOps engineer, I want separate Docker configurations for each component, so that I can deploy only what I need.

#### Acceptance Criteria

1. THE project SHALL provide `Dockerfile.api` for Backend_API (no GPU dependencies)
2. THE project SHALL provide `Dockerfile.miner` for Miner_Engine (with GPU dependencies)
3. THE project SHALL provide `Dockerfile.serving` for Serving_Node (with GPU dependencies)
4. THE project SHALL provide `docker-compose.infrastructure.yml` for Backend_API + PostgreSQL + Redis + Nginx
5. THE project SHALL provide `docker-compose.miner.yml` for Miner with GPU support
6. WHEN deploying infrastructure only, THE system SHALL not require NVIDIA runtime

### Requirement 8: Configuration Management

**User Story:** As a system administrator, I want clear environment variable documentation, so that I can configure each component correctly.

#### Acceptance Criteria

1. THE project SHALL provide `.env.example` files for each component type
2. WHEN `R3MES_INFERENCE_MODE` is not set, THE Backend_API SHALL default to "disabled" mode
3. THE Backend_API SHALL validate required environment variables on startup
4. IF required environment variables are missing, THEN THE Backend_API SHALL display clear error messages
5. THE Documentation SHALL list all environment variables with descriptions for each component

### Requirement 9: Dependency Management

**User Story:** As a developer, I want clear separation of dependencies, so that I can understand what each component needs.

#### Acceptance Criteria

1. THE Backend_API SHALL have `requirements-api.txt` without GPU libraries
2. THE Backend_API SHALL have `requirements-inference.txt` with GPU libraries (optional)
3. THE Miner_Engine SHALL have its own `requirements.txt` with GPU libraries
4. THE Serving_Node SHALL share GPU dependencies with Miner_Engine or have its own requirements file
5. WHEN installing Backend_API for infrastructure-only deployment, THE system SHALL not install torch, transformers, bitsandbytes, or accelerate
