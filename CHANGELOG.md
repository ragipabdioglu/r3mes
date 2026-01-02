# Changelog

All notable changes to the R3MES project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### IBC Modules Re-enablement - Senior Level (2 Ocak 2026)
- **Full IBC Support**: Cross-chain communication enabled
  - IBC Core Keeper - Inter-blockchain communication core
  - IBC Transfer - Cross-chain token transfers
  - ICA Controller - Interchain accounts controller
  - ICA Host - Interchain accounts host
  - Capability Keeper - IBC capability management
- **Keeper Methods**: Missing genesis and aggregation methods added
  - `InitGenesis` - Initialize module state from genesis
  - `ExportGenesis` - Export module state to genesis
  - `FinalizeExpiredAggregations` - Finalize expired training aggregations
- **Dashboard API**: REST API endpoints for web dashboard
  - `/api/dashboard/stats` - Network statistics
  - `/api/dashboard/blocks` - Recent blocks
  - `/api/dashboard/miners` - Miner information
  - `/api/dashboard/validators` - Validator information
  - `/api/dashboard/governance` - Governance proposals
  - `/api/dashboard/health` - Health check
- **WebSocket Support**: Real-time updates via WebSocket
- **Sentry Integration**: Error tracking with optional Sentry support
- **Proto Files Fixed**: Regenerated with buf, fixed file descriptors
  - Updated `min_stake` type from `github.com/cosmos/cosmos-sdk/types.Int` to `cosmossdk.io/math.Int`
  - Fixed non-constant format string errors in error constructors
  - Fixed integration tests with proper multistore setup
- **All Tests Passing**: 9 tests passing (7 keeper + 2 types)
- **Performance Benchmarking**: Comprehensive benchmarks added
  - Keeper Memory: 22.96 KB (very efficient)
  - Average Throughput: 336,521 ops/sec
  - Params Get: 905,223 ops/sec
  - Node Registration: 373,092 ops/sec
- **Build Verified**: Production build successful

#### CLI Tools v0.3.0 - Senior Level Upgrade (2 Ocak 2026)
- **Cobra/Viper Framework**: Professional CLI framework integration
  - Full Cobra command structure with subcommands
  - Viper configuration management (YAML, ENV, flags)
  - Auto-generated shell completions (bash, zsh, fish, powershell)
- **Transaction Signing**: Real implementation for on-chain transactions
  - `tx send` - Send tokens with proper signing
  - `tx sign` - Offline transaction signing
  - `tx broadcast` - Broadcast signed transactions
  - Governance voting now fully functional with signing
- **Context & Graceful Shutdown**: Signal handling (SIGINT, SIGTERM)
  - All commands respect context cancellation
  - Clean shutdown for long-running operations
- **Structured Logging**: Zap logger integration
  - JSON and console output formats
  - Debug/Info/Warn/Error levels
  - Verbose mode flag
- **JSON Output Mode**: `--json` flag for all commands
- **Watch Modes**: `miner stats --watch`, `node sync --watch`
- **Version Bump**: v0.2.0 → v0.3.0

#### Web Dashboard - Full Tailwind Migration (2 Ocak 2026)
- **CSS → Tailwind Migration**: All 8 component CSS files migrated
  - MinersTable.tsx - Miner leaderboard with tier badges
  - ValidatorList.tsx - Validator list with trust scores
  - StakingDashboard.tsx - Staking overview and delegations
  - GovernancePanel.tsx - Governance proposals and voting
  - CreateProposalModal.tsx - Proposal creation modal
  - DelegateForm.tsx - Delegation form modal
  - NetworkStats.tsx - Network statistics cards
  - RecentBlocks.tsx - Recent blocks grid
- **Deleted CSS Files**: 8 CSS files removed from styles/components/
- **Build Verified**: Production build successful

#### CLI Tools v0.2.0 - Modular Architecture (2 Ocak 2026)
- **Modular Command Structure**: CLI refactored from monolithic to modular architecture
  - `cmd/config.go` - Configuration management
  - `cmd/wallet.go` - Wallet operations (create, import, balance, export, list)
  - `cmd/miner.go` - Miner operations (start, stop, status, stats)
  - `cmd/node.go` - Node operations (start, stop, status, sync)
  - `cmd/governance.go` - Governance operations (vote, proposals, proposal)
- **Comprehensive Test Suite**: Added unit tests for all modules
  - `cmd/wallet_test.go` - Wallet encryption, address generation tests
  - `cmd/config_test.go` - Configuration and environment variable tests
- **Improved Code Organization**: Separation of concerns for better maintainability
- **Version Bump**: v0.1.0 → v0.2.0

#### Blockchain Node - Keeper Refactoring Complete (2 Ocak 2026)
- **Domain-Based Separation**: 50+ collections split into 8 domain-specific keepers
  - `core/keeper.go` - Core functionality (params, nonces, timestamps)
  - `model/keeper.go` - Model registry and versioning
  - `training/keeper.go` - Gradient submissions and aggregation
  - `node/keeper.go` - Node registration and management
  - `dataset/keeper.go` - Dataset proposals and governance
  - `economics/keeper.go` - Rewards and treasury management
  - `security/keeper.go` - Trap jobs, challenges, fraud detection
  - `infra/keeper.go` - IPFS integration and caching
- **Interface-Based Design**: Clean interfaces for each keeper domain
- **Backward Compatibility**: Legacy methods maintained for gradual migration

#### Web Dashboard Improvements (2 Ocak 2026)
- **Mock API → Backend Integration**: 13 mock functions connected to real endpoints
- **Hook Integrations**: useCSRF, useAnnouncer, useVirtualization integrated across pages
- **Accessibility**: Screen reader announcements and ARIA attributes on 7 pages
- **Redis-Ready Rate Limiter**: Production-ready rate limiting with Redis support
- **Formatters Utility**: 10 common formatting functions centralized

#### Frontend Features
- **Faucet UI**: Complete faucet interface for claiming free tokens
  - Wallet address input with auto-fill from connected wallet
  - Optional amount specification
  - Rate limit information and next claim time display
  - Transaction hash display with copy functionality
  - Success/error message handling
  - Navigation link added to main navbar

- **Staking Dashboard**: Full staking interface with claim rewards
  - Staking overview cards (total staked, pending rewards, active delegations)
  - Validator list with delegate/undelegate/redelegate actions
  - Claim rewards functionality (withdraws from all validators)
  - Integration with Keplr wallet for transaction signing
  - Navigation link added to main navbar

#### Backend Features
- **Notification Service Expansion**: Comprehensive alerting system
  - Database connection failure notifications
  - Blockchain connection failure notifications
  - High error rate monitoring and alerts
  - Health check failure notifications
  - Configurable notification channels (email, Slack, in-app)
  - Error rate monitor with configurable thresholds

- **Semantic Router Activation**: Enabled by default
  - Embedding-based intelligent routing for LoRA adapter selection
  - Improved error handling with graceful fallback
  - CPU-friendly model (all-MiniLM-L6-v2) - no CUDA required
  - Configurable similarity threshold

- **Blockchain Indexer Improvements**: Enhanced indexing capabilities
  - Transaction-level event parsing (Cosmos SDK events)
  - Batch processing for better performance
  - Indexing lag monitoring
  - Health check endpoint (`/health/indexer`)
  - Configurable batch size

#### Documentation
- **Feature Documentation**:
  - `docs/faucet.md` - Complete faucet feature documentation
  - `docs/staking.md` - Complete staking feature documentation
  - `remes/docs/HANDLERS.md` - Blockchain handler implementation status
  - `docs/FEATURE_FLAGS.md` - Feature flags and configuration guide

- **API Documentation**:
  - `docs/api/faucet.md` - Faucet API reference
  - `docs/api/staking.md` - Staking API reference

- **Deployment Documentation Updates**:
  - Updated `docs/12_production_deployment.md` with notification service, semantic router, and indexer information
  - Updated `backend/README.md` with semantic router and notification service configuration

### Changed

- **Semantic Router**: Default changed from `false` to `true`
  - Now enabled by default for better adapter selection
  - Improved error handling and fallback mechanism
  - Better logging and initialization messages

- **Notification Service**: Expanded to cover more system events
  - Database connection failures now trigger critical alerts
  - Blockchain connection failures now trigger critical alerts
  - Error rate monitoring added with configurable thresholds
  - Health check failures now trigger critical alerts

- **Blockchain Indexer**: Enhanced with transaction-level parsing
  - Now parses Cosmos SDK events from transactions
  - Batch processing for improved performance
  - Indexing lag tracking and monitoring

### Fixed

- **Staking Dashboard**: Claim rewards functionality implemented
  - Previously showed "coming soon" alert
  - Now properly withdraws rewards from all validators
  - Uses Cosmos SDK `MsgWithdrawDelegatorReward` messages

- **Faucet UI**: Complete implementation
  - Previously only backend endpoint existed
  - Now has full frontend interface with proper error handling

- **Semantic Router**: Improved error handling
  - Better fallback to keyword router on initialization failure
  - Improved error messages and logging
  - CPU-only operation (no CUDA dependency for semantic router itself)

### Security

- **Notification Service**: Secure alerting for critical system events
  - Database connection failures are immediately reported
  - Blockchain connection failures are immediately reported
  - High error rates trigger alerts before service degradation

### Performance

- **Blockchain Indexer**: Batch processing for better performance
  - Processes blocks in configurable batches (default: 10)
  - Reduces database connection overhead
  - Improves indexing throughput

### Configuration

- **Environment Variables Added**:
  - `USE_SEMANTIC_ROUTER` (default: `true`) - Enable/disable semantic router
  - `SEMANTIC_ROUTER_THRESHOLD` (default: `0.7`) - Similarity threshold
  - `NOTIFICATION_CHANNELS` (default: `email,slack`) - Notification channels
  - `ERROR_RATE_THRESHOLD` (default: `0.1`) - Error rate threshold
  - `ERROR_RATE_CHECK_INTERVAL` (default: `60`) - Check interval in seconds
  - `ERROR_RATE_MIN_REQUESTS` (default: `100`) - Minimum requests before alerting
  - `INDEXER_BATCH_SIZE` (default: `10`) - Indexer batch size
  - `FAUCET_ENABLED` (default: `true`) - Enable/disable faucet
  - `FAUCET_AMOUNT` (default: `1000000uremes`) - Default faucet amount
  - `FAUCET_DAILY_LIMIT` (default: `5000000uremes`) - Daily faucet limit

## [Previous Versions]

See git history for previous changelog entries.

---

## Migration Guide

### Upgrading to This Version

1. **Update Environment Variables**:
   - Review `docker/env.production.example` for new variables
   - Add notification service configuration if desired
   - Configure semantic router threshold if needed

2. **Update Docker Secrets**:
   - Run `bash scripts/create_secrets.sh` to ensure all secrets are created
   - Secrets are now required for Grafana admin password

3. **Review Feature Flags**:
   - Check `docs/FEATURE_FLAGS.md` for all available flags
   - Configure according to your deployment needs

4. **Test New Features**:
   - Test faucet functionality
   - Test staking and claim rewards
   - Verify notification service is working
   - Check semantic router is functioning correctly

### Breaking Changes

None in this release.

### Deprecations

- Keyword router is now deprecated in favor of semantic router
  - Still available as fallback
  - Will be removed in a future version

---

## Contributors

- Development Team
- Community Contributors

---

## Links

- [Documentation](docs/)
- [API Reference](docs/api/)
- [Deployment Guide](docs/12_production_deployment.md)
- [Feature Flags](docs/FEATURE_FLAGS.md)

