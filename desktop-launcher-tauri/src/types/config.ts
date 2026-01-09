/**
 * R3MES Desktop Launcher - Configuration Types
 * 
 * TypeScript type definitions that mirror the Rust config module.
 * These ensure type safety between frontend and backend.
 */

// Chain ID constants - must match config.rs chain_ids module
export const CHAIN_IDS = {
  MAINNET: 'remes-1',
  TESTNET: 'remes-testnet-1',
  LOCAL: 'remes-local',
} as const;

export type ChainId = typeof CHAIN_IDS[keyof typeof CHAIN_IDS];

// Model configuration constants - must match config.rs model_config module
export const MODEL_CONFIG = {
  BITNET_EXPECTED_SIZE_GB: 28.0,
  BITNET_MIN_SIZE_GB: 26.0,
  BITNET_VERSION: 'BitNet b1.58',
} as const;

// Network presets - matches config.rs NetworkConfig defaults
export const NETWORK_PRESETS = {
  local: {
    chain_id: CHAIN_IDS.LOCAL,
    rpc_endpoint: 'http://localhost:26657',
    rest_endpoint: 'http://localhost:1317',
    grpc_endpoint: 'localhost:9090',
    websocket_endpoint: 'ws://localhost:26657/websocket',
  },
  testnet: {
    chain_id: CHAIN_IDS.TESTNET,
    rpc_endpoint: 'https://rpc.r3mes.network',
    rest_endpoint: 'https://rest.r3mes.network',
    grpc_endpoint: 'rpc.r3mes.network:9090',
    websocket_endpoint: 'wss://rpc.r3mes.network/websocket',
    backend_url: 'https://api.r3mes.network',
  },
  mainnet: {
    chain_id: CHAIN_IDS.MAINNET,
    rpc_endpoint: 'https://rpc.r3mes.network',
    rest_endpoint: 'https://api.r3mes.network',
    grpc_endpoint: 'grpc.r3mes.network:9090',
    websocket_endpoint: 'wss://rpc.r3mes.network/websocket',
  },
} as const;

export type NetworkPreset = keyof typeof NETWORK_PRESETS;

// Miner configuration interface
export interface MinerConfig {
  wallet_address: string;
  backend_url: string;
  blockchain_rpc: string;
  ipfs_gateway: string;
  gpu_memory_limit: number;
  batch_size: number;
  gradient_accumulation_steps: number;
  mixed_precision: boolean;
  auto_restart: boolean;
  log_level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
}

// Network configuration interface
export interface NetworkConfig {
  chain_id: ChainId | string;
  rpc_endpoint: string;
  rest_endpoint: string;
  grpc_endpoint: string;
  websocket_endpoint: string;
}

// Advanced configuration interface
export interface AdvancedConfig {
  max_workers: number;
  checkpoint_interval: number;
  telemetry_enabled: boolean;
  debug_mode: boolean;
  auto_update: boolean;
  update_channel: 'stable' | 'beta' | 'nightly';
}

// Full configuration interface
export interface FullConfig {
  miner: MinerConfig;
  network: NetworkConfig;
  advanced: AdvancedConfig;
}

// Process types - matches ProcessType enum in process_manager.rs
export const PROCESS_TYPES = {
  NODE: 'node',
  MINER: 'miner',
  IPFS: 'ipfs',
  SERVING: 'serving',
  VALIDATOR: 'validator',
  PROPOSER: 'proposer',
} as const;

export type ProcessType = typeof PROCESS_TYPES[keyof typeof PROCESS_TYPES];

// Process status interface
export interface ProcessInfo {
  running: boolean;
  pid: number | null;
}

export interface ProcessStatus {
  node: ProcessInfo;
  miner: ProcessInfo;
  ipfs: ProcessInfo;
  serving: ProcessInfo;
  validator: ProcessInfo;
  proposer: ProcessInfo;
}

// System status interface - matches get_system_status response
export interface SystemStatus {
  chain_sync: {
    synced: boolean;
    percentage: number;
    block_height: number | null;
    latest_block_height: number | null;
  };
  ipfs: {
    connected: boolean;
    peers: number;
    status: string;
  };
  model: {
    downloaded: boolean;
    progress: number;
    file_name: string | null;
    file_size_gb: number | null;
    integrity_verified: boolean;
  };
  node: {
    running: boolean;
    rpc_endpoint: string;
    grpc_endpoint: string;
    last_block_time: number | null;
  };
}

// Model status interface
export interface ModelStatus {
  downloaded: boolean;
  progress: number;
  version: string;
  size_gb: number;
  status: 'ready' | 'downloading' | 'not_downloaded';
}

// Wallet interfaces
export interface WalletInfo {
  address: string;
  balance: string;
  exists: boolean;
}

export interface Transaction {
  hash: string;
  from: string;
  to: string;
  amount: string;
  timestamp: number;
  status: 'pending' | 'confirmed' | 'failed';
  block_height?: number;
}

// Hardware check result
export interface HardwareCheckResult {
  docker: {
    installed: boolean;
    running: boolean;
    version: string | null;
  };
  gpu: {
    available: boolean;
    name: string | null;
    vram_gb: number | null;
    driver_version: string | null;
  };
  disk: {
    available_gb: number;
    required_gb: number;
    sufficient: boolean;
  };
  ram: {
    total_gb: number;
    minimum_gb: number;
    sufficient: boolean;
  };
  cuda: {
    installed: boolean;
    version: string | null;
    compatible: boolean;
  };
  all_checks_passed: boolean;
}

// Mining stats interface
export interface MiningStats {
  hashrate: number;
  loss: number;
  loss_trend: 'decreasing' | 'increasing' | 'stable';
  estimated_earnings_per_day: number;
  current_balance: number;
  gpu_temp: number;
  gpu_temp_status: 'normal' | 'high' | 'critical';
  vram_usage_mb: number;
  vram_total_mb: number;
  training_epoch: number;
  gradient_norm: number;
  uptime_seconds: number;
}

// Helper functions
export function isValidChainId(chainId: string): chainId is ChainId {
  return Object.values(CHAIN_IDS).includes(chainId as ChainId);
}

export function getNetworkPreset(preset: NetworkPreset): NetworkConfig {
  return { ...NETWORK_PRESETS[preset] };
}

export function getDefaultMinerConfig(): MinerConfig {
  return {
    wallet_address: '',
    backend_url: NETWORK_PRESETS.testnet.rest_endpoint,
    blockchain_rpc: NETWORK_PRESETS.testnet.rpc_endpoint,
    ipfs_gateway: 'http://localhost:5001',
    gpu_memory_limit: 80,
    batch_size: 4,
    gradient_accumulation_steps: 4,
    mixed_precision: true,
    auto_restart: true,
    log_level: 'INFO',
  };
}

export function getDefaultNetworkConfig(): NetworkConfig {
  return getNetworkPreset('testnet');
}

export function getDefaultAdvancedConfig(): AdvancedConfig {
  return {
    max_workers: 1,
    checkpoint_interval: 100,
    telemetry_enabled: true,
    debug_mode: false,
    auto_update: true,
    update_channel: 'stable',
  };
}

export function getDefaultFullConfig(): FullConfig {
  return {
    miner: getDefaultMinerConfig(),
    network: getDefaultNetworkConfig(),
    advanced: getDefaultAdvancedConfig(),
  };
}
