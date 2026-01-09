//! Configuration management for R3MES Launcher
//! 
//! Handles environment variables and config file for network settings.
//! Supports both testnet and mainnet configurations.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;
use std::fs;

/// Canonical chain IDs - use these constants for consistency
pub mod chain_ids {
    pub const MAINNET: &str = "remes-1";
    pub const TESTNET: &str = "remes-testnet-1";
    pub const LOCAL: &str = "remes-local";
}

/// Model configuration constants
pub mod model_config {
    /// Expected size of BitNet b1.58 model in GB
    pub const BITNET_EXPECTED_SIZE_GB: f64 = 6.0;
    /// Minimum size to consider model as downloaded (allows for compression variance)
    pub const BITNET_MIN_SIZE_GB: f64 = 4.0;
    /// Model version string
    pub const BITNET_VERSION: &str = "BitNet b1.58";
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LauncherConfig {
    pub chain_id: String,
    pub rpc_url: String,
    pub grpc_url: String,
    pub rest_url: String,
    pub ipfs_url: String,
    pub ipfs_gateway_url: String,  // KRİTİK: IPFS gateway for downloading content
    pub web_dashboard_url: String,
}

/// Miner-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerConfig {
    pub wallet_address: String,
    pub backend_url: String,
    pub blockchain_rpc: String,
    pub ipfs_gateway: String,
    pub gpu_memory_limit: u32,
    pub batch_size: u32,
    pub gradient_accumulation_steps: u32,
    pub mixed_precision: bool,
    pub auto_restart: bool,
    pub log_level: String,
}

impl Default for MinerConfig {
    fn default() -> Self {
        // Use environment variables or sensible defaults
        // In production, these should come from config file or env vars
        let network_mode = env::var("R3MES_NETWORK")
            .unwrap_or_else(|_| "testnet".to_string())
            .to_lowercase();
        
        let (backend_url, blockchain_rpc) = if network_mode == "development" || network_mode == "dev" {
            ("http://localhost:8000".to_string(), "http://localhost:26657".to_string())
        } else if network_mode == "mainnet" {
            ("https://api.r3mes.network".to_string(), "https://rpc.r3mes.network".to_string())
        } else {
            // Default to testnet
            ("https://api.r3mes.network".to_string(), "https://rpc.r3mes.network".to_string())
        };
        
        Self {
            wallet_address: String::new(),
            backend_url,
            blockchain_rpc,
            ipfs_gateway: "http://localhost:5001".to_string(), // IPFS is always local
            gpu_memory_limit: 80,
            batch_size: 4,
            gradient_accumulation_steps: 4,
            mixed_precision: true,
            auto_restart: true,
            log_level: "INFO".to_string(),
        }
    }
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub chain_id: String,
    pub rpc_endpoint: String,
    pub rest_endpoint: String,
    pub grpc_endpoint: String,
    pub websocket_endpoint: String,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        // Use environment variables or sensible defaults based on network mode
        let network_mode = env::var("R3MES_NETWORK")
            .unwrap_or_else(|_| "testnet".to_string())
            .to_lowercase();
        
        if network_mode == "development" || network_mode == "dev" {
            Self {
                chain_id: chain_ids::LOCAL.to_string(),
                rpc_endpoint: "http://localhost:26657".to_string(),
                rest_endpoint: "http://localhost:1317".to_string(),
                grpc_endpoint: "localhost:9090".to_string(),
                websocket_endpoint: "ws://localhost:26657/websocket".to_string(),
            }
        } else if network_mode == "mainnet" {
            Self {
                chain_id: chain_ids::MAINNET.to_string(),
                rpc_endpoint: "https://rpc.r3mes.network".to_string(),
                rest_endpoint: "https://api.r3mes.network".to_string(),
                grpc_endpoint: "grpc.r3mes.network:9090".to_string(),
                websocket_endpoint: "wss://rpc.r3mes.network/websocket".to_string(),
            }
        } else {
            // Default to testnet
            Self {
                chain_id: chain_ids::TESTNET.to_string(),
                rpc_endpoint: "https://rpc.r3mes.network".to_string(),
                rest_endpoint: "https://rest.r3mes.network".to_string(),
                grpc_endpoint: "rpc.r3mes.network:9090".to_string(),
                websocket_endpoint: "wss://rpc.r3mes.network/websocket".to_string(),
            }
        }
    }
}

/// Advanced configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedConfig {
    pub max_workers: u32,
    pub checkpoint_interval: u32,
    pub telemetry_enabled: bool,
    pub debug_mode: bool,
    pub auto_update: bool,
    pub update_channel: String,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            max_workers: 1,
            checkpoint_interval: 100,
            telemetry_enabled: true,
            debug_mode: false,
            auto_update: true,
            update_channel: "stable".to_string(),
        }
    }
}

/// Full application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullConfig {
    pub miner: MinerConfig,
    pub network: NetworkConfig,
    pub advanced: AdvancedConfig,
}

impl Default for FullConfig {
    fn default() -> Self {
        Self {
            miner: MinerConfig::default(),
            network: NetworkConfig::default(),
            advanced: AdvancedConfig::default(),
        }
    }
}

impl FullConfig {
    /// Get the full config file path
    pub fn config_file_path() -> PathBuf {
        #[cfg(windows)]
        {
            let appdata = env::var("APPDATA").unwrap_or_else(|_| {
                env::var("USERPROFILE").unwrap_or_else(|_| "C:\\".to_string())
            });
            PathBuf::from(&appdata).join("R3MES").join("config.json")
        }
        
        #[cfg(not(windows))]
        {
            let home = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(&home).join(".r3mes").join("config.json")
        }
    }

    /// Load full configuration from file
    pub fn load() -> Self {
        let config_file = Self::config_file_path();
        
        if config_file.exists() {
            if let Ok(content) = fs::read_to_string(&config_file) {
                if let Ok(config) = serde_json::from_str(&content) {
                    return config;
                }
            }
        }
        
        Self::default()
    }

    /// Save full configuration to file
    pub fn save(&self) -> Result<(), String> {
        let config_file = Self::config_file_path();
        let config_dir = config_file.parent().ok_or("Invalid config path")?;
        
        fs::create_dir_all(config_dir)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        fs::write(&config_file, json)
            .map_err(|e| format!("Failed to write config file: {}", e))?;

        Ok(())
    }
}

impl LauncherConfig {
    /// Load configuration from environment variables or defaults
    pub fn load() -> Self {
        // Get network mode from environment (default: testnet for safety)
        let network_mode = env::var("R3MES_NETWORK")
            .unwrap_or_else(|_| "testnet".to_string())
            .to_lowercase();

        // Load from environment variables if set, otherwise use defaults based on network mode
        // Use canonical chain IDs from chain_ids module for consistency
        let (default_chain_id, default_rpc, default_grpc, default_rest, default_ipfs, default_ipfs_gateway, default_dashboard) = 
            if network_mode == "mainnet" {
                (
                    chain_ids::MAINNET,
                    "https://rpc.r3mes.network",
                    "rpc.r3mes.network:9090",
                    "https://api.r3mes.network",
                    "http://localhost:5001", // IPFS API stays local
                    "https://ipfs.r3mes.network", // IPFS gateway for downloads
                    "https://r3mes.network",
                )
            } else if network_mode == "development" || network_mode == "dev" {
                (
                    chain_ids::LOCAL,
                    "http://localhost:26657",
                    "localhost:9090",
                    "http://localhost:1317",
                    "http://localhost:5001",
                    "http://localhost:8080", // Local IPFS gateway
                    "http://localhost:3000",
                )
            } else {
                // Default to testnet
                (
                    chain_ids::TESTNET,
                    "https://rpc.r3mes.network",
                    "rpc.r3mes.network:9090",
                    "https://rest.r3mes.network",
                    "http://localhost:5001", // IPFS API stays local
                    "https://ipfs.r3mes.network", // IPFS gateway for downloads
                    "https://testnet.r3mes.network",
                )
            };

        Self {
            chain_id: env::var("R3MES_CHAIN_ID")
                .or_else(|_| env::var("CHAIN_ID"))
                .unwrap_or_else(|_| default_chain_id.to_string()),
            rpc_url: env::var("R3MES_RPC_URL")
                .or_else(|_| env::var("BLOCKCHAIN_RPC_URL"))
                .unwrap_or_else(|_| default_rpc.to_string()),
            grpc_url: env::var("R3MES_GRPC_URL")
                .or_else(|_| env::var("BLOCKCHAIN_GRPC_URL"))
                .unwrap_or_else(|_| default_grpc.to_string()),
            rest_url: env::var("R3MES_REST_URL")
                .or_else(|_| env::var("BLOCKCHAIN_REST_URL"))
                .unwrap_or_else(|_| default_rest.to_string()),
            ipfs_url: env::var("R3MES_IPFS_URL")
                .or_else(|_| env::var("IPFS_API_URL"))
                .unwrap_or_else(|_| default_ipfs.to_string()),
            ipfs_gateway_url: env::var("R3MES_IPFS_GATEWAY_URL")
                .or_else(|_| env::var("IPFS_GATEWAY_URL"))
                .unwrap_or_else(|_| default_ipfs_gateway.to_string()),
            web_dashboard_url: env::var("R3MES_WEB_DASHBOARD_URL")
                .unwrap_or_else(|_| default_dashboard.to_string()),
        }
    }

    /// Load configuration from config file (future: support for user settings)
    pub fn load_from_file() -> Option<Self> {
        let config_file = Self::config_file_path();

        if !config_file.exists() {
            return None;
        }

        let content = fs::read_to_string(&config_file).ok()?;
        serde_json::from_str(&content).ok()
    }

    /// Get the config file path
    pub fn config_file_path() -> PathBuf {
        #[cfg(windows)]
        {
            let appdata = env::var("APPDATA").unwrap_or_else(|_| {
                env::var("USERPROFILE").unwrap_or_else(|_| "C:\\".to_string())
            });
            PathBuf::from(&appdata).join("R3MES").join("launcher_config.json")
        }
        
        #[cfg(not(windows))]
        {
            let home = env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(&home).join(".r3mes").join("launcher_config.json")
        }
    }

    /// Save configuration to file
    pub fn save_to_file(&self) -> Result<(), String> {
        let config_file = Self::config_file_path();
        let config_dir = config_file.parent().ok_or("Invalid config path")?;
        
        fs::create_dir_all(config_dir)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        fs::write(&config_file, json)
            .map_err(|e| format!("Failed to write config file: {}", e))?;

        Ok(())
    }
}

// Global config instance (lazy static pattern)
lazy_static::lazy_static! {
    pub static ref CONFIG: LauncherConfig = LauncherConfig::load();
}

/// Get the current configuration
pub fn get_config() -> &'static LauncherConfig {
    &CONFIG
}

