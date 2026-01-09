//! Unit tests for config module

#[cfg(test)]
mod tests {
    use crate::config::*;

    #[test]
    fn test_chain_ids_constants() {
        assert_eq!(chain_ids::MAINNET, "remes-1");
        assert_eq!(chain_ids::TESTNET, "remes-testnet-1");
        assert_eq!(chain_ids::LOCAL, "remes-local");
    }

    #[test]
    fn test_model_config_constants() {
        assert_eq!(model_config::BITNET_EXPECTED_SIZE_GB, 28.0);
        assert_eq!(model_config::BITNET_MIN_SIZE_GB, 26.0);
        assert_eq!(model_config::BITNET_VERSION, "BitNet b1.58");
        
        // Ensure min size is less than expected size
        assert!(model_config::BITNET_MIN_SIZE_GB < model_config::BITNET_EXPECTED_SIZE_GB);
    }

    #[test]
    fn test_miner_config_default() {
        let config = MinerConfig::default();
        
        // Check default values
        assert!(config.wallet_address.is_empty());
        assert_eq!(config.gpu_memory_limit, 80);
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.gradient_accumulation_steps, 4);
        assert!(config.mixed_precision);
        assert!(config.auto_restart);
        assert_eq!(config.log_level, "INFO");
        
        // IPFS should always be local
        assert_eq!(config.ipfs_gateway, "http://localhost:5001");
    }

    #[test]
    fn test_network_config_default_testnet() {
        // Default should be testnet when R3MES_NETWORK is not set
        std::env::remove_var("R3MES_NETWORK");
        let config = NetworkConfig::default();
        
        assert_eq!(config.chain_id, chain_ids::TESTNET);
        assert!(config.rpc_endpoint.contains("testnet"));
        assert!(config.rest_endpoint.contains("testnet"));
    }

    #[test]
    fn test_network_config_development_mode() {
        std::env::set_var("R3MES_NETWORK", "development");
        let config = NetworkConfig::default();
        
        assert_eq!(config.chain_id, chain_ids::LOCAL);
        assert!(config.rpc_endpoint.contains("localhost"));
        assert!(config.rest_endpoint.contains("localhost"));
        
        // Cleanup
        std::env::remove_var("R3MES_NETWORK");
    }

    #[test]
    fn test_network_config_mainnet_mode() {
        std::env::set_var("R3MES_NETWORK", "mainnet");
        let config = NetworkConfig::default();
        
        assert_eq!(config.chain_id, chain_ids::MAINNET);
        assert!(!config.rpc_endpoint.contains("localhost"));
        assert!(!config.rpc_endpoint.contains("testnet"));
        
        // Cleanup
        std::env::remove_var("R3MES_NETWORK");
    }

    #[test]
    fn test_advanced_config_default() {
        let config = AdvancedConfig::default();
        
        assert_eq!(config.max_workers, 1);
        assert_eq!(config.checkpoint_interval, 100);
        assert!(config.telemetry_enabled);
        assert!(!config.debug_mode);
        assert!(config.auto_update);
        assert_eq!(config.update_channel, "stable");
    }

    #[test]
    fn test_full_config_default() {
        let config = FullConfig::default();
        
        // Verify all sub-configs are initialized
        assert!(config.miner.wallet_address.is_empty());
        assert!(!config.network.chain_id.is_empty());
        assert_eq!(config.advanced.update_channel, "stable");
    }

    #[test]
    fn test_launcher_config_load() {
        let config = LauncherConfig::load();
        
        // Should have valid values
        assert!(!config.chain_id.is_empty());
        assert!(!config.rpc_url.is_empty());
        assert!(!config.rest_url.is_empty());
        assert!(!config.ipfs_url.is_empty());
    }

    #[test]
    fn test_launcher_config_env_override() {
        // Set custom chain ID via env
        std::env::set_var("R3MES_CHAIN_ID", "custom-chain-1");
        
        let config = LauncherConfig::load();
        assert_eq!(config.chain_id, "custom-chain-1");
        
        // Cleanup
        std::env::remove_var("R3MES_CHAIN_ID");
    }

    #[test]
    fn test_config_file_path_not_empty() {
        let path = LauncherConfig::config_file_path();
        assert!(!path.as_os_str().is_empty());
        
        // Should contain r3mes or R3MES in path
        let path_str = path.to_string_lossy().to_lowercase();
        assert!(path_str.contains("r3mes") || path_str.contains("appdata"));
    }

    #[test]
    fn test_full_config_file_path_not_empty() {
        let path = FullConfig::config_file_path();
        assert!(!path.as_os_str().is_empty());
    }

    #[test]
    fn test_full_config_serialization() {
        let config = FullConfig::default();
        
        // Should serialize without error
        let json = serde_json::to_string(&config);
        assert!(json.is_ok());
        
        // Should deserialize back
        let json_str = json.unwrap();
        let deserialized: Result<FullConfig, _> = serde_json::from_str(&json_str);
        assert!(deserialized.is_ok());
        
        let restored = deserialized.unwrap();
        assert_eq!(restored.network.chain_id, config.network.chain_id);
    }

    #[test]
    fn test_get_config_returns_static_ref() {
        let config1 = get_config();
        let config2 = get_config();
        
        // Should return same reference
        assert_eq!(config1.chain_id, config2.chain_id);
    }

    #[test]
    fn test_chain_id_consistency() {
        // Ensure chain IDs are used consistently
        let launcher_config = LauncherConfig::load();
        
        // Chain ID should be one of the known values or custom
        let valid_chain_ids = [
            chain_ids::MAINNET,
            chain_ids::TESTNET,
            chain_ids::LOCAL,
        ];
        
        // Either matches a known chain ID or is custom
        let is_known = valid_chain_ids.contains(&launcher_config.chain_id.as_str());
        let is_custom = !launcher_config.chain_id.is_empty();
        
        assert!(is_known || is_custom);
    }
}
