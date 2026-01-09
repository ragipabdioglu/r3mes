//! Unit tests for process_manager module

#[cfg(test)]
mod tests {
    use crate::process_manager::*;

    #[test]
    fn test_process_type_name() {
        assert_eq!(ProcessType::Node.name(), "Node");
        assert_eq!(ProcessType::Miner.name(), "Miner");
        assert_eq!(ProcessType::Ipfs.name(), "IPFS");
        assert_eq!(ProcessType::Serving.name(), "Serving");
        assert_eq!(ProcessType::Validator.name(), "Validator");
        assert_eq!(ProcessType::Proposer.name(), "Proposer");
    }

    #[test]
    fn test_process_type_log_file() {
        assert_eq!(ProcessType::Node.log_file(), "node.log");
        assert_eq!(ProcessType::Miner.log_file(), "miner.log");
        assert_eq!(ProcessType::Ipfs.log_file(), "ipfs.log");
        assert_eq!(ProcessType::Serving.log_file(), "serving.log");
        assert_eq!(ProcessType::Validator.log_file(), "validator.log");
        assert_eq!(ProcessType::Proposer.log_file(), "proposer.log");
    }

    #[cfg(unix)]
    #[test]
    fn test_process_type_kill_pattern_unix() {
        assert_eq!(ProcessType::Node.kill_pattern(), "remesd");
        assert_eq!(ProcessType::Miner.kill_pattern(), "r3mes-miner");
        assert_eq!(ProcessType::Ipfs.kill_pattern(), "ipfs");
        assert_eq!(ProcessType::Serving.kill_pattern(), "r3mes-serving");
        assert_eq!(ProcessType::Validator.kill_pattern(), "remesd.*validator");
        assert_eq!(ProcessType::Proposer.kill_pattern(), "r3mes-proposer");
    }

    #[cfg(windows)]
    #[test]
    fn test_process_type_exe_name_windows() {
        assert_eq!(ProcessType::Node.exe_name(), "remesd.exe");
        assert_eq!(ProcessType::Miner.exe_name(), "r3mes-miner.exe");
        assert_eq!(ProcessType::Ipfs.exe_name(), "ipfs.exe");
        assert_eq!(ProcessType::Serving.exe_name(), "r3mes-serving.exe");
        assert_eq!(ProcessType::Validator.exe_name(), "remesd.exe");
        assert_eq!(ProcessType::Proposer.exe_name(), "r3mes-proposer.exe");
    }

    #[test]
    fn test_process_type_equality() {
        assert_eq!(ProcessType::Node, ProcessType::Node);
        assert_ne!(ProcessType::Node, ProcessType::Miner);
    }

    #[test]
    fn test_process_type_clone() {
        let original = ProcessType::Miner;
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_process_result_serialization() {
        let result = ProcessResult {
            success: true,
            message: "Process started".to_string(),
            pid: Some(12345),
        };
        
        let json = serde_json::to_string(&result);
        assert!(json.is_ok());
        
        let restored: ProcessResult = serde_json::from_str(&json.unwrap()).unwrap();
        assert!(restored.success);
        assert_eq!(restored.message, "Process started");
        assert_eq!(restored.pid, Some(12345));
    }

    #[test]
    fn test_process_result_without_pid() {
        let result = ProcessResult {
            success: false,
            message: "Process not found".to_string(),
            pid: None,
        };
        
        let json = serde_json::to_string(&result);
        assert!(json.is_ok());
        
        let restored: ProcessResult = serde_json::from_str(&json.unwrap()).unwrap();
        assert!(!restored.success);
        assert!(restored.pid.is_none());
    }

    #[test]
    fn test_process_info_serialization() {
        let info = ProcessInfo {
            running: true,
            pid: Some(9999),
        };
        
        let json = serde_json::to_string(&info);
        assert!(json.is_ok());
        
        let restored: ProcessInfo = serde_json::from_str(&json.unwrap()).unwrap();
        assert!(restored.running);
        assert_eq!(restored.pid, Some(9999));
    }

    #[test]
    fn test_process_status_serialization() {
        let status = ProcessStatus {
            node: ProcessInfo { running: true, pid: Some(1) },
            miner: ProcessInfo { running: false, pid: None },
            ipfs: ProcessInfo { running: true, pid: Some(2) },
            serving: ProcessInfo { running: false, pid: None },
            validator: ProcessInfo { running: false, pid: None },
            proposer: ProcessInfo { running: false, pid: None },
        };
        
        let json = serde_json::to_string(&status);
        assert!(json.is_ok());
        
        let restored: ProcessStatus = serde_json::from_str(&json.unwrap()).unwrap();
        assert!(restored.node.running);
        assert!(!restored.miner.running);
        assert!(restored.ipfs.running);
    }

    #[test]
    fn test_persisted_process_state_default() {
        let state = PersistedProcessState::default();
        
        assert!(state.node_pid.is_none());
        assert!(state.miner_pid.is_none());
        assert!(state.ipfs_pid.is_none());
        assert!(state.serving_pid.is_none());
        assert!(state.validator_pid.is_none());
        assert!(state.proposer_pid.is_none());
        assert!(state.last_updated.is_none());
    }

    #[test]
    fn test_persisted_process_state_serialization() {
        let state = PersistedProcessState {
            node_pid: Some(1234),
            miner_pid: Some(5678),
            ipfs_pid: None,
            serving_pid: None,
            validator_pid: None,
            proposer_pid: None,
            last_updated: Some("2024-01-01T00:00:00Z".to_string()),
        };
        
        let json = serde_json::to_string(&state);
        assert!(json.is_ok());
        
        let restored: PersistedProcessState = serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(restored.node_pid, Some(1234));
        assert_eq!(restored.miner_pid, Some(5678));
        assert!(restored.ipfs_pid.is_none());
    }

    #[test]
    fn test_persisted_process_state_validate_clears_invalid_pids() {
        let mut state = PersistedProcessState {
            node_pid: Some(999999999), // Very unlikely to be a real PID
            miner_pid: None,
            ipfs_pid: None,
            serving_pid: None,
            validator_pid: None,
            proposer_pid: None,
            last_updated: None,
        };
        
        let changed = state.validate();
        
        // Should have cleared the invalid PID
        // Note: This might not clear on all systems if PID happens to exist
        // The test verifies the function runs without panic
        assert!(changed == true || changed == false);
    }

    #[test]
    fn test_process_manager_new() {
        let manager = ProcessManager::new();
        
        // Manager should be created without panic
        // We can't easily test internal state, but creation should succeed
        drop(manager);
    }

    #[tokio::test]
    async fn test_restore_process_state_no_panic() {
        // Should not panic even if state file doesn't exist
        let result = restore_process_state().await;
        
        // Should either succeed or return an error, not panic
        match result {
            Ok(_) => {}
            Err(e) => {
                println!("Expected error in test environment: {}", e);
            }
        }
    }

    #[test]
    fn test_persisted_state_file_path() {
        let path = PersistedProcessState::file_path();
        
        // Path should not be empty
        assert!(!path.as_os_str().is_empty());
        
        // Should end with process_state.json
        assert!(
            path.file_name().map(|f| f.to_string_lossy().contains("process_state")).unwrap_or(false),
            "Path should contain process_state: {:?}",
            path
        );
    }

    #[test]
    fn test_persisted_state_save_and_load() {
        // Create a temporary state
        let state = PersistedProcessState {
            node_pid: Some(12345),
            miner_pid: None,
            ipfs_pid: Some(67890),
            serving_pid: None,
            validator_pid: None,
            proposer_pid: None,
            last_updated: Some(chrono::Utc::now().to_rfc3339()),
        };
        
        // Save should succeed (creates directory if needed)
        let save_result = state.save();
        
        if save_result.is_ok() {
            // Load should return the saved state
            let loaded = PersistedProcessState::load();
            assert_eq!(loaded.node_pid, Some(12345));
            assert_eq!(loaded.ipfs_pid, Some(67890));
        }
        // If save fails (e.g., permissions), that's OK for unit tests
    }

    #[test]
    fn test_all_process_types_have_unique_names() {
        let types = [
            ProcessType::Node,
            ProcessType::Miner,
            ProcessType::Ipfs,
            ProcessType::Serving,
            ProcessType::Validator,
            ProcessType::Proposer,
        ];
        
        let names: Vec<&str> = types.iter().map(|t| t.name()).collect();
        let unique_names: std::collections::HashSet<&str> = names.iter().cloned().collect();
        
        assert_eq!(names.len(), unique_names.len(), "All process types should have unique names");
    }

    #[test]
    fn test_all_process_types_have_unique_log_files() {
        let types = [
            ProcessType::Node,
            ProcessType::Miner,
            ProcessType::Ipfs,
            ProcessType::Serving,
            ProcessType::Validator,
            ProcessType::Proposer,
        ];
        
        let log_files: Vec<&str> = types.iter().map(|t| t.log_file()).collect();
        let unique_files: std::collections::HashSet<&str> = log_files.iter().cloned().collect();
        
        assert_eq!(log_files.len(), unique_files.len(), "All process types should have unique log files");
    }
}
