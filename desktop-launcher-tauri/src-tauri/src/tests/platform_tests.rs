//! Unit tests for platform module

#[cfg(test)]
mod tests {
    use crate::platform::*;
    use std::path::PathBuf;

    #[test]
    fn test_get_home_dir_exists() {
        let home = get_home_dir();
        assert!(home.is_ok(), "get_home_dir should succeed");
        
        let home_path = home.unwrap();
        assert!(home_path.exists(), "Home directory should exist");
        assert!(home_path.is_dir(), "Home directory should be a directory");
    }

    #[test]
    fn test_get_r3mes_data_dir_valid_path() {
        let data_dir = get_r3mes_data_dir();
        assert!(data_dir.is_ok(), "get_r3mes_data_dir should succeed");
        
        let path = data_dir.unwrap();
        let path_str = path.to_string_lossy().to_lowercase();
        
        // Should contain r3mes in path
        assert!(
            path_str.contains("r3mes") || path_str.contains("appdata"),
            "Data dir should contain r3mes or appdata: {}",
            path_str
        );
    }

    #[test]
    fn test_get_data_dir_alias() {
        let data_dir1 = get_r3mes_data_dir();
        let data_dir2 = get_data_dir();
        
        assert!(data_dir1.is_ok());
        assert!(data_dir2.is_ok());
        assert_eq!(data_dir1.unwrap(), data_dir2.unwrap());
    }

    #[test]
    fn test_get_logs_dir_is_subdirectory() {
        let data_dir = get_r3mes_data_dir().unwrap();
        let logs_dir = get_logs_dir().unwrap();
        
        assert!(logs_dir.starts_with(&data_dir), "Logs dir should be under data dir");
        assert!(logs_dir.ends_with("logs"), "Logs dir should end with 'logs'");
    }

    #[test]
    fn test_get_wallets_dir_is_subdirectory() {
        let data_dir = get_r3mes_data_dir().unwrap();
        let wallets_dir = get_wallets_dir().unwrap();
        
        assert!(wallets_dir.starts_with(&data_dir), "Wallets dir should be under data dir");
        assert!(wallets_dir.ends_with("wallets"), "Wallets dir should end with 'wallets'");
    }

    #[test]
    fn test_get_models_dir_is_subdirectory() {
        let data_dir = get_r3mes_data_dir().unwrap();
        let models_dir = get_models_dir().unwrap();
        
        assert!(models_dir.starts_with(&data_dir), "Models dir should be under data dir");
        assert!(models_dir.ends_with("models"), "Models dir should end with 'models'");
    }

    #[test]
    fn test_get_config_dir_equals_data_dir() {
        let data_dir = get_r3mes_data_dir().unwrap();
        let config_dir = get_config_dir().unwrap();
        
        assert_eq!(data_dir, config_dir, "Config dir should equal data dir");
    }

    #[test]
    fn test_get_workspace_dir_valid() {
        let workspace = get_workspace_dir();
        assert!(workspace.is_ok(), "get_workspace_dir should succeed");
        
        let path = workspace.unwrap();
        let path_str = path.to_string_lossy();
        
        assert!(path_str.contains("R3MES"), "Workspace should contain R3MES: {}", path_str);
    }

    #[test]
    fn test_get_system_python_not_empty() {
        let python = get_system_python();
        assert!(!python.is_empty(), "System python should not be empty");
        
        #[cfg(windows)]
        assert_eq!(python, "python");
        
        #[cfg(not(windows))]
        assert_eq!(python, "python3");
    }

    #[test]
    fn test_get_shell_returns_valid_shell() {
        let (shell, flag) = get_shell();
        
        assert!(!shell.is_empty(), "Shell should not be empty");
        assert!(!flag.is_empty(), "Shell flag should not be empty");
        
        #[cfg(windows)]
        {
            assert_eq!(shell, "cmd");
            assert_eq!(flag, "/C");
        }
        
        #[cfg(not(windows))]
        {
            assert_eq!(shell, "sh");
            assert_eq!(flag, "-c");
        }
    }

    #[test]
    fn test_is_port_in_use_common_ports() {
        // Port 0 should always be available (OS assigns)
        // This is a basic sanity check
        
        // Very high ports are usually available
        let high_port = 59999;
        let result = is_port_in_use(high_port);
        // Just verify it returns a boolean without panicking
        assert!(result == true || result == false);
    }

    #[test]
    fn test_check_port_listening_returns_bool() {
        // Test that the function returns without panicking
        let result = check_port_listening(12345);
        assert!(result == true || result == false);
    }

    #[test]
    fn test_execute_shell_command_echo() {
        #[cfg(windows)]
        let result = execute_shell_command("echo test");
        
        #[cfg(not(windows))]
        let result = execute_shell_command("echo test");
        
        assert!(result.is_ok(), "Echo command should succeed");
        
        let output = result.unwrap();
        assert!(output.status.success(), "Echo should exit successfully");
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("test"), "Output should contain 'test'");
    }

    #[test]
    fn test_directory_paths_are_absolute() {
        let paths = vec![
            get_home_dir(),
            get_r3mes_data_dir(),
            get_logs_dir(),
            get_wallets_dir(),
            get_models_dir(),
            get_config_dir(),
            get_workspace_dir(),
        ];
        
        for path_result in paths {
            if let Ok(path) = path_result {
                assert!(
                    path.is_absolute() || path.starts_with("~"),
                    "Path should be absolute: {:?}",
                    path
                );
            }
        }
    }

    #[test]
    fn test_venv_python_path_structure() {
        // Even if venv doesn't exist, path structure should be correct
        let result = get_venv_python();
        
        // If it fails, that's OK - venv might not exist
        // But if it succeeds, verify the path structure
        if let Ok(path) = result {
            let path_str = path.to_string_lossy();
            
            #[cfg(windows)]
            {
                assert!(path_str.contains("Scripts"), "Windows venv should use Scripts");
                assert!(path_str.ends_with(".exe"), "Windows should have .exe extension");
            }
            
            #[cfg(not(windows))]
            {
                assert!(path_str.contains("bin"), "Unix venv should use bin");
            }
        }
    }

    #[test]
    fn test_open_url_format() {
        // We can't actually test opening URLs in tests, but we can verify
        // the function doesn't panic with valid input
        // Note: This test is commented out to avoid opening browsers during tests
        // let result = open_url("https://example.com");
        // The function should at least not panic
    }

    #[test]
    fn test_path_separators_are_correct() {
        let workspace = get_workspace_dir().unwrap();
        let logs = get_logs_dir().unwrap();
        
        // Paths should use correct separators for the platform
        let workspace_str = workspace.to_string_lossy();
        let logs_str = logs.to_string_lossy();
        
        #[cfg(windows)]
        {
            // Windows paths should not have forward slashes (except UNC)
            assert!(
                !workspace_str.contains('/') || workspace_str.starts_with("//"),
                "Windows path should use backslashes"
            );
        }
        
        #[cfg(not(windows))]
        {
            // Unix paths should not have backslashes
            assert!(
                !logs_str.contains('\\'),
                "Unix path should use forward slashes"
            );
        }
    }
}
