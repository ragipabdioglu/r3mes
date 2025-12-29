/// Debug utilities for R3MES Desktop Launcher
///
/// Provides debug logging configuration, process state inspection, and system resource monitoring.

use std::env;
use std::process::Command;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    pub enabled: bool,
    pub log_level: String,
    pub components: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessState {
    pub process_name: String,
    pub pid: Option<u32>,
    pub status: String,
    pub memory_mb: Option<f64>,
    pub cpu_percent: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResources {
    pub memory_total_mb: f64,
    pub memory_used_mb: f64,
    pub memory_available_mb: f64,
    pub cpu_usage_percent: Option<f64>,
}

/// Load debug configuration from environment variables
pub fn load_debug_config() -> DebugConfig {
    let enabled = env::var("R3MES_DEBUG_MODE")
        .unwrap_or_else(|_| "false".to_string())
        .eq_ignore_ascii_case("true");
    
    let log_level = env::var("R3MES_DEBUG_LOG_LEVEL")
        .unwrap_or_else(|_| "INFO".to_string());
    
    let components_str = env::var("R3MES_DEBUG_COMPONENTS")
        .unwrap_or_else(|_| "".to_string());
    
    let components: Vec<String> = if components_str.is_empty() {
        vec!["*".to_string()]
    } else {
        components_str
            .split(',')
            .map(|s| s.trim().to_lowercase())
            .collect()
    };
    
    DebugConfig {
        enabled,
        log_level,
        components,
    }
}

/// Check if debug is enabled for a component
pub fn is_component_enabled(config: &DebugConfig, component: &str) -> bool {
    if !config.enabled {
        return false;
    }
    
    if config.components.contains(&"*".to_string()) || config.components.is_empty() {
        return true;
    }
    
    config.components.contains(&component.to_lowercase())
}

/// Get process state information
#[tauri::command]
pub fn get_process_state(process_name: String) -> Result<ProcessState, String> {
    let output = Command::new("ps")
        .args(&["aux"])
        .output()
        .map_err(|e| format!("Failed to execute ps: {}", e))?;
    
    let output_str = String::from_utf8_lossy(&output.stdout);
    
    // Parse ps output to find process
    for line in output_str.lines() {
        if line.contains(&process_name) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 11 {
                let pid = parts[1].parse::<u32>().ok();
                let memory_percent = parts[3].parse::<f64>().ok();
                let cpu_percent = parts[2].parse::<f64>().ok();
                
                // Try to get memory in MB (simplified)
                let memory_mb = memory_percent.map(|p| p * 1024.0 / 100.0);
                
                return Ok(ProcessState {
                    process_name: process_name.clone(),
                    pid,
                    status: "running".to_string(),
                    memory_mb,
                    cpu_percent,
                });
            }
        }
    }
    
    Ok(ProcessState {
        process_name,
        pid: None,
        status: "not_found".to_string(),
        memory_mb: None,
        cpu_percent: None,
    })
}

/// Get system resource information
#[tauri::command]
pub fn get_system_resources() -> Result<SystemResources, String> {
    // Try to get memory info
    let memory_info = Command::new("free")
        .arg("-m")
        .output();
    
    let (memory_total_mb, memory_used_mb, memory_available_mb) = if let Ok(output) = memory_info {
        let output_str = String::from_utf8_lossy(&output.stdout);
        // Parse free output (Linux format)
        for line in output_str.lines() {
            if line.starts_with("Mem:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 4 {
                    let total = parts[1].parse::<f64>().unwrap_or(0.0);
                    let used = parts[2].parse::<f64>().unwrap_or(0.0);
                    let available = parts[3].parse::<f64>().unwrap_or(0.0);
                    return Ok(SystemResources {
                        memory_total_mb: total,
                        memory_used_mb: used,
                        memory_available_mb: available,
                        cpu_usage_percent: None, // Would need additional tooling
                    });
                }
            }
        }
        (0.0, 0.0, 0.0)
    } else {
        // Fallback values
        (0.0, 0.0, 0.0)
    };
    
    Ok(SystemResources {
        memory_total_mb,
        memory_used_mb,
        memory_available_mb,
        cpu_usage_percent: None,
    })
}

/// Setup debug logging based on configuration
pub fn setup_debug_logging() {
    let config = load_debug_config();
    
    if config.enabled && is_component_enabled(&config, "launcher") {
        // Set log level based on config
        env::set_var("RUST_LOG", &config.log_level);
        
        // Log level is set via RUST_LOG environment variable
        // Actual logger initialization should be done in main.rs
    }
}
