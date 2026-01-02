/// Debug utilities for R3MES Desktop Launcher
/// 
/// Provides debugging information, performance profiling, and troubleshooting tools.
/// Used for development and production debugging.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Serialize, Deserialize)]
pub struct DebugInfo {
    pub system: SystemInfo,
    pub processes: Vec<ProcessDebugInfo>,
    pub logs: Vec<LogEntry>,
    pub config: ConfigDebugInfo,
    pub network: NetworkDebugInfo,
    pub performance: PerformanceInfo,
    pub timestamp: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemInfo {
    pub os: String,
    pub arch: String,
    pub hostname: String,
    pub uptime: u64,
    pub memory_total: u64,
    pub memory_available: u64,
    pub disk_space: u64,
    pub cpu_count: u32,
    pub load_average: Vec<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessDebugInfo {
    pub name: String,
    pub pid: Option<u32>,
    pub status: String,
    pub cpu_usage: f64,
    pub memory_usage: u64,
    pub start_time: u64,
    pub command_line: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub level: String,
    pub source: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigDebugInfo {
    pub config_files: Vec<ConfigFileInfo>,
    pub environment_vars: HashMap<String, String>,
    pub workspace_path: String,
    pub log_level: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConfigFileInfo {
    pub path: String,
    pub exists: bool,
    pub size: u64,
    pub modified: u64,
    pub readable: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkDebugInfo {
    pub rpc_connectivity: bool,
    pub grpc_connectivity: bool,
    pub ipfs_connectivity: bool,
    pub dashboard_connectivity: bool,
    pub response_times: HashMap<String, u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceInfo {
    pub startup_time: u64,
    pub memory_usage: u64,
    pub cpu_usage: f64,
    pub disk_io: DiskIOInfo,
    pub network_io: NetworkIOInfo,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiskIOInfo {
    pub read_bytes: u64,
    pub write_bytes: u64,
    pub read_ops: u64,
    pub write_ops: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkIOInfo {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
}

/// Collect comprehensive debug information
pub async fn collect_debug_info() -> Result<DebugInfo, String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Ok(DebugInfo {
        system: collect_system_info()?,
        processes: collect_process_info().await?,
        logs: collect_recent_logs().await?,
        config: collect_config_info()?,
        network: collect_network_info().await?,
        performance: collect_performance_info()?,
        timestamp,
    })
}

/// Collect system information
fn collect_system_info() -> Result<SystemInfo, String> {
    let os = std::env::consts::OS.to_string();
    let arch = std::env::consts::ARCH.to_string();
    
    let hostname = Command::new("hostname")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    // Get system uptime (Linux/macOS)
    let uptime = if cfg!(unix) {
        Command::new("uptime")
            .arg("-s")
            .output()
            .ok()
            .and_then(|output| {
                let uptime_str = String::from_utf8_lossy(&output.stdout);
                // Parse uptime and calculate seconds since boot
                // This is a simplified implementation
                Some(3600) // Placeholder: 1 hour
            })
            .unwrap_or(0)
    } else {
        0 // Windows implementation would be different
    };

    // Get memory information
    let (memory_total, memory_available) = get_memory_info();
    
    // Get disk space
    let disk_space = get_disk_space();
    
    // Get CPU count
    let cpu_count = num_cpus::get() as u32;
    
    // Get load average (Unix only)
    let load_average = if cfg!(unix) {
        get_load_average()
    } else {
        vec![0.0, 0.0, 0.0]
    };

    Ok(SystemInfo {
        os,
        arch,
        hostname,
        uptime,
        memory_total,
        memory_available,
        disk_space,
        cpu_count,
        load_average,
    })
}

/// Get memory information
fn get_memory_info() -> (u64, u64) {
    #[cfg(unix)]
    {
        // Read /proc/meminfo on Linux
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            let mut total = 0;
            let mut available = 0;
            
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        total = value.parse::<u64>().unwrap_or(0) * 1024; // Convert KB to bytes
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(value) = line.split_whitespace().nth(1) {
                        available = value.parse::<u64>().unwrap_or(0) * 1024; // Convert KB to bytes
                    }
                }
            }
            
            return (total, available);
        }
    }
    
    // Fallback or Windows implementation
    (0, 0)
}

/// Get disk space information
fn get_disk_space() -> u64 {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
    let workspace = PathBuf::from(&home).join("R3MES");
    
    // Use statvfs on Unix or GetDiskFreeSpaceEx on Windows
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let Ok(metadata) = fs::metadata(&workspace) {
            // This is a simplified implementation
            // In practice, you'd use statvfs system call
            return metadata.size();
        }
    }
    
    0
}

/// Get load average (Unix only)
fn get_load_average() -> Vec<f64> {
    #[cfg(unix)]
    {
        if let Ok(loadavg) = fs::read_to_string("/proc/loadavg") {
            let parts: Vec<&str> = loadavg.split_whitespace().collect();
            if parts.len() >= 3 {
                return vec![
                    parts[0].parse().unwrap_or(0.0),
                    parts[1].parse().unwrap_or(0.0),
                    parts[2].parse().unwrap_or(0.0),
                ];
            }
        }
    }
    
    vec![0.0, 0.0, 0.0]
}

/// Collect process information
async fn collect_process_info() -> Result<Vec<ProcessDebugInfo>, String> {
    let mut processes = Vec::new();
    
    // List of processes we're interested in
    let process_names = vec![
        "remesd", "r3mes-miner", "ipfs", "r3mes-serving", 
        "r3mes-validator", "r3mes-proposer", "python3"
    ];
    
    for name in process_names {
        if let Ok(info) = get_process_info(name).await {
            processes.push(info);
        }
    }
    
    Ok(processes)
}

/// Get information about a specific process
async fn get_process_info(process_name: &str) -> Result<ProcessDebugInfo, String> {
    #[cfg(unix)]
    {
        let output = Command::new("pgrep")
            .arg("-f")
            .arg(process_name)
            .output()
            .map_err(|e| format!("Failed to run pgrep: {}", e))?;
        
        if output.status.success() {
            let pid_str = String::from_utf8_lossy(&output.stdout).trim();
            if let Ok(pid) = pid_str.parse::<u32>() {
                // Get detailed process info
                let (cpu_usage, memory_usage, start_time, command_line) = get_process_details(pid)?;
                
                return Ok(ProcessDebugInfo {
                    name: process_name.to_string(),
                    pid: Some(pid),
                    status: "running".to_string(),
                    cpu_usage,
                    memory_usage,
                    start_time,
                    command_line,
                });
            }
        }
    }
    
    // Process not found or Windows
    Ok(ProcessDebugInfo {
        name: process_name.to_string(),
        pid: None,
        status: "stopped".to_string(),
        cpu_usage: 0.0,
        memory_usage: 0,
        start_time: 0,
        command_line: String::new(),
    })
}

/// Get detailed process information by PID
fn get_process_details(pid: u32) -> Result<(f64, u64, u64, String), String> {
    #[cfg(unix)]
    {
        // Read /proc/PID/stat for CPU and memory info
        let stat_path = format!("/proc/{}/stat", pid);
        let cmdline_path = format!("/proc/{}/cmdline", pid);
        
        let cpu_usage = 0.0; // Simplified - would need to calculate from /proc/stat
        let memory_usage = 0; // Simplified - would read from /proc/PID/status
        let start_time = 0; // Simplified - would parse from /proc/PID/stat
        
        let command_line = fs::read_to_string(&cmdline_path)
            .unwrap_or_else(|_| String::new())
            .replace('\0', " ");
        
        return Ok((cpu_usage, memory_usage, start_time, command_line));
    }
    
    Ok((0.0, 0, 0, String::new()))
}

/// Collect recent log entries
async fn collect_recent_logs() -> Result<Vec<LogEntry>, String> {
    let mut logs = Vec::new();
    
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let log_dir = PathBuf::from(&home).join("R3MES").join("logs");
    
    if log_dir.exists() {
        let log_files = vec!["node.log", "miner.log", "ipfs.log", "serving.log"];
        
        for log_file in log_files {
            let log_path = log_dir.join(log_file);
            if let Ok(content) = fs::read_to_string(&log_path) {
                // Parse last 10 lines
                for line in content.lines().rev().take(10) {
                    if let Some(entry) = parse_log_line(line, log_file) {
                        logs.push(entry);
                    }
                }
            }
        }
    }
    
    // Sort by timestamp (newest first)
    logs.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    logs.truncate(50); // Keep only last 50 entries
    
    Ok(logs)
}

/// Parse a log line into a LogEntry
fn parse_log_line(line: &str, source: &str) -> Option<LogEntry> {
    // Simple log parsing - in practice, you'd have more sophisticated parsing
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let level = if line.contains("ERROR") {
        "ERROR"
    } else if line.contains("WARN") {
        "WARN"
    } else if line.contains("INFO") {
        "INFO"
    } else {
        "DEBUG"
    };
    
    Some(LogEntry {
        timestamp,
        level: level.to_string(),
        source: source.replace(".log", ""),
        message: line.to_string(),
    })
}

/// Collect configuration information
fn collect_config_info() -> Result<ConfigDebugInfo, String> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let workspace_path = PathBuf::from(&home).join("R3MES");
    
    let config_files = vec![
        workspace_path.join(".r3mes").join("config.json"),
        workspace_path.join(".r3mes").join("launcher_config.json"),
        workspace_path.join("miner-engine").join(".env"),
        workspace_path.join("miner-engine").join("config.yaml"),
    ];
    
    let mut config_file_infos = Vec::new();
    
    for config_file in config_files {
        let info = ConfigFileInfo {
            path: config_file.to_string_lossy().to_string(),
            exists: config_file.exists(),
            size: config_file.metadata().map(|m| m.len()).unwrap_or(0),
            modified: config_file.metadata()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0),
            readable: config_file.exists() && fs::read_to_string(&config_file).is_ok(),
        };
        config_file_infos.push(info);
    }
    
    // Collect relevant environment variables
    let mut environment_vars = HashMap::new();
    let env_vars = vec![
        "R3MES_NETWORK", "R3MES_CHAIN_ID", "R3MES_RPC_URL", 
        "BLOCKCHAIN_RPC_URL", "IPFS_API_URL", "CUDA_VISIBLE_DEVICES"
    ];
    
    for var in env_vars {
        if let Ok(value) = std::env::var(var) {
            environment_vars.insert(var.to_string(), value);
        }
    }
    
    Ok(ConfigDebugInfo {
        config_files: config_file_infos,
        environment_vars,
        workspace_path: workspace_path.to_string_lossy().to_string(),
        log_level: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
    })
}

/// Collect network connectivity information
async fn collect_network_info() -> Result<NetworkDebugInfo, String> {
    let config = crate::config::get_config();
    
    let rpc_connectivity = test_connectivity(&config.rpc_url).await;
    let grpc_connectivity = test_connectivity(&format!("http://{}", config.grpc_url)).await;
    let ipfs_connectivity = test_connectivity(&config.ipfs_url).await;
    let dashboard_connectivity = test_connectivity(&config.web_dashboard_url).await;
    
    let mut response_times = HashMap::new();
    response_times.insert("rpc".to_string(), measure_response_time(&config.rpc_url).await);
    response_times.insert("ipfs".to_string(), measure_response_time(&config.ipfs_url).await);
    
    Ok(NetworkDebugInfo {
        rpc_connectivity,
        grpc_connectivity,
        ipfs_connectivity,
        dashboard_connectivity,
        response_times,
    })
}

/// Test connectivity to a URL
async fn test_connectivity(url: &str) -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    
    client.get(url).send().await.is_ok()
}

/// Measure response time to a URL
async fn measure_response_time(url: &str) -> u64 {
    let start = std::time::Instant::now();
    
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .unwrap();
    
    let _ = client.get(url).send().await;
    
    start.elapsed().as_millis() as u64
}

/// Collect performance information
fn collect_performance_info() -> Result<PerformanceInfo, String> {
    // This would collect actual performance metrics
    // For now, returning placeholder values
    
    Ok(PerformanceInfo {
        startup_time: 1500, // milliseconds
        memory_usage: 50 * 1024 * 1024, // 50 MB
        cpu_usage: 5.0, // 5%
        disk_io: DiskIOInfo {
            read_bytes: 1024 * 1024,
            write_bytes: 512 * 1024,
            read_ops: 100,
            write_ops: 50,
        },
        network_io: NetworkIOInfo {
            bytes_sent: 10240,
            bytes_received: 20480,
            packets_sent: 100,
            packets_received: 150,
        },
    })
}

/// Export debug information to file
pub async fn export_debug_info() -> Result<String, String> {
    let debug_info = collect_debug_info().await?;
    
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let debug_dir = PathBuf::from(&home).join("R3MES").join("debug");
    
    fs::create_dir_all(&debug_dir)
        .map_err(|e| format!("Failed to create debug directory: {}", e))?;
    
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let debug_file = debug_dir.join(format!("debug_info_{}.json", timestamp));
    
    let json = serde_json::to_string_pretty(&debug_info)
        .map_err(|e| format!("Failed to serialize debug info: {}", e))?;
    
    fs::write(&debug_file, json)
        .map_err(|e| format!("Failed to write debug file: {}", e))?;
    
    Ok(debug_file.to_string_lossy().to_string())
}

/// Get system troubleshooting recommendations
pub fn get_troubleshooting_recommendations(debug_info: &DebugInfo) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    // Check memory usage
    if debug_info.system.memory_available < 1024 * 1024 * 1024 { // Less than 1GB
        recommendations.push("Low memory available. Consider closing other applications.".to_string());
    }
    
    // Check disk space
    if debug_info.system.disk_space < 10 * 1024 * 1024 * 1024 { // Less than 10GB
        recommendations.push("Low disk space. Consider freeing up space.".to_string());
    }
    
    // Check network connectivity
    if !debug_info.network.rpc_connectivity {
        recommendations.push("RPC connection failed. Check network settings and firewall.".to_string());
    }
    
    if !debug_info.network.ipfs_connectivity {
        recommendations.push("IPFS connection failed. Ensure IPFS daemon is running.".to_string());
    }
    
    // Check process status
    let running_processes = debug_info.processes.iter()
        .filter(|p| p.status == "running")
        .count();
    
    if running_processes == 0 {
        recommendations.push("No R3MES processes are running. Start the required services.".to_string());
    }
    
    // Check configuration files
    let missing_configs = debug_info.config.config_files.iter()
        .filter(|c| !c.exists)
        .count();
    
    if missing_configs > 0 {
        recommendations.push("Some configuration files are missing. Run setup wizard.".to_string());
    }
    
    if recommendations.is_empty() {
        recommendations.push("System appears to be healthy.".to_string());
    }
    
    recommendations
}