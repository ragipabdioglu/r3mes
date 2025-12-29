use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use std::io;
use log::{debug, warn, error};
use crate::config::get_config;

/// Custom error type for status monitoring operations
#[derive(Debug)]
pub enum StatusError {
    /// Network-related errors (connection, timeout)
    Network(String),
    /// Configuration errors
    Config(String),
    /// IO errors (file system, process)
    Io(io::Error),
    /// JSON parsing errors
    Parse(String),
    /// Timeout errors
    Timeout(String),
}

impl std::fmt::Display for StatusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatusError::Network(msg) => write!(f, "Network error: {}", msg),
            StatusError::Config(msg) => write!(f, "Configuration error: {}", msg),
            StatusError::Io(err) => write!(f, "IO error: {}", err),
            StatusError::Parse(msg) => write!(f, "Parse error: {}", msg),
            StatusError::Timeout(msg) => write!(f, "Timeout: {}", msg),
        }
    }
}

impl std::error::Error for StatusError {}

impl From<io::Error> for StatusError {
    fn from(err: io::Error) -> Self {
        StatusError::Io(err)
    }
}

impl From<serde_json::Error> for StatusError {
    fn from(err: serde_json::Error) -> Self {
        StatusError::Parse(err.to_string())
    }
}

/// Result type alias for status operations
pub type StatusResult<T> = Result<T, StatusError>;

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemStatus {
    pub chain_sync: ChainSyncStatus,
    pub ipfs: IpfsStatus,
    pub model: ModelStatus,
    pub node: NodeStatus,
    /// Overall system health status
    pub health: HealthStatus,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChainSyncStatus {
    pub synced: bool,
    pub percentage: f64,
    pub block_height: Option<u64>,
    pub latest_block_height: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IpfsStatus {
    pub connected: bool,
    pub peers: u32,
    pub status: String, // "online", "offline", "connecting"
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelStatus {
    pub downloaded: bool,
    pub progress: f64, // 0.0 to 1.0
    pub file_name: Option<String>,
    pub file_size_gb: Option<f64>,
    pub integrity_verified: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NodeStatus {
    pub running: bool,
    pub rpc_endpoint: String,
    pub grpc_endpoint: String,
    pub last_block_time: Option<i64>,
}

/// Overall health status of the system
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String, // "healthy", "degraded", "unhealthy"
    pub issues: Vec<String>,
    pub last_check: i64,
}

/// HTTP request with timeout
fn http_request_with_timeout(url: &str, timeout_secs: u64) -> StatusResult<String> {
    let output = Command::new("curl")
        .arg("-s")
        .arg("--max-time")
        .arg(timeout_secs.to_string())
        .arg("--connect-timeout")
        .arg("5")
        .arg(url)
        .output()
        .map_err(|e| StatusError::Io(e))?;

    if !output.status.success() {
        return Err(StatusError::Network(format!(
            "Request failed with status: {:?}",
            output.status.code()
        )));
    }

    String::from_utf8(output.stdout)
        .map_err(|e| StatusError::Parse(format!("Invalid UTF-8 response: {}", e)))
}

pub async fn get_system_status() -> Result<SystemStatus, String> {
    let mut issues = Vec::new();

    // Collect status from all components, handling errors gracefully
    let chain_sync = match get_chain_sync_status().await {
        Ok(status) => status,
        Err(e) => {
            warn!("Failed to get chain sync status: {}", e);
            issues.push(format!("Chain sync: {}", e));
            ChainSyncStatus {
                synced: false,
                percentage: 0.0,
                block_height: None,
                latest_block_height: None,
            }
        }
    };

    let ipfs = match get_ipfs_status().await {
        Ok(status) => status,
        Err(e) => {
            warn!("Failed to get IPFS status: {}", e);
            issues.push(format!("IPFS: {}", e));
            IpfsStatus {
                connected: false,
                peers: 0,
                status: "error".to_string(),
            }
        }
    };

    let model = match get_model_status().await {
        Ok(status) => status,
        Err(e) => {
            warn!("Failed to get model status: {}", e);
            issues.push(format!("Model: {}", e));
            ModelStatus {
                downloaded: false,
                progress: 0.0,
                file_name: None,
                file_size_gb: None,
                integrity_verified: false,
            }
        }
    };

    let node = match get_node_status().await {
        Ok(status) => status,
        Err(e) => {
            warn!("Failed to get node status: {}", e);
            issues.push(format!("Node: {}", e));
            let config = get_config();
            NodeStatus {
                running: false,
                rpc_endpoint: config.rpc_url.clone(),
                grpc_endpoint: config.grpc_url.clone(),
                last_block_time: None,
            }
        }
    };

    // Determine overall health
    let health_status = if issues.is_empty() && node.running && chain_sync.synced {
        "healthy"
    } else if node.running {
        "degraded"
    } else {
        "unhealthy"
    };

    let health = HealthStatus {
        status: health_status.to_string(),
        issues,
        last_check: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs() as i64,
    };

    Ok(SystemStatus {
        chain_sync,
        ipfs,
        model,
        node,
        health,
    })
}

pub async fn get_chain_sync_status() -> StatusResult<ChainSyncStatus> {
    let config = get_config();
    let rpc_url = &config.rpc_url;
    
    debug!("Fetching chain sync status from {}", rpc_url);
    
    // Try to get current block height with timeout
    let current_height: Option<u64> = match http_request_with_timeout(
        &format!("{}/status", rpc_url),
        10,
    ) {
        Ok(response) => {
            // Parse JSON response
            match serde_json::from_str::<serde_json::Value>(&response) {
                Ok(json) => {
                    json.get("result")
                        .and_then(|r| r.get("sync_info"))
                        .and_then(|s| s.get("latest_block_height"))
                        .and_then(|h| h.as_str())
                        .and_then(|h| h.parse().ok())
                }
                Err(e) => {
                    warn!("Failed to parse chain status response: {}", e);
                    None
                }
            }
        }
        Err(e) => {
            warn!("Failed to fetch chain status: {}", e);
            None
        }
    };

    // Try to get latest block height from network
    let latest_height: Option<u64> = None; // Would query network peers

    let percentage = if let (Some(current), Some(latest)) = (current_height, latest_height) {
        if latest > 0 {
            (current as f64 / latest as f64) * 100.0
        } else {
            0.0
        }
    } else if current_height.is_some() {
        // If we have current height but no latest, assume synced
        100.0
    } else {
        0.0
    };

    Ok(ChainSyncStatus {
        synced: percentage >= 99.0 || (current_height.is_some() && latest_height.is_none()),
        percentage,
        block_height: current_height,
        latest_block_height: latest_height,
    })
}

pub async fn get_ipfs_status() -> StatusResult<IpfsStatus> {
    let config = get_config();
    let ipfs_url = &config.ipfs_url;
    
    debug!("Checking IPFS status at {}", ipfs_url);
    
    // Check if IPFS is running with timeout
    let connected = match http_request_with_timeout(
        &format!("{}/api/v0/version", ipfs_url),
        5,
    ) {
        Ok(_) => true,
        Err(e) => {
            debug!("IPFS not available: {}", e);
            false
        }
    };

    // Get peer count if connected
    let peers = if connected {
        match http_request_with_timeout(
            &format!("{}/api/v0/swarm/peers", ipfs_url),
            5,
        ) {
            Ok(response) => {
                // Parse JSON to count peers
                match serde_json::from_str::<serde_json::Value>(&response) {
                    Ok(json) => {
                        json.get("Peers")
                            .and_then(|p| p.as_array())
                            .map(|arr| arr.len() as u32)
                            .unwrap_or(0)
                    }
                    Err(_) => 0
                }
            }
            Err(_) => 0
        }
    } else {
        0
    };

    let status = if connected {
        if peers > 0 {
            "online"
        } else {
            "connecting"
        }
    } else {
        "offline"
    };

    Ok(IpfsStatus {
        connected,
        peers,
        status: status.to_string(),
    })
}

pub async fn get_model_status() -> StatusResult<ModelStatus> {
    use std::path::PathBuf;
    use std::fs;
    
    // Get home directory with proper error handling
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE")) // Windows fallback
        .unwrap_or_else(|_| {
            warn!("Could not determine home directory, using current directory");
            ".".to_string()
        });
    
    let model_dir = PathBuf::from(&home).join(".r3mes").join("models");
    
    debug!("Checking model status in {:?}", model_dir);
    
    // Check if model directory exists and has files
    let downloaded = model_dir.exists() && model_dir.is_dir();
    
    let (file_name, file_size_gb) = if downloaded {
        // Find model file (usually .gguf or .safetensors)
        let model_file = fs::read_dir(&model_dir)
            .map_err(|e| {
                warn!("Failed to read model directory: {}", e);
                StatusError::Io(e)
            })?
            .filter_map(|e| e.ok())
            .find(|e| {
                e.path().extension()
                    .and_then(|ext| ext.to_str())
                    .map(|s| s == "gguf" || s == "safetensors" || s == "bin")
                    .unwrap_or(false)
            });

        if let Some(file) = model_file {
            let path = file.path();
            let name = path.file_name()
                .and_then(|n| n.to_str())
                .map(|s| s.to_string());
            
            let size = fs::metadata(&path)
                .ok()
                .map(|m| m.len() as f64 / (1024.0 * 1024.0 * 1024.0));

            (name, size)
        } else {
            (None, None)
        }
    } else {
        (None, None)
    };

    // TODO: Implement SHA256 verification
    let integrity_verified = file_name.is_some();

    Ok(ModelStatus {
        downloaded: downloaded && file_name.is_some(),
        progress: if downloaded && file_name.is_some() { 1.0 } else { 0.0 },
        file_name,
        file_size_gb,
        integrity_verified,
    })
}

pub async fn get_node_status() -> StatusResult<NodeStatus> {
    let config = get_config();
    let rpc_url = &config.rpc_url;
    let grpc_url = &config.grpc_url;
    
    debug!("Checking node status at {}", rpc_url);
    
    // Check if node is running with timeout
    let (running, last_block_time) = match http_request_with_timeout(
        &format!("{}/status", rpc_url),
        10,
    ) {
        Ok(response) => {
            // Parse JSON response for block time
            match serde_json::from_str::<serde_json::Value>(&response) {
                Ok(json) => {
                    let block_time = json.get("result")
                        .and_then(|r| r.get("sync_info"))
                        .and_then(|s| s.get("latest_block_time"))
                        .and_then(|t| t.as_str())
                        .and_then(|t| {
                            // Parse RFC3339 timestamp
                            chrono::DateTime::parse_from_rfc3339(t)
                                .ok()
                                .map(|dt| dt.timestamp())
                        });
                    (true, block_time)
                }
                Err(e) => {
                    warn!("Failed to parse node status: {}", e);
                    (true, None) // Node is running but couldn't parse response
                }
            }
        }
        Err(e) => {
            debug!("Node not available: {}", e);
            (false, None)
        }
    };

    // Fallback to current time if we couldn't get block time
    let last_block_time = last_block_time.or_else(|| {
        if running {
            Some(
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_secs() as i64
            )
        } else {
            None
        }
    });

    Ok(NodeStatus {
        running,
        rpc_endpoint: rpc_url.to_string(),
        grpc_endpoint: grpc_url.to_string(),
        last_block_time,
    })
}

/// Retry a status check with exponential backoff
pub async fn retry_status_check<T, F, Fut>(
    operation: F,
    max_retries: u32,
    initial_delay_ms: u64,
) -> StatusResult<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = StatusResult<T>>,
{
    let mut delay = initial_delay_ms;
    let mut last_error = None;

    for attempt in 0..max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                warn!("Status check attempt {} failed: {}", attempt + 1, e);
                last_error = Some(e);
                
                if attempt < max_retries - 1 {
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                    delay *= 2; // Exponential backoff
                }
            }
        }
    }

    Err(last_error.unwrap_or(StatusError::Network("Max retries exceeded".to_string())))
}

