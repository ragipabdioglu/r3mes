use serde::{Deserialize, Serialize};
use std::process::Command;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize)]
pub struct IpfsNodeStatus {
    pub running: bool,
    pub peer_count: u32,
    pub repo_size_mb: u64,
    pub version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PinInfo {
    pub cid: String,
    pub name: String,
    pub size_bytes: u64,
    pub pinned_at: String,
}

pub async fn start_ipfs_node() -> Result<String, String> {
    // Start IPFS daemon
    let output = Command::new("ipfs")
        .arg("daemon")
        .spawn()
        .map_err(|e| format!("Failed to start IPFS: {}", e))?;
    
    Ok(format!("IPFS daemon started (PID: {})", output.id()))
}

pub async fn stop_ipfs_node() -> Result<String, String> {
    // Stop IPFS daemon
    let output = Command::new("pkill")
        .arg("ipfs")
        .output()
        .map_err(|e| format!("Failed to stop IPFS: {}", e))?;
    
    if output.status.success() {
        Ok("IPFS daemon stopped".to_string())
    } else {
        Err("Failed to stop IPFS daemon".to_string())
    }
}

pub async fn get_ipfs_status() -> Result<IpfsNodeStatus, String> {
    // Check if IPFS is running
    let running = Command::new("ipfs")
        .arg("id")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    
    if !running {
        return Ok(IpfsNodeStatus {
            running: false,
            peer_count: 0,
            repo_size_mb: 0,
            version: None,
        });
    }
    
    // Get peer count
    let peer_count = Command::new("ipfs")
        .arg("swarm")
        .arg("peers")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                Some(stdout.lines().count() as u32)
            } else {
                Some(0)
            }
        })
        .unwrap_or(0);
    
    // Get version
    let version = Command::new("ipfs")
        .arg("version")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout).ok()
            } else {
                None
            }
        });
    
    // Get repo size (simplified)
    let repo_size_mb = 0; // Would calculate actual repo size
    
    Ok(IpfsNodeStatus {
        running: true,
        peer_count,
        repo_size_mb,
        version,
    })
}

pub async fn pin_content(cid: String, name: String) -> Result<PinInfo, String> {
    // Pin content to IPFS
    let output = Command::new("ipfs")
        .arg("pin")
        .arg("add")
        .arg(&cid)
        .output()
        .map_err(|e| format!("Failed to pin content: {}", e))?;
    
    if !output.status.success() {
        return Err(format!("Failed to pin content: {}", String::from_utf8_lossy(&output.stderr)));
    }
    
    // Get content size
    let size_output = Command::new("ipfs")
        .arg("object")
        .arg("stat")
        .arg(&cid)
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                // Parse size from output (simplified)
                Some(0)
            } else {
                None
            }
        })
        .unwrap_or(0);
    
    Ok(PinInfo {
        cid,
        name,
        size_bytes: size_output,
        pinned_at: chrono::Utc::now().to_rfc3339(),
    })
}

pub async fn check_content_availability(cid: String) -> Result<bool, String> {
    // Check if content is available in IPFS network
    let output = Command::new("ipfs")
        .arg("dht")
        .arg("findprovs")
        .arg(&cid)
        .output()
        .map_err(|e| format!("Failed to check availability: {}", e))?;
    
    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // If there are providers, content is available
        Ok(!stdout.trim().is_empty())
    } else {
        Ok(false)
    }
}

pub async fn list_pins() -> Result<Vec<PinInfo>, String> {
    // List all pinned content
    let output = Command::new("ipfs")
        .arg("pin")
        .arg("ls")
        .arg("--type=recursive")
        .output()
        .map_err(|e| format!("Failed to list pins: {}", e))?;
    
    if !output.status.success() {
        return Err(format!("Failed to list pins: {}", String::from_utf8_lossy(&output.stderr)));
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut pins = Vec::new();
    
    for line in stdout.lines() {
        if let Some(cid) = line.split_whitespace().next() {
            pins.push(PinInfo {
                cid: cid.to_string(),
                name: "".to_string(), // Would parse name from pin metadata
                size_bytes: 0,
                pinned_at: chrono::Utc::now().to_rfc3339(),
            });
        }
    }
    
    Ok(pins)
}

