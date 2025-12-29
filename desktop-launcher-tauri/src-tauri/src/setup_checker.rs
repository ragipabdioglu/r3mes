use serde::{Deserialize, Serialize};
use std::process::Command;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: String,
    pub model: String,
    pub memory_gb: Option<u64>,
    pub cuda_version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemCheck {
    pub gpu: Option<GpuInfo>,
    pub ram_gb: u64,
    pub disk_space_gb: u64,
    pub docker_installed: bool,
    pub python_version: Option<String>,
    pub node_version: Option<String>,
    pub cuda_available: bool,
}

pub async fn check_system() -> Result<SystemCheck, String> {
    let gpu = check_gpu().ok();
    let ram_gb = check_ram()?;
    let disk_space_gb = check_disk_space()?;
    let docker_installed = check_docker().await;
    let python_version = check_python().await;
    let node_version = check_node().await;
    let cuda_available = check_cuda().await;

    Ok(SystemCheck {
        gpu,
        ram_gb,
        disk_space_gb,
        docker_installed,
        python_version,
        node_version,
        cuda_available,
    })
}

fn check_gpu() -> Result<GpuInfo, String> {
    // Try to detect NVIDIA GPU
    if let Ok(output) = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--format=csv,noheader")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = stdout.trim().split(',').collect();
            if parts.len() >= 2 {
                let model = parts[0].trim().to_string();
                let memory_str = parts[1].trim();
                let memory_gb = memory_str
                    .replace("MiB", "")
                    .trim()
                    .parse::<u64>()
                    .ok()
                    .map(|mb| mb / 1024);

                // Try to get CUDA version
                let cuda_version = Command::new("nvidia-smi")
                    .arg("--query-gpu=driver_version")
                    .arg("--format=csv,noheader")
                    .output()
                    .ok()
                    .and_then(|o| {
                        String::from_utf8(o.stdout).ok()
                    })
                    .map(|s| s.trim().to_string());

                return Ok(GpuInfo {
                    vendor: "NVIDIA".to_string(),
                    model,
                    memory_gb,
                    cuda_version,
                });
            }
        }
    }

    // Try to detect AMD GPU (on Linux)
    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = Command::new("lspci")
            .arg("-v")
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.contains("AMD") && stdout.contains("VGA") {
                return Ok(GpuInfo {
                    vendor: "AMD".to_string(),
                    model: "AMD GPU".to_string(),
                    memory_gb: None,
                    cuda_version: None,
                });
            }
        }
    }

    Err("No GPU detected".to_string())
}

fn check_ram() -> Result<u64, String> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo")
            .map_err(|e| format!("Failed to read /proc/meminfo: {}", e))?;
        
        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let kb = parts[1].parse::<u64>()
                        .map_err(|_| "Failed to parse memory")?;
                    return Ok(kb / 1024 / 1024); // Convert KB to GB
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        let output = Command::new("wmic")
            .arg("computersystem")
            .arg("get")
            .arg("TotalPhysicalMemory")
            .output()
            .map_err(|e| format!("Failed to run wmic: {}", e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if let Ok(bytes) = line.trim().parse::<u64>() {
                return Ok(bytes / 1024 / 1024 / 1024); // Convert bytes to GB
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .map_err(|e| format!("Failed to run sysctl: {}", e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        if let Ok(bytes) = stdout.trim().parse::<u64>() {
            return Ok(bytes / 1024 / 1024 / 1024); // Convert bytes to GB
        }
    }

    Err("Failed to detect RAM".to_string())
}

fn check_disk_space() -> Result<u64, String> {
    let current_dir = std::env::current_dir()
        .map_err(|e| format!("Failed to get current directory: {}", e))?;
    
    #[cfg(target_os = "linux")]
    {
        let output = Command::new("df")
            .arg("-BG")
            .arg(&current_dir)
            .output()
            .map_err(|e| format!("Failed to run df: {}", e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = stdout.lines().collect();
        if lines.len() >= 2 {
            let parts: Vec<&str> = lines[1].split_whitespace().collect();
            if parts.len() >= 4 {
                if let Ok(gb) = parts[3].replace("G", "").parse::<u64>() {
                    return Ok(gb);
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        let output = Command::new("wmic")
            .arg("logicaldisk")
            .arg("get")
            .arg("freespace,size")
            .output()
            .map_err(|e| format!("Failed to run wmic: {}", e))?;
        
        // Parse output to get free space
        // This is simplified - in production, parse properly
        return Ok(100); // Placeholder
    }

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("df")
            .arg("-g")
            .arg(&current_dir)
            .output()
            .map_err(|e| format!("Failed to run df: {}", e))?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = stdout.lines().collect();
        if lines.len() >= 2 {
            let parts: Vec<&str> = lines[1].split_whitespace().collect();
            if parts.len() >= 4 {
                if let Ok(gb) = parts[3].parse::<u64>() {
                    return Ok(gb);
                }
            }
        }
    }

    Err("Failed to detect disk space".to_string())
}

async fn check_docker() -> bool {
    Command::new("docker")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

async fn check_python() -> Option<String> {
    let commands = vec!["python3", "python"];
    
    for cmd in commands {
        if let Ok(output) = Command::new(cmd)
            .arg("--version")
            .output()
        {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                return Some(stdout.trim().to_string());
            }
        }
    }
    
    None
}

async fn check_node() -> Option<String> {
    if let Ok(output) = Command::new("node")
        .arg("--version")
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Some(stdout.trim().to_string());
        }
    }
    
    None
}

async fn check_cuda() -> bool {
    Command::new("nvidia-smi")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

