use serde::{Deserialize, Serialize};
use std::process::Command;
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareCheckResult {
    pub docker: DockerCheck,
    pub gpu: GpuCheck,
    pub disk: DiskCheck,
    pub ram: RamCheck,
    pub cuda: CudaCheck,
    pub network: NetworkCheck,
    pub all_checks_passed: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DockerCheck {
    pub installed: bool,
    pub running: bool,
    pub version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuCheck {
    pub available: bool,
    pub vendor: Option<String>, // "NVIDIA", "AMD", "Intel"
    pub name: Option<String>,
    pub vram_gb: Option<u32>,
    pub driver_version: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiskCheck {
    pub available_gb: u64,
    pub required_gb: u64,
    pub sufficient: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RamCheck {
    pub total_gb: u64,
    pub minimum_gb: u64,
    pub sufficient: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaCheck {
    pub installed: bool,
    pub version: Option<String>,
    pub compatible: bool, // >= 12.1
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkCheck {
    pub blockchain_reachable: bool,
    pub ipfs_reachable: bool,
    pub blockchain_url: Option<String>,
    pub ipfs_url: Option<String>,
}

pub fn check_docker() -> DockerCheck {
    // Check if docker command exists
    let installed = Command::new("docker")
        .arg("--version")
        .output()
        .is_ok();

    if !installed {
        return DockerCheck {
            installed: false,
            running: false,
            version: None,
        };
    }

    // Check if docker daemon is running
    let running = Command::new("docker")
        .arg("ps")
        .output()
        .is_ok();

    // Get docker version
    let version = Command::new("docker")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout).ok()
        });

    DockerCheck {
        installed: true,
        running,
        version,
    }
}

pub fn check_gpu() -> GpuCheck {
    // Try NVIDIA first
    let nvidia_smi_output = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,driver_version")
        .arg("--format=csv,noheader")
        .output();

    if let Ok(output) = nvidia_smi_output {
        let output_str = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = output_str.trim().lines().collect();
        
        if !lines.is_empty() {
            let first_line = lines[0];
            let parts: Vec<&str> = first_line.split(',').map(|s| s.trim()).collect();
            
            let name = parts.get(0).map(|s| s.to_string());
            
            let vram_gb = parts.get(1)
                .and_then(|s| {
                    s.replace("MiB", "").trim().parse::<u32>().ok()
                })
                .map(|mb| mb / 1024);

            let driver_version = parts.get(2).map(|s| s.to_string());

            return GpuCheck {
                available: true,
                vendor: Some("NVIDIA".to_string()),
                name,
                vram_gb,
                driver_version,
            };
        }
    }

    // Try AMD (rocm-smi)
    let amd_output = Command::new("rocm-smi")
        .arg("--showid")
        .output();

    if let Ok(output) = amd_output {
        let output_str = String::from_utf8_lossy(&output.stdout);
        if !output_str.trim().is_empty() {
            // Try to get GPU name
            let name_output = Command::new("rocm-smi")
                .arg("--showproductname")
                .output();
            
            let name = name_output.ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| s.lines().next().map(|l| l.trim().to_string()));

            return GpuCheck {
                available: true,
                vendor: Some("AMD".to_string()),
                name,
                vram_gb: None, // ROCm doesn't provide easy VRAM query
                driver_version: None,
            };
        }
    }

    // Try Intel GPU (intel_gpu_top)
    let intel_output = Command::new("intel_gpu_top")
        .arg("-l")
        .output();

    if let Ok(output) = intel_output {
        let output_str = String::from_utf8_lossy(&output.stdout);
        if !output_str.trim().is_empty() {
            return GpuCheck {
                available: true,
                vendor: Some("Intel".to_string()),
                name: Some("Intel GPU".to_string()),
                vram_gb: None,
                driver_version: None,
            };
        }
    }

    // No GPU found
    GpuCheck {
        available: false,
        vendor: None,
        name: None,
        vram_gb: None,
        driver_version: None,
    }
}

pub fn check_disk() -> DiskCheck {
    let home_dir = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let home_path = Path::new(&home_dir);

    // Check available disk space
    let available_gb = if cfg!(target_os = "linux") {
        // Use df command on Linux
        Command::new("df")
            .arg("-BG")
            .arg(&home_dir)
            .output()
            .ok()
            .and_then(|output| {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let lines: Vec<&str> = output_str.lines().collect();
                if lines.len() > 1 {
                    let parts: Vec<&str> = lines[1].split_whitespace().collect();
                    parts.get(3)
                        .and_then(|s| s.replace("G", "").parse::<u64>().ok())
                } else {
                    None
                }
            })
            .unwrap_or(0)
    } else if cfg!(target_os = "windows") {
        // Windows: Use fsutil or PowerShell
        // Simplified - would need Windows-specific implementation
        0
    } else if cfg!(target_os = "macos") {
        // macOS: Use df command
        Command::new("df")
            .arg("-g")
            .arg(&home_dir)
            .output()
            .ok()
            .and_then(|output| {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let lines: Vec<&str> = output_str.lines().collect();
                if lines.len() > 1 {
                    let parts: Vec<&str> = lines[1].split_whitespace().collect();
                    parts.get(3).and_then(|s| s.parse::<u64>().ok())
                } else {
                    None
                }
            })
            .unwrap_or(0)
    } else {
        0
    };

    let required_gb = 100; // 28GB model + 72GB for chain data and other files
    let sufficient = available_gb >= required_gb;

    DiskCheck {
        available_gb,
        required_gb,
        sufficient,
    }
}

pub fn check_ram() -> RamCheck {
    let minimum_gb = 16;

    // Read /proc/meminfo on Linux
    let total_gb = if cfg!(target_os = "linux") {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if let Some(kb_str) = parts.get(1) {
                            if let Ok(kb) = kb_str.parse::<u64>() {
                                return Some(kb / 1024 / 1024); // Convert KB to GB
                            }
                        }
                    }
                }
                None
            })
            .unwrap_or(0)
    } else if cfg!(target_os = "macos") {
        // macOS: Use sysctl
        Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .ok()
            .and_then(|output| {
                String::from_utf8(output.stdout)
                    .ok()
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .map(|bytes| bytes / 1024 / 1024 / 1024) // Convert bytes to GB
            })
            .unwrap_or(0)
    } else {
        // Windows: Would need Windows-specific implementation
        0
    };

    RamCheck {
        total_gb,
        minimum_gb,
        sufficient: total_gb >= minimum_gb,
    }
}

pub fn check_cuda() -> CudaCheck {
    // Check nvcc version
    let nvcc_output = Command::new("nvcc")
        .arg("--version")
        .output();

    match nvcc_output {
        Ok(output) => {
            let output_str = String::from_utf8_lossy(&output.stdout);
            
            // Parse version (format: "release X.Y")
            let version = output_str
                .lines()
                .find(|line| line.contains("release"))
                .and_then(|line| {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    parts.iter()
                        .position(|&s| s == "release")
                        .and_then(|idx| parts.get(idx + 1))
                        .map(|s| s.replace(",", ""))
                });

            // Check if version >= 12.1
            let compatible = version.as_ref()
                .and_then(|v| {
                    let parts: Vec<&str> = v.split('.').collect();
                    if let (Some(major), Some(minor)) = (parts.get(0), parts.get(1)) {
                        if let (Ok(maj), Ok(min)) = (major.parse::<u32>(), minor.parse::<u32>()) {
                            return Some((maj > 12) || (maj == 12 && min >= 1));
                        }
                    }
                    None
                })
                .unwrap_or(false);

            CudaCheck {
                installed: true,
                version,
                compatible,
            }
        }
        Err(_) => CudaCheck {
            installed: false,
            version: None,
            compatible: false,
        },
    }
}

pub fn check_network() -> NetworkCheck {
    use crate::config::get_config;
    let config = get_config();
    
    // Check blockchain connectivity (use configured RPC URL)
    let blockchain_url = &config.rpc_url;
    let blockchain_reachable = Command::new("curl")
        .arg("-s")
        .arg("--connect-timeout")
        .arg("3")
        .arg(blockchain_url)
        .output()
        .is_ok() || Command::new("wget")
        .arg("--spider")
        .arg("--timeout=3")
        .arg(blockchain_url)
        .output()
        .is_ok();

    // Check IPFS connectivity (use configured IPFS URL)
    let ipfs_url = format!("{}/api/v0/version", config.ipfs_url);
    let ipfs_reachable = Command::new("curl")
        .arg("-s")
        .arg("--connect-timeout")
        .arg("3")
        .arg(ipfs_url)
        .output()
        .is_ok() || Command::new("wget")
        .arg("--spider")
        .arg("--timeout=3")
        .arg(ipfs_url)
        .output()
        .is_ok();

    NetworkCheck {
        blockchain_reachable,
        ipfs_reachable,
        blockchain_url: Some(blockchain_url.to_string()),
        ipfs_url: Some(config.ipfs_url.clone()),
    }
}

pub fn check_hardware() -> HardwareCheckResult {
    let docker = check_docker();
    let gpu = check_gpu();
    let disk = check_disk();
    let ram = check_ram();
    let cuda = check_cuda();
    let network = check_network();

    let all_checks_passed = docker.installed
        && gpu.available
        && disk.sufficient
        && ram.sufficient
        && (cuda.compatible || gpu.vendor.as_deref() == Some("AMD") || gpu.vendor.as_deref() == Some("Intel")) // CUDA only required for NVIDIA
        && network.blockchain_reachable
        && network.ipfs_reachable;

    HardwareCheckResult {
        docker,
        gpu,
        disk,
        ram,
        cuda,
        network,
        all_checks_passed,
    }
}

