//! Hardware requirements checking for R3MES Desktop Launcher
//! 
//! Validates system requirements for mining, serving, and validation operations.
//! Checks GPU, CUDA, Docker, memory, disk space, and other dependencies.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use crate::platform::silent_command;

#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareCheckResult {
    pub docker: DockerInfo,
    pub gpu: GpuInfo,
    pub cuda: CudaInfo,
    pub disk: DiskInfo,
    pub ram: RamInfo,
    pub network: NetworkInfo,
    pub python: PythonInfo,
    pub all_checks_passed: bool,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DockerInfo {
    pub installed: bool,
    pub running: bool,
    pub version: Option<String>,
    pub compose_available: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GpuInfo {
    pub available: bool,
    pub name: Option<String>,
    pub vram_gb: Option<f64>,
    pub driver_version: Option<String>,
    pub compute_capability: Option<String>,
    pub gpu_count: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaInfo {
    pub installed: bool,
    pub version: Option<String>,
    pub compatible: bool,
    pub runtime_version: Option<String>,
    pub nvcc_available: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DiskInfo {
    pub available_gb: f64,
    pub required_gb: f64,
    pub sufficient: bool,
    pub workspace_path: String,
    pub ssd: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RamInfo {
    pub total_gb: f64,
    pub available_gb: f64,
    pub minimum_gb: f64,
    pub recommended_gb: f64,
    pub sufficient: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub internet_connected: bool,
    pub ports_available: Vec<PortInfo>,
    pub bandwidth_mbps: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PortInfo {
    pub port: u16,
    pub available: bool,
    pub service: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PythonInfo {
    pub installed: bool,
    pub version: Option<String>,
    pub pip_available: bool,
    pub venv_support: bool,
}

/// Perform comprehensive hardware check
pub fn check_hardware() -> HardwareCheckResult {
    let docker = check_docker();
    let gpu = check_gpu();
    let cuda = check_cuda();
    let disk = check_disk_space();
    let ram = check_memory();
    let network = check_network();
    let python = check_python();
    
    let mut warnings = Vec::new();
    let mut recommendations = Vec::new();
    
    // Analyze results and generate warnings/recommendations
    let all_checks_passed = analyze_results(
        &docker, &gpu, &cuda, &disk, &ram, &network, &python,
        &mut warnings, &mut recommendations
    );
    
    HardwareCheckResult {
        docker,
        gpu,
        cuda,
        disk,
        ram,
        network,
        python,
        all_checks_passed,
        warnings,
        recommendations,
    }
}

/// Check Docker installation and status
fn check_docker() -> DockerInfo {
    let installed = silent_command("docker")
        .arg("--version")
        .output()
        .is_ok();
    
    let version = if installed {
        silent_command("docker")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                let version_str = String::from_utf8_lossy(&output.stdout);
                // Parse "Docker version 20.10.17, build 100c701"
                version_str.split_whitespace()
                    .nth(2)
                    .map(|v| v.trim_end_matches(',').to_string())
            })
    } else {
        None
    };
    
    let running = if installed {
        silent_command("docker")
            .arg("info")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    } else {
        false
    };
    
    let compose_available = silent_command("docker")
        .arg("compose")
        .arg("version")
        .output()
        .is_ok();
    
    DockerInfo {
        installed,
        running,
        version,
        compose_available,
    }
}

/// Check GPU availability and specifications
fn check_gpu() -> GpuInfo {
    // Try nvidia-smi first
    if let Ok(output) = silent_command("nvidia-smi")
        .arg("--query-gpu=name,memory.total,driver_version,compute_cap")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if output.status.success() {
            let gpu_info = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = gpu_info.trim().split('\n').collect();
            
            if !lines.is_empty() && !lines[0].is_empty() {
                let parts: Vec<&str> = lines[0].split(',').map(|s| s.trim()).collect();
                
                if parts.len() >= 4 {
                    let name = Some(parts[0].to_string());
                    let vram_gb = parts[1].parse::<f64>().ok().map(|mb| mb / 1024.0);
                    let driver_version = Some(parts[2].to_string());
                    let compute_capability = Some(parts[3].to_string());
                    
                    return GpuInfo {
                        available: true,
                        name,
                        vram_gb,
                        driver_version,
                        compute_capability,
                        gpu_count: lines.len() as u32,
                    };
                }
            }
        }
    }
    
    // Try alternative methods for AMD GPUs or integrated graphics
    let available = check_alternative_gpu();
    
    GpuInfo {
        available,
        name: None,
        vram_gb: None,
        driver_version: None,
        compute_capability: None,
        gpu_count: if available { 1 } else { 0 },
    }
}

/// Check for alternative GPU solutions (AMD, Intel)
fn check_alternative_gpu() -> bool {
    // Check for AMD ROCm
    if silent_command("rocm-smi").output().is_ok() {
        return true;
    }
    
    // Check for Intel GPU
    #[cfg(target_os = "linux")]
    {
        if fs::read_to_string("/proc/cpuinfo")
            .unwrap_or_default()
            .contains("Intel")
        {
            // Check for Intel integrated graphics
            if PathBuf::from("/dev/dri").exists() {
                return true;
            }
        }
    }
    
    false
}

/// Check CUDA installation and compatibility
fn check_cuda() -> CudaInfo {
    let nvcc_available = silent_command("nvcc")
        .arg("--version")
        .output()
        .is_ok();
    
    let version = if nvcc_available {
        silent_command("nvcc")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                let version_str = String::from_utf8_lossy(&output.stdout);
                // Parse "Cuda compilation tools, release 12.1, V12.1.105"
                version_str.lines()
                    .find(|line| line.contains("release"))
                    .and_then(|line| {
                        line.split("release ")
                            .nth(1)
                            .and_then(|s| s.split(',').next())
                            .map(|v| v.trim().to_string())
                    })
            })
    } else {
        None
    };
    
    // Check runtime version
    let runtime_version = silent_command("nvidia-smi")
        .output()
        .ok()
        .and_then(|output| {
            let smi_output = String::from_utf8_lossy(&output.stdout);
            // Parse CUDA Version from nvidia-smi output
            smi_output.lines()
                .find(|line| line.contains("CUDA Version:"))
                .and_then(|line| {
                    line.split("CUDA Version: ")
                        .nth(1)
                        .and_then(|s| s.split_whitespace().next())
                        .map(|v| v.to_string())
                })
        });
    
    let installed = nvcc_available || runtime_version.is_some();
    
    // Check compatibility (CUDA >= 12.1 required)
    let compatible = version.as_ref()
        .or(runtime_version.as_ref())
        .map(|v| {
            let version_parts: Vec<&str> = v.split('.').collect();
            if version_parts.len() >= 2 {
                let major: u32 = version_parts[0].parse().unwrap_or(0);
                let minor: u32 = version_parts[1].parse().unwrap_or(0);
                major > 12 || (major == 12 && minor >= 1)
            } else {
                false
            }
        })
        .unwrap_or(false);
    
    CudaInfo {
        installed,
        version,
        compatible,
        runtime_version,
        nvcc_available,
    }
}

/// Check available disk space
fn check_disk_space() -> DiskInfo {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
    let workspace_path = PathBuf::from(&home).join("R3MES");
    
    // Required space: 6GB for models + 4GB for logs/cache/adapters
    let required_gb = 10.0;
    
    let available_gb = get_available_disk_space(&workspace_path);
    let sufficient = available_gb >= required_gb;
    
    // Check if it's an SSD (simplified check)
    let ssd = is_ssd(&workspace_path);
    
    DiskInfo {
        available_gb,
        required_gb,
        sufficient,
        workspace_path: workspace_path.to_string_lossy().to_string(),
        ssd,
    }
}

/// Get available disk space in GB
fn get_available_disk_space(path: &PathBuf) -> f64 {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        
        // Use statvfs to get filesystem statistics
        let path_cstr = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
        let mut statvfs = std::mem::MaybeUninit::uninit();
        
        unsafe {
            if libc::statvfs(path_cstr.as_ptr(), statvfs.as_mut_ptr()) == 0 {
                let statvfs = statvfs.assume_init();
                let available_bytes = statvfs.f_bavail * statvfs.f_frsize;
                return available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            }
        }
    }
    
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use std::ffi::OsStr;
        
        let path_str = path.to_string_lossy().to_string();
        let path_wide: Vec<u16> = OsStr::new(&path_str)
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        
        let mut free_bytes: winapi::um::winnt::ULARGE_INTEGER = unsafe { std::mem::zeroed() };
        let mut total_bytes: winapi::um::winnt::ULARGE_INTEGER = unsafe { std::mem::zeroed() };
        
        unsafe {
            if winapi::um::fileapi::GetDiskFreeSpaceExW(
                path_wide.as_ptr(),
                &mut free_bytes as *mut _,
                &mut total_bytes as *mut _,
                std::ptr::null_mut(),
            ) != 0 {
                return *free_bytes.QuadPart() as f64 / (1024.0 * 1024.0 * 1024.0);
            }
        }
    }
    
    // Fallback
    0.0
}

/// Check if the storage is SSD (simplified)
fn is_ssd(_path: &PathBuf) -> bool {
    #[cfg(target_os = "linux")]
    {
        // Check /sys/block for rotational flag
        if let Ok(mounts) = fs::read_to_string("/proc/mounts") {
            for line in mounts.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 && path.starts_with(parts[1]) {
                    let device = parts[0];
                    if let Some(device_name) = device.split('/').last() {
                        let rotational_path = format!("/sys/block/{}/queue/rotational", device_name);
                        if let Ok(rotational) = fs::read_to_string(&rotational_path) {
                            return rotational.trim() == "0";
                        }
                    }
                }
            }
        }
    }
    
    // Default assumption for modern systems
    true
}

/// Check system memory
fn check_memory() -> RamInfo {
    let (total_bytes, available_bytes) = get_memory_info();
    
    let total_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    let available_gb = available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    
    // Requirements: 8GB minimum, 16GB recommended
    let minimum_gb = 8.0;
    let recommended_gb = 16.0;
    let sufficient = total_gb >= minimum_gb;
    
    RamInfo {
        total_gb,
        available_gb,
        minimum_gb,
        recommended_gb,
        sufficient,
    }
}

/// Get memory information (total and available)
fn get_memory_info() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
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
    
    #[cfg(target_os = "macos")]
    {
        // Use sysctl for macOS
        if let Ok(output) = silent_command("sysctl")
            .arg("hw.memsize")
            .output()
        {
            if let Ok(output_str) = String::from_utf8(output.stdout) {
                if let Some(value_str) = output_str.split_whitespace().nth(1) {
                    if let Ok(total) = value_str.parse::<u64>() {
                        // For available, use vm_stat (simplified)
                        let available = total / 2; // Rough estimate
                        return (total, available);
                    }
                }
            }
        }
    }
    
    #[cfg(windows)]
    {
        use winapi::um::sysinfoapi::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        
        unsafe {
            let mut mem_status: MEMORYSTATUSEX = std::mem::zeroed();
            mem_status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            
            if GlobalMemoryStatusEx(&mut mem_status) != 0 {
                return (mem_status.ullTotalPhys, mem_status.ullAvailPhys);
            }
        }
    }
    
    (0, 0)
}

/// Check network connectivity and port availability
fn check_network() -> NetworkInfo {
    let internet_connected = test_internet_connectivity();
    
    // Check required ports
    let required_ports = vec![
        (26657, "Tendermint RPC"),
        (9090, "gRPC"),
        (1317, "REST API"),
        (5001, "IPFS API"),
        (4001, "IPFS Swarm"),
        (8080, "Miner Stats"),
    ];
    
    let mut ports_available = Vec::new();
    for (port, service) in required_ports {
        let available = is_port_available(port);
        ports_available.push(PortInfo {
            port,
            available,
            service: service.to_string(),
        });
    }
    
    NetworkInfo {
        internet_connected,
        ports_available,
        bandwidth_mbps: None, // Would require speed test
    }
}

/// Test internet connectivity
fn test_internet_connectivity() -> bool {
    // Try to connect to multiple reliable hosts
    let test_hosts = vec![
        "8.8.8.8:53",      // Google DNS
        "1.1.1.1:53",      // Cloudflare DNS
        "208.67.222.222:53", // OpenDNS
    ];
    
    for host in test_hosts {
        if std::net::TcpStream::connect_timeout(
            &host.parse().unwrap(),
            std::time::Duration::from_secs(3)
        ).is_ok() {
            return true;
        }
    }
    
    false
}

/// Check if a port is available
fn is_port_available(port: u16) -> bool {
    std::net::TcpListener::bind(format!("127.0.0.1:{}", port)).is_ok()
}

/// Check Python installation
fn check_python() -> PythonInfo {
    let python_commands = vec!["python3", "python"];
    let mut installed = false;
    let mut version = None;
    
    for cmd in python_commands {
        if let Ok(output) = silent_command(cmd)
            .arg("--version")
            .output()
        {
            if output.status.success() {
                installed = true;
                let version_str = String::from_utf8_lossy(&output.stdout);
                version = version_str.split_whitespace()
                    .nth(1)
                    .map(|v| v.to_string());
                break;
            }
        }
    }
    
    let pip_available = if installed {
        silent_command("pip3")
            .arg("--version")
            .output()
            .is_ok() || silent_command("pip")
            .arg("--version")
            .output()
            .is_ok()
    } else {
        false
    };
    
    let venv_support = if installed {
        silent_command("python3")
            .arg("-m")
            .arg("venv")
            .arg("--help")
            .output()
            .is_ok()
    } else {
        false
    };
    
    PythonInfo {
        installed,
        version,
        pip_available,
        venv_support,
    }
}

/// Analyze all check results and generate warnings/recommendations
fn analyze_results(
    docker: &DockerInfo,
    gpu: &GpuInfo,
    cuda: &CudaInfo,
    disk: &DiskInfo,
    ram: &RamInfo,
    network: &NetworkInfo,
    python: &PythonInfo,
    warnings: &mut Vec<String>,
    recommendations: &mut Vec<String>,
) -> bool {
    let mut critical_issues = 0;
    
    // Docker checks
    if !docker.installed {
        warnings.push("Docker is not installed".to_string());
        recommendations.push("Install Docker Desktop from https://docker.com/products/docker-desktop".to_string());
        critical_issues += 1;
    } else if !docker.running {
        warnings.push("Docker is installed but not running".to_string());
        recommendations.push("Start Docker Desktop or run 'sudo systemctl start docker'".to_string());
    }
    
    // GPU checks
    if !gpu.available {
        warnings.push("No GPU detected".to_string());
        recommendations.push("GPU recommended for faster inference. CPU-only mode available but slower.".to_string());
    } else if let Some(vram) = gpu.vram_gb {
        if vram < 4.0 {
            warnings.push(format!("GPU has only {:.1}GB VRAM, 4GB+ recommended", vram));
            recommendations.push("Consider upgrading GPU for better performance".to_string());
        }
    }
    
    // CUDA checks
    if gpu.available && !cuda.installed {
        warnings.push("GPU detected but CUDA not installed".to_string());
        recommendations.push("Install CUDA Toolkit 12.1+ from https://developer.nvidia.com/cuda-downloads".to_string());
        critical_issues += 1;
    } else if cuda.installed && !cuda.compatible {
        warnings.push("CUDA version is too old (12.1+ required)".to_string());
        recommendations.push("Update CUDA to version 12.1 or newer".to_string());
        critical_issues += 1;
    }
    
    // Disk checks
    if !disk.sufficient {
        warnings.push(format!("Insufficient disk space: {:.1}GB available, {:.1}GB required", 
                             disk.available_gb, disk.required_gb));
        recommendations.push("Free up disk space or use external storage".to_string());
        critical_issues += 1;
    }
    
    if !disk.ssd {
        warnings.push("Using HDD storage, SSD recommended for better performance".to_string());
        recommendations.push("Consider moving workspace to SSD for faster I/O".to_string());
    }
    
    // Memory checks
    if !ram.sufficient {
        warnings.push(format!("Insufficient RAM: {:.1}GB total, {:.1}GB minimum required", 
                             ram.total_gb, ram.minimum_gb));
        recommendations.push("Upgrade RAM to at least 8GB, 16GB recommended".to_string());
        critical_issues += 1;
    } else if ram.total_gb < ram.recommended_gb {
        warnings.push(format!("RAM below recommended: {:.1}GB total, {:.1}GB recommended", 
                             ram.total_gb, ram.recommended_gb));
        recommendations.push("Consider upgrading to 16GB RAM for optimal performance".to_string());
    }
    
    // Network checks
    if !network.internet_connected {
        warnings.push("No internet connection detected".to_string());
        recommendations.push("Check network connection and firewall settings".to_string());
        critical_issues += 1;
    }
    
    let blocked_ports: Vec<&PortInfo> = network.ports_available.iter()
        .filter(|p| !p.available)
        .collect();
    
    if !blocked_ports.is_empty() {
        let port_list: Vec<String> = blocked_ports.iter()
            .map(|p| format!("{} ({})", p.port, p.service))
            .collect();
        warnings.push(format!("Some ports are not available: {}", port_list.join(", ")));
        recommendations.push("Check firewall settings and close conflicting applications".to_string());
    }
    
    // Python checks
    if !python.installed {
        warnings.push("Python is not installed".to_string());
        recommendations.push("Install Python 3.8+ from https://python.org/downloads".to_string());
        critical_issues += 1;
    } else if !python.pip_available {
        warnings.push("pip is not available".to_string());
        recommendations.push("Install pip: python -m ensurepip --upgrade".to_string());
    }
    
    // Overall assessment
    critical_issues == 0
}

/// Get hardware recommendations based on intended use
pub fn get_hardware_recommendations(roles: &[String]) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    if roles.contains(&"miner".to_string()) {
        recommendations.push("For Mining:".to_string());
        recommendations.push("- NVIDIA GPU with 4GB+ VRAM (GTX 1650 or better) - optional".to_string());
        recommendations.push("- CUDA 12.1+ installed (for GPU acceleration)".to_string());
        recommendations.push("- 8GB+ RAM".to_string());
        recommendations.push("- SSD storage with 15GB+ free space".to_string());
        recommendations.push("- Stable internet connection (10+ Mbps)".to_string());
    }
    
    if roles.contains(&"serving".to_string()) {
        recommendations.push("For Serving:".to_string());
        recommendations.push("- GPU recommended but not required".to_string());
        recommendations.push("- 8GB+ RAM".to_string());
        recommendations.push("- Fast SSD storage".to_string());
        recommendations.push("- Low-latency internet connection".to_string());
    }
    
    if roles.contains(&"validator".to_string()) {
        recommendations.push("For Validation:".to_string());
        recommendations.push("- Reliable hardware with 99.9% uptime".to_string());
        recommendations.push("- Redundant internet connections".to_string());
        recommendations.push("- UPS (Uninterruptible Power Supply)".to_string());
        recommendations.push("- Monitoring and alerting setup".to_string());
    }
    
    recommendations
}