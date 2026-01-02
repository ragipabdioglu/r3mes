/// System and process status monitoring for R3MES Desktop Launcher
/// 
/// Monitors system resources, process health, network connectivity,
/// and provides alerts and recommendations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub cpu: CpuStatus,
    pub memory: MemoryStatus,
    pub disk: DiskStatus,
    pub network: NetworkStatus,
    pub processes: HashMap<String, ProcessStatus>,
    pub services: HashMap<String, ServiceStatus>,
    pub alerts: Vec<Alert>,
    pub timestamp: u64,
    pub uptime: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStatus {
    pub usage_percent: f64,
    pub load_average: Vec<f64>,
    pub core_count: u32,
    pub temperature: Option<f64>,
    pub frequency_mhz: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatus {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub usage_percent: f64,
    pub swap_total: u64,
    pub swap_used: u64,
    pub cached_bytes: u64,
    pub buffers_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskStatus {
    pub total_bytes: u64,
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub usage_percent: f64,
    pub read_bytes_per_sec: u64,
    pub write_bytes_per_sec: u64,
    pub read_ops_per_sec: u64,
    pub write_ops_per_sec: u64,
    pub mount_point: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    pub interfaces: HashMap<String, NetworkInterface>,
    pub total_bytes_sent: u64,
    pub total_bytes_received: u64,
    pub connectivity: ConnectivityStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInterface {
    pub name: String,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub errors: u64,
    pub drops: u64,
    pub speed_mbps: Option<u64>,
    pub is_up: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityStatus {
    pub internet_connected: bool,
    pub rpc_reachable: bool,
    pub ipfs_reachable: bool,
    pub dashboard_reachable: bool,
    pub latency_ms: HashMap<String, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessStatus {
    pub name: String,
    pub pid: Option<u32>,
    pub status: String,
    pub cpu_percent: f64,
    pub memory_bytes: u64,
    pub memory_percent: f64,
    pub start_time: u64,
    pub running_time: u64,
    pub threads: u32,
    pub file_descriptors: Option<u32>,
    pub command_line: String,
    pub health: HealthStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub name: String,
    pub status: String,
    pub port: Option<u16>,
    pub response_time_ms: Option<u64>,
    pub last_check: u64,
    pub error_count: u64,
    pub health: HealthStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub level: AlertLevel,
    pub title: String,
    pub message: String,
    pub source: String,
    pub timestamp: u64,
    pub acknowledged: bool,
    pub auto_resolve: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
}

pub struct StatusMonitor {
    system_status: Arc<RwLock<SystemStatus>>,
    monitoring_interval: Duration,
    alert_thresholds: AlertThresholds,
    process_names: Vec<String>,
    service_endpoints: HashMap<String, String>,
    previous_stats: Arc<RwLock<Option<SystemStatus>>>,
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub cpu_warning: f64,
    pub cpu_critical: f64,
    pub memory_warning: f64,
    pub memory_critical: f64,
    pub disk_warning: f64,
    pub disk_critical: f64,
    pub response_time_warning: u64,
    pub response_time_critical: u64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_warning: 80.0,
            cpu_critical: 95.0,
            memory_warning: 85.0,
            memory_critical: 95.0,
            disk_warning: 85.0,
            disk_critical: 95.0,
            response_time_warning: 5000,  // 5 seconds
            response_time_critical: 10000, // 10 seconds
        }
    }
}

impl StatusMonitor {
    /// Create a new status monitor
    pub fn new() -> Self {
        let process_names = vec![
            "remesd".to_string(),
            "r3mes-miner".to_string(),
            "ipfs".to_string(),
            "r3mes-serving".to_string(),
            "r3mes-validator".to_string(),
            "r3mes-proposer".to_string(),
        ];
        
        let mut service_endpoints = HashMap::new();
        service_endpoints.insert("rpc".to_string(), "http://localhost:26657/status".to_string());
        service_endpoints.insert("api".to_string(), "http://localhost:1317/cosmos/base/tendermint/v1beta1/node_info".to_string());
        service_endpoints.insert("ipfs".to_string(), "http://localhost:5001/api/v0/version".to_string());
        service_endpoints.insert("dashboard".to_string(), "http://localhost:3000/api/health".to_string());
        
        Self {
            system_status: Arc::new(RwLock::new(Self::default_system_status())),
            monitoring_interval: Duration::from_secs(5),
            alert_thresholds: AlertThresholds::default(),
            process_names,
            service_endpoints,
            previous_stats: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Start monitoring
    pub async fn start_monitoring(&self) {
        let system_status = Arc::clone(&self.system_status);
        let previous_stats = Arc::clone(&self.previous_stats);
        let interval_duration = self.monitoring_interval;
        let thresholds = self.alert_thresholds.clone();
        let process_names = self.process_names.clone();
        let service_endpoints = self.service_endpoints.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(interval_duration);
            
            loop {
                interval.tick().await;
                
                let mut status = Self::collect_system_status(&process_names, &service_endpoints).await;
                
                // Generate alerts based on thresholds
                status.alerts = Self::generate_alerts(&status, &thresholds);
                
                // Store previous stats for rate calculations
                {
                    let mut prev = previous_stats.write().await;
                    *prev = Some(status.clone());
                }
                
                // Update current status
                {
                    let mut current = system_status.write().await;
                    *current = status;
                }
            }
        });
    }
    
    /// Get current system status
    pub async fn get_status(&self) -> SystemStatus {
        self.system_status.read().await.clone()
    }
    
    /// Get specific process status
    pub async fn get_process_status(&self, process_name: &str) -> Option<ProcessStatus> {
        let status = self.system_status.read().await;
        status.processes.get(process_name).cloned()
    }
    
    /// Get active alerts
    pub async fn get_alerts(&self) -> Vec<Alert> {
        let status = self.system_status.read().await;
        status.alerts.clone()
    }
    
    /// Acknowledge an alert
    pub async fn acknowledge_alert(&self, alert_id: &str) {
        let mut status = self.system_status.write().await;
        for alert in &mut status.alerts {
            if alert.id == alert_id {
                alert.acknowledged = true;
                break;
            }
        }
    }
    
    /// Collect comprehensive system status
    async fn collect_system_status(
        process_names: &[String],
        service_endpoints: &HashMap<String, String>,
    ) -> SystemStatus {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let cpu = Self::collect_cpu_status().await;
        let memory = Self::collect_memory_status().await;
        let disk = Self::collect_disk_status().await;
        let network = Self::collect_network_status().await;
        let processes = Self::collect_process_status(process_names).await;
        let services = Self::collect_service_status(service_endpoints).await;
        let uptime = Self::get_system_uptime().await;
        
        SystemStatus {
            cpu,
            memory,
            disk,
            network,
            processes,
            services,
            alerts: Vec::new(), // Will be populated by generate_alerts
            timestamp,
            uptime,
        }
    }
    
    /// Collect CPU status
    async fn collect_cpu_status() -> CpuStatus {
        let core_count = num_cpus::get() as u32;
        let usage_percent = Self::get_cpu_usage().await;
        let load_average = Self::get_load_average().await;
        let temperature = Self::get_cpu_temperature().await;
        let frequency_mhz = Self::get_cpu_frequency().await;
        
        CpuStatus {
            usage_percent,
            load_average,
            core_count,
            temperature,
            frequency_mhz,
        }
    }
    
    /// Get CPU usage percentage
    async fn get_cpu_usage() -> f64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(stat) = std::fs::read_to_string("/proc/stat") {
                if let Some(line) = stat.lines().next() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 8 && parts[0] == "cpu" {
                        let user: u64 = parts[1].parse().unwrap_or(0);
                        let nice: u64 = parts[2].parse().unwrap_or(0);
                        let system: u64 = parts[3].parse().unwrap_or(0);
                        let idle: u64 = parts[4].parse().unwrap_or(0);
                        let iowait: u64 = parts[5].parse().unwrap_or(0);
                        let irq: u64 = parts[6].parse().unwrap_or(0);
                        let softirq: u64 = parts[7].parse().unwrap_or(0);
                        
                        let total = user + nice + system + idle + iowait + irq + softirq;
                        let active = total - idle - iowait;
                        
                        if total > 0 {
                            return (active as f64 / total as f64) * 100.0;
                        }
                    }
                }
            }
        }
        
        // Fallback: use system load
        0.0
    }
    
    /// Get system load average
    async fn get_load_average() -> Vec<f64> {
        #[cfg(unix)]
        {
            if let Ok(loadavg) = std::fs::read_to_string("/proc/loadavg") {
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
    
    /// Get CPU temperature
    async fn get_cpu_temperature() -> Option<f64> {
        #[cfg(target_os = "linux")]
        {
            // Try different thermal zones
            for i in 0..10 {
                let temp_path = format!("/sys/class/thermal/thermal_zone{}/temp", i);
                if let Ok(temp_str) = std::fs::read_to_string(&temp_path) {
                    if let Ok(temp_millicelsius) = temp_str.trim().parse::<u64>() {
                        return Some(temp_millicelsius as f64 / 1000.0);
                    }
                }
            }
        }
        
        None
    }
    
    /// Get CPU frequency
    async fn get_cpu_frequency() -> Option<u32> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(freq_str) = std::fs::read_to_string("/proc/cpuinfo") {
                for line in freq_str.lines() {
                    if line.starts_with("cpu MHz") {
                        if let Some(freq_part) = line.split(':').nth(1) {
                            if let Ok(freq) = freq_part.trim().parse::<f64>() {
                                return Some(freq as u32);
                            }
                        }
                    }
                }
            }
        }
        
        None
    }
    
    /// Collect memory status
    async fn collect_memory_status() -> MemoryStatus {
        #[cfg(target_os = "linux")]
        {
            if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
                let mut total_bytes = 0;
                let mut available_bytes = 0;
                let mut cached_bytes = 0;
                let mut buffers_bytes = 0;
                let mut swap_total = 0;
                let mut swap_used = 0;
                
                for line in meminfo.lines() {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        let value = parts[1].parse::<u64>().unwrap_or(0) * 1024; // Convert KB to bytes
                        
                        match parts[0] {
                            "MemTotal:" => total_bytes = value,
                            "MemAvailable:" => available_bytes = value,
                            "Cached:" => cached_bytes = value,
                            "Buffers:" => buffers_bytes = value,
                            "SwapTotal:" => swap_total = value,
                            "SwapFree:" => swap_used = swap_total - value,
                            _ => {}
                        }
                    }
                }
                
                let used_bytes = total_bytes - available_bytes;
                let usage_percent = if total_bytes > 0 {
                    (used_bytes as f64 / total_bytes as f64) * 100.0
                } else {
                    0.0
                };
                
                return MemoryStatus {
                    total_bytes,
                    used_bytes,
                    available_bytes,
                    usage_percent,
                    swap_total,
                    swap_used,
                    cached_bytes,
                    buffers_bytes,
                };
            }
        }
        
        // Fallback for other platforms
        MemoryStatus {
            total_bytes: 0,
            used_bytes: 0,
            available_bytes: 0,
            usage_percent: 0.0,
            swap_total: 0,
            swap_used: 0,
            cached_bytes: 0,
            buffers_bytes: 0,
        }
    }
    
    /// Collect disk status
    async fn collect_disk_status() -> DiskStatus {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/".to_string());
        let workspace_path = std::path::PathBuf::from(&home).join("R3MES");
        
        // Get disk space information
        let (total_bytes, available_bytes) = Self::get_disk_space(&workspace_path).await;
        let used_bytes = total_bytes - available_bytes;
        let usage_percent = if total_bytes > 0 {
            (used_bytes as f64 / total_bytes as f64) * 100.0
        } else {
            0.0
        };
        
        // Get disk I/O statistics (simplified)
        let (read_bytes_per_sec, write_bytes_per_sec, read_ops_per_sec, write_ops_per_sec) = 
            Self::get_disk_io_stats().await;
        
        DiskStatus {
            total_bytes,
            used_bytes,
            available_bytes,
            usage_percent,
            read_bytes_per_sec,
            write_bytes_per_sec,
            read_ops_per_sec,
            write_ops_per_sec,
            mount_point: workspace_path.to_string_lossy().to_string(),
        }
    }
    
    /// Get disk space information
    async fn get_disk_space(path: &std::path::Path) -> (u64, u64) {
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            
            let path_cstr = std::ffi::CString::new(path.to_string_lossy().as_bytes()).unwrap();
            let mut statvfs = std::mem::MaybeUninit::uninit();
            
            unsafe {
                if libc::statvfs(path_cstr.as_ptr(), statvfs.as_mut_ptr()) == 0 {
                    let statvfs = statvfs.assume_init();
                    let total_bytes = statvfs.f_blocks * statvfs.f_frsize;
                    let available_bytes = statvfs.f_bavail * statvfs.f_frsize;
                    return (total_bytes, available_bytes);
                }
            }
        }
        
        (0, 0)
    }
    
    /// Get disk I/O statistics
    async fn get_disk_io_stats() -> (u64, u64, u64, u64) {
        // This would require reading /proc/diskstats and calculating rates
        // For now, return zeros
        (0, 0, 0, 0)
    }
    
    /// Collect network status
    async fn collect_network_status() -> NetworkStatus {
        let interfaces = Self::get_network_interfaces().await;
        let connectivity = Self::check_connectivity().await;
        
        let total_bytes_sent = interfaces.values().map(|i| i.bytes_sent).sum();
        let total_bytes_received = interfaces.values().map(|i| i.bytes_received).sum();
        
        NetworkStatus {
            interfaces,
            total_bytes_sent,
            total_bytes_received,
            connectivity,
        }
    }
    
    /// Get network interfaces
    async fn get_network_interfaces() -> HashMap<String, NetworkInterface> {
        let mut interfaces = HashMap::new();
        
        #[cfg(target_os = "linux")]
        {
            if let Ok(net_dev) = std::fs::read_to_string("/proc/net/dev") {
                for line in net_dev.lines().skip(2) {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 17 {
                        let name = parts[0].trim_end_matches(':').to_string();
                        let bytes_received = parts[1].parse().unwrap_or(0);
                        let packets_received = parts[2].parse().unwrap_or(0);
                        let errors_rx = parts[3].parse().unwrap_or(0);
                        let drops_rx = parts[4].parse().unwrap_or(0);
                        let bytes_sent = parts[9].parse().unwrap_or(0);
                        let packets_sent = parts[10].parse().unwrap_or(0);
                        let errors_tx = parts[11].parse().unwrap_or(0);
                        let drops_tx = parts[12].parse().unwrap_or(0);
                        
                        interfaces.insert(name.clone(), NetworkInterface {
                            name,
                            bytes_sent,
                            bytes_received,
                            packets_sent,
                            packets_received,
                            errors: errors_rx + errors_tx,
                            drops: drops_rx + drops_tx,
                            speed_mbps: None,
                            is_up: true, // Simplified
                        });
                    }
                }
            }
        }
        
        interfaces
    }
    
    /// Check network connectivity
    async fn check_connectivity() -> ConnectivityStatus {
        let internet_connected = Self::test_internet_connectivity().await;
        let rpc_reachable = Self::test_endpoint_connectivity("http://localhost:26657/status").await;
        let ipfs_reachable = Self::test_endpoint_connectivity("http://localhost:5001/api/v0/version").await;
        let dashboard_reachable = Self::test_endpoint_connectivity("http://localhost:3000").await;
        
        let mut latency_ms = HashMap::new();
        latency_ms.insert("rpc".to_string(), Self::measure_latency("http://localhost:26657/status").await);
        latency_ms.insert("ipfs".to_string(), Self::measure_latency("http://localhost:5001/api/v0/version").await);
        
        ConnectivityStatus {
            internet_connected,
            rpc_reachable,
            ipfs_reachable,
            dashboard_reachable,
            latency_ms,
        }
    }
    
    /// Test internet connectivity
    async fn test_internet_connectivity() -> bool {
        let test_hosts = vec!["8.8.8.8:53", "1.1.1.1:53"];
        
        for host in test_hosts {
            if tokio::net::TcpStream::connect(host).await.is_ok() {
                return true;
            }
        }
        
        false
    }
    
    /// Test endpoint connectivity
    async fn test_endpoint_connectivity(url: &str) -> bool {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        
        client.get(url).send().await.is_ok()
    }
    
    /// Measure endpoint latency
    async fn measure_latency(url: &str) -> u64 {
        let start = std::time::Instant::now();
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        
        let _ = client.get(url).send().await;
        
        start.elapsed().as_millis() as u64
    }
    
    /// Collect process status
    async fn collect_process_status(process_names: &[String]) -> HashMap<String, ProcessStatus> {
        let mut processes = HashMap::new();
        
        for process_name in process_names {
            if let Some(status) = Self::get_process_status_by_name(process_name).await {
                processes.insert(process_name.clone(), status);
            }
        }
        
        processes
    }
    
    /// Get process status by name
    async fn get_process_status_by_name(process_name: &str) -> Option<ProcessStatus> {
        #[cfg(unix)]
        {
            let output = Command::new("pgrep")
                .arg("-f")
                .arg(process_name)
                .output()
                .ok()?;
            
            if output.status.success() {
                let pid_str = String::from_utf8_lossy(&output.stdout).trim();
                if let Ok(pid) = pid_str.parse::<u32>() {
                    return Self::get_process_details(pid, process_name).await;
                }
            }
        }
        
        None
    }
    
    /// Get detailed process information
    async fn get_process_details(pid: u32, name: &str) -> Option<ProcessStatus> {
        // This would read from /proc/PID/* files on Linux
        // For now, return a basic status
        Some(ProcessStatus {
            name: name.to_string(),
            pid: Some(pid),
            status: "running".to_string(),
            cpu_percent: 0.0,
            memory_bytes: 0,
            memory_percent: 0.0,
            start_time: 0,
            running_time: 0,
            threads: 1,
            file_descriptors: None,
            command_line: String::new(),
            health: HealthStatus::Healthy,
        })
    }
    
    /// Collect service status
    async fn collect_service_status(endpoints: &HashMap<String, String>) -> HashMap<String, ServiceStatus> {
        let mut services = HashMap::new();
        
        for (name, url) in endpoints {
            let status = Self::check_service_status(name, url).await;
            services.insert(name.clone(), status);
        }
        
        services
    }
    
    /// Check individual service status
    async fn check_service_status(name: &str, url: &str) -> ServiceStatus {
        let start = std::time::Instant::now();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .unwrap();
        
        let (status, response_time_ms, error_count) = match client.get(url).send().await {
            Ok(response) => {
                let response_time = start.elapsed().as_millis() as u64;
                if response.status().is_success() {
                    ("healthy".to_string(), Some(response_time), 0)
                } else {
                    ("unhealthy".to_string(), Some(response_time), 1)
                }
            }
            Err(_) => ("unreachable".to_string(), None, 1),
        };
        
        let health = match status.as_str() {
            "healthy" => HealthStatus::Healthy,
            "unhealthy" => HealthStatus::Warning,
            "unreachable" => HealthStatus::Critical,
            _ => HealthStatus::Unknown,
        };
        
        // Extract port from URL
        let port = url.parse::<url::Url>()
            .ok()
            .and_then(|u| u.port())
            .or_else(|| {
                if url.starts_with("https://") { Some(443) }
                else if url.starts_with("http://") { Some(80) }
                else { None }
            });
        
        ServiceStatus {
            name: name.to_string(),
            status,
            port,
            response_time_ms,
            last_check: timestamp,
            error_count,
            health,
        }
    }
    
    /// Get system uptime
    async fn get_system_uptime() -> u64 {
        #[cfg(target_os = "linux")]
        {
            if let Ok(uptime_str) = std::fs::read_to_string("/proc/uptime") {
                if let Some(uptime_part) = uptime_str.split_whitespace().next() {
                    if let Ok(uptime_seconds) = uptime_part.parse::<f64>() {
                        return uptime_seconds as u64;
                    }
                }
            }
        }
        
        0
    }
    
    /// Generate alerts based on current status and thresholds
    fn generate_alerts(status: &SystemStatus, thresholds: &AlertThresholds) -> Vec<Alert> {
        let mut alerts = Vec::new();
        let timestamp = status.timestamp;
        
        // CPU alerts
        if status.cpu.usage_percent > thresholds.cpu_critical {
            alerts.push(Alert {
                id: "cpu_critical".to_string(),
                level: AlertLevel::Critical,
                title: "Critical CPU Usage".to_string(),
                message: format!("CPU usage is {:.1}%", status.cpu.usage_percent),
                source: "system".to_string(),
                timestamp,
                acknowledged: false,
                auto_resolve: true,
            });
        } else if status.cpu.usage_percent > thresholds.cpu_warning {
            alerts.push(Alert {
                id: "cpu_warning".to_string(),
                level: AlertLevel::Warning,
                title: "High CPU Usage".to_string(),
                message: format!("CPU usage is {:.1}%", status.cpu.usage_percent),
                source: "system".to_string(),
                timestamp,
                acknowledged: false,
                auto_resolve: true,
            });
        }
        
        // Memory alerts
        if status.memory.usage_percent > thresholds.memory_critical {
            alerts.push(Alert {
                id: "memory_critical".to_string(),
                level: AlertLevel::Critical,
                title: "Critical Memory Usage".to_string(),
                message: format!("Memory usage is {:.1}%", status.memory.usage_percent),
                source: "system".to_string(),
                timestamp,
                acknowledged: false,
                auto_resolve: true,
            });
        } else if status.memory.usage_percent > thresholds.memory_warning {
            alerts.push(Alert {
                id: "memory_warning".to_string(),
                level: AlertLevel::Warning,
                title: "High Memory Usage".to_string(),
                message: format!("Memory usage is {:.1}%", status.memory.usage_percent),
                source: "system".to_string(),
                timestamp,
                acknowledged: false,
                auto_resolve: true,
            });
        }
        
        // Disk alerts
        if status.disk.usage_percent > thresholds.disk_critical {
            alerts.push(Alert {
                id: "disk_critical".to_string(),
                level: AlertLevel::Critical,
                title: "Critical Disk Usage".to_string(),
                message: format!("Disk usage is {:.1}%", status.disk.usage_percent),
                source: "system".to_string(),
                timestamp,
                acknowledged: false,
                auto_resolve: true,
            });
        } else if status.disk.usage_percent > thresholds.disk_warning {
            alerts.push(Alert {
                id: "disk_warning".to_string(),
                level: AlertLevel::Warning,
                title: "High Disk Usage".to_string(),
                message: format!("Disk usage is {:.1}%", status.disk.usage_percent),
                source: "system".to_string(),
                timestamp,
                acknowledged: false,
                auto_resolve: true,
            });
        }
        
        // Connectivity alerts
        if !status.network.connectivity.internet_connected {
            alerts.push(Alert {
                id: "internet_disconnected".to_string(),
                level: AlertLevel::Critical,
                title: "Internet Connection Lost".to_string(),
                message: "No internet connectivity detected".to_string(),
                source: "network".to_string(),
                timestamp,
                acknowledged: false,
                auto_resolve: true,
            });
        }
        
        // Service alerts
        for (name, service) in &status.services {
            if matches!(service.health, HealthStatus::Critical) {
                alerts.push(Alert {
                    id: format!("service_{}_critical", name),
                    level: AlertLevel::Critical,
                    title: format!("Service {} Unreachable", name),
                    message: format!("Service {} is not responding", name),
                    source: "service".to_string(),
                    timestamp,
                    acknowledged: false,
                    auto_resolve: true,
                });
            }
        }
        
        alerts
    }
    
    /// Create default system status
    fn default_system_status() -> SystemStatus {
        SystemStatus {
            cpu: CpuStatus {
                usage_percent: 0.0,
                load_average: vec![0.0, 0.0, 0.0],
                core_count: 1,
                temperature: None,
                frequency_mhz: None,
            },
            memory: MemoryStatus {
                total_bytes: 0,
                used_bytes: 0,
                available_bytes: 0,
                usage_percent: 0.0,
                swap_total: 0,
                swap_used: 0,
                cached_bytes: 0,
                buffers_bytes: 0,
            },
            disk: DiskStatus {
                total_bytes: 0,
                used_bytes: 0,
                available_bytes: 0,
                usage_percent: 0.0,
                read_bytes_per_sec: 0,
                write_bytes_per_sec: 0,
                read_ops_per_sec: 0,
                write_ops_per_sec: 0,
                mount_point: "/".to_string(),
            },
            network: NetworkStatus {
                interfaces: HashMap::new(),
                total_bytes_sent: 0,
                total_bytes_received: 0,
                connectivity: ConnectivityStatus {
                    internet_connected: false,
                    rpc_reachable: false,
                    ipfs_reachable: false,
                    dashboard_reachable: false,
                    latency_ms: HashMap::new(),
                },
            },
            processes: HashMap::new(),
            services: HashMap::new(),
            alerts: Vec::new(),
            timestamp: 0,
            uptime: 0,
        }
    }
}