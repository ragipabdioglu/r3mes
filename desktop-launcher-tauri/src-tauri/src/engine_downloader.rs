/// Engine downloader for R3MES Desktop Launcher
/// 
/// Handles downloading, verification, and installation of the R3MES mining engine
/// across different platforms (Windows, Linux, macOS).

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::io::Write;
use reqwest;
use sha2::{Sha256, Digest};
use futures_util::StreamExt;

#[derive(Debug, Serialize, Deserialize)]
pub struct EngineStatus {
    pub installed: bool,
    pub version: Option<String>,
    pub checksum_valid: bool,
    pub executable_path: Option<String>,
    pub size_bytes: u64,
    pub last_updated: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub percentage: f64,
    pub speed_bps: u64,
    pub eta_seconds: u64,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DownloadResult {
    pub success: bool,
    pub message: String,
    pub file_path: Option<String>,
    pub checksum: Option<String>,
}

pub struct EngineDownloader {
    engine_dir: PathBuf,
    download_url: String,
    expected_checksum: String,
    client: reqwest::Client,
}

impl EngineDownloader {
    /// Create a new engine downloader
    pub fn new(engine_dir: PathBuf, download_url: String, expected_checksum: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300)) // 5 minutes
            .build()
            .unwrap();
        
        Self {
            engine_dir,
            download_url,
            expected_checksum,
            client,
        }
    }
    
    /// Get the default engine directory
    pub fn default_engine_dir() -> PathBuf {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        
        PathBuf::from(home).join(".r3mes").join("engine")
    }
    
    /// Get the download URL for the current platform
    pub fn get_download_url() -> String {
        let base_url = "https://github.com/R3MES-Network/releases/download/latest";
        
        #[cfg(target_os = "windows")]
        {
            format!("{}/r3mes-engine-windows-x64.zip", base_url)
        }
        
        #[cfg(target_os = "macos")]
        {
            if cfg!(target_arch = "aarch64") {
                format!("{}/r3mes-engine-macos-arm64.tar.gz", base_url)
            } else {
                format!("{}/r3mes-engine-macos-x64.tar.gz", base_url)
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            if cfg!(target_arch = "aarch64") {
                format!("{}/r3mes-engine-linux-arm64.tar.gz", base_url)
            } else {
                format!("{}/r3mes-engine-linux-x64.tar.gz", base_url)
            }
        }
        
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            format!("{}/r3mes-engine-generic.tar.gz", base_url)
        }
    }
    
    /// Get the expected checksum for the current platform
    pub fn get_expected_checksum() -> String {
        // In a real implementation, these would be fetched from a manifest file
        // or embedded in the application during build
        
        #[cfg(target_os = "windows")]
        {
            "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456".to_string()
        }
        
        #[cfg(target_os = "macos")]
        {
            if cfg!(target_arch = "aarch64") {
                "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567a".to_string()
            } else {
                "c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567ab2".to_string()
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            if cfg!(target_arch = "aarch64") {
                "d4e5f6789012345678901234567890abcdef1234567890abcdef1234567ab2c3".to_string()
            } else {
                "e5f6789012345678901234567890abcdef1234567890abcdef1234567ab2c3d4".to_string()
            }
        }
        
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            "f6789012345678901234567890abcdef1234567890abcdef1234567ab2c3d4e5".to_string()
        }
    }
    
    /// Check the current engine status
    pub async fn check_engine_status(&self) -> EngineStatus {
        let executable_name = if cfg!(windows) { "engine.exe" } else { "engine" };
        let executable_path = self.engine_dir.join(executable_name);
        
        let installed = executable_path.exists();
        let mut version = None;
        let mut checksum_valid = false;
        let mut size_bytes = 0;
        let mut last_updated = None;
        
        if installed {
            // Get file size
            if let Ok(metadata) = fs::metadata(&executable_path) {
                size_bytes = metadata.len();
                last_updated = metadata.modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| d.as_secs());
            }
            
            // Get version
            version = self.get_engine_version(&executable_path).await;
            
            // Verify checksum
            checksum_valid = self.verify_checksum(&executable_path).await.unwrap_or(false);
        }
        
        EngineStatus {
            installed,
            version,
            checksum_valid,
            executable_path: if installed { 
                Some(executable_path.to_string_lossy().to_string()) 
            } else { 
                None 
            },
            size_bytes,
            last_updated,
        }
    }
    
    /// Download and install the engine
    pub async fn download_and_install(&self) -> Result<DownloadResult, String> {
        // Create engine directory
        fs::create_dir_all(&self.engine_dir)
            .map_err(|e| format!("Failed to create engine directory: {}", e))?;
        
        // Determine download file name
        let url_path = self.download_url.split('/').last().unwrap_or("engine.zip");
        let download_path = self.engine_dir.join(format!("download_{}", url_path));
        
        // Download the file
        println!("Downloading engine from: {}", self.download_url);
        let download_result = self.download_file(&download_path).await?;
        
        if !download_result.success {
            return Ok(download_result);
        }
        
        // Verify checksum
        println!("Verifying checksum...");
        let actual_checksum = self.calculate_file_checksum(&download_path).await?;
        
        if actual_checksum != self.expected_checksum {
            fs::remove_file(&download_path).ok();
            return Ok(DownloadResult {
                success: false,
                message: format!(
                    "Checksum verification failed. Expected: {}, Got: {}", 
                    self.expected_checksum, actual_checksum
                ),
                file_path: None,
                checksum: Some(actual_checksum),
            });
        }
        
        // Extract the archive
        println!("Extracting engine...");
        self.extract_archive(&download_path).await?;
        
        // Clean up download file
        fs::remove_file(&download_path).ok();
        
        // Set executable permissions (Unix)
        #[cfg(unix)]
        {
            let executable_name = "engine";
            let executable_path = self.engine_dir.join(executable_name);
            if executable_path.exists() {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&executable_path)
                    .map_err(|e| format!("Failed to get file permissions: {}", e))?
                    .permissions();
                perms.set_mode(0o755); // rwxr-xr-x
                fs::set_permissions(&executable_path, perms)
                    .map_err(|e| format!("Failed to set executable permissions: {}", e))?;
            }
        }
        
        Ok(DownloadResult {
            success: true,
            message: "Engine downloaded and installed successfully".to_string(),
            file_path: Some(self.engine_dir.to_string_lossy().to_string()),
            checksum: Some(actual_checksum),
        })
    }
    
    /// Download a file with progress tracking
    async fn download_file(&self, file_path: &PathBuf) -> Result<DownloadResult, String> {
        let response = self.client
            .get(&self.download_url)
            .send()
            .await
            .map_err(|e| format!("Failed to start download: {}", e))?;
        
        if !response.status().is_success() {
            return Ok(DownloadResult {
                success: false,
                message: format!("Download failed with status: {}", response.status()),
                file_path: None,
                checksum: None,
            });
        }
        
        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        
        let mut file = fs::File::create(file_path)
            .map_err(|e| format!("Failed to create file: {}", e))?;
        
        let start_time = std::time::Instant::now();
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download error: {}", e))?;
            
            file.write_all(&chunk)
                .map_err(|e| format!("Failed to write to file: {}", e))?;
            
            downloaded += chunk.len() as u64;
            
            // Calculate progress
            let elapsed = start_time.elapsed().as_secs_f64();
            let speed = if elapsed > 0.0 { downloaded as f64 / elapsed } else { 0.0 };
            let percentage = if total_size > 0 { 
                (downloaded as f64 / total_size as f64) * 100.0 
            } else { 
                0.0 
            };
            
            // Log progress every 1MB or 10%
            if downloaded % (1024 * 1024) == 0 || (downloaded * 10 / total_size.max(1)) > ((downloaded - chunk.len() as u64) * 10 / total_size.max(1)) {
                println!("Download progress: {:.1}% ({} / {} bytes)", 
                        percentage, downloaded, total_size);
            }
        }
        
        file.flush().map_err(|e| format!("Failed to flush file: {}", e))?;
        
        Ok(DownloadResult {
            success: true,
            message: format!("Downloaded {} bytes", downloaded),
            file_path: Some(file_path.to_string_lossy().to_string()),
            checksum: None,
        })
    }
    
    /// Extract archive based on file extension
    async fn extract_archive(&self, archive_path: &PathBuf) -> Result<(), String> {
        let file_name = archive_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        
        if file_name.ends_with(".zip") {
            self.extract_zip(archive_path).await
        } else if file_name.ends_with(".tar.gz") || file_name.ends_with(".tgz") {
            self.extract_tar_gz(archive_path).await
        } else {
            Err(format!("Unsupported archive format: {}", file_name))
        }
    }
    
    /// Extract ZIP archive
    async fn extract_zip(&self, archive_path: &PathBuf) -> Result<(), String> {
        use zip::ZipArchive;
        use std::io::Read;
        
        let file = fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open ZIP file: {}", e))?;
        
        let mut archive = ZipArchive::new(file)
            .map_err(|e| format!("Failed to read ZIP archive: {}", e))?;
        
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| format!("Failed to read ZIP entry {}: {}", i, e))?;
            
            let outpath = self.engine_dir.join(file.name());
            
            if file.name().ends_with('/') {
                // Directory
                fs::create_dir_all(&outpath)
                    .map_err(|e| format!("Failed to create directory: {}", e))?;
            } else {
                // File
                if let Some(parent) = outpath.parent() {
                    fs::create_dir_all(parent)
                        .map_err(|e| format!("Failed to create parent directory: {}", e))?;
                }
                
                let mut outfile = fs::File::create(&outpath)
                    .map_err(|e| format!("Failed to create output file: {}", e))?;
                
                std::io::copy(&mut file, &mut outfile)
                    .map_err(|e| format!("Failed to extract file: {}", e))?;
            }
        }
        
        Ok(())
    }
    
    /// Extract TAR.GZ archive
    async fn extract_tar_gz(&self, archive_path: &PathBuf) -> Result<(), String> {
        use flate2::read::GzDecoder;
        use tar::Archive;
        
        let file = fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open TAR.GZ file: {}", e))?;
        
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        
        archive.unpack(&self.engine_dir)
            .map_err(|e| format!("Failed to extract TAR.GZ archive: {}", e))?;
        
        Ok(())
    }
    
    /// Calculate SHA256 checksum of a file
    async fn calculate_file_checksum(&self, file_path: &PathBuf) -> Result<String, String> {
        let mut file = fs::File::open(file_path)
            .map_err(|e| format!("Failed to open file for checksum: {}", e))?;
        
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)
            .map_err(|e| format!("Failed to read file for checksum: {}", e))?;
        
        let hash = hasher.finalize();
        Ok(format!("{:x}", hash))
    }
    
    /// Verify file checksum
    async fn verify_checksum(&self, file_path: &PathBuf) -> Result<bool, String> {
        let actual_checksum = self.calculate_file_checksum(file_path).await?;
        Ok(actual_checksum == self.expected_checksum)
    }
    
    /// Get engine version by running the executable
    async fn get_engine_version(&self, executable_path: &PathBuf) -> Option<String> {
        use std::process::Command;
        
        let output = Command::new(executable_path)
            .arg("--version")
            .output()
            .ok()?;
        
        if output.status.success() {
            let version_str = String::from_utf8_lossy(&output.stdout);
            // Parse version from output like "R3MES Engine v1.2.3"
            version_str.lines()
                .next()
                .and_then(|line| {
                    line.split_whitespace()
                        .find(|word| word.starts_with('v'))
                        .map(|v| v.to_string())
                })
        } else {
            None
        }
    }
    
    /// Remove installed engine
    pub async fn remove_engine(&self) -> Result<(), String> {
        if self.engine_dir.exists() {
            fs::remove_dir_all(&self.engine_dir)
                .map_err(|e| format!("Failed to remove engine directory: {}", e))?;
        }
        Ok(())
    }
    
    /// Update engine to latest version
    pub async fn update_engine(&self) -> Result<DownloadResult, String> {
        // Check if update is needed
        let status = self.check_engine_status().await;
        
        if status.installed && status.checksum_valid {
            // Check for newer version online (simplified)
            let latest_version = self.get_latest_version().await?;
            
            if let Some(current_version) = status.version {
                if self.is_version_newer(&latest_version, &current_version) {
                    println!("Updating engine from {} to {}", current_version, latest_version);
                    return self.download_and_install().await;
                } else {
                    return Ok(DownloadResult {
                        success: true,
                        message: "Engine is already up to date".to_string(),
                        file_path: status.executable_path,
                        checksum: None,
                    });
                }
            }
        }
        
        // Install or reinstall
        self.download_and_install().await
    }
    
    /// Get latest version from remote
    async fn get_latest_version(&self) -> Result<String, String> {
        // In a real implementation, this would query a version API
        // For now, return a placeholder version
        Ok("v1.0.0".to_string())
    }
    
    /// Compare version strings (simplified)
    fn is_version_newer(&self, new_version: &str, current_version: &str) -> bool {
        // Remove 'v' prefix if present
        let new_ver = new_version.trim_start_matches('v');
        let current_ver = current_version.trim_start_matches('v');
        
        // Simple string comparison (in practice, use proper semver parsing)
        new_ver > current_ver
    }
    
    /// Get download progress (for UI updates)
    pub async fn get_download_progress(&self) -> Option<DownloadProgress> {
        // This would be implemented with shared state between download and UI
        // For now, return None (no active download)
        None
    }
    
    /// Cancel ongoing download
    pub async fn cancel_download(&self) -> Result<(), String> {
        // This would require cancellation token implementation
        // For now, just return success
        Ok(())
    }
}

/// Utility functions
impl EngineDownloader {
    /// Check if engine is compatible with current system
    pub fn is_engine_compatible() -> bool {
        // Check system requirements
        let os_supported = cfg!(any(
            target_os = "windows",
            target_os = "macos", 
            target_os = "linux"
        ));
        
        let arch_supported = cfg!(any(
            target_arch = "x86_64",
            target_arch = "aarch64"
        ));
        
        os_supported && arch_supported
    }
    
    /// Get system information for engine compatibility
    pub fn get_system_info() -> (String, String) {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        (os, arch)
    }
    
    /// Estimate download time based on file size and connection speed
    pub fn estimate_download_time(file_size_bytes: u64, speed_bps: u64) -> u64 {
        if speed_bps == 0 {
            return 0;
        }
        file_size_bytes / speed_bps
    }
    
    /// Format bytes for human-readable display
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.1} {}", size, UNITS[unit_index])
        }
    }
}