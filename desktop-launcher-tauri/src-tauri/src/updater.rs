//! Auto-updater for R3MES Desktop Launcher
//! 
//! Handles checking for updates, downloading, and installing new versions
//! of the desktop launcher application.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    pub available: bool,
    pub current_version: String,
    pub latest_version: String,
    pub release_notes: String,
    pub download_url: String,
    pub file_size: u64,
    pub checksum: String,
    pub release_date: String,
    pub critical: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateProgress {
    pub stage: UpdateStage,
    pub progress_percent: f64,
    pub message: String,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateStage {
    Checking,
    Downloading,
    Verifying,
    Installing,
    Complete,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateResult {
    pub success: bool,
    pub message: String,
    pub requires_restart: bool,
    pub backup_created: bool,
}

pub struct Updater {
    current_version: String,
    update_server_url: String,
    temp_dir: PathBuf,
    backup_dir: PathBuf,
    app_dir: PathBuf,
}

impl Updater {
    /// Create a new updater
    pub fn new() -> Self {
        let current_version = env!("CARGO_PKG_VERSION").to_string();
        let update_server_url = "https://api.github.com/repos/R3MES-Network/desktop-launcher/releases".to_string();
        
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        
        let temp_dir = std::env::temp_dir().join("r3mes_updater");
        let backup_dir = PathBuf::from(&home).join(".r3mes").join("backups");
        
        // Get current application directory
        let app_dir = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        
        Self {
            current_version,
            update_server_url,
            temp_dir,
            backup_dir,
            app_dir,
        }
    }
    
    /// Check for available updates
    pub async fn check_for_updates(&self) -> Result<UpdateInfo, String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .user_agent("R3MES-Desktop-Launcher")
            .build()
            .unwrap();
        
        let response = client
            .get(&format!("{}/latest", self.update_server_url))
            .send()
            .await
            .map_err(|e| format!("Failed to check for updates: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Update check failed with status: {}", response.status()));
        }
        
        let release_data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse update response: {}", e))?;
        
        let latest_version = release_data["tag_name"]
            .as_str()
            .unwrap_or("")
            .trim_start_matches('v')
            .to_string();
        
        let release_notes = release_data["body"]
            .as_str()
            .unwrap_or("No release notes available")
            .to_string();
        
        let release_date = release_data["published_at"]
            .as_str()
            .unwrap_or("")
            .to_string();
        
        // Find the appropriate asset for current platform
        let empty_vec = vec![];
        let assets = release_data["assets"].as_array().unwrap_or(&empty_vec);
        let (download_url, file_size, checksum) = self.find_platform_asset(assets)?;
        
        let available = self.is_newer_version(&latest_version, &self.current_version);
        
        // Check if this is a critical update (security fix)
        let critical = release_notes.to_lowercase().contains("security") ||
                      release_notes.to_lowercase().contains("critical") ||
                      release_notes.to_lowercase().contains("urgent");
        
        Ok(UpdateInfo {
            available,
            current_version: self.current_version.clone(),
            latest_version,
            release_notes,
            download_url,
            file_size,
            checksum,
            release_date,
            critical,
        })
    }
    
    /// Download and install update
    pub async fn download_and_install_update(
        &self,
        update_info: &UpdateInfo,
    ) -> Result<UpdateResult, String> {
        // Create necessary directories
        fs::create_dir_all(&self.temp_dir)
            .map_err(|e| format!("Failed to create temp directory: {}", e))?;
        
        fs::create_dir_all(&self.backup_dir)
            .map_err(|e| format!("Failed to create backup directory: {}", e))?;
        
        // Download update file
        let download_path = self.temp_dir.join("update.zip");
        self.download_update(&update_info.download_url, &download_path, update_info.file_size).await?;
        
        // Verify checksum
        self.verify_checksum(&download_path, &update_info.checksum).await?;
        
        // Create backup of current installation
        let backup_created = self.create_backup().await?;
        
        // Install update
        self.install_update(&download_path).await?;
        
        // Cleanup
        let _ = fs::remove_file(&download_path);
        
        Ok(UpdateResult {
            success: true,
            message: format!("Successfully updated to version {}", update_info.latest_version),
            requires_restart: true,
            backup_created,
        })
    }
    
    /// Download update file with progress tracking
    async fn download_update(
        &self,
        url: &str,
        path: &PathBuf,
        expected_size: u64,
    ) -> Result<(), String> {
        use futures_util::StreamExt;
        use std::io::Write;
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap();
        
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("Failed to start download: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Download failed with status: {}", response.status()));
        }
        
        let mut file = fs::File::create(path)
            .map_err(|e| format!("Failed to create download file: {}", e))?;
        
        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download error: {}", e))?;
            
            file.write_all(&chunk)
                .map_err(|e| format!("Failed to write to file: {}", e))?;
            
            downloaded += chunk.len() as u64;
            
            // Log progress every 1MB
            if downloaded % (1024 * 1024) == 0 {
                let progress = if expected_size > 0 {
                    (downloaded as f64 / expected_size as f64) * 100.0
                } else {
                    0.0
                };
                println!("Download progress: {:.1}% ({} / {} bytes)", 
                        progress, downloaded, expected_size);
            }
        }
        
        file.flush().map_err(|e| format!("Failed to flush file: {}", e))?;
        
        Ok(())
    }
    
    /// Verify file checksum
    async fn verify_checksum(&self, file_path: &PathBuf, expected_checksum: &str) -> Result<(), String> {
        use sha2::{Sha256, Digest};
        
        
        let mut file = fs::File::open(file_path)
            .map_err(|e| format!("Failed to open file for checksum: {}", e))?;
        
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)
            .map_err(|e| format!("Failed to read file for checksum: {}", e))?;
        
        let actual_checksum = format!("{:x}", hasher.finalize());
        
        if actual_checksum != expected_checksum {
            return Err(format!(
                "Checksum verification failed. Expected: {}, Got: {}",
                expected_checksum, actual_checksum
            ));
        }
        
        Ok(())
    }
    
    /// Create backup of current installation
    async fn create_backup(&self) -> Result<bool, String> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
        let backup_path = self.backup_dir.join(format!("backup_{}", timestamp));
        
        // Copy current application directory to backup
        self.copy_directory_sync(&self.app_dir, &backup_path)?;
        
        println!("Backup created at: {:?}", backup_path);
        Ok(true)
    }
    
    /// Install update from downloaded file
    async fn install_update(&self, update_file: &PathBuf) -> Result<(), String> {
        // Extract update file
        let extract_dir = self.temp_dir.join("extracted");
        fs::create_dir_all(&extract_dir)
            .map_err(|e| format!("Failed to create extract directory: {}", e))?;
        
        self.extract_archive(update_file, &extract_dir).await?;
        
        // Stop current application processes (if any)
        self.stop_application_processes().await?;
        
        // Replace application files
        self.replace_application_files(&extract_dir).await?;
        
        Ok(())
    }
    
    /// Extract archive
    async fn extract_archive(&self, archive_path: &PathBuf, extract_to: &PathBuf) -> Result<(), String> {
        use zip::ZipArchive;
        
        
        let file = fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open archive: {}", e))?;
        
        let mut archive = ZipArchive::new(file)
            .map_err(|e| format!("Failed to read archive: {}", e))?;
        
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| format!("Failed to read archive entry: {}", e))?;
            
            let outpath = extract_to.join(file.name());
            
            if file.name().ends_with('/') {
                fs::create_dir_all(&outpath)
                    .map_err(|e| format!("Failed to create directory: {}", e))?;
            } else {
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
    
    /// Stop application processes
    async fn stop_application_processes(&self) -> Result<(), String> {
        // This would stop any running instances of the application
        // For now, just return success
        Ok(())
    }
    
    /// Replace application files
    async fn replace_application_files(&self, source_dir: &PathBuf) -> Result<(), String> {
        // Copy new files over existing ones
        self.copy_directory_sync(source_dir, &self.app_dir)?;
        
        // Set executable permissions on Unix
        #[cfg(unix)]
        {
            let executable_name = "r3mes-desktop-launcher";
            let executable_path = self.app_dir.join(executable_name);
            
            if executable_path.exists() {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(&executable_path)
                    .map_err(|e| format!("Failed to get file permissions: {}", e))?
                    .permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&executable_path, perms)
                    .map_err(|e| format!("Failed to set executable permissions: {}", e))?;
            }
        }
        
        Ok(())
    }
    
    /// Copy directory recursively
    fn copy_directory_sync(&self, source: &PathBuf, destination: &PathBuf) -> Result<(), String> {
        if !source.exists() {
            return Err(format!("Source directory does not exist: {:?}", source));
        }
        
        fs::create_dir_all(destination)
            .map_err(|e| format!("Failed to create destination directory: {}", e))?;
        
        for entry in fs::read_dir(source)
            .map_err(|e| format!("Failed to read source directory: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let source_path = entry.path();
            let dest_path = destination.join(entry.file_name());
            
            if source_path.is_dir() {
                self.copy_directory_sync(&source_path, &dest_path)?;
            } else {
                fs::copy(&source_path, &dest_path)
                    .map_err(|e| format!("Failed to copy file: {}", e))?;
            }
        }
        
        Ok(())
    }
    
    /// Find appropriate asset for current platform
    fn find_platform_asset(&self, assets: &[serde_json::Value]) -> Result<(String, u64, String), String> {
        let platform_suffix = self.get_platform_suffix();
        
        for asset in assets {
            let name = asset["name"].as_str().unwrap_or("");
            
            if name.contains(&platform_suffix) {
                let download_url = asset["browser_download_url"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                
                let file_size = asset["size"].as_u64().unwrap_or(0);
                
                // In a real implementation, checksums would be provided
                // For now, use a placeholder
                let checksum = "placeholder_checksum".to_string();
                
                return Ok((download_url, file_size, checksum));
            }
        }
        
        Err(format!("No asset found for platform: {}", platform_suffix))
    }
    
    /// Get platform-specific suffix for asset names
    fn get_platform_suffix(&self) -> String {
        #[cfg(target_os = "windows")]
        {
            "windows-x64.zip".to_string()
        }
        
        #[cfg(target_os = "macos")]
        {
            if cfg!(target_arch = "aarch64") {
                "macos-arm64.zip".to_string()
            } else {
                "macos-x64.zip".to_string()
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            if cfg!(target_arch = "aarch64") {
                "linux-arm64.zip".to_string()
            } else {
                "linux-x64.zip".to_string()
            }
        }
        
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            "generic.zip".to_string()
        }
    }
    
    /// Check if a version is newer than another
    fn is_newer_version(&self, new_version: &str, current_version: &str) -> bool {
        // Simple version comparison (in practice, use proper semver parsing)
        let new_parts: Vec<u32> = new_version.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        
        let current_parts: Vec<u32> = current_version.split('.')
            .filter_map(|s| s.parse().ok())
            .collect();
        
        for i in 0..std::cmp::max(new_parts.len(), current_parts.len()) {
            let new_part = new_parts.get(i).unwrap_or(&0);
            let current_part = current_parts.get(i).unwrap_or(&0);
            
            if new_part > current_part {
                return true;
            } else if new_part < current_part {
                return false;
            }
        }
        
        false
    }
    
    /// Rollback to previous version
    pub async fn rollback_update(&self) -> Result<UpdateResult, String> {
        // Find the most recent backup
        let backup_entries = fs::read_dir(&self.backup_dir)
            .map_err(|e| format!("Failed to read backup directory: {}", e))?;
        
        let mut backups: Vec<_> = backup_entries
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().ok().map_or(false, |ft| ft.is_dir()))
            .collect();
        
        backups.sort_by(|a, b| {
            b.metadata().ok()
                .and_then(|m| m.modified().ok())
                .cmp(&a.metadata().ok().and_then(|m| m.modified().ok()))
        });
        
        if let Some(latest_backup) = backups.first() {
            let backup_path = latest_backup.path();
            
            // Stop current application
            self.stop_application_processes().await?;
            
            // Restore from backup
            self.copy_directory_sync(&backup_path, &self.app_dir)?;
            
            Ok(UpdateResult {
                success: true,
                message: "Successfully rolled back to previous version".to_string(),
                requires_restart: true,
                backup_created: false,
            })
        } else {
            Err("No backup found for rollback".to_string())
        }
    }
    
    /// Clean up old backups
    pub async fn cleanup_old_backups(&self, keep_count: usize) -> Result<(), String> {
        let backup_entries = fs::read_dir(&self.backup_dir)
            .map_err(|e| format!("Failed to read backup directory: {}", e))?;
        
        let mut backups: Vec<_> = backup_entries
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_type().ok().map_or(false, |ft| ft.is_dir()))
            .collect();
        
        // Sort by modification time (newest first)
        backups.sort_by(|a, b| {
            b.metadata().ok()
                .and_then(|m| m.modified().ok())
                .cmp(&a.metadata().ok().and_then(|m| m.modified().ok()))
        });
        
        // Remove old backups beyond keep_count
        for backup in backups.iter().skip(keep_count) {
            let backup_path = backup.path();
            if let Err(e) = fs::remove_dir_all(&backup_path) {
                eprintln!("Failed to remove old backup {:?}: {}", backup_path, e);
            } else {
                println!("Removed old backup: {:?}", backup_path);
            }
        }
        
        Ok(())
    }
    
    /// Get current version
    pub fn get_current_version(&self) -> &str {
        &self.current_version
    }
    
    /// Set update server URL
    pub fn set_update_server_url(&mut self, url: String) {
        self.update_server_url = url;
    }
    
    /// Check if auto-update is enabled
    pub fn is_auto_update_enabled(&self) -> bool {
        // Check configuration or environment variable
        std::env::var("R3MES_AUTO_UPDATE")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true)
    }
    
    /// Enable or disable auto-update
    pub fn set_auto_update_enabled(&self, enabled: bool) {
        // This would save to configuration file
        // For now, just set environment variable
        std::env::set_var("R3MES_AUTO_UPDATE", enabled.to_string());
    }
}