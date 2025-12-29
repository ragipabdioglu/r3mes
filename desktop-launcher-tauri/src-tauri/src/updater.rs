/// Silent Auto-Update (Sessiz Güncelleme)
/// 
/// Production olmanın altın kuralı: 1000 kişiye "yeni exe indirin" diyemezsin.
/// Sistemde bir bug bulursan veya model güncellersen (BitNet v2 çıkarsa) otomatik güncelleme yapılmalı.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;
use std::io::Write;
use std::process::Command;
use std::collections::HashMap;
use futures_util::StreamExt;
use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UpdateManifest {
    pub version: String,
    pub chain_binary: UpdateInfo,
    pub model_weights: UpdateInfo,
    pub miner_engine: UpdateInfo,
    pub launcher: UpdateInfo,
    #[serde(default)]
    pub release_notes: String,
    #[serde(default)]
    pub min_launcher_version: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UpdateInfo {
    pub version: String,
    pub download_url: String,
    pub checksum: String,
    pub required: bool,  // If false, update is optional
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default)]
    pub release_date: String,
}

/// Update progress event for frontend
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UpdateProgress {
    pub component: String,
    pub status: UpdateStatus,
    pub progress_percent: f32,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum UpdateStatus {
    Checking,
    Downloading,
    Verifying,
    Installing,
    Completed,
    Failed,
    RolledBack,
}

/// Backup entry for rollback
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct BackupEntry {
    pub component: String,
    pub original_path: PathBuf,
    pub backup_path: PathBuf,
    pub version: String,
    pub created_at: DateTime<Utc>,
}

/// Rollback manager for safe updates
#[derive(Debug)]
pub struct RollbackManager {
    backup_dir: PathBuf,
    backups: HashMap<String, BackupEntry>,
    manifest_path: PathBuf,
}

impl RollbackManager {
    pub fn new() -> Result<Self, String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        let backup_dir = PathBuf::from(&home).join(".r3mes").join("backups");
        let manifest_path = backup_dir.join("rollback_manifest.json");
        
        fs::create_dir_all(&backup_dir)
            .map_err(|e| format!("Failed to create backup directory: {}", e))?;
        
        // Load existing backups
        let backups = if manifest_path.exists() {
            let content = fs::read_to_string(&manifest_path).unwrap_or_default();
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            HashMap::new()
        };
        
        Ok(Self {
            backup_dir,
            backups,
            manifest_path,
        })
    }
    
    /// Create backup before update
    pub fn backup_component(&mut self, component: &str, current_path: &PathBuf, version: &str) -> Result<PathBuf, String> {
        if !current_path.exists() {
            return Err(format!("Component path does not exist: {}", current_path.display()));
        }
        
        let timestamp = Utc::now().timestamp();
        let backup_name = format!("{}_{}_backup_{}", component, version.replace('.', "_"), timestamp);
        let backup_path = self.backup_dir.join(&backup_name);
        
        // Copy file or directory
        if current_path.is_dir() {
            self.copy_dir_recursive(current_path, &backup_path)?;
        } else {
            fs::copy(current_path, &backup_path)
                .map_err(|e| format!("Failed to backup {}: {}", component, e))?;
        }
        
        let entry = BackupEntry {
            component: component.to_string(),
            original_path: current_path.clone(),
            backup_path: backup_path.clone(),
            version: version.to_string(),
            created_at: Utc::now(),
        };
        
        self.backups.insert(component.to_string(), entry);
        self.save_manifest()?;
        
        println!("📦 Backed up {} (version {}) to {}", component, version, backup_path.display());
        Ok(backup_path)
    }
    
    /// Rollback component to previous version
    pub fn rollback(&mut self, component: &str) -> Result<(), String> {
        let entry = self.backups.get(component)
            .ok_or_else(|| format!("No backup found for component: {}", component))?
            .clone();
        
        if !entry.backup_path.exists() {
            return Err(format!("Backup file not found: {}", entry.backup_path.display()));
        }
        
        // Restore from backup
        if entry.backup_path.is_dir() {
            // Remove current directory first
            if entry.original_path.exists() {
                fs::remove_dir_all(&entry.original_path)
                    .map_err(|e| format!("Failed to remove current {}: {}", component, e))?;
            }
            self.copy_dir_recursive(&entry.backup_path, &entry.original_path)?;
        } else {
            fs::copy(&entry.backup_path, &entry.original_path)
                .map_err(|e| format!("Failed to restore {}: {}", component, e))?;
        }
        
        println!("⏪ Rolled back {} to version {}", component, entry.version);
        Ok(())
    }
    
    /// Cleanup old backups (keep only last N backups per component)
    pub fn cleanup_old_backups(&mut self, keep_count: usize) -> Result<(), String> {
        let mut component_backups: HashMap<String, Vec<(PathBuf, DateTime<Utc>)>> = HashMap::new();
        
        // Scan backup directory
        if let Ok(entries) = fs::read_dir(&self.backup_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    // Parse component name from backup filename
                    if let Some(component) = name.split('_').next() {
                        let metadata = fs::metadata(&path).ok();
                        let created = metadata
                            .and_then(|m| m.created().ok())
                            .map(|t| DateTime::<Utc>::from(t))
                            .unwrap_or_else(Utc::now);
                        
                        component_backups
                            .entry(component.to_string())
                            .or_default()
                            .push((path, created));
                    }
                }
            }
        }
        
        // Remove old backups
        for (component, mut backups) in component_backups {
            backups.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by date descending
            
            for (path, _) in backups.into_iter().skip(keep_count) {
                if path.is_dir() {
                    let _ = fs::remove_dir_all(&path);
                } else {
                    let _ = fs::remove_file(&path);
                }
                println!("🗑️  Cleaned up old backup: {}", path.display());
            }
        }
        
        Ok(())
    }
    
    fn copy_dir_recursive(&self, src: &PathBuf, dst: &PathBuf) -> Result<(), String> {
        fs::create_dir_all(dst)
            .map_err(|e| format!("Failed to create directory {}: {}", dst.display(), e))?;
        
        for entry in fs::read_dir(src).map_err(|e| format!("Failed to read directory: {}", e))? {
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            
            if src_path.is_dir() {
                self.copy_dir_recursive(&src_path, &dst_path)?;
            } else {
                fs::copy(&src_path, &dst_path)
                    .map_err(|e| format!("Failed to copy file: {}", e))?;
            }
        }
        
        Ok(())
    }
    
    fn save_manifest(&self) -> Result<(), String> {
        let content = serde_json::to_string_pretty(&self.backups)
            .map_err(|e| format!("Failed to serialize manifest: {}", e))?;
        fs::write(&self.manifest_path, content)
            .map_err(|e| format!("Failed to write manifest: {}", e))?;
        Ok(())
    }
}

/// Update result for tracking
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UpdateResult {
    pub component: String,
    pub success: bool,
    pub old_version: String,
    pub new_version: String,
    pub error_message: Option<String>,
    pub rolled_back: bool,
}

pub struct SilentUpdater {
    manifest_url: String,
    update_dir: PathBuf,
    rollback_manager: RollbackManager,
    progress_callback: Option<Box<dyn Fn(UpdateProgress) + Send + Sync>>,
}

impl SilentUpdater {
    pub fn new() -> Result<Self, String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        let update_dir = PathBuf::from(&home).join(".r3mes").join("updates");
        
        fs::create_dir_all(&update_dir)
            .map_err(|e| format!("Failed to create update directory: {}", e))?;
        
        Ok(Self {
            manifest_url: std::env::var("R3MES_UPDATE_MANIFEST_URL")
                .unwrap_or_else(|_| "https://releases.r3mes.network/manifest.json".to_string()),
            update_dir,
            rollback_manager: RollbackManager::new()?,
            progress_callback: None,
        })
    }
    
    /// Set progress callback for UI updates
    pub fn set_progress_callback<F>(&mut self, callback: F)
    where
        F: Fn(UpdateProgress) + Send + Sync + 'static,
    {
        self.progress_callback = Some(Box::new(callback));
    }
    
    /// Emit progress event
    fn emit_progress(&self, component: &str, status: UpdateStatus, progress: f32, downloaded: u64, total: u64, message: &str) {
        if let Some(ref callback) = self.progress_callback {
            callback(UpdateProgress {
                component: component.to_string(),
                status,
                progress_percent: progress,
                downloaded_bytes: downloaded,
                total_bytes: total,
                message: message.to_string(),
            });
        }
    }
    
    /// Check for updates without installing
    pub async fn check_for_updates(&self) -> Result<Vec<(String, String, String)>, String> {
        let manifest = self.fetch_manifest().await?;
        let mut available_updates = Vec::new();
        
        // Check chain binary
        let current_chain = self.get_chain_binary_version()?;
        if current_chain != manifest.chain_binary.version {
            available_updates.push(("chain_binary".to_string(), current_chain, manifest.chain_binary.version.clone()));
        }
        
        // Check model weights
        let current_model = self.get_model_version()?;
        if current_model != manifest.model_weights.version {
            available_updates.push(("model_weights".to_string(), current_model, manifest.model_weights.version.clone()));
        }
        
        // Check miner engine
        let current_miner = self.get_miner_engine_version()?;
        if current_miner != manifest.miner_engine.version {
            available_updates.push(("miner_engine".to_string(), current_miner, manifest.miner_engine.version.clone()));
        }
        
        // Check launcher
        let current_launcher = env!("CARGO_PKG_VERSION").to_string();
        if current_launcher != manifest.launcher.version {
            available_updates.push(("launcher".to_string(), current_launcher, manifest.launcher.version.clone()));
        }
        
        Ok(available_updates)
    }
    
    /// Main update function with rollback support
    pub async fn check_and_update(&mut self) -> Result<Vec<UpdateResult>, String> {
        // Create update directory if it doesn't exist
        fs::create_dir_all(&self.update_dir)
            .map_err(|e| format!("Failed to create update directory: {}", e))?;
        
        let mut results = Vec::new();
        
        // 1. Fetch manifest
        self.emit_progress("manifest", UpdateStatus::Checking, 0.0, 0, 0, "Fetching update manifest...");
        let manifest = self.fetch_manifest().await?;
        
        // 2. Check and update each component with rollback support
        results.push(self.update_component_with_rollback("chain_binary", &manifest.chain_binary).await);
        results.push(self.update_component_with_rollback("model_weights", &manifest.model_weights).await);
        results.push(self.update_component_with_rollback("miner_engine", &manifest.miner_engine).await);
        results.push(self.update_component_with_rollback("launcher", &manifest.launcher).await);
        
        // 3. Cleanup old backups (keep last 3)
        let _ = self.rollback_manager.cleanup_old_backups(3);
        
        // Check if any required update failed
        for result in &results {
            if !result.success && !result.rolled_back {
                // Check if this was a required update
                let is_required = match result.component.as_str() {
                    "chain_binary" => manifest.chain_binary.required,
                    "model_weights" => manifest.model_weights.required,
                    "miner_engine" => manifest.miner_engine.required,
                    "launcher" => manifest.launcher.required,
                    _ => false,
                };
                
                if is_required {
                    return Err(format!("Required update failed for {}: {:?}", result.component, result.error_message));
                }
            }
        }
        
        Ok(results)
    }
    
    /// Update a single component with automatic rollback on failure
    async fn update_component_with_rollback(&mut self, component: &str, update_info: &UpdateInfo) -> UpdateResult {
        let current_version = match component {
            "chain_binary" => self.get_chain_binary_version().unwrap_or_default(),
            "model_weights" => self.get_model_version().unwrap_or_default(),
            "miner_engine" => self.get_miner_engine_version().unwrap_or_default(),
            "launcher" => env!("CARGO_PKG_VERSION").to_string(),
            _ => "unknown".to_string(),
        };
        
        // Skip if already up to date
        if current_version == update_info.version {
            return UpdateResult {
                component: component.to_string(),
                success: true,
                old_version: current_version.clone(),
                new_version: update_info.version.clone(),
                error_message: None,
                rolled_back: false,
            };
        }
        
        // Get component path for backup
        let component_path = match component {
            "chain_binary" => self.get_chain_binary_path(),
            "model_weights" => self.get_model_directory(),
            "miner_engine" => self.get_miner_engine_path(),
            "launcher" => Ok(std::env::current_exe().unwrap_or_default()),
            _ => Err("Unknown component".to_string()),
        };
        
        let component_path = match component_path {
            Ok(p) => p,
            Err(e) => {
                return UpdateResult {
                    component: component.to_string(),
                    success: false,
                    old_version: current_version,
                    new_version: update_info.version.clone(),
                    error_message: Some(e),
                    rolled_back: false,
                };
            }
        };
        
        // Create backup before update (if path exists)
        if component_path.exists() {
            if let Err(e) = self.rollback_manager.backup_component(component, &component_path, &current_version) {
                eprintln!("⚠️  Failed to backup {}: {}", component, e);
                // Continue anyway, but log the warning
            }
        }
        
        // Perform update
        self.emit_progress(component, UpdateStatus::Downloading, 0.0, 0, update_info.size_bytes, &format!("Updating {}...", component));
        
        let update_result = match component {
            "chain_binary" => self.update_chain_binary(update_info).await,
            "model_weights" => self.update_model_weights(update_info).await,
            "miner_engine" => self.update_miner_engine(update_info).await,
            "launcher" => self.update_launcher(update_info).await,
            _ => Err("Unknown component".to_string()),
        };
        
        match update_result {
            Ok(()) => {
                self.emit_progress(component, UpdateStatus::Completed, 100.0, update_info.size_bytes, update_info.size_bytes, &format!("{} updated successfully", component));
                UpdateResult {
                    component: component.to_string(),
                    success: true,
                    old_version: current_version,
                    new_version: update_info.version.clone(),
                    error_message: None,
                    rolled_back: false,
                }
            }
            Err(e) => {
                self.emit_progress(component, UpdateStatus::Failed, 0.0, 0, 0, &format!("Update failed: {}", e));
                
                // Attempt rollback
                let rolled_back = if let Err(rollback_err) = self.rollback_manager.rollback(component) {
                    eprintln!("⚠️  Rollback failed for {}: {}", component, rollback_err);
                    false
                } else {
                    self.emit_progress(component, UpdateStatus::RolledBack, 0.0, 0, 0, &format!("{} rolled back to previous version", component));
                    true
                };
                
                UpdateResult {
                    component: component.to_string(),
                    success: false,
                    old_version: current_version,
                    new_version: update_info.version.clone(),
                    error_message: Some(e),
                    rolled_back,
                }
            }
        }
    }
    
    /// Manual rollback for a specific component
    pub fn rollback_component(&mut self, component: &str) -> Result<(), String> {
        self.rollback_manager.rollback(component)
    }
    
    async fn update_miner_engine(&self, update_info: &UpdateInfo) -> Result<(), String> {
        let current_version = self.get_miner_engine_version()?;
        
        if current_version == update_info.version {
            return Ok(());  // Already up to date
        }
        
        println!("🔄 Updating miner engine: {} → {}", current_version, update_info.version);
        
        // Download new miner engine
        let download_path = self.download_file(&update_info.download_url).await?;
        
        // Verify checksum
        self.verify_checksum(&download_path, &update_info.checksum)?;
        
        // Replace old miner engine
        let miner_path = self.get_miner_engine_path()?;
        fs::copy(&download_path, &miner_path)
            .map_err(|e| format!("Failed to replace miner engine: {}", e))?;
        
        println!("✅ Miner engine updated to {}", update_info.version);
        Ok(())
    }
    
    async fn update_model_weights(&self, update_info: &UpdateInfo) -> Result<(), String> {
        let current_version = self.get_model_version()?;
        
        if current_version == update_info.version {
            return Ok(());  // Already up to date
        }
        
        println!("🔄 Updating model weights: {} → {}", current_version, update_info.version);
        
        // Download new model weights
        let download_path = self.download_file(&update_info.download_url).await?;
        
        // Verify checksum
        self.verify_checksum(&download_path, &update_info.checksum)?;
        
        // Extract model weights to model directory
        let model_dir = self.get_model_directory()?;
        self.extract_model_weights(&download_path, &model_dir)?;
        
        println!("✅ Model weights updated to {}", update_info.version);
        Ok(())
    }
    
    async fn update_chain_binary(&self, update_info: &UpdateInfo) -> Result<(), String> {
        let current_version = self.get_chain_binary_version()?;
        
        if current_version == update_info.version {
            return Ok(());  // Already up to date
        }
        
        println!("🔄 Updating chain binary: {} → {}", current_version, update_info.version);
        
        // Download new chain binary
        let download_path = self.download_file(&update_info.download_url).await?;
        
        // Verify checksum
        self.verify_checksum(&download_path, &update_info.checksum)?;
        
        // Replace old chain binary
        let chain_path = self.get_chain_binary_path()?;
        fs::copy(&download_path, &chain_path)
            .map_err(|e| format!("Failed to replace chain binary: {}", e))?;
        
        // Make executable (Unix only)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&chain_path)
                .map_err(|e| format!("Failed to get metadata: {}", e))?
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&chain_path, perms)
                .map_err(|e| format!("Failed to set permissions: {}", e))?;
        }
        
        println!("✅ Chain binary updated to {}", update_info.version);
        Ok(())
    }
    
    async fn update_launcher(&self, update_info: &UpdateInfo) -> Result<(), String> {
        // Launcher updates require restart, so we just log it
        println!("ℹ️  Launcher update available: {}", update_info.version);
        println!("   Please download from: {}", update_info.download_url);
        Ok(())
    }
    
    async fn fetch_manifest(&self) -> Result<UpdateManifest, String> {
        // Fetch manifest from URL using reqwest
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        
        let response = client
            .get(&self.manifest_url)
            .send()
            .await
            .map_err(|e| format!("Failed to fetch manifest from {}: {}", self.manifest_url, e))?;
        
        if !response.status().is_success() {
            return Err(format!(
                "Failed to fetch manifest: HTTP {}",
                response.status()
            ));
        }
        
        let manifest_json = response
            .text()
            .await
            .map_err(|e| format!("Failed to read manifest response: {}", e))?;
        
        let manifest: UpdateManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| format!("Failed to parse manifest JSON: {}", e))?;
        
        Ok(manifest)
    }
    
    async fn download_file(&self, url: &str) -> Result<PathBuf, String> {
        // Download file using reqwest with progress tracking
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))  // 1 hour timeout for large files
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("Failed to download from {}: {}", url, e))?;
        
        if !response.status().is_success() {
            return Err(format!(
                "Failed to download file: HTTP {}",
                response.status()
            ));
        }
        
        // Get filename from URL or Content-Disposition header
        let filename = url.split('/').last().unwrap_or("download");
        let download_path = self.update_dir.join(filename);
        
        // Check if file already exists (resume capability)
        let mut file = if download_path.exists() {
            // Resume download by opening in append mode
            std::fs::OpenOptions::new()
                .append(true)
                .open(&download_path)
                .map_err(|e| format!("Failed to open file for resume: {}", e))?
        } else {
            // Create new file
            fs::File::create(&download_path)
                .map_err(|e| format!("Failed to create download file: {}", e))?
        };
        
        // Stream download with progress tracking
        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download error: {}", e))?;
            file.write_all(&chunk)
                .map_err(|e| format!("Failed to write chunk: {}", e))?;
            downloaded += chunk.len() as u64;
            
            // Log progress every 10MB
            if downloaded % (10 * 1024 * 1024) == 0 {
                println!("   Downloaded: {} MB", downloaded / (1024 * 1024));
            }
        }
        
        println!("✅ Download complete: {} bytes", downloaded);
        Ok(download_path)
    }
    
    fn verify_checksum(&self, file_path: &PathBuf, expected_checksum: &str) -> Result<(), String> {
        use sha2::{Sha256, Digest};
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(file_path)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192];
        
        loop {
            let bytes_read = file.read(&mut buffer)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            
            if bytes_read == 0 {
                break;
            }
            
            hasher.update(&buffer[..bytes_read]);
        }
        
        let computed_hash = format!("{:x}", hasher.finalize());
        
        if computed_hash != expected_checksum {
            return Err(format!(
                "Checksum mismatch: expected {}, got {}",
                expected_checksum, computed_hash
            ));
        }
        
        Ok(())
    }
    
    fn get_miner_engine_version(&self) -> Result<String, String> {
        // Try to read version from miner engine binary or config file
        let miner_path = self.get_miner_engine_path()?;
        
        // First, try to read from version file if it exists
        let version_file = miner_path.join("VERSION");
        if version_file.exists() {
            if let Ok(version) = fs::read_to_string(&version_file) {
                return Ok(version.trim().to_string());
            }
        }
        
        // Try to read from pyproject.toml or setup.py if Python package
        let pyproject_toml = miner_path.join("pyproject.toml");
        if pyproject_toml.exists() {
            if let Ok(content) = fs::read_to_string(&pyproject_toml) {
                // Simple version extraction from pyproject.toml
                for line in content.lines() {
                    if line.trim().starts_with("version =") {
                        let version = line
                            .split('=')
                            .nth(1)
                            .and_then(|s| s.trim().strip_prefix('"').and_then(|s| s.strip_suffix('"')))
                            .unwrap_or("")
                            .to_string();
                        if !version.is_empty() {
                            return Ok(version);
                        }
                    }
                }
            }
        }
        
        // Fallback: try to run miner engine with --version flag if it's executable
        let miner_binary = miner_path.join("r3mes-miner");
        if miner_binary.exists() {
            if let Ok(output) = Command::new(&miner_binary)
                .arg("--version")
                .output()
            {
                if output.status.success() {
                    let version_str = String::from_utf8_lossy(&output.stdout);
                    // Extract version number (e.g., "r3mes-miner 1.0.0")
                    for word in version_str.split_whitespace() {
                        if word.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                            return Ok(word.to_string());
                        }
                    }
                }
            }
        }
        
        // Default fallback
        Ok("1.0.0".to_string())
    }
    
    fn get_model_version(&self) -> Result<String, String> {
        // Read version from model directory version file
        let model_dir = self.get_model_directory()?;
        
        // Try to read from VERSION file
        let version_file = model_dir.join("VERSION");
        if version_file.exists() {
            if let Ok(version) = fs::read_to_string(&version_file) {
                return Ok(version.trim().to_string());
            }
        }
        
        // Try to read from model_info.json or similar
        let model_info_file = model_dir.join("model_info.json");
        if model_info_file.exists() {
            if let Ok(content) = fs::read_to_string(&model_info_file) {
                // Simple JSON parsing for version field
                if let Ok(json_value) = serde_json::from_str::<serde_json::Value>(&content) {
                    if let Some(version) = json_value.get("version").and_then(|v| v.as_str()) {
                        return Ok(version.to_string());
                    }
                }
            }
        }
        
        // Default fallback
        Ok("bitnet-v1".to_string())
    }
    
    fn get_chain_binary_version(&self) -> Result<String, String> {
        // Run remesd --version command and parse the output
        let chain_path = self.get_chain_binary_path()?;
        
        if !chain_path.exists() {
            return Ok("v0.1.0".to_string());  // Default if binary doesn't exist
        }
        
        // Try to execute remesd --version
        let output = Command::new(&chain_path)
            .arg("--version")
            .output()
            .map_err(|e| format!("Failed to execute {} --version: {}", chain_path.display(), e))?;
        
        if !output.status.success() {
            return Ok("v0.1.0".to_string());  // Default if command fails
        }
        
        let version_output = String::from_utf8_lossy(&output.stdout);
        
        // Parse version from output (format may vary, e.g., "remesd version v0.1.0" or "v0.1.0")
        // Look for version pattern: v followed by digits and dots
        for line in version_output.lines() {
            for word in line.split_whitespace() {
                if word.starts_with('v') && word.chars().skip(1).any(|c| c.is_ascii_digit()) {
                    return Ok(word.to_string());
                }
                // Also check for version without 'v' prefix
                if word.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                    let parts: Vec<&str> = word.split('.').collect();
                    if parts.len() >= 2 && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit())) {
                        return Ok(format!("v{}", word));
                    }
                }
            }
        }
        
        // Default fallback
        Ok("v0.1.0".to_string())
    }
    
    fn get_miner_engine_path(&self) -> Result<PathBuf, String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        Ok(PathBuf::from(&home).join("R3MES").join("miner-engine"))
    }
    
    fn get_model_directory(&self) -> Result<PathBuf, String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        Ok(PathBuf::from(&home).join(".r3mes").join("models"))
    }
    
    fn get_chain_binary_path(&self) -> Result<PathBuf, String> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        Ok(PathBuf::from(&home).join("R3MES").join("remes").join("build").join("remesd"))
    }
    
    fn extract_model_weights(&self, archive_path: &PathBuf, target_dir: &PathBuf) -> Result<(), String> {
        // Create target directory
        fs::create_dir_all(target_dir)
            .map_err(|e| format!("Failed to create model directory: {}", e))?;
        
        let archive_str = archive_path.to_string_lossy();
        
        // Determine archive type from extension
        if archive_str.ends_with(".tar.gz") || archive_str.ends_with(".tgz") {
            self.extract_tar_gz(archive_path, target_dir)?;
        } else if archive_str.ends_with(".zip") {
            self.extract_zip(archive_path, target_dir)?;
        } else if archive_str.ends_with(".tar") {
            self.extract_tar(archive_path, target_dir)?;
        } else {
            return Err(format!("Unsupported archive format: {}", archive_str));
        }
        
        Ok(())
    }
    
    fn extract_tar_gz(&self, archive_path: &PathBuf, target_dir: &PathBuf) -> Result<(), String> {
        use flate2::read::GzDecoder;
        use tar::Archive;
        
        let file = fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open archive file: {}", e))?;
        
        let tar = GzDecoder::new(file);
        let mut archive = Archive::new(tar);
        
        archive.unpack(target_dir)
            .map_err(|e| format!("Failed to extract tar.gz archive: {}", e))?;
        
        println!("✅ Extracted tar.gz archive to {}", target_dir.display());
        Ok(())
    }
    
    fn extract_tar(&self, archive_path: &PathBuf, target_dir: &PathBuf) -> Result<(), String> {
        use tar::Archive;
        
        let file = fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open archive file: {}", e))?;
        
        let mut archive = Archive::new(file);
        
        archive.unpack(target_dir)
            .map_err(|e| format!("Failed to extract tar archive: {}", e))?;
        
        println!("✅ Extracted tar archive to {}", target_dir.display());
        Ok(())
    }
    
    fn extract_zip(&self, archive_path: &PathBuf, target_dir: &PathBuf) -> Result<(), String> {
        use zip::ZipArchive;
        
        let file = fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open archive file: {}", e))?;
        
        let mut archive = ZipArchive::new(file)
            .map_err(|e| format!("Failed to open zip archive: {}", e))?;
        
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| format!("Failed to read file {} from zip: {}", i, e))?;
            
            let outpath = match file.enclosed_name() {
                Some(path) => target_dir.join(path),
                None => continue,
            };
            
            // Create parent directories if needed
            if let Some(p) = outpath.parent() {
                fs::create_dir_all(p)
                    .map_err(|e| format!("Failed to create directory: {}", e))?;
            }
            
            // Extract file
            if file.is_dir() {
                fs::create_dir_all(&outpath)
                    .map_err(|e| format!("Failed to create directory: {}", e))?;
            } else {
                let mut outfile = fs::File::create(&outpath)
                    .map_err(|e| format!("Failed to create file: {}", e))?;
                std::io::copy(&mut file, &mut outfile)
                    .map_err(|e| format!("Failed to write file: {}", e))?;
            }
        }
        
        println!("✅ Extracted zip archive to {}", target_dir.display());
        Ok(())
    }
}



// ============================================================================
// Tauri Commands for Frontend Integration
// ============================================================================

use std::sync::Mutex;
use once_cell::sync::Lazy;

static UPDATER: Lazy<Mutex<Option<SilentUpdater>>> = Lazy::new(|| Mutex::new(None));

/// Initialize the updater (call once at app startup)
#[tauri::command]
pub fn init_updater() -> Result<(), String> {
    let mut updater_guard = UPDATER.lock().map_err(|e| e.to_string())?;
    *updater_guard = Some(SilentUpdater::new()?);
    Ok(())
}

/// Check for available updates without installing
#[tauri::command]
pub async fn check_updates() -> Result<Vec<(String, String, String)>, String> {
    let updater_guard = UPDATER.lock().map_err(|e| e.to_string())?;
    let updater = updater_guard.as_ref().ok_or("Updater not initialized")?;
    updater.check_for_updates().await
}

/// Perform updates with automatic rollback on failure
#[tauri::command]
pub async fn perform_updates() -> Result<Vec<UpdateResult>, String> {
    let mut updater_guard = UPDATER.lock().map_err(|e| e.to_string())?;
    let updater = updater_guard.as_mut().ok_or("Updater not initialized")?;
    updater.check_and_update().await
}

/// Rollback a specific component to previous version
#[tauri::command]
pub fn rollback_update(component: String) -> Result<(), String> {
    let mut updater_guard = UPDATER.lock().map_err(|e| e.to_string())?;
    let updater = updater_guard.as_mut().ok_or("Updater not initialized")?;
    updater.rollback_component(&component)
}

/// Get current versions of all components
#[tauri::command]
pub fn get_current_versions() -> Result<HashMap<String, String>, String> {
    let updater_guard = UPDATER.lock().map_err(|e| e.to_string())?;
    let updater = updater_guard.as_ref().ok_or("Updater not initialized")?;
    
    let mut versions = HashMap::new();
    versions.insert("chain_binary".to_string(), updater.get_chain_binary_version().unwrap_or_default());
    versions.insert("model_weights".to_string(), updater.get_model_version().unwrap_or_default());
    versions.insert("miner_engine".to_string(), updater.get_miner_engine_version().unwrap_or_default());
    versions.insert("launcher".to_string(), env!("CARGO_PKG_VERSION").to_string());
    
    Ok(versions)
}
