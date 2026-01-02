/// AI model downloader and manager for R3MES Desktop Launcher
/// 
/// Handles downloading, verification, and management of AI models
/// from IPFS and other sources for mining and serving operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use sha2::{Sha256, Digest};
use futures_util::StreamExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub size_bytes: u64,
    pub checksum: String,
    pub ipfs_hash: String,
    pub download_urls: Vec<String>,
    pub model_type: ModelType,
    pub requirements: ModelRequirements,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    BitNet,
    Transformer,
    CNN,
    RNN,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRequirements {
    pub min_vram_gb: f64,
    pub min_ram_gb: f64,
    pub cuda_required: bool,
    pub python_version: String,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub name: String,
    pub installed: bool,
    pub version: Option<String>,
    pub size_bytes: u64,
    pub checksum_valid: bool,
    pub last_updated: Option<u64>,
    pub installation_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub model_name: String,
    pub stage: DownloadStage,
    pub progress_percent: f64,
    pub bytes_downloaded: u64,
    pub total_bytes: u64,
    pub speed_bps: u64,
    pub eta_seconds: u64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DownloadStage {
    Initializing,
    Downloading,
    Verifying,
    Installing,
    Complete,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadResult {
    pub success: bool,
    pub model_name: String,
    pub message: String,
    pub installation_path: Option<String>,
    pub checksum: Option<String>,
}

pub struct ModelDownloader {
    models_dir: PathBuf,
    cache_dir: PathBuf,
    ipfs_gateway: String,
    fallback_urls: Vec<String>,
    client: reqwest::Client,
}

impl ModelDownloader {
    /// Create a new model downloader
    pub fn new() -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        
        let models_dir = PathBuf::from(&home).join(".r3mes").join("models");
        let cache_dir = PathBuf::from(&home).join(".r3mes").join("cache");
        
        let ipfs_gateway = std::env::var("IPFS_GATEWAY")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());
        
        let fallback_urls = vec![
            "https://models.r3mes.network".to_string(),
            "https://huggingface.co/R3MES-Network".to_string(),
        ];
        
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600)) // 1 hour for large models
            .build()
            .unwrap();
        
        Self {
            models_dir,
            cache_dir,
            ipfs_gateway,
            fallback_urls,
            client,
        }
    }
    
    /// Get available models
    pub async fn get_available_models(&self) -> Result<Vec<ModelInfo>, String> {
        // In a real implementation, this would fetch from a registry
        Ok(vec![
            ModelInfo {
                name: "bitnet-b1.58".to_string(),
                version: "1.0.0".to_string(),
                description: "BitNet b1.58 model for efficient inference".to_string(),
                size_bytes: 28 * 1024 * 1024 * 1024, // 28GB
                checksum: "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456".to_string(),
                ipfs_hash: "QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG".to_string(),
                download_urls: vec![
                    "https://models.r3mes.network/bitnet-b1.58-v1.0.0.tar.gz".to_string(),
                ],
                model_type: ModelType::BitNet,
                requirements: ModelRequirements {
                    min_vram_gb: 8.0,
                    min_ram_gb: 16.0,
                    cuda_required: true,
                    python_version: ">=3.8".to_string(),
                    dependencies: vec![
                        "torch>=2.0.0".to_string(),
                        "transformers>=4.30.0".to_string(),
                        "numpy>=1.21.0".to_string(),
                    ],
                },
                metadata: HashMap::new(),
            },
            ModelInfo {
                name: "llama2-7b".to_string(),
                version: "1.0.0".to_string(),
                description: "LLaMA 2 7B parameter model".to_string(),
                size_bytes: 13 * 1024 * 1024 * 1024, // 13GB
                checksum: "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567a".to_string(),
                ipfs_hash: "QmXwBPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdH".to_string(),
                download_urls: vec![
                    "https://models.r3mes.network/llama2-7b-v1.0.0.tar.gz".to_string(),
                ],
                model_type: ModelType::Transformer,
                requirements: ModelRequirements {
                    min_vram_gb: 16.0,
                    min_ram_gb: 32.0,
                    cuda_required: true,
                    python_version: ">=3.8".to_string(),
                    dependencies: vec![
                        "torch>=2.0.0".to_string(),
                        "transformers>=4.30.0".to_string(),
                        "sentencepiece>=0.1.99".to_string(),
                    ],
                },
                metadata: HashMap::new(),
            },
        ])
    }
    
    /// Get installed models
    pub async fn get_installed_models(&self) -> Result<Vec<ModelStatus>, String> {
        let mut models = Vec::new();
        
        if !self.models_dir.exists() {
            return Ok(models);
        }
        
        for entry in fs::read_dir(&self.models_dir)
            .map_err(|e| format!("Failed to read models directory: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            
            if path.is_dir() {
                if let Some(model_name) = path.file_name().and_then(|n| n.to_str()) {
                    let status = self.get_model_status(model_name).await?;
                    models.push(status);
                }
            }
        }
        
        Ok(models)
    }
    
    /// Get status of a specific model
    pub async fn get_model_status(&self, model_name: &str) -> Result<ModelStatus, String> {
        let model_dir = self.models_dir.join(model_name);
        let installed = model_dir.exists();
        
        let mut version = None;
        let mut size_bytes = 0;
        let mut checksum_valid = false;
        let mut last_updated = None;
        let mut installation_path = None;
        
        if installed {
            installation_path = Some(model_dir.to_string_lossy().to_string());
            
            // Read metadata file
            let metadata_file = model_dir.join("metadata.json");
            if metadata_file.exists() {
                if let Ok(metadata_content) = fs::read_to_string(&metadata_file) {
                    if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&metadata_content) {
                        version = metadata["version"].as_str().map(|s| s.to_string());
                        size_bytes = metadata["size_bytes"].as_u64().unwrap_or(0);
                        last_updated = metadata["installed_at"].as_u64();
                    }
                }
            }
            
            // Calculate directory size if not in metadata
            if size_bytes == 0 {
                size_bytes = self.calculate_directory_size(&model_dir)?;
            }
            
            // Verify checksum
            checksum_valid = self.verify_model_checksum(model_name).await.unwrap_or(false);
        }
        
        Ok(ModelStatus {
            name: model_name.to_string(),
            installed,
            version,
            size_bytes,
            checksum_valid,
            last_updated,
            installation_path,
        })
    }
    
    /// Download and install a model
    pub async fn download_model(&self, model_info: &ModelInfo) -> Result<DownloadResult, String> {
        // Create directories
        fs::create_dir_all(&self.models_dir)
            .map_err(|e| format!("Failed to create models directory: {}", e))?;
        
        fs::create_dir_all(&self.cache_dir)
            .map_err(|e| format!("Failed to create cache directory: {}", e))?;
        
        let model_dir = self.models_dir.join(&model_info.name);
        let cache_file = self.cache_dir.join(format!("{}.tar.gz", model_info.name));
        
        // Try different download sources
        let mut download_success = false;
        let mut last_error = String::new();
        
        // Try IPFS first
        if !model_info.ipfs_hash.is_empty() {
            match self.download_from_ipfs(&model_info.ipfs_hash, &cache_file).await {
                Ok(_) => download_success = true,
                Err(e) => {
                    last_error = format!("IPFS download failed: {}", e);
                    println!("IPFS download failed, trying fallback URLs...");
                }
            }
        }
        
        // Try direct URLs if IPFS failed
        if !download_success {
            for url in &model_info.download_urls {
                match self.download_from_url(url, &cache_file, model_info.size_bytes).await {
                    Ok(_) => {
                        download_success = true;
                        break;
                    }
                    Err(e) => {
                        last_error = format!("URL download failed: {}", e);
                        println!("Download from {} failed: {}", url, e);
                    }
                }
            }
        }
        
        if !download_success {
            return Err(format!("All download attempts failed. Last error: {}", last_error));
        }
        
        // Verify checksum
        println!("Verifying model checksum...");
        let actual_checksum = self.calculate_file_checksum(&cache_file).await?;
        if actual_checksum != model_info.checksum {
            fs::remove_file(&cache_file).ok();
            return Err(format!(
                "Checksum verification failed. Expected: {}, Got: {}",
                model_info.checksum, actual_checksum
            ));
        }
        
        // Extract model
        println!("Extracting model...");
        self.extract_model(&cache_file, &model_dir).await?;
        
        // Save metadata
        self.save_model_metadata(model_info, &model_dir).await?;
        
        // Clean up cache file
        fs::remove_file(&cache_file).ok();
        
        Ok(DownloadResult {
            success: true,
            model_name: model_info.name.clone(),
            message: format!("Model {} downloaded and installed successfully", model_info.name),
            installation_path: Some(model_dir.to_string_lossy().to_string()),
            checksum: Some(actual_checksum),
        })
    }
    
    /// Download from IPFS
    async fn download_from_ipfs(&self, ipfs_hash: &str, output_path: &PathBuf) -> Result<(), String> {
        let url = format!("{}/ipfs/{}", self.ipfs_gateway, ipfs_hash);
        self.download_from_url(&url, output_path, 0).await
    }
    
    /// Download from URL with progress tracking
    async fn download_from_url(
        &self,
        url: &str,
        output_path: &PathBuf,
        expected_size: u64,
    ) -> Result<(), String> {
        println!("Downloading from: {}", url);
        
        let response = self.client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("Failed to start download: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Download failed with status: {}", response.status()));
        }
        
        let total_size = response.content_length().unwrap_or(expected_size);
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        
        let mut file = fs::File::create(output_path)
            .map_err(|e| format!("Failed to create output file: {}", e))?;
        
        use std::io::Write;
        let start_time = std::time::Instant::now();
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download error: {}", e))?;
            
            file.write_all(&chunk)
                .map_err(|e| format!("Failed to write to file: {}", e))?;
            
            downloaded += chunk.len() as u64;
            
            // Log progress every 100MB or 5%
            let progress_threshold = std::cmp::max(100 * 1024 * 1024, total_size / 20);
            if downloaded % progress_threshold == 0 || downloaded == total_size {
                let elapsed = start_time.elapsed().as_secs_f64();
                let speed = if elapsed > 0.0 { downloaded as f64 / elapsed } else { 0.0 };
                let percentage = if total_size > 0 {
                    (downloaded as f64 / total_size as f64) * 100.0
                } else {
                    0.0
                };
                
                println!(
                    "Download progress: {:.1}% ({} / {} bytes) - Speed: {:.1} MB/s",
                    percentage,
                    downloaded,
                    total_size,
                    speed / (1024.0 * 1024.0)
                );
            }
        }
        
        file.flush().map_err(|e| format!("Failed to flush file: {}", e))?;
        
        println!("Download completed: {} bytes", downloaded);
        Ok(())
    }
    
    /// Extract model archive
    async fn extract_model(&self, archive_path: &PathBuf, extract_to: &PathBuf) -> Result<(), String> {
        use flate2::read::GzDecoder;
        use tar::Archive;
        
        // Remove existing directory if it exists
        if extract_to.exists() {
            fs::remove_dir_all(extract_to)
                .map_err(|e| format!("Failed to remove existing model directory: {}", e))?;
        }
        
        fs::create_dir_all(extract_to)
            .map_err(|e| format!("Failed to create model directory: {}", e))?;
        
        let file = fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open archive: {}", e))?;
        
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        
        archive.unpack(extract_to)
            .map_err(|e| format!("Failed to extract archive: {}", e))?;
        
        Ok(())
    }
    
    /// Save model metadata
    async fn save_model_metadata(&self, model_info: &ModelInfo, model_dir: &PathBuf) -> Result<(), String> {
        let metadata = serde_json::json!({
            "name": model_info.name,
            "version": model_info.version,
            "description": model_info.description,
            "size_bytes": model_info.size_bytes,
            "checksum": model_info.checksum,
            "ipfs_hash": model_info.ipfs_hash,
            "model_type": model_info.model_type,
            "requirements": model_info.requirements,
            "installed_at": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            "metadata": model_info.metadata,
        });
        
        let metadata_file = model_dir.join("metadata.json");
        fs::write(&metadata_file, serde_json::to_string_pretty(&metadata).unwrap())
            .map_err(|e| format!("Failed to save metadata: {}", e))?;
        
        Ok(())
    }
    
    /// Calculate file checksum
    async fn calculate_file_checksum(&self, file_path: &PathBuf) -> Result<String, String> {
        use std::io::Read;
        
        let mut file = fs::File::open(file_path)
            .map_err(|e| format!("Failed to open file for checksum: {}", e))?;
        
        let mut hasher = Sha256::new();
        std::io::copy(&mut file, &mut hasher)
            .map_err(|e| format!("Failed to read file for checksum: {}", e))?;
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Verify model checksum
    async fn verify_model_checksum(&self, model_name: &str) -> Result<bool, String> {
        let model_dir = self.models_dir.join(model_name);
        let metadata_file = model_dir.join("metadata.json");
        
        if !metadata_file.exists() {
            return Ok(false);
        }
        
        let metadata_content = fs::read_to_string(&metadata_file)
            .map_err(|e| format!("Failed to read metadata: {}", e))?;
        
        let metadata: serde_json::Value = serde_json::from_str(&metadata_content)
            .map_err(|e| format!("Failed to parse metadata: {}", e))?;
        
        let expected_checksum = metadata["checksum"]
            .as_str()
            .ok_or("No checksum in metadata")?;
        
        // Calculate checksum of all files in model directory
        let actual_checksum = self.calculate_directory_checksum(&model_dir).await?;
        
        Ok(actual_checksum == expected_checksum)
    }
    
    /// Calculate directory checksum
    async fn calculate_directory_checksum(&self, dir_path: &PathBuf) -> Result<String, String> {
        let mut hasher = Sha256::new();
        
        // Get all files in directory recursively
        let mut files = Vec::new();
        self.collect_files(dir_path, &mut files)?;
        
        // Sort files for consistent checksum
        files.sort();
        
        for file_path in files {
            // Skip metadata file to avoid circular dependency
            if file_path.file_name().and_then(|n| n.to_str()) == Some("metadata.json") {
                continue;
            }
            
            let mut file = fs::File::open(&file_path)
                .map_err(|e| format!("Failed to open file for checksum: {}", e))?;
            
            std::io::copy(&mut file, &mut hasher)
                .map_err(|e| format!("Failed to read file for checksum: {}", e))?;
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Collect all files in directory recursively
    fn collect_files(&self, dir_path: &PathBuf, files: &mut Vec<PathBuf>) -> Result<(), String> {
        for entry in fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            
            if path.is_dir() {
                self.collect_files(&path, files)?;
            } else {
                files.push(path);
            }
        }
        
        Ok(())
    }
    
    /// Calculate directory size
    fn calculate_directory_size(&self, dir_path: &PathBuf) -> Result<u64, String> {
        let mut total_size = 0;
        
        for entry in fs::read_dir(dir_path)
            .map_err(|e| format!("Failed to read directory: {}", e))?
        {
            let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
            let path = entry.path();
            
            if path.is_dir() {
                total_size += self.calculate_directory_size(&path)?;
            } else {
                let metadata = fs::metadata(&path)
                    .map_err(|e| format!("Failed to get file metadata: {}", e))?;
                total_size += metadata.len();
            }
        }
        
        Ok(total_size)
    }
    
    /// Remove a model
    pub async fn remove_model(&self, model_name: &str) -> Result<(), String> {
        let model_dir = self.models_dir.join(model_name);
        
        if model_dir.exists() {
            fs::remove_dir_all(&model_dir)
                .map_err(|e| format!("Failed to remove model directory: {}", e))?;
        }
        
        Ok(())
    }
    
    /// Update a model
    pub async fn update_model(&self, model_name: &str) -> Result<DownloadResult, String> {
        // Get current model info
        let available_models = self.get_available_models().await?;
        let model_info = available_models
            .iter()
            .find(|m| m.name == model_name)
            .ok_or_else(|| format!("Model {} not found", model_name))?;
        
        // Check if update is needed
        let current_status = self.get_model_status(model_name).await?;
        
        if current_status.installed {
            if let Some(current_version) = current_status.version {
                if current_version == model_info.version {
                    return Ok(DownloadResult {
                        success: true,
                        model_name: model_name.to_string(),
                        message: "Model is already up to date".to_string(),
                        installation_path: current_status.installation_path,
                        checksum: None,
                    });
                }
            }
        }
        
        // Download and install update
        self.download_model(model_info).await
    }
    
    /// Get model download progress (placeholder for UI integration)
    pub async fn get_download_progress(&self, model_name: &str) -> Option<DownloadProgress> {
        // This would be implemented with shared state between download and UI
        // For now, return None (no active download)
        None
    }
    
    /// Cancel model download
    pub async fn cancel_download(&self, model_name: &str) -> Result<(), String> {
        // This would require cancellation token implementation
        // For now, just return success
        Ok(())
    }
}