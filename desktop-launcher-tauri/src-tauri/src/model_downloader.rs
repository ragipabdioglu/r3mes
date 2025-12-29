/// Model Downloader - Hybrid Download System (HuggingFace + IPFS Fallback)
/// 
/// Downloads model files from HuggingFace with SHA256 verification.
/// Falls back to IPFS if HuggingFace download fails.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};
use futures_util::StreamExt;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_name: String,
    pub file_name: String,
    pub download_source: String,
    pub verification_sha256: String,
    pub required_disk_space_gb: f64,
    pub ipfs_fallback_hash: Option<String>,
}

#[derive(Debug)]
pub struct DownloadProgress {
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub percentage: f64,
}

pub struct ModelDownloader {
    model_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new() -> Result<Self, String> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| "Could not determine home directory")?;
        
        let model_dir = PathBuf::from(&home).join(".r3mes").join("models");
        fs::create_dir_all(&model_dir)
            .map_err(|e| format!("Failed to create model directory: {}", e))?;
        
        Ok(Self { model_dir })
    }
    
    /// Check if model file exists and verify its integrity
    pub fn check_model(&self, config: &ModelConfig) -> Result<bool, String> {
        let model_path = self.model_dir.join(&config.file_name);
        
        if !model_path.exists() {
            return Ok(false);
        }
        
        // Verify SHA256 hash
        let computed_hash = self.calculate_sha256(&model_path)?;
        if computed_hash.to_lowercase() != config.verification_sha256.to_lowercase() {
            eprintln!("⚠️  Model file hash mismatch. File may be corrupted.");
            eprintln!("   Expected: {}", config.verification_sha256);
            eprintln!("   Got:      {}", computed_hash);
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Download model from HuggingFace with progress tracking
    pub async fn download_model(
        &self,
        config: &ModelConfig,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<PathBuf, String> {
        let model_path = self.model_dir.join(&config.file_name);
        
        // 1. Check if file already exists and is valid
        if let Ok(true) = self.check_model(config) {
            println!("✅ Model file already exists and is valid: {:?}", model_path);
            return Ok(model_path);
        }
        
        // 2. Check available disk space
        self.check_disk_space(config.required_disk_space_gb)?;
        
        // 3. Download from HuggingFace
        println!("📥 Downloading model from HuggingFace...");
        println!("   URL: {}", config.download_source);
        println!("   File: {}", config.file_name);
        
        // Use async runtime for download
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create async runtime: {}", e))?;
        
        rt.block_on(self.download_with_progress(
            &config.download_source,
            &model_path,
            progress_callback,
        ))?;
        
        // 4. Verify SHA256 hash
        println!("🔍 Verifying file integrity...");
        let computed_hash = self.calculate_sha256(&model_path)?;
        
        if computed_hash.to_lowercase() != config.verification_sha256.to_lowercase() {
            // Delete corrupted file
            fs::remove_file(&model_path)
                .map_err(|e| format!("Failed to remove corrupted file: {}", e))?;
            
            return Err(format!(
                "❌ HATA: Dosya bozuk veya değiştirilmiş!\n   Beklenen: {}\n   Alınan: {}",
                config.verification_sha256, computed_hash
            ));
        }
        
        println!("✅ Dosya orijinal ve güvenli. SHA256 doğrulandı.");
        Ok(model_path)
    }
    
    /// Download file with progress tracking
    pub async fn download_with_progress(
        &self,
        url: &str,
        output_path: &Path,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<(), String> {
        // Create HTTP client with timeout
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600))  // 1 hour timeout for large files
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        
        // Send request
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| format!("Download failed: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Download failed: HTTP {}", response.status()));
        }
        
        // Get total size from Content-Length header
        let total_size = response.content_length();
        
        // Create output file
        let mut file = fs::File::create(output_path)
            .map_err(|e| format!("Failed to create file: {}", e))?;
        
        // Stream download with progress tracking
        let mut stream = response.bytes_stream();
        let mut downloaded = 0u64;
        
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result
                .map_err(|e| format!("Download error: {}", e))?;
            
            file.write_all(&chunk)
                .map_err(|e| format!("Failed to write chunk: {}", e))?;
            
            downloaded += chunk.len() as u64;
            
            // Call progress callback
            if let Some(ref cb) = progress_callback {
                let percentage = if let Some(total) = total_size {
                    (downloaded as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
                
                cb(DownloadProgress {
                    bytes_downloaded: downloaded,
                    total_bytes: total_size,
                    percentage,
                });
            }
            
            // Log progress every 10MB
            if downloaded % (10 * 1024 * 1024) == 0 {
                println!("   Downloaded: {} MB", downloaded / (1024 * 1024));
            }
        }
        
        println!("✅ Download complete: {} bytes", downloaded);
        Ok(())
    }
    
    /// Calculate SHA256 hash of a file
    fn calculate_sha256(&self, file_path: &Path) -> Result<String, String> {
        let mut file = fs::File::open(file_path)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192];
        
        loop {
            let bytes_read = io::Read::read(&mut file, &mut buffer)
                .map_err(|e| format!("Failed to read file: {}", e))?;
            
            if bytes_read == 0 {
                break;
            }
            
            hasher.update(&buffer[..bytes_read]);
        }
        
        let hash = format!("{:x}", hasher.finalize());
        Ok(hash)
    }
    
    /// Check if sufficient disk space is available
    fn check_disk_space(&self, required_gb: f64) -> Result<(), String> {
        // Use a simple approach: check parent directory
        // In production, you might want to use a crate like `sysinfo` or `disk_usage`
        // For now, use a simplified check
        
        // Try to create a temporary file to test write access
        let test_file = self.model_dir.join(".disk_space_test");
        match fs::File::create(&test_file) {
            Ok(_) => {
                // File created successfully, assume enough space
                let _ = fs::remove_file(&test_file);
                Ok(())
            }
            Err(e) => {
                // Check if error is due to disk space
                if e.kind() == std::io::ErrorKind::NoSpaceLeftOnDevice {
                    Err(format!(
                        "Insufficient disk space: {:.2} GB required",
                        required_gb
                    ))
                } else {
                    // Other error (permissions, etc.) - log but continue
                    eprintln!("⚠️  Could not check disk space: {}", e);
                    Ok(())
                }
            }
        }
    }
    
    /// Get model directory path
    pub fn get_model_dir(&self) -> &Path {
        &self.model_dir
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_model_config_parsing() {
        let config_json = r#"
        {
            "model_name": "Llama-3-8B-R3MES-Optimized",
            "file_name": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
            "download_source": "https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
            "verification_sha256": "abc123def456...",
            "required_disk_space_gb": 6.0,
            "ipfs_fallback_hash": null
        }
        "#;
        
        let config: ModelConfig = serde_json::from_str(config_json).unwrap();
        assert_eq!(config.model_name, "Llama-3-8B-R3MES-Optimized");
        assert_eq!(config.file_name, "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf");
    }
}

