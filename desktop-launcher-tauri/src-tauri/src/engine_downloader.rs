use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;
use std::io::{self, Write, Read};
use sha2::{Sha256, Digest};
use futures_util::StreamExt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub percentage: f64,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub speed_bytes_per_sec: f64,
    pub eta_seconds: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EngineStatus {
    pub installed: bool,
    pub version: Option<String>,
    pub path: Option<PathBuf>,
    pub checksum_valid: bool,
}

pub struct EngineDownloader {
    engine_dir: PathBuf,
    download_url: String,
    expected_checksum: Option<String>,
}

impl EngineDownloader {
    /// Create a new EngineDownloader instance
    pub fn new(engine_dir: PathBuf, download_url: String, expected_checksum: Option<String>) -> Self {
        Self {
            engine_dir,
            download_url,
            expected_checksum,
        }
    }

    /// Get the default engine directory
    pub fn default_engine_dir() -> PathBuf {
        #[cfg(windows)]
        {
            let appdata = std::env::var("APPDATA").unwrap_or_else(|_| {
                std::env::var("USERPROFILE").unwrap() + "\\AppData\\Roaming"
            });
            PathBuf::from(appdata).join("R3MES").join("engine")
        }
        
        #[cfg(not(windows))]
        {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            PathBuf::from(home).join(".r3mes").join("engine")
        }
    }

    /// Get the engine executable path
    pub fn engine_path(&self) -> PathBuf {
        #[cfg(windows)]
        {
            self.engine_dir.join("engine.exe")
        }
        
        #[cfg(not(windows))]
        {
            self.engine_dir.join("engine")
        }
    }

    /// Check if engine is installed and valid
    pub async fn check_engine_status(&self) -> EngineStatus {
        let engine_path = self.engine_path();
        
        if !engine_path.exists() {
            return EngineStatus {
                installed: false,
                version: None,
                path: None,
                checksum_valid: false,
            };
        }

        // Check checksum if expected
        let checksum_valid = if let Some(expected) = &self.expected_checksum {
            self.verify_checksum(&engine_path, expected).await.unwrap_or(false)
        } else {
            true // If no checksum provided, assume valid
        };

        // Try to get version (by running engine --version)
        let version = Self::get_engine_version(&engine_path).await;

        EngineStatus {
            installed: true,
            version,
            path: Some(engine_path),
            checksum_valid,
        }
    }

    /// Get engine version by running engine --version
    async fn get_engine_version(engine_path: &Path) -> Option<String> {
        use std::process::Command;
        
        // For version check, we can use blocking Command since it's fast
        let output = Command::new(engine_path)
            .arg("--version")
            .output()
            .ok()?;
        
        if output.status.success() {
            String::from_utf8(output.stdout)
                .ok()
                .map(|s| s.trim().to_string())
        } else {
            None
        }
    }

    /// Ensure engine is installed and ready
    pub async fn ensure_engine_installed(&self) -> Result<PathBuf, String> {
        let status = self.check_engine_status().await;
        
        if status.installed && status.checksum_valid {
            return Ok(status.path.unwrap());
        }

        // Engine not installed or invalid, need to download
        Err("Engine not installed or invalid checksum".to_string())
    }

    /// Download engine with progress tracking
    pub async fn download_with_progress<F>(
        &self,
        mut progress_callback: F,
    ) -> Result<PathBuf, String>
    where
        F: FnMut(DownloadProgress),
    {
        // Create engine directory if it doesn't exist
        fs::create_dir_all(&self.engine_dir)
            .map_err(|e| format!("Failed to create engine directory: {}", e))?;

        let engine_path = self.engine_path();
        let temp_path = engine_path.with_extension("tmp");

        // Get file size from server (HEAD request)
        let client = reqwest::Client::new();
        let response = client
            .head(&self.download_url)
            .send()
            .await
            .map_err(|e| format!("Failed to check download URL: {}", e))?;

        let total_bytes = response
            .headers()
            .get("content-length")
            .and_then(|h| h.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());

        // Check if partial download exists
        let mut downloaded_bytes = 0u64;
        if temp_path.exists() {
            downloaded_bytes = fs::metadata(&temp_path)
                .map(|m| m.len())
                .unwrap_or(0);
            
            if let Some(total) = total_bytes {
                if downloaded_bytes >= total {
                    // File already fully downloaded
                    fs::rename(&temp_path, &engine_path)
                        .map_err(|e| format!("Failed to rename temp file: {}", e))?;
                    return Ok(engine_path);
                }
            }
        }

        // Create request with Range header for resume if needed
        let mut request_builder = client.get(&self.download_url);
        if downloaded_bytes > 0 {
            request_builder = request_builder.header("Range", format!("bytes={}-", downloaded_bytes));
        }

        let response = request_builder
            .send()
            .await
            .map_err(|e| format!("Failed to download engine: {}", e))?;

        if !response.status().is_success() && response.status() != 206 {
            return Err(format!("Download failed: HTTP {}", response.status()));
        }

        // Open file for appending if resuming, or create new if starting fresh
        let mut file = if downloaded_bytes > 0 {
            fs::OpenOptions::new()
                .append(true)
                .open(&temp_path)
                .map_err(|e| format!("Failed to open temp file for appending: {}", e))?
        } else {
            fs::File::create(&temp_path)
                .map_err(|e| format!("Failed to create temp file: {}", e))?
        };

        // Stream download with progress tracking
        let mut stream = response.bytes_stream();
        let start_time = std::time::Instant::now();
        let mut last_update = std::time::Instant::now();
        let mut last_bytes = downloaded_bytes;

        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result.map_err(|e| format!("Failed to read chunk: {}", e))?;
            let bytes_read = chunk.len();

            file.write_all(&chunk)
                .map_err(|e| format!("Failed to write to temp file: {}", e))?;

            downloaded_bytes += bytes_read as u64;

            // Update progress every 100ms
            if last_update.elapsed().as_millis() >= 100 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let speed = if elapsed > 0.0 {
                    downloaded_bytes as f64 / elapsed
                } else {
                    0.0
                };

                let percentage = if let Some(total) = total_bytes {
                    (downloaded_bytes as f64 / total as f64) * 100.0
                } else {
                    // If total unknown, we can't calculate percentage
                    0.0
                };

                let eta = if let Some(total) = total_bytes {
                    let remaining = total.saturating_sub(downloaded_bytes);
                    if speed > 0.0 {
                        Some((remaining as f64 / speed) as u64)
                    } else {
                        None
                    }
                } else {
                    None
                };

                progress_callback(DownloadProgress {
                    percentage,
                    downloaded_bytes,
                    total_bytes,
                    speed_bytes_per_sec: speed,
                    eta_seconds: eta,
                });

                last_update = std::time::Instant::now();
                last_bytes = downloaded_bytes;
            }
        }

        // Move temp file to final location
        drop(file); // Close file before renaming
        fs::rename(&temp_path, &engine_path)
            .map_err(|e| format!("Failed to rename temp file: {}", e))?;

        // Verify checksum if provided
        if let Some(expected_checksum) = &self.expected_checksum {
            if !self.verify_checksum(&engine_path, expected_checksum).await? {
                fs::remove_file(&engine_path).ok(); // Clean up invalid file
                return Err("Checksum verification failed".to_string());
            }
        }

        // Make executable on Unix systems
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&engine_path)
                .map_err(|e| format!("Failed to get file metadata: {}", e))?
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&engine_path, perms)
                .map_err(|e| format!("Failed to set executable permissions: {}", e))?;
        }

        Ok(engine_path)
    }

    /// Verify SHA256 checksum of a file
    pub async fn verify_checksum(&self, file_path: &Path, expected_checksum: &str) -> Result<bool, String> {
        let mut file = fs::File::open(file_path)
            .map_err(|e| format!("Failed to open file for checksum verification: {}", e))?;

        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192];

        loop {
            let bytes_read = std::io::Read::read(&mut file, &mut buffer)
                .map_err(|e| format!("Failed to read file for checksum: {}", e))?;

            if bytes_read == 0 {
                break;
            }

            hasher.update(&buffer[..bytes_read]);
        }

        let computed_checksum = format!("{:x}", hasher.finalize());
        let expected_checksum_clean = expected_checksum.trim().to_lowercase().replace("sha256:", "");

        Ok(computed_checksum == expected_checksum_clean)
    }

    /// Get engine download URL from manifest or use default
    pub fn get_download_url() -> String {
        // Default CDN URL (can be overridden via environment variable)
        std::env::var("R3MES_ENGINE_CDN_URL")
            .unwrap_or_else(|_| "https://releases.r3mes.network/engine-v1.0.0.zip".to_string())
    }

    /// Get expected checksum from manifest or environment
    pub fn get_expected_checksum() -> Option<String> {
        std::env::var("R3MES_ENGINE_CHECKSUM").ok()
    }
}

