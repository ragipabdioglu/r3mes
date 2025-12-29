use std::process::Command;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct InstallationResult {
    pub success: bool,
    pub message: String,
    pub progress: f64, // 0.0 to 1.0
}

pub async fn install_docker() -> Result<InstallationResult, String> {
    // Check OS and install Docker accordingly
    #[cfg(target_os = "linux")]
    {
        // Linux installation
        let output = Command::new("sh")
            .arg("-c")
            .arg("curl -fsSL https://get.docker.com -o get-docker.sh && sh get-docker.sh")
            .output()
            .map_err(|e| format!("Failed to install Docker: {}", e))?;

        if output.status.success() {
            Ok(InstallationResult {
                success: true,
                message: "Docker installed successfully".to_string(),
                progress: 1.0,
            })
        } else {
            Err(format!("Docker installation failed: {}", String::from_utf8_lossy(&output.stderr)))
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows: Direct user to download Docker Desktop
        Err("Please install Docker Desktop from https://www.docker.com/products/docker-desktop".to_string())
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: Use Homebrew
        let output = Command::new("brew")
            .arg("install")
            .arg("--cask")
            .arg("docker")
            .output()
            .map_err(|e| format!("Failed to install Docker: {}", e))?;

        if output.status.success() {
            Ok(InstallationResult {
                success: true,
                message: "Docker installed successfully".to_string(),
                progress: 1.0,
            })
        } else {
            Err(format!("Docker installation failed: {}", String::from_utf8_lossy(&output.stderr)))
        }
    }
}

pub async fn install_python() -> Result<InstallationResult, String> {
    #[cfg(target_os = "linux")]
    {
        let output = Command::new("sh")
            .arg("-c")
            .arg("sudo apt-get update && sudo apt-get install -y python3 python3-pip")
            .output()
            .map_err(|e| format!("Failed to install Python: {}", e))?;

        if output.status.success() {
            Ok(InstallationResult {
                success: true,
                message: "Python installed successfully".to_string(),
                progress: 1.0,
            })
        } else {
            Err(format!("Python installation failed: {}", String::from_utf8_lossy(&output.stderr)))
        }
    }

    #[cfg(target_os = "windows")]
    {
        Err("Please install Python from https://www.python.org/downloads/".to_string())
    }

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("brew")
            .arg("install")
            .arg("python@3.10")
            .output()
            .map_err(|e| format!("Failed to install Python: {}", e))?;

        if output.status.success() {
            Ok(InstallationResult {
                success: true,
                message: "Python installed successfully".to_string(),
                progress: 1.0,
            })
        } else {
            Err(format!("Python installation failed: {}", String::from_utf8_lossy(&output.stderr)))
        }
    }
}

pub async fn install_nodejs() -> Result<InstallationResult, String> {
    #[cfg(target_os = "linux")]
    {
        let output = Command::new("sh")
            .arg("-c")
            .arg("curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs")
            .output()
            .map_err(|e| format!("Failed to install Node.js: {}", e))?;

        if output.status.success() {
            Ok(InstallationResult {
                success: true,
                message: "Node.js installed successfully".to_string(),
                progress: 1.0,
            })
        } else {
            Err(format!("Node.js installation failed: {}", String::from_utf8_lossy(&output.stderr)))
        }
    }

    #[cfg(target_os = "windows")]
    {
        Err("Please install Node.js from https://nodejs.org/".to_string())
    }

    #[cfg(target_os = "macos")]
    {
        let output = Command::new("brew")
            .arg("install")
            .arg("node")
            .output()
            .map_err(|e| format!("Failed to install Node.js: {}", e))?;

        if output.status.success() {
            Ok(InstallationResult {
                success: true,
                message: "Node.js installed successfully".to_string(),
                progress: 1.0,
            })
        } else {
            Err(format!("Node.js installation failed: {}", String::from_utf8_lossy(&output.stderr)))
        }
    }
}

pub async fn install_cuda() -> Result<InstallationResult, String> {
    // CUDA installation is complex and OS-specific
    // For now, provide instructions
    Err("Please install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads".to_string())
}

