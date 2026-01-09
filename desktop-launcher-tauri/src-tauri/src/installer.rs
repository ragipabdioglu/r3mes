//! Component installer for R3MES Desktop Launcher
//! 
//! Handles installation of system dependencies, tools, and components
//! required for R3MES operations across different platforms.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::process::Command;
use std::path::PathBuf;
use std::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallationResult {
    pub success: bool,
    pub component: String,
    pub message: String,
    pub version: Option<String>,
    pub installation_path: Option<String>,
    pub requires_restart: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentInfo {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub installed: bool,
    pub version: Option<String>,
    pub installation_method: InstallationMethod,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InstallationMethod {
    PackageManager,
    Download,
    Script,
    Manual,
}

pub struct ComponentInstaller {
    temp_dir: PathBuf,
    install_dir: PathBuf,
}

impl ComponentInstaller {
    /// Create a new component installer
    pub fn new() -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .unwrap_or_else(|_| ".".to_string());
        
        let install_dir = PathBuf::from(&home).join(".r3mes").join("tools");
        let temp_dir = std::env::temp_dir().join("r3mes_installer");
        
        Self {
            temp_dir,
            install_dir,
        }
    }
    
    /// Install a component by name
    pub async fn install_component(&self, component: &str) -> Result<InstallationResult, String> {
        match component {
            "docker" => self.install_docker().await,
            "cuda" => self.install_cuda().await,
            "python" => self.install_python().await,
            "ipfs" => self.install_ipfs().await,
            "nodejs" => self.install_nodejs().await,
            "git" => self.install_git().await,
            _ => Err(format!("Unknown component: {}", component)),
        }
    }
    
    /// Get list of available components
    pub fn get_available_components(&self) -> Vec<ComponentInfo> {
        vec![
            ComponentInfo {
                name: "docker".to_string(),
                description: "Docker containerization platform".to_string(),
                required: true,
                installed: self.is_docker_installed(),
                version: self.get_docker_version(),
                installation_method: InstallationMethod::Download,
                dependencies: vec![],
            },
            ComponentInfo {
                name: "cuda".to_string(),
                description: "NVIDIA CUDA Toolkit for GPU computing".to_string(),
                required: false,
                installed: self.is_cuda_installed(),
                version: self.get_cuda_version(),
                installation_method: InstallationMethod::Download,
                dependencies: vec!["nvidia-driver".to_string()],
            },
            ComponentInfo {
                name: "python".to_string(),
                description: "Python programming language".to_string(),
                required: true,
                installed: self.is_python_installed(),
                version: self.get_python_version(),
                installation_method: InstallationMethod::PackageManager,
                dependencies: vec![],
            },
            ComponentInfo {
                name: "ipfs".to_string(),
                description: "InterPlanetary File System".to_string(),
                required: true,
                installed: self.is_ipfs_installed(),
                version: self.get_ipfs_version(),
                installation_method: InstallationMethod::Download,
                dependencies: vec![],
            },
            ComponentInfo {
                name: "nodejs".to_string(),
                description: "Node.js JavaScript runtime".to_string(),
                required: false,
                installed: self.is_nodejs_installed(),
                version: self.get_nodejs_version(),
                installation_method: InstallationMethod::PackageManager,
                dependencies: vec![],
            },
            ComponentInfo {
                name: "git".to_string(),
                description: "Git version control system".to_string(),
                required: false,
                installed: self.is_git_installed(),
                version: self.get_git_version(),
                installation_method: InstallationMethod::PackageManager,
                dependencies: vec![],
            },
        ]
    }
    
    /// Install Docker
    async fn install_docker(&self) -> Result<InstallationResult, String> {
        #[cfg(target_os = "windows")]
        {
            // Windows: Download Docker Desktop installer
            let download_url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe";
            let installer_path = self.temp_dir.join("DockerDesktopInstaller.exe");
            
            // Create temp directory
            fs::create_dir_all(&self.temp_dir)
                .map_err(|e| format!("Failed to create temp directory: {}", e))?;
            
            // Download installer
            println!("Downloading Docker Desktop installer...");
            self.download_file(download_url, &installer_path).await?;
            
            // Run installer
            println!("Running Docker Desktop installer...");
            let output = Command::new(&installer_path)
                .arg("install")
                .arg("--quiet")
                .output()
                .map_err(|e| format!("Failed to run Docker installer: {}", e))?;
            
            if output.status.success() {
                Ok(InstallationResult {
                    success: true,
                    component: "docker".to_string(),
                    message: "Docker Desktop installed successfully. Please restart your computer.".to_string(),
                    version: None,
                    installation_path: Some("C:\\Program Files\\Docker\\Docker".to_string()),
                    requires_restart: true,
                })
            } else {
                let error = String::from_utf8_lossy(&output.stderr);
                Err(format!("Docker installation failed: {}", error))
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS: Download Docker Desktop DMG
            let download_url = if cfg!(target_arch = "aarch64") {
                "https://desktop.docker.com/mac/main/arm64/Docker.dmg"
            } else {
                "https://desktop.docker.com/mac/main/amd64/Docker.dmg"
            };
            
            let dmg_path = self.temp_dir.join("Docker.dmg");
            
            fs::create_dir_all(&self.temp_dir)
                .map_err(|e| format!("Failed to create temp directory: {}", e))?;
            
            println!("Downloading Docker Desktop...");
            self.download_file(download_url, &dmg_path).await?;
            
            // Mount DMG and copy application
            let output = Command::new("hdiutil")
                .arg("attach")
                .arg(&dmg_path)
                .output()
                .map_err(|e| format!("Failed to mount DMG: {}", e))?;
            
            if output.status.success() {
                // Copy Docker.app to Applications
                let copy_output = Command::new("cp")
                    .arg("-R")
                    .arg("/Volumes/Docker/Docker.app")
                    .arg("/Applications/")
                    .output()
                    .map_err(|e| format!("Failed to copy Docker.app: {}", e))?;
                
                // Unmount DMG
                let _ = Command::new("hdiutil")
                    .arg("detach")
                    .arg("/Volumes/Docker")
                    .output();
                
                if copy_output.status.success() {
                    Ok(InstallationResult {
                        success: true,
                        component: "docker".to_string(),
                        message: "Docker Desktop installed successfully. Please start Docker from Applications.".to_string(),
                        version: None,
                        installation_path: Some("/Applications/Docker.app".to_string()),
                        requires_restart: false,
                    })
                } else {
                    Err("Failed to install Docker Desktop".to_string())
                }
            } else {
                Err("Failed to mount Docker DMG".to_string())
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            // Linux: Use package manager or install script
            if self.has_command("apt") {
                self.install_docker_ubuntu().await
            } else if self.has_command("yum") || self.has_command("dnf") {
                self.install_docker_rhel().await
            } else if self.has_command("pacman") {
                self.install_docker_arch().await
            } else {
                self.install_docker_generic().await
            }
        }
    }
    
    #[cfg(target_os = "linux")]
    async fn install_docker_ubuntu(&self) -> Result<InstallationResult, String> {
        // Update package index
        let _ = Command::new("sudo")
            .arg("apt")
            .arg("update")
            .output();
        
        // Install prerequisites
        let prereq_output = Command::new("sudo")
            .arg("apt")
            .arg("install")
            .arg("-y")
            .arg("ca-certificates")
            .arg("curl")
            .arg("gnupg")
            .arg("lsb-release")
            .output()
            .map_err(|e| format!("Failed to install prerequisites: {}", e))?;
        
        if !prereq_output.status.success() {
            return Err("Failed to install Docker prerequisites".to_string());
        }
        
        // Add Docker GPG key
        let key_output = Command::new("curl")
            .arg("-fsSL")
            .arg("https://download.docker.com/linux/ubuntu/gpg")
            .arg("|")
            .arg("sudo")
            .arg("gpg")
            .arg("--dearmor")
            .arg("-o")
            .arg("/usr/share/keyrings/docker-archive-keyring.gpg")
            .output();
        
        // Add Docker repository
        let repo_output = Command::new("sh")
            .arg("-c")
            .arg(r#"echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null"#)
            .output();
        
        // Update package index again
        let _ = Command::new("sudo")
            .arg("apt")
            .arg("update")
            .output();
        
        // Install Docker
        let install_output = Command::new("sudo")
            .arg("apt")
            .arg("install")
            .arg("-y")
            .arg("docker-ce")
            .arg("docker-ce-cli")
            .arg("containerd.io")
            .arg("docker-compose-plugin")
            .output()
            .map_err(|e| format!("Failed to install Docker: {}", e))?;
        
        if install_output.status.success() {
            // Add user to docker group
            let user = std::env::var("USER").unwrap_or_else(|_| "user".to_string());
            let _ = Command::new("sudo")
                .arg("usermod")
                .arg("-aG")
                .arg("docker")
                .arg(&user)
                .output();
            
            // Start Docker service
            let _ = Command::new("sudo")
                .arg("systemctl")
                .arg("start")
                .arg("docker")
                .output();
            
            let _ = Command::new("sudo")
                .arg("systemctl")
                .arg("enable")
                .arg("docker")
                .output();
            
            Ok(InstallationResult {
                success: true,
                component: "docker".to_string(),
                message: "Docker installed successfully. Please log out and log back in to use Docker without sudo.".to_string(),
                version: self.get_docker_version(),
                installation_path: Some("/usr/bin/docker".to_string()),
                requires_restart: false,
            })
        } else {
            let error = String::from_utf8_lossy(&install_output.stderr);
            Err(format!("Docker installation failed: {}", error))
        }
    }
    
    #[cfg(target_os = "linux")]
    async fn install_docker_rhel(&self) -> Result<InstallationResult, String> {
        let package_manager = if self.has_command("dnf") { "dnf" } else { "yum" };
        
        // Install prerequisites
        let prereq_output = Command::new("sudo")
            .arg(package_manager)
            .arg("install")
            .arg("-y")
            .arg("yum-utils")
            .output()
            .map_err(|e| format!("Failed to install prerequisites: {}", e))?;
        
        if !prereq_output.status.success() {
            return Err("Failed to install Docker prerequisites".to_string());
        }
        
        // Add Docker repository
        let repo_output = Command::new("sudo")
            .arg("yum-config-manager")
            .arg("--add-repo")
            .arg("https://download.docker.com/linux/centos/docker-ce.repo")
            .output();
        
        // Install Docker
        let install_output = Command::new("sudo")
            .arg(package_manager)
            .arg("install")
            .arg("-y")
            .arg("docker-ce")
            .arg("docker-ce-cli")
            .arg("containerd.io")
            .arg("docker-compose-plugin")
            .output()
            .map_err(|e| format!("Failed to install Docker: {}", e))?;
        
        if install_output.status.success() {
            // Start Docker service
            let _ = Command::new("sudo")
                .arg("systemctl")
                .arg("start")
                .arg("docker")
                .output();
            
            let _ = Command::new("sudo")
                .arg("systemctl")
                .arg("enable")
                .arg("docker")
                .output();
            
            Ok(InstallationResult {
                success: true,
                component: "docker".to_string(),
                message: "Docker installed successfully.".to_string(),
                version: self.get_docker_version(),
                installation_path: Some("/usr/bin/docker".to_string()),
                requires_restart: false,
            })
        } else {
            let error = String::from_utf8_lossy(&install_output.stderr);
            Err(format!("Docker installation failed: {}", error))
        }
    }
    
    #[cfg(target_os = "linux")]
    async fn install_docker_arch(&self) -> Result<InstallationResult, String> {
        // Install Docker using pacman
        let install_output = Command::new("sudo")
            .arg("pacman")
            .arg("-S")
            .arg("--noconfirm")
            .arg("docker")
            .arg("docker-compose")
            .output()
            .map_err(|e| format!("Failed to install Docker: {}", e))?;
        
        if install_output.status.success() {
            // Start Docker service
            let _ = Command::new("sudo")
                .arg("systemctl")
                .arg("start")
                .arg("docker")
                .output();
            
            let _ = Command::new("sudo")
                .arg("systemctl")
                .arg("enable")
                .arg("docker")
                .output();
            
            Ok(InstallationResult {
                success: true,
                component: "docker".to_string(),
                message: "Docker installed successfully.".to_string(),
                version: self.get_docker_version(),
                installation_path: Some("/usr/bin/docker".to_string()),
                requires_restart: false,
            })
        } else {
            let error = String::from_utf8_lossy(&install_output.stderr);
            Err(format!("Docker installation failed: {}", error))
        }
    }
    
    #[cfg(target_os = "linux")]
    async fn install_docker_generic(&self) -> Result<InstallationResult, String> {
        // Use Docker's convenience script
        let script_path = self.temp_dir.join("get-docker.sh");
        
        fs::create_dir_all(&self.temp_dir)
            .map_err(|e| format!("Failed to create temp directory: {}", e))?;
        
        // Download install script
        self.download_file("https://get.docker.com", &script_path).await?;
        
        // Make script executable
        let _ = Command::new("chmod")
            .arg("+x")
            .arg(&script_path)
            .output();
        
        // Run install script
        let install_output = Command::new("sudo")
            .arg("sh")
            .arg(&script_path)
            .output()
            .map_err(|e| format!("Failed to run Docker install script: {}", e))?;
        
        if install_output.status.success() {
            Ok(InstallationResult {
                success: true,
                component: "docker".to_string(),
                message: "Docker installed successfully using convenience script.".to_string(),
                version: self.get_docker_version(),
                installation_path: Some("/usr/bin/docker".to_string()),
                requires_restart: false,
            })
        } else {
            let error = String::from_utf8_lossy(&install_output.stderr);
            Err(format!("Docker installation failed: {}", error))
        }
    }
    
    /// Install CUDA Toolkit
    async fn install_cuda(&self) -> Result<InstallationResult, String> {
        #[cfg(target_os = "windows")]
        {
            // Windows: Download CUDA installer
            let download_url = "https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_546.12_windows.exe";
            let installer_path = self.temp_dir.join("cuda_installer.exe");
            
            fs::create_dir_all(&self.temp_dir)
                .map_err(|e| format!("Failed to create temp directory: {}", e))?;
            
            println!("Downloading CUDA Toolkit installer...");
            self.download_file(download_url, &installer_path).await?;
            
            println!("Running CUDA installer...");
            let output = Command::new(&installer_path)
                .arg("-s")
                .output()
                .map_err(|e| format!("Failed to run CUDA installer: {}", e))?;
            
            if output.status.success() {
                Ok(InstallationResult {
                    success: true,
                    component: "cuda".to_string(),
                    message: "CUDA Toolkit installed successfully.".to_string(),
                    version: Some("12.3".to_string()),
                    installation_path: Some("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.3".to_string()),
                    requires_restart: true,
                })
            } else {
                let error = String::from_utf8_lossy(&output.stderr);
                Err(format!("CUDA installation failed: {}", error))
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            // Linux: Use package manager or runfile
            if self.has_command("apt") {
                self.install_cuda_ubuntu().await
            } else {
                Err("CUDA installation not supported on this Linux distribution. Please install manually from https://developer.nvidia.com/cuda-downloads".to_string())
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            Err("CUDA is not supported on macOS. Use Metal Performance Shaders instead.".to_string())
        }
    }
    
    #[cfg(target_os = "linux")]
    async fn install_cuda_ubuntu(&self) -> Result<InstallationResult, String> {
        // Add NVIDIA package repository
        let keyring_output = Command::new("wget")
            .arg("https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb")
            .arg("-O")
            .arg("/tmp/cuda-keyring.deb")
            .output();
        
        if keyring_output.is_ok() {
            let _ = Command::new("sudo")
                .arg("dpkg")
                .arg("-i")
                .arg("/tmp/cuda-keyring.deb")
                .output();
        }
        
        // Update package list
        let _ = Command::new("sudo")
            .arg("apt")
            .arg("update")
            .output();
        
        // Install CUDA
        let install_output = Command::new("sudo")
            .arg("apt")
            .arg("install")
            .arg("-y")
            .arg("cuda")
            .output()
            .map_err(|e| format!("Failed to install CUDA: {}", e))?;
        
        if install_output.status.success() {
            Ok(InstallationResult {
                success: true,
                component: "cuda".to_string(),
                message: "CUDA Toolkit installed successfully. Please reboot and add CUDA to your PATH.".to_string(),
                version: self.get_cuda_version(),
                installation_path: Some("/usr/local/cuda".to_string()),
                requires_restart: true,
            })
        } else {
            let error = String::from_utf8_lossy(&install_output.stderr);
            Err(format!("CUDA installation failed: {}", error))
        }
    }
    
    /// Install Python
    async fn install_python(&self) -> Result<InstallationResult, String> {
        if self.is_python_installed() {
            return Ok(InstallationResult {
                success: true,
                component: "python".to_string(),
                message: "Python is already installed.".to_string(),
                version: self.get_python_version(),
                installation_path: None,
                requires_restart: false,
            });
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows: Download Python installer from python.org
            let download_url = "https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe";
            let installer_path = self.temp_dir.join("python-installer.exe");
            
            fs::create_dir_all(&self.temp_dir)
                .map_err(|e| format!("Failed to create temp directory: {}", e))?;
            
            println!("Downloading Python installer...");
            self.download_file(download_url, &installer_path).await?;
            
            println!("Installing Python...");
            let output = Command::new(&installer_path)
                .arg("/quiet")
                .arg("InstallAllUsers=1")
                .arg("PrependPath=1")
                .output()
                .map_err(|e| format!("Failed to run Python installer: {}", e))?;
            
            if output.status.success() {
                Ok(InstallationResult {
                    success: true,
                    component: "python".to_string(),
                    message: "Python installed successfully.".to_string(),
                    version: Some("3.11.6".to_string()),
                    installation_path: Some("C:\\Program Files\\Python311".to_string()),
                    requires_restart: false,
                })
            } else {
                let error = String::from_utf8_lossy(&output.stderr);
                Err(format!("Python installation failed: {}", error))
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS: Use Homebrew if available, otherwise download from python.org
            if self.has_command("brew") {
                let output = Command::new("brew")
                    .arg("install")
                    .arg("python@3.11")
                    .output()
                    .map_err(|e| format!("Failed to install Python via Homebrew: {}", e))?;
                
                if output.status.success() {
                    Ok(InstallationResult {
                        success: true,
                        component: "python".to_string(),
                        message: "Python installed successfully via Homebrew.".to_string(),
                        version: self.get_python_version(),
                        installation_path: Some("/opt/homebrew/bin/python3".to_string()),
                        requires_restart: false,
                    })
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    Err(format!("Python installation failed: {}", error))
                }
            } else {
                Err("Please install Homebrew first or download Python from https://python.org".to_string())
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            // Linux: Use package manager
            if self.has_command("apt") {
                let output = Command::new("sudo")
                    .arg("apt")
                    .arg("install")
                    .arg("-y")
                    .arg("python3")
                    .arg("python3-pip")
                    .arg("python3-venv")
                    .output()
                    .map_err(|e| format!("Failed to install Python: {}", e))?;
                
                if output.status.success() {
                    Ok(InstallationResult {
                        success: true,
                        component: "python".to_string(),
                        message: "Python installed successfully.".to_string(),
                        version: self.get_python_version(),
                        installation_path: Some("/usr/bin/python3".to_string()),
                        requires_restart: false,
                    })
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    Err(format!("Python installation failed: {}", error))
                }
            } else if self.has_command("yum") || self.has_command("dnf") {
                let pm = if self.has_command("dnf") { "dnf" } else { "yum" };
                let output = Command::new("sudo")
                    .arg(pm)
                    .arg("install")
                    .arg("-y")
                    .arg("python3")
                    .arg("python3-pip")
                    .output()
                    .map_err(|e| format!("Failed to install Python: {}", e))?;
                
                if output.status.success() {
                    Ok(InstallationResult {
                        success: true,
                        component: "python".to_string(),
                        message: "Python installed successfully.".to_string(),
                        version: self.get_python_version(),
                        installation_path: Some("/usr/bin/python3".to_string()),
                        requires_restart: false,
                    })
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    Err(format!("Python installation failed: {}", error))
                }
            } else {
                Err("Unsupported package manager. Please install Python manually.".to_string())
            }
        }
    }
    
    /// Install IPFS
    async fn install_ipfs(&self) -> Result<InstallationResult, String> {
        let download_url = self.get_ipfs_download_url();
        let archive_name = if cfg!(windows) { "go-ipfs.zip" } else { "go-ipfs.tar.gz" };
        let archive_path = self.temp_dir.join(archive_name);
        
        fs::create_dir_all(&self.temp_dir)
            .map_err(|e| format!("Failed to create temp directory: {}", e))?;
        
        fs::create_dir_all(&self.install_dir)
            .map_err(|e| format!("Failed to create install directory: {}", e))?;
        
        println!("Downloading IPFS...");
        self.download_file(&download_url, &archive_path).await?;
        
        println!("Extracting IPFS...");
        self.extract_archive(&archive_path, &self.temp_dir).await?;
        
        // Find the extracted binary
        let binary_name = if cfg!(windows) { "ipfs.exe" } else { "ipfs" };
        let extracted_binary = self.temp_dir.join("go-ipfs").join(binary_name);
        let install_binary = self.install_dir.join(binary_name);
        
        // Copy binary to install directory
        fs::copy(&extracted_binary, &install_binary)
            .map_err(|e| format!("Failed to copy IPFS binary: {}", e))?;
        
        // Make executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&install_binary)
                .map_err(|e| format!("Failed to get file permissions: {}", e))?
                .permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&install_binary, perms)
                .map_err(|e| format!("Failed to set executable permissions: {}", e))?;
        }
        
        Ok(InstallationResult {
            success: true,
            component: "ipfs".to_string(),
            message: format!("IPFS installed successfully to {}", install_binary.display()),
            version: self.get_ipfs_version(),
            installation_path: Some(install_binary.to_string_lossy().to_string()),
            requires_restart: false,
        })
    }
    
    /// Install Node.js
    async fn install_nodejs(&self) -> Result<InstallationResult, String> {
        #[cfg(target_os = "windows")]
        {
            let download_url = "https://nodejs.org/dist/v18.18.2/node-v18.18.2-x64.msi";
            let installer_path = self.temp_dir.join("nodejs-installer.msi");
            
            fs::create_dir_all(&self.temp_dir)
                .map_err(|e| format!("Failed to create temp directory: {}", e))?;
            
            println!("Downloading Node.js installer...");
            self.download_file(download_url, &installer_path).await?;
            
            println!("Installing Node.js...");
            let output = Command::new("msiexec")
                .arg("/i")
                .arg(&installer_path)
                .arg("/quiet")
                .output()
                .map_err(|e| format!("Failed to run Node.js installer: {}", e))?;
            
            if output.status.success() {
                Ok(InstallationResult {
                    success: true,
                    component: "nodejs".to_string(),
                    message: "Node.js installed successfully.".to_string(),
                    version: Some("18.18.2".to_string()),
                    installation_path: Some("C:\\Program Files\\nodejs".to_string()),
                    requires_restart: false,
                })
            } else {
                let error = String::from_utf8_lossy(&output.stderr);
                Err(format!("Node.js installation failed: {}", error))
            }
        }
        
        #[cfg(not(target_os = "windows"))]
        {
            // Use package manager on Unix systems
            if self.has_command("brew") {
                let output = Command::new("brew")
                    .arg("install")
                    .arg("node")
                    .output()
                    .map_err(|e| format!("Failed to install Node.js via Homebrew: {}", e))?;
                
                if output.status.success() {
                    Ok(InstallationResult {
                        success: true,
                        component: "nodejs".to_string(),
                        message: "Node.js installed successfully via Homebrew.".to_string(),
                        version: self.get_nodejs_version(),
                        installation_path: Some("/opt/homebrew/bin/node".to_string()),
                        requires_restart: false,
                    })
                } else {
                    let error = String::from_utf8_lossy(&output.stderr);
                    Err(format!("Node.js installation failed: {}", error))
                }
            } else if self.has_command("apt") {
                // Install Node.js via NodeSource repository
                let setup_output = Command::new("curl")
                    .arg("-fsSL")
                    .arg("https://deb.nodesource.com/setup_18.x")
                    .arg("|")
                    .arg("sudo")
                    .arg("-E")
                    .arg("bash")
                    .arg("-")
                    .output();
                
                let install_output = Command::new("sudo")
                    .arg("apt")
                    .arg("install")
                    .arg("-y")
                    .arg("nodejs")
                    .output()
                    .map_err(|e| format!("Failed to install Node.js: {}", e))?;
                
                if install_output.status.success() {
                    Ok(InstallationResult {
                        success: true,
                        component: "nodejs".to_string(),
                        message: "Node.js installed successfully.".to_string(),
                        version: self.get_nodejs_version(),
                        installation_path: Some("/usr/bin/node".to_string()),
                        requires_restart: false,
                    })
                } else {
                    let error = String::from_utf8_lossy(&install_output.stderr);
                    Err(format!("Node.js installation failed: {}", error))
                }
            } else {
                Err("Unsupported package manager. Please install Node.js manually from https://nodejs.org".to_string())
            }
        }
    }
    
    /// Install Git
    async fn install_git(&self) -> Result<InstallationResult, String> {
        if self.is_git_installed() {
            return Ok(InstallationResult {
                success: true,
                component: "git".to_string(),
                message: "Git is already installed.".to_string(),
                version: self.get_git_version(),
                installation_path: None,
                requires_restart: false,
            });
        }
        
        #[cfg(target_os = "windows")]
        {
            let download_url = "https://github.com/git-for-windows/git/releases/download/v2.42.0.windows.2/Git-2.42.0.2-64-bit.exe";
            let installer_path = self.temp_dir.join("git-installer.exe");
            
            fs::create_dir_all(&self.temp_dir)
                .map_err(|e| format!("Failed to create temp directory: {}", e))?;
            
            println!("Downloading Git installer...");
            self.download_file(download_url, &installer_path).await?;
            
            println!("Installing Git...");
            let output = Command::new(&installer_path)
                .arg("/VERYSILENT")
                .arg("/NORESTART")
                .output()
                .map_err(|e| format!("Failed to run Git installer: {}", e))?;
            
            if output.status.success() {
                Ok(InstallationResult {
                    success: true,
                    component: "git".to_string(),
                    message: "Git installed successfully.".to_string(),
                    version: Some("2.42.0".to_string()),
                    installation_path: Some("C:\\Program Files\\Git".to_string()),
                    requires_restart: false,
                })
            } else {
                let error = String::from_utf8_lossy(&output.stderr);
                Err(format!("Git installation failed: {}", error))
            }
        }
        
        #[cfg(not(target_os = "windows"))]
        {
            // Use package manager
            let (command, args) = if self.has_command("apt") {
                ("sudo", vec!["apt", "install", "-y", "git"])
            } else if self.has_command("yum") {
                ("sudo", vec!["yum", "install", "-y", "git"])
            } else if self.has_command("dnf") {
                ("sudo", vec!["dnf", "install", "-y", "git"])
            } else if self.has_command("brew") {
                ("brew", vec!["install", "git"])
            } else {
                return Err("Unsupported package manager. Please install Git manually.".to_string());
            };
            
            let output = Command::new(command)
                .args(&args)
                .output()
                .map_err(|e| format!("Failed to install Git: {}", e))?;
            
            if output.status.success() {
                Ok(InstallationResult {
                    success: true,
                    component: "git".to_string(),
                    message: "Git installed successfully.".to_string(),
                    version: self.get_git_version(),
                    installation_path: Some("/usr/bin/git".to_string()),
                    requires_restart: false,
                })
            } else {
                let error = String::from_utf8_lossy(&output.stderr);
                Err(format!("Git installation failed: {}", error))
            }
        }
    }
    
    /// Download a file from URL
    async fn download_file(&self, url: &str, path: &PathBuf) -> Result<(), String> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap();
        
        let response = client.get(url)
            .send()
            .await
            .map_err(|e| format!("Failed to download file: {}", e))?;
        
        if !response.status().is_success() {
            return Err(format!("Download failed with status: {}", response.status()));
        }
        
        let bytes = response.bytes()
            .await
            .map_err(|e| format!("Failed to read response bytes: {}", e))?;
        
        fs::write(path, bytes)
            .map_err(|e| format!("Failed to write file: {}", e))?;
        
        Ok(())
    }
    
    /// Extract archive
    async fn extract_archive(&self, archive_path: &PathBuf, extract_to: &PathBuf) -> Result<(), String> {
        let file_name = archive_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        
        if file_name.ends_with(".zip") {
            self.extract_zip(archive_path, extract_to).await
        } else if file_name.ends_with(".tar.gz") {
            self.extract_tar_gz(archive_path, extract_to).await
        } else {
            Err(format!("Unsupported archive format: {}", file_name))
        }
    }
    
    async fn extract_zip(&self, archive_path: &PathBuf, extract_to: &PathBuf) -> Result<(), String> {
        use zip::ZipArchive;
        
        
        let file = fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open ZIP file: {}", e))?;
        
        let mut archive = ZipArchive::new(file)
            .map_err(|e| format!("Failed to read ZIP archive: {}", e))?;
        
        for i in 0..archive.len() {
            let mut file = archive.by_index(i)
                .map_err(|e| format!("Failed to read ZIP entry: {}", e))?;
            
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
    
    async fn extract_tar_gz(&self, archive_path: &PathBuf, extract_to: &PathBuf) -> Result<(), String> {
        use flate2::read::GzDecoder;
        use tar::Archive;
        
        let file = fs::File::open(archive_path)
            .map_err(|e| format!("Failed to open TAR.GZ file: {}", e))?;
        
        let decoder = GzDecoder::new(file);
        let mut archive = Archive::new(decoder);
        
        archive.unpack(extract_to)
            .map_err(|e| format!("Failed to extract TAR.GZ archive: {}", e))?;
        
        Ok(())
    }
    
    /// Get IPFS download URL for current platform
    fn get_ipfs_download_url(&self) -> String {
        let version = "v0.23.0";
        let base_url = format!("https://github.com/ipfs/go-ipfs/releases/download/{}", version);
        
        #[cfg(target_os = "windows")]
        {
            format!("{}/go-ipfs_{}_windows-amd64.zip", base_url, version)
        }
        
        #[cfg(target_os = "macos")]
        {
            if cfg!(target_arch = "aarch64") {
                format!("{}/go-ipfs_{}_darwin-arm64.tar.gz", base_url, version)
            } else {
                format!("{}/go-ipfs_{}_darwin-amd64.tar.gz", base_url, version)
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            if cfg!(target_arch = "aarch64") {
                format!("{}/go-ipfs_{}_linux-arm64.tar.gz", base_url, version)
            } else {
                format!("{}/go-ipfs_{}_linux-amd64.tar.gz", base_url, version)
            }
        }
        
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            format!("{}/go-ipfs_{}_linux-amd64.tar.gz", base_url, version)
        }
    }
    
    /// Check if a command exists
    fn has_command(&self, command: &str) -> bool {
        Command::new("which")
            .arg(command)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false) ||
        Command::new("where")
            .arg(command)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }
    
    // Component detection methods
    fn is_docker_installed(&self) -> bool {
        self.has_command("docker")
    }
    
    fn is_cuda_installed(&self) -> bool {
        self.has_command("nvcc") || self.has_command("nvidia-smi")
    }
    
    fn is_python_installed(&self) -> bool {
        self.has_command("python3") || self.has_command("python")
    }
    
    fn is_ipfs_installed(&self) -> bool {
        self.has_command("ipfs")
    }
    
    fn is_nodejs_installed(&self) -> bool {
        self.has_command("node")
    }
    
    fn is_git_installed(&self) -> bool {
        self.has_command("git")
    }
    
    // Version detection methods
    fn get_docker_version(&self) -> Option<String> {
        Command::new("docker")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                let version_str = String::from_utf8_lossy(&output.stdout);
                version_str.split_whitespace()
                    .nth(2)
                    .map(|v| v.trim_end_matches(',').to_string())
            })
    }
    
    fn get_cuda_version(&self) -> Option<String> {
        Command::new("nvcc")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                let version_str = String::from_utf8_lossy(&output.stdout);
                version_str.lines()
                    .find(|line| line.contains("release"))
                    .and_then(|line| {
                        line.split("release ")
                            .nth(1)
                            .and_then(|s| s.split(',').next())
                            .map(|v| v.trim().to_string())
                    })
            })
    }
    
    fn get_python_version(&self) -> Option<String> {
        let commands = vec!["python3", "python"];
        
        for cmd in commands {
            if let Ok(output) = Command::new(cmd).arg("--version").output() {
                let version_str = String::from_utf8_lossy(&output.stdout);
                if let Some(version) = version_str.split_whitespace().nth(1) {
                    return Some(version.to_string());
                }
            }
        }
        
        None
    }
    
    fn get_ipfs_version(&self) -> Option<String> {
        Command::new("ipfs")
            .arg("version")
            .output()
            .ok()
            .and_then(|output| {
                let version_str = String::from_utf8_lossy(&output.stdout);
                version_str.split_whitespace()
                    .nth(2)
                    .map(|v| v.to_string())
            })
    }
    
    fn get_nodejs_version(&self) -> Option<String> {
        Command::new("node")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                let version_str = String::from_utf8_lossy(&output.stdout);
                Some(version_str.trim().trim_start_matches('v').to_string())
            })
    }
    
    fn get_git_version(&self) -> Option<String> {
        Command::new("git")
            .arg("--version")
            .output()
            .ok()
            .and_then(|output| {
                let version_str = String::from_utf8_lossy(&output.stdout);
                version_str.split_whitespace()
                    .nth(2)
                    .map(|v| v.to_string())
            })
    }
}