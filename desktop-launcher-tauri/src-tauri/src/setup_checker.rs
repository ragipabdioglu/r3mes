use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use log::{info, warn, error};
use crate::hardware_check::{check_system_requirements, SystemRequirements};
use crate::config::Config;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SetupStatus {
    pub is_first_run: bool,
    pub docker_installed: bool,
    pub cuda_available: bool,
    pub python_installed: bool,
    pub git_installed: bool,
    pub engine_downloaded: bool,
    pub models_downloaded: bool,
    pub wallet_configured: bool,
    pub firewall_configured: bool,
    pub setup_complete: bool,
    pub missing_components: Vec<String>,
    pub warnings: Vec<String>,
}

impl Default for SetupStatus {
    fn default() -> Self {
        Self {
            is_first_run: true,
            docker_installed: false,
            cuda_available: false,
            python_installed: false,
            git_installed: false,
            engine_downloaded: false,
            models_downloaded: false,
            wallet_configured: false,
            firewall_configured: false,
            setup_complete: false,
            missing_components: Vec::new(),
            warnings: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SetupStep {
    pub id: String,
    pub name: String,
    pub description: String,
    pub required: bool,
    pub completed: bool,
    pub error: Option<String>,
}

pub struct SetupChecker;

impl SetupChecker {
    pub fn new() -> Self {
        Self
    }

    /// Comprehensive setup validation
    pub async fn check_complete_setup() -> Result<SetupStatus, String> {
        info!("Starting comprehensive setup check...");
        
        let mut status = SetupStatus::default();
        
        // Check if this is first run
        status.is_first_run = Self::is_first_run().await?;
        
        // Check system requirements
        let sys_req = check_system_requirements().await
            .map_err(|e| format!("Failed to check system requirements: {}", e))?;
        
        status.docker_installed = sys_req.docker_available;
        status.cuda_available = sys_req.cuda_available;
        
        // Check development tools
        status.python_installed = Self::check_python_installation().await;
        status.git_installed = Self::check_git_installation().await;
        
        // Check R3MES components
        status.engine_downloaded = Self::check_engine_installation().await;
        status.models_downloaded = Self::check_models_installation().await;
        
        // Check configuration
        status.wallet_configured = Self::check_wallet_configuration().await;
        status.firewall_configured = Self::check_firewall_configuration().await;
        
        // Determine missing components
        status.missing_components = Self::get_missing_components(&status);
        status.warnings = Self::get_setup_warnings(&status);
        
        // Overall setup completion status
        status.setup_complete = status.missing_components.is_empty() && 
                               status.docker_installed && 
                               status.python_installed &&
                               status.engine_downloaded;
        
        info!("Setup check completed. Status: {}", if status.setup_complete { "Complete" } else { "Incomplete" });
        Ok(status)
    }

    /// Check if this is the first run
    pub async fn is_first_run() -> Result<bool, String> {
        let config_dir = dirs::config_dir()
            .ok_or("Could not find config directory")?
            .join("r3mes-launcher");
        
        let setup_marker = config_dir.join(".setup_complete");
        Ok(!setup_marker.exists())
    }

    /// Mark setup as complete
    pub async fn mark_setup_complete() -> Result<(), String> {
        let config_dir = dirs::config_dir()
            .ok_or("Could not find config directory")?
            .join("r3mes-launcher");
        
        tokio::fs::create_dir_all(&config_dir).await
            .map_err(|e| format!("Failed to create config directory: {}", e))?;
        
        let setup_marker = config_dir.join(".setup_complete");
        tokio::fs::write(&setup_marker, "setup_complete").await
            .map_err(|e| format!("Failed to create setup marker: {}", e))?;
        
        info!("Setup marked as complete");
        Ok(())
    }

    /// Check Python installation
    async fn check_python_installation() -> bool {
        match Command::new("python").arg("--version").output() {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    info!("Python found: {}", version.trim());
                    true
                } else {
                    // Try python3
                    match Command::new("python3").arg("--version").output() {
                        Ok(output) => {
                            if output.status.success() {
                                let version = String::from_utf8_lossy(&output.stdout);
                                info!("Python3 found: {}", version.trim());
                                true
                            } else {
                                false
                            }
                        }
                        Err(_) => false,
                    }
                }
            }
            Err(_) => false,
        }
    }

    /// Check Git installation
    async fn check_git_installation() -> bool {
        match Command::new("git").arg("--version").output() {
            Ok(output) => {
                if output.status.success() {
                    let version = String::from_utf8_lossy(&output.stdout);
                    info!("Git found: {}", version.trim());
                    true
                } else {
                    false
                }
            }
            Err(_) => false,
        }
    }

    /// Check if R3MES engine is installed
    async fn check_engine_installation() -> bool {
        let engine_paths = vec![
            "miner-engine",
            "../miner-engine",
            "../../miner-engine",
            dirs::home_dir().map(|p| p.join("r3mes").join("miner-engine")),
        ];

        for path_opt in engine_paths {
            if let Some(path) = path_opt {
                if path.exists() && path.join("main.py").exists() {
                    info!("R3MES engine found at: {:?}", path);
                    return true;
                }
            }
        }

        warn!("R3MES engine not found");
        false
    }

    /// Check if AI models are downloaded
    async fn check_models_installation() -> bool {
        let model_paths = vec![
            "models",
            "../models", 
            "../../models",
            dirs::home_dir().map(|p| p.join("r3mes").join("models")),
        ];

        for path_opt in model_paths {
            if let Some(path) = path_opt {
                if path.exists() {
                    // Check if there are any model files
                    if let Ok(entries) = std::fs::read_dir(&path) {
                        let model_count = entries.count();
                        if model_count > 0 {
                            info!("Models found at: {:?} ({} files)", path, model_count);
                            return true;
                        }
                    }
                }
            }
        }

        warn!("AI models not found");
        false
    }

    /// Check wallet configuration
    async fn check_wallet_configuration() -> bool {
        match Config::load().await {
            Ok(config) => {
                !config.wallet.private_key.is_empty() || !config.wallet.mnemonic.is_empty()
            }
            Err(_) => false,
        }
    }

    /// Check firewall configuration
    async fn check_firewall_configuration() -> bool {
        // Check if required ports are accessible
        let required_ports = vec![8080, 4001, 5001, 9090];
        
        for port in required_ports {
            if !Self::is_port_available(port).await {
                warn!("Port {} may be blocked by firewall", port);
                return false;
            }
        }
        
        true
    }

    /// Check if a port is available
    async fn is_port_available(port: u16) -> bool {
        use std::net::{TcpListener, SocketAddr};
        
        let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
        TcpListener::bind(addr).is_ok()
    }

    /// Get list of missing components
    fn get_missing_components(status: &SetupStatus) -> Vec<String> {
        let mut missing = Vec::new();
        
        if !status.docker_installed {
            missing.push("Docker Desktop".to_string());
        }
        
        if !status.python_installed {
            missing.push("Python 3.8+".to_string());
        }
        
        if !status.git_installed {
            missing.push("Git".to_string());
        }
        
        if !status.engine_downloaded {
            missing.push("R3MES Mining Engine".to_string());
        }
        
        if !status.models_downloaded {
            missing.push("AI Models".to_string());
        }
        
        if !status.wallet_configured {
            missing.push("Wallet Configuration".to_string());
        }
        
        missing
    }

    /// Get setup warnings
    fn get_setup_warnings(status: &SetupStatus) -> Vec<String> {
        let mut warnings = Vec::new();
        
        if !status.cuda_available {
            warnings.push("CUDA not available - GPU mining will be disabled".to_string());
        }
        
        if !status.firewall_configured {
            warnings.push("Firewall may block required ports".to_string());
        }
        
        warnings
    }

    /// Get setup steps with their completion status
    pub async fn get_setup_steps() -> Result<Vec<SetupStep>, String> {
        let status = Self::check_complete_setup().await?;
        
        let steps = vec![
            SetupStep {
                id: "docker".to_string(),
                name: "Docker Installation".to_string(),
                description: "Install Docker Desktop for containerization".to_string(),
                required: true,
                completed: status.docker_installed,
                error: if !status.docker_installed { 
                    Some("Docker Desktop not found".to_string()) 
                } else { 
                    None 
                },
            },
            SetupStep {
                id: "python".to_string(),
                name: "Python Installation".to_string(),
                description: "Install Python 3.8 or higher".to_string(),
                required: true,
                completed: status.python_installed,
                error: if !status.python_installed { 
                    Some("Python not found in PATH".to_string()) 
                } else { 
                    None 
                },
            },
            SetupStep {
                id: "git".to_string(),
                name: "Git Installation".to_string(),
                description: "Install Git for version control".to_string(),
                required: true,
                completed: status.git_installed,
                error: if !status.git_installed { 
                    Some("Git not found in PATH".to_string()) 
                } else { 
                    None 
                },
            },
            SetupStep {
                id: "cuda".to_string(),
                name: "CUDA Support".to_string(),
                description: "Install NVIDIA CUDA for GPU acceleration".to_string(),
                required: false,
                completed: status.cuda_available,
                error: if !status.cuda_available { 
                    Some("CUDA toolkit not found".to_string()) 
                } else { 
                    None 
                },
            },
            SetupStep {
                id: "engine".to_string(),
                name: "Mining Engine".to_string(),
                description: "Download and setup R3MES mining engine".to_string(),
                required: true,
                completed: status.engine_downloaded,
                error: if !status.engine_downloaded { 
                    Some("Mining engine not found".to_string()) 
                } else { 
                    None 
                },
            },
            SetupStep {
                id: "models".to_string(),
                name: "AI Models".to_string(),
                description: "Download required AI models".to_string(),
                required: true,
                completed: status.models_downloaded,
                error: if !status.models_downloaded { 
                    Some("AI models not downloaded".to_string()) 
                } else { 
                    None 
                },
            },
            SetupStep {
                id: "wallet".to_string(),
                name: "Wallet Configuration".to_string(),
                description: "Create or import wallet for mining rewards".to_string(),
                required: true,
                completed: status.wallet_configured,
                error: if !status.wallet_configured { 
                    Some("Wallet not configured".to_string()) 
                } else { 
                    None 
                },
            },
            SetupStep {
                id: "firewall".to_string(),
                name: "Firewall Configuration".to_string(),
                description: "Configure firewall for network access".to_string(),
                required: false,
                completed: status.firewall_configured,
                error: if !status.firewall_configured { 
                    Some("Firewall may block required ports".to_string()) 
                } else { 
                    None 
                },
            },
        ];
        
        Ok(steps)
    }

    /// Validate specific component
    pub async fn validate_component(component: &str) -> Result<bool, String> {
        match component {
            "docker" => Ok(crate::hardware_check::check_docker_availability().await.unwrap_or(false)),
            "python" => Ok(Self::check_python_installation().await),
            "git" => Ok(Self::check_git_installation().await),
            "cuda" => Ok(crate::hardware_check::check_cuda_availability().await.unwrap_or(false)),
            "engine" => Ok(Self::check_engine_installation().await),
            "models" => Ok(Self::check_models_installation().await),
            "wallet" => Ok(Self::check_wallet_configuration().await),
            "firewall" => Ok(Self::check_firewall_configuration().await),
            _ => Err(format!("Unknown component: {}", component)),
        }
    }

    /// Get setup progress percentage
    pub async fn get_setup_progress() -> Result<f32, String> {
        let steps = Self::get_setup_steps().await?;
        let total_required = steps.iter().filter(|s| s.required).count() as f32;
        let completed_required = steps.iter().filter(|s| s.required && s.completed).count() as f32;
        
        if total_required == 0.0 {
            Ok(100.0)
        } else {
            Ok((completed_required / total_required) * 100.0)
        }
    }
}

// Tauri command exports
#[tauri::command]
pub async fn is_first_run() -> Result<bool, String> {
    SetupChecker::is_first_run().await
}

#[tauri::command]
pub async fn mark_setup_complete() -> Result<(), String> {
    SetupChecker::mark_setup_complete().await
}

#[tauri::command]
pub async fn check_setup_status() -> Result<SetupStatus, String> {
    SetupChecker::check_complete_setup().await
}

#[tauri::command]
pub async fn get_setup_steps() -> Result<Vec<SetupStep>, String> {
    SetupChecker::get_setup_steps().await
}

#[tauri::command]
pub async fn validate_component(component: String) -> Result<bool, String> {
    SetupChecker::validate_component(&component).await
}

#[tauri::command]
pub async fn get_setup_progress() -> Result<f32, String> {
    SetupChecker::get_setup_progress().await
}