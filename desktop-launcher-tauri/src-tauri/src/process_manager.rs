//! Process management for R3MES Desktop Launcher

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Mutex;
use once_cell::sync::Lazy;
use crate::platform::silent_command;

#[cfg(windows)]
use crate::platform::hide_console_window;

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessResult {
    pub success: bool,
    pub message: String,
    pub pid: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessStatus {
    pub node: ProcessInfo,
    pub miner: ProcessInfo,
    pub ipfs: ProcessInfo,
    pub serving: ProcessInfo,
    pub validator: ProcessInfo,
    pub proposer: ProcessInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    pub running: bool,
    pub pid: Option<u32>,
}

/// Process type enum for generic process management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessType {
    Node,
    Miner,
    Ipfs,
    Serving,
    Validator,
    Proposer,
}

impl ProcessType {
    /// Get the display name for this process type
    pub fn name(&self) -> &'static str {
        match self {
            ProcessType::Node => "Node",
            ProcessType::Miner => "Miner",
            ProcessType::Ipfs => "IPFS",
            ProcessType::Serving => "Serving",
            ProcessType::Validator => "Validator",
            ProcessType::Proposer => "Proposer",
        }
    }
    
    /// Get the log file name for this process type
    pub fn log_file(&self) -> &'static str {
        match self {
            ProcessType::Node => "node.log",
            ProcessType::Miner => "miner.log",
            ProcessType::Ipfs => "ipfs.log",
            ProcessType::Serving => "serving.log",
            ProcessType::Validator => "validator.log",
            ProcessType::Proposer => "proposer.log",
        }
    }
    
    /// Get the process name for force kill (Unix pkill pattern)
    #[cfg(unix)]
    pub fn kill_pattern(&self) -> &'static str {
        match self {
            ProcessType::Node => "remesd",
            ProcessType::Miner => "r3mes-miner",
            ProcessType::Ipfs => "ipfs",
            ProcessType::Serving => "r3mes-serving",
            ProcessType::Validator => "remesd.*validator",
            ProcessType::Proposer => "r3mes-proposer",
        }
    }
    
    /// Get the process name for force kill (Windows taskkill)
    #[cfg(windows)]
    pub fn exe_name(&self) -> &'static str {
        match self {
            ProcessType::Node => "remesd.exe",
            ProcessType::Miner => "r3mes-miner.exe",
            ProcessType::Ipfs => "ipfs.exe",
            ProcessType::Serving => "r3mes-serving.exe",
            ProcessType::Validator => "remesd.exe",
            ProcessType::Proposer => "r3mes-proposer.exe",
        }
    }
}

/// Persisted process state - survives application restarts
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PersistedProcessState {
    pub node_pid: Option<u32>,
    pub miner_pid: Option<u32>,
    pub ipfs_pid: Option<u32>,
    pub serving_pid: Option<u32>,
    pub validator_pid: Option<u32>,
    pub proposer_pid: Option<u32>,
    pub last_updated: Option<String>,
}

impl PersistedProcessState {
    /// Get the persistence file path
    fn file_path() -> PathBuf {
        crate::platform::get_data_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join("process_state.json")
    }
    
    /// Load persisted state from file
    pub fn load() -> Self {
        let path = Self::file_path();
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(state) = serde_json::from_str(&content) {
                    return state;
                }
            }
        }
        Self::default()
    }
    
    /// Save state to file
    pub fn save(&self) -> Result<(), String> {
        let path = Self::file_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }
        
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize state: {}", e))?;
        
        std::fs::write(&path, json)
            .map_err(|e| format!("Failed to write state file: {}", e))?;
        
        Ok(())
    }
    
    /// Check if a PID is still running
    #[cfg(unix)]
    fn is_pid_running(pid: u32) -> bool {
        silent_command("kill")
            .arg("-0")
            .arg(pid.to_string())
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
    
    #[cfg(windows)]
    fn is_pid_running(pid: u32) -> bool {
        silent_command("tasklist")
            .arg("/FI")
            .arg(format!("PID eq {}", pid))
            .output()
            .map(|o| {
                let output = String::from_utf8_lossy(&o.stdout);
                output.contains(&pid.to_string())
            })
            .unwrap_or(false)
    }
    
    /// Validate and clean up stale PIDs
    pub fn validate(&mut self) -> bool {
        let mut changed = false;
        
        if let Some(pid) = self.node_pid {
            if !Self::is_pid_running(pid) {
                self.node_pid = None;
                changed = true;
            }
        }
        if let Some(pid) = self.miner_pid {
            if !Self::is_pid_running(pid) {
                self.miner_pid = None;
                changed = true;
            }
        }
        if let Some(pid) = self.ipfs_pid {
            if !Self::is_pid_running(pid) {
                self.ipfs_pid = None;
                changed = true;
            }
        }
        if let Some(pid) = self.serving_pid {
            if !Self::is_pid_running(pid) {
                self.serving_pid = None;
                changed = true;
            }
        }
        if let Some(pid) = self.validator_pid {
            if !Self::is_pid_running(pid) {
                self.validator_pid = None;
                changed = true;
            }
        }
        if let Some(pid) = self.proposer_pid {
            if !Self::is_pid_running(pid) {
                self.proposer_pid = None;
                changed = true;
            }
        }
        
        changed
    }
}

/// Global process state for persistence
static PERSISTED_STATE: Lazy<Arc<Mutex<PersistedProcessState>>> = 
    Lazy::new(|| Arc::new(Mutex::new(PersistedProcessState::load())));

/// Restore process state on startup
pub async fn restore_process_state() -> Result<(), String> {
    let mut state = PERSISTED_STATE.lock().await;
    
    // Validate PIDs - remove any that are no longer running
    if state.validate() {
        state.save()?;
    }
    
    log::info!("Restored process state: node={:?}, miner={:?}, ipfs={:?}", 
        state.node_pid, state.miner_pid, state.ipfs_pid);
    
    Ok(())
}

/// Save current process state
async fn save_process_state(
    node_pid: Option<u32>,
    miner_pid: Option<u32>,
    ipfs_pid: Option<u32>,
    serving_pid: Option<u32>,
    validator_pid: Option<u32>,
    proposer_pid: Option<u32>,
) -> Result<(), String> {
    let mut state = PERSISTED_STATE.lock().await;
    state.node_pid = node_pid;
    state.miner_pid = miner_pid;
    state.ipfs_pid = ipfs_pid;
    state.serving_pid = serving_pid;
    state.validator_pid = validator_pid;
    state.proposer_pid = proposer_pid;
    state.last_updated = Some(chrono::Utc::now().to_rfc3339());
    state.save()
}

pub struct ProcessManager {
    workspace: PathBuf,
    node_pid: Arc<Mutex<Option<u32>>>,
    miner_pid: Arc<Mutex<Option<u32>>>,
    ipfs_pid: Arc<Mutex<Option<u32>>>,
    serving_pid: Arc<Mutex<Option<u32>>>,
    validator_pid: Arc<Mutex<Option<u32>>>,
    proposer_pid: Arc<Mutex<Option<u32>>>,
}

impl ProcessManager {
    pub fn new() -> Self {
        // Use platform module for cross-platform workspace directory
        let workspace = crate::platform::get_workspace_dir()
            .unwrap_or_else(|_| {
                // Fallback to home directory
                #[cfg(windows)]
                {
                    let home = std::env::var("USERPROFILE")
                        .unwrap_or_else(|_| "C:\\Users\\Default".to_string());
                    PathBuf::from(home).join("R3MES")
                }
                #[cfg(not(windows))]
                {
                    let home = std::env::var("HOME")
                        .unwrap_or_else(|_| "/tmp".to_string());
                    PathBuf::from(home).join("R3MES")
                }
            });
        
        // Load persisted state synchronously (blocking is OK during init)
        let persisted = PersistedProcessState::load();
        
        Self {
            workspace,
            node_pid: Arc::new(Mutex::new(persisted.node_pid)),
            miner_pid: Arc::new(Mutex::new(persisted.miner_pid)),
            ipfs_pid: Arc::new(Mutex::new(persisted.ipfs_pid)),
            serving_pid: Arc::new(Mutex::new(persisted.serving_pid)),
            validator_pid: Arc::new(Mutex::new(persisted.validator_pid)),
            proposer_pid: Arc::new(Mutex::new(persisted.proposer_pid)),
        }
    }
    
    /// Save current state to persistence file
    async fn persist_state(&self) {
        let node = *self.node_pid.lock().await;
        let miner = *self.miner_pid.lock().await;
        let ipfs = *self.ipfs_pid.lock().await;
        let serving = *self.serving_pid.lock().await;
        let validator = *self.validator_pid.lock().await;
        let proposer = *self.proposer_pid.lock().await;
        
        if let Err(e) = save_process_state(node, miner, ipfs, serving, validator, proposer).await {
            log::warn!("Failed to persist process state: {}", e);
        }
    }
    
    /// Get the PID mutex for a given process type
    fn get_pid_mutex(&self, process_type: ProcessType) -> &Arc<Mutex<Option<u32>>> {
        match process_type {
            ProcessType::Node => &self.node_pid,
            ProcessType::Miner => &self.miner_pid,
            ProcessType::Ipfs => &self.ipfs_pid,
            ProcessType::Serving => &self.serving_pid,
            ProcessType::Validator => &self.validator_pid,
            ProcessType::Proposer => &self.proposer_pid,
        }
    }
    
    /// Generic process stop helper - reduces code duplication
    async fn stop_process_generic(&self, process_type: ProcessType) -> Result<ProcessResult, String> {
        let pid_mutex = self.get_pid_mutex(process_type);
        let mut pid_guard = pid_mutex.lock().await;
        let name = process_type.name();
        
        if let Some(pid) = pid_guard.take() {
            drop(pid_guard); // Release lock before async operations
            
            #[cfg(unix)]
            {
                let _ = silent_command("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .output();
                
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                let _ = silent_command("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
            
            #[cfg(windows)]
            {
                let _ = silent_command("taskkill")
                    .arg("/T")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .output();
            }
            
            // Persist state after stop
            self.persist_state().await;

            Ok(ProcessResult {
                success: true,
                message: format!("{} stopped", name),
                pid: None,
            })
        } else {
            // Force kill any remaining processes
            #[cfg(unix)]
            {
                let _ = silent_command("pkill")
                    .arg("-9")
                    .arg("-f")
                    .arg(process_type.kill_pattern())
                    .output();
            }
            #[cfg(windows)]
            {
                let _ = silent_command("taskkill")
                    .arg("/F")
                    .arg("/IM")
                    .arg(process_type.exe_name())
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: format!("{} stopped (force)", name),
                pid: None,
            })
        }
    }
    
    /// Generic process spawn helper - reduces code duplication
    async fn spawn_and_track(
        &self,
        process_type: ProcessType,
        cmd: &mut Command,
        log_file: PathBuf,
    ) -> Result<u32, String> {
        #[cfg(unix)]
        let pid = {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child); // Detach process
            pid
        };
        
        #[cfg(windows)]
        let pid = {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            child.id()
        };
        
        // Update PID in the appropriate mutex
        let pid_mutex = self.get_pid_mutex(process_type);
        *pid_mutex.lock().await = Some(pid);
        
        // Persist state after successful start
        self.persist_state().await;
        
        Ok(pid)
    }
    
    /// Get the log file path for a process type
    fn get_log_file(&self, process_type: ProcessType) -> Result<PathBuf, String> {
        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        Ok(log_dir.join(process_type.log_file()))
    }

    pub async fn start_node(&mut self) -> Result<ProcessResult, String> {
        let mut node_pid = self.node_pid.lock().await;
        if node_pid.is_some() {
            return Ok(ProcessResult {
                success: false,
                message: "Node is already running".to_string(),
                pid: None,
            });
        }

        let remesd_path = self.workspace.join("remes").join("build").join("remesd");
        if !remesd_path.exists() {
            return Ok(ProcessResult {
                success: false,
                message: format!("remesd binary not found at {:?}", remesd_path),
                pid: None,
            });
        }

        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        let log_file = log_dir.join("node.log");

        // Start process in detached mode
        let mut cmd = Command::new(&remesd_path);
        #[cfg(windows)]
        hide_console_window(&mut cmd);
        
        cmd.arg("start")
            .current_dir(self.workspace.join("remes"));

        #[cfg(unix)]
        let result = {
            use std::os::unix::process::CommandExt;
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            // Spawn detached process
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            
            // Detach process
            std::mem::forget(child);
            
            *node_pid = Some(pid);
            drop(node_pid); // Release lock before persist
            
            pid
        };
        
        #[cfg(windows)]
        let result = {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *node_pid = Some(pid);
            drop(node_pid); // Release lock before persist
            
            pid
        };
        
        // Persist state after successful start
        self.persist_state().await;
        
        Ok(ProcessResult {
            success: true,
            message: "Node started".to_string(),
            pid: Some(result),
        })
    }

    pub async fn stop_node(&mut self) -> Result<ProcessResult, String> {
        self.stop_process_generic(ProcessType::Node).await
    }

    pub async fn start_miner(&mut self) -> Result<ProcessResult, String> {
        use crate::engine_downloader::EngineDownloader;
        
        let mut miner_pid = self.miner_pid.lock().await;
        if miner_pid.is_some() {
            return Ok(ProcessResult {
                success: false,
                message: "Miner is already running".to_string(),
                pid: None,
            });
        }

        // On Windows, check if engine.exe exists, otherwise use Python
        #[cfg(windows)]
        {
            let engine_dir = EngineDownloader::default_engine_dir();
            let engine_path = engine_dir.join("engine.exe");
            
            // If engine.exe exists, use it
            if engine_path.exists() {
                let log_dir = self.workspace.join("logs");
                std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
                let log_file = log_dir.join("miner.log");

                let mut cmd = Command::new(&engine_path);
                hide_console_window(&mut cmd);
                cmd.arg("start")
                    .current_dir(&engine_dir);

                let child = cmd
                    .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                    .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                    .spawn()
                    .map_err(|e| e.to_string())?;
                
                let pid = child.id();
                *miner_pid = Some(pid);
                
                return Ok(ProcessResult {
                    success: true,
                    message: "Miner started (engine.exe)".to_string(),
                    pid: Some(pid),
                });
            }
            // Otherwise fall through to Python/miner command
        }

        // Try to find r3mes-miner or use venv python
        let venv_python = {
            #[cfg(windows)]
            {
                self.workspace.join("miner-engine").join("venv").join("Scripts").join("python.exe")
            }
            #[cfg(not(windows))]
            {
                self.workspace.join("miner-engine").join("venv").join("bin").join("python3")
            }
        };
        let (miner_cmd, miner_args) = if venv_python.exists() {
            (venv_python.to_string_lossy().to_string(), vec!["-m", "r3mes.cli.commands", "start"])
        } else {
            ("r3mes-miner".to_string(), vec!["start"])
        };

        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        let log_file = log_dir.join("miner.log");

        let mut cmd = Command::new(&miner_cmd);
        #[cfg(windows)]
        hide_console_window(&mut cmd);
        cmd.args(&miner_args)
            .current_dir(self.workspace.join("miner-engine"));

        #[cfg(unix)]
        let result = {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *miner_pid = Some(pid);
            drop(miner_pid); // Release lock before persist
            
            pid
        };
        
        #[cfg(windows)]
        let result = {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *miner_pid = Some(pid);
            drop(miner_pid); // Release lock before persist
            
            pid
        };
        
        // Persist state after successful start
        self.persist_state().await;
        
        Ok(ProcessResult {
            success: true,
            message: "Miner started".to_string(),
            pid: Some(result),
        })
    }

    pub async fn stop_miner(&mut self) -> Result<ProcessResult, String> {
        self.stop_process_generic(ProcessType::Miner).await
    }

    pub async fn start_ipfs(&mut self) -> Result<ProcessResult, String> {
        let mut ipfs_pid = self.ipfs_pid.lock().await;
        if ipfs_pid.is_some() {
            return Ok(ProcessResult {
                success: false,
                message: "IPFS is already running".to_string(),
                pid: None,
            });
        }

        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        let log_file = log_dir.join("ipfs.log");

        let mut cmd = Command::new("ipfs");
        #[cfg(windows)]
        hide_console_window(&mut cmd);
        cmd.arg("daemon");

        #[cfg(unix)]
        let result = {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *ipfs_pid = Some(pid);
            drop(ipfs_pid); // Release lock before persist
            
            pid
        };
        
        #[cfg(windows)]
        let result = {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *ipfs_pid = Some(pid);
            drop(ipfs_pid); // Release lock before persist
            
            pid
        };
        
        // Persist state after successful start
        self.persist_state().await;
        
        Ok(ProcessResult {
            success: true,
            message: "IPFS started".to_string(),
            pid: Some(result),
        })
    }

    pub async fn stop_ipfs(&mut self) -> Result<ProcessResult, String> {
        self.stop_process_generic(ProcessType::Ipfs).await
    }

    pub async fn start_serving(&mut self) -> Result<ProcessResult, String> {
        let mut serving_pid = self.serving_pid.lock().await;
        if serving_pid.is_some() {
            return Ok(ProcessResult {
                success: false,
                message: "Serving node is already running".to_string(),
                pid: None,
            });
        }

        // Try to find r3mes-serving command
        let venv_python = {
            #[cfg(windows)]
            {
                self.workspace.join("miner-engine").join("venv").join("Scripts").join("python.exe")
            }
            #[cfg(not(windows))]
            {
                self.workspace.join("miner-engine").join("venv").join("bin").join("python3")
            }
        };
        let (serving_cmd, serving_args) = if venv_python.exists() {
            (venv_python.to_string_lossy().to_string(), vec!["-m", "r3mes.cli.serving_commands", "start"])
        } else {
            ("r3mes-serving".to_string(), vec!["start"])
        };

        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        let log_file = log_dir.join("serving.log");

        let mut cmd = Command::new(&serving_cmd);
        #[cfg(windows)]
        hide_console_window(&mut cmd);
        cmd.args(&serving_args)
            .current_dir(self.workspace.join("miner-engine"));

        #[cfg(unix)]
        let result = {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *serving_pid = Some(pid);
            drop(serving_pid); // Release lock before persist
            
            pid
        };
        
        #[cfg(windows)]
        let result = {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *serving_pid = Some(pid);
            drop(serving_pid); // Release lock before persist
            
            pid
        };
        
        // Persist state after successful start
        self.persist_state().await;
        
        Ok(ProcessResult {
            success: true,
            message: "Serving node started".to_string(),
            pid: Some(result),
        })
    }

    pub async fn stop_serving(&mut self) -> Result<ProcessResult, String> {
        self.stop_process_generic(ProcessType::Serving).await
    }

    pub async fn start_validator(&mut self) -> Result<ProcessResult, String> {
        let mut validator_pid = self.validator_pid.lock().await;
        if validator_pid.is_some() {
            return Ok(ProcessResult {
                success: false,
                message: "Validator is already running".to_string(),
                pid: None,
            });
        }

        // Validator uses remesd with validator mode
        let remesd_path = self.workspace.join("remes").join("build").join("remesd");
        if !remesd_path.exists() {
            return Ok(ProcessResult {
                success: false,
                message: format!("remesd binary not found at {:?}", remesd_path),
                pid: None,
            });
        }

        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        let log_file = log_dir.join("validator.log");

        let mut cmd = Command::new(&remesd_path);
        #[cfg(windows)]
        hide_console_window(&mut cmd);
        cmd.arg("start")
            .current_dir(self.workspace.join("remes"));

        #[cfg(unix)]
        let result = {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *validator_pid = Some(pid);
            drop(validator_pid); // Release lock before persist
            
            pid
        };
        
        #[cfg(windows)]
        let result = {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *validator_pid = Some(pid);
            drop(validator_pid); // Release lock before persist
            
            pid
        };
        
        // Persist state after successful start
        self.persist_state().await;
        
        Ok(ProcessResult {
            success: true,
            message: "Validator started".to_string(),
            pid: Some(result),
        })
    }

    pub async fn stop_validator(&mut self) -> Result<ProcessResult, String> {
        self.stop_process_generic(ProcessType::Validator).await
    }

    pub async fn start_proposer(&mut self) -> Result<ProcessResult, String> {
        let mut proposer_pid = self.proposer_pid.lock().await;
        if proposer_pid.is_some() {
            return Ok(ProcessResult {
                success: false,
                message: "Proposer is already running".to_string(),
                pid: None,
            });
        }

        // Try to find r3mes-proposer command
        let venv_python = {
            #[cfg(windows)]
            {
                self.workspace.join("miner-engine").join("venv").join("Scripts").join("python.exe")
            }
            #[cfg(not(windows))]
            {
                self.workspace.join("miner-engine").join("venv").join("bin").join("python3")
            }
        };
        let (proposer_cmd, proposer_args) = if venv_python.exists() {
            (venv_python.to_string_lossy().to_string(), vec!["-m", "r3mes.cli.proposer_commands", "start"])
        } else {
            ("r3mes-proposer".to_string(), vec!["start"])
        };

        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        let log_file = log_dir.join("proposer.log");

        let mut cmd = Command::new(&proposer_cmd);
        #[cfg(windows)]
        hide_console_window(&mut cmd);
        cmd.args(&proposer_args)
            .current_dir(self.workspace.join("miner-engine"));

        #[cfg(unix)]
        let result = {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *proposer_pid = Some(pid);
            drop(proposer_pid); // Release lock before persist
            
            pid
        };
        
        #[cfg(windows)]
        let result = {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *proposer_pid = Some(pid);
            drop(proposer_pid); // Release lock before persist
            
            pid
        };
        
        // Persist state after successful start
        self.persist_state().await;
        
        Ok(ProcessResult {
            success: true,
            message: "Proposer started".to_string(),
            pid: Some(result),
        })
    }

    pub async fn stop_proposer(&mut self) -> Result<ProcessResult, String> {
        self.stop_process_generic(ProcessType::Proposer).await
    }

    pub async fn get_status(&self) -> ProcessStatus {
        let node_pid = self.node_pid.lock().await;
        let miner_pid = self.miner_pid.lock().await;
        let ipfs_pid = self.ipfs_pid.lock().await;
        let serving_pid = self.serving_pid.lock().await;
        let validator_pid = self.validator_pid.lock().await;
        let proposer_pid = self.proposer_pid.lock().await;

        ProcessStatus {
            node: ProcessInfo {
                running: node_pid.is_some(),
                pid: *node_pid,
            },
            miner: ProcessInfo {
                running: miner_pid.is_some(),
                pid: *miner_pid,
            },
            ipfs: ProcessInfo {
                running: ipfs_pid.is_some(),
                pid: *ipfs_pid,
            },
            serving: ProcessInfo {
                running: serving_pid.is_some(),
                pid: *serving_pid,
            },
            validator: ProcessInfo {
                running: validator_pid.is_some(),
                pid: *validator_pid,
            },
            proposer: ProcessInfo {
                running: proposer_pid.is_some(),
                pid: *proposer_pid,
            },
        }
    }

    pub async fn get_logs(&self) -> Vec<String> {
        let log_dir = self.workspace.join("logs");
        let mut logs = Vec::new();

        for log_file in &["node.log", "miner.log", "ipfs.log", "serving.log", "validator.log", "proposer.log"] {
            let path = log_dir.join(log_file);
            if let Ok(content) = std::fs::read_to_string(&path) {
                for line in content.lines().rev().take(10) {
                    logs.push(format!("[{}] {}", log_file.replace(".log", ""), line));
                }
            }
        }

        logs.reverse();
        logs
    }

    pub async fn get_logs_tail(&self, process: &str, lines: usize) -> Vec<String> {
        let log_dir = self.workspace.join("logs");
        let log_file = format!("{}.log", process);
        let path = log_dir.join(&log_file);
        
        if let Ok(content) = std::fs::read_to_string(&path) {
            content
                .lines()
                .rev()
                .take(lines)
                .map(|line| line.to_string())
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect()
        } else {
            Vec::new()
        }
    }

    pub async fn get_logs_by_level(&self, process: &str, level: &str) -> Vec<String> {
        let log_dir = self.workspace.join("logs");
        let log_file = format!("{}.log", process);
        let path = log_dir.join(&log_file);
        
        if let Ok(content) = std::fs::read_to_string(&path) {
            content
                .lines()
                .filter(|line| {
                    let line_lower = line.to_lowercase();
                    match level {
                        "error" => line_lower.contains("error") || line_lower.contains("fatal"),
                        "warning" => line_lower.contains("warning") || line_lower.contains("warn"),
                        "info" => line_lower.contains("info"),
                        "debug" => line_lower.contains("debug"),
                        _ => true,
                    }
                })
                .map(|line| line.to_string())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Cleanup all child processes (node, miner, IPFS)
    /// This should be called on application shutdown to prevent zombie processes
    pub async fn cleanup_all(&mut self) -> Result<(), String> {
        let _ = self.stop_node().await;
        let _ = self.stop_miner().await;
        let _ = self.stop_ipfs().await;
        let _ = self.stop_serving().await;
        let _ = self.stop_validator().await;
        let _ = self.stop_proposer().await;
        
        // Additional cleanup: kill any remaining processes by name
        #[cfg(unix)]
        {
            let _ = silent_command("pkill")
                .arg("-9")
                .arg("-f")
                .arg("remesd")
                .output();
            let _ = silent_command("pkill")
                .arg("-9")
                .arg("-f")
                .arg("r3mes-miner")
                .output();
            let _ = silent_command("pkill")
                .arg("-9")
                .arg("-f")
                .arg("r3mes-serving")
                .output();
            let _ = silent_command("pkill")
                .arg("-9")
                .arg("-f")
                .arg("r3mes-proposer")
                .output();
            let _ = silent_command("pkill")
                .arg("-9")
                .arg("ipfs")
                .output();
        }
        
        #[cfg(windows)]
        {
            let _ = silent_command("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("remesd.exe")
                .output();
            let _ = silent_command("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("r3mes-miner.exe")
                .output();
            let _ = silent_command("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("r3mes-serving.exe")
                .output();
            let _ = silent_command("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("r3mes-proposer.exe")
                .output();
            let _ = silent_command("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("engine.exe")
                .output();
            let _ = silent_command("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("ipfs.exe")
                .output();
        }
        
        Ok(())
    }
}
