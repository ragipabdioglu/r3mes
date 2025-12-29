use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use tokio::sync::Mutex;

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

#[derive(Debug, Serialize, Deserialize)]
pub struct ProcessInfo {
    pub running: bool,
    pub pid: Option<u32>,
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
        let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
        let workspace = PathBuf::from(home).join("R3MES");
        
        Self {
            workspace,
            node_pid: Arc::new(Mutex::new(None)),
            miner_pid: Arc::new(Mutex::new(None)),
            ipfs_pid: Arc::new(Mutex::new(None)),
            serving_pid: Arc::new(Mutex::new(None)),
            validator_pid: Arc::new(Mutex::new(None)),
            proposer_pid: Arc::new(Mutex::new(None)),
        }
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
        cmd.arg("start")
            .current_dir(self.workspace.join("remes"));

        #[cfg(unix)]
        {
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
            
            Ok(ProcessResult {
                success: true,
                message: "Node started".to_string(),
                pid: Some(pid),
            })
        }
        
        #[cfg(windows)]
        {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *node_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "Node started".to_string(),
                pid: Some(pid),
            })
        }
    }

    pub async fn stop_node(&mut self) -> Result<ProcessResult, String> {
        let mut node_pid = self.node_pid.lock().await;
        
        if let Some(pid) = node_pid.take() {
            #[cfg(unix)]
            {
                let _ = Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .output();
                
                // Wait a bit, then force kill
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                let _ = Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
            
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/T")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Node stopped".to_string(),
                pid: None,
            })
        } else {
            // Force kill any remaining processes
            #[cfg(unix)]
            {
                let _ = Command::new("pkill")
                    .arg("-9")
                    .arg("-f")
                    .arg("remesd")
                    .output();
            }
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/F")
                    .arg("/IM")
                    .arg("remesd.exe")
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Node stopped (force)".to_string(),
                pid: None,
            })
        }
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
        let venv_python = self.workspace.join("miner-engine").join("venv").join("bin").join("python3");
        let (miner_cmd, miner_args) = if venv_python.exists() {
            (venv_python.to_string_lossy().to_string(), vec!["-m", "r3mes.cli.commands", "start"])
        } else {
            ("r3mes-miner".to_string(), vec!["start"])
        };

        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        let log_file = log_dir.join("miner.log");

        let mut cmd = Command::new(&miner_cmd);
        cmd.args(&miner_args)
            .current_dir(self.workspace.join("miner-engine"));

        #[cfg(unix)]
        {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *miner_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "Miner started".to_string(),
                pid: Some(pid),
            })
        }
        
        #[cfg(windows)]
        {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *miner_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "Miner started".to_string(),
                pid: Some(pid),
            })
        }
    }

    pub async fn stop_miner(&mut self) -> Result<ProcessResult, String> {
        let mut miner_pid = self.miner_pid.lock().await;
        
        if let Some(pid) = miner_pid.take() {
            #[cfg(unix)]
            {
                let _ = Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .output();
                
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                let _ = Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
            
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/T")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Miner stopped".to_string(),
                pid: None,
            })
        } else {
            // Force kill
            #[cfg(unix)]
            {
                let _ = Command::new("pkill")
                    .arg("-9")
                    .arg("-f")
                    .arg("r3mes-miner")
                    .output();
            }
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/F")
                    .arg("/IM")
                    .arg("r3mes-miner.exe")
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Miner stopped (force)".to_string(),
                pid: None,
            })
        }
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
        cmd.arg("daemon");

        #[cfg(unix)]
        {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *ipfs_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "IPFS started".to_string(),
                pid: Some(pid),
            })
        }
        
        #[cfg(windows)]
        {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *ipfs_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "IPFS started".to_string(),
                pid: Some(pid),
            })
        }
    }

    pub async fn stop_ipfs(&mut self) -> Result<ProcessResult, String> {
        let mut ipfs_pid = self.ipfs_pid.lock().await;
        
        if let Some(pid) = ipfs_pid.take() {
            #[cfg(unix)]
            {
                let _ = Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .output();
                
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                let _ = Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
            
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/T")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "IPFS stopped".to_string(),
                pid: None,
            })
        } else {
            // Force kill
            #[cfg(unix)]
            {
                let _ = Command::new("pkill")
                    .arg("-9")
                    .arg("ipfs")
                    .output();
            }
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/F")
                    .arg("/IM")
                    .arg("ipfs.exe")
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "IPFS stopped (force)".to_string(),
                pid: None,
            })
        }
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
        let venv_python = self.workspace.join("miner-engine").join("venv").join("bin").join("python3");
        let (serving_cmd, serving_args) = if venv_python.exists() {
            (venv_python.to_string_lossy().to_string(), vec!["-m", "r3mes.cli.serving_commands", "start"])
        } else {
            ("r3mes-serving".to_string(), vec!["start"])
        };

        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        let log_file = log_dir.join("serving.log");

        let mut cmd = Command::new(&serving_cmd);
        cmd.args(&serving_args)
            .current_dir(self.workspace.join("miner-engine"));

        #[cfg(unix)]
        {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *serving_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "Serving node started".to_string(),
                pid: Some(pid),
            })
        }
        
        #[cfg(windows)]
        {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *serving_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "Serving node started".to_string(),
                pid: Some(pid),
            })
        }
    }

    pub async fn stop_serving(&mut self) -> Result<ProcessResult, String> {
        let mut serving_pid = self.serving_pid.lock().await;
        
        if let Some(pid) = serving_pid.take() {
            #[cfg(unix)]
            {
                let _ = Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .output();
                
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                let _ = Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
            
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/T")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Serving node stopped".to_string(),
                pid: None,
            })
        } else {
            #[cfg(unix)]
            {
                let _ = Command::new("pkill")
                    .arg("-9")
                    .arg("-f")
                    .arg("r3mes-serving")
                    .output();
            }
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/F")
                    .arg("/IM")
                    .arg("r3mes-serving.exe")
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Serving node stopped (force)".to_string(),
                pid: None,
            })
        }
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
        cmd.arg("start")
            .current_dir(self.workspace.join("remes"));

        #[cfg(unix)]
        {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *validator_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "Validator started".to_string(),
                pid: Some(pid),
            })
        }
        
        #[cfg(windows)]
        {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *validator_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "Validator started".to_string(),
                pid: Some(pid),
            })
        }
    }

    pub async fn stop_validator(&mut self) -> Result<ProcessResult, String> {
        let mut validator_pid = self.validator_pid.lock().await;
        
        if let Some(pid) = validator_pid.take() {
            #[cfg(unix)]
            {
                let _ = Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .output();
                
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                let _ = Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
            
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/T")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Validator stopped".to_string(),
                pid: None,
            })
        } else {
            #[cfg(unix)]
            {
                let _ = Command::new("pkill")
                    .arg("-9")
                    .arg("-f")
                    .arg("remesd.*validator")
                    .output();
            }
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/F")
                    .arg("/IM")
                    .arg("remesd.exe")
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Validator stopped (force)".to_string(),
                pid: None,
            })
        }
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
        let venv_python = self.workspace.join("miner-engine").join("venv").join("bin").join("python3");
        let (proposer_cmd, proposer_args) = if venv_python.exists() {
            (venv_python.to_string_lossy().to_string(), vec!["-m", "r3mes.cli.proposer_commands", "start"])
        } else {
            ("r3mes-proposer".to_string(), vec!["start"])
        };

        let log_dir = self.workspace.join("logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;
        let log_file = log_dir.join("proposer.log");

        let mut cmd = Command::new(&proposer_cmd);
        cmd.args(&proposer_args)
            .current_dir(self.workspace.join("miner-engine"));

        #[cfg(unix)]
        {
            let log_file_clone = log_file.clone();
            cmd.stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file_clone).map_err(|e| e.to_string())?);
            
            let child = cmd.spawn().map_err(|e| e.to_string())?;
            let pid = child.id();
            std::mem::forget(child);
            
            *proposer_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "Proposer started".to_string(),
                pid: Some(pid),
            })
        }
        
        #[cfg(windows)]
        {
            let child = cmd
                .stdout(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .stderr(std::fs::File::create(&log_file).map_err(|e| e.to_string())?)
                .spawn()
                .map_err(|e| e.to_string())?;
            
            let pid = child.id();
            *proposer_pid = Some(pid);
            
            Ok(ProcessResult {
                success: true,
                message: "Proposer started".to_string(),
                pid: Some(pid),
            })
        }
    }

    pub async fn stop_proposer(&mut self) -> Result<ProcessResult, String> {
        let mut proposer_pid = self.proposer_pid.lock().await;
        
        if let Some(pid) = proposer_pid.take() {
            #[cfg(unix)]
            {
                let _ = Command::new("kill")
                    .arg("-TERM")
                    .arg(pid.to_string())
                    .output();
                
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                let _ = Command::new("kill")
                    .arg("-9")
                    .arg(pid.to_string())
                    .output();
            }
            
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/T")
                    .arg("/PID")
                    .arg(pid.to_string())
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Proposer stopped".to_string(),
                pid: None,
            })
        } else {
            #[cfg(unix)]
            {
                let _ = Command::new("pkill")
                    .arg("-9")
                    .arg("-f")
                    .arg("r3mes-proposer")
                    .output();
            }
            #[cfg(windows)]
            {
                let _ = Command::new("taskkill")
                    .arg("/F")
                    .arg("/IM")
                    .arg("r3mes-proposer.exe")
                    .output();
            }

            Ok(ProcessResult {
                success: true,
                message: "Proposer stopped (force)".to_string(),
                pid: None,
            })
        }
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
            let _ = Command::new("pkill")
                .arg("-9")
                .arg("-f")
                .arg("remesd")
                .output();
            let _ = Command::new("pkill")
                .arg("-9")
                .arg("-f")
                .arg("r3mes-miner")
                .output();
            let _ = Command::new("pkill")
                .arg("-9")
                .arg("-f")
                .arg("r3mes-serving")
                .output();
            let _ = Command::new("pkill")
                .arg("-9")
                .arg("-f")
                .arg("r3mes-proposer")
                .output();
            let _ = Command::new("pkill")
                .arg("-9")
                .arg("ipfs")
                .output();
        }
        
        #[cfg(windows)]
        {
            let _ = Command::new("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("remesd.exe")
                .output();
            let _ = Command::new("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("r3mes-miner.exe")
                .output();
            let _ = Command::new("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("r3mes-serving.exe")
                .output();
            let _ = Command::new("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("r3mes-proposer.exe")
                .output();
            let _ = Command::new("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("engine.exe")
                .output();
            let _ = Command::new("taskkill")
                .arg("/F")
                .arg("/IM")
                .arg("ipfs.exe")
                .output();
        }
        
        Ok(())
    }
}
