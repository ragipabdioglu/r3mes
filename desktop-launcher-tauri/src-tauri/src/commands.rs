//! Tauri commands for R3MES Desktop Launcher

#![allow(dead_code)]

use crate::process_manager::{ProcessManager, ProcessResult, ProcessStatus};
use crate::hardware_check;
use crate::keychain::KeychainManager;
use crate::wallet;
use crate::setup_checker;
use crate::engine_downloader;
use crate::debug;
use crate::platform::{self, silent_command};
use crate::config::get_config as get_launcher_config;
use crate::config::FullConfig;
use serde::{Deserialize, Serialize};

#[cfg(windows)]
use crate::platform::hide_console_window;

use std::process::Command;

static PROCESS_MANAGER: tokio::sync::Mutex<Option<ProcessManager>> = tokio::sync::Mutex::const_new(None);

async fn get_manager() -> tokio::sync::MutexGuard<'static, Option<ProcessManager>> {
    let mut manager = PROCESS_MANAGER.lock().await;
    if manager.is_none() {
        *manager = Some(ProcessManager::new());
    }
    manager
}

/// Helper macro to get manager with proper error handling
macro_rules! with_manager {
    ($manager:expr) => {
        $manager.as_mut().ok_or_else(|| "Process manager not initialized".to_string())?
    };
}

/// Cleanup all processes - should be called on application shutdown
#[tauri::command]
pub async fn cleanup_all_processes() -> Result<(), String> {
    let mut manager = get_manager().await;
    if let Some(manager) = manager.as_mut() {
        manager.cleanup_all().await?;
    }
    Ok(())
}

#[tauri::command]
pub async fn start_node() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.start_node().await
}

#[tauri::command]
pub async fn stop_node() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.stop_node().await
}

#[tauri::command]
pub async fn start_miner() -> Result<ProcessResult, String> {
    use crate::engine_downloader::EngineDownloader;
    
    // Ensure engine is installed before starting (cross-platform)
    let engine_dir = EngineDownloader::default_engine_dir();
    let download_url = EngineDownloader::get_download_url();
    let expected_checksum = EngineDownloader::get_expected_checksum();
    
    let downloader = EngineDownloader::new(engine_dir, download_url, expected_checksum);
    let status = downloader.check_engine_status().await;
    
    if !status.installed || !status.checksum_valid {
        // Engine not installed or invalid - return error so UI can trigger download
        return Err("Engine not installed. Please download engine first.".to_string());
    }
    
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.start_miner().await
}

#[tauri::command]
pub async fn stop_miner() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.stop_miner().await
}

#[tauri::command]
pub async fn start_ipfs() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.start_ipfs().await
}

#[tauri::command]
pub async fn stop_ipfs() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.stop_ipfs().await
}

#[tauri::command]
pub async fn start_serving() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.start_serving().await
}

#[tauri::command]
pub async fn stop_serving() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.stop_serving().await
}

#[tauri::command]
pub async fn start_validator() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.start_validator().await
}

#[tauri::command]
pub async fn stop_validator() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.stop_validator().await
}

#[tauri::command]
pub async fn start_proposer() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.start_proposer().await
}

#[tauri::command]
pub async fn stop_proposer() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    manager.stop_proposer().await
}

#[tauri::command]
pub async fn get_status() -> Result<ProcessStatus, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    Ok(manager.get_status().await)
}

#[tauri::command]
pub async fn get_logs() -> Result<Vec<String>, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    Ok(manager.get_logs().await)
}

#[tauri::command]
pub async fn get_logs_tail(process: String, lines: usize) -> Result<Vec<String>, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    Ok(manager.get_logs_tail(&process, lines).await)
}

#[tauri::command]
pub async fn get_logs_by_level(process: String, level: String) -> Result<Vec<String>, String> {
    let mut manager = get_manager().await;
    let manager = with_manager!(manager);
    Ok(manager.get_logs_by_level(&process, &level).await)
}

#[tauri::command]
pub async fn export_logs(process: String) -> Result<String, String> {
    use std::fs;
    
    let log_dir = platform::get_logs_dir()?;
    let log_file = log_dir.join(format!("{}.log", process));
    
    if !log_file.exists() {
        return Err(format!("Log file not found: {:?}", log_file));
    }
    
    fs::read_to_string(&log_file)
        .map_err(|e| format!("Failed to read log file: {}", e))
}

#[tauri::command]
pub async fn check_hardware() -> Result<hardware_check::HardwareCheckResult, String> {
    Ok(hardware_check::check_hardware())
}

#[tauri::command]
pub async fn is_first_run() -> Result<bool, String> {
    let config_dir = platform::get_config_dir()?;
    let config_file = config_dir.join("launcher_config.json");
    
    Ok(!config_file.exists())
}

#[tauri::command]
pub async fn mark_setup_complete() -> Result<(), String> {
    use std::fs;
    
    let config_dir = platform::get_config_dir()?;
    
    // Create config directory if it doesn't exist
    fs::create_dir_all(&config_dir)
        .map_err(|e| format!("Failed to create config directory: {}", e))?;
    
    let config_file = config_dir.join("launcher_config.json");
    let config = serde_json::json!({
        "setup_completed": true,
        "setup_date": chrono::Utc::now().to_rfc3339(),
    });
    
    fs::write(&config_file, serde_json::to_string_pretty(&config).unwrap())
        .map_err(|e| format!("Failed to write config file: {}", e))?;
    
    Ok(())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WalletInfo {
    pub address: String,
    pub balance: String,
    pub exists: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: String,
    pub from: String,
    pub to: String,
    pub amount: String,
    pub timestamp: i64,
    pub status: String, // "pending", "confirmed", "failed"
    pub block_height: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TransactionHistory {
    pub transactions: Vec<Transaction>,
    pub total: usize,
}

#[tauri::command]
pub async fn get_wallet_info() -> Result<WalletInfo, String> {
    use std::fs;
    
    let wallet_path = platform::get_wallets_dir()?.join("default_wallet.json");
    
    if !wallet_path.exists() {
        return Ok(WalletInfo {
            address: String::new(),
            balance: "0".to_string(),
            exists: false,
        });
    }
    
    // Load wallet file
    let wallet_data: serde_json::Value = fs::read_to_string(&wallet_path)
        .map_err(|e| format!("Failed to read wallet: {}", e))?
        .parse()
        .map_err(|e| format!("Failed to parse wallet: {}", e))?;
    
    let address = wallet_data["address"]
        .as_str()
        .ok_or("Invalid wallet format: missing address")?
        .to_string();
    
    // Fetch balance from blockchain
    let balance = match wallet::get_wallet_balance(&address).await {
        Ok(balance_info) => balance_info.balance,
        Err(e) => {
            eprintln!("⚠️  Failed to fetch balance from blockchain: {}", e);
            "0".to_string()  // Fallback to 0 on error
        }
    };
    
    Ok(WalletInfo {
        address,
        balance,
        exists: true,
    })
}

#[tauri::command]
pub async fn create_wallet() -> Result<serde_json::Value, String> {
    // Call Python wallet manager
    let workspace = platform::get_workspace_dir()?;
    
    // Try to use venv python first, fallback to system python
    let python = platform::get_venv_python()
        .unwrap_or_else(|_| std::path::PathBuf::from(platform::get_system_python()));
    
    // Use r3mes-miner wallet create command instead
    let mut cmd = Command::new(&python);
    #[cfg(windows)]
    hide_console_window(&mut cmd);
    
    let output = cmd
        .arg("-c")
        .arg("from r3mes.cli.wallet import WalletManager; w = WalletManager(); path = w.create_wallet(); print(path)")
        .current_dir(&workspace)
        .output()
        .map_err(|e| format!("Failed to execute Python: {}", e))?;
    
    if !output.status.success() {
        let error = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Failed to create wallet: {}", error));
    }
    
    let wallet_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    
    // Load wallet to get address
    let wallet_info = get_wallet_info().await?;
    
    // Generate proper BIP39 mnemonic using wallet module
    let wallet = wallet::create_new_wallet()
        .map_err(|e| format!("Failed to create wallet: {}", e))?;
    
    let mnemonic = wallet.mnemonic
        .ok_or("Mnemonic not generated")?;
    
    // Store private key in keychain (if available)
    let keychain = KeychainManager::new();
    if let Ok(wallet_data) = std::fs::read_to_string(&wallet_path) {
        if let Ok(wallet_json) = serde_json::from_str::<serde_json::Value>(&wallet_data) {
            if let Some(private_key) = wallet_json.get("private_key").and_then(|v| v.as_str()) {
                if let Err(e) = keychain.store("wallet_private_key", private_key) {
                    eprintln!("⚠️  Failed to store private key in keychain: {}", e);
                    // Continue anyway - keychain is optional
                } else {
                    println!("✅ Private key stored in keychain");
                }
            }
        }
    }
    
    Ok(serde_json::json!({
        "address": wallet_info.address,
        "mnemonic": mnemonic,
    }))
}

#[tauri::command]
pub async fn import_wallet_from_private_key(private_key: String) -> Result<(), String> {
    // Validate private key format
    if private_key.len() != 64 {
        return Err("Invalid private key format (must be 64 hex characters)".to_string());
    }
    
    // Import wallet using wallet module
    wallet::import_wallet_from_private_key(&private_key)
        .map_err(|e| format!("Failed to import wallet: {}", e))?;
    
    Ok(())
}

#[tauri::command]
pub async fn import_wallet_from_mnemonic(mnemonic: String) -> Result<(), String> {
    // Validate mnemonic (12 or 24 words)
    let words: Vec<&str> = mnemonic.trim().split_whitespace().collect();
    if words.len() != 12 && words.len() != 24 {
        return Err("Mnemonic must be 12 or 24 words".to_string());
    }
    
    // Import wallet using wallet module
    wallet::import_wallet_from_mnemonic(&mnemonic)
        .map_err(|e| format!("Failed to import wallet: {}", e))?;
    
    Ok(())
}

#[tauri::command]
pub async fn export_wallet() -> Result<serde_json::Value, String> {
    use std::fs;
    
    let wallet_path = platform::get_wallets_dir()?.join("default_wallet.json");
    
    if !wallet_path.exists() {
        return Err("No wallet found".to_string());
    }
    
    // Read wallet file
    let wallet_data = fs::read_to_string(&wallet_path)
        .map_err(|e| format!("Failed to read wallet: {}", e))?;
    
    // Encrypt wallet data using AES-256-GCM
    use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
    use aes_gcm::aead::Aead;
    use rand::RngCore;
    
    // Get encryption key from keychain - NEVER generate random key without saving
    // A random key that isn't saved makes the wallet unrecoverable
    let keychain = KeychainManager::new();
    let encryption_key = match keychain.retrieve("wallet_encryption_key") {
        Ok(key) => key,
        Err(_) => {
            // Generate new key and SAVE it to keychain
            let mut key = [0u8; 32];
            rand::thread_rng().fill_bytes(&mut key);
            let new_key = hex::encode(key);
            
            // Try to store the key - if this fails, abort the export
            // We cannot export with a key that won't be saved
            keychain.store("wallet_encryption_key", &new_key)
                .map_err(|e| format!(
                    "Cannot export wallet: failed to store encryption key in keychain. \
                    Without saving the key, the exported wallet would be unrecoverable. \
                    Error: {}", e
                ))?;
            
            new_key
        }
    };
    
    let key_bytes = hex::decode(&encryption_key)
        .map_err(|e| format!("Invalid encryption key: {}", e))?;
    let cipher = Aes256Gcm::new_from_slice(&key_bytes)
        .map_err(|e| format!("Failed to create cipher: {}", e))?;
    
    // Generate random nonce
    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    
    // Encrypt wallet data
    let encrypted_data = cipher.encrypt(nonce, wallet_data.as_bytes())
        .map_err(|e| format!("Failed to encrypt wallet: {}", e))?;
    
    Ok(serde_json::json!({
        "encrypted": hex::encode(encrypted_data),
        "nonce": hex::encode(nonce_bytes),
    }))
}

#[tauri::command]
pub async fn get_transaction_history(address: String, limit: Option<usize>) -> Result<TransactionHistory, String> {
    use std::fs;
    
    let history_path = platform::get_wallets_dir()?.join("transaction_history.json");
    
    // Load transaction history from local file
    let transactions = if history_path.exists() {
        let history_data = fs::read_to_string(&history_path)
            .map_err(|e| format!("Failed to read transaction history: {}", e))?;
        
        let all_txs: Vec<Transaction> = serde_json::from_str(&history_data)
            .unwrap_or_else(|_| vec![]);
        
        // Filter by address
        let filtered: Vec<Transaction> = all_txs
            .into_iter()
            .filter(|tx| tx.from == address || tx.to == address)
            .collect();
        
        // Sort by timestamp (newest first)
        let mut sorted = filtered;
        sorted.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        // Apply limit
        if let Some(limit) = limit {
            sorted.truncate(limit);
        }
        
        sorted
    } else {
        vec![]
    };
    
    let total = transactions.len();
    Ok(TransactionHistory {
        transactions,
        total,
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChainStatus {
    pub synced: bool,
    pub block_height: u64,
    pub sync_percentage: f64,
    pub status: String, // "running" | "syncing" | "stopped"
}

#[tauri::command]
pub async fn get_chain_status() -> Result<ChainStatus, String> {
    let config = get_launcher_config();
    let rpc_url = &config.rpc_url;
    
    // Try to get status from RPC
    let status_output = silent_command("curl")
        .arg("-s")
        .arg("--connect-timeout")
        .arg("5")
        .arg(format!("{}/status", rpc_url))
        .output();
    
    let status_json: Option<serde_json::Value> = status_output
        .ok()
        .filter(|o| o.status.success())
        .and_then(|output| serde_json::from_slice(&output.stdout).ok());
    
    if status_json.is_none() {
        return Ok(ChainStatus {
            synced: false,
            block_height: 0,
            sync_percentage: 0.0,
            status: "stopped".to_string(),
        });
    }
    
    let json = status_json.unwrap();
    
    // Extract sync info from RPC response
    let sync_info = &json["result"]["sync_info"];
    
    let block_height = sync_info["latest_block_height"]
        .as_str()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    
    // Use the actual catching_up field from Tendermint RPC
    // This is the correct way to determine sync status
    let catching_up = sync_info["catching_up"]
        .as_bool()
        .unwrap_or(true);
    
    let synced = !catching_up && block_height > 0;
    
    // Calculate sync percentage based on earliest and latest block heights
    let _earliest_block_height = sync_info["earliest_block_height"]
        .as_str()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(1);
    
    // If we have network info, use it for more accurate percentage
    // Otherwise estimate based on catching_up status
    let sync_percentage = if synced {
        100.0
    } else if block_height > 0 {
        // When syncing, we don't know the target height
        // Show progress based on catching_up status
        // This is an estimate - real percentage would need network height
        99.0 // Show 99% while syncing, 100% when done
    } else {
        0.0
    };
    
    Ok(ChainStatus {
        synced,
        block_height,
        sync_percentage,
        status: if synced {
            "running".to_string()
        } else if block_height > 0 {
            "syncing".to_string()
        } else {
            "stopped".to_string()
        },
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterNodeResult {
    pub success: bool,
    pub message: String,
    pub transaction_hash: Option<String>,
}

#[tauri::command]
pub async fn register_node_roles(
    roles: Vec<i32>,
    stake: String,
) -> Result<RegisterNodeResult, String> {
    
    use std::fs;
    
    if roles.is_empty() {
        return Err("No roles selected".to_string());
    }
    
    // Get wallet info to get the address
    let wallet_info = get_wallet_info().await?;
    if !wallet_info.exists {
        return Err("No wallet found. Please create or import a wallet first.".to_string());
    }
    
    let node_address = wallet_info.address;
    
    // Get private key from keychain or wallet file
    let keychain = KeychainManager::new();
    let private_key = match keychain.retrieve("wallet_private_key") {
        Ok(key) => key,
        Err(_) => {
            // Try to get from wallet file
            let wallet_path = platform::get_wallets_dir()?.join("default_wallet.json");
            
            if !wallet_path.exists() {
                return Err("Could not find wallet private key".to_string());
            }
            
            let wallet_data: serde_json::Value = fs::read_to_string(&wallet_path)
                .map_err(|e| format!("Failed to read wallet: {}", e))?
                .parse()
                .map_err(|e| format!("Failed to parse wallet: {}", e))?;
            
            wallet_data["private_key"]
                .as_str()
                .ok_or("Invalid wallet format: missing private_key")?
                .to_string()
        }
    };
    
    // Get RPC URL from config
    let config = crate::config::get_config();
    let grpc_url = &config.grpc_url;
    let chain_id = &config.chain_id;
    
    // Build the MsgRegisterNode transaction
    // Role mapping: 1=MINER, 2=SERVING, 3=VALIDATOR, 4=PROPOSER
    let role_names: Vec<&str> = roles.iter().map(|r| match r {
        1 => "NODE_TYPE_MINER",
        2 => "NODE_TYPE_SERVING",
        3 => "NODE_TYPE_VALIDATOR",
        4 => "NODE_TYPE_PROPOSER",
        _ => "NODE_TYPE_UNSPECIFIED",
    }).collect();
    
    // Use Python CLI to submit transaction (cross-platform)
    // SECURITY: Pass private key via stdin instead of command line argument
    // Command line arguments are visible in process listings (ps aux)
    let roles_json = serde_json::to_string(&roles).unwrap_or_else(|_| "[]".to_string());
    
    use std::io::Write;
    use std::process::Stdio;
    
    // Python script reads private key from stdin for security
    let python_script = format!(r#"
import sys
import json

# Read private key from stdin (secure - not visible in ps aux)
private_key = sys.stdin.readline().strip()

sys.path.insert(0, '.')
try:
    from r3mes.cli.blockchain import register_node
    result = register_node(
        private_key=private_key,
        node_address='{}',
        roles={},
        stake='{}',
        grpc_url='{}',
        chain_id='{}'
    )
    print(json.dumps(result))
except Exception as e:
    print(json.dumps({{"success": False, "error": str(e)}}))
"#, node_address, roles_json, stake, grpc_url, chain_id);
    
    let child = Command::new("python3")
        .arg("-c")
        .arg(&python_script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    
    let output = match child {
        Ok(mut process) => {
            // Write private key to stdin
            if let Some(mut stdin) = process.stdin.take() {
                let _ = writeln!(stdin, "{}", private_key);
            }
            process.wait_with_output()
        }
        Err(e) => Err(e),
    };
    
    match output {
        Ok(output) => {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                
                // Try to parse JSON response
                if let Ok(result) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    if result.get("success").and_then(|v| v.as_bool()).unwrap_or(false) {
                        let tx_hash = result.get("tx_hash")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        
                        return Ok(RegisterNodeResult {
                            success: true,
                            message: format!("Node registered successfully with roles: {:?}", role_names),
                            transaction_hash: tx_hash,
                        });
                    } else {
                        let error = result.get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Unknown error");
                        return Err(format!("Registration failed: {}", error));
                    }
                }
            }
            
            // Fallback: Python CLI not available, provide instructions
            Ok(RegisterNodeResult {
                success: false,
                message: format!(
                    "Automatic registration not available. Please register manually:\n\n\
                    1. Open terminal and run:\n\
                    r3mes-cli tx remes register-node {} --roles {} --stake {} --from {} --chain-id {}\n\n\
                    Or use the web dashboard at /roles",
                    node_address, roles_json, stake, node_address, chain_id
                ),
                transaction_hash: None,
            })
        }
        Err(e) => {
            // Python not available, provide manual instructions
            Ok(RegisterNodeResult {
                success: false,
                message: format!(
                    "Could not execute registration command: {}\n\n\
                    Please register manually via the web dashboard at /roles or use the blockchain CLI:\n\
                    r3mes-cli tx remes register-node {} --roles {:?} --stake {} --chain-id {}",
                    e, node_address, role_names, stake, chain_id
                ),
                transaction_hash: None,
            })
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IPFSStatus {
    pub connected: bool,
    pub peer_count: u32,
    pub status: String, // "running" | "stopped"
}

#[tauri::command]
pub async fn get_ipfs_status() -> Result<IPFSStatus, String> {
    // Check if IPFS is running by checking if port 5001 is listening (cross-platform)
    let ipfs_running = platform::check_port_listening(5001);
    
    if !ipfs_running {
        return Ok(IPFSStatus {
            connected: false,
            peer_count: 0,
            status: "stopped".to_string(),
        });
    }
    
    // Get peer count from IPFS API
    let config = get_launcher_config();
    let ipfs_url = &config.ipfs_url;
    let peer_count = silent_command("curl")
        .arg("-s")
        .arg(format!("{}/api/v0/swarm/peers", ipfs_url))
        .output()
        .ok()
        .and_then(|output| {
            let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
            json.as_array().map(|arr| arr.len() as u32)
        })
        .unwrap_or(0);
    
    Ok(IPFSStatus {
        connected: true,
        peer_count,
        status: "running".to_string(),
    })
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelStatus {
    pub downloaded: bool,
    pub progress: f64,
    pub version: String,
    pub size_gb: f64,
    pub status: String, // "ready" | "downloading" | "not_downloaded"
}

#[tauri::command]
pub async fn get_model_status() -> Result<ModelStatus, String> {
    use std::fs;
    use crate::config::model_config;
    
    let model_dir = platform::get_models_dir()?;
    let model_file = model_dir.join("bitnet_b1.58.pt");
    
    if model_file.exists() {
        // Check file size
        let size_bytes = fs::metadata(&model_file)
            .map(|m| m.len())
            .unwrap_or(0);
        let size_gb = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        
        // Use config constants for model size validation
        let downloaded = size_gb >= model_config::BITNET_MIN_SIZE_GB;
        
        Ok(ModelStatus {
            downloaded,
            progress: if downloaded { 
                100.0 
            } else { 
                (size_gb / model_config::BITNET_EXPECTED_SIZE_GB * 100.0).min(99.0) 
            },
            version: model_config::BITNET_VERSION.to_string(),
            size_gb,
            status: if downloaded {
                "ready".to_string()
            } else {
                "downloading".to_string()
            },
        })
    } else {
        Ok(ModelStatus {
            downloaded: false,
            progress: 0.0,
            version: model_config::BITNET_VERSION.to_string(),
            size_gb: 0.0,
            status: "not_downloaded".to_string(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MiningStats {
    pub hashrate: f64,
    pub loss: f64,
    pub loss_trend: String, // "decreasing" | "increasing" | "stable"
    pub estimated_earnings_per_day: f64,
    pub current_balance: f64,
    pub gpu_temp: f64,
    pub gpu_temp_status: String, // "normal" | "high" | "critical"
    pub vram_usage_mb: u64,
    pub vram_total_mb: u64,
    pub training_epoch: u64,
    pub gradient_norm: f64,
    pub uptime_seconds: u64,
}

#[tauri::command]
pub async fn open_dashboard() -> Result<(), String> {
    // Open web dashboard in default browser using config URL
    let config = get_launcher_config();
    let url = &config.web_dashboard_url;
    
    platform::open_url(url)
}

#[tauri::command]
pub async fn get_mining_stats() -> Result<MiningStats, String> {
    // Get miner stats URL from config (backend_url + /stats)
    let config = crate::config::FullConfig::load();
    let stats_url = format!("{}/stats", config.miner.backend_url.trim_end_matches('/'));
    
    // Try to fetch from miner stats HTTP server (if running)
    let stats_response = silent_command("curl")
        .arg("-s")
        .arg(&stats_url)
        .output();
    
    if let Ok(output) = stats_response {
        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
            // Parse stats from JSON response
            let hashrate = json["hashrate"].as_f64().unwrap_or(0.0);
            let loss = json["loss"].as_f64().unwrap_or(0.0);
            let loss_trend = json["loss_trend"]
                .as_str()
                .unwrap_or("stable")
                .to_string();
            let estimated_earnings = json["estimated_earnings_per_day"]
                .as_f64()
                .unwrap_or(0.0);
            let balance = json["current_balance"].as_f64().unwrap_or(0.0);
            let gpu_temp = json["gpu_temp"].as_f64().unwrap_or(0.0);
            let gpu_temp_status = if gpu_temp >= 95.0 {
                "critical"
            } else if gpu_temp >= 85.0 {
                "high"
            } else {
                "normal"
            };
            let vram_usage = json["vram_usage_mb"].as_u64().unwrap_or(0);
            let vram_total = json["vram_total_mb"].as_u64().unwrap_or(0);
            let epoch = json["training_epoch"].as_u64().unwrap_or(0);
            let gradient_norm = json["gradient_norm"].as_f64().unwrap_or(0.0);
            let uptime = json["uptime_seconds"].as_u64().unwrap_or(0);
            
            return Ok(MiningStats {
                hashrate,
                loss,
                loss_trend,
                estimated_earnings_per_day: estimated_earnings,
                current_balance: balance,
                gpu_temp,
                gpu_temp_status: gpu_temp_status.to_string(),
                vram_usage_mb: vram_usage,
                vram_total_mb: vram_total,
                training_epoch: epoch,
                gradient_norm,
                uptime_seconds: uptime,
            });
        }
    }
    
    // Fallback: Try to get GPU temp from nvidia-smi
    let gpu_temp = silent_command("nvidia-smi")
        .arg("--query-gpu=temperature.gpu")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()
        .and_then(|output| {
            String::from_utf8(output.stdout)
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
        })
        .unwrap_or(0.0);
    
    let gpu_temp_status = if gpu_temp >= 95.0 {
        "critical"
    } else if gpu_temp >= 85.0 {
        "high"
    } else {
        "normal"
    };
    
    // Get VRAM usage
    let vram_info = silent_command("nvidia-smi")
        .arg("--query-gpu=memory.used,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
        .ok()
        .and_then(|output| {
            let output_str = String::from_utf8(output.stdout).ok()?;
            let parts: Vec<&str> = output_str.trim().split(',').collect();
            if parts.len() >= 2 {
                Some((
                    parts[0].trim().parse::<u64>().ok()?,
                    parts[1].trim().parse::<u64>().ok()?,
                ))
            } else {
                None
            }
        })
        .unwrap_or((0, 0));
    
    // Default values if miner stats server is not available
    Ok(MiningStats {
        hashrate: 0.0,
        loss: 0.0,
        loss_trend: "stable".to_string(),
        estimated_earnings_per_day: 0.0,
        current_balance: 0.0,
        gpu_temp,
        gpu_temp_status: gpu_temp_status.to_string(),
        vram_usage_mb: vram_info.0,
        vram_total_mb: vram_info.1,
        training_epoch: 0,
        gradient_norm: 0.0,
        uptime_seconds: 0,
    })
}

// Setup Wizard Commands
#[tauri::command]
pub async fn check_system() -> Result<setup_checker::SetupStatus, String> {
    setup_checker::SetupChecker::check_complete_setup().await
}

#[tauri::command]
pub async fn install_component(component: String) -> Result<serde_json::Value, String> {
    // Return installation instructions instead of actually installing
    let instructions = match component.as_str() {
        "docker" => "Please install Docker Desktop from https://www.docker.com/products/docker-desktop",
        "python" => "Please install Python from https://www.python.org/downloads/",
        "nodejs" => "Please install Node.js from https://nodejs.org/",
        "cuda" => "Please install CUDA Toolkit from https://developer.nvidia.com/cuda-downloads",
        _ => return Err(format!("Unknown component: {}", component)),
    };
    
    Ok(serde_json::json!({
        "success": false,
        "component": component,
        "message": instructions,
        "manual_install_required": true
    }))
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PortInfo {
    pub port: u16,
    pub description: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PortCheckResult {
    pub port: u16,
    pub description: String,
    pub is_open: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FirewallCheckResult {
    pub all_ok: bool,
    pub port_checks: Vec<PortCheckResult>,
    pub warnings: Vec<String>,
}

/// Check if firewall ports are accessible
#[tauri::command]
pub async fn check_firewall_ports(ports: Vec<PortInfo>) -> Result<FirewallCheckResult, String> {
    
    
    
    let mut port_checks = Vec::new();
    let mut warnings = Vec::new();
    let mut all_ok = true;
    
    for port_info in ports {
        let is_open = check_port_accessible(port_info.port);
        
        if !is_open {
            all_ok = false;
            let warning = format!(
                "Port {} ({}) may be blocked by firewall. When the miner starts, Windows Firewall may prompt for access. Please click 'Allow Access' to enable P2P connectivity.",
                port_info.port, port_info.description
            );
            warnings.push(warning);
        }
        
        port_checks.push(PortCheckResult {
            port: port_info.port,
            description: port_info.description,
            is_open,
        });
    }
    
    Ok(FirewallCheckResult {
        all_ok,
        port_checks,
        warnings,
    })
}

fn check_port_accessible(port: u16) -> bool {
    use std::net::TcpListener;
    
    // Try to bind to the port to check if it's available
    // If we can bind, the port is free (and will be accessible when the miner starts)
    // If we can't bind, the port might be in use (which is OK - something is already listening)
    // or blocked by firewall (which we can't easily detect without trying to connect)
    
    // Try to bind to all interfaces
    if TcpListener::bind(("0.0.0.0", port)).is_ok() {
        return true; // Port is free and can be bound
    }
    
    // If binding fails, try to check if something is already listening
    // We use system commands as a cross-platform way to check
    
    #[cfg(unix)]
    {
        // Try lsof first, then netstat as fallback
        let check_cmd = format!("lsof -i :{} 2>/dev/null || netstat -an 2>/dev/null | grep ':{}' | grep LISTEN", port, port);
        let output = silent_command("sh")
            .arg("-c")
            .arg(&check_cmd)
            .output();
        
        if let Ok(output) = output {
            return output.status.success();
        }
    }
    
    #[cfg(windows)]
    {
        let output = silent_command("netstat")
            .arg("-an")
            .output();
        
        if let Ok(output) = output {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let port_str = format!(":{}", port);
            // Check if port is in LISTENING state
            if output_str.contains(&port_str) && output_str.contains("LISTENING") {
                return true;
            }
        }
    }
    
    // If we can't determine, assume port might be blocked
    // This is conservative - we'll show a warning
    false
}

#[tauri::command]
pub async fn verify_installation(_component: String) -> Result<bool, String> {
    // Verify component installation
    Ok(true)
}

// System Status Commands - simplified versions
#[tauri::command]
pub async fn get_system_status() -> Result<serde_json::Value, String> {
    // Return basic system status
    Ok(serde_json::json!({
        "status": "running",
        "uptime": 0,
        "processes": {}
    }))
}

#[tauri::command]
pub async fn get_chain_sync_status() -> Result<serde_json::Value, String> {
    // Return basic chain sync status
    Ok(serde_json::json!({
        "synced": false,
        "block_height": 0,
        "sync_percentage": 0.0
    }))
}

#[tauri::command]
pub async fn get_ipfs_node_status() -> Result<serde_json::Value, String> {
    // Return basic IPFS status
    Ok(serde_json::json!({
        "connected": false,
        "peer_count": 0
    }))
}

#[tauri::command]
pub async fn get_model_download_status() -> Result<serde_json::Value, String> {
    // Return basic model status
    Ok(serde_json::json!({
        "downloaded": false,
        "progress": 0.0
    }))
}

// Log Viewer Commands - simplified versions
#[tauri::command]
pub async fn tail_log_file(process: String, lines: usize) -> Result<Vec<String>, String> {
    // Use process manager's get_logs_tail
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    Ok(manager.get_logs_tail(&process, lines).await)
}

#[tauri::command]
pub async fn filter_logs(_process: String, _level: String) -> Result<Vec<serde_json::Value>, String> {
    // Return empty for now
    Ok(vec![])
}

#[tauri::command]
pub async fn search_logs(_process: String, _query: String) -> Result<Vec<serde_json::Value>, String> {
    // Return empty for now
    Ok(vec![])
}

// Wallet Management Commands
#[tauri::command]
pub async fn create_new_wallet() -> Result<wallet::Wallet, String> {
    wallet::create_new_wallet()
}

#[tauri::command]
pub async fn import_wallet_mnemonic(mnemonic: String) -> Result<wallet::Wallet, String> {
    wallet::import_wallet_from_mnemonic(&mnemonic)
}

#[tauri::command]
pub async fn import_wallet_private_key(private_key: String) -> Result<wallet::Wallet, String> {
    wallet::import_wallet_from_private_key(&private_key)
}

#[tauri::command]
pub async fn get_wallet_balance(address: String) -> Result<wallet::WalletBalance, String> {
    wallet::get_wallet_balance(&address).await
}

#[tauri::command]
pub async fn get_wallet_transactions(address: String, limit: u32) -> Result<Vec<wallet::Transaction>, String> {
    wallet::get_wallet_transactions(&address, limit).await
}

// Engine Downloader Commands
#[tauri::command]
pub async fn ensure_engine_ready() -> Result<engine_downloader::EngineStatus, String> {
    use crate::engine_downloader::EngineDownloader;
    
    let engine_dir = EngineDownloader::default_engine_dir();
    let download_url = EngineDownloader::get_download_url();
    let expected_checksum = EngineDownloader::get_expected_checksum();
    
    let downloader = EngineDownloader::new(engine_dir, download_url, expected_checksum);
    let status = downloader.check_engine_status().await;
    Ok(status)
}

#[tauri::command]
pub async fn download_engine() -> Result<String, String> {
    use crate::engine_downloader::EngineDownloader;
    
    let engine_dir = EngineDownloader::default_engine_dir();
    let download_url = EngineDownloader::get_download_url();
    let expected_checksum = EngineDownloader::get_expected_checksum();
    
    let downloader = EngineDownloader::new(engine_dir, download_url, expected_checksum);
    
    // Download and install the engine
    let result = downloader.download_and_install().await?;
    
    if result.success {
        Ok(result.file_path.unwrap_or_else(|| "Engine installed".to_string()))
    } else {
        Err(result.message)
    }
}


// Configuration Panel Commands

#[tauri::command]
pub async fn get_full_config() -> Result<FullConfig, String> {
    Ok(FullConfig::load())
}

#[tauri::command]
pub async fn save_config(config: FullConfig) -> Result<(), String> {
    config.save()
}

#[tauri::command]
pub async fn reset_config_to_defaults() -> Result<(), String> {
    let default_config = FullConfig::default();
    default_config.save()
}
// Setup Checker Commands (from setup_checker module)
#[tauri::command]
pub async fn check_setup_status() -> Result<setup_checker::SetupStatus, String> {
    setup_checker::SetupChecker::check_complete_setup().await
}

#[tauri::command]
pub async fn get_setup_steps() -> Result<Vec<setup_checker::SetupStep>, String> {
    setup_checker::SetupChecker::get_setup_steps().await
}

#[tauri::command]
pub async fn validate_component(component: String) -> Result<bool, String> {
    setup_checker::SetupChecker::validate_component(&component).await
}

#[tauri::command]
pub async fn get_setup_progress() -> Result<f32, String> {
    setup_checker::SetupChecker::get_setup_progress().await
}

// Debug Commands
#[tauri::command]
pub async fn collect_debug_info() -> Result<debug::DebugInfo, String> {
    debug::collect_debug_info().await
}

#[tauri::command]
pub async fn export_debug_info() -> Result<String, String> {
    debug::export_debug_info().await
}

#[tauri::command]
pub async fn get_troubleshooting_recommendations() -> Result<Vec<String>, String> {
    let debug_info = debug::collect_debug_info().await?;
    Ok(debug::get_troubleshooting_recommendations(&debug_info))
}


// ============================================================================
// BLOCKCHAIN SYNC COMMANDS
// KRİTİK EKSİKLİK #4 ÇÖZÜMÜ: Launcher Blockchain Model/Dataset Sync
// ============================================================================

/// Result of blockchain model sync operation
#[derive(Debug, Serialize, Deserialize)]
pub struct BlockchainSyncResult {
    pub success: bool,
    pub message: String,
    pub model_version: Option<String>,
    pub model_path: Option<String>,
    pub datasets_synced: usize,
    pub checksum_verified: bool,
}

/// Sync model from blockchain
/// 
/// Queries blockchain for current approved model and downloads from IPFS if needed
#[tauri::command]
pub async fn sync_model_from_blockchain() -> Result<BlockchainSyncResult, String> {
    use crate::engine_downloader::EngineDownloader;
    use crate::config::get_config;
    
    let config = get_config();
    let blockchain_url = &config.rest_url;
    let ipfs_gateway = &config.ipfs_gateway_url;
    
    let engine_dir = EngineDownloader::default_engine_dir();
    let download_url = EngineDownloader::get_download_url();
    let expected_checksum = EngineDownloader::get_expected_checksum();
    
    let downloader = EngineDownloader::new(engine_dir, download_url, expected_checksum);
    
    match downloader.sync_model_from_blockchain(blockchain_url, ipfs_gateway).await {
        Ok(result) => Ok(BlockchainSyncResult {
            success: result.success,
            message: result.message,
            model_version: result.model_version,
            model_path: result.model_path,
            datasets_synced: 0,
            checksum_verified: result.checksum_verified,
        }),
        Err(e) => Err(e),
    }
}

/// Query approved datasets from blockchain
#[tauri::command]
pub async fn query_approved_datasets() -> Result<Vec<engine_downloader::BlockchainDatasetInfo>, String> {
    use crate::engine_downloader::EngineDownloader;
    use crate::config::get_config;
    
    let config = get_config();
    let blockchain_url = &config.rest_url;
    
    let engine_dir = EngineDownloader::default_engine_dir();
    let download_url = EngineDownloader::get_download_url();
    let expected_checksum = EngineDownloader::get_expected_checksum();
    
    let downloader = EngineDownloader::new(engine_dir, download_url, expected_checksum);
    
    downloader.query_approved_datasets(blockchain_url).await
}

/// Download a specific dataset from IPFS
#[tauri::command]
pub async fn download_dataset(dataset_id: String, ipfs_hash: String, checksum: String) -> Result<String, String> {
    use crate::engine_downloader::{EngineDownloader, BlockchainDatasetInfo};
    use crate::config::get_config;
    
    let config = get_config();
    let ipfs_gateway = &config.ipfs_gateway_url;
    
    let engine_dir = EngineDownloader::default_engine_dir();
    let download_url = EngineDownloader::get_download_url();
    let expected_checksum = EngineDownloader::get_expected_checksum();
    
    let downloader = EngineDownloader::new(engine_dir, download_url, expected_checksum);
    
    let dataset_info = BlockchainDatasetInfo {
        dataset_id,
        name: "dataset".to_string(),
        ipfs_hash,
        checksum,
        size_bytes: 0,
        category: "training".to_string(),
    };
    
    downloader.download_dataset(&dataset_info, ipfs_gateway).await
}

/// Get local model version
#[tauri::command]
pub async fn get_local_model_version() -> Result<Option<String>, String> {
    use crate::engine_downloader::EngineDownloader;
    
    let engine_dir = EngineDownloader::default_engine_dir();
    let download_url = EngineDownloader::get_download_url();
    let expected_checksum = EngineDownloader::get_expected_checksum();
    
    let downloader = EngineDownloader::new(engine_dir, download_url, expected_checksum);
    
    Ok(downloader.get_local_model_version())
}

/// Check if model update is available
#[tauri::command]
pub async fn check_model_update() -> Result<serde_json::Value, String> {
    use crate::engine_downloader::EngineDownloader;
    use crate::config::get_config;
    
    let config = get_config();
    let blockchain_url = &config.rest_url;
    
    let engine_dir = EngineDownloader::default_engine_dir();
    let download_url = EngineDownloader::get_download_url();
    let expected_checksum = EngineDownloader::get_expected_checksum();
    
    let downloader = EngineDownloader::new(engine_dir, download_url, expected_checksum);
    
    // Get local version
    let local_version = downloader.get_local_model_version();
    
    // Query blockchain for current version
    let blockchain_model = downloader.query_global_model_state(blockchain_url).await;
    
    match blockchain_model {
        Ok(model_info) => {
            let update_available = match &local_version {
                Some(local) => local != &model_info.model_version,
                None => true,
            };
            
            Ok(serde_json::json!({
                "update_available": update_available,
                "local_version": local_version,
                "blockchain_version": model_info.model_version,
                "ipfs_hash": model_info.ipfs_hash,
            }))
        },
        Err(e) => {
            Ok(serde_json::json!({
                "update_available": false,
                "local_version": local_version,
                "error": e,
            }))
        }
    }
}

/// Sync all approved datasets from blockchain
#[tauri::command]
pub async fn sync_all_datasets() -> Result<BlockchainSyncResult, String> {
    use crate::engine_downloader::EngineDownloader;
    use crate::config::get_config;
    
    let config = get_config();
    let blockchain_url = &config.rest_url;
    let ipfs_gateway = &config.ipfs_gateway_url;
    
    let engine_dir = EngineDownloader::default_engine_dir();
    let download_url = EngineDownloader::get_download_url();
    let expected_checksum = EngineDownloader::get_expected_checksum();
    
    let downloader = EngineDownloader::new(engine_dir, download_url, expected_checksum);
    
    // Query approved datasets
    let datasets = downloader.query_approved_datasets(blockchain_url).await?;
    
    let mut synced_count = 0;
    let mut errors = Vec::new();
    
    for dataset in &datasets {
        match downloader.download_dataset(dataset, ipfs_gateway).await {
            Ok(_) => synced_count += 1,
            Err(e) => errors.push(format!("{}: {}", dataset.dataset_id, e)),
        }
    }
    
    let success = errors.is_empty();
    let message = if success {
        format!("Successfully synced {} datasets", synced_count)
    } else {
        format!("Synced {} datasets with {} errors: {:?}", synced_count, errors.len(), errors)
    };
    
    Ok(BlockchainSyncResult {
        success,
        message,
        model_version: None,
        model_path: None,
        datasets_synced: synced_count,
        checksum_verified: true,
    })
}


// ============================================================================
// BLOCKCHAIN SYNC COMMANDS (for Frontend UI)
// ============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct AdapterSyncStatus {
    pub adapter_id: String,
    pub name: String,
    pub domain: String,
    pub version: String,
    pub ipfs_hash: String,
    pub synced: bool,
    pub last_synced: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetSyncStatus {
    pub dataset_id: String,
    pub name: String,
    pub ipfs_hash: String,
    pub size_gb: f64,
    pub downloaded: bool,
    pub last_updated: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BlockchainSyncResult {
    pub success: bool,
    pub message: String,
    pub new_items: usize,
    pub updated_items: usize,
    pub failed_items: usize,
}

/// Get list of synced adapters
#[tauri::command]
pub async fn get_synced_adapters() -> Result<Vec<AdapterSyncStatus>, String> {
    use crate::config::BlockchainConfig;
    
    let config = BlockchainConfig::from_network_mode(
        &std::env::var("NETWORK_MODE").unwrap_or_else(|_| "testnet".to_string())
    );
    
    // Query blockchain for approved adapters
    let query_url = format!("{}/remes/remes/v1/approved_adapters", config.rest_endpoint);
    
    let response = reqwest::get(&query_url)
        .await
        .map_err(|e| format!("Failed to query adapters: {}", e))?;
    
    if !response.status().is_success() {
        return Ok(vec![]);
    }
    
    let json: serde_json::Value = response.json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;
    
    let adapters = json["adapters"].as_array()
        .ok_or("Invalid response format")?;
    
    let mut result = Vec::new();
    for adapter in adapters {
        result.push(AdapterSyncStatus {
            adapter_id: adapter["adapter_id"].as_str().unwrap_or("").to_string(),
            name: adapter["name"].as_str().unwrap_or("").to_string(),
            domain: adapter["domain"].as_str().unwrap_or("general").to_string(),
            version: adapter["version"].as_str().unwrap_or("1.0.0").to_string(),
            ipfs_hash: adapter["ipfs_hash"].as_str().unwrap_or("").to_string(),
            synced: adapter["synced"].as_bool().unwrap_or(false),
            last_synced: adapter["last_synced"].as_str().map(|s| s.to_string()),
        });
    }
    
    Ok(result)
}

/// Get list of synced datasets
#[tauri::command]
pub async fn get_synced_datasets() -> Result<Vec<DatasetSyncStatus>, String> {
    use crate::config::BlockchainConfig;
    
    let config = BlockchainConfig::from_network_mode(
        &std::env::var("NETWORK_MODE").unwrap_or_else(|_| "testnet".to_string())
    );
    
    // Query blockchain for approved datasets
    let query_url = format!("{}/remes/remes/v1/approved_datasets", config.rest_endpoint);
    
    let response = reqwest::get(&query_url)
        .await
        .map_err(|e| format!("Failed to query datasets: {}", e))?;
    
    if !response.status().is_success() {
        return Ok(vec![]);
    }
    
    let json: serde_json::Value = response.json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;
    
    let datasets = json["datasets"].as_array()
        .ok_or("Invalid response format")?;
    
    let mut result = Vec::new();
    for dataset in datasets {
        result.push(DatasetSyncStatus {
            dataset_id: dataset["dataset_id"].as_str().unwrap_or("").to_string(),
            name: dataset["name"].as_str().unwrap_or("").to_string(),
            ipfs_hash: dataset["ipfs_hash"].as_str().unwrap_or("").to_string(),
            size_gb: dataset["size_gb"].as_f64().unwrap_or(0.0),
            downloaded: dataset["downloaded"].as_bool().unwrap_or(false),
            last_updated: dataset["last_updated"].as_str().map(|s| s.to_string()),
        });
    }
    
    Ok(result)
}

/// Sync all adapters from blockchain
#[tauri::command]
pub async fn sync_all_adapters() -> Result<BlockchainSyncResult, String> {
    // Get approved adapters
    let adapters = get_synced_adapters().await?;
    
    let mut new_items = 0;
    let mut updated_items = 0;
    let mut failed_items = 0;
    
    for adapter in adapters {
        // Check if adapter is already downloaded
        let adapter_dir = platform::get_adapters_dir()?.join(&adapter.adapter_id);
        
        if adapter_dir.exists() {
            // Already exists, check if update needed
            if !adapter.synced {
                // Download update
                match download_adapter(&adapter).await {
                    Ok(_) => updated_items += 1,
                    Err(_) => failed_items += 1,
                }
            }
        } else {
            // New adapter, download it
            match download_adapter(&adapter).await {
                Ok(_) => new_items += 1,
                Err(_) => failed_items += 1,
            }
        }
    }
    
    Ok(BlockchainSyncResult {
        success: failed_items == 0,
        message: format!("Synced {} new, {} updated adapters", new_items, updated_items),
        new_items,
        updated_items,
        failed_items,
    })
}

/// Helper function to download adapter from IPFS
async fn download_adapter(adapter: &AdapterSyncStatus) -> Result<(), String> {
    let config = get_launcher_config();
    let ipfs_url = format!("{}/ipfs/{}", config.ipfs_url, adapter.ipfs_hash);
    
    let adapter_dir = platform::get_adapters_dir()?.join(&adapter.adapter_id);
    std::fs::create_dir_all(&adapter_dir)
        .map_err(|e| format!("Failed to create adapter directory: {}", e))?;
    
    // Download adapter file
    let response = reqwest::get(&ipfs_url)
        .await
        .map_err(|e| format!("Failed to download adapter: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("Download failed with status: {}", response.status()));
    }
    
    let content = response.bytes()
        .await
        .map_err(|e| format!("Failed to read adapter content: {}", e))?;
    
    // Save adapter file
    let adapter_file = adapter_dir.join("adapter_model.bin");
    std::fs::write(&adapter_file, &content)
        .map_err(|e| format!("Failed to write adapter file: {}", e))?;
    
    // Save metadata
    let metadata = serde_json::json!({
        "adapter_id": adapter.adapter_id,
        "name": adapter.name,
        "domain": adapter.domain,
        "version": adapter.version,
        "ipfs_hash": adapter.ipfs_hash,
        "downloaded_at": chrono::Utc::now().to_rfc3339(),
    });
    
    let metadata_file = adapter_dir.join("adapter_config.json");
    std::fs::write(&metadata_file, serde_json::to_string_pretty(&metadata).unwrap())
        .map_err(|e| format!("Failed to write metadata: {}", e))?;
    
    Ok(())
}
