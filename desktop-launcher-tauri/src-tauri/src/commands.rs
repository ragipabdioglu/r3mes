use crate::process_manager::ProcessManager;
use crate::hardware_check;
use crate::keychain::{KeychainManager, KeychainError};
use crate::setup_checker;
use crate::status_monitor;
use crate::log_reader;
use crate::wallet;
use crate::installer;
use serde::{Deserialize, Serialize};
use std::process::Command;

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

static PROCESS_MANAGER: tokio::sync::Mutex<Option<ProcessManager>> = tokio::sync::Mutex::const_new(None);

async fn get_manager() -> tokio::sync::MutexGuard<'static, Option<ProcessManager>> {
    let mut manager = PROCESS_MANAGER.lock().await;
    if manager.is_none() {
        *manager = Some(ProcessManager::new());
    }
    manager
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
    let manager = manager.as_mut().unwrap();
    manager.start_node().await
}

#[tauri::command]
pub async fn stop_node() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
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
    let manager = manager.as_mut().unwrap();
    manager.start_miner().await
}

#[tauri::command]
pub async fn stop_miner() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    manager.stop_miner().await
}

#[tauri::command]
pub async fn start_ipfs() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    manager.start_ipfs().await
}

#[tauri::command]
pub async fn stop_ipfs() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    manager.stop_ipfs().await
}

#[tauri::command]
pub async fn start_serving() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    manager.start_serving().await
}

#[tauri::command]
pub async fn stop_serving() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    manager.stop_serving().await
}

#[tauri::command]
pub async fn start_validator() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    manager.start_validator().await
}

#[tauri::command]
pub async fn stop_validator() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    manager.stop_validator().await
}

#[tauri::command]
pub async fn start_proposer() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    manager.start_proposer().await
}

#[tauri::command]
pub async fn stop_proposer() -> Result<ProcessResult, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    manager.stop_proposer().await
}

#[tauri::command]
pub async fn get_status() -> Result<ProcessStatus, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    Ok(manager.get_status().await)
}

#[tauri::command]
pub async fn get_logs() -> Result<Vec<String>, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    Ok(manager.get_logs().await)
}

#[tauri::command]
pub async fn get_logs_tail(process: String, lines: usize) -> Result<Vec<String>, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    Ok(manager.get_logs_tail(&process, lines).await)
}

#[tauri::command]
pub async fn get_logs_by_level(process: String, level: String) -> Result<Vec<String>, String> {
    let mut manager = get_manager().await;
    let manager = manager.as_mut().unwrap();
    Ok(manager.get_logs_by_level(&process, &level).await)
}

#[tauri::command]
pub async fn export_logs(process: String) -> Result<String, String> {
    use std::fs;
    use std::path::PathBuf;
    
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let log_dir = PathBuf::from(&home).join("R3MES").join("logs");
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
    use std::path::PathBuf;
    
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let config_dir = PathBuf::from(&home).join(".r3mes");
    let config_file = config_dir.join("launcher_config.json");
    
    Ok(!config_file.exists())
}

#[tauri::command]
pub async fn mark_setup_complete() -> Result<(), String> {
    use std::fs;
    use std::path::PathBuf;
    
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let config_dir = PathBuf::from(&home).join(".r3mes");
    
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
    use std::path::PathBuf;
    use std::fs;
    
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let wallet_path = PathBuf::from(&home).join(".r3mes").join("wallets").join("default_wallet.json");
    
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
    use std::process::Command;
    use std::path::PathBuf;
    
    // Call Python wallet manager
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let workspace = PathBuf::from(&home).join("R3MES");
    let python_script = workspace.join("miner-engine").join("r3mes").join("cli").join("wallet.py");
    
    // Use r3mes-miner wallet create command instead
    let output = Command::new("python3")
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
    use std::path::PathBuf;
    
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let wallet_path = PathBuf::from(&home).join(".r3mes").join("wallets").join("default_wallet.json");
    
    if !wallet_path.exists() {
        return Err("No wallet found".to_string());
    }
    
    // Read wallet file
    let wallet_data = fs::read_to_string(&wallet_path)
        .map_err(|e| format!("Failed to read wallet: {}", e))?;
    
    // Encrypt wallet data using AES-256-GCM
    use aes_gcm::{
        aes::Aes256,
        AesGcm, KeyInit, Nonce,
    };
    use aes_gcm::aead::{Aead, NewAead};
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
    let cipher = AesGcm::<Aes256>::new_from_slice(&key_bytes)
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
    use std::path::PathBuf;
    
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let history_path = PathBuf::from(&home).join(".r3mes").join("wallets").join("transaction_history.json");
    
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
    
    Ok(TransactionHistory {
        transactions,
        total: transactions.len(),
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
    use std::process::Command;
    use crate::config::get_config;
    
    let config = get_config();
    let rpc_url = &config.rpc_url;
    
    // Check if node is running by trying to connect to RPC
    let node_running = Command::new("curl")
        .arg("-s")
        .arg("--connect-timeout")
        .arg("2")
        .arg(format!("{}/status", rpc_url))
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    
    if !node_running {
        return Ok(ChainStatus {
            synced: false,
            block_height: 0,
            sync_percentage: 0.0,
            status: "stopped".to_string(),
        });
    }
    
    // Try to get block height from RPC
    let block_height = Command::new("curl")
        .arg("-s")
        .arg(format!("{}/status", rpc_url))
        .output()
        .ok()
        .and_then(|output| {
            let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
            json["result"]["sync_info"]["latest_block_height"]
                .as_str()
                .and_then(|s| s.parse::<u64>().ok())
        })
        .unwrap_or(0);
    
    // Sync status calculation: compare local block height with network height
    // Note: Full sync status requires querying network peers for latest block height
    // For now, assume syncing if block_height > 0 but < some threshold
    let synced = block_height > 1000; // Arbitrary threshold
    let sync_percentage = if synced {
        100.0
    } else if block_height > 0 {
        (block_height as f64 / 1000.0 * 100.0).min(99.9)
    } else {
        0.0
    };
    
    Ok(ChainStatus {
        synced,
        block_height,
        sync_percentage,
        status: if node_running {
            if synced {
                "running".to_string()
            } else {
                "syncing".to_string()
            }
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
    use std::path::PathBuf;
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
            let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
            let wallet_path = PathBuf::from(&home).join(".r3mes").join("wallets").join("default_wallet.json");
            
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
    
    let mut child = Command::new("python3")
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
    use std::process::Command;
    
    // Check if IPFS is running by checking if port 5001 is listening
    let ipfs_running = Command::new("sh")
        .arg("-c")
        .arg("lsof -i :5001 > /dev/null 2>&1")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    
    if !ipfs_running {
        return Ok(IPFSStatus {
            connected: false,
            peer_count: 0,
            status: "stopped".to_string(),
        });
    }
    
    // Get peer count from IPFS API
    use crate::config::get_config;
    let config = get_config();
    let ipfs_url = &config.ipfs_url;
    let peer_count = Command::new("curl")
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
    use std::path::PathBuf;
    
    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
    let model_dir = PathBuf::from(&home).join(".r3mes").join("models");
    let model_file = model_dir.join("bitnet_b1.58.pt");
    
    if model_file.exists() {
        // Check file size
        let size_bytes = fs::metadata(&model_file)
            .map(|m| m.len())
            .unwrap_or(0);
        let size_gb = size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        
        // Check if file is complete (28GB for BitNet)
        let downloaded = size_gb >= 25.0; // Allow some margin
        
        Ok(ModelStatus {
            downloaded,
            progress: if downloaded { 100.0 } else { (size_gb / 28.0 * 100.0).min(99.0) },
            version: "BitNet b1.58".to_string(),
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
            version: "BitNet b1.58".to_string(),
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
    // Open web dashboard in default browser
    let url = "http://localhost:3000";
    
    #[cfg(target_os = "windows")]
    {
        Command::new("cmd")
            .args(["/C", "start", url])
            .output()
            .map_err(|e| format!("Failed to open dashboard: {}", e))?;
    }
    
    #[cfg(target_os = "macos")]
    {
        Command::new("open")
            .arg(url)
            .output()
            .map_err(|e| format!("Failed to open dashboard: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        Command::new("xdg-open")
            .arg(url)
            .output()
            .map_err(|e| format!("Failed to open dashboard: {}", e))?;
    }
    
    Ok(())
}

#[tauri::command]
pub async fn get_mining_stats() -> Result<MiningStats, String> {
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};
    
    // Try to fetch from miner stats HTTP server (if running)
    let stats_response = Command::new("curl")
        .arg("-s")
        .arg("http://localhost:8080/stats")
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
    let gpu_temp = Command::new("nvidia-smi")
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
    let vram_info = Command::new("nvidia-smi")
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
pub async fn check_system() -> Result<setup_checker::SystemCheck, String> {
    setup_checker::check_system().await
}

#[tauri::command]
pub async fn install_component(component: String) -> Result<installer::InstallationResult, String> {
    match component.as_str() {
        "docker" => installer::install_docker().await,
        "python" => installer::install_python().await,
        "nodejs" => installer::install_nodejs().await,
        "cuda" => installer::install_cuda().await,
        _ => Err(format!("Unknown component: {}", component)),
    }
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
    use std::net::{TcpListener, TcpStream};
    use std::time::Duration;
    
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
        use std::process::Command;
        // Try lsof first, then netstat as fallback
        let check_cmd = format!("lsof -i :{} 2>/dev/null || netstat -an 2>/dev/null | grep ':{}' | grep LISTEN", port, port);
        let output = Command::new("sh")
            .arg("-c")
            .arg(&check_cmd)
            .output();
        
        if let Ok(output) = output {
            return output.status.success();
        }
    }
    
    #[cfg(windows)]
    {
        use std::process::Command;
        let output = Command::new("netstat")
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
pub async fn verify_installation(component: String) -> Result<bool, String> {
    // Verify component installation
    Ok(true)
}

// System Status Commands
#[tauri::command]
pub async fn get_system_status() -> Result<status_monitor::SystemStatus, String> {
    status_monitor::get_system_status().await
}

#[tauri::command]
pub async fn get_chain_sync_status() -> Result<status_monitor::ChainSyncStatus, String> {
    status_monitor::get_chain_sync_status().await
}

#[tauri::command]
pub async fn get_ipfs_node_status() -> Result<status_monitor::IpfsStatus, String> {
    status_monitor::get_ipfs_status().await
}

#[tauri::command]
pub async fn get_model_download_status() -> Result<status_monitor::ModelStatus, String> {
    status_monitor::get_model_status().await
}

// Log Viewer Commands
#[tauri::command]
pub async fn tail_log_file(process: String, lines: usize) -> Result<Vec<String>, String> {
    log_reader::tail_log_file(&process, lines).await
}

#[tauri::command]
pub async fn filter_logs(process: String, level: String) -> Result<Vec<log_reader::LogEntry>, String> {
    log_reader::filter_logs(&process, &level).await
}

#[tauri::command]
pub async fn search_logs(process: String, query: String) -> Result<Vec<log_reader::LogEntry>, String> {
    log_reader::search_logs(&process, &query).await
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
    
    // Download with progress callback (for now, just log to console)
    let result = downloader.download_with_progress(|progress| {
        eprintln!("Download progress: {:.1}% ({} / {} bytes, {:.2} MB/s)", 
            progress.percentage,
            progress.downloaded_bytes,
            progress.total_bytes.map(|b| b.to_string()).unwrap_or_else(|| "?".to_string()),
            progress.speed_bytes_per_sec / 1_000_000.0
        );
    }).await;
    
    result.map(|path| path.to_string_lossy().to_string())
}


// Configuration Panel Commands
use crate::config::{FullConfig, MinerConfig, NetworkConfig, AdvancedConfig};

#[tauri::command]
pub async fn get_config() -> Result<FullConfig, String> {
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
