use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::fs;
use cosmrs::crypto::secp256k1;
use cosmrs::AccountId;
use bip39::{Mnemonic, Language};

#[derive(Debug, Serialize, Deserialize)]
pub struct Wallet {
    pub address: String,
    pub mnemonic: Option<String>, // Only stored temporarily during creation
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WalletBalance {
    pub address: String,
    pub balance: String,
    pub denom: String,
}

pub fn create_new_wallet() -> Result<Wallet, String> {
    // Generate new mnemonic
    let mnemonic = Mnemonic::generate_in(Language::English, 24)
        .map_err(|e| format!("Failed to generate mnemonic: {}", e))?;
    
    let mnemonic_str = mnemonic.to_string();
    
    // Derive private key from mnemonic
    let seed = mnemonic.to_seed("");
    let private_key = secp256k1::SigningKey::from_bytes(&seed[..32])
        .map_err(|e| format!("Failed to derive private key: {}", e))?;
    
    // Get public key and address
    let public_key = private_key.public_key();
    let account_id = AccountId::new("remes", &public_key)
        .map_err(|e| format!("Failed to create account ID: {}", e))?;
    
    let address = account_id.to_string();
    
    // Store wallet securely (in production, use OS keychain)
    save_wallet(&address, &mnemonic_str)?;
    
    Ok(Wallet {
        address,
        mnemonic: Some(mnemonic_str), // Return mnemonic only once
    })
}

pub fn import_wallet_from_mnemonic(mnemonic_str: &str) -> Result<Wallet, String> {
    let mnemonic = Mnemonic::parse_in(Language::English, mnemonic_str)
        .map_err(|e| format!("Invalid mnemonic: {}", e))?;
    
    let seed = mnemonic.to_seed("");
    let private_key = secp256k1::SigningKey::from_bytes(&seed[..32])
        .map_err(|e| format!("Failed to derive private key: {}", e))?;
    
    let public_key = private_key.public_key();
    let account_id = AccountId::new("remes", &public_key)
        .map_err(|e| format!("Failed to create account ID: {}", e))?;
    
    let address = account_id.to_string();
    
    save_wallet(&address, mnemonic_str)?;
    
    Ok(Wallet {
        address,
        mnemonic: None, // Don't return mnemonic on import
    })
}

pub fn import_wallet_from_private_key(private_key_hex: &str) -> Result<Wallet, String> {
    let private_key_bytes = hex::decode(private_key_hex)
        .map_err(|e| format!("Invalid hex: {}", e))?;
    
    let private_key = secp256k1::SigningKey::from_bytes(&private_key_bytes)
        .map_err(|e| format!("Failed to create signing key: {}", e))?;
    
    let public_key = private_key.public_key();
    let account_id = AccountId::new("remes", &public_key)
        .map_err(|e| format!("Failed to create account ID: {}", e))?;
    
    let address = account_id.to_string();
    
    // Store wallet (without mnemonic for private key imports)
    save_wallet(&address, "")?;
    
    Ok(Wallet {
        address,
        mnemonic: None,
    })
}

pub async fn get_wallet_balance(address: &str) -> Result<WalletBalance, String> {
    // Query blockchain for balance using reqwest
    let rpc_url = std::env::var("BLOCKCHAIN_REST_URL")
        .unwrap_or_else(|_| "http://localhost:1317".to_string());
    let query_url = format!("{}/cosmos/bank/v1beta1/balances/{}", rpc_url, address);
    
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
    
    let response = client
        .get(&query_url)
        .send()
        .await
        .map_err(|e| format!("Failed to query balance: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("Failed to get balance: HTTP {}", response.status()));
    }
    
    let balance_data: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse balance response: {}", e))?;
    
    // Extract balance from response
    let balance = balance_data["balances"]
        .as_array()
        .and_then(|b| b.first())
        .and_then(|b| b["amount"].as_str())
        .unwrap_or("0")
        .to_string();
    
    let denom = balance_data["balances"]
        .as_array()
        .and_then(|b| b.first())
        .and_then(|b| b["denom"].as_str())
        .unwrap_or("uremes")
        .to_string();
    
    Ok(WalletBalance {
        address: address.to_string(),
        balance,
        denom,
    })
}

pub async fn get_wallet_transactions(address: &str, limit: u32) -> Result<Vec<Transaction>, String> {
    // Query blockchain for transactions
    // In production, use proper RPC client
    Ok(vec![])
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Transaction {
    pub hash: String,
    pub height: u64,
    pub timestamp: i64,
    pub from: String,
    pub to: String,
    pub amount: String,
    pub denom: String,
    pub status: String,
}

fn save_wallet(address: &str, mnemonic: &str) -> Result<(), String> {
    let home = std::env::var("HOME")
        .map_err(|_| "HOME environment variable not set".to_string())?;
    
    let wallet_dir = PathBuf::from(&home).join(".r3mes").join("wallets");
    fs::create_dir_all(&wallet_dir)
        .map_err(|e| format!("Failed to create wallet directory: {}", e))?;
    
    let wallet_file = wallet_dir.join("default_wallet.json");
    
    // In production, encrypt mnemonic before storing
    let wallet_data = serde_json::json!({
        "address": address,
        "mnemonic": mnemonic, // Should be encrypted
        "created_at": chrono::Utc::now().to_rfc3339(),
    });
    
    fs::write(&wallet_file, serde_json::to_string_pretty(&wallet_data).unwrap())
        .map_err(|e| format!("Failed to write wallet file: {}", e))?;
    
    Ok(())
}

