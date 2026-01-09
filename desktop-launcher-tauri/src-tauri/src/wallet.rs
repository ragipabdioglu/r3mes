//! Wallet management for R3MES Desktop Launcher

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::fs;
use cosmrs::crypto::secp256k1;
use cosmrs::AccountId;
use bip39::{Mnemonic, Language};
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::Aead;
use rand::RngCore;

use crate::platform;
use crate::keychain::KeychainManager;

#[derive(Debug, Serialize, Deserialize)]
pub struct Wallet {
    pub address: String,
    pub mnemonic: Option<String>, // Only returned temporarily during creation, never stored in plain text
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WalletBalance {
    pub address: String,
    pub balance: String,
    pub denom: String,
}

/// Encrypted wallet data structure stored on disk
#[derive(Debug, Serialize, Deserialize)]
struct EncryptedWalletData {
    pub address: String,
    pub encrypted_mnemonic: String,  // Hex-encoded encrypted mnemonic
    pub nonce: String,               // Hex-encoded nonce for AES-GCM
    pub created_at: String,
    pub version: u32,                // Schema version for future migrations
}

/// Get or create the wallet encryption key from keychain
fn get_wallet_encryption_key() -> Result<[u8; 32], String> {
    let keychain = KeychainManager::new();
    
    // Try to retrieve existing key
    if let Ok(key_hex) = keychain.retrieve("wallet_master_key") {
        let key_bytes = hex::decode(&key_hex)
            .map_err(|e| format!("Invalid encryption key format: {}", e))?;
        
        if key_bytes.len() == 32 {
            let mut key = [0u8; 32];
            key.copy_from_slice(&key_bytes);
            return Ok(key);
        }
    }
    
    // Generate new key and store it
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    
    let key_hex = hex::encode(&key);
    keychain.store("wallet_master_key", &key_hex)
        .map_err(|e| format!("Failed to store encryption key: {}", e))?;
    
    Ok(key)
}

/// Encrypt mnemonic using AES-256-GCM
fn encrypt_mnemonic(mnemonic: &str) -> Result<(String, String), String> {
    let key = get_wallet_encryption_key()?;
    
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| format!("Failed to create cipher: {}", e))?;
    
    // Generate random nonce
    let mut nonce_bytes = [0u8; 12];
    rand::thread_rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    
    // Encrypt
    let ciphertext = cipher.encrypt(nonce, mnemonic.as_bytes())
        .map_err(|e| format!("Encryption failed: {}", e))?;
    
    Ok((hex::encode(ciphertext), hex::encode(nonce_bytes)))
}

/// Decrypt mnemonic using AES-256-GCM
fn decrypt_mnemonic(encrypted_hex: &str, nonce_hex: &str) -> Result<String, String> {
    let key = get_wallet_encryption_key()?;
    
    let cipher = Aes256Gcm::new_from_slice(&key)
        .map_err(|e| format!("Failed to create cipher: {}", e))?;
    
    let ciphertext = hex::decode(encrypted_hex)
        .map_err(|e| format!("Invalid encrypted data: {}", e))?;
    
    let nonce_bytes = hex::decode(nonce_hex)
        .map_err(|e| format!("Invalid nonce: {}", e))?;
    
    if nonce_bytes.len() != 12 {
        return Err("Invalid nonce length".to_string());
    }
    
    let nonce = Nonce::from_slice(&nonce_bytes);
    
    let plaintext = cipher.decrypt(nonce, ciphertext.as_ref())
        .map_err(|e| format!("Decryption failed: {}", e))?;
    
    String::from_utf8(plaintext)
        .map_err(|e| format!("Invalid UTF-8 in decrypted data: {}", e))
}

pub fn create_new_wallet() -> Result<Wallet, String> {
    // Generate new mnemonic
    let mnemonic = Mnemonic::generate_in(Language::English, 24)
        .map_err(|e| format!("Failed to generate mnemonic: {}", e))?;
    
    let mnemonic_str = mnemonic.to_string();
    
    // Derive private key from mnemonic
    let seed = mnemonic.to_seed("");
    let private_key = secp256k1::SigningKey::from_slice(&seed[..32])
        .map_err(|e| format!("Failed to derive private key: {}", e))?;
    
    // Get public key and address
    let public_key = private_key.public_key();
    
    // Get the raw public key bytes for AccountId
    let public_key_bytes = public_key.to_bytes();
    let account_id = AccountId::new("remes", &public_key_bytes)
        .map_err(|e| format!("Failed to create account ID: {}", e))?;
    
    let address = account_id.to_string();
    
    // Store wallet securely with encrypted mnemonic
    save_wallet_encrypted(&address, &mnemonic_str)?;
    
    // Also store private key in keychain for quick access
    // Use k256 to get the raw bytes
    let keychain = KeychainManager::new();
    let private_key_hex = hex::encode(&seed[..32]);
    if let Err(e) = keychain.store_wallet_private_key(&address, &private_key_hex) {
        eprintln!("⚠️  Warning: Could not store private key in keychain: {}", e);
        // Continue anyway - mnemonic is stored encrypted
    }
    
    Ok(Wallet {
        address,
        mnemonic: Some(mnemonic_str), // Return mnemonic only once for user to backup
    })
}

pub fn import_wallet_from_mnemonic(mnemonic_str: &str) -> Result<Wallet, String> {
    let mnemonic = Mnemonic::parse_in(Language::English, mnemonic_str)
        .map_err(|e| format!("Invalid mnemonic: {}", e))?;
    
    let seed = mnemonic.to_seed("");
    let private_key = secp256k1::SigningKey::from_slice(&seed[..32])
        .map_err(|e| format!("Failed to derive private key: {}", e))?;
    
    let public_key = private_key.public_key();
    let public_key_bytes = public_key.to_bytes();
    let account_id = AccountId::new("remes", &public_key_bytes)
        .map_err(|e| format!("Failed to create account ID: {}", e))?;
    
    let address = account_id.to_string();
    
    // Store wallet securely with encrypted mnemonic
    save_wallet_encrypted(&address, mnemonic_str)?;
    
    // Also store private key in keychain
    let keychain = KeychainManager::new();
    let private_key_hex = hex::encode(&seed[..32]);
    if let Err(e) = keychain.store_wallet_private_key(&address, &private_key_hex) {
        eprintln!("⚠️  Warning: Could not store private key in keychain: {}", e);
    }
    
    Ok(Wallet {
        address,
        mnemonic: None, // Don't return mnemonic on import
    })
}

pub fn import_wallet_from_private_key(private_key_hex: &str) -> Result<Wallet, String> {
    let private_key_bytes = hex::decode(private_key_hex)
        .map_err(|e| format!("Invalid hex: {}", e))?;
    
    let private_key = secp256k1::SigningKey::from_slice(&private_key_bytes)
        .map_err(|e| format!("Failed to create signing key: {}", e))?;
    
    let public_key = private_key.public_key();
    let public_key_bytes = public_key.to_bytes();
    let account_id = AccountId::new("remes", &public_key_bytes)
        .map_err(|e| format!("Failed to create account ID: {}", e))?;
    
    let address = account_id.to_string();
    
    // Store wallet (without mnemonic for private key imports)
    save_wallet_encrypted(&address, "")?;
    
    // Store private key in keychain
    let keychain = KeychainManager::new();
    if let Err(e) = keychain.store_wallet_private_key(&address, private_key_hex) {
        eprintln!("⚠️  Warning: Could not store private key in keychain: {}", e);
    }
    
    Ok(Wallet {
        address,
        mnemonic: None,
    })
}

pub async fn get_wallet_balance(address: &str) -> Result<WalletBalance, String> {
    // Get REST URL from config
    let config = crate::config::get_config();
    let rest_url = &config.rest_url;
    let query_url = format!("{}/cosmos/bank/v1beta1/balances/{}", rest_url, address);
    
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

pub async fn get_wallet_transactions(_address: &str, _limit: u32) -> Result<Vec<Transaction>, String> {
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

/// Save wallet with encrypted mnemonic
fn save_wallet_encrypted(address: &str, mnemonic: &str) -> Result<(), String> {
    let wallet_dir = platform::get_wallets_dir()?;
    fs::create_dir_all(&wallet_dir)
        .map_err(|e| format!("Failed to create wallet directory: {}", e))?;
    
    let wallet_file = wallet_dir.join("default_wallet.json");
    
    // Encrypt mnemonic if provided
    let (encrypted_mnemonic, nonce) = if !mnemonic.is_empty() {
        encrypt_mnemonic(mnemonic)?
    } else {
        (String::new(), String::new())
    };
    
    let wallet_data = EncryptedWalletData {
        address: address.to_string(),
        encrypted_mnemonic,
        nonce,
        created_at: chrono::Utc::now().to_rfc3339(),
        version: 1,
    };
    
    let json = serde_json::to_string_pretty(&wallet_data)
        .map_err(|e| format!("Failed to serialize wallet: {}", e))?;
    
    fs::write(&wallet_file, json)
        .map_err(|e| format!("Failed to write wallet file: {}", e))?;
    
    // Set restrictive permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(&wallet_file)
            .map_err(|e| format!("Failed to get file metadata: {}", e))?
            .permissions();
        perms.set_mode(0o600); // Read/write for owner only
        fs::set_permissions(&wallet_file, perms)
            .map_err(|e| format!("Failed to set file permissions: {}", e))?;
    }
    
    Ok(())
}

/// Load and decrypt wallet mnemonic (for recovery purposes only)
pub fn load_wallet_mnemonic(address: &str) -> Result<String, String> {
    let wallet_dir = platform::get_wallets_dir()?;
    let wallet_file = wallet_dir.join("default_wallet.json");
    
    if !wallet_file.exists() {
        return Err("Wallet file not found".to_string());
    }
    
    let content = fs::read_to_string(&wallet_file)
        .map_err(|e| format!("Failed to read wallet file: {}", e))?;
    
    let wallet_data: EncryptedWalletData = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse wallet file: {}", e))?;
    
    if wallet_data.address != address {
        return Err("Address mismatch".to_string());
    }
    
    if wallet_data.encrypted_mnemonic.is_empty() {
        return Err("No mnemonic stored (wallet was imported from private key)".to_string());
    }
    
    decrypt_mnemonic(&wallet_data.encrypted_mnemonic, &wallet_data.nonce)
}

/// Get wallet address from stored wallet file
pub fn get_stored_wallet_address() -> Result<String, String> {
    let wallet_dir = platform::get_wallets_dir()?;
    let wallet_file = wallet_dir.join("default_wallet.json");
    
    if !wallet_file.exists() {
        return Err("No wallet found".to_string());
    }
    
    let content = fs::read_to_string(&wallet_file)
        .map_err(|e| format!("Failed to read wallet file: {}", e))?;
    
    // Try new encrypted format first
    if let Ok(wallet_data) = serde_json::from_str::<EncryptedWalletData>(&content) {
        return Ok(wallet_data.address);
    }
    
    // Fallback to old format for migration
    let wallet_json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse wallet file: {}", e))?;
    
    wallet_json["address"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "Invalid wallet format".to_string())
}

/// Migrate old unencrypted wallet to new encrypted format
pub fn migrate_wallet_if_needed() -> Result<bool, String> {
    let wallet_dir = platform::get_wallets_dir()?;
    let wallet_file = wallet_dir.join("default_wallet.json");
    
    if !wallet_file.exists() {
        return Ok(false);
    }
    
    let content = fs::read_to_string(&wallet_file)
        .map_err(|e| format!("Failed to read wallet file: {}", e))?;
    
    // Check if already in new format
    if serde_json::from_str::<EncryptedWalletData>(&content).is_ok() {
        return Ok(false); // Already migrated
    }
    
    // Parse old format
    let old_wallet: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse wallet file: {}", e))?;
    
    let address = old_wallet["address"]
        .as_str()
        .ok_or("Invalid wallet format: missing address")?;
    
    let mnemonic = old_wallet["mnemonic"]
        .as_str()
        .unwrap_or("");
    
    // Backup old wallet
    let backup_file = wallet_dir.join("default_wallet.json.backup");
    fs::copy(&wallet_file, &backup_file)
        .map_err(|e| format!("Failed to backup wallet: {}", e))?;
    
    // Save in new encrypted format
    save_wallet_encrypted(address, mnemonic)?;
    
    println!("✅ Wallet migrated to encrypted format");
    Ok(true)
}
