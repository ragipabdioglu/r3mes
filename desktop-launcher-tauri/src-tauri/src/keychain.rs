//! Secure credential storage using OS keychain/keyring
//! 
//! Provides cross-platform secure storage for sensitive data like private keys,
//! API tokens, and encryption keys using the operating system's native keychain.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug)]
pub enum KeychainError {
    NotSupported,
    AccessDenied,
    ItemNotFound,
    InvalidData,
    SystemError(String),
}

impl std::fmt::Display for KeychainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KeychainError::NotSupported => write!(f, "Keychain not supported on this platform"),
            KeychainError::AccessDenied => write!(f, "Access denied to keychain"),
            KeychainError::ItemNotFound => write!(f, "Item not found in keychain"),
            KeychainError::InvalidData => write!(f, "Invalid data format"),
            KeychainError::SystemError(msg) => write!(f, "System error: {}", msg),
        }
    }
}

impl std::error::Error for KeychainError {}

#[derive(Debug, Serialize, Deserialize)]
pub struct KeychainItem {
    pub service: String,
    pub account: String,
    pub data: String,
    pub created_at: u64,
    pub accessed_at: u64,
}

pub struct KeychainManager {
    service_name: String,
    fallback_storage: HashMap<String, String>,
    use_fallback: bool,
}

impl KeychainManager {
    /// Create a new keychain manager
    pub fn new() -> Self {
        Self {
            service_name: "io.r3mes.desktop-launcher".to_string(),
            fallback_storage: HashMap::new(),
            use_fallback: !Self::is_keychain_available(),
        }
    }
    
    /// Create a keychain manager with custom service name
    pub fn with_service(service_name: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            fallback_storage: HashMap::new(),
            use_fallback: !Self::is_keychain_available(),
        }
    }
    
    /// Check if native keychain is available
    pub fn is_keychain_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            // macOS always has Keychain
            true
        }
        
        #[cfg(target_os = "windows")]
        {
            // Windows has Credential Manager
            true
        }
        
        #[cfg(target_os = "linux")]
        {
            // Check for libsecret/gnome-keyring
            Self::check_linux_keyring()
        }
        
        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            false
        }
    }
    
    #[cfg(target_os = "linux")]
    fn check_linux_keyring() -> bool {
        use std::process::Command;
        
        // Check if secret-tool is available (libsecret)
        if Command::new("secret-tool").arg("--version").output().is_ok() {
            return true;
        }
        
        // Check if gnome-keyring is running
        if let Ok(output) = Command::new("ps").arg("aux").output() {
            let processes = String::from_utf8_lossy(&output.stdout);
            if processes.contains("gnome-keyring") {
                return true;
            }
        }
        
        // Check for KDE Wallet
        if Command::new("kwalletcli").arg("--help").output().is_ok() {
            return true;
        }
        
        false
    }
    
    /// Store a credential in the keychain
    pub fn store(&self, key: &str, value: &str) -> Result<(), KeychainError> {
        if self.use_fallback {
            return self.store_fallback(key, value);
        }
        
        #[cfg(target_os = "macos")]
        {
            self.store_macos(key, value)
        }
        
        #[cfg(target_os = "windows")]
        {
            self.store_windows(key, value)
        }
        
        #[cfg(target_os = "linux")]
        {
            self.store_linux(key, value)
        }
        
        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            self.store_fallback(key, value)
        }
    }
    
    /// Retrieve a credential from the keychain
    pub fn retrieve(&self, key: &str) -> Result<String, KeychainError> {
        if self.use_fallback {
            return self.retrieve_fallback(key);
        }
        
        #[cfg(target_os = "macos")]
        {
            self.retrieve_macos(key)
        }
        
        #[cfg(target_os = "windows")]
        {
            self.retrieve_windows(key)
        }
        
        #[cfg(target_os = "linux")]
        {
            self.retrieve_linux(key)
        }
        
        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            self.retrieve_fallback(key)
        }
    }
    
    /// Delete a credential from the keychain
    pub fn delete(&self, key: &str) -> Result<(), KeychainError> {
        if self.use_fallback {
            return self.delete_fallback(key);
        }
        
        #[cfg(target_os = "macos")]
        {
            self.delete_macos(key)
        }
        
        #[cfg(target_os = "windows")]
        {
            self.delete_windows(key)
        }
        
        #[cfg(target_os = "linux")]
        {
            self.delete_linux(key)
        }
        
        #[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
        {
            self.delete_fallback(key)
        }
    }
    
    /// List all stored keys
    pub fn list_keys(&self) -> Result<Vec<String>, KeychainError> {
        if self.use_fallback {
            return Ok(self.fallback_storage.keys().cloned().collect());
        }
        
        // Platform-specific implementations would go here
        // For now, return empty list
        Ok(Vec::new())
    }
    
    /// Check if a key exists
    pub fn exists(&self, key: &str) -> bool {
        self.retrieve(key).is_ok()
    }
    
    /// Clear all stored credentials (use with caution)
    pub fn clear_all(&mut self) -> Result<(), KeychainError> {
        if self.use_fallback {
            self.fallback_storage.clear();
            return Ok(());
        }
        
        // For native keychains, we'd need to list and delete each item
        // This is a potentially dangerous operation
        Err(KeychainError::NotSupported)
    }
    
    // macOS implementation using Security framework
    #[cfg(target_os = "macos")]
    fn store_macos(&self, key: &str, value: &str) -> Result<(), KeychainError> {
        use std::process::Command;
        
        let output = Command::new("security")
            .arg("add-generic-password")
            .arg("-a") // account
            .arg(key)
            .arg("-s") // service
            .arg(&self.service_name)
            .arg("-w") // password
            .arg(value)
            .arg("-U") // update if exists
            .output()
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        if output.status.success() {
            Ok(())
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(KeychainError::SystemError(error.to_string()))
        }
    }
    
    #[cfg(target_os = "macos")]
    fn retrieve_macos(&self, key: &str) -> Result<String, KeychainError> {
        use std::process::Command;
        
        let output = Command::new("security")
            .arg("find-generic-password")
            .arg("-a") // account
            .arg(key)
            .arg("-s") // service
            .arg(&self.service_name)
            .arg("-w") // print password only
            .output()
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        if output.status.success() {
            let password = String::from_utf8_lossy(&output.stdout);
            Ok(password.trim().to_string())
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            if error.contains("could not be found") {
                Err(KeychainError::ItemNotFound)
            } else {
                Err(KeychainError::SystemError(error.to_string()))
            }
        }
    }
    
    #[cfg(target_os = "macos")]
    fn delete_macos(&self, key: &str) -> Result<(), KeychainError> {
        use std::process::Command;
        
        let output = Command::new("security")
            .arg("delete-generic-password")
            .arg("-a") // account
            .arg(key)
            .arg("-s") // service
            .arg(&self.service_name)
            .output()
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        if output.status.success() {
            Ok(())
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            if error.contains("could not be found") {
                Err(KeychainError::ItemNotFound)
            } else {
                Err(KeychainError::SystemError(error.to_string()))
            }
        }
    }
    
    // Windows implementation using Credential Manager
    #[cfg(target_os = "windows")]
    fn store_windows(&self, key: &str, value: &str) -> Result<(), KeychainError> {
        use std::process::Command;
        
        let target_name = format!("{}:{}", self.service_name, key);
        
        let output = Command::new("cmdkey")
            .arg("/generic")
            .arg(&target_name)
            .arg("/user")
            .arg(key)
            .arg("/pass")
            .arg(value)
            .output()
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        if output.status.success() {
            Ok(())
        } else {
            let error = String::from_utf8_lossy(&output.stderr);
            Err(KeychainError::SystemError(error.to_string()))
        }
    }
    
    #[cfg(target_os = "windows")]
    fn retrieve_windows(&self, key: &str) -> Result<String, KeychainError> {
        // Windows credential retrieval is more complex and would require
        // Windows API calls or PowerShell. For now, use fallback.
        self.retrieve_fallback(key)
    }
    
    #[cfg(target_os = "windows")]
    fn delete_windows(&self, key: &str) -> Result<(), KeychainError> {
        use std::process::Command;
        
        let target_name = format!("{}:{}", self.service_name, key);
        
        let output = Command::new("cmdkey")
            .arg("/delete")
            .arg(&target_name)
            .output()
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        if output.status.success() {
            Ok(())
        } else {
            Err(KeychainError::ItemNotFound)
        }
    }
    
    // Linux implementation using libsecret/gnome-keyring
    #[cfg(target_os = "linux")]
    fn store_linux(&self, key: &str, value: &str) -> Result<(), KeychainError> {
        use std::process::Command;
        
        // Try secret-tool first (libsecret)
        let output = Command::new("secret-tool")
            .arg("store")
            .arg("--label")
            .arg(&format!("R3MES {}", key))
            .arg("service")
            .arg(&self.service_name)
            .arg("account")
            .arg(key)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn();
        
        if let Ok(mut child) = output {
            use std::io::Write;
            if let Some(stdin) = child.stdin.as_mut() {
                let _ = stdin.write_all(value.as_bytes());
            }
            
            let output = child.wait_with_output()
                .map_err(|e| KeychainError::SystemError(e.to_string()))?;
            
            if output.status.success() {
                return Ok(());
            }
        }
        
        // Fallback to encrypted file storage
        self.store_fallback(key, value)
    }
    
    #[cfg(target_os = "linux")]
    fn retrieve_linux(&self, key: &str) -> Result<String, KeychainError> {
        use std::process::Command;
        
        // Try secret-tool first
        let output = Command::new("secret-tool")
            .arg("lookup")
            .arg("service")
            .arg(&self.service_name)
            .arg("account")
            .arg(key)
            .output();
        
        if let Ok(output) = output {
            if output.status.success() {
                let value = String::from_utf8_lossy(&output.stdout);
                return Ok(value.trim().to_string());
            }
        }
        
        // Fallback to encrypted file storage
        self.retrieve_fallback(key)
    }
    
    #[cfg(target_os = "linux")]
    fn delete_linux(&self, key: &str) -> Result<(), KeychainError> {
        use std::process::Command;
        
        // Try secret-tool first
        let output = Command::new("secret-tool")
            .arg("clear")
            .arg("service")
            .arg(&self.service_name)
            .arg("account")
            .arg(key)
            .output();
        
        if let Ok(output) = output {
            if output.status.success() {
                return Ok(());
            }
        }
        
        // Fallback to encrypted file storage
        self.delete_fallback(key)
    }
    
    // Fallback implementation using encrypted file storage
    fn store_fallback(&self, key: &str, value: &str) -> Result<(), KeychainError> {
        use std::fs;
        
        
        let storage_dir = self.get_fallback_storage_dir()?;
        fs::create_dir_all(&storage_dir)
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        let encrypted_value = self.encrypt_value(value)?;
        let file_path = storage_dir.join(format!("{}.enc", key));
        
        fs::write(&file_path, encrypted_value)
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        Ok(())
    }
    
    fn retrieve_fallback(&self, key: &str) -> Result<String, KeychainError> {
        use std::fs;
        
        
        let storage_dir = self.get_fallback_storage_dir()?;
        let file_path = storage_dir.join(format!("{}.enc", key));
        
        if !file_path.exists() {
            return Err(KeychainError::ItemNotFound);
        }
        
        let encrypted_data = fs::read(&file_path)
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        self.decrypt_value(&encrypted_data)
    }
    
    fn delete_fallback(&self, key: &str) -> Result<(), KeychainError> {
        use std::fs;
        
        
        let storage_dir = self.get_fallback_storage_dir()?;
        let file_path = storage_dir.join(format!("{}.enc", key));
        
        if !file_path.exists() {
            return Err(KeychainError::ItemNotFound);
        }
        
        fs::remove_file(&file_path)
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        Ok(())
    }
    
    fn get_fallback_storage_dir(&self) -> Result<PathBuf, KeychainError> {
        // Use platform module for cross-platform home directory
        #[cfg(windows)]
        {
            let home = std::env::var("USERPROFILE")
                .or_else(|_| std::env::var("APPDATA").map(|p| {
                    std::path::Path::new(&p).parent()
                        .map(|p| p.to_string_lossy().to_string())
                        .unwrap_or(p)
                }))
                .map_err(|_| KeychainError::SystemError("Cannot determine home directory".to_string()))?;
            Ok(PathBuf::from(home).join(".r3mes").join("keychain"))
        }
        
        #[cfg(not(windows))]
        {
            let home = std::env::var("HOME")
                .map_err(|_| KeychainError::SystemError("Cannot determine home directory".to_string()))?;
            Ok(PathBuf::from(home).join(".r3mes").join("keychain"))
        }
    }
    
    fn encrypt_value(&self, value: &str) -> Result<Vec<u8>, KeychainError> {
        use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
        use aes_gcm::aead::{Aead, OsRng};
        use rand::RngCore;
        
        // Generate or retrieve encryption key
        let key = self.get_or_create_encryption_key()?;
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| KeychainError::InvalidData)?;
        
        // Generate random nonce
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        // Encrypt the value
        let ciphertext = cipher.encrypt(nonce, value.as_bytes())
            .map_err(|_| KeychainError::SystemError("Encryption failed".to_string()))?;
        
        // Combine nonce and ciphertext
        let mut result = nonce_bytes.to_vec();
        result.extend_from_slice(&ciphertext);
        
        Ok(result)
    }
    
    fn decrypt_value(&self, encrypted_data: &[u8]) -> Result<String, KeychainError> {
        use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
        use aes_gcm::aead::Aead;
        
        if encrypted_data.len() < 12 {
            return Err(KeychainError::InvalidData);
        }
        
        // Extract nonce and ciphertext
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);
        
        // Get encryption key
        let key = self.get_or_create_encryption_key()?;
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| KeychainError::InvalidData)?;
        
        // Decrypt
        let plaintext = cipher.decrypt(nonce, ciphertext)
            .map_err(|_| KeychainError::SystemError("Decryption failed".to_string()))?;
        
        String::from_utf8(plaintext)
            .map_err(|_| KeychainError::InvalidData)
    }
    
    fn get_or_create_encryption_key(&self) -> Result<[u8; 32], KeychainError> {
        use std::fs;
        use rand::RngCore;
        
        let storage_dir = self.get_fallback_storage_dir()?;
        let key_file = storage_dir.join("master.key");
        
        if key_file.exists() {
            let key_data = fs::read(&key_file)
                .map_err(|e| KeychainError::SystemError(e.to_string()))?;
            
            if key_data.len() == 32 {
                let mut key = [0u8; 32];
                key.copy_from_slice(&key_data);
                return Ok(key);
            }
        }
        
        // Generate new key
        let mut key = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut key);
        
        // Save key
        fs::create_dir_all(&storage_dir)
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        fs::write(&key_file, &key)
            .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        
        // Set restrictive permissions
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&key_file)
                .map_err(|e| KeychainError::SystemError(e.to_string()))?
                .permissions();
            perms.set_mode(0o600); // Read/write for owner only
            fs::set_permissions(&key_file, perms)
                .map_err(|e| KeychainError::SystemError(e.to_string()))?;
        }
        
        #[cfg(windows)]
        {
            // On Windows, files in user's home directory are already protected by NTFS ACLs
            // The file is only accessible by the current user by default
            // For additional security, we could use Windows ACL APIs, but that requires winapi crate
            // For now, the default permissions are sufficient for most use cases
        }
        
        Ok(key)
    }
}

impl Default for KeychainManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common operations
impl KeychainManager {
    /// Store a wallet private key securely
    pub fn store_wallet_private_key(&self, address: &str, private_key: &str) -> Result<(), KeychainError> {
        let key = format!("wallet_private_key_{}", address);
        self.store(&key, private_key)
    }
    
    /// Retrieve a wallet private key
    pub fn retrieve_wallet_private_key(&self, address: &str) -> Result<String, KeychainError> {
        let key = format!("wallet_private_key_{}", address);
        self.retrieve(&key)
    }
    
    /// Store an API key
    pub fn store_api_key(&self, service: &str, api_key: &str) -> Result<(), KeychainError> {
        let key = format!("api_key_{}", service);
        self.store(&key, api_key)
    }
    
    /// Retrieve an API key
    pub fn retrieve_api_key(&self, service: &str) -> Result<String, KeychainError> {
        let key = format!("api_key_{}", service);
        self.retrieve(&key)
    }
    
    /// Store an encryption key
    pub fn store_encryption_key(&self, purpose: &str, encryption_key: &str) -> Result<(), KeychainError> {
        let key = format!("encryption_key_{}", purpose);
        self.store(&key, encryption_key)
    }
    
    /// Retrieve an encryption key
    pub fn retrieve_encryption_key(&self, purpose: &str) -> Result<String, KeychainError> {
        let key = format!("encryption_key_{}", purpose);
        self.retrieve(&key)
    }
}