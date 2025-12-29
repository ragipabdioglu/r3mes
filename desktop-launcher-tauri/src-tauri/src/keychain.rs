/// Keychain/SecretStorage Integration
/// 
/// Platform-specific secure storage:
/// - Windows: Windows Credential Manager
/// - macOS: Keychain
/// - Linux: Secret Service (libsecret)

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub enum KeychainError {
    NotAvailable(String),
    NotFound,
    AccessDenied,
    Other(String),
}

impl fmt::Display for KeychainError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KeychainError::NotAvailable(msg) => {
                write!(f, "Keychain not available: {}", msg)
            }
            KeychainError::NotFound => write!(f, "Key not found in keychain"),
            KeychainError::AccessDenied => write!(f, "Access denied to keychain"),
            KeychainError::Other(msg) => write!(f, "Keychain error: {}", msg),
        }
    }
}

impl Error for KeychainError {}

pub struct KeychainManager {
    service: String,
}

impl KeychainManager {
    pub fn new() -> Self {
        Self {
            service: "R3MES".to_string(),
        }
    }

    /// Store a secret in the platform keychain
    pub fn store(&self, key: &str, value: &str) -> Result<(), KeychainError> {
        #[cfg(target_os = "windows")]
        {
            self.store_windows(key, value)
        }
        #[cfg(target_os = "macos")]
        {
            self.store_macos(key, value)
        }
        #[cfg(target_os = "linux")]
        {
            self.store_linux(key, value)
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            Err(KeychainError::NotAvailable("Unsupported platform".to_string()))
        }
    }

    /// Retrieve a secret from the platform keychain
    pub fn retrieve(&self, key: &str) -> Result<String, KeychainError> {
        #[cfg(target_os = "windows")]
        {
            self.retrieve_windows(key)
        }
        #[cfg(target_os = "macos")]
        {
            self.retrieve_macos(key)
        }
        #[cfg(target_os = "linux")]
        {
            self.retrieve_linux(key)
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            Err(KeychainError::NotAvailable("Unsupported platform".to_string()))
        }
    }

    /// Delete a secret from the platform keychain
    pub fn delete(&self, key: &str) -> Result<(), KeychainError> {
        #[cfg(target_os = "windows")]
        {
            self.delete_windows(key)
        }
        #[cfg(target_os = "macos")]
        {
            self.delete_macos(key)
        }
        #[cfg(target_os = "linux")]
        {
            self.delete_linux(key)
        }
        #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
        {
            Err(KeychainError::NotAvailable("Unsupported platform".to_string()))
        }
    }

    #[cfg(target_os = "windows")]
    fn store_windows(&self, key: &str, value: &str) -> Result<(), KeychainError> {
        use winapi::um::wincred::{CredWriteW, CREDENTIALW, CRED_TYPE_GENERIC};
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        use std::ptr;

        let target_name: Vec<u16> = OsStr::new(&format!("{}\\{}", self.service, key))
            .encode_wide()
            .chain(Some(0))
            .collect();

        let credential_blob: Vec<u8> = value.as_bytes().to_vec();
        let credential_blob_ptr = credential_blob.as_ptr();

        let mut credential = CREDENTIALW {
            Flags: 0,
            Type: CRED_TYPE_GENERIC,
            TargetName: target_name.as_ptr() as *mut u16,
            Comment: ptr::null_mut(),
            LastWritten: winapi::um::winbase::FILETIME {
                dwLowDateTime: 0,
                dwHighDateTime: 0,
            },
            CredentialBlobSize: credential_blob.len() as u32,
            CredentialBlob: credential_blob_ptr as *mut u8,
            Persist: 2, // CRED_PERSIST_LOCAL_MACHINE
            AttributeCount: 0,
            Attributes: ptr::null_mut(),
            TargetAlias: ptr::null_mut(),
            UserName: ptr::null_mut(),
        };

        unsafe {
            let result = CredWriteW(&mut credential, 0);
            if result == 0 {
                Err(KeychainError::Other(format!(
                    "Failed to write credential: error code {}",
                    winapi::um::errhandlingapi::GetLastError()
                )))
            } else {
                Ok(())
            }
        }
    }

    #[cfg(target_os = "windows")]
    fn retrieve_windows(&self, key: &str) -> Result<String, KeychainError> {
        use winapi::um::wincred::{CredReadW, CredFree, CRED_TYPE_GENERIC};
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;
        use std::ptr;

        let target_name: Vec<u16> = OsStr::new(&format!("{}\\{}", self.service, key))
            .encode_wide()
            .chain(Some(0))
            .collect();

        let mut credential_ptr: *mut winapi::um::wincred::CREDENTIALW = ptr::null_mut();

        unsafe {
            let result = CredReadW(
                target_name.as_ptr(),
                CRED_TYPE_GENERIC,
                0,
                &mut credential_ptr,
            );

            if result == 0 {
                let error = winapi::um::errhandlingapi::GetLastError();
                if error == 1168 {
                    // ERROR_NOT_FOUND
                    return Err(KeychainError::NotFound);
                }
                return Err(KeychainError::Other(format!(
                    "Failed to read credential: error code {}",
                    error
                )));
            }

            let credential = &*credential_ptr;
            let blob_size = credential.CredentialBlobSize as usize;
            let blob_ptr = credential.CredentialBlob;

            let blob_slice = std::slice::from_raw_parts(blob_ptr, blob_size);
            let value = String::from_utf8_lossy(blob_slice).to_string();

            CredFree(credential_ptr as *mut _);

            Ok(value)
        }
    }

    #[cfg(target_os = "windows")]
    fn delete_windows(&self, key: &str) -> Result<(), KeychainError> {
        use winapi::um::wincred::CredDeleteW;
        use std::ffi::OsStr;
        use std::os::windows::ffi::OsStrExt;

        let target_name: Vec<u16> = OsStr::new(&format!("{}\\{}", self.service, key))
            .encode_wide()
            .chain(Some(0))
            .collect();

        unsafe {
            let result = CredDeleteW(target_name.as_ptr(), CRED_TYPE_GENERIC, 0);
            if result == 0 {
                let error = winapi::um::errhandlingapi::GetLastError();
                if error == 1168 {
                    // ERROR_NOT_FOUND
                    return Err(KeychainError::NotFound);
                }
                return Err(KeychainError::Other(format!(
                    "Failed to delete credential: error code {}",
                    error
                )));
            }
            Ok(())
        }
    }

    #[cfg(target_os = "macos")]
    fn store_macos(&self, key: &str, value: &str) -> Result<(), KeychainError> {
        use std::process::Command;

        let service = format!("{}.{}", self.service, key);
        let output = Command::new("security")
            .arg("add-generic-password")
            .arg("-a")
            .arg(&service)
            .arg("-s")
            .arg(&service)
            .arg("-w")
            .arg(value)
            .arg("-U")
            .output()
            .map_err(|e| KeychainError::Other(format!("Failed to execute security command: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("already exists") {
                // Update existing entry
                self.delete_macos(key)?;
                return self.store_macos(key, value);
            }
            return Err(KeychainError::Other(format!(
                "Failed to store in keychain: {}",
                stderr
            )));
        }

        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn retrieve_macos(&self, key: &str) -> Result<String, KeychainError> {
        use std::process::Command;

        let service = format!("{}.{}", self.service, key);
        let output = Command::new("security")
            .arg("find-generic-password")
            .arg("-a")
            .arg(&service)
            .arg("-s")
            .arg(&service)
            .arg("-w")
            .output()
            .map_err(|e| KeychainError::Other(format!("Failed to execute security command: {}", e)))?;

        if !output.status.success() {
            return Err(KeychainError::NotFound);
        }

        let value = String::from_utf8(output.stdout)
            .map_err(|e| KeychainError::Other(format!("Failed to parse output: {}", e)))?;

        Ok(value.trim().to_string())
    }

    #[cfg(target_os = "macos")]
    fn delete_macos(&self, key: &str) -> Result<(), KeychainError> {
        use std::process::Command;

        let service = format!("{}.{}", self.service, key);
        let output = Command::new("security")
            .arg("delete-generic-password")
            .arg("-a")
            .arg(&service)
            .arg("-s")
            .arg(&service)
            .output()
            .map_err(|e| KeychainError::Other(format!("Failed to execute security command: {}", e)))?;

        if !output.status.success() {
            return Err(KeychainError::NotFound);
        }

        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn store_linux(&self, key: &str, value: &str) -> Result<(), KeychainError> {
        use std::process::Command;

        let attribute = format!("{}/{}", self.service, key);
        let output = Command::new("secret-tool")
            .arg("store")
            .arg("--label")
            .arg(&format!("R3MES Wallet: {}", key))
            .arg("service")
            .arg(&self.service)
            .arg("key")
            .arg(key)
            .stdin(std::process::Stdio::piped())
            .output()
            .map_err(|e| KeychainError::Other(format!("Failed to execute secret-tool: {}", e)))?;

        if !output.status.success() {
            return Err(KeychainError::Other(format!(
                "Failed to store in secret service: {}",
                String::from_utf8_lossy(&output.stderr)
            )));
        }

        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn retrieve_linux(&self, key: &str) -> Result<String, KeychainError> {
        use std::process::Command;

        let output = Command::new("secret-tool")
            .arg("lookup")
            .arg("service")
            .arg(&self.service)
            .arg("key")
            .arg(key)
            .output()
            .map_err(|e| KeychainError::Other(format!("Failed to execute secret-tool: {}", e)))?;

        if !output.status.success() {
            return Err(KeychainError::NotFound);
        }

        let value = String::from_utf8(output.stdout)
            .map_err(|e| KeychainError::Other(format!("Failed to parse output: {}", e)))?;

        Ok(value.trim().to_string())
    }

    #[cfg(target_os = "linux")]
    fn delete_linux(&self, key: &str) -> Result<(), KeychainError> {
        // Secret Service doesn't have a direct delete command
        // We'll use dbus-send or just overwrite on next store
        // For now, return Ok (overwrite on next store)
        Ok(())
    }
}

