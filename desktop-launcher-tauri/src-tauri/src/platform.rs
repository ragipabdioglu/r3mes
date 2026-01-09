//! Platform-specific utilities for cross-platform compatibility
//! 
//! This module provides utilities that work correctly on Windows, macOS, and Linux.

#![allow(dead_code)]

use std::path::PathBuf;
use std::env;
use std::process::Command;

/// Configure a Command to hide the console window on Windows
/// This prevents terminal windows from flashing when running subprocesses
#[cfg(windows)]
pub fn hide_console_window(cmd: &mut Command) -> &mut Command {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    cmd.creation_flags(CREATE_NO_WINDOW)
}

#[cfg(not(windows))]
pub fn hide_console_window(cmd: &mut Command) -> &mut Command {
    cmd
}

/// Create a new Command with hidden console window (Windows)
pub fn silent_command(program: &str) -> Command {
    let mut cmd = Command::new(program);
    hide_console_window(&mut cmd);
    cmd
}

/// Get the user's home directory in a cross-platform way
/// 
/// On Windows: Uses USERPROFILE or HOMEDRIVE+HOMEPATH
/// On Unix: Uses HOME
pub fn get_home_dir() -> Result<PathBuf, String> {
    // Try dirs crate first (most reliable)
    if let Some(home) = dirs::home_dir() {
        return Ok(home);
    }
    
    // Fallback to environment variables
    #[cfg(windows)]
    {
        if let Ok(userprofile) = env::var("USERPROFILE") {
            return Ok(PathBuf::from(userprofile));
        }
        
        // Try HOMEDRIVE + HOMEPATH combination
        if let (Ok(drive), Ok(path)) = (env::var("HOMEDRIVE"), env::var("HOMEPATH")) {
            return Ok(PathBuf::from(format!("{}{}", drive, path)));
        }
        
        // Last resort: use APPDATA parent
        if let Ok(appdata) = env::var("APPDATA") {
            if let Some(parent) = PathBuf::from(appdata).parent() {
                return Ok(parent.to_path_buf());
            }
        }
    }
    
    #[cfg(not(windows))]
    {
        if let Ok(home) = env::var("HOME") {
            return Ok(PathBuf::from(home));
        }
    }
    
    Err("Could not determine home directory".to_string())
}

/// Get the R3MES data directory
/// 
/// On Windows: %APPDATA%\R3MES or %USERPROFILE%\.r3mes
/// On Unix: ~/.r3mes
pub fn get_r3mes_data_dir() -> Result<PathBuf, String> {
    #[cfg(windows)]
    {
        // Prefer APPDATA on Windows for proper app data storage
        if let Ok(appdata) = env::var("APPDATA") {
            return Ok(PathBuf::from(appdata).join("R3MES"));
        }
    }
    
    // Fallback to home directory
    let home = get_home_dir()?;
    Ok(home.join(".r3mes"))
}

/// Alias for get_r3mes_data_dir for convenience
pub fn get_data_dir() -> Result<PathBuf, String> {
    get_r3mes_data_dir()
}

/// Get the R3MES logs directory
pub fn get_logs_dir() -> Result<PathBuf, String> {
    Ok(get_r3mes_data_dir()?.join("logs"))
}

/// Get the R3MES wallets directory
pub fn get_wallets_dir() -> Result<PathBuf, String> {
    Ok(get_r3mes_data_dir()?.join("wallets"))
}

/// Get the R3MES models directory
pub fn get_models_dir() -> Result<PathBuf, String> {
    Ok(get_r3mes_data_dir()?.join("models"))
}

/// Get the R3MES adapters directory
pub fn get_adapters_dir() -> Result<PathBuf, String> {
    Ok(get_r3mes_data_dir()?.join("adapters"))
}

/// Get the R3MES config directory
pub fn get_config_dir() -> Result<PathBuf, String> {
    Ok(get_r3mes_data_dir()?)
}

/// Get the R3MES workspace directory (where miner-engine etc. are installed)
pub fn get_workspace_dir() -> Result<PathBuf, String> {
    #[cfg(windows)]
    {
        // On Windows, use a dedicated folder in user's home
        let home = get_home_dir()?;
        return Ok(home.join("R3MES"));
    }
    
    #[cfg(not(windows))]
    {
        let home = get_home_dir()?;
        return Ok(home.join("R3MES"));
    }
}

/// Get the Python executable path for the virtual environment
pub fn get_venv_python() -> Result<PathBuf, String> {
    let workspace = get_workspace_dir()?;
    let venv_dir = workspace.join("miner-engine").join("venv");
    
    #[cfg(windows)]
    {
        // Windows uses Scripts/python.exe
        let python = venv_dir.join("Scripts").join("python.exe");
        if python.exists() {
            return Ok(python);
        }
        // Fallback to python3.exe
        let python3 = venv_dir.join("Scripts").join("python3.exe");
        if python3.exists() {
            return Ok(python3);
        }
    }
    
    #[cfg(not(windows))]
    {
        // Unix uses bin/python3
        let python3 = venv_dir.join("bin").join("python3");
        if python3.exists() {
            return Ok(python3);
        }
        // Fallback to python
        let python = venv_dir.join("bin").join("python");
        if python.exists() {
            return Ok(python);
        }
    }
    
    Err("Python virtual environment not found".to_string())
}

/// Get the system Python executable
pub fn get_system_python() -> &'static str {
    #[cfg(windows)]
    {
        "python"
    }
    
    #[cfg(not(windows))]
    {
        "python3"
    }
}

/// Check if a port is in use (cross-platform)
pub fn is_port_in_use(port: u16) -> bool {
    use std::net::TcpListener;
    TcpListener::bind(("127.0.0.1", port)).is_err()
}

/// Check if a process is running by checking if a port is listening
pub fn check_port_listening(port: u16) -> bool {
    #[cfg(windows)]
    {
        let output = silent_command("netstat")
            .args(["-an"])
            .output();
        
        if let Ok(output) = output {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let port_str = format!(":{}", port);
            return output_str.lines().any(|line| {
                line.contains(&port_str) && line.contains("LISTENING")
            });
        }
        false
    }
    
    #[cfg(target_os = "macos")]
    {
        let output = silent_command("lsof")
            .args(["-i", &format!(":{}", port), "-sTCP:LISTEN"])
            .output();
        
        if let Ok(output) = output {
            return output.status.success() && !output.stdout.is_empty();
        }
        false
    }
    
    #[cfg(target_os = "linux")]
    {
        // Try ss first (faster), then netstat
        let output = silent_command("ss")
            .args(["-tln"])
            .output();
        
        if let Ok(output) = output {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let port_str = format!(":{}", port);
            if output_str.contains(&port_str) {
                return true;
            }
        }
        
        // Fallback to netstat
        let output = silent_command("netstat")
            .args(["-tln"])
            .output();
        
        if let Ok(output) = output {
            let output_str = String::from_utf8_lossy(&output.stdout);
            let port_str = format!(":{}", port);
            return output_str.contains(&port_str);
        }
        
        false
    }
}

/// Open a URL in the default browser (cross-platform)
pub fn open_url(url: &str) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        silent_command("cmd")
            .args(["/C", "start", "", url])
            .spawn()
            .map_err(|e| format!("Failed to open URL: {}", e))?;
    }
    
    #[cfg(target_os = "macos")]
    {
        silent_command("open")
            .arg(url)
            .spawn()
            .map_err(|e| format!("Failed to open URL: {}", e))?;
    }
    
    #[cfg(target_os = "linux")]
    {
        silent_command("xdg-open")
            .arg(url)
            .spawn()
            .map_err(|e| format!("Failed to open URL: {}", e))?;
    }
    
    Ok(())
}

/// Get the shell command for the current platform
pub fn get_shell() -> (&'static str, &'static str) {
    #[cfg(windows)]
    {
        ("cmd", "/C")
    }
    
    #[cfg(not(windows))]
    {
        ("sh", "-c")
    }
}

/// Execute a shell command (cross-platform)
pub fn execute_shell_command(command: &str) -> Result<std::process::Output, String> {
    let (shell, flag) = get_shell();
    
    silent_command(shell)
        .arg(flag)
        .arg(command)
        .output()
        .map_err(|e| format!("Failed to execute command: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_get_home_dir() {
        let home = get_home_dir();
        assert!(home.is_ok());
        assert!(home.unwrap().exists());
    }
    
    #[test]
    fn test_get_r3mes_data_dir() {
        let data_dir = get_r3mes_data_dir();
        assert!(data_dir.is_ok());
    }
}
