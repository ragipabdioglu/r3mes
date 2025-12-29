use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{BufRead, BufReader};
use tokio::sync::mpsc;
use tokio::task;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub message: String,
    pub process: String,
}

pub async fn tail_log_file(
    process: &str,
    lines: usize,
) -> Result<Vec<String>, String> {
    let log_path = get_log_path(process)?;
    
    if !log_path.exists() {
        return Ok(vec![]);
    }

    let file = File::open(&log_path)
        .map_err(|e| format!("Failed to open log file: {}", e))?;
    
    let reader = BufReader::new(file);
    let all_lines: Vec<String> = reader
        .lines()
        .filter_map(|l| l.ok())
        .collect();
    
    // Return last N lines
    let start = all_lines.len().saturating_sub(lines);
    Ok(all_lines[start..].to_vec())
}

pub async fn filter_logs(
    process: &str,
    level: &str,
) -> Result<Vec<LogEntry>, String> {
    let log_path = get_log_path(process)?;
    
    if !log_path.exists() {
        return Ok(vec![]);
    }

    let file = File::open(&log_path)
        .map_err(|e| format!("Failed to open log file: {}", e))?;
    
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for line in reader.lines().filter_map(|l| l.ok()) {
        // Parse log line (simplified - would need proper parsing)
        if line.to_uppercase().contains(&level.to_uppercase()) {
            entries.push(LogEntry {
                timestamp: "".to_string(),
                level: level.to_string(),
                message: line,
                process: process.to_string(),
            });
        }
    }

    Ok(entries)
}

pub async fn search_logs(
    process: &str,
    query: &str,
) -> Result<Vec<LogEntry>, String> {
    let log_path = get_log_path(process)?;
    
    if !log_path.exists() {
        return Ok(vec![]);
    }

    let file = File::open(&log_path)
        .map_err(|e| format!("Failed to open log file: {}", e))?;
    
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    for line in reader.lines().filter_map(|l| l.ok()) {
        if line.contains(query) {
            entries.push(LogEntry {
                timestamp: "".to_string(),
                level: "INFO".to_string(),
                message: line,
                process: process.to_string(),
            });
        }
    }

    Ok(entries)
}

pub async fn stream_logs(
    process: &str,
    sender: mpsc::Sender<String>,
) -> Result<(), String> {
    let log_path = get_log_path(process)?;
    
    if !log_path.exists() {
        return Err("Log file not found".to_string());
    }

    task::spawn_blocking(move || {
        let file = File::open(&log_path)
            .map_err(|e| format!("Failed to open log file: {}", e))?;
        
        let reader = BufReader::new(file);
        
        // Read from end of file and stream new lines
        // This is simplified - in production, use proper file tailing
        for line in reader.lines().filter_map(|l| l.ok()) {
            if sender.blocking_send(line).is_err() {
                break;
            }
        }
        
        Ok::<(), String>(())
    }).await
    .map_err(|e| format!("Task error: {}", e))??;

    Ok(())
}

fn get_log_path(process: &str) -> Result<PathBuf, String> {
    let home = std::env::var("HOME")
        .map_err(|_| "HOME environment variable not set".to_string())?;
    
    Ok(PathBuf::from(&home)
        .join("R3MES")
        .join("logs")
        .join(format!("{}.log", process)))
}

