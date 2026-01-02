/// Log reader and processor for R3MES Desktop Launcher
/// 
/// Provides real-time log tailing, filtering, searching, and processing
/// for all R3MES processes and services.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio::time::{interval, Duration};
use regex::Regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: u64,
    pub level: LogLevel,
    pub source: String,
    pub message: String,
    pub raw_line: String,
    pub line_number: u64,
    pub file_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Fatal,
    Unknown,
}

impl LogLevel {
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "TRACE" | "TRC" => LogLevel::Trace,
            "DEBUG" | "DBG" => LogLevel::Debug,
            "INFO" | "INF" => LogLevel::Info,
            "WARN" | "WARNING" | "WRN" => LogLevel::Warn,
            "ERROR" | "ERR" => LogLevel::Error,
            "FATAL" | "FTL" => LogLevel::Fatal,
            _ => LogLevel::Unknown,
        }
    }
    
    pub fn to_string(&self) -> String {
        match self {
            LogLevel::Trace => "TRACE".to_string(),
            LogLevel::Debug => "DEBUG".to_string(),
            LogLevel::Info => "INFO".to_string(),
            LogLevel::Warn => "WARN".to_string(),
            LogLevel::Error => "ERROR".to_string(),
            LogLevel::Fatal => "FATAL".to_string(),
            LogLevel::Unknown => "UNKNOWN".to_string(),
        }
    }
    
    pub fn priority(&self) -> u8 {
        match self {
            LogLevel::Trace => 0,
            LogLevel::Debug => 1,
            LogLevel::Info => 2,
            LogLevel::Warn => 3,
            LogLevel::Error => 4,
            LogLevel::Fatal => 5,
            LogLevel::Unknown => 6,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogFilter {
    pub levels: Vec<LogLevel>,
    pub sources: Vec<String>,
    pub search_term: Option<String>,
    pub regex_pattern: Option<String>,
    pub time_range: Option<TimeRange>,
    pub max_entries: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogStats {
    pub total_entries: u64,
    pub entries_by_level: HashMap<String, u64>,
    pub entries_by_source: HashMap<String, u64>,
    pub first_entry_time: Option<u64>,
    pub last_entry_time: Option<u64>,
    pub file_size_bytes: u64,
}

pub struct LogReader {
    log_dir: PathBuf,
    watchers: Arc<RwLock<HashMap<String, LogWatcher>>>,
    buffer: Arc<RwLock<VecDeque<LogEntry>>>,
    max_buffer_size: usize,
    parsers: HashMap<String, Box<dyn LogParser + Send + Sync>>,
}

pub struct LogWatcher {
    file_path: PathBuf,
    file: File,
    position: u64,
    line_number: u64,
    source: String,
    sender: mpsc::UnboundedSender<LogEntry>,
}

pub trait LogParser: Send + Sync {
    fn parse_line(&self, line: &str, source: &str, line_number: u64, file_path: &str) -> Option<LogEntry>;
    fn get_source_name(&self) -> &str;
}

impl LogReader {
    /// Create a new log reader
    pub fn new(log_dir: PathBuf) -> Self {
        let mut parsers: HashMap<String, Box<dyn LogParser + Send + Sync>> = HashMap::new();
        
        // Register default parsers
        parsers.insert("node.log".to_string(), Box::new(TendermintLogParser));
        parsers.insert("miner.log".to_string(), Box::new(PythonLogParser));
        parsers.insert("ipfs.log".to_string(), Box::new(IPFSLogParser));
        parsers.insert("serving.log".to_string(), Box::new(PythonLogParser));
        parsers.insert("validator.log".to_string(), Box::new(TendermintLogParser));
        parsers.insert("proposer.log".to_string(), Box::new(PythonLogParser));
        
        Self {
            log_dir,
            watchers: Arc::new(RwLock::new(HashMap::new())),
            buffer: Arc::new(RwLock::new(VecDeque::new())),
            max_buffer_size: 10000,
            parsers,
        }
    }
    
    /// Start watching log files
    pub async fn start_watching(&self) -> Result<(), String> {
        let log_files = vec![
            "node.log", "miner.log", "ipfs.log", 
            "serving.log", "validator.log", "proposer.log"
        ];
        
        let (tx, mut rx) = mpsc::unbounded_channel::<LogEntry>();
        
        // Start watchers for each log file
        for log_file in log_files {
            let file_path = self.log_dir.join(log_file);
            if file_path.exists() {
                let watcher = self.create_watcher(file_path, log_file.to_string(), tx.clone()).await?;
                self.watchers.write().await.insert(log_file.to_string(), watcher);
            }
        }
        
        // Start buffer management task
        let buffer = Arc::clone(&self.buffer);
        let max_size = self.max_buffer_size;
        
        tokio::spawn(async move {
            while let Some(entry) = rx.recv().await {
                let mut buffer = buffer.write().await;
                
                // Add new entry
                buffer.push_back(entry);
                
                // Trim buffer if too large
                while buffer.len() > max_size {
                    buffer.pop_front();
                }
            }
        });
        
        // Start periodic file checking
        self.start_file_checker().await;
        
        Ok(())
    }
    
    /// Create a log watcher for a specific file
    async fn create_watcher(
        &self,
        file_path: PathBuf,
        source: String,
        sender: mpsc::UnboundedSender<LogEntry>,
    ) -> Result<LogWatcher, String> {
        let file = File::open(&file_path)
            .map_err(|e| format!("Failed to open log file {:?}: {}", file_path, e))?;
        
        // Start from end of file for new entries only
        let mut file_for_seek = file.try_clone()
            .map_err(|e| format!("Failed to clone file handle: {}", e))?;
        
        let position = file_for_seek.seek(SeekFrom::End(0))
            .map_err(|e| format!("Failed to seek to end of file: {}", e))?;
        
        Ok(LogWatcher {
            file_path,
            file,
            position,
            line_number: 0,
            source,
            sender,
        })
    }
    
    /// Start periodic file checking for new content
    async fn start_file_checker(&self) {
        let watchers = Arc::clone(&self.watchers);
        let parsers = self.parsers.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(500));
            
            loop {
                interval.tick().await;
                
                let mut watchers = watchers.write().await;
                for (file_name, watcher) in watchers.iter_mut() {
                    if let Some(parser) = parsers.get(file_name) {
                        if let Err(e) = Self::check_file_for_updates(watcher, parser.as_ref()).await {
                            log::error!("Error checking file {}: {}", file_name, e);
                        }
                    }
                }
            }
        });
    }
    
    /// Check a file for new content
    async fn check_file_for_updates(
        watcher: &mut LogWatcher,
        parser: &dyn LogParser,
    ) -> Result<(), String> {
        // Check if file has grown
        let metadata = std::fs::metadata(&watcher.file_path)
            .map_err(|e| format!("Failed to get file metadata: {}", e))?;
        
        let current_size = metadata.len();
        
        if current_size > watcher.position {
            // File has new content
            let mut file = File::open(&watcher.file_path)
                .map_err(|e| format!("Failed to reopen file: {}", e))?;
            
            file.seek(SeekFrom::Start(watcher.position))
                .map_err(|e| format!("Failed to seek to position: {}", e))?;
            
            let reader = BufReader::new(file);
            
            for line in reader.lines() {
                let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
                watcher.line_number += 1;
                
                if let Some(entry) = parser.parse_line(
                    &line,
                    &watcher.source,
                    watcher.line_number,
                    &watcher.file_path.to_string_lossy(),
                ) {
                    if let Err(_) = watcher.sender.send(entry) {
                        // Receiver dropped, stop watching
                        break;
                    }
                }
            }
            
            watcher.position = current_size;
        }
        
        Ok(())
    }
    
    /// Get recent log entries with optional filtering
    pub async fn get_logs(&self, filter: Option<LogFilter>) -> Vec<LogEntry> {
        let buffer = self.buffer.read().await;
        let mut entries: Vec<LogEntry> = buffer.iter().cloned().collect();
        
        if let Some(filter) = filter {
            entries = self.apply_filter(entries, &filter);
        }
        
        // Sort by timestamp (newest first)
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        entries
    }
    
    /// Get logs from a specific file
    pub async fn get_logs_from_file(
        &self,
        file_name: &str,
        lines: Option<usize>,
        filter: Option<LogFilter>,
    ) -> Result<Vec<LogEntry>, String> {
        let file_path = self.log_dir.join(file_name);
        
        if !file_path.exists() {
            return Ok(Vec::new());
        }
        
        let parser = self.parsers.get(file_name)
            .ok_or_else(|| format!("No parser found for file: {}", file_name))?;
        
        let file = File::open(&file_path)
            .map_err(|e| format!("Failed to open file: {}", e))?;
        
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        let mut line_number = 0;
        
        for line in reader.lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            line_number += 1;
            
            if let Some(entry) = parser.parse_line(
                &line,
                file_name,
                line_number,
                &file_path.to_string_lossy(),
            ) {
                entries.push(entry);
            }
        }
        
        // Apply filter if provided
        if let Some(filter) = filter {
            entries = self.apply_filter(entries, &filter);
        }
        
        // Sort by timestamp (newest first)
        entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        
        // Limit number of entries if specified
        if let Some(limit) = lines {
            entries.truncate(limit);
        }
        
        Ok(entries)
    }
    
    /// Apply filter to log entries
    fn apply_filter(&self, mut entries: Vec<LogEntry>, filter: &LogFilter) -> Vec<LogEntry> {
        // Filter by log levels
        if !filter.levels.is_empty() {
            entries.retain(|entry| filter.levels.contains(&entry.level));
        }
        
        // Filter by sources
        if !filter.sources.is_empty() {
            entries.retain(|entry| filter.sources.contains(&entry.source));
        }
        
        // Filter by search term
        if let Some(search_term) = &filter.search_term {
            let search_lower = search_term.to_lowercase();
            entries.retain(|entry| {
                entry.message.to_lowercase().contains(&search_lower) ||
                entry.raw_line.to_lowercase().contains(&search_lower)
            });
        }
        
        // Filter by regex pattern
        if let Some(pattern) = &filter.regex_pattern {
            if let Ok(regex) = Regex::new(pattern) {
                entries.retain(|entry| {
                    regex.is_match(&entry.message) || regex.is_match(&entry.raw_line)
                });
            }
        }
        
        // Filter by time range
        if let Some(time_range) = &filter.time_range {
            entries.retain(|entry| {
                entry.timestamp >= time_range.start && entry.timestamp <= time_range.end
            });
        }
        
        // Limit number of entries
        if let Some(max_entries) = filter.max_entries {
            entries.truncate(max_entries);
        }
        
        entries
    }
    
    /// Get log statistics
    pub async fn get_log_stats(&self, file_name: Option<String>) -> Result<LogStats, String> {
        let entries = if let Some(file_name) = file_name {
            self.get_logs_from_file(&file_name, None, None).await?
        } else {
            self.get_logs(None).await
        };
        
        let mut entries_by_level = HashMap::new();
        let mut entries_by_source = HashMap::new();
        let mut first_entry_time = None;
        let mut last_entry_time = None;
        
        for entry in &entries {
            // Count by level
            let level_str = entry.level.to_string();
            *entries_by_level.entry(level_str).or_insert(0) += 1;
            
            // Count by source
            *entries_by_source.entry(entry.source.clone()).or_insert(0) += 1;
            
            // Track time range
            if first_entry_time.is_none() || entry.timestamp < first_entry_time.unwrap() {
                first_entry_time = Some(entry.timestamp);
            }
            if last_entry_time.is_none() || entry.timestamp > last_entry_time.unwrap() {
                last_entry_time = Some(entry.timestamp);
            }
        }
        
        // Calculate total file size
        let mut file_size_bytes = 0;
        if let Ok(entries) = std::fs::read_dir(&self.log_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    file_size_bytes += metadata.len();
                }
            }
        }
        
        Ok(LogStats {
            total_entries: entries.len() as u64,
            entries_by_level,
            entries_by_source,
            first_entry_time,
            last_entry_time,
            file_size_bytes,
        })
    }
    
    /// Export logs to file
    pub async fn export_logs(
        &self,
        output_path: PathBuf,
        filter: Option<LogFilter>,
        format: ExportFormat,
    ) -> Result<(), String> {
        let entries = self.get_logs(filter).await;
        
        match format {
            ExportFormat::Json => self.export_json(&entries, output_path).await,
            ExportFormat::Csv => self.export_csv(&entries, output_path).await,
            ExportFormat::Text => self.export_text(&entries, output_path).await,
        }
    }
    
    async fn export_json(&self, entries: &[LogEntry], output_path: PathBuf) -> Result<(), String> {
        let json = serde_json::to_string_pretty(entries)
            .map_err(|e| format!("Failed to serialize to JSON: {}", e))?;
        
        std::fs::write(output_path, json)
            .map_err(|e| format!("Failed to write JSON file: {}", e))?;
        
        Ok(())
    }
    
    async fn export_csv(&self, entries: &[LogEntry], output_path: PathBuf) -> Result<(), String> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(output_path)
            .map_err(|e| format!("Failed to create CSV file: {}", e))?;
        
        // Write header
        writeln!(file, "timestamp,level,source,message,line_number,file_path")
            .map_err(|e| format!("Failed to write CSV header: {}", e))?;
        
        // Write entries
        for entry in entries {
            writeln!(
                file,
                "{},{},{},{},{},{}",
                entry.timestamp,
                entry.level.to_string(),
                entry.source,
                entry.message.replace(',', ";"), // Escape commas
                entry.line_number,
                entry.file_path
            ).map_err(|e| format!("Failed to write CSV entry: {}", e))?;
        }
        
        Ok(())
    }
    
    async fn export_text(&self, entries: &[LogEntry], output_path: PathBuf) -> Result<(), String> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(output_path)
            .map_err(|e| format!("Failed to create text file: {}", e))?;
        
        for entry in entries {
            writeln!(
                file,
                "[{}] [{}] [{}] {}",
                chrono::DateTime::from_timestamp(entry.timestamp as i64, 0)
                    .unwrap_or_default()
                    .format("%Y-%m-%d %H:%M:%S"),
                entry.level.to_string(),
                entry.source,
                entry.message
            ).map_err(|e| format!("Failed to write text entry: {}", e))?;
        }
        
        Ok(())
    }
    
    /// Clear log buffer
    pub async fn clear_buffer(&self) {
        self.buffer.write().await.clear();
    }
    
    /// Stop watching all files
    pub async fn stop_watching(&self) {
        self.watchers.write().await.clear();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Text,
}

// Log parsers for different services

pub struct TendermintLogParser;

impl LogParser for TendermintLogParser {
    fn parse_line(&self, line: &str, source: &str, line_number: u64, file_path: &str) -> Option<LogEntry> {
        // Parse Tendermint log format: I[2024-01-01|12:00:00.000] message module=consensus
        let re = Regex::new(r"^([IWEF])\[([^\]]+)\]\s+(.+)").ok()?;
        
        if let Some(captures) = re.captures(line) {
            let level_char = captures.get(1)?.as_str();
            let timestamp_str = captures.get(2)?.as_str();
            let message = captures.get(3)?.as_str();
            
            let level = match level_char {
                "I" => LogLevel::Info,
                "W" => LogLevel::Warn,
                "E" => LogLevel::Error,
                "F" => LogLevel::Fatal,
                _ => LogLevel::Unknown,
            };
            
            // Parse timestamp (simplified)
            let timestamp = chrono::NaiveDateTime::parse_from_str(timestamp_str, "%Y-%m-%d|%H:%M:%S%.3f")
                .ok()?
                .and_utc()
                .timestamp() as u64;
            
            Some(LogEntry {
                timestamp,
                level,
                source: source.to_string(),
                message: message.to_string(),
                raw_line: line.to_string(),
                line_number,
                file_path: file_path.to_string(),
            })
        } else {
            // Fallback for unstructured lines
            Some(LogEntry {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                level: LogLevel::Unknown,
                source: source.to_string(),
                message: line.to_string(),
                raw_line: line.to_string(),
                line_number,
                file_path: file_path.to_string(),
            })
        }
    }
    
    fn get_source_name(&self) -> &str {
        "tendermint"
    }
}

pub struct PythonLogParser;

impl LogParser for PythonLogParser {
    fn parse_line(&self, line: &str, source: &str, line_number: u64, file_path: &str) -> Option<LogEntry> {
        // Parse Python log format: 2024-01-01 12:00:00,000 - INFO - message
        let re = Regex::new(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)").ok()?;
        
        if let Some(captures) = re.captures(line) {
            let timestamp_str = captures.get(1)?.as_str();
            let level_str = captures.get(2)?.as_str();
            let message = captures.get(3)?.as_str();
            
            let level = LogLevel::from_str(level_str);
            
            // Parse timestamp
            let timestamp = chrono::NaiveDateTime::parse_from_str(timestamp_str, "%Y-%m-%d %H:%M:%S,%3f")
                .ok()?
                .and_utc()
                .timestamp() as u64;
            
            Some(LogEntry {
                timestamp,
                level,
                source: source.to_string(),
                message: message.to_string(),
                raw_line: line.to_string(),
                line_number,
                file_path: file_path.to_string(),
            })
        } else {
            // Fallback for unstructured lines
            Some(LogEntry {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                level: LogLevel::Unknown,
                source: source.to_string(),
                message: line.to_string(),
                raw_line: line.to_string(),
                line_number,
                file_path: file_path.to_string(),
            })
        }
    }
    
    fn get_source_name(&self) -> &str {
        "python"
    }
}

pub struct IPFSLogParser;

impl LogParser for IPFSLogParser {
    fn parse_line(&self, line: &str, source: &str, line_number: u64, file_path: &str) -> Option<LogEntry> {
        // Parse IPFS log format: 2024-01-01T12:00:00.000Z	INFO	message
        let re = Regex::new(r"^([^\t]+)\t(\w+)\t(.+)").ok()?;
        
        if let Some(captures) = re.captures(line) {
            let timestamp_str = captures.get(1)?.as_str();
            let level_str = captures.get(2)?.as_str();
            let message = captures.get(3)?.as_str();
            
            let level = LogLevel::from_str(level_str);
            
            // Parse ISO timestamp
            let timestamp = chrono::DateTime::parse_from_rfc3339(timestamp_str)
                .ok()?
                .timestamp() as u64;
            
            Some(LogEntry {
                timestamp,
                level,
                source: source.to_string(),
                message: message.to_string(),
                raw_line: line.to_string(),
                line_number,
                file_path: file_path.to_string(),
            })
        } else {
            // Fallback for unstructured lines
            Some(LogEntry {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                level: LogLevel::Unknown,
                source: source.to_string(),
                message: line.to_string(),
                raw_line: line.to_string(),
                line_number,
                file_path: file_path.to_string(),
            })
        }
    }
    
    fn get_source_name(&self) -> &str {
        "ipfs"
    }
}