//! WebSocket client for real-time communication with R3MES services
//! 
//! Provides real-time updates for mining stats, process status, blockchain events,
//! and other live data streams.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, mpsc};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use futures_util::{SinkExt, StreamExt};
use once_cell::sync::Lazy;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    pub message_type: String,
    pub data: serde_json::Value,
    pub timestamp: u64,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningUpdate {
    pub hashrate: f64,
    pub loss: f64,
    pub gpu_temp: f64,
    pub vram_usage: u64,
    pub earnings: f64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessUpdate {
    pub process_name: String,
    pub status: String,
    pub pid: Option<u32>,
    pub cpu_usage: f64,
    pub memory_usage: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockchainUpdate {
    pub block_height: u64,
    pub sync_status: String,
    pub peer_count: u32,
    pub latest_block_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemUpdate {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_io: NetworkIO,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIO {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
}

pub type MessageHandler = Arc<dyn Fn(WebSocketMessage) + Send + Sync>;

pub struct WebSocketClient {
    url: String,
    connected: Arc<RwLock<bool>>,
    handlers: Arc<RwLock<HashMap<String, MessageHandler>>>,
    sender: Option<mpsc::UnboundedSender<Message>>,
    reconnect_attempts: Arc<RwLock<u32>>,
    max_reconnect_attempts: u32,
    reconnect_delay: std::time::Duration,
}

impl WebSocketClient {
    /// Create a new WebSocket client
    pub fn new(url: String) -> Self {
        Self {
            url,
            connected: Arc::new(RwLock::new(false)),
            handlers: Arc::new(RwLock::new(HashMap::new())),
            sender: None,
            reconnect_attempts: Arc::new(RwLock::new(0)),
            max_reconnect_attempts: 5,
            reconnect_delay: std::time::Duration::from_secs(5),
        }
    }
    
    /// Connect to the WebSocket server
    pub async fn connect(&mut self) -> Result<(), String> {
        log::info!("Connecting to WebSocket: {}", self.url);
        
        let (ws_stream, _) = connect_async(&self.url)
            .await
            .map_err(|e| format!("Failed to connect to WebSocket: {}", e))?;
        
        let (mut ws_sender, mut ws_receiver) = ws_stream.split();
        let (tx, mut rx) = mpsc::unbounded_channel::<Message>();
        
        self.sender = Some(tx);
        *self.connected.write().await = true;
        *self.reconnect_attempts.write().await = 0;
        
        log::info!("WebSocket connected successfully");
        
        // Spawn sender task
        let _sender_task = {
            tokio::spawn(async move {
                while let Some(message) = rx.recv().await {
                    if let Err(e) = ws_sender.send(message).await {
                        log::error!("Failed to send WebSocket message: {}", e);
                        break;
                    }
                }
            })
        };
        
        // Spawn receiver task
        let handlers = Arc::clone(&self.handlers);
        let connected = Arc::clone(&self.connected);
        let _receiver_task = {
            tokio::spawn(async move {
                while let Some(message) = ws_receiver.next().await {
                    match message {
                        Ok(Message::Text(text)) => {
                            if let Ok(ws_message) = serde_json::from_str::<WebSocketMessage>(&text) {
                                Self::handle_message(ws_message, &handlers).await;
                            } else {
                                log::warn!("Failed to parse WebSocket message: {}", text);
                            }
                        }
                        Ok(Message::Binary(data)) => {
                            log::debug!("Received binary message: {} bytes", data.len());
                        }
                        Ok(Message::Ping(data)) => {
                            log::debug!("Received ping: {:?}", data);
                        }
                        Ok(Message::Pong(data)) => {
                            log::debug!("Received pong: {:?}", data);
                        }
                        Ok(Message::Close(_)) => {
                            log::info!("WebSocket connection closed by server");
                            *connected.write().await = false;
                            break;
                        }
                        Ok(Message::Frame(_)) => {
                            // Raw frame - ignore
                            log::debug!("Received raw frame");
                        }
                        Err(e) => {
                            log::error!("WebSocket error: {}", e);
                            *connected.write().await = false;
                            break;
                        }
                    }
                }
            })
        };
        
        // Start reconnection task
        let _url = self.url.clone();
        let connected_clone = Arc::clone(&self.connected);
        let reconnect_attempts = Arc::clone(&self.reconnect_attempts);
        let max_attempts = self.max_reconnect_attempts;
        let delay = self.reconnect_delay;
        
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(10)).await;
                
                if !*connected_clone.read().await {
                    let attempts = *reconnect_attempts.read().await;
                    if attempts < max_attempts {
                        log::info!("Attempting to reconnect... (attempt {}/{})", attempts + 1, max_attempts);
                        
                        // This is a simplified reconnection - in practice, you'd need
                        // to recreate the entire connection
                        *reconnect_attempts.write().await += 1;
                        tokio::time::sleep(delay).await;
                    } else {
                        log::error!("Max reconnection attempts reached, giving up");
                        break;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Disconnect from the WebSocket server
    pub async fn disconnect(&mut self) {
        if let Some(sender) = &self.sender {
            let _ = sender.send(Message::Close(None));
        }
        
        *self.connected.write().await = false;
        self.sender = None;
        
        log::info!("WebSocket disconnected");
    }
    
    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        *self.connected.read().await
    }
    
    /// Send a message to the server
    pub async fn send_message(&self, message: WebSocketMessage) -> Result<(), String> {
        if let Some(sender) = &self.sender {
            let json = serde_json::to_string(&message)
                .map_err(|e| format!("Failed to serialize message: {}", e))?;
            
            sender.send(Message::Text(json))
                .map_err(|e| format!("Failed to send message: {}", e))?;
            
            Ok(())
        } else {
            Err("Not connected".to_string())
        }
    }
    
    /// Subscribe to a message type
    pub async fn subscribe(&self, message_type: String, handler: MessageHandler) {
        let mut handlers = self.handlers.write().await;
        handlers.insert(message_type.clone(), handler);
        
        // Send subscription message to server
        let subscribe_msg = WebSocketMessage {
            message_type: "subscribe".to_string(),
            data: serde_json::json!({ "type": message_type }),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            source: "client".to_string(),
        };
        
        let _ = self.send_message(subscribe_msg).await;
    }
    
    /// Unsubscribe from a message type
    pub async fn unsubscribe(&self, message_type: &str) {
        let mut handlers = self.handlers.write().await;
        handlers.remove(message_type);
        
        // Send unsubscription message to server
        let unsubscribe_msg = WebSocketMessage {
            message_type: "unsubscribe".to_string(),
            data: serde_json::json!({ "type": message_type }),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            source: "client".to_string(),
        };
        
        let _ = self.send_message(unsubscribe_msg).await;
    }
    
    /// Handle incoming message
    async fn handle_message(
        message: WebSocketMessage,
        handlers: &Arc<RwLock<HashMap<String, MessageHandler>>>,
    ) {
        let handlers_read = handlers.read().await;
        
        if let Some(handler) = handlers_read.get(&message.message_type) {
            handler(message);
        } else {
            log::debug!("No handler for message type: {}", message.message_type);
        }
    }
    
    /// Send ping to keep connection alive
    pub async fn ping(&self) -> Result<(), String> {
        if let Some(sender) = &self.sender {
            sender.send(Message::Ping(vec![]))
                .map_err(|e| format!("Failed to send ping: {}", e))?;
            Ok(())
        } else {
            Err("Not connected".to_string())
        }
    }
    
    /// Get connection statistics
    pub async fn get_stats(&self) -> ConnectionStats {
        ConnectionStats {
            connected: *self.connected.read().await,
            reconnect_attempts: *self.reconnect_attempts.read().await,
            subscriptions: self.handlers.read().await.len(),
            url: self.url.clone(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ConnectionStats {
    pub connected: bool,
    pub reconnect_attempts: u32,
    pub subscriptions: usize,
    pub url: String,
}

// Global WebSocket client instance
static WS_CLIENT: Lazy<Arc<RwLock<Option<WebSocketClient>>>> = 
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Get the global WebSocket client
pub fn get_ws_client() -> Arc<RwLock<Option<WebSocketClient>>> {
    Arc::clone(&WS_CLIENT)
}

/// Initialize the global WebSocket client
pub async fn init_ws_client(url: String) -> Result<(), String> {
    let mut client = WebSocketClient::new(url);
    client.connect().await?;
    
    *WS_CLIENT.write().await = Some(client);
    Ok(())
}

/// Convenience functions for common subscriptions
pub async fn subscribe_to_mining_updates<F>(handler: F) -> Result<(), String>
where
    F: Fn(MiningUpdate) + Send + Sync + 'static,
{
    let ws_client = WS_CLIENT.read().await;
    if let Some(client) = ws_client.as_ref() {
        let handler = Arc::new(move |message: WebSocketMessage| {
            if let Ok(update) = serde_json::from_value::<MiningUpdate>(message.data) {
                handler(update);
            }
        });
        
        client.subscribe("mining_update".to_string(), handler).await;
        Ok(())
    } else {
        Err("WebSocket client not initialized".to_string())
    }
}

pub async fn subscribe_to_process_updates<F>(handler: F) -> Result<(), String>
where
    F: Fn(ProcessUpdate) + Send + Sync + 'static,
{
    let ws_client = WS_CLIENT.read().await;
    if let Some(client) = ws_client.as_ref() {
        let handler = Arc::new(move |message: WebSocketMessage| {
            if let Ok(update) = serde_json::from_value::<ProcessUpdate>(message.data) {
                handler(update);
            }
        });
        
        client.subscribe("process_update".to_string(), handler).await;
        Ok(())
    } else {
        Err("WebSocket client not initialized".to_string())
    }
}

pub async fn subscribe_to_blockchain_updates<F>(handler: F) -> Result<(), String>
where
    F: Fn(BlockchainUpdate) + Send + Sync + 'static,
{
    let ws_client = WS_CLIENT.read().await;
    if let Some(client) = ws_client.as_ref() {
        let handler = Arc::new(move |message: WebSocketMessage| {
            if let Ok(update) = serde_json::from_value::<BlockchainUpdate>(message.data) {
                handler(update);
            }
        });
        
        client.subscribe("blockchain_update".to_string(), handler).await;
        Ok(())
    } else {
        Err("WebSocket client not initialized".to_string())
    }
}

pub async fn subscribe_to_system_updates<F>(handler: F) -> Result<(), String>
where
    F: Fn(SystemUpdate) + Send + Sync + 'static,
{
    let ws_client = WS_CLIENT.read().await;
    if let Some(client) = ws_client.as_ref() {
        let handler = Arc::new(move |message: WebSocketMessage| {
            if let Ok(update) = serde_json::from_value::<SystemUpdate>(message.data) {
                handler(update);
            }
        });
        
        client.subscribe("system_update".to_string(), handler).await;
        Ok(())
    } else {
        Err("WebSocket client not initialized".to_string())
    }
}

/// Send a request for current status
pub async fn request_current_status() -> Result<(), String> {
    let ws_client = WS_CLIENT.read().await;
    if let Some(client) = ws_client.as_ref() {
        let message = WebSocketMessage {
            message_type: "request_status".to_string(),
            data: serde_json::json!({}),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            source: "desktop_launcher".to_string(),
        };
        
        client.send_message(message).await
    } else {
        Err("WebSocket client not initialized".to_string())
    }
}

/// Send a command to a service
pub async fn send_command(service: String, command: String, params: serde_json::Value) -> Result<(), String> {
    let ws_client = WS_CLIENT.read().await;
    if let Some(client) = ws_client.as_ref() {
        let message = WebSocketMessage {
            message_type: "command".to_string(),
            data: serde_json::json!({
                "service": service,
                "command": command,
                "params": params
            }),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            source: "desktop_launcher".to_string(),
        };
        
        client.send_message(message).await
    } else {
        Err("WebSocket client not initialized".to_string())
    }
}

/// Cleanup WebSocket client
pub async fn cleanup_ws_client() {
    let mut ws_client = WS_CLIENT.write().await;
    if let Some(mut client) = ws_client.take() {
        client.disconnect().await;
    }
}

/// WebSocket client builder for easier configuration
pub struct WebSocketClientBuilder {
    url: String,
    max_reconnect_attempts: u32,
    reconnect_delay: std::time::Duration,
    ping_interval: Option<std::time::Duration>,
}

impl WebSocketClientBuilder {
    pub fn new(url: String) -> Self {
        Self {
            url,
            max_reconnect_attempts: 5,
            reconnect_delay: std::time::Duration::from_secs(5),
            ping_interval: Some(std::time::Duration::from_secs(30)),
        }
    }
    
    pub fn max_reconnect_attempts(mut self, attempts: u32) -> Self {
        self.max_reconnect_attempts = attempts;
        self
    }
    
    pub fn reconnect_delay(mut self, delay: std::time::Duration) -> Self {
        self.reconnect_delay = delay;
        self
    }
    
    pub fn ping_interval(mut self, interval: Option<std::time::Duration>) -> Self {
        self.ping_interval = interval;
        self
    }
    
    pub fn build(self) -> WebSocketClient {
        let mut client = WebSocketClient::new(self.url);
        client.max_reconnect_attempts = self.max_reconnect_attempts;
        client.reconnect_delay = self.reconnect_delay;
        
        // Start ping task if interval is set
        if let Some(interval) = self.ping_interval {
            let connected = Arc::clone(&client.connected);
            let sender = client.sender.clone();
            
            tokio::spawn(async move {
                let mut ping_interval = tokio::time::interval(interval);
                loop {
                    ping_interval.tick().await;
                    
                    if *connected.read().await {
                        if let Some(sender) = &sender {
                            let _ = sender.send(Message::Ping(vec![]));
                        }
                    }
                }
            });
        }
        
        client
    }
}