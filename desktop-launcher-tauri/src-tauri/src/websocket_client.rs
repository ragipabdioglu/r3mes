//! WebSocket Client for Real-time Updates
//!
//! This module provides WebSocket connectivity to the R3MES backend
//! for real-time status updates, replacing HTTP polling.
//!
//! Channels:
//! - network_status: Network-wide statistics
//! - block_updates: New block notifications
//! - miner_stats: Miner-specific statistics (requires auth)
//! - training_metrics: Training progress updates (requires auth)

use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use log::{info, warn, error, debug};

/// WebSocket connection state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Error(String),
}

/// WebSocket message types from backend
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WebSocketMessage {
    #[serde(rename = "network_status")]
    NetworkStatus(NetworkStatusData),
    #[serde(rename = "block_update")]
    BlockUpdate(BlockUpdateData),
    #[serde(rename = "miner_stats")]
    MinerStats(MinerStatsData),
    #[serde(rename = "training_metrics")]
    TrainingMetrics(TrainingMetricsData),
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "pong")]
    Pong,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatusData {
    pub active_miners: u64,
    pub total_hashrate: f64,
    pub block_height: u64,
    pub network_difficulty: f64,
    pub connected_peers: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockUpdateData {
    pub height: u64,
    pub hash: String,
    pub proposer: String,
    pub timestamp: i64,
    pub tx_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinerStatsData {
    pub wallet_address: String,
    pub hashrate: f64,
    pub gpu_temperature: f64,
    pub blocks_found: u64,
    pub earnings_today: f64,
    pub uptime_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetricsData {
    pub training_round_id: u64,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub loss: f64,
    pub accuracy: f64,
    pub gradients_submitted: u64,
}

/// WebSocket client for R3MES backend
pub struct WebSocketClient {
    /// Backend WebSocket URL
    ws_url: String,
    /// Authentication token (for protected channels)
    auth_token: Option<String>,
    /// Current connection state
    state: Arc<RwLock<ConnectionState>>,
    /// Channel for sending messages to the WebSocket
    tx: Option<mpsc::Sender<String>>,
    /// Subscribed channels
    subscribed_channels: Arc<RwLock<Vec<String>>>,
    /// Reconnection attempts
    reconnect_attempts: Arc<RwLock<u32>>,
    /// Maximum reconnection attempts
    max_reconnect_attempts: u32,
}

impl WebSocketClient {
    /// Create a new WebSocket client
    pub fn new(backend_url: &str, auth_token: Option<String>) -> Self {
        // Convert HTTP URL to WebSocket URL
        let ws_url = backend_url
            .replace("https://", "wss://")
            .replace("http://", "ws://");
        
        Self {
            ws_url: format!("{}/ws", ws_url),
            auth_token,
            state: Arc::new(RwLock::new(ConnectionState::Disconnected)),
            tx: None,
            subscribed_channels: Arc::new(RwLock::new(Vec::new())),
            reconnect_attempts: Arc::new(RwLock::new(0)),
            max_reconnect_attempts: 5,
        }
    }

    /// Get current connection state
    pub async fn get_state(&self) -> ConnectionState {
        self.state.read().await.clone()
    }

    /// Connect to WebSocket server
    pub async fn connect(&mut self) -> Result<mpsc::Receiver<WebSocketMessage>, String> {
        // Update state
        *self.state.write().await = ConnectionState::Connecting;
        
        // Build URL with auth token if available
        let url = if let Some(ref token) = self.auth_token {
            format!("{}?token={}", self.ws_url, token)
        } else {
            self.ws_url.clone()
        };

        info!("Connecting to WebSocket: {}", self.ws_url);

        // Connect to WebSocket
        let (ws_stream, _) = connect_async(&url)
            .await
            .map_err(|e| format!("WebSocket connection failed: {}", e))?;

        let (mut write, mut read) = ws_stream.split();

        // Create channels for communication
        let (msg_tx, msg_rx) = mpsc::channel::<WebSocketMessage>(100);
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<String>(100);

        self.tx = Some(cmd_tx);
        *self.state.write().await = ConnectionState::Connected;
        *self.reconnect_attempts.write().await = 0;

        info!("WebSocket connected successfully");

        // Spawn task to handle outgoing messages
        let state_clone = self.state.clone();
        tokio::spawn(async move {
            while let Some(msg) = cmd_rx.recv().await {
                if write.send(Message::Text(msg)).await.is_err() {
                    error!("Failed to send WebSocket message");
                    *state_clone.write().await = ConnectionState::Error("Send failed".to_string());
                    break;
                }
            }
        });

        // Spawn task to handle incoming messages
        let state_clone = self.state.clone();
        tokio::spawn(async move {
            while let Some(msg_result) = read.next().await {
                match msg_result {
                    Ok(Message::Text(text)) => {
                        match serde_json::from_str::<WebSocketMessage>(&text) {
                            Ok(ws_msg) => {
                                // Handle ping/pong internally
                                if matches!(ws_msg, WebSocketMessage::Ping) {
                                    debug!("Received ping, sending pong");
                                    continue;
                                }
                                
                                if msg_tx.send(ws_msg).await.is_err() {
                                    warn!("Message receiver dropped");
                                    break;
                                }
                            }
                            Err(e) => {
                                warn!("Failed to parse WebSocket message: {}", e);
                            }
                        }
                    }
                    Ok(Message::Ping(_)) => {
                        debug!("Received WebSocket ping");
                    }
                    Ok(Message::Close(_)) => {
                        info!("WebSocket connection closed by server");
                        *state_clone.write().await = ConnectionState::Disconnected;
                        break;
                    }
                    Err(e) => {
                        error!("WebSocket error: {}", e);
                        *state_clone.write().await = ConnectionState::Error(e.to_string());
                        break;
                    }
                    _ => {}
                }
            }
        });

        Ok(msg_rx)
    }

    /// Subscribe to a channel
    pub async fn subscribe(&self, channel: &str) -> Result<(), String> {
        let tx = self.tx.as_ref().ok_or("Not connected")?;
        
        let subscribe_msg = serde_json::json!({
            "action": "subscribe",
            "channel": channel
        }).to_string();

        tx.send(subscribe_msg)
            .await
            .map_err(|e| format!("Failed to subscribe: {}", e))?;

        // Track subscription
        self.subscribed_channels.write().await.push(channel.to_string());
        
        info!("Subscribed to channel: {}", channel);
        Ok(())
    }

    /// Unsubscribe from a channel
    pub async fn unsubscribe(&self, channel: &str) -> Result<(), String> {
        let tx = self.tx.as_ref().ok_or("Not connected")?;
        
        let unsubscribe_msg = serde_json::json!({
            "action": "unsubscribe",
            "channel": channel
        }).to_string();

        tx.send(unsubscribe_msg)
            .await
            .map_err(|e| format!("Failed to unsubscribe: {}", e))?;

        // Remove from tracked subscriptions
        self.subscribed_channels.write().await.retain(|c| c != channel);
        
        info!("Unsubscribed from channel: {}", channel);
        Ok(())
    }

    /// Disconnect from WebSocket server
    pub async fn disconnect(&mut self) {
        if let Some(tx) = self.tx.take() {
            drop(tx);
        }
        *self.state.write().await = ConnectionState::Disconnected;
        info!("WebSocket disconnected");
    }

    /// Check if connected
    pub async fn is_connected(&self) -> bool {
        matches!(*self.state.read().await, ConnectionState::Connected)
    }
}

/// Global WebSocket client instance
static WS_CLIENT: once_cell::sync::OnceCell<Arc<RwLock<Option<WebSocketClient>>>> = 
    once_cell::sync::OnceCell::new();

/// Get or create the global WebSocket client
pub fn get_ws_client() -> Arc<RwLock<Option<WebSocketClient>>> {
    WS_CLIENT.get_or_init(|| Arc::new(RwLock::new(None))).clone()
}

/// Initialize WebSocket connection
pub async fn init_websocket(backend_url: &str, auth_token: Option<String>) -> Result<(), String> {
    let client = WebSocketClient::new(backend_url, auth_token);
    *get_ws_client().write().await = Some(client);
    Ok(())
}

/// Connect and subscribe to default channels
pub async fn connect_and_subscribe() -> Result<mpsc::Receiver<WebSocketMessage>, String> {
    let client_lock = get_ws_client();
    let mut client_guard = client_lock.write().await;
    
    let client = client_guard.as_mut().ok_or("WebSocket client not initialized")?;
    
    let rx = client.connect().await?;
    
    // Subscribe to public channels by default
    client.subscribe("network_status").await?;
    client.subscribe("block_updates").await?;
    
    Ok(rx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_conversion() {
        let client = WebSocketClient::new("https://api.r3mes.network", None);
        assert!(client.ws_url.starts_with("wss://"));
        
        let client = WebSocketClient::new("http://localhost:8000", None);
        assert!(client.ws_url.starts_with("ws://"));
    }
}
