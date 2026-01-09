//! R3MES Inference Module
//!
//! Provides Tauri commands for AI inference using the BitNet + DoRA + RAG pipeline.
//! Communicates with the backend API for inference operations.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Inference request parameters
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wallet_address: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: i32,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    #[serde(default = "default_top_k")]
    pub top_k: i32,
    #[serde(default)]
    pub skip_rag: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub force_experts: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
}

fn default_max_tokens() -> i32 { 512 }
fn default_temperature() -> f64 { 0.7 }
fn default_top_p() -> f64 { 0.9 }
fn default_top_k() -> i32 { 50 }

/// Expert usage information
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExpertUsage {
    pub id: String,
    pub weight: f64,
}

/// Inference response
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub request_id: String,
    pub text: String,
    pub tokens_generated: i32,
    pub latency_ms: f64,
    pub experts_used: Vec<ExpertUsage>,
    pub rag_context_used: bool,
    pub model_version: String,
    pub credits_used: f64,
}

/// Inference health status
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceHealth {
    pub status: String,
    pub inference_mode: String,
    pub is_ready: bool,
    pub is_healthy: bool,
    pub pipeline_initialized: bool,
    pub model_loaded: bool,
    pub total_requests: i64,
    pub successful_requests: i64,
    pub failed_requests: i64,
    pub avg_latency_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// Inference metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct InferenceMetrics {
    pub serving_engine_requests_total: i64,
    pub serving_engine_requests_success: i64,
    pub serving_engine_requests_failed: i64,
    pub serving_engine_latency_avg_ms: f64,
    pub serving_engine_ready: i32,
    pub serving_engine_healthy: i32,
    pub pipeline_total_requests: i64,
    pub pipeline_error_rate: f64,
    pub cache_vram_used_mb: f64,
    pub cache_ram_used_mb: f64,
    pub cache_hits: i64,
    pub cache_misses: i64,
}

/// Get the API base URL from config or environment
fn get_api_base_url() -> String {
    // Check environment variable first
    if let Ok(url) = std::env::var("R3MES_API_URL") {
        return url;
    }
    
    // Use config - prefer miner backend_url for inference
    let full_config = crate::config::FullConfig::load();
    if !full_config.miner.backend_url.is_empty() {
        return full_config.miner.backend_url.clone();
    }
    
    // Fallback to launcher config rest_url
    let config = crate::config::get_config();
    if !config.rest_url.is_empty() {
        return config.rest_url.clone();
    }
    
    // Final fallback based on network mode
    let network_mode = std::env::var("R3MES_NETWORK")
        .unwrap_or_else(|_| "testnet".to_string())
        .to_lowercase();
    
    if network_mode == "development" || network_mode == "dev" {
        "http://localhost:8000".to_string()
    } else if network_mode == "mainnet" {
        "https://api.r3mes.network".to_string()
    } else {
        "https://testnet-api.r3mes.network".to_string()
    }
}

/// Run inference query
#[tauri::command]
pub async fn run_inference(request: InferenceRequest) -> Result<InferenceResponse, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
    
    let api_url = format!("{}/api/inference/generate", get_api_base_url());
    
    let response = client
        .post(&api_url)
        .json(&request)
        .send()
        .await
        .map_err(|e| format!("Inference request failed: {}", e))?;
    
    if !response.status().is_success() {
        let status = response.status();
        let error_text = response.text().await.unwrap_or_default();
        return Err(format!("Inference failed ({}): {}", status, error_text));
    }
    
    response
        .json::<InferenceResponse>()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))
}

/// Get inference health status
#[tauri::command]
pub async fn get_inference_health() -> Result<InferenceHealth, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
    
    let api_url = format!("{}/api/inference/health", get_api_base_url());
    
    let response = client
        .get(&api_url)
        .send()
        .await
        .map_err(|e| format!("Health check failed: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("Health check failed: {}", response.status()));
    }
    
    response
        .json::<InferenceHealth>()
        .await
        .map_err(|e| format!("Failed to parse health response: {}", e))
}

/// Check if inference is ready
#[tauri::command]
pub async fn check_inference_ready() -> Result<bool, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
    
    let api_url = format!("{}/api/inference/health/ready", get_api_base_url());
    
    match client.get(&api_url).send().await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

/// Get inference metrics
#[tauri::command]
pub async fn get_inference_metrics() -> Result<InferenceMetrics, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
    
    let api_url = format!("{}/api/inference/metrics", get_api_base_url());
    
    let response = client
        .get(&api_url)
        .send()
        .await
        .map_err(|e| format!("Metrics fetch failed: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("Metrics fetch failed: {}", response.status()));
    }
    
    response
        .json::<InferenceMetrics>()
        .await
        .map_err(|e| format!("Failed to parse metrics: {}", e))
}

/// Warmup inference pipeline
#[tauri::command]
pub async fn warmup_inference_pipeline() -> Result<serde_json::Value, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
    
    let api_url = format!("{}/api/inference/pipeline/warmup", get_api_base_url());
    
    let response = client
        .post(&api_url)
        .send()
        .await
        .map_err(|e| format!("Warmup request failed: {}", e))?;
    
    response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| format!("Failed to parse warmup response: {}", e))
}

/// Preload DoRA adapters
#[tauri::command]
pub async fn preload_adapters(adapter_ids: Vec<String>) -> Result<serde_json::Value, String> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
    
    let api_url = format!("{}/api/inference/adapters/preload", get_api_base_url());
    
    let response = client
        .post(&api_url)
        .json(&adapter_ids)
        .send()
        .await
        .map_err(|e| format!("Preload request failed: {}", e))?;
    
    response
        .json::<serde_json::Value>()
        .await
        .map_err(|e| format!("Failed to parse preload response: {}", e))
}
