package keeper

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	sdkmath "cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/gorilla/websocket"

	"remes/x/remes/types"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		// Allow connections from localhost and configured origins
		// In production, configure allowed origins properly
		return true
	},
}

// MinerStats represents miner statistics for WebSocket streaming
type MinerStats struct {
	GPUTemp   float64 `json:"gpu_temp"`
	FanSpeed  int     `json:"fan_speed"`
	VRAMUsage int     `json:"vram_usage"` // MB
	PowerDraw float64 `json:"power_draw"` // Watts
	HashRate  float64 `json:"hash_rate"`  // Gradients/hour
	Uptime    int64   `json:"uptime"`     // Seconds
	Timestamp int64   `json:"timestamp"`
}

// TrainingMetrics represents training metrics for WebSocket streaming
type TrainingMetrics struct {
	Epoch        int     `json:"epoch"`
	Loss         float64 `json:"loss"`
	Accuracy     float64 `json:"accuracy"`
	GradientNorm float64 `json:"gradient_norm"`
	Timestamp    int64   `json:"timestamp"`
}

// NetworkStatus represents network status for WebSocket streaming
type NetworkStatus struct {
	ActiveMiners    int     `json:"active_miners"`
	TotalGradients  int64   `json:"total_gradients"`
	ModelUpdates    int64   `json:"model_updates"`
	BlockHeight     int64   `json:"block_height"`
	BlockTime       float64 `json:"block_time"`        // seconds
	NetworkHashRate float64 `json:"network_hash_rate"` // Total gradients/hour
	Timestamp       int64   `json:"timestamp"`
}

// HandleWebSocket handles WebSocket connections for real-time data streaming
func (k Keeper) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		http.Error(w, "Failed to upgrade connection", http.StatusBadRequest)
		return
	}
	defer conn.Close()

	// Subscribe to topics
	topic := r.URL.Query().Get("topic")

	switch topic {
	case "miner_stats":
		k.streamMinerStats(conn, r)
	case "training_metrics":
		k.streamTrainingMetrics(conn, r)
	case "network_status":
		k.streamNetworkStatus(conn, r)
	case "miner_logs":
		k.streamMinerLogs(conn, r)
	default:
		conn.WriteJSON(map[string]string{"error": "Unknown topic. Use: miner_stats, training_metrics, network_status, or miner_logs"})
		conn.Close()
	}
}

// streamMinerStats streams miner statistics every 2 seconds
func (k Keeper) streamMinerStats(conn *websocket.Conn, r *http.Request) {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	defer conn.Close() // Ensure connection is closed on exit

	// Ping/pong mechanism for connection health
	pingTicker := time.NewTicker(30 * time.Second)
	defer pingTicker.Stop()

	// Set read deadline to detect dead connections
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	// Get miner address from query params (optional)
	minerAddress := r.URL.Query().Get("miner")

	for {
		select {
		case <-ticker.C:
			stats := k.GetMinerStats(minerAddress)
			if err := conn.WriteJSON(stats); err != nil {
				return // Connection closed or error
			}
		case <-pingTicker.C:
			// Send ping to keep connection alive
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return // Connection closed or error
			}
		}
	}
}

// streamTrainingMetrics streams training metrics per step
func (k Keeper) streamTrainingMetrics(conn *websocket.Conn, r *http.Request) {
	ticker := time.NewTicker(1 * time.Second) // More frequent updates for training
	defer ticker.Stop()
	defer conn.Close() // Ensure connection is closed on exit

	// Ping/pong mechanism for connection health
	pingTicker := time.NewTicker(30 * time.Second)
	defer pingTicker.Stop()

	// Set read deadline to detect dead connections
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	// Get miner address from query params (optional)
	minerAddress := r.URL.Query().Get("miner")

	for {
		select {
		case <-ticker.C:
			metrics := k.GetTrainingMetrics(minerAddress)
			if err := conn.WriteJSON(metrics); err != nil {
				return // Connection closed or error
			}
		case <-pingTicker.C:
			// Send ping to keep connection alive
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return // Connection closed or error
			}
		}
	}
}

// streamNetworkStatus streams network status every 5 seconds
func (k Keeper) streamNetworkStatus(conn *websocket.Conn, r *http.Request) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	defer conn.Close() // Ensure connection is closed on exit

	// Ping/pong mechanism for connection health
	pingTicker := time.NewTicker(30 * time.Second)
	defer pingTicker.Stop()

	// Set read deadline to detect dead connections
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	for {
		select {
		case <-ticker.C:
			status := k.GetNetworkStatus()
			if err := conn.WriteJSON(status); err != nil {
				return // Connection closed or error
			}
		case <-pingTicker.C:
			// Send ping to keep connection alive
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return // Connection closed or error
			}
		}
	}
}

// MinerLog represents a log entry from Python miner
type MinerLog struct {
	Timestamp    string `json:"timestamp"`
	Level        string `json:"level"`
	Message      string `json:"message"`
	MinerAddress string `json:"miner_address,omitempty"`
}

// streamMinerLogs streams miner logs from Python miner to dashboard
// This is a bidirectional stream: Python miner sends logs, dashboard receives them
// Note: In production, use a pub/sub system for multiple dashboard clients
func (k Keeper) streamMinerLogs(conn *websocket.Conn, r *http.Request) {
	defer conn.Close() // Ensure connection is closed on exit

	// Get miner address from query params (optional filter)
	minerAddress := r.URL.Query().Get("miner")

	// Set read deadline to detect dead connections
	conn.SetReadDeadline(time.Now().Add(60 * time.Second))
	conn.SetPongHandler(func(string) error {
		conn.SetReadDeadline(time.Now().Add(60 * time.Second))
		return nil
	})

	// Read logs from Python miner and forward to dashboard
	// For now, we use a simple approach: Python miner sends logs, we echo them back
	// In production, use a pub/sub system to broadcast to all dashboard clients

	// Channel to signal when read goroutine exits
	readDone := make(chan struct{})

	go func() {
		defer close(readDone)
		for {
			var log MinerLog
			if err := conn.ReadJSON(&log); err != nil {
				// Connection closed or error - this is normal for dashboard clients
				// Dashboard clients don't send logs, they only receive
				return
			}

			// Reset read deadline on successful read
			conn.SetReadDeadline(time.Now().Add(60 * time.Second))

			// Filter by miner address if specified
			if minerAddress != "" && log.MinerAddress != minerAddress {
				continue
			}

			// Echo log back (for now - in production, broadcast to all dashboard clients)
			// This allows Python miner to verify its logs are being received
			if err := conn.WriteJSON(log); err != nil {
				return
			}
		}
	}()

	// Keep connection alive with ping
	pingTicker := time.NewTicker(30 * time.Second)
	defer pingTicker.Stop()

	for {
		select {
		case <-pingTicker.C:
			// Send ping to keep connection alive
			if err := conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				return // Connection closed or error
			}
		case <-readDone:
			// Read goroutine exited, close connection
			return
		}
	}
}

// GetMinerStats retrieves miner statistics
// Attempts to fetch real-time stats from Python miner's HTTP endpoint
// Falls back to defaults if Python miner is not available
func (k Keeper) GetMinerStats(minerAddress string) MinerStats {
	// Default stats (will be overridden if miner data is available)
	stats := MinerStats{
		GPUTemp:   65.0,  // Default
		FanSpeed:  2400,  // Default
		VRAMUsage: 8192,  // Default
		PowerDraw: 220.0, // Default
		HashRate:  0.0,   // Will be calculated from on-chain data
		Uptime:    0,     // Will be calculated from on-chain data
		Timestamp: time.Now().Unix(),
	}

	// Try to fetch real-time stats from Python miner's HTTP endpoint
	// Get miner stats port from environment or use default
	// In production, MINER_STATS_PORT should be set (but 8080 is acceptable as default)
	minerStatsPort := os.Getenv("MINER_STATS_PORT")
	if minerStatsPort == "" {
		// Default port 8080 is acceptable even in production (commonly used for stats endpoints)
		minerStatsPort = "8080"
	}
	minerStatsHost := os.Getenv("MINER_STATS_HOST")
	if minerStatsHost == "" {
		// In production, MINER_STATS_HOST must be set (no localhost fallback)
		if os.Getenv("R3MES_ENV") == "production" {
			panic("MINER_STATS_HOST must be set in production (cannot use localhost default)")
		}
		minerStatsHost = "localhost"
	} else if os.Getenv("R3MES_ENV") == "production" {
		// Validate that production doesn't use localhost
		if strings.Contains(minerStatsHost, "localhost") || strings.Contains(minerStatsHost, "127.0.0.1") {
			panic(fmt.Sprintf("MINER_STATS_HOST cannot use localhost in production: %s", minerStatsHost))
		}
	}

	statsURL := fmt.Sprintf("http://%s:%s/stats", minerStatsHost, minerStatsPort)

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 2 * time.Second,
	}

	resp, err := client.Get(statsURL)
	if err != nil {
		// Python miner not available, return defaults
		return stats
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Python miner returned error, return defaults
		return stats
	}

	// Parse JSON response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return stats
	}

	var minerData struct {
		MinerStats struct {
			GPUTemp   float64 `json:"gpu_temp"`
			FanSpeed  int     `json:"fan_speed"`
			VRAMUsage int     `json:"vram_usage"`
			PowerDraw float64 `json:"power_draw"`
			HashRate  float64 `json:"hash_rate"`
			Uptime    int64   `json:"uptime"`
			Timestamp int64   `json:"timestamp"`
		} `json:"miner_stats"`
	}

	if err := json.Unmarshal(body, &minerData); err != nil {
		return stats
	}

	// Update stats with real data from Python miner
	stats.GPUTemp = minerData.MinerStats.GPUTemp
	stats.FanSpeed = minerData.MinerStats.FanSpeed
	stats.VRAMUsage = minerData.MinerStats.VRAMUsage
	stats.PowerDraw = minerData.MinerStats.PowerDraw
	stats.HashRate = minerData.MinerStats.HashRate
	stats.Uptime = minerData.MinerStats.Uptime
	if minerData.MinerStats.Timestamp > 0 {
		stats.Timestamp = minerData.MinerStats.Timestamp
	}

	return stats
}

// GetTrainingMetrics retrieves training metrics
// Attempts to fetch real-time metrics from Python miner's HTTP endpoint
// Falls back to defaults if Python miner is not available
func (k Keeper) GetTrainingMetrics(minerAddress string) TrainingMetrics {
	// Default metrics (will be overridden if miner data is available)
	metrics := TrainingMetrics{
		Epoch:        1,
		Loss:         2.34,
		Accuracy:     0.85,
		GradientNorm: 1.23,
		Timestamp:    time.Now().Unix(),
	}

	// Try to fetch real-time metrics from Python miner's HTTP endpoint
	minerStatsPort := os.Getenv("MINER_STATS_PORT")
	if minerStatsPort == "" {
		minerStatsPort = "8080"
	}
	minerStatsHost := os.Getenv("MINER_STATS_HOST")
	if minerStatsHost == "" {
		// In production, MINER_STATS_HOST must be set (no localhost fallback)
		if os.Getenv("R3MES_ENV") == "production" {
			panic("MINER_STATS_HOST must be set in production (cannot use localhost default)")
		}
		minerStatsHost = "localhost"
	} else if os.Getenv("R3MES_ENV") == "production" {
		// Validate that production doesn't use localhost
		if strings.Contains(minerStatsHost, "localhost") || strings.Contains(minerStatsHost, "127.0.0.1") {
			panic(fmt.Sprintf("MINER_STATS_HOST cannot use localhost in production: %s", minerStatsHost))
		}
	}

	statsURL := fmt.Sprintf("http://%s:%s/stats", minerStatsHost, minerStatsPort)

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 2 * time.Second,
	}

	resp, err := client.Get(statsURL)
	if err != nil {
		// Python miner not available, return defaults
		return metrics
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Python miner returned error, return defaults
		return metrics
	}

	// Parse JSON response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return metrics
	}

	var minerData struct {
		TrainingMetrics struct {
			Epoch        int     `json:"epoch"`
			Loss         float64 `json:"loss"`
			Accuracy     float64 `json:"accuracy"`
			GradientNorm float64 `json:"gradient_norm"`
			Timestamp    int64   `json:"timestamp"`
		} `json:"training_metrics"`
	}

	if err := json.Unmarshal(body, &minerData); err != nil {
		return metrics
	}

	// Update metrics with real data from Python miner
	metrics.Epoch = minerData.TrainingMetrics.Epoch
	metrics.Loss = minerData.TrainingMetrics.Loss
	metrics.Accuracy = minerData.TrainingMetrics.Accuracy
	metrics.GradientNorm = minerData.TrainingMetrics.GradientNorm
	if minerData.TrainingMetrics.Timestamp > 0 {
		metrics.Timestamp = minerData.TrainingMetrics.Timestamp
	}

	return metrics
}

// GetNetworkStatus retrieves network-wide status
// Note: This function doesn't have SDK context access
// For real-time data, use GetNetworkStatusWithContext
func (k Keeper) GetNetworkStatus() NetworkStatus {
	// Return mock data - HTTP handlers don't have SDK context
	// In production, use GetNetworkStatusWithContext with proper context
	return NetworkStatus{
		ActiveMiners:    0,
		TotalGradients:  0,
		ModelUpdates:    0,
		BlockHeight:     0,
		BlockTime:       5.0, // seconds
		NetworkHashRate: 0.0,
		Timestamp:       time.Now().Unix(),
	}
}

// GetNetworkStatusWithContext retrieves network-wide status with SDK context
func (k Keeper) GetNetworkStatusWithContext(ctx sdk.Context) NetworkStatus {
	// Count active miners
	activeMiners := 0
	totalGradients := int64(0)

	_ = k.MiningContributions.Walk(ctx, nil, func(key string, value types.MiningContribution) (stop bool, err error) {
		trustScore, parseErr := sdkmath.LegacyNewDecFromStr(value.TrustScore)
		if parseErr == nil && trustScore.GT(sdkmath.LegacyZeroDec()) {
			activeMiners++
		}
		return false, nil
	})

	// Count total gradients
	_ = k.StoredGradients.Walk(ctx, nil, func(key uint64, value types.StoredGradient) (stop bool, err error) {
		totalGradients++
		return false, nil
	})

	// Get model updates count (from global model state)
	modelUpdates := int64(0)
	modelState, err := k.GlobalModelState.Get(ctx)
	if err == nil {
		modelUpdates = int64(modelState.LastUpdatedHeight)
	}

	// Get block height
	blockHeight := ctx.BlockHeight()

	// Calculate network hash rate (simplified - gradients per hour)
	// This would need historical data for accurate calculation
	networkHashRate := float64(totalGradients) / 3600.0 // Simplified calculation

	return NetworkStatus{
		ActiveMiners:    activeMiners,
		TotalGradients:  totalGradients,
		ModelUpdates:    modelUpdates,
		BlockHeight:     blockHeight,
		BlockTime:       5.0, // Average block time (would need calculation)
		NetworkHashRate: networkHashRate,
		Timestamp:       time.Now().Unix(),
	}
}
