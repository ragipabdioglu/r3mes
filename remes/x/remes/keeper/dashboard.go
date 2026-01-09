package keeper

import (
	"encoding/json"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
)

// DashboardAPI provides REST API endpoints for the web dashboard
type DashboardAPI struct {
	keeper Keeper
}

// NewDashboardAPI creates a new DashboardAPI instance
func NewDashboardAPI(keeper Keeper) *DashboardAPI {
	return &DashboardAPI{keeper: keeper}
}

// RegisterRoutes registers all dashboard API routes
func (api *DashboardAPI) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/stats", api.handleStats)
	mux.HandleFunc("/blocks", api.handleBlocks)
	mux.HandleFunc("/miners", api.handleMiners)
	mux.HandleFunc("/validators", api.handleValidators)
	mux.HandleFunc("/governance", api.handleGovernance)
	mux.HandleFunc("/health", api.handleHealth)
}

// handleStats returns network statistics
func (api *DashboardAPI) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	stats := map[string]interface{}{
		"totalBlocks":       0,
		"totalTransactions": 0,
		"activeMiners":      0,
		"activeValidators":  0,
		"networkHashrate":   "0 H/s",
		"status":            "operational",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(stats)
}

// handleBlocks returns recent blocks
func (api *DashboardAPI) handleBlocks(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	blocks := []map[string]interface{}{}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(blocks)
}

// handleMiners returns miner information
func (api *DashboardAPI) handleMiners(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	miners := []map[string]interface{}{}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(miners)
}

// handleValidators returns validator information
func (api *DashboardAPI) handleValidators(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	validators := []map[string]interface{}{}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(validators)
}

// handleGovernance returns governance proposals
func (api *DashboardAPI) handleGovernance(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	proposals := []map[string]interface{}{}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(proposals)
}

// handleHealth returns health status
func (api *DashboardAPI) handleHealth(w http.ResponseWriter, r *http.Request) {
	health := map[string]interface{}{
		"status":  "healthy",
		"version": "1.0.0",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// WebSocket support
var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins in development
	},
}

// WebSocketHub manages WebSocket connections
type WebSocketHub struct {
	clients    map[*websocket.Conn]bool
	broadcast  chan []byte
	register   chan *websocket.Conn
	unregister chan *websocket.Conn
	mu         sync.RWMutex
}

var hub = &WebSocketHub{
	clients:    make(map[*websocket.Conn]bool),
	broadcast:  make(chan []byte),
	register:   make(chan *websocket.Conn),
	unregister: make(chan *websocket.Conn),
}

func init() {
	go hub.run()
}

func (h *WebSocketHub) run() {
	for {
		select {
		case client := <-h.register:
			h.mu.Lock()
			h.clients[client] = true
			h.mu.Unlock()
		case client := <-h.unregister:
			h.mu.Lock()
			if _, ok := h.clients[client]; ok {
				delete(h.clients, client)
				client.Close()
			}
			h.mu.Unlock()
		case message := <-h.broadcast:
			h.mu.RLock()
			for client := range h.clients {
				err := client.WriteMessage(websocket.TextMessage, message)
				if err != nil {
					client.Close()
					delete(h.clients, client)
				}
			}
			h.mu.RUnlock()
		}
	}
}

// HandleWebSocket handles WebSocket connections for real-time updates
func (k *Keeper) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		return
	}

	hub.register <- conn

	// Send initial connection message
	initialMsg := map[string]interface{}{
		"type":    "connected",
		"message": "WebSocket connection established",
	}
	msgBytes, _ := json.Marshal(initialMsg)
	conn.WriteMessage(websocket.TextMessage, msgBytes)

	// Handle incoming messages
	go func() {
		defer func() {
			hub.unregister <- conn
		}()
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				break
			}
		}
	}()
}

// BroadcastUpdate sends an update to all connected WebSocket clients
func BroadcastUpdate(eventType string, data interface{}) {
	msg := map[string]interface{}{
		"type": eventType,
		"data": data,
	}
	msgBytes, err := json.Marshal(msg)
	if err != nil {
		return
	}
	hub.broadcast <- msgBytes
}
