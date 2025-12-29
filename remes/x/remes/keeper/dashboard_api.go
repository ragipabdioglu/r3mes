package keeper

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	sdkmath "cosmossdk.io/math"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/cosmos/cosmos-sdk/types/query"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"remes/x/remes/types"
)

// DashboardCache stores cached data for dashboard API responses
type DashboardCache struct {
	MinersCache struct {
		Data      interface{}
		ExpiresAt time.Time
	}
	StatisticsCache struct {
		Data      interface{}
		ExpiresAt time.Time
	}
	mu sync.RWMutex
}

// DashboardAPI handles REST API endpoints for the web dashboard
type DashboardAPI struct {
	keeper            Keeper
	queryServer       types.QueryServer
	grpcAddr          string // gRPC server address (configurable via environment or default)
	tendermintRPCAddr string // Tendermint RPC address (configurable via environment or default)
	cache             DashboardCache
	cacheTTL          time.Duration // Cache time-to-live (default: 5 seconds)
}

// NewDashboardAPI creates a new DashboardAPI instance
func NewDashboardAPI(keeper Keeper) *DashboardAPI {
	// Get gRPC address from environment or use default
	// In production, R3MES_GRPC_ADDR must be set (no localhost fallback)
	grpcAddr := getEnvOrDefault("R3MES_GRPC_ADDR", "localhost:9090")
	// Get Tendermint RPC address from environment or use default
	// In production, R3MES_TENDERMINT_RPC_ADDR must be set (no localhost fallback)
	tendermintRPCAddr := getEnvOrDefault("R3MES_TENDERMINT_RPC_ADDR", "http://localhost:26657")

	// Get cache TTL from environment or use default (30 seconds for production)
	// Default increased from 5 to 30 seconds for better performance
	cacheTTLSeconds := 30
	if ttlStr := os.Getenv("R3MES_DASHBOARD_CACHE_TTL_SECONDS"); ttlStr != "" {
		if ttl, err := strconv.Atoi(ttlStr); err == nil && ttl > 0 {
			cacheTTLSeconds = ttl
		}
	}

	return &DashboardAPI{
		keeper:            keeper,
		queryServer:       NewQueryServerImpl(keeper),
		grpcAddr:          grpcAddr,
		tendermintRPCAddr: tendermintRPCAddr,
		cache:             DashboardCache{},
		cacheTTL:          time.Duration(cacheTTLSeconds) * time.Second,
	}
}

// getEnvOrDefault gets environment variable or returns default value
// In production, localhost defaults are not allowed
func getEnvOrDefault(key, defaultValue string) string {
	value := os.Getenv(key)
	if value != "" {
		// In production, validate that value is not localhost
		if os.Getenv("R3MES_ENV") == "production" {
			if strings.Contains(value, "localhost") || strings.Contains(value, "127.0.0.1") {
				panic(fmt.Sprintf("%s cannot use localhost in production: %s", key, value))
			}
		}
		return value
	}
	
	// In production, do not allow localhost defaults
	if os.Getenv("R3MES_ENV") == "production" {
		if strings.Contains(defaultValue, "localhost") || strings.Contains(defaultValue, "127.0.0.1") {
			panic(fmt.Sprintf("%s must be set in production (cannot use localhost default: %s)", key, defaultValue))
		}
	}
	
	return defaultValue
}

// getGRPCClient creates a gRPC client connection to the query server
func (api *DashboardAPI) getGRPCClient() (types.QueryClient, func(), error) {
	// Use background context for connection, timeout will be handled per-request
	// Use WithInsecure for local development (localhost)
	// In production, use proper TLS credentials
	conn, err := grpc.Dial(api.grpcAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithTimeout(5*time.Second))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to connect to gRPC server at %s: %w", api.grpcAddr, err)
	}

	client := types.NewQueryClient(conn)
	cleanup := func() {
		conn.Close()
	}

	return client, cleanup, nil
}

// getSDKContextFromRequest extracts SDK context from HTTP request
// NOTE: HTTP handlers in Cosmos SDK don't have direct access to SDK context
// This is a known limitation. For production, consider:
// 1. Using gRPC query endpoints instead of HTTP handlers
// 2. Passing context through middleware
// 3. Using query server to get context
func (api *DashboardAPI) getSDKContextFromRequest(r *http.Request) (sdk.Context, error) {
	// HTTP handlers don't have SDK context access
	// Return error to indicate context is not available
	// Dashboard API endpoints should use gRPC query endpoints for real-time data
	return sdk.Context{}, fmt.Errorf("SDK context not available in HTTP handler")
}

// RegisterRoutes registers dashboard API routes
func (api *DashboardAPI) RegisterRoutes(mux *http.ServeMux) {
	// Apply panic recovery middleware to all routes
	// Wrap all handlers with panic recovery
	wrapWithPanicRecovery := func(handler http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if err := recover(); err != nil {
					// Log panic
					fmt.Fprintf(os.Stderr, "Panic in dashboard API: %v\n", err)
					// Stack trace is logged via panic recovery middleware
					
					// Return error response
					w.WriteHeader(http.StatusInternalServerError)
					w.Header().Set("Content-Type", "application/json")
					
					isProduction := os.Getenv("R3MES_ENV") == "production"
					if isProduction {
						fmt.Fprintf(w, `{"error":"INTERNAL_SERVER_ERROR","message":"An internal error occurred"}`)
					} else {
						fmt.Fprintf(w, `{"error":"INTERNAL_SERVER_ERROR","message":"%v"}`, err)
					}
				}
			}()
			handler(w, r)
		}
	}
	
	// CORS middleware with configurable allowed origins
	// Production-ready: strict origin validation, no wildcard in production
	corsHandler := func(next http.HandlerFunc) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			// Get allowed origins from environment variable or use default
			// In production, CORS_ALLOWED_ORIGINS must be set (no localhost fallback)
			allowedOriginsEnv := os.Getenv("CORS_ALLOWED_ORIGINS")
			isProduction := os.Getenv("R3MES_ENV") == "production"
			
			var allowedOrigins string
			if allowedOriginsEnv != "" {
				allowedOrigins = allowedOriginsEnv
			} else {
				if isProduction {
					// Production: must be set explicitly
					http.Error(w, "CORS_ALLOWED_ORIGINS must be set in production", http.StatusInternalServerError)
					return
				}
				// Development: use default
				allowedOrigins = "http://localhost:3000"
			}

			// In production, validate that no wildcard or localhost is used
			if isProduction {
				allowedList := strings.Split(allowedOrigins, ",")
				for _, allowed := range allowedList {
					allowed = strings.TrimSpace(allowed)
					if allowed == "*" {
						http.Error(w, "Wildcard CORS origin is not allowed in production", http.StatusInternalServerError)
						return
					}
					if strings.Contains(allowed, "localhost") || strings.Contains(allowed, "127.0.0.1") {
						http.Error(w, "Localhost CORS origin is not allowed in production", http.StatusInternalServerError)
						return
					}
				}
			}

			// Validate origin against allowed list
			origin := r.Header.Get("Origin")
			if origin != "" {
				// Check if origin is in allowed list (comma-separated)
				allowedList := strings.Split(allowedOrigins, ",")
				originAllowed := false
				for _, allowed := range allowedList {
					allowed = strings.TrimSpace(allowed)
					if allowed == "*" && !isProduction {
						// Development mode: allow all origins
						w.Header().Set("Access-Control-Allow-Origin", "*")
						originAllowed = true
						break
					} else if origin == allowed {
						// Production mode: only allow specific origins
						w.Header().Set("Access-Control-Allow-Origin", origin)
						originAllowed = true
						break
					}
				}
				if !originAllowed && isProduction {
					// Production: reject unauthorized origin
					http.Error(w, "Origin not allowed", http.StatusForbidden)
					return
				}
			} else if !isProduction && (allowedOrigins == "*" || strings.Contains(allowedOrigins, "*")) {
				// Fallback for development: allow all if no origin header
				w.Header().Set("Access-Control-Allow-Origin", "*")
			}

			// Set CORS headers
			w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
			w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, X-Requested-With, Accept")
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			w.Header().Set("Access-Control-Expose-Headers", "X-Request-ID, X-RateLimit-Limit, X-RateLimit-Remaining")
			w.Header().Set("Access-Control-Max-Age", "3600") // Cache preflight for 1 hour

			if r.Method == "OPTIONS" {
				w.WriteHeader(http.StatusOK)
				return
			}

			next(w, r)
		}
	}

	// Wrap all handlers with panic recovery
	wrapHandler := func(handler http.HandlerFunc) http.HandlerFunc {
		return wrapWithPanicRecovery(corsHandler(handler))
	}
	
	// API endpoints (without /api/dashboard prefix - it's added in app.go)
	mux.HandleFunc("/status", wrapHandler(api.handleStatus))
	mux.HandleFunc("/miners", wrapHandler(api.handleMiners))
	mux.HandleFunc("/miner/", wrapHandler(api.handleMiner))
	mux.HandleFunc("/blocks", wrapHandler(api.handleBlocks))
	mux.HandleFunc("/block/", wrapHandler(api.handleBlock))
	mux.HandleFunc("/statistics", wrapHandler(api.handleStatistics))
	mux.HandleFunc("/locations", wrapHandler(api.handleMinerLocations))
	mux.HandleFunc("/ipfs/health", wrapHandler(api.handleIPFSHealth))
	// Prometheus metrics endpoint
	mux.HandleFunc("/metrics", wrapHandler(api.handleMetrics))
}

// handleStatus returns overall network status
func (api *DashboardAPI) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Try to get block height from Tendermint RPC
	blockHeight := int64(0)

	// Query Tendermint RPC for latest block (if available)
	// This is a fallback when SDK context is not available
	resp, err := http.Get(api.tendermintRPCAddr + "/status")
	if err == nil {
		defer resp.Body.Close()
		var statusResp struct {
			Result struct {
				SyncInfo struct {
					LatestBlockHeight string `json:"latest_block_height"`
				} `json:"sync_info"`
			} `json:"result"`
		}
		if json.NewDecoder(resp.Body).Decode(&statusResp) == nil {
			if height, parseErr := strconv.ParseInt(statusResp.Result.SyncInfo.LatestBlockHeight, 10, 64); parseErr == nil {
				blockHeight = height
			}
		}
	}

	// Get SDK context from request (if available)
	ctx, err := api.getSDKContextFromRequest(r)
	if err != nil {
		// Fallback: Use GetNetworkStatus without context, but with block height from RPC
		status := api.keeper.GetNetworkStatus()
		status.BlockHeight = blockHeight
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(status)
		return
	}

	// Use context-aware version
	status := api.keeper.GetNetworkStatusWithContext(ctx)
	// Override with RPC block height if available
	if blockHeight > 0 {
		status.BlockHeight = blockHeight
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}

// handleMiners returns list of active miners
func (api *DashboardAPI) handleMiners(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Check cache first
	api.cache.mu.RLock()
	if time.Now().Before(api.cache.MinersCache.ExpiresAt) && api.cache.MinersCache.Data != nil {
		// Cache hit - return cached data
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Cache", "HIT")
		json.NewEncoder(w).Encode(api.cache.MinersCache.Data)
		api.cache.mu.RUnlock()
		return
	}
	api.cache.mu.RUnlock()

	// Cache miss - query gRPC
	grpcClient, cleanup, err := api.getGRPCClient()
	if err != nil {
		// Fallback: Return empty list if gRPC connection fails
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"miners": []map[string]interface{}{},
			"total":  0,
			"error":  fmt.Sprintf("gRPC connection failed: %v", err),
		})
		return
	}
	defer cleanup()

	// Get pagination parameters from query string
	limitStr := r.URL.Query().Get("limit")
	offsetStr := r.URL.Query().Get("offset")

	var pagination *query.PageRequest
	if limitStr != "" || offsetStr != "" {
		limit := uint64(100) // Default limit
		offset := uint64(0)

		if limitStr != "" {
			if parsed, err := strconv.ParseUint(limitStr, 10, 64); err == nil && parsed > 0 && parsed <= 1000 {
				limit = parsed
			}
		}

		if offsetStr != "" {
			if parsed, err := strconv.ParseUint(offsetStr, 10, 64); err == nil {
				offset = parsed
			}
		}

		pagination = &query.PageRequest{
			Limit:  limit,
			Offset: offset,
		}
	}

	// Call gRPC query endpoint
	req := &types.QueryMinersRequest{
		Pagination: pagination,
	}

	resp, err := grpcClient.QueryMiners(r.Context(), req)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to query miners: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert MinerInfo to map for JSON response
	miners := make([]map[string]interface{}, len(resp.Miners))
	for i, miner := range resp.Miners {
		miners[i] = map[string]interface{}{
			"address":                miner.Address,
			"status":                 miner.Status,
			"total_submissions":      miner.TotalSubmissions,
			"successful_submissions": miner.SuccessfulSubmissions,
			"trust_score":            miner.TrustScore,
			"reputation_tier":        miner.ReputationTier,
			"slashing_events":        miner.SlashingEvents,
			"last_submission_height": miner.LastSubmissionHeight,
			"total_gradients":        miner.TotalGradients,
		}
	}

	responseData := map[string]interface{}{
		"miners": miners,
		"total":  resp.Total,
	}

	// Update cache
	api.cache.mu.Lock()
	api.cache.MinersCache.Data = responseData
	api.cache.MinersCache.ExpiresAt = time.Now().Add(api.cacheTTL)
	api.cache.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", "MISS")
	json.NewEncoder(w).Encode(responseData)
}

// validateMinerAddress validates miner address format to prevent injection attacks
func validateMinerAddress(address string) bool {
	// Cosmos address format: bech32 encoded, typically 20-45 characters
	// Allow only alphanumeric and specific characters
	if len(address) < 20 || len(address) > 45 {
		return false
	}

	// Whitelist approach: only allow valid bech32 characters
	validChars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	for _, char := range address {
		if !strings.ContainsRune(validChars, char) {
			return false
		}
	}

	return true
}

// handleMiner returns specific miner details
func (api *DashboardAPI) handleMiner(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract miner address from URL path (path is already stripped of /api/dashboard prefix)
	minerAddress := r.URL.Path[len("/miner/"):]
	if minerAddress == "" {
		http.Error(w, "Miner address required", http.StatusBadRequest)
		return
	}

	// Validate miner address format to prevent injection attacks
	if !validateMinerAddress(minerAddress) {
		http.Error(w, "Invalid miner address format", http.StatusBadRequest)
		return
	}

	// Get SDK context from request (if available)
	ctx, err := api.getSDKContextFromRequest(r)
	if err != nil {
		// Fallback: Return error
		http.Error(w, "SDK context not available - use gRPC query endpoints", http.StatusServiceUnavailable)
		return
	}

	// Get miner contribution
	contribution, err := api.keeper.MiningContributions.Get(ctx, minerAddress)
	if err != nil {
		// Miner not found
		http.Error(w, fmt.Sprintf("Miner not found: %s", minerAddress), http.StatusNotFound)
		return
	}

	// Count gradients submitted by this miner
	totalGradients := uint64(0)
	_ = api.keeper.StoredGradients.Walk(ctx, nil, func(key uint64, value types.StoredGradient) (stop bool, err error) {
		if value.Miner == minerAddress {
			totalGradients++
		}
		return false, nil
	})

	// Determine status
	status := "inactive"
	trustScore, parseErr := sdkmath.LegacyNewDecFromStr(contribution.TrustScore)
	if parseErr == nil && trustScore.GT(sdkmath.LegacyZeroDec()) {
		status = "active"
	}

	minerDetails := map[string]interface{}{
		"address":                minerAddress,
		"status":                 status,
		"total_submissions":      contribution.TotalSubmissions,
		"successful_submissions": contribution.SuccessfulSubmissions,
		"trust_score":            contribution.TrustScore,
		"reputation_tier":        contribution.ReputationTier,
		"slashing_events":        contribution.SlashingEvents,
		"last_submission_height": contribution.LastSubmissionHeight,
		"total_gradients":        totalGradients,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(minerDetails)
}

// handleBlocks returns recent blocks
// NOTE: Block information requires SDK context which is not available in HTTP handlers
// For production, use gRPC query endpoints or implement context middleware
func (api *DashboardAPI) handleBlocks(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Get limit from query params (default: 10)
	limitStr := r.URL.Query().Get("limit")
	limit := 10
	if limitStr != "" {
		// Validate limit format (only digits)
		for _, char := range limitStr {
			if char < '0' || char > '9' {
				http.Error(w, "Invalid limit format", http.StatusBadRequest)
				return
			}
		}

		if parsed, err := strconv.Atoi(limitStr); err == nil {
			// Validate limit range (prevent DoS with very large limits)
			// Max limit reduced from 1000 to 100 for blocks endpoint
			if parsed < 1 || parsed > 100 {
				http.Error(w, "Limit must be between 1 and 100", http.StatusBadRequest)
				return
			}
			limit = parsed
		} else {
			http.Error(w, "Invalid limit value", http.StatusBadRequest)
			return
		}
	}

	// Use gRPC query endpoint to get block information
	grpcClient, cleanup, err := api.getGRPCClient()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to connect to gRPC server: %v", err), http.StatusInternalServerError)
		return
	}
	defer cleanup()

	// Get offset from query params (default: 0)
	offsetStr := r.URL.Query().Get("offset")
	offset := uint64(0)
	if offsetStr != "" {
		// Validate offset format (only digits)
		for _, char := range offsetStr {
			if char < '0' || char > '9' {
				http.Error(w, "Invalid offset format", http.StatusBadRequest)
				return
			}
		}

		if parsed, err := strconv.ParseUint(offsetStr, 10, 64); err == nil {
			offset = parsed
		} else {
			http.Error(w, "Invalid offset value", http.StatusBadRequest)
			return
		}
	}

	// Create request
	req := &types.QueryBlocksRequest{
		Limit:  uint64(limit),
		Offset: offset,
	}

	// Query blocks via gRPC
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := grpcClient.QueryBlocks(ctx, req)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to query blocks: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to JSON format
	blocks := make([]map[string]interface{}, len(resp.Blocks))
	for i, block := range resp.Blocks {
		blocks[i] = map[string]interface{}{
			"height":    block.Height,
			"hash":      block.Hash,
			"timestamp": block.Timestamp,
			"tx_count":  block.TxCount,
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"blocks": blocks,
		"limit":  limit,
		"total":  resp.Total,
	})
}

// handleBlock returns specific block details
// NOTE: Block information requires SDK context which is not available in HTTP handlers
func (api *DashboardAPI) handleBlock(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract block height from URL path (path is already stripped of /api/dashboard prefix)
	heightStr := r.URL.Path[len("/block/"):]
	if heightStr == "" {
		http.Error(w, "Block height required", http.StatusBadRequest)
		return
	}

	// Validate block height format (only digits)
	for _, char := range heightStr {
		if char < '0' || char > '9' {
			http.Error(w, "Invalid block height format", http.StatusBadRequest)
			return
		}
	}

	height, err := strconv.ParseInt(heightStr, 10, 64)
	if err != nil {
		http.Error(w, "Invalid block height", http.StatusBadRequest)
		return
	}

	// Validate block height range (must be positive and reasonable)
	if height < 0 || height > 1000000000 { // Max 1 billion blocks
		http.Error(w, "Block height out of valid range", http.StatusBadRequest)
		return
	}

	// Use gRPC query endpoint to get block information
	grpcClient, cleanup, err := api.getGRPCClient()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to connect to gRPC server: %v", err), http.StatusInternalServerError)
		return
	}
	defer cleanup()

	// Create request
	req := &types.QueryBlockRequest{
		Height: height,
	}

	// Query block via gRPC
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	resp, err := grpcClient.QueryBlock(ctx, req)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to query block: %v", err), http.StatusInternalServerError)
		return
	}

	// Convert to JSON format
	block := map[string]interface{}{
		"height":    resp.Block.Height,
		"hash":      resp.Block.Hash,
		"timestamp": resp.Block.Timestamp,
		"tx_count":  resp.Block.TxCount,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(block)
}

// handleStatistics returns network statistics
func (api *DashboardAPI) handleStatistics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Check cache first
	api.cache.mu.RLock()
	if time.Now().Before(api.cache.StatisticsCache.ExpiresAt) && api.cache.StatisticsCache.Data != nil {
		// Cache hit - return cached data
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Cache", "HIT")
		json.NewEncoder(w).Encode(api.cache.StatisticsCache.Data)
		api.cache.mu.RUnlock()
		return
	}
	api.cache.mu.RUnlock()

	// Cache miss - query gRPC
	grpcClient, cleanup, err := api.getGRPCClient()
	if err != nil {
		// Fallback: Return error if gRPC connection fails
		http.Error(w, fmt.Sprintf("gRPC connection failed: %v", err), http.StatusServiceUnavailable)
		return
	}
	defer cleanup()

	// Call gRPC query endpoint
	req := &types.QueryStatisticsRequest{}

	resp, err := grpcClient.QueryStatistics(r.Context(), req)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to query statistics: %v", err), http.StatusInternalServerError)
		return
	}

	statistics := map[string]interface{}{
		"total_miners":          resp.TotalMiners,
		"active_miners":         resp.ActiveMiners,
		"total_gradients":       resp.TotalGradients,
		"total_aggregations":    resp.TotalAggregations,
		"pending_gradients":     resp.PendingGradients,
		"pending_aggregations":  resp.PendingAggregations,
		"average_gradient_size": resp.AverageGradientSize,
		"last_updated":          resp.LastUpdated,
	}

	// Update cache
	api.cache.mu.Lock()
	api.cache.StatisticsCache.Data = statistics
	api.cache.StatisticsCache.ExpiresAt = time.Now().Add(api.cacheTTL)
	api.cache.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Cache", "MISS")
	json.NewEncoder(w).Encode(statistics)
}

// MinerLocation represents a miner's approximate geographic location for visualization
type MinerLocation struct {
	Address string  `json:"address"`
	Lat     float64 `json:"lat"`
	Lng     float64 `json:"lng"`
	Size    float64 `json:"size"`
}

// deriveLatLngFromAddress deterministically derives a pseudo-random latitude and longitude
// from a miner address. This is used for visualization only and does NOT reveal real locations.
func deriveLatLngFromAddress(address string) (float64, float64) {
	// Hash the address
	sum := sha256.Sum256([]byte(address))

	// Use first 4 bytes to derive lat/lng in a deterministic way
	latRaw := binary.BigEndian.Uint16(sum[0:2])
	lngRaw := binary.BigEndian.Uint16(sum[2:4])

	// Normalize to [0,1]
	latNorm := float64(latRaw) / 65535.0
	lngNorm := float64(lngRaw) / 65535.0

	// Map to latitude [-60, 60] to avoid poles (better visualization)
	lat := -60.0 + latNorm*120.0

	// Map to longitude [-180, 180]
	lng := -180.0 + lngNorm*360.0

	return lat, lng
}

// handleMinerLocations returns approximate miner locations for the Network Explorer
// NOTE: This does NOT expose real IP/location data. Locations are derived deterministically
// from miner addresses for visualization purposes only.
// Uses gRPC query endpoints to get real miner data instead of SDK context.
func (api *DashboardAPI) handleMinerLocations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Use gRPC query to get miners (real data, no SDK context needed)
	grpcClient, cleanup, err := api.getGRPCClient()
	if err != nil {
		// Fallback: Return empty list if gRPC connection fails
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"locations": []MinerLocation{},
			"total":     0,
			"error":     fmt.Sprintf("gRPC connection failed: %v", err),
		})
		return
	}
	defer cleanup()

	// Query all miners via gRPC (no pagination limit for locations)
	req := &types.QueryMinersRequest{
		Pagination: &query.PageRequest{
			Limit: 10000, // Large limit to get all miners for visualization
		},
	}

	resp, err := grpcClient.QueryMiners(r.Context(), req)
	if err != nil {
		// Fallback: Return empty list if query fails
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"locations": []MinerLocation{},
			"total":     0,
			"error":     fmt.Sprintf("Failed to query miners: %v", err),
		})
		return
	}

	// Convert MinerInfo to MinerLocation with derived coordinates
	locations := make([]MinerLocation, 0, len(resp.Miners))
	for _, miner := range resp.Miners {
		// Only include active miners
		if miner.Status != "active" {
			continue
		}

		lat, lng := deriveLatLngFromAddress(miner.Address)

		// Size based on trust score or total submissions (visualization only)
		size := 0.3
		if miner.TotalSubmissions > 0 {
			// Normalize based on submissions (0.1 to 1.0)
			// Simple log-based scaling
			submissionsFloat := float64(miner.TotalSubmissions)
			size = 0.1 + (0.9 * (1.0 / (1.0 + submissionsFloat/100.0)))
			if size > 1.0 {
				size = 1.0
			}
			if size < 0.1 {
				size = 0.1
			}
		}

		location := MinerLocation{
			Address: miner.Address,
			Lat:     lat,
			Lng:     lng,
			Size:    size,
		}

		locations = append(locations, location)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"locations": locations,
		"total":     len(locations),
	})
}

// handleIPFSHealth returns IPFS connection health status
func (api *DashboardAPI) handleIPFSHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Check if IPFS manager is available
	health := map[string]interface{}{
		"status":    "unknown",
		"connected": false,
		"message":   "",
	}

	if api.keeper.ipfsManager == nil {
		health["status"] = "not_configured"
		health["message"] = "IPFS manager not configured"
	} else {
		// Try to check IPFS connection
		// Note: This is a simple check - in production, you might want to ping IPFS
		// For now, we just check if the manager exists
		health["status"] = "configured"
		health["connected"] = true
		health["message"] = "IPFS manager is configured"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// handleMetrics returns Prometheus metrics
func (api *DashboardAPI) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Return Prometheus metrics
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	MetricsHandler().ServeHTTP(w, r)
}
