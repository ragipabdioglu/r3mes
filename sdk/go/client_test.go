package r3mes

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestDefaultConfig(t *testing.T) {
	config := DefaultConfig()

	if config.RPCEndpoint == "" {
		t.Error("RPCEndpoint should not be empty")
	}
	if config.RESTEndpoint == "" {
		t.Error("RESTEndpoint should not be empty")
	}
	if config.BackendEndpoint == "" {
		t.Error("BackendEndpoint should not be empty")
	}
	if config.Timeout == 0 {
		t.Error("Timeout should not be zero")
	}
}

func TestNewClient(t *testing.T) {
	tests := []struct {
		name   string
		config Config
	}{
		{
			name:   "empty config uses defaults",
			config: Config{},
		},
		{
			name: "custom config",
			config: Config{
				RPCEndpoint:     "http://localhost:26657",
				RESTEndpoint:    "http://localhost:1317",
				BackendEndpoint: "http://localhost:8000",
				Timeout:         10 * time.Second,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client, err := NewClient(tt.config)
			if err != nil {
				t.Fatalf("NewClient() error = %v", err)
			}
			if client == nil {
				t.Fatal("NewClient() returned nil")
			}
			if client.httpClient == nil {
				t.Error("httpClient should not be nil")
			}
		})
	}
}

func TestClientClose(t *testing.T) {
	client, _ := NewClient(Config{})
	err := client.Close()
	if err != nil {
		t.Errorf("Close() error = %v", err)
	}
}

func TestGetNetworkStats(t *testing.T) {
	expected := NetworkStats{
		ActiveMiners: 100,
		TotalUsers:   1000,
		TotalCredits: 50000.5,
		BlockHeight:  12345,
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/network/stats" {
			t.Errorf("Expected path /network/stats, got %s", r.URL.Path)
		}
		if r.Method != http.MethodGet {
			t.Errorf("Expected GET method, got %s", r.Method)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(expected)
	}))
	defer server.Close()

	client, _ := NewClient(Config{BackendEndpoint: server.URL})
	stats, err := client.GetNetworkStats(context.Background())

	if err != nil {
		t.Fatalf("GetNetworkStats() error = %v", err)
	}
	if stats.ActiveMiners != expected.ActiveMiners {
		t.Errorf("ActiveMiners = %d, want %d", stats.ActiveMiners, expected.ActiveMiners)
	}
	if stats.TotalUsers != expected.TotalUsers {
		t.Errorf("TotalUsers = %d, want %d", stats.TotalUsers, expected.TotalUsers)
	}
}

func TestGetNetworkStatsError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("Internal Server Error"))
	}))
	defer server.Close()

	client, _ := NewClient(Config{BackendEndpoint: server.URL})
	_, err := client.GetNetworkStats(context.Background())

	if err == nil {
		t.Error("Expected error for 500 response")
	}
}

func TestGetUserInfo(t *testing.T) {
	expected := UserInfo{
		WalletAddress: "remes1abc123",
		Credits:       100.5,
		IsMiner:       true,
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/user/info/remes1abc123" {
			t.Errorf("Expected path /user/info/remes1abc123, got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(expected)
	}))
	defer server.Close()

	client, _ := NewClient(Config{BackendEndpoint: server.URL})
	info, err := client.GetUserInfo(context.Background(), "remes1abc123")

	if err != nil {
		t.Fatalf("GetUserInfo() error = %v", err)
	}
	if info.WalletAddress != expected.WalletAddress {
		t.Errorf("WalletAddress = %s, want %s", info.WalletAddress, expected.WalletAddress)
	}
	if info.Credits != expected.Credits {
		t.Errorf("Credits = %f, want %f", info.Credits, expected.Credits)
	}
}

func TestGetUserInfoNotFound(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	client, _ := NewClient(Config{BackendEndpoint: server.URL})
	_, err := client.GetUserInfo(context.Background(), "remes1notfound")

	if err != ErrUserNotFound {
		t.Errorf("Expected ErrUserNotFound, got %v", err)
	}
}

func TestGetMinerStats(t *testing.T) {
	expected := MinerStats{
		WalletAddress:    "remes1miner",
		Hashrate:         1500.5,
		TotalEarnings:    250.75,
		BlocksFound:      10,
		UptimePercentage: 99.5,
		IsActive:         true,
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(expected)
	}))
	defer server.Close()

	client, _ := NewClient(Config{BackendEndpoint: server.URL})
	stats, err := client.GetMinerStats(context.Background(), "remes1miner")

	if err != nil {
		t.Fatalf("GetMinerStats() error = %v", err)
	}
	if stats.Hashrate != expected.Hashrate {
		t.Errorf("Hashrate = %f, want %f", stats.Hashrate, expected.Hashrate)
	}
	if stats.IsActive != expected.IsActive {
		t.Errorf("IsActive = %v, want %v", stats.IsActive, expected.IsActive)
	}
}

func TestGetMinerStatsNotFound(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer server.Close()

	client, _ := NewClient(Config{BackendEndpoint: server.URL})
	stats, err := client.GetMinerStats(context.Background(), "remes1notfound")

	if err != nil {
		t.Fatalf("GetMinerStats() error = %v", err)
	}
	if stats.IsActive {
		t.Error("Expected IsActive to be false for not found miner")
	}
}

func TestGetBalance(t *testing.T) {
	expected := []Balance{
		{Denom: "uremes", Amount: "1000000"},
		{Denom: "uatom", Amount: "500000"},
	}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"balances": expected,
		})
	}))
	defer server.Close()

	client, _ := NewClient(Config{RESTEndpoint: server.URL})
	balances, err := client.GetBalance(context.Background(), "remes1abc")

	if err != nil {
		t.Fatalf("GetBalance() error = %v", err)
	}
	if len(balances) != 2 {
		t.Errorf("Expected 2 balances, got %d", len(balances))
	}
	if balances[0].Denom != "uremes" {
		t.Errorf("Expected denom uremes, got %s", balances[0].Denom)
	}
}

func TestContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client, _ := NewClient(Config{BackendEndpoint: server.URL, Timeout: 100 * time.Millisecond})

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	_, err := client.GetNetworkStats(ctx)
	if err == nil {
		t.Error("Expected error due to context cancellation")
	}
}
