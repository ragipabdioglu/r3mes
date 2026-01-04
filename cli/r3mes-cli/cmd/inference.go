package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

// InferenceRequest represents an inference API request
type InferenceRequest struct {
	Prompt       string   `json:"prompt"`
	WalletAddr   string   `json:"wallet_address,omitempty"`
	MaxTokens    int      `json:"max_tokens"`
	Temperature  float64  `json:"temperature"`
	TopP         float64  `json:"top_p"`
	TopK         int      `json:"top_k"`
	SkipRAG      bool     `json:"skip_rag"`
	ForceExperts []string `json:"force_experts,omitempty"`
	Stream       bool     `json:"stream"`
}

// InferenceResponse represents an inference API response
type InferenceResponse struct {
	RequestID       string        `json:"request_id"`
	Text            string        `json:"text"`
	TokensGenerated int           `json:"tokens_generated"`
	LatencyMs       float64       `json:"latency_ms"`
	ExpertsUsed     []ExpertUsage `json:"experts_used"`
	RAGContextUsed  bool          `json:"rag_context_used"`
	ModelVersion    string        `json:"model_version"`
	CreditsUsed     float64       `json:"credits_used"`
}

// ExpertUsage represents DoRA expert usage info
type ExpertUsage struct {
	ID     string  `json:"id"`
	Weight float64 `json:"weight"`
}

// InferenceHealth represents inference health status
type InferenceHealth struct {
	Status              string  `json:"status"`
	InferenceMode       string  `json:"inference_mode"`
	IsReady             bool    `json:"is_ready"`
	IsHealthy           bool    `json:"is_healthy"`
	PipelineInitialized bool    `json:"pipeline_initialized"`
	ModelLoaded         bool    `json:"model_loaded"`
	TotalRequests       int     `json:"total_requests"`
	SuccessfulRequests  int     `json:"successful_requests"`
	FailedRequests      int     `json:"failed_requests"`
	AvgLatencyMs        float64 `json:"avg_latency_ms"`
	ErrorMessage        string  `json:"error_message,omitempty"`
}

// InferenceMetrics represents inference metrics
type InferenceMetrics struct {
	RequestsTotal   int     `json:"serving_engine_requests_total"`
	RequestsSuccess int     `json:"serving_engine_requests_success"`
	RequestsFailed  int     `json:"serving_engine_requests_failed"`
	LatencyAvgMs    float64 `json:"serving_engine_latency_avg_ms"`
	Ready           int     `json:"serving_engine_ready"`
	Healthy         int     `json:"serving_engine_healthy"`
	CacheHits       int     `json:"cache_hits"`
	CacheMisses     int     `json:"cache_misses"`
	VRAMUsedMB      float64 `json:"cache_vram_used_mb"`
	RAMUsedMB       float64 `json:"cache_ram_used_mb"`
}

func newInferenceCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "inference",
		Short: "AI inference operations",
		Long: `Run AI inference queries using the BitNet + DoRA + RAG pipeline.

Examples:
  r3mes inference query "What is R3MES?"
  r3mes inference query "Explain blockchain" --stream
  r3mes inference health
  r3mes inference metrics`,
	}

	cmd.AddCommand(newInferenceQueryCmd())
	cmd.AddCommand(newInferenceStreamCmd())
	cmd.AddCommand(newInferenceHealthCmd())
	cmd.AddCommand(newInferenceMetricsCmd())
	cmd.AddCommand(newInferenceWarmupCmd())

	return cmd
}

func newInferenceQueryCmd() *cobra.Command {
	var (
		maxTokens   int
		temperature float64
		topP        float64
		topK        int
		skipRAG     bool
		experts     []string
		stream      bool
		wallet      string
	)

	cmd := &cobra.Command{
		Use:   "query [prompt]",
		Short: "Run inference query",
		Long: `Run an AI inference query through the BitNet + DoRA + RAG pipeline.

Examples:
  r3mes inference query "What is R3MES?"
  r3mes inference query "Explain DoRA" --max-tokens 1024
  r3mes inference query "Code example" --skip-rag --temperature 0.5`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			prompt := strings.Join(args, " ")

			if stream {
				return runInferenceStream(ctx, prompt, maxTokens, temperature, topP, topK, skipRAG, experts, wallet)
			}
			return runInferenceQuery(ctx, prompt, maxTokens, temperature, topP, topK, skipRAG, experts, wallet)
		},
	}

	cmd.Flags().IntVar(&maxTokens, "max-tokens", 512, "Maximum tokens to generate (1-4096)")
	cmd.Flags().Float64Var(&temperature, "temperature", 0.7, "Sampling temperature (0.0-2.0)")
	cmd.Flags().Float64Var(&topP, "top-p", 0.9, "Top-p sampling (0.0-1.0)")
	cmd.Flags().IntVar(&topK, "top-k", 50, "Top-k sampling (0-100)")
	cmd.Flags().BoolVar(&skipRAG, "skip-rag", false, "Skip RAG context retrieval")
	cmd.Flags().StringSliceVar(&experts, "experts", nil, "Force specific DoRA experts (comma-separated)")
	cmd.Flags().BoolVarP(&stream, "stream", "s", false, "Enable streaming output")
	cmd.Flags().StringVarP(&wallet, "wallet", "w", "", "Wallet address for credit deduction")

	return cmd
}

func newInferenceStreamCmd() *cobra.Command {
	var (
		maxTokens   int
		temperature float64
		skipRAG     bool
		wallet      string
	)

	cmd := &cobra.Command{
		Use:   "stream [prompt]",
		Short: "Run streaming inference query",
		Long: `Run a streaming AI inference query. Tokens are printed as they are generated.

Examples:
  r3mes inference stream "Write a poem about AI"
  r3mes inference stream "Explain quantum computing" --max-tokens 2048`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			prompt := strings.Join(args, " ")
			return runInferenceStream(ctx, prompt, maxTokens, temperature, 0.9, 50, skipRAG, nil, wallet)
		},
	}

	cmd.Flags().IntVar(&maxTokens, "max-tokens", 512, "Maximum tokens to generate")
	cmd.Flags().Float64Var(&temperature, "temperature", 0.7, "Sampling temperature")
	cmd.Flags().BoolVar(&skipRAG, "skip-rag", false, "Skip RAG context retrieval")
	cmd.Flags().StringVarP(&wallet, "wallet", "w", "", "Wallet address")

	return cmd
}

func newInferenceHealthCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "health",
		Short: "Check inference engine health",
		Long: `Check the health status of the inference engine.

Shows:
  - Engine status (ready, healthy, mode)
  - Pipeline initialization state
  - Model loading state
  - Request statistics`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return getInferenceHealth(ctx)
		},
	}
}

func newInferenceMetricsCmd() *cobra.Command {
	var watch bool

	cmd := &cobra.Command{
		Use:   "metrics",
		Short: "Get inference metrics",
		Long: `Get Prometheus-compatible inference metrics.

Shows:
  - Request counts (total, success, failed)
  - Latency statistics
  - Cache utilization (VRAM, RAM)
  - Cache hit/miss rates`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			if watch {
				return watchInferenceMetrics(ctx)
			}
			return getInferenceMetrics(ctx)
		},
	}

	cmd.Flags().BoolVarP(&watch, "watch", "w", false, "Watch metrics continuously")

	return cmd
}

func newInferenceWarmupCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "warmup",
		Short: "Warmup inference pipeline",
		Long:  `Pre-load models and caches for faster first inference.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return warmupPipeline(ctx)
		},
	}
}

// Implementation functions

func getAPIBaseURL() string {
	// Check environment variable first
	if url := os.Getenv("R3MES_API_URL"); url != "" {
		return url
	}
	// Check config
	config := GetConfig()
	if config != nil && config.RPCEndpoint != "" {
		return config.RPCEndpoint
	}
	// Default
	return "http://localhost:8000"
}

func runInferenceQuery(ctx context.Context, prompt string, maxTokens int, temperature, topP float64, topK int, skipRAG bool, experts []string, wallet string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	logger.Info("Running inference query",
		zap.String("prompt", truncateString(prompt, 50)),
		zap.Int("max_tokens", maxTokens),
	)

	req := InferenceRequest{
		Prompt:       prompt,
		WalletAddr:   wallet,
		MaxTokens:    maxTokens,
		Temperature:  temperature,
		TopP:         topP,
		TopK:         topK,
		SkipRAG:      skipRAG,
		ForceExperts: experts,
		Stream:       false,
	}

	reqBody, _ := json.Marshal(req)
	apiURL := fmt.Sprintf("%s/api/inference/generate", getAPIBaseURL())

	httpReq, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("inference request failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		return fmt.Errorf("inference failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result InferenceResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	if config.JSONOutput {
		data, _ := json.MarshalIndent(result, "", "  ")
		fmt.Println(string(data))
		return nil
	}

	// Print response
	fmt.Println()
	fmt.Println(result.Text)
	fmt.Println()
	fmt.Printf("─────────────────────────────────────────\n")
	fmt.Printf("Tokens: %d | Latency: %.0fms | RAG: %v\n",
		result.TokensGenerated, result.LatencyMs, result.RAGContextUsed)
	if len(result.ExpertsUsed) > 0 {
		expertNames := make([]string, len(result.ExpertsUsed))
		for i, e := range result.ExpertsUsed {
			expertNames[i] = e.ID
		}
		fmt.Printf("Experts: %s\n", strings.Join(expertNames, ", "))
	}

	return nil
}

func runInferenceStream(ctx context.Context, prompt string, maxTokens int, temperature, topP float64, topK int, skipRAG bool, experts []string, wallet string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	logger.Info("Running streaming inference",
		zap.String("prompt", truncateString(prompt, 50)),
	)

	req := InferenceRequest{
		Prompt:       prompt,
		WalletAddr:   wallet,
		MaxTokens:    maxTokens,
		Temperature:  temperature,
		TopP:         topP,
		TopK:         topK,
		SkipRAG:      skipRAG,
		ForceExperts: experts,
		Stream:       true,
	}

	reqBody, _ := json.Marshal(req)
	apiURL := fmt.Sprintf("%s/api/inference/generate/stream", getAPIBaseURL())

	httpReq, err := http.NewRequestWithContext(ctx, "POST", apiURL, bytes.NewBuffer(reqBody))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	client := &http.Client{Timeout: 0} // No timeout for streaming
	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("streaming request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("streaming failed (status %d): %s", resp.StatusCode, string(body))
	}

	fmt.Println()

	// Read SSE stream
	reader := resp.Body
	buf := make([]byte, 1024)
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\n\n[Interrupted]")
			return nil
		default:
		}

		n, err := reader.Read(buf)
		if err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("stream read error: %w", err)
		}

		chunk := string(buf[:n])
		lines := strings.Split(chunk, "\n")

		for _, line := range lines {
			if strings.HasPrefix(line, "data: ") {
				data := strings.TrimPrefix(line, "data: ")
				data = strings.TrimSpace(data)

				if data == "[DONE]" {
					fmt.Println("\n")
					return nil
				}
				if strings.HasPrefix(data, "[ERROR]") {
					return fmt.Errorf("stream error: %s", data)
				}

				fmt.Print(data)
			}
		}
	}

	fmt.Println("\n")
	return nil
}

func getInferenceHealth(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	apiURL := fmt.Sprintf("%s/api/inference/health", getAPIBaseURL())

	httpReq, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return err
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var health InferenceHealth
	if err := json.Unmarshal(body, &health); err != nil {
		return fmt.Errorf("failed to parse health: %w", err)
	}

	if config.JSONOutput {
		data, _ := json.MarshalIndent(health, "", "  ")
		fmt.Println(string(data))
		return nil
	}

	// Print health status
	fmt.Println("\nInference Engine Health:")
	fmt.Printf("  Status:     %s\n", getStatusEmoji(health.Status))
	fmt.Printf("  Mode:       %s\n", health.InferenceMode)
	fmt.Printf("  Ready:      %s\n", getBoolEmoji(health.IsReady))
	fmt.Printf("  Healthy:    %s\n", getBoolEmoji(health.IsHealthy))
	fmt.Printf("  Pipeline:   %s\n", getBoolEmoji(health.PipelineInitialized))
	fmt.Printf("  Model:      %s\n", getBoolEmoji(health.ModelLoaded))
	fmt.Println()
	fmt.Printf("  Requests:   %d total, %d success, %d failed\n",
		health.TotalRequests, health.SuccessfulRequests, health.FailedRequests)
	fmt.Printf("  Avg Latency: %.2f ms\n", health.AvgLatencyMs)

	if health.ErrorMessage != "" {
		fmt.Printf("\n  ⚠️ Error: %s\n", health.ErrorMessage)
	}

	return nil
}

func getInferenceMetrics(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	apiURL := fmt.Sprintf("%s/api/inference/metrics", getAPIBaseURL())

	httpReq, err := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
	if err != nil {
		return err
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("metrics fetch failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var metrics InferenceMetrics
	if err := json.Unmarshal(body, &metrics); err != nil {
		return fmt.Errorf("failed to parse metrics: %w", err)
	}

	if config.JSONOutput {
		data, _ := json.MarshalIndent(metrics, "", "  ")
		fmt.Println(string(data))
		return nil
	}

	printInferenceMetrics(&metrics)
	return nil
}

func watchInferenceMetrics(ctx context.Context) error {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	fmt.Println("Watching inference metrics (Ctrl+C to stop)...")
	fmt.Println()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			apiURL := fmt.Sprintf("%s/api/inference/metrics", getAPIBaseURL())
			httpReq, _ := http.NewRequestWithContext(ctx, "GET", apiURL, nil)
			client := &http.Client{Timeout: 5 * time.Second}

			resp, err := client.Do(httpReq)
			if err != nil {
				fmt.Printf("\r⚠️ Error: %v", err)
				continue
			}

			body, _ := io.ReadAll(resp.Body)
			resp.Body.Close()

			var metrics InferenceMetrics
			if err := json.Unmarshal(body, &metrics); err != nil {
				continue
			}

			// Clear and print
			fmt.Printf("\r\033[K")
			fmt.Printf("Requests: %d (✓%d ✗%d) | Latency: %.0fms | Cache: %d/%d | VRAM: %.0fMB",
				metrics.RequestsTotal, metrics.RequestsSuccess, metrics.RequestsFailed,
				metrics.LatencyAvgMs, metrics.CacheHits, metrics.CacheHits+metrics.CacheMisses,
				metrics.VRAMUsedMB)
		}
	}
}

func warmupPipeline(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	fmt.Println("Warming up inference pipeline...")

	apiURL := fmt.Sprintf("%s/api/inference/pipeline/warmup", getAPIBaseURL())
	httpReq, err := http.NewRequestWithContext(ctx, "POST", apiURL, nil)
	if err != nil {
		return err
	}

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("warmup failed: %w", err)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var result map[string]interface{}
	json.Unmarshal(body, &result)

	if resp.StatusCode == 200 {
		fmt.Println("✅ Pipeline warmed up successfully")
	} else {
		fmt.Printf("⚠️ Warmup status: %v\n", result["status"])
		if reason, ok := result["reason"]; ok {
			fmt.Printf("   Reason: %v\n", reason)
		}
	}

	return nil
}

// Helper functions

func printInferenceMetrics(m *InferenceMetrics) {
	fmt.Println("\nInference Metrics:")
	fmt.Printf("  Requests:\n")
	fmt.Printf("    Total:     %d\n", m.RequestsTotal)
	fmt.Printf("    Success:   %d\n", m.RequestsSuccess)
	fmt.Printf("    Failed:    %d\n", m.RequestsFailed)
	fmt.Printf("  Performance:\n")
	fmt.Printf("    Avg Latency: %.2f ms\n", m.LatencyAvgMs)
	fmt.Printf("    Ready:       %s\n", getBoolEmoji(m.Ready == 1))
	fmt.Printf("    Healthy:     %s\n", getBoolEmoji(m.Healthy == 1))
	fmt.Printf("  Cache:\n")
	fmt.Printf("    Hits:      %d\n", m.CacheHits)
	fmt.Printf("    Misses:    %d\n", m.CacheMisses)
	fmt.Printf("    VRAM Used: %.2f MB\n", m.VRAMUsedMB)
	fmt.Printf("    RAM Used:  %.2f MB\n", m.RAMUsedMB)
}

func getStatusEmoji(status string) string {
	switch status {
	case "ready", "healthy", "mock":
		return "✅ " + status
	case "remote":
		return "🔗 " + status
	case "disabled", "unavailable":
		return "❌ " + status
	default:
		return "⚠️ " + status
	}
}

func getBoolEmoji(b bool) string {
	if b {
		return "✅ Yes"
	}
	return "❌ No"
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
