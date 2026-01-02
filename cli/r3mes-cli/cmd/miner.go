package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"time"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

// MinerStats represents miner statistics
type MinerStats struct {
	Hashrate    float64 `json:"hashrate"`
	Loss        float64 `json:"loss"`
	LossTrend   string  `json:"loss_trend"`
	GPUTemp     float64 `json:"gpu_temp"`
	VRAMUsage   float64 `json:"vram_usage_mb"`
	VRAMTotal   float64 `json:"vram_total_mb"`
	Uptime      int64   `json:"uptime_seconds"`
	TasksTotal  int     `json:"tasks_total"`
	TasksActive int     `json:"tasks_active"`
}

func newMinerCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "miner",
		Short: "Miner operations",
		Long:  "Start, stop, and monitor the R3MES miner.",
	}

	cmd.AddCommand(newMinerStartCmd())
	cmd.AddCommand(newMinerStopCmd())
	cmd.AddCommand(newMinerStatusCmd())
	cmd.AddCommand(newMinerStatsCmd())

	return cmd
}

func newMinerStartCmd() *cobra.Command {
	var (
		detach    bool
		gpuDevice int
	)

	cmd := &cobra.Command{
		Use:   "start",
		Short: "Start the miner",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return startMinerCobra(ctx, detach, gpuDevice)
		},
	}

	cmd.Flags().BoolVarP(&detach, "detach", "d", true, "Run miner in background")
	cmd.Flags().IntVar(&gpuDevice, "gpu", 0, "GPU device index to use")

	return cmd
}

func newMinerStopCmd() *cobra.Command {
	var force bool

	cmd := &cobra.Command{
		Use:   "stop",
		Short: "Stop the miner",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return stopMinerCobra(ctx, force)
		},
	}

	cmd.Flags().BoolVarP(&force, "force", "f", false, "Force stop without graceful shutdown")

	return cmd
}

func newMinerStatusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Check miner status",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return getMinerStatusCobra(ctx)
		},
	}
}

func newMinerStatsCmd() *cobra.Command {
	var watch bool

	cmd := &cobra.Command{
		Use:   "stats",
		Short: "Get miner statistics",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			if watch {
				return watchMinerStats(ctx)
			}
			return getMinerStatsCobra(ctx)
		},
	}

	cmd.Flags().BoolVarP(&watch, "watch", "w", false, "Watch stats continuously")

	return cmd
}

func startMinerCobra(ctx context.Context, detach bool, gpuDevice int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	logger.Info("Starting miner", zap.Int("gpu", gpuDevice), zap.Bool("detach", detach))

	// Check if already running
	if isRunning, _ := checkMinerRunning(ctx, config); isRunning {
		return fmt.Errorf("miner is already running")
	}

	// Build command with environment
	cmdArgs := []string{"-m", "r3mes.cli.commands", "start"}
	if gpuDevice > 0 {
		cmdArgs = append(cmdArgs, "--gpu", fmt.Sprintf("%d", gpuDevice))
	}

	minerCmd := exec.CommandContext(ctx, "python3", cmdArgs...)
	minerCmd.Env = append(os.Environ(),
		fmt.Sprintf("R3MES_RPC_ENDPOINT=%s", config.RPCEndpoint),
		fmt.Sprintf("R3MES_CHAIN_ID=%s", config.ChainID),
	)

	if detach {
		minerCmd.Stdout = nil
		minerCmd.Stderr = nil

		if err := minerCmd.Start(); err != nil {
			logger.Error("Failed to start miner", zap.Error(err))
			return fmt.Errorf("failed to start miner: %w\nMake sure r3mes miner-engine is installed", err)
		}

		logger.Info("Miner started", zap.Int("pid", minerCmd.Process.Pid))
		fmt.Printf("✅ Miner started (PID: %d)\n", minerCmd.Process.Pid)
		fmt.Printf("   Use 'r3mes miner status' to check status\n")
		fmt.Printf("   Use 'r3mes miner stats' to view statistics\n")
	} else {
		minerCmd.Stdout = os.Stdout
		minerCmd.Stderr = os.Stderr

		fmt.Println("Starting miner in foreground (Ctrl+C to stop)...")
		if err := minerCmd.Run(); err != nil {
			return fmt.Errorf("miner exited with error: %w", err)
		}
	}

	return nil
}

func stopMinerCobra(ctx context.Context, force bool) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	logger.Info("Stopping miner", zap.Bool("force", force))

	// Try graceful shutdown first
	if !force {
		minerURL := fmt.Sprintf("http://localhost:%s/shutdown", config.MinerPort)
		req, _ := http.NewRequestWithContext(ctx, "POST", minerURL, nil)
		client := &http.Client{Timeout: 5 * time.Second}

		if resp, err := client.Do(req); err == nil {
			resp.Body.Close()
			fmt.Println("✅ Miner shutdown signal sent")

			// Wait for graceful shutdown
			for i := 0; i < 10; i++ {
				time.Sleep(500 * time.Millisecond)
				if running, _ := checkMinerRunning(ctx, config); !running {
					fmt.Println("✅ Miner stopped gracefully")
					return nil
				}
			}
		}
	}

	// Force kill
	stopCmd := exec.CommandContext(ctx, "python3", "-m", "r3mes.cli.commands", "stop")
	if err := stopCmd.Run(); err != nil {
		// Fallback to pkill
		exec.Command("pkill", "-f", "r3mes-miner").Run()
		exec.Command("pkill", "-f", "r3mes.cli.commands").Run()
	}

	logger.Info("Miner stopped")
	fmt.Println("✅ Miner stopped")
	return nil
}

func getMinerStatusCobra(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	logger.Debug("Checking miner status")

	running, health := checkMinerRunning(ctx, config)

	if config.JSONOutput {
		output := map[string]any{
			"running": running,
			"health":  health,
		}
		data, _ := json.MarshalIndent(output, "", "  ")
		fmt.Println(string(data))
		return nil
	}

	if !running {
		fmt.Println("Status: ❌ Not running")
		return nil
	}

	switch health {
	case "healthy":
		fmt.Println("Status: ✅ Running")
	case "unhealthy":
		fmt.Println("Status: ⚠️ Unhealthy")
	default:
		fmt.Println("Status: 🔄 Unknown")
	}

	return nil
}

func getMinerStatsCobra(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	stats, err := fetchMinerStats(ctx, config)
	if err != nil {
		return err
	}

	if config.JSONOutput {
		data, _ := json.MarshalIndent(stats, "", "  ")
		fmt.Println(string(data))
		return nil
	}

	printMinerStats(stats)
	return nil
}

func watchMinerStats(ctx context.Context) error {
	config := GetConfig()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	fmt.Println("Watching miner stats (Ctrl+C to stop)...")
	fmt.Println()

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			stats, err := fetchMinerStats(ctx, config)
			if err != nil {
				fmt.Printf("\r⚠️ Error: %v", err)
				continue
			}

			// Clear line and print stats
			fmt.Printf("\r\033[K")
			fmt.Printf("Hashrate: %.2f g/h | Loss: %.4f (%s) | GPU: %.1f°C | VRAM: %.0f/%.0f MB",
				stats.Hashrate, stats.Loss, stats.LossTrend, stats.GPUTemp, stats.VRAMUsage, stats.VRAMTotal)
		}
	}
}

func checkMinerRunning(ctx context.Context, config *Config) (bool, string) {
	minerURL := fmt.Sprintf("http://localhost:%s/health", config.MinerPort)

	req, err := http.NewRequestWithContext(ctx, "GET", minerURL, nil)
	if err != nil {
		return false, ""
	}

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return false, ""
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		return true, "healthy"
	}
	return true, "unhealthy"
}

func fetchMinerStats(ctx context.Context, config *Config) (*MinerStats, error) {
	minerURL := fmt.Sprintf("http://localhost:%s/stats", config.MinerPort)

	req, err := http.NewRequestWithContext(ctx, "GET", minerURL, nil)
	if err != nil {
		return nil, err
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("miner not running or stats endpoint unavailable")
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var stats MinerStats
	if err := json.Unmarshal(body, &stats); err != nil {
		return nil, fmt.Errorf("failed to parse stats: %w", err)
	}

	return &stats, nil
}

func printMinerStats(stats *MinerStats) {
	fmt.Println("\nMiner Statistics:")
	fmt.Printf("  Hashrate:    %.2f gradients/hour\n", stats.Hashrate)
	fmt.Printf("  Loss:        %.4f\n", stats.Loss)
	fmt.Printf("  Loss Trend:  %s\n", stats.LossTrend)
	fmt.Printf("  GPU Temp:    %.1f°C\n", stats.GPUTemp)
	fmt.Printf("  VRAM Usage:  %.0f MB / %.0f MB\n", stats.VRAMUsage, stats.VRAMTotal)
	fmt.Printf("  Uptime:      %d seconds\n", stats.Uptime)
	if stats.TasksTotal > 0 {
		fmt.Printf("  Tasks:       %d active / %d total\n", stats.TasksActive, stats.TasksTotal)
	}
}
