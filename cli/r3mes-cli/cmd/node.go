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

// NodeStatus represents node status information
type NodeStatus struct {
	Running     bool   `json:"running"`
	BlockHeight string `json:"block_height"`
	Syncing     bool   `json:"syncing"`
	NodeID      string `json:"node_id,omitempty"`
	Network     string `json:"network,omitempty"`
}

func newNodeCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "node",
		Short: "Node operations",
		Long:  "Start, stop, and monitor the R3MES blockchain node.",
	}

	cmd.AddCommand(newNodeStartCmd())
	cmd.AddCommand(newNodeStopCmd())
	cmd.AddCommand(newNodeStatusCmd())
	cmd.AddCommand(newNodeSyncCmd())

	return cmd
}

func newNodeStartCmd() *cobra.Command {
	var (
		detach   bool
		dataDir  string
		logLevel string
	)

	cmd := &cobra.Command{
		Use:   "start",
		Short: "Start the blockchain node",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return startNodeCobra(ctx, detach, dataDir, logLevel)
		},
	}

	cmd.Flags().BoolVarP(&detach, "detach", "d", true, "Run node in background")
	cmd.Flags().StringVar(&dataDir, "data-dir", "", "Data directory (default: ~/.remes)")
	cmd.Flags().StringVar(&logLevel, "log-level", "info", "Log level (debug, info, warn, error)")

	return cmd
}

func newNodeStopCmd() *cobra.Command {
	var force bool

	cmd := &cobra.Command{
		Use:   "stop",
		Short: "Stop the blockchain node",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return stopNodeCobra(ctx, force)
		},
	}

	cmd.Flags().BoolVarP(&force, "force", "f", false, "Force stop without graceful shutdown")

	return cmd
}

func newNodeStatusCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "status",
		Short: "Check node status",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return getNodeStatusCobra(ctx)
		},
	}
}

func newNodeSyncCmd() *cobra.Command {
	var watch bool

	cmd := &cobra.Command{
		Use:   "sync",
		Short: "Check sync status",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			if watch {
				return watchSyncStatus(ctx)
			}
			return getNodeStatusCobra(ctx)
		},
	}

	cmd.Flags().BoolVarP(&watch, "watch", "w", false, "Watch sync progress continuously")

	return cmd
}

func startNodeCobra(ctx context.Context, detach bool, dataDir, logLevel string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	logger.Info("Starting node",
		zap.Bool("detach", detach),
		zap.String("data_dir", dataDir),
		zap.String("log_level", logLevel))

	// Check if already running
	if status, _ := fetchNodeStatus(ctx, config); status != nil && status.Running {
		return fmt.Errorf("node is already running at block height %s", status.BlockHeight)
	}

	// Build command
	cmdArgs := []string{"start"}
	if dataDir != "" {
		cmdArgs = append(cmdArgs, "--home", dataDir)
	}
	cmdArgs = append(cmdArgs, "--log_level", logLevel)

	nodeCmd := exec.CommandContext(ctx, "remesd", cmdArgs...)

	if detach {
		nodeCmd.Stdout = nil
		nodeCmd.Stderr = nil

		if err := nodeCmd.Start(); err != nil {
			logger.Error("Failed to start node", zap.Error(err))
			return fmt.Errorf("failed to start node: %w\nMake sure remesd is installed and in PATH", err)
		}

		logger.Info("Node started", zap.Int("pid", nodeCmd.Process.Pid))
		fmt.Printf("✅ Node started (PID: %d)\n", nodeCmd.Process.Pid)
		fmt.Printf("   Use 'r3mes node status' to check status\n")
		fmt.Printf("   Use 'r3mes node sync --watch' to monitor sync progress\n")
	} else {
		nodeCmd.Stdout = os.Stdout
		nodeCmd.Stderr = os.Stderr

		fmt.Println("Starting node in foreground (Ctrl+C to stop)...")
		if err := nodeCmd.Run(); err != nil {
			return fmt.Errorf("node exited with error: %w", err)
		}
	}

	return nil
}

func stopNodeCobra(ctx context.Context, force bool) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	logger.Info("Stopping node", zap.Bool("force", force))

	if force {
		exec.Command("pkill", "-9", "-f", "remesd").Run()
	} else {
		// Graceful shutdown
		exec.Command("pkill", "-SIGTERM", "-f", "remesd").Run()

		// Wait for graceful shutdown
		config := GetConfig()
		for i := 0; i < 20; i++ {
			time.Sleep(500 * time.Millisecond)
			if status, _ := fetchNodeStatus(ctx, config); status == nil || !status.Running {
				break
			}
		}
	}

	logger.Info("Node stopped")
	fmt.Println("✅ Node stopped")
	return nil
}

func getNodeStatusCobra(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	status, err := fetchNodeStatus(ctx, config)

	if config.JSONOutput {
		if err != nil {
			output := map[string]any{"running": false, "error": err.Error()}
			data, _ := json.MarshalIndent(output, "", "  ")
			fmt.Println(string(data))
			return nil
		}
		data, _ := json.MarshalIndent(status, "", "  ")
		fmt.Println(string(data))
		return nil
	}

	if err != nil || status == nil || !status.Running {
		fmt.Println("Status: ❌ Not running")
		return nil
	}

	fmt.Println("Status: ✅ Running")
	fmt.Printf("Block Height: %s\n", status.BlockHeight)

	if status.Syncing {
		fmt.Println("Sync Status: 🔄 Syncing...")
	} else {
		fmt.Println("Sync Status: ✅ Synced")
	}

	if status.NodeID != "" {
		fmt.Printf("Node ID: %s\n", status.NodeID)
	}
	if status.Network != "" {
		fmt.Printf("Network: %s\n", status.Network)
	}

	return nil
}

func watchSyncStatus(ctx context.Context) error {
	config := GetConfig()
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	fmt.Println("Watching sync status (Ctrl+C to stop)...")
	fmt.Println()

	var lastHeight string

	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			status, err := fetchNodeStatus(ctx, config)
			if err != nil {
				fmt.Printf("\r\033[K⚠️ Error: %v", err)
				continue
			}

			if status == nil || !status.Running {
				fmt.Printf("\r\033[K❌ Node not running")
				continue
			}

			syncIcon := "✅"
			if status.Syncing {
				syncIcon = "🔄"
			}

			blocksPerSec := ""
			if lastHeight != "" && lastHeight != status.BlockHeight {
				blocksPerSec = " (syncing)"
			}
			lastHeight = status.BlockHeight

			fmt.Printf("\r\033[KBlock: %s %s%s", status.BlockHeight, syncIcon, blocksPerSec)
		}
	}
}

func fetchNodeStatus(ctx context.Context, config *Config) (*NodeStatus, error) {
	if config.RPCEndpoint == "" {
		return nil, fmt.Errorf("RPC endpoint not configured")
	}

	req, err := http.NewRequestWithContext(ctx, "GET", config.RPCEndpoint+"/status", nil)
	if err != nil {
		return nil, err
	}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var result struct {
		Result struct {
			NodeInfo struct {
				ID      string `json:"id"`
				Network string `json:"network"`
			} `json:"node_info"`
			SyncInfo struct {
				LatestBlockHeight string `json:"latest_block_height"`
				CatchingUp        bool   `json:"catching_up"`
			} `json:"sync_info"`
		} `json:"result"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &NodeStatus{
		Running:     true,
		BlockHeight: result.Result.SyncInfo.LatestBlockHeight,
		Syncing:     result.Result.SyncInfo.CatchingUp,
		NodeID:      result.Result.NodeInfo.ID,
		Network:     result.Result.NodeInfo.Network,
	}, nil
}
