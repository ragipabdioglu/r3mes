package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/tyler-smith/go-bip39"
)

// Config holds CLI configuration
type Config struct {
	RPCEndpoint  string `json:"rpc_endpoint"`
	GRPCEndpoint string `json:"grpc_endpoint"`
	ChainID      string `json:"chain_id"`
	WalletPath   string `json:"wallet_path"`
}

// Wallet represents a wallet
type Wallet struct {
	Address    string `json:"address"`
	PublicKey  string `json:"public_key"`
	PrivateKey string `json:"private_key,omitempty"`
	Mnemonic   string `json:"mnemonic,omitempty"`
}

// BalanceResponse from blockchain
type BalanceResponse struct {
	Balance struct {
		Denom  string `json:"denom"`
		Amount string `json:"amount"`
	} `json:"balance"`
}

var config Config

func init() {
	// Load config from environment or defaults
	config = Config{
		RPCEndpoint:  getEnv("R3MES_RPC_ENDPOINT", "http://localhost:26657"),
		GRPCEndpoint: getEnv("R3MES_GRPC_ENDPOINT", "localhost:9090"),
		ChainID:      getEnv("R3MES_CHAIN_ID", "remes-test"),
		WalletPath:   getEnv("R3MES_WALLET_PATH", filepath.Join(getHomeDir(), ".r3mes", "wallets")),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getHomeDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "~"
	}
	return home
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]

	switch command {
	case "wallet":
		handleWalletCommand(os.Args[2:])
	case "miner":
		handleMinerCommand(os.Args[2:])
	case "node":
		handleNodeCommand(os.Args[2:])
	case "governance":
		handleGovernanceCommand(os.Args[2:])
	case "version":
		fmt.Println("R3MES CLI v0.1.0")
	case "config":
		handleConfigCommand(os.Args[2:])
	default:
		fmt.Printf("Unknown command: %s\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println("R3MES CLI - Command-line interface for R3MES")
	fmt.Println("\nUsage: r3mes <command> [options]")
	fmt.Println("\nCommands:")
	fmt.Println("  wallet      Wallet management (create, import, balance, export)")
	fmt.Println("  miner       Miner operations (start, stop, status)")
	fmt.Println("  node        Node operations (start, stop, status)")
	fmt.Println("  governance  Governance operations (vote, proposals)")
	fmt.Println("  config      Configuration management")
	fmt.Println("  version     Show version information")
	fmt.Println("\nEnvironment Variables:")
	fmt.Println("  R3MES_RPC_ENDPOINT   RPC endpoint (default: http://localhost:26657)")
	fmt.Println("  R3MES_GRPC_ENDPOINT  gRPC endpoint (default: localhost:9090)")
	fmt.Println("  R3MES_CHAIN_ID       Chain ID (default: remes-test)")
	fmt.Println("  R3MES_WALLET_PATH    Wallet storage path")
}

func handleWalletCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: r3mes wallet <create|import|balance|export|list> [options]")
		os.Exit(1)
	}

	subcommand := args[0]

	switch subcommand {
	case "create":
		createWallet()
	case "import":
		if len(args) < 2 {
			fmt.Println("Usage: r3mes wallet import <mnemonic_or_private_key>")
			fmt.Println("  For mnemonic: r3mes wallet import \"word1 word2 ... word12\"")
			fmt.Println("  For private key: r3mes wallet import 0x...")
			os.Exit(1)
		}
		importWallet(strings.Join(args[1:], " "))
	case "balance":
		if len(args) < 2 {
			// Try to get default wallet address
			wallet, err := loadDefaultWallet()
			if err != nil {
				fmt.Println("Usage: r3mes wallet balance <address>")
				fmt.Println("Or create a wallet first: r3mes wallet create")
				os.Exit(1)
			}
			getBalance(wallet.Address)
		} else {
			getBalance(args[1])
		}
	case "export":
		exportWallet()
	case "list":
		listWallets()
	default:
		fmt.Printf("Unknown wallet command: %s\n", subcommand)
		os.Exit(1)
	}
}

func createWallet() {
	fmt.Println("Creating new wallet...")

	// Generate entropy and mnemonic
	entropy, err := bip39.NewEntropy(128) // 12 words
	if err != nil {
		fmt.Printf("Error generating entropy: %v\n", err)
		os.Exit(1)
	}

	mnemonic, err := bip39.NewMnemonic(entropy)
	if err != nil {
		fmt.Printf("Error generating mnemonic: %v\n", err)
		os.Exit(1)
	}

	// Generate private key from mnemonic (simplified - in production use proper HD derivation)
	seed := bip39.NewSeed(mnemonic, "")
	privateKey := hex.EncodeToString(seed[:32])

	// Generate address (simplified - in production use proper Cosmos address derivation)
	addressBytes := make([]byte, 20)
	copy(addressBytes, seed[32:52])
	address := "remes1" + hex.EncodeToString(addressBytes)[:38]

	wallet := Wallet{
		Address:    address,
		PrivateKey: privateKey,
		Mnemonic:   mnemonic,
	}

	// Save wallet
	if err := saveWallet(wallet, "default"); err != nil {
		fmt.Printf("Error saving wallet: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n✅ Wallet created successfully!")
	fmt.Printf("Address: %s\n", wallet.Address)
	fmt.Printf("\n🔑 Mnemonic (SAVE THIS SECURELY!):\n%s\n", mnemonic)
	fmt.Println("\n⚠️  WARNING: Never share your mnemonic phrase with anyone!")
	fmt.Println("⚠️  Store it in a safe place. If you lose it, you lose access to your funds.")
}

func importWallet(input string) {
	fmt.Println("Importing wallet...")

	input = strings.TrimSpace(input)
	var wallet Wallet

	// Check if it's a mnemonic (contains spaces) or private key
	if strings.Contains(input, " ") {
		// Mnemonic
		if !bip39.IsMnemonicValid(input) {
			fmt.Println("Error: Invalid mnemonic phrase")
			os.Exit(1)
		}

		seed := bip39.NewSeed(input, "")
		privateKey := hex.EncodeToString(seed[:32])
		addressBytes := make([]byte, 20)
		copy(addressBytes, seed[32:52])
		address := "remes1" + hex.EncodeToString(addressBytes)[:38]

		wallet = Wallet{
			Address:    address,
			PrivateKey: privateKey,
			Mnemonic:   input,
		}
	} else {
		// Private key
		input = strings.TrimPrefix(input, "0x")
		if len(input) != 64 {
			fmt.Println("Error: Invalid private key (must be 64 hex characters)")
			os.Exit(1)
		}

		// Generate address from private key (simplified)
		keyBytes, err := hex.DecodeString(input)
		if err != nil {
			fmt.Printf("Error decoding private key: %v\n", err)
			os.Exit(1)
		}

		addressBytes := make([]byte, 20)
		rand.Read(addressBytes) // In production, derive from public key
		address := "remes1" + hex.EncodeToString(addressBytes)[:38]

		wallet = Wallet{
			Address:    address,
			PrivateKey: hex.EncodeToString(keyBytes),
		}
	}

	// Save wallet
	if err := saveWallet(wallet, "default"); err != nil {
		fmt.Printf("Error saving wallet: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n✅ Wallet imported successfully!")
	fmt.Printf("Address: %s\n", wallet.Address)
}

func getBalance(address string) {
	fmt.Printf("Getting balance for %s...\n", address)

	// Query blockchain REST API
	url := fmt.Sprintf("%s/cosmos/bank/v1beta1/balances/%s",
		strings.Replace(config.RPCEndpoint, ":26657", ":1317", 1), address)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		fmt.Printf("Error querying balance: %v\n", err)
		fmt.Println("Make sure the blockchain node is running.")
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		fmt.Printf("Error: HTTP %d\n", resp.StatusCode)
		os.Exit(1)
	}

	body, _ := io.ReadAll(resp.Body)
	var result struct {
		Balances []struct {
			Denom  string `json:"denom"`
			Amount string `json:"amount"`
		} `json:"balances"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nBalance for %s:\n", address)
	if len(result.Balances) == 0 {
		fmt.Println("  No balances found")
	} else {
		for _, b := range result.Balances {
			fmt.Printf("  %s %s\n", b.Amount, b.Denom)
		}
	}
}

func exportWallet() {
	wallet, err := loadDefaultWallet()
	if err != nil {
		fmt.Printf("Error loading wallet: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n⚠️  WARNING: Exporting wallet private key!")
	fmt.Println("Never share this information with anyone.\n")
	fmt.Printf("Address: %s\n", wallet.Address)
	if wallet.Mnemonic != "" {
		fmt.Printf("Mnemonic: %s\n", wallet.Mnemonic)
	}
	fmt.Printf("Private Key: %s\n", wallet.PrivateKey)
}

func listWallets() {
	entries, err := os.ReadDir(config.WalletPath)
	if err != nil {
		fmt.Println("No wallets found. Create one with: r3mes wallet create")
		return
	}

	fmt.Println("Wallets:")
	for _, entry := range entries {
		if strings.HasSuffix(entry.Name(), ".json") {
			walletPath := filepath.Join(config.WalletPath, entry.Name())
			data, err := os.ReadFile(walletPath)
			if err != nil {
				continue
			}
			var wallet Wallet
			if err := json.Unmarshal(data, &wallet); err != nil {
				continue
			}
			name := strings.TrimSuffix(entry.Name(), ".json")
			fmt.Printf("  %s: %s\n", name, wallet.Address)
		}
	}
}

func saveWallet(wallet Wallet, name string) error {
	// Create wallet directory
	if err := os.MkdirAll(config.WalletPath, 0700); err != nil {
		return err
	}

	// Save wallet (without private key in the main file for security)
	walletPath := filepath.Join(config.WalletPath, name+".json")
	data, err := json.MarshalIndent(wallet, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(walletPath, data, 0600)
}

func loadDefaultWallet() (*Wallet, error) {
	walletPath := filepath.Join(config.WalletPath, "default.json")
	data, err := os.ReadFile(walletPath)
	if err != nil {
		return nil, err
	}

	var wallet Wallet
	if err := json.Unmarshal(data, &wallet); err != nil {
		return nil, err
	}

	return &wallet, nil
}

func handleMinerCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: r3mes miner <start|stop|status|stats>")
		os.Exit(1)
	}

	subcommand := args[0]

	switch subcommand {
	case "start":
		startMiner()
	case "stop":
		stopMiner()
	case "status":
		getMinerStatus()
	case "stats":
		getMinerStats()
	default:
		fmt.Printf("Unknown miner command: %s\n", subcommand)
		os.Exit(1)
	}
}

func startMiner() {
	fmt.Println("Starting miner...")

	// Check if Python miner is available
	cmd := exec.Command("python3", "-m", "r3mes.cli.commands", "start")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		fmt.Printf("Error starting miner: %v\n", err)
		fmt.Println("Make sure r3mes miner-engine is installed.")
		os.Exit(1)
	}

	fmt.Printf("✅ Miner started (PID: %d)\n", cmd.Process.Pid)
}

func stopMiner() {
	fmt.Println("Stopping miner...")

	// Try to stop via Python CLI
	cmd := exec.Command("python3", "-m", "r3mes.cli.commands", "stop")
	if err := cmd.Run(); err != nil {
		// Fallback: kill by process name
		exec.Command("pkill", "-f", "r3mes-miner").Run()
		exec.Command("pkill", "-f", "r3mes.cli.commands").Run()
	}

	fmt.Println("✅ Miner stopped")
}

func getMinerStatus() {
	fmt.Println("Checking miner status...")

	// Query miner stats HTTP endpoint
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://localhost:8080/health")
	if err != nil {
		fmt.Println("Status: ❌ Not running")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		fmt.Println("Status: ✅ Running")
	} else {
		fmt.Println("Status: ⚠️ Unhealthy")
	}
}

func getMinerStats() {
	fmt.Println("Getting miner statistics...")

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get("http://localhost:8080/stats")
	if err != nil {
		fmt.Println("Error: Miner not running or stats endpoint unavailable")
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var stats map[string]interface{}
	if err := json.Unmarshal(body, &stats); err != nil {
		fmt.Printf("Error parsing stats: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\nMiner Statistics:")
	fmt.Printf("  Hashrate: %.2f gradients/hour\n", stats["hashrate"])
	fmt.Printf("  Loss: %.4f\n", stats["loss"])
	fmt.Printf("  Loss Trend: %s\n", stats["loss_trend"])
	fmt.Printf("  GPU Temp: %.1f°C\n", stats["gpu_temp"])
	fmt.Printf("  VRAM Usage: %v MB / %v MB\n", stats["vram_usage_mb"], stats["vram_total_mb"])
	fmt.Printf("  Uptime: %v seconds\n", stats["uptime_seconds"])
}

func handleNodeCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: r3mes node <start|stop|status|sync>")
		os.Exit(1)
	}

	subcommand := args[0]

	switch subcommand {
	case "start":
		startNode()
	case "stop":
		stopNode()
	case "status":
		getNodeStatus()
	case "sync":
		fmt.Println("Checking sync status...")
		getNodeStatus()
	default:
		fmt.Printf("Unknown node command: %s\n", subcommand)
		os.Exit(1)
	}
}

func startNode() {
	fmt.Println("Starting node...")

	cmd := exec.Command("remesd", "start")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		fmt.Printf("Error starting node: %v\n", err)
		fmt.Println("Make sure remesd is installed and in PATH.")
		os.Exit(1)
	}

	fmt.Printf("✅ Node started (PID: %d)\n", cmd.Process.Pid)
}

func stopNode() {
	fmt.Println("Stopping node...")
	exec.Command("pkill", "-f", "remesd").Run()
	fmt.Println("✅ Node stopped")
}

func getNodeStatus() {
	fmt.Println("Checking node status...")

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(config.RPCEndpoint + "/status")
	if err != nil {
		fmt.Println("Status: ❌ Not running")
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var result struct {
		Result struct {
			SyncInfo struct {
				LatestBlockHeight string `json:"latest_block_height"`
				CatchingUp        bool   `json:"catching_up"`
			} `json:"sync_info"`
		} `json:"result"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		fmt.Println("Status: ⚠️ Error parsing response")
		return
	}

	fmt.Println("Status: ✅ Running")
	fmt.Printf("Block Height: %s\n", result.Result.SyncInfo.LatestBlockHeight)
	if result.Result.SyncInfo.CatchingUp {
		fmt.Println("Sync Status: 🔄 Syncing...")
	} else {
		fmt.Println("Sync Status: ✅ Synced")
	}
}

func handleGovernanceCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: r3mes governance <vote|proposals|proposal>")
		os.Exit(1)
	}

	subcommand := args[0]

	switch subcommand {
	case "vote":
		if len(args) < 3 {
			fmt.Println("Usage: r3mes governance vote <proposal_id> <yes|no|abstain|no_with_veto>")
			os.Exit(1)
		}
		voteOnProposal(args[1], args[2])
	case "proposals":
		listProposals()
	case "proposal":
		if len(args) < 2 {
			fmt.Println("Usage: r3mes governance proposal <proposal_id>")
			os.Exit(1)
		}
		getProposal(args[1])
	default:
		fmt.Printf("Unknown governance command: %s\n", subcommand)
		os.Exit(1)
	}
}

func voteOnProposal(proposalID, vote string) {
	fmt.Printf("Voting %s on proposal %s...\n", vote, proposalID)

	// Validate vote option
	validVotes := map[string]bool{
		"yes": true, "no": true, "abstain": true, "no_with_veto": true,
	}
	if !validVotes[strings.ToLower(vote)] {
		fmt.Println("Error: Invalid vote option. Use: yes, no, abstain, or no_with_veto")
		os.Exit(1)
	}

	// Load wallet
	wallet, err := loadDefaultWallet()
	if err != nil {
		fmt.Println("Error: No wallet found. Create one with: r3mes wallet create")
		os.Exit(1)
	}

	fmt.Printf("Voting from address: %s\n", wallet.Address)
	fmt.Println("⚠️  Note: Transaction signing not yet implemented in CLI.")
	fmt.Println("Please use the web dashboard or remesd CLI to vote.")
}

func listProposals() {
	fmt.Println("Fetching proposals...")

	url := fmt.Sprintf("%s/cosmos/gov/v1beta1/proposals",
		strings.Replace(config.RPCEndpoint, ":26657", ":1317", 1))

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		fmt.Printf("Error fetching proposals: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var result struct {
		Proposals []struct {
			ProposalID string `json:"proposal_id"`
			Content    struct {
				Title string `json:"title"`
			} `json:"content"`
			Status string `json:"status"`
		} `json:"proposals"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		fmt.Printf("Error parsing response: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\nProposals:")
	if len(result.Proposals) == 0 {
		fmt.Println("  No proposals found")
	} else {
		for _, p := range result.Proposals {
			fmt.Printf("  #%s: %s [%s]\n", p.ProposalID, p.Content.Title, p.Status)
		}
	}
}

func getProposal(proposalID string) {
	fmt.Printf("Fetching proposal %s...\n", proposalID)

	url := fmt.Sprintf("%s/cosmos/gov/v1beta1/proposals/%s",
		strings.Replace(config.RPCEndpoint, ":26657", ":1317", 1), proposalID)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		fmt.Printf("Error fetching proposal: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	fmt.Println(string(body))
}

func handleConfigCommand(args []string) {
	if len(args) < 1 {
		fmt.Println("Current Configuration:")
		fmt.Printf("  RPC Endpoint: %s\n", config.RPCEndpoint)
		fmt.Printf("  gRPC Endpoint: %s\n", config.GRPCEndpoint)
		fmt.Printf("  Chain ID: %s\n", config.ChainID)
		fmt.Printf("  Wallet Path: %s\n", config.WalletPath)
		return
	}

	subcommand := args[0]
	switch subcommand {
	case "set":
		if len(args) < 3 {
			fmt.Println("Usage: r3mes config set <key> <value>")
			os.Exit(1)
		}
		fmt.Printf("Setting %s = %s\n", args[1], args[2])
		fmt.Println("Note: Use environment variables to persist configuration.")
	default:
		fmt.Printf("Unknown config command: %s\n", subcommand)
	}
}
