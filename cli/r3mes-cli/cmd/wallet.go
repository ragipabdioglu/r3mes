package cmd

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/btcsuite/btcd/btcec/v2"
	"github.com/cosmos/cosmos-sdk/types/bech32"
	"github.com/spf13/cobra"
	"github.com/tyler-smith/go-bip39"
	"go.uber.org/zap"
	"golang.org/x/crypto/pbkdf2"
	"golang.org/x/crypto/ripemd160"
	"golang.org/x/term"
)

// Wallet represents a wallet with encrypted storage
type Wallet struct {
	Address             string `json:"address"`
	PublicKey           string `json:"public_key"`
	EncryptedPrivateKey string `json:"encrypted_private_key,omitempty"`
	EncryptedMnemonic   string `json:"encrypted_mnemonic,omitempty"`
	Salt                string `json:"salt"`
	CreatedAt           string `json:"created_at"`
	PrivateKey          string `json:"-"`
	Mnemonic            string `json:"-"`
}

func newWalletCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "wallet",
		Short: "Wallet management",
		Long:  "Create, import, export, and manage R3MES wallets.",
	}

	cmd.AddCommand(newWalletCreateCmd())
	cmd.AddCommand(newWalletImportCmd())
	cmd.AddCommand(newWalletBalanceCmd())
	cmd.AddCommand(newWalletExportCmd())
	cmd.AddCommand(newWalletListCmd())

	return cmd
}

func newWalletCreateCmd() *cobra.Command {
	var noEncrypt bool

	cmd := &cobra.Command{
		Use:   "create",
		Short: "Create a new wallet",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return createWalletCobra(ctx, noEncrypt)
		},
	}

	cmd.Flags().BoolVar(&noEncrypt, "no-encrypt", false, "Skip encryption (not recommended)")

	return cmd
}

func newWalletImportCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "import <mnemonic_or_private_key>",
		Short: "Import wallet from mnemonic or private key",
		Long: `Import a wallet using either:
  - A 12/24 word mnemonic phrase: "word1 word2 ... word12"
  - A hex private key: 0x... or without prefix`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			input := strings.Join(args, " ")
			return importWalletCobra(ctx, input)
		},
	}
}

func newWalletBalanceCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "balance [address]",
		Short: "Check wallet balance",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			config := GetConfig()

			var address string
			if len(args) > 0 {
				address = args[0]
			} else {
				wallet, err := LoadDefaultWallet(config)
				if err != nil {
					return fmt.Errorf("no address provided and no default wallet found")
				}
				address = wallet.Address
			}

			return getBalanceCobra(ctx, address)
		},
	}
}

func newWalletExportCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "export",
		Short: "Export wallet private key and mnemonic",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return exportWalletCobra(ctx)
		},
	}
}

func newWalletListCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List all wallets",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			return listWalletsCobra(ctx)
		},
	}
}

func createWalletCobra(ctx context.Context, noEncrypt bool) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	logger.Info("Creating new wallet")

	entropy, err := bip39.NewEntropy(128)
	if err != nil {
		return fmt.Errorf("failed to generate entropy: %w", err)
	}

	mnemonic, err := bip39.NewMnemonic(entropy)
	if err != nil {
		return fmt.Errorf("failed to generate mnemonic: %w", err)
	}

	seed := bip39.NewSeed(mnemonic, "")
	privateKey := hex.EncodeToString(seed[:32])

	address, err := GenerateCosmosAddress(seed[:32])
	if err != nil {
		return fmt.Errorf("failed to generate address: %w", err)
	}

	wallet := Wallet{
		Address:   address,
		CreatedAt: time.Now().UTC().Format(time.RFC3339),
	}

	if !noEncrypt {
		fmt.Print("Enter password to encrypt wallet: ")
		password, err := readPassword()
		if err != nil {
			return fmt.Errorf("failed to read password: %w", err)
		}

		if len(password) > 0 {
			salt := generateSalt()
			wallet.Salt = hex.EncodeToString(salt)

			encryptedPrivateKey, err := encryptData(privateKey, password, salt)
			if err != nil {
				return fmt.Errorf("failed to encrypt private key: %w", err)
			}
			wallet.EncryptedPrivateKey = encryptedPrivateKey

			encryptedMnemonic, err := encryptData(mnemonic, password, salt)
			if err != nil {
				return fmt.Errorf("failed to encrypt mnemonic: %w", err)
			}
			wallet.EncryptedMnemonic = encryptedMnemonic

			logger.Info("Wallet encrypted successfully")
		}
	}

	if err := SaveWallet(wallet, "default", config); err != nil {
		return fmt.Errorf("failed to save wallet: %w", err)
	}

	logger.Info("Wallet created", zap.String("address", wallet.Address))

	if config.JSONOutput {
		output := map[string]string{"address": wallet.Address}
		if wallet.EncryptedPrivateKey == "" {
			output["mnemonic"] = mnemonic
		}
		data, _ := json.MarshalIndent(output, "", "  ")
		fmt.Println(string(data))
		return nil
	}

	fmt.Println("\n✅ Wallet created successfully!")
	fmt.Printf("Address: %s\n", wallet.Address)

	if wallet.EncryptedPrivateKey == "" {
		fmt.Printf("\n🔑 Mnemonic (SAVE THIS SECURELY!):\n%s\n", mnemonic)
		fmt.Println("\n⚠️  WARNING: Never share your mnemonic phrase with anyone!")
	} else {
		fmt.Println("\n🔐 Wallet encrypted. Use 'r3mes wallet export' to view mnemonic.")
	}

	return nil
}

func importWalletCobra(ctx context.Context, input string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	input = strings.TrimSpace(input)

	logger.Info("Importing wallet")

	var wallet Wallet

	if strings.Contains(input, " ") {
		// Mnemonic import
		if !bip39.IsMnemonicValid(input) {
			return fmt.Errorf("invalid mnemonic phrase")
		}

		seed := bip39.NewSeed(input, "")
		address, err := GenerateCosmosAddress(seed[:32])
		if err != nil {
			return fmt.Errorf("failed to generate address: %w", err)
		}

		wallet = Wallet{
			Address:    address,
			PrivateKey: hex.EncodeToString(seed[:32]),
			Mnemonic:   input,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339),
		}
	} else {
		// Private key import
		input = strings.TrimPrefix(input, "0x")
		if len(input) != 64 {
			return fmt.Errorf("invalid private key (must be 64 hex characters)")
		}

		keyBytes, err := hex.DecodeString(input)
		if err != nil {
			return fmt.Errorf("failed to decode private key: %w", err)
		}

		address, err := GenerateCosmosAddress(keyBytes)
		if err != nil {
			return fmt.Errorf("failed to generate address: %w", err)
		}

		wallet = Wallet{
			Address:    address,
			PrivateKey: input,
			CreatedAt:  time.Now().UTC().Format(time.RFC3339),
		}
	}

	// Encrypt if password provided
	fmt.Print("Enter password to encrypt wallet (leave empty for no encryption): ")
	password, _ := readPassword()

	if len(password) > 0 {
		salt := generateSalt()
		wallet.Salt = hex.EncodeToString(salt)

		encryptedPrivateKey, err := encryptData(wallet.PrivateKey, password, salt)
		if err != nil {
			return fmt.Errorf("failed to encrypt: %w", err)
		}
		wallet.EncryptedPrivateKey = encryptedPrivateKey

		if wallet.Mnemonic != "" {
			encryptedMnemonic, err := encryptData(wallet.Mnemonic, password, salt)
			if err != nil {
				return fmt.Errorf("failed to encrypt mnemonic: %w", err)
			}
			wallet.EncryptedMnemonic = encryptedMnemonic
		}

		// Clear plaintext
		wallet.PrivateKey = ""
		wallet.Mnemonic = ""
	}

	if err := SaveWallet(wallet, "default", config); err != nil {
		return fmt.Errorf("failed to save wallet: %w", err)
	}

	logger.Info("Wallet imported", zap.String("address", wallet.Address))

	fmt.Println("\n✅ Wallet imported successfully!")
	fmt.Printf("Address: %s\n", wallet.Address)

	return nil
}

func getBalanceCobra(ctx context.Context, address string) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	if config.RPCEndpoint == "" {
		return fmt.Errorf("RPC endpoint not configured")
	}

	logger.Debug("Fetching balance", zap.String("address", address))

	url := fmt.Sprintf("%s/cosmos/bank/v1beta1/balances/%s",
		getRESTEndpoint(config.RPCEndpoint), address)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to query balance: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP error: %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	var result struct {
		Balances []struct {
			Denom  string `json:"denom"`
			Amount string `json:"amount"`
		} `json:"balances"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return fmt.Errorf("failed to parse response: %w", err)
	}

	if config.JSONOutput {
		fmt.Println(string(body))
		return nil
	}

	fmt.Printf("\nBalance for %s:\n", address)
	if len(result.Balances) == 0 {
		fmt.Println("  No balances found")
	} else {
		for _, b := range result.Balances {
			fmt.Printf("  %s %s\n", b.Amount, b.Denom)
		}
	}

	return nil
}

func exportWalletCobra(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	wallet, err := LoadDefaultWallet(config)
	if err != nil {
		return fmt.Errorf("failed to load wallet: %w", err)
	}

	fmt.Println("\n⚠️  WARNING: Exporting wallet private information!")

	if wallet.EncryptedPrivateKey != "" {
		fmt.Print("Enter wallet password: ")
		password, err := readPassword()
		if err != nil {
			return fmt.Errorf("failed to read password: %w", err)
		}

		salt, _ := hex.DecodeString(wallet.Salt)
		privateKey, err := decryptData(wallet.EncryptedPrivateKey, password, salt)
		if err != nil {
			return fmt.Errorf("invalid password or corrupted wallet")
		}

		mnemonic, _ := decryptData(wallet.EncryptedMnemonic, password, salt)

		if config.JSONOutput {
			output := map[string]string{
				"address":     wallet.Address,
				"private_key": privateKey,
			}
			if mnemonic != "" {
				output["mnemonic"] = mnemonic
			}
			data, _ := json.MarshalIndent(output, "", "  ")
			fmt.Println(string(data))
			return nil
		}

		fmt.Printf("\nAddress: %s\n", wallet.Address)
		if mnemonic != "" {
			fmt.Printf("Mnemonic: %s\n", mnemonic)
		}
		fmt.Printf("Private Key: %s\n", privateKey)
	} else {
		fmt.Printf("Address: %s\n", wallet.Address)
		fmt.Println("Private Key: [Not stored - wallet was created without encryption]")
	}

	return nil
}

func listWalletsCobra(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	config := GetConfig()
	entries, err := os.ReadDir(config.WalletPath)
	if err != nil {
		fmt.Println("No wallets found. Create one with: r3mes wallet create")
		return nil
	}

	type walletInfo struct {
		Name    string `json:"name"`
		Address string `json:"address"`
	}
	var wallets []walletInfo

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
			wallets = append(wallets, walletInfo{Name: name, Address: wallet.Address})
		}
	}

	if config.JSONOutput {
		data, _ := json.MarshalIndent(wallets, "", "  ")
		fmt.Println(string(data))
		return nil
	}

	fmt.Println("Wallets:")
	if len(wallets) == 0 {
		fmt.Println("  No wallets found")
	} else {
		for _, w := range wallets {
			fmt.Printf("  %s: %s\n", w.Name, w.Address)
		}
	}

	return nil
}

// GenerateCosmosAddress generates a Cosmos address from private key bytes
func GenerateCosmosAddress(privateKeyBytes []byte) (string, error) {
	if len(privateKeyBytes) == 0 {
		return "", fmt.Errorf("private key cannot be empty")
	}

	privKey, pubKey := btcec.PrivKeyFromBytes(privateKeyBytes)
	if privKey == nil {
		return "", fmt.Errorf("invalid private key")
	}

	compressedPubKey := pubKey.SerializeCompressed()
	sha256Hash := sha256.Sum256(compressedPubKey)

	// Note: ripemd160 is required for Cosmos address compatibility
	ripemd160Hasher := ripemd160.New()
	ripemd160Hasher.Write(sha256Hash[:])
	addressBytes := ripemd160Hasher.Sum(nil)

	address, err := bech32.ConvertAndEncode("remes", addressBytes)
	if err != nil {
		return "", fmt.Errorf("failed to encode bech32 address: %w", err)
	}

	return address, nil
}

// SaveWallet saves a wallet to disk
func SaveWallet(wallet Wallet, name string, config *Config) error {
	if err := os.MkdirAll(config.WalletPath, 0700); err != nil {
		return err
	}

	walletPath := filepath.Join(config.WalletPath, name+".json")
	data, err := json.MarshalIndent(wallet, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(walletPath, data, 0600)
}

// LoadDefaultWallet loads the default wallet
func LoadDefaultWallet(config *Config) (*Wallet, error) {
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

// Encryption utilities
func generateSalt() []byte {
	salt := make([]byte, 32)
	rand.Read(salt)
	return salt
}

func deriveKey(password string, salt []byte) []byte {
	return pbkdf2.Key([]byte(password), salt, 100000, 32, sha256.New)
}

func encryptData(plaintext, password string, salt []byte) (string, error) {
	key := deriveKey(password, salt)
	block, err := aes.NewCipher(key)
	if err != nil {
		return "", err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}

	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", err
	}

	ciphertext := gcm.Seal(nonce, nonce, []byte(plaintext), nil)
	return hex.EncodeToString(ciphertext), nil
}

func decryptData(encrypted, password string, salt []byte) (string, error) {
	key := deriveKey(password, salt)
	ciphertext, err := hex.DecodeString(encrypted)
	if err != nil {
		return "", err
	}

	block, err := aes.NewCipher(key)
	if err != nil {
		return "", err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}

	nonceSize := gcm.NonceSize()
	if len(ciphertext) < nonceSize {
		return "", fmt.Errorf("ciphertext too short")
	}

	nonce, ciphertext := ciphertext[:nonceSize], ciphertext[nonceSize:]
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return "", err
	}

	return string(plaintext), nil
}

func readPassword() (string, error) {
	bytePassword, err := term.ReadPassword(int(syscall.Stdin))
	if err != nil {
		return "", err
	}
	fmt.Println()
	return string(bytePassword), nil
}
