package cmd

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"github.com/btcsuite/btcd/btcec/v2"
	"github.com/btcsuite/btcd/btcec/v2/ecdsa"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

// TxBuilder handles transaction construction and signing
type TxBuilder struct {
	ChainID       string
	AccountNum    uint64
	Sequence      uint64
	GasLimit      uint64
	GasPrice      string
	Memo          string
	TimeoutHeight uint64
}

// SignedTx represents a signed transaction ready for broadcast
type SignedTx struct {
	Body       TxBody   `json:"body"`
	AuthInfo   AuthInfo `json:"auth_info"`
	Signatures []string `json:"signatures"`
}

// TxBody contains the transaction messages
type TxBody struct {
	Messages      []json.RawMessage `json:"messages"`
	Memo          string            `json:"memo,omitempty"`
	TimeoutHeight string            `json:"timeout_height,omitempty"`
}

// AuthInfo contains signer info and fee
type AuthInfo struct {
	SignerInfos []SignerInfo `json:"signer_infos"`
	Fee         Fee          `json:"fee"`
}

// SignerInfo contains public key and signing mode
type SignerInfo struct {
	PublicKey PublicKey `json:"public_key"`
	ModeInfo  ModeInfo  `json:"mode_info"`
	Sequence  string    `json:"sequence"`
}

// PublicKey for Cosmos SDK
type PublicKey struct {
	Type string `json:"@type"`
	Key  string `json:"key"`
}

// ModeInfo specifies signing mode
type ModeInfo struct {
	Single SingleMode `json:"single"`
}

// SingleMode for direct signing
type SingleMode struct {
	Mode string `json:"mode"`
}

// Fee structure
type Fee struct {
	Amount   []Coin `json:"amount"`
	GasLimit string `json:"gas_limit"`
}

// Coin represents a token amount
type Coin struct {
	Denom  string `json:"denom"`
	Amount string `json:"amount"`
}

// BroadcastRequest for REST API
type BroadcastRequest struct {
	TxBytes string `json:"tx_bytes"`
	Mode    string `json:"mode"`
}

// BroadcastResponse from REST API
type BroadcastResponse struct {
	TxResponse struct {
		Code      int    `json:"code"`
		TxHash    string `json:"txhash"`
		RawLog    string `json:"raw_log"`
		GasWanted string `json:"gas_wanted"`
		GasUsed   string `json:"gas_used"`
	} `json:"tx_response"`
}

func newTxCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "tx",
		Short: "Transaction operations",
		Long:  "Sign and broadcast transactions to the R3MES network.",
	}

	cmd.AddCommand(newTxSendCmd())
	cmd.AddCommand(newTxSignCmd())
	cmd.AddCommand(newTxBroadcastCmd())

	return cmd
}

func newTxSendCmd() *cobra.Command {
	var (
		amount   string
		denom    string
		memo     string
		gasLimit uint64
		gasPrice string
	)

	cmd := &cobra.Command{
		Use:   "send <to_address>",
		Short: "Send tokens to another address",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			toAddress := args[0]

			config := GetConfig()
			if config.RPCEndpoint == "" {
				return fmt.Errorf("RPC endpoint not configured. Run: r3mes config init")
			}

			// Load wallet
			wallet, err := LoadDefaultWallet(config)
			if err != nil {
				return fmt.Errorf("no wallet found. Create one with: r3mes wallet create")
			}

			// Decrypt private key if needed
			privateKey, err := getPrivateKey(wallet)
			if err != nil {
				return fmt.Errorf("failed to get private key: %w", err)
			}

			// Get account info
			accountNum, sequence, err := getAccountInfo(ctx, config, wallet.Address)
			if err != nil {
				return fmt.Errorf("failed to get account info: %w", err)
			}

			// Build send message
			sendMsg := buildSendMsg(wallet.Address, toAddress, amount, denom)

			// Create and sign transaction
			builder := &TxBuilder{
				ChainID:    config.ChainID,
				AccountNum: accountNum,
				Sequence:   sequence,
				GasLimit:   gasLimit,
				GasPrice:   gasPrice,
				Memo:       memo,
			}

			signedTx, err := builder.BuildAndSign([]json.RawMessage{sendMsg}, privateKey)
			if err != nil {
				return fmt.Errorf("failed to sign transaction: %w", err)
			}

			// Broadcast
			result, err := broadcastTx(ctx, config, signedTx)
			if err != nil {
				return fmt.Errorf("failed to broadcast: %w", err)
			}

			if result.TxResponse.Code != 0 {
				return fmt.Errorf("transaction failed: %s", result.TxResponse.RawLog)
			}

			logger.Info("Transaction sent successfully",
				zap.String("txhash", result.TxResponse.TxHash),
				zap.String("gas_used", result.TxResponse.GasUsed))

			fmt.Printf("✅ Transaction sent!\n")
			fmt.Printf("   TxHash: %s\n", result.TxResponse.TxHash)
			fmt.Printf("   Gas Used: %s\n", result.TxResponse.GasUsed)

			return nil
		},
	}

	cmd.Flags().StringVar(&amount, "amount", "", "Amount to send (required)")
	cmd.Flags().StringVar(&denom, "denom", "uremes", "Token denomination")
	cmd.Flags().StringVar(&memo, "memo", "", "Transaction memo")
	cmd.Flags().Uint64Var(&gasLimit, "gas", 200000, "Gas limit")
	cmd.Flags().StringVar(&gasPrice, "gas-price", "0.025uremes", "Gas price")
	cmd.MarkFlagRequired("amount")

	return cmd
}

func newTxSignCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "sign <tx_file>",
		Short: "Sign a transaction from file",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			// Implementation for offline signing
			fmt.Println("Offline signing - reading transaction from file...")
			return fmt.Errorf("offline signing not yet implemented")
		},
	}
}

func newTxBroadcastCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "broadcast <signed_tx_file>",
		Short: "Broadcast a signed transaction",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("Broadcasting signed transaction...")
			return fmt.Errorf("broadcast from file not yet implemented")
		},
	}
}

// BuildAndSign creates and signs a transaction
func (b *TxBuilder) BuildAndSign(messages []json.RawMessage, privateKeyHex string) (*SignedTx, error) {
	// Decode private key
	privKeyBytes, err := hex.DecodeString(privateKeyHex)
	if err != nil {
		return nil, fmt.Errorf("invalid private key: %w", err)
	}

	privKey, pubKey := btcec.PrivKeyFromBytes(privKeyBytes)
	if privKey == nil {
		return nil, fmt.Errorf("failed to parse private key")
	}

	compressedPubKey := pubKey.SerializeCompressed()

	// Build transaction body
	txBody := TxBody{
		Messages: messages,
		Memo:     b.Memo,
	}
	if b.TimeoutHeight > 0 {
		txBody.TimeoutHeight = strconv.FormatUint(b.TimeoutHeight, 10)
	}

	// Build auth info
	authInfo := AuthInfo{
		SignerInfos: []SignerInfo{
			{
				PublicKey: PublicKey{
					Type: "/cosmos.crypto.secp256k1.PubKey",
					Key:  base64.StdEncoding.EncodeToString(compressedPubKey),
				},
				ModeInfo: ModeInfo{
					Single: SingleMode{Mode: "SIGN_MODE_DIRECT"},
				},
				Sequence: strconv.FormatUint(b.Sequence, 10),
			},
		},
		Fee: Fee{
			Amount:   []Coin{{Denom: "uremes", Amount: "5000"}},
			GasLimit: strconv.FormatUint(b.GasLimit, 10),
		},
	}

	// Create sign doc
	signDoc := map[string]any{
		"body_bytes":      txBody,
		"auth_info_bytes": authInfo,
		"chain_id":        b.ChainID,
		"account_number":  strconv.FormatUint(b.AccountNum, 10),
	}

	signDocBytes, err := json.Marshal(signDoc)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal sign doc: %w", err)
	}

	// Sign
	hash := sha256.Sum256(signDocBytes)
	signature := ecdsa.Sign(privKey, hash[:])
	sigBytes := signature.Serialize()

	return &SignedTx{
		Body:       txBody,
		AuthInfo:   authInfo,
		Signatures: []string{base64.StdEncoding.EncodeToString(sigBytes)},
	}, nil
}

func buildSendMsg(from, to, amount, denom string) json.RawMessage {
	msg := map[string]any{
		"@type":        "/cosmos.bank.v1beta1.MsgSend",
		"from_address": from,
		"to_address":   to,
		"amount": []map[string]string{
			{"denom": denom, "amount": amount},
		},
	}
	data, _ := json.Marshal(msg)
	return data
}

func getAccountInfo(ctx context.Context, config *Config, address string) (uint64, uint64, error) {
	url := fmt.Sprintf("%s/cosmos/auth/v1beta1/accounts/%s",
		getRESTEndpoint(config.RPCEndpoint), address)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return 0, 0, err
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return 0, 0, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var result struct {
		Account struct {
			AccountNumber string `json:"account_number"`
			Sequence      string `json:"sequence"`
		} `json:"account"`
	}

	if err := json.Unmarshal(body, &result); err != nil {
		return 0, 0, err
	}

	accountNum, _ := strconv.ParseUint(result.Account.AccountNumber, 10, 64)
	sequence, _ := strconv.ParseUint(result.Account.Sequence, 10, 64)

	return accountNum, sequence, nil
}

func broadcastTx(ctx context.Context, config *Config, tx *SignedTx) (*BroadcastResponse, error) {
	txBytes, err := json.Marshal(tx)
	if err != nil {
		return nil, err
	}

	broadcastReq := BroadcastRequest{
		TxBytes: base64.StdEncoding.EncodeToString(txBytes),
		Mode:    "BROADCAST_MODE_SYNC",
	}

	reqBody, _ := json.Marshal(broadcastReq)

	url := fmt.Sprintf("%s/cosmos/tx/v1beta1/txs", getRESTEndpoint(config.RPCEndpoint))
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)

	var result BroadcastResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	return &result, nil
}

func getRESTEndpoint(rpcEndpoint string) string {
	// Convert RPC endpoint to REST endpoint
	// RPC: :26657 -> REST: :1317
	return rpcEndpoint[:len(rpcEndpoint)-5] + "1317"
}

func getPrivateKey(wallet *Wallet) (string, error) {
	if wallet.PrivateKey != "" {
		return wallet.PrivateKey, nil
	}

	if wallet.EncryptedPrivateKey == "" {
		return "", fmt.Errorf("wallet has no private key stored")
	}

	fmt.Print("Enter wallet password: ")
	password, err := readPassword()
	if err != nil {
		return "", err
	}

	salt, err := hex.DecodeString(wallet.Salt)
	if err != nil {
		return "", err
	}

	return decryptData(wallet.EncryptedPrivateKey, password, salt)
}
