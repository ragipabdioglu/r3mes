import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import "./WalletManager.css";

interface WalletInfo {
  address: string;
  balance: string;
  exists: boolean;
}

interface WalletManagerProps {
  onClose: () => void;
}

export default function WalletManager({ onClose }: WalletManagerProps) {
  const [mode, setMode] = useState<"view" | "create" | "import">("view");
  const [walletInfo, setWalletInfo] = useState<WalletInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mnemonic, setMnemonic] = useState<string | null>(null);
  const [showMnemonic, setShowMnemonic] = useState(false);
  const [importKey, setImportKey] = useState("");
  const [importMnemonic, setImportMnemonic] = useState("");

  useEffect(() => {
    loadWalletInfo();
  }, []);

  const loadWalletInfo = async () => {
    setLoading(true);
    try {
      const info = await invoke<WalletInfo>("get_wallet_info");
      setWalletInfo(info);
    } catch (error: any) {
      console.error("Failed to load wallet:", error);
      setError(error.message || "Failed to load wallet");
    } finally {
      setLoading(false);
    }
  };

  const handleCreateWallet = async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await invoke<{ address: string; mnemonic: string }>("create_wallet");
      setMnemonic(result.mnemonic);
      setShowMnemonic(true);
      await loadWalletInfo();
    } catch (error: any) {
      setError(error.message || "Failed to create wallet");
    } finally {
      setLoading(false);
    }
  };

  const handleImportPrivateKey = async () => {
    if (!importKey.trim()) {
      setError("Private key cannot be empty");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await invoke("import_wallet_from_private_key", { privateKey: importKey.trim() });
      setImportKey("");
      setMode("view");
      await loadWalletInfo();
    } catch (error: any) {
      setError(error.message || "Failed to import wallet");
    } finally {
      setLoading(false);
    }
  };

  const handleImportMnemonic = async () => {
    if (!importMnemonic.trim()) {
      setError("Mnemonic phrase cannot be empty");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await invoke("import_wallet_from_mnemonic", { mnemonic: importMnemonic.trim() });
      setImportMnemonic("");
      setMode("view");
      await loadWalletInfo();
    } catch (error: any) {
      setError(error.message || "Failed to import wallet");
    } finally {
      setLoading(false);
    }
  };

  const handleCopyAddress = async () => {
    if (walletInfo?.address) {
      await navigator.clipboard.writeText(walletInfo.address);
      // Show toast or notification
    }
  };

  const handleExportWallet = async () => {
    try {
      const exported = await invoke<{ encrypted: string }>("export_wallet");
      // Download or show export data
      const blob = new Blob([exported.encrypted], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "r3mes_wallet_backup.txt";
      a.click();
      URL.revokeObjectURL(url);
    } catch (error: any) {
      setError(error.message || "Failed to export wallet");
    }
  };

  if (loading && !walletInfo) {
    return (
      <div className="wallet-manager-overlay">
        <div className="wallet-manager">
          <div className="loading-spinner"></div>
          <p>Loading wallet...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="wallet-manager-overlay" onClick={onClose}>
      <div className="wallet-manager" onClick={(e) => e.stopPropagation()}>
        <div className="wallet-header">
          <h2>Wallet Management</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {mode === "view" && (
          <div className="wallet-content">
            {walletInfo?.exists ? (
              <>
                <div className="wallet-info">
                  <div className="info-item">
                    <label>Address</label>
                    <div className="address-display">
                      <code>{walletInfo.address}</code>
                      <button onClick={handleCopyAddress} className="copy-btn">Copy</button>
                    </div>
                  </div>
                  <div className="info-item">
                    <label>Balance</label>
                    <div className="balance-display">
                      {walletInfo.balance} REMES
                    </div>
                  </div>
                </div>
                <div className="wallet-actions">
                  <button onClick={handleExportWallet} className="btn-secondary">
                    Export Wallet
                  </button>
                  <button onClick={() => setMode("import")} className="btn-secondary">
                    Import New Wallet
                  </button>
                </div>
              </>
            ) : (
              <div className="no-wallet">
                <p>No wallet found. Create a new wallet or import an existing one.</p>
                <div className="wallet-actions">
                  <button onClick={handleCreateWallet} className="btn-primary" disabled={loading}>
                    Create New Wallet
                  </button>
                  <button onClick={() => setMode("import")} className="btn-secondary">
                    Import Wallet
                  </button>
                </div>
              </div>
            )}
          </div>
        )}

        {mode === "create" && showMnemonic && mnemonic && (
          <div className="wallet-content">
            <div className="mnemonic-warning">
              <h3>⚠️ Save Your Recovery Phrase</h3>
              <p>Write down these 12 words in order. Keep them safe and never share them with anyone.</p>
            </div>
            <div className="mnemonic-display">
              {mnemonic.split(" ").map((word, index) => (
                <div key={index} className="mnemonic-word">
                  <span className="word-number">{index + 1}</span>
                  <span className="word-text">{word}</span>
                </div>
              ))}
            </div>
            <div className="wallet-actions">
              <button onClick={() => { setShowMnemonic(false); setMode("view"); }} className="btn-primary">
                I've Saved It
              </button>
            </div>
          </div>
        )}

        {mode === "import" && (
          <div className="wallet-content">
            <div className="import-tabs">
              <button
                className={importKey ? "tab active" : "tab"}
                onClick={() => { setImportKey(""); setImportMnemonic(""); }}
              >
                Private Key
              </button>
              <button
                className={importMnemonic ? "tab active" : "tab"}
                onClick={() => { setImportKey(""); setImportMnemonic(""); }}
              >
                Mnemonic
              </button>
            </div>

            {importKey ? (
              <div className="import-form">
                <label>Private Key (Hex)</label>
                <textarea
                  value={importKey}
                  onChange={(e) => setImportKey(e.target.value)}
                  placeholder="Enter your private key (hex string)"
                  rows={3}
                />
                <button onClick={handleImportPrivateKey} className="btn-primary" disabled={loading}>
                  Import
                </button>
              </div>
            ) : (
              <div className="import-form">
                <label>Mnemonic Phrase (12 words)</label>
                <textarea
                  value={importMnemonic}
                  onChange={(e) => setImportMnemonic(e.target.value)}
                  placeholder="Enter your 12-word mnemonic phrase"
                  rows={3}
                />
                <button onClick={handleImportMnemonic} className="btn-primary" disabled={loading}>
                  Import
                </button>
              </div>
            )}

            <div className="wallet-actions">
              <button onClick={() => { setMode("view"); setImportKey(""); setImportMnemonic(""); }} className="btn-secondary">
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

