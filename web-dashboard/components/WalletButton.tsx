"use client";

import { useState, useEffect } from "react";
import { Wallet } from "lucide-react";
import { detectAvailableWallets } from "@/lib/cosmos-kit";
import { connectKeplrWallet, connectLeapWallet, connectCosmostationWallet, getKeplrBalance } from "@/lib/keplr";
import { logger } from "@/lib/logger";

export default function WalletButton() {
  const [isConnected, setIsConnected] = useState(false);
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [balance, setBalance] = useState<string>("0");
  const [availableWallets, setAvailableWallets] = useState<string[]>([]);
  const [showWalletMenu, setShowWalletMenu] = useState(false);

  useEffect(() => {
    // Detect available wallets
    const wallets = detectAvailableWallets();
    setAvailableWallets(wallets);

    // Check if wallet was previously connected
    const savedAddress = localStorage.getItem("keplr_address");
    if (savedAddress) {
      setWalletAddress(savedAddress);
      setIsConnected(true);
      getKeplrBalance(savedAddress)
        .then((bal) => setBalance(bal))
        .catch((err) => logger.error("Failed to get balance:", err));
    }
  }, []);

  const connectWallet = async (walletType: "keplr" | "leap" | "cosmostation") => {
    try {
      let address: string | null = null;

      switch (walletType) {
        case "keplr":
          address = await connectKeplrWallet();
          break;
        case "leap":
          address = await connectLeapWallet();
          break;
        case "cosmostation":
          address = await connectCosmostationWallet();
          break;
      }

      if (address) {
        setWalletAddress(address);
        setIsConnected(true);
        localStorage.setItem("keplr_address", address);
        // Dispatch custom event to notify WalletContext
        window.dispatchEvent(new Event("wallet-changed"));
        setShowWalletMenu(false);

        // Fetch balance
        try {
          const bal = await getKeplrBalance(address);
          setBalance(bal);
        } catch (error) {
          logger.error("Failed to fetch balance:", error);
        }
      }
    } catch (error: any) {
      alert(error.message || `Failed to connect ${walletType} wallet`);
    }
  };

  const disconnectWallet = () => {
    setWalletAddress(null);
    setIsConnected(false);
    setBalance("0");
    localStorage.removeItem("keplr_address");
    // Dispatch custom event to notify WalletContext
    window.dispatchEvent(new Event("wallet-changed"));
  };

  if (isConnected) {
    return (
      <div className="flex items-center gap-2">
        <div className="hidden md:flex flex-col items-end">
          <span className="text-[9px] sm:text-[10px] uppercase tracking-[0.15em] text-[var(--text-secondary)]">
            Balance
          </span>
          <span className="text-xs sm:text-sm font-semibold text-[var(--accent-primary)]">
            {(parseFloat(balance) / 1e6).toFixed(2)} REMES
          </span>
        </div>
        <div className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 md:px-4 py-1 sm:py-1.5 md:py-2 rounded-full bg-[var(--bg-secondary)] border border-[var(--border-color)]">
          <Wallet className="w-3 h-3 sm:w-3.5 sm:h-3.5 md:w-4 md:h-4 text-[var(--text-primary)]" />
          <span className="text-[9px] sm:text-xs font-mono text-[var(--text-primary)]">
            {walletAddress?.slice(0, 6)}...{walletAddress?.slice(-4)}
          </span>
        </div>
        <button
          onClick={disconnectWallet}
          className="hidden sm:block text-[9px] sm:text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
        >
          Disconnect
        </button>
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setShowWalletMenu(!showWalletMenu)}
        className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 md:px-4 py-1 sm:py-1.5 md:py-2 
                   rounded-full bg-[var(--accent-primary)] text-white font-medium 
                   hover:opacity-90 transition-all duration-200 hover:scale-105
                   text-[10px] sm:text-xs md:text-sm"
      >
        <Wallet className="w-3 h-3 sm:w-3.5 sm:h-3.5 md:w-4 md:h-4" />
        <span className="hidden sm:inline">Connect</span>
        <span className="sm:hidden">Connect</span>
      </button>

      {showWalletMenu && (
        <div className="absolute right-0 mt-2 w-48 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg shadow-lg z-50">
          <div className="p-2">
            {availableWallets.length === 0 ? (
              <div className="px-3 py-2 text-sm text-[var(--text-secondary)]">
                No wallets detected. Please install Keplr, Leap, or Cosmostation.
              </div>
            ) : (
              <>
                {availableWallets.includes("keplr") && (
                  <button
                    onClick={() => connectWallet("keplr")}
                    className="w-full text-left px-3 py-2 text-sm text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded transition-colors"
                  >
                    Keplr Wallet
                  </button>
                )}
                {availableWallets.includes("leap") && (
                  <button
                    onClick={() => connectWallet("leap")}
                    className="w-full text-left px-3 py-2 text-sm text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded transition-colors"
                  >
                    Leap Wallet
                  </button>
                )}
                {availableWallets.includes("cosmostation") && (
                  <button
                    onClick={() => connectWallet("cosmostation")}
                    className="w-full text-left px-3 py-2 text-sm text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded transition-colors"
                  >
                    Cosmostation Wallet
                  </button>
                )}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

