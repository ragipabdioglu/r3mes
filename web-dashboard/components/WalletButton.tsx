"use client";

import { useState, useEffect, useRef } from "react";
import { Wallet, ChevronDown } from "lucide-react";
import { detectAvailableWallets } from "@/lib/cosmos-kit";
import { connectKeplrWallet, connectLeapWallet, connectCosmostationWallet, getKeplrBalance } from "@/lib/keplr";
import { logger } from "@/lib/logger";

export default function WalletButton() {
  const [isConnected, setIsConnected] = useState(false);
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [balance, setBalance] = useState<string>("0");
  const [availableWallets, setAvailableWallets] = useState<string[]>([]);
  const [showWalletMenu, setShowWalletMenu] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

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

  // Handle click outside to close menu
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node) &&
          buttonRef.current && !buttonRef.current.contains(event.target as Node)) {
        setShowWalletMenu(false);
      }
    };

    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && showWalletMenu) {
        setShowWalletMenu(false);
        buttonRef.current?.focus();
      }
    };

    if (showWalletMenu) {
      document.addEventListener('mousedown', handleClickOutside);
      document.addEventListener('keydown', handleEscape);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [showWalletMenu]);

  const connectWallet = async (walletType: "keplr" | "leap" | "cosmostation") => {
    setIsConnecting(true);
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
      logger.error(`Failed to connect ${walletType} wallet:`, error);
      // Show user-friendly error message
      const errorMessage = error.message?.includes('rejected') 
        ? 'Connection was cancelled by user'
        : `Failed to connect ${walletType} wallet. Please try again.`;
      alert(errorMessage);
    } finally {
      setIsConnecting(false);
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
          <span 
            className="text-[9px] sm:text-[10px] uppercase tracking-[0.15em] text-[var(--text-secondary)]"
            aria-label="Wallet balance"
          >
            Balance
          </span>
          <span 
            className="text-xs sm:text-sm font-semibold text-[var(--accent-primary)]"
            aria-label={`${(parseFloat(balance) / 1e6).toFixed(2)} REMES tokens`}
          >
            {(parseFloat(balance) / 1e6).toFixed(2)} REMES
          </span>
        </div>
        <div 
          className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 md:px-4 py-1 sm:py-1.5 md:py-2 rounded-full bg-[var(--bg-secondary)] border border-[var(--border-color)]"
          role="status"
          aria-label={`Connected wallet: ${walletAddress?.slice(0, 6)}...${walletAddress?.slice(-4)}`}
        >
          <Wallet 
            className="w-3 h-3 sm:w-3.5 sm:h-3.5 md:w-4 md:h-4 text-[var(--text-primary)]" 
            aria-hidden="true"
          />
          <span 
            className="text-[9px] sm:text-xs font-mono text-[var(--text-primary)]"
            aria-hidden="true"
          >
            {walletAddress?.slice(0, 6)}...{walletAddress?.slice(-4)}
          </span>
        </div>
        <button
          onClick={disconnectWallet}
          className="hidden sm:block text-[9px] sm:text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-2 py-1"
          style={{ minHeight: '44px', minWidth: '44px' }}
          aria-label="Disconnect wallet"
          tabIndex={0}
        >
          Disconnect
        </button>
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        ref={buttonRef}
        onClick={() => setShowWalletMenu(!showWalletMenu)}
        disabled={isConnecting}
        className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 md:px-4 py-1 sm:py-1.5 md:py-2 
                   rounded-full bg-[var(--accent-primary)] text-white font-medium 
                   hover:opacity-90 transition-all duration-200 hover:scale-105
                   text-[10px] sm:text-xs md:text-sm
                   focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2
                   disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        style={{ minHeight: '44px', minWidth: '44px' }}
        aria-label={showWalletMenu ? "Close wallet menu" : "Open wallet menu"}
        aria-expanded={showWalletMenu}
        aria-haspopup="menu"
        tabIndex={0}
      >
        <Wallet className="w-3 h-3 sm:w-3.5 sm:h-3.5 md:w-4 md:h-4" aria-hidden="true" />
        <span className="hidden sm:inline">
          {isConnecting ? 'Connecting...' : 'Connect'}
        </span>
        <span className="sm:hidden">
          {isConnecting ? 'Connecting...' : 'Connect'}
        </span>
        <ChevronDown 
          className={`w-3 h-3 transition-transform ${showWalletMenu ? 'rotate-180' : ''}`}
          aria-hidden="true"
        />
      </button>

      {showWalletMenu && (
        <div 
          ref={menuRef}
          className="absolute right-0 mt-2 w-48 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg shadow-lg z-50"
          role="menu"
          aria-label="Wallet selection menu"
        >
          <div className="p-2">
            {availableWallets.length === 0 ? (
              <div 
                className="px-3 py-2 text-sm text-[var(--text-secondary)]"
                role="status"
                aria-live="polite"
              >
                No wallets detected. Please install Keplr, Leap, or Cosmostation.
              </div>
            ) : (
              <>
                <div className="sr-only" aria-live="polite">
                  {availableWallets.length} wallet{availableWallets.length > 1 ? 's' : ''} available
                </div>
                {availableWallets.includes("keplr") && (
                  <button
                    onClick={() => connectWallet("keplr")}
                    disabled={isConnecting}
                    className="w-full text-left px-3 py-2 text-sm text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{ minHeight: '44px' }}
                    role="menuitem"
                    tabIndex={0}
                    aria-label="Connect with Keplr Wallet"
                  >
                    Keplr Wallet
                  </button>
                )}
                {availableWallets.includes("leap") && (
                  <button
                    onClick={() => connectWallet("leap")}
                    disabled={isConnecting}
                    className="w-full text-left px-3 py-2 text-sm text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{ minHeight: '44px' }}
                    role="menuitem"
                    tabIndex={0}
                    aria-label="Connect with Leap Wallet"
                  >
                    Leap Wallet
                  </button>
                )}
                {availableWallets.includes("cosmostation") && (
                  <button
                    onClick={() => connectWallet("cosmostation")}
                    disabled={isConnecting}
                    className="w-full text-left px-3 py-2 text-sm text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] rounded transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{ minHeight: '44px' }}
                    role="menuitem"
                    tabIndex={0}
                    aria-label="Connect with Cosmostation Wallet"
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

