"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { getUserInfo, UserInfo } from "@/lib/api";
import { logger } from "@/lib/logger";

interface WalletContextType {
  walletAddress: string | null;
  credits: number | null;
  userInfo: UserInfo | null;
  isLoading: boolean;
  refreshUserInfo: () => Promise<void>;
}

const WalletContext = createContext<WalletContextType | undefined>(undefined);

export function WalletProvider({ children }: { children: ReactNode }) {
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [credits, setCredits] = useState<number | null>(null);
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refreshUserInfo = async () => {
    if (!walletAddress) {
      setCredits(null);
      setUserInfo(null);
      return;
    }

    try {
      const info = await getUserInfo(walletAddress);
      setUserInfo(info);
      setCredits(info.credits);
    } catch (err) {
      logger.error("Failed to fetch user info:", err);
      setCredits(null);
      setUserInfo(null);
    }
  };

  useEffect(() => {
    // Initial load
    const savedAddress = localStorage.getItem("keplr_address");
    setWalletAddress(savedAddress);
    setIsLoading(false);

    // Listen for storage changes (from other tabs/windows)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === "keplr_address") {
        const address = e.newValue;
        setWalletAddress(address);
      }
    };

    // Listen for custom events (from same tab)
    const handleWalletChange = () => {
      const address = localStorage.getItem("keplr_address");
      setWalletAddress(address);
    };

    window.addEventListener("storage", handleStorageChange);
    window.addEventListener("wallet-changed", handleWalletChange);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      window.removeEventListener("wallet-changed", handleWalletChange);
    };
  }, []);

  useEffect(() => {
    if (walletAddress) {
      refreshUserInfo();
    } else {
      setCredits(null);
      setUserInfo(null);
    }
  }, [walletAddress]);

  return (
    <WalletContext.Provider
      value={{
        walletAddress,
        credits,
        userInfo,
        isLoading,
        refreshUserInfo,
      }}
    >
      {children}
    </WalletContext.Provider>
  );
}

export function useWallet() {
  const context = useContext(WalletContext);
  if (context === undefined) {
    throw new Error("useWallet must be used within a WalletProvider");
  }
  return context;
}

