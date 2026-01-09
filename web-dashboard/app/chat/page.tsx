"use client";

import { useState, useEffect } from "react";
import ChatInterface from "@/components/ChatInterface";
import WalletGuard from "@/components/WalletGuard";
import { useUserInfo } from "@/hooks/useMinerData";
import { logger } from "@/lib/logger";

export default function ChatPage() {
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  
  // Use React Query hook instead of setInterval polling
  const { data: userInfo, error: userInfoError } = useUserInfo(walletAddress);
  const credits = userInfo?.credits ?? null;

  useEffect(() => {
    const address = localStorage.getItem("keplr_address");
    setWalletAddress(address);

    // Listen for wallet address changes
    const handleStorageChange = () => {
      const addr = localStorage.getItem("keplr_address");
      if (addr !== walletAddress) {
        setWalletAddress(addr);
      }
    };

    window.addEventListener("storage", handleStorageChange);
    // Also check periodically (less frequent, as React Query handles refetching)
    const interval = setInterval(handleStorageChange, 10000); // Check every 10 seconds

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      clearInterval(interval);
    };
  }, [walletAddress]);

  useEffect(() => {
    if (userInfoError) {
      logger.error("Failed to fetch credits:", userInfoError);
    }
  }, [userInfoError]);

  return (
    <WalletGuard>
      <div className="flex flex-col md:flex-row flex-1 overflow-hidden min-h-0">
        {/* Sidebar */}
        <div className="w-full md:w-64 lg:w-80 border-r border-[var(--border-color)] 
                        bg-[var(--bg-secondary)] flex flex-col overflow-hidden shrink-0">
          <div className="p-4 sm:p-5 md:p-6 space-y-6 overflow-y-auto flex-1">
            {/* Model Selector */}
            <div>
              <label className="text-sm text-[var(--text-secondary)] font-medium mb-2 block">
                Model
              </label>
              <select className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] 
                               rounded-lg px-4 py-2 text-[var(--text-primary)] text-sm 
                               focus:outline-none focus:border-[var(--accent-primary)]">
                <option>BitNet Base</option>
                <option>Coder</option>
                <option>Law</option>
              </select>
            </div>

            {/* Credits */}
            <div className="card">
              <p className="text-sm text-[var(--text-secondary)] font-medium mb-1">Remaining</p>
              <p className="text-2xl font-bold text-[var(--accent-primary)]">
                {credits !== null ? Math.floor(credits) : "500"} Messages
              </p>
            </div>

            {/* New Chat */}
            <button className="w-full btn-secondary text-sm py-2">
              New Chat
            </button>
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col overflow-hidden min-w-0">
          <ChatInterface walletAddress={walletAddress || ""} />
        </div>
      </div>
    </WalletGuard>
  );
}
