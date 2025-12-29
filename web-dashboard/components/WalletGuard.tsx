"use client";

import { useEffect, useState } from "react";
import WalletButton from "./WalletButton";

interface WalletGuardProps {
  children: React.ReactNode;
}

export default function WalletGuard({ children }: WalletGuardProps) {
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    const address = localStorage.getItem("keplr_address");
    setWalletAddress(address);

    const handleStorageChange = () => {
      const addr = localStorage.getItem("keplr_address");
      setWalletAddress(addr);
    };

    window.addEventListener("storage", handleStorageChange);
    const interval = setInterval(() => {
      const addr = localStorage.getItem("keplr_address");
      if (addr !== walletAddress) {
        setWalletAddress(addr);
      }
    }, 1000);

    return () => {
      window.removeEventListener("storage", handleStorageChange);
      clearInterval(interval);
    };
  }, [walletAddress]);

  if (!mounted) {
    return (
      <div className="flex items-center justify-center h-screen bg-slate-900">
        <div className="text-slate-400">Loading...</div>
      </div>
    );
  }

  if (!walletAddress) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/95 backdrop-blur-md">
        <div className="text-center max-w-md p-8 card">
          <h2 className="text-2xl font-bold mb-4 gradient-text">
            Erişim için Cüzdan Bağlayın
          </h2>
          <p className="text-slate-400 mb-6">
            Bu sayfayı kullanmak için cüzdanınızı bağlamanız gerekiyor.
          </p>
          <WalletButton />
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
