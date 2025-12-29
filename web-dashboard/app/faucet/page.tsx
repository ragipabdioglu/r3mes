"use client";

import { useState, useEffect } from "react";
import { useWallet } from "@/contexts/WalletContext";
import { claimFaucet, getFaucetStatus, FaucetStatus } from "@/lib/api";
import { logger } from "@/lib/logger";
import { toast } from "@/lib/toast";
import { Copy, ExternalLink, CheckCircle, XCircle, Clock, AlertCircle, Droplet } from "lucide-react";
import WalletGuard from "@/components/WalletGuard";

function FaucetPageContent() {
  const { walletAddress } = useWallet();
  const [address, setAddress] = useState("");
  const [amount, setAmount] = useState("");
  const [isClaiming, setIsClaiming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<{ txHash: string; amount: string; nextClaim: string } | null>(null);
  const [faucetStatus, setFaucetStatus] = useState<FaucetStatus | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    // Load faucet status
    getFaucetStatus()
      .then(setFaucetStatus)
      .catch((err) => {
        logger.error("Failed to load faucet status:", err);
      });
  }, []);

  useEffect(() => {
    if (walletAddress) {
      setAddress(walletAddress);
    }
  }, [walletAddress]);

  const handleClaim = async () => {
    if (!address.trim()) {
      setError("Please enter a wallet address");
      return;
    }

    // Validate address format (basic check)
    if (!address.startsWith("remes")) {
      setError("Invalid address format. R3MES addresses start with 'remes'");
      return;
    }

    setIsClaiming(true);
    setError(null);
    setSuccess(null);

    try {
      const response = await claimFaucet({
        address: address.trim(),
        amount: amount.trim() || undefined,
      });

      if (response.success) {
        setSuccess({
          txHash: response.tx_hash || "",
          amount: response.amount,
          nextClaim: response.next_claim_available_at || "",
        });
        setAmount(""); // Reset amount for next claim
      } else {
        setError(response.message || "Failed to claim tokens");
      }
    } catch (err: any) {
      if (err.response?.status === 429) {
        const errorDetail = err.response.data?.detail;
        if (typeof errorDetail === "object" && errorDetail.next_claim_available_at) {
          const nextClaim = new Date(errorDetail.next_claim_available_at);
          const now = new Date();
          const hoursUntil = Math.ceil((nextClaim.getTime() - now.getTime()) / (1000 * 60 * 60));
          setError(
            `Rate limit exceeded. You can only claim once per day. Next claim available in ${hoursUntil} hour(s).`
          );
        } else {
          setError("Rate limit exceeded. You can only claim once per day.");
        }
      } else if (err.response?.status === 503) {
        setError("Faucet is currently disabled. Please try again later.");
      } else if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError(err.message || "Failed to claim tokens. Please try again later.");
      }
    } finally {
      setIsClaiming(false);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success("Copied to clipboard");
  };

  const formatAmount = (amountStr: string): string => {
    // Convert uremes to REMES (1 REMES = 1,000,000 uremes)
    const uremes = amountStr.replace("uremes", "").replace("remes", "").trim();
    const num = parseInt(uremes);
    if (isNaN(num)) return amountStr;
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(2)} REMES`;
    }
    return `${num} uremes`;
  };

  const formatNextClaim = (isoString: string): string => {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffHours = Math.ceil(diffMs / (1000 * 60 * 60));
    const diffMins = Math.ceil(diffMs / (1000 * 60));

    if (diffMins < 60) {
      return `${diffMins} minute(s)`;
    } else if (diffHours < 24) {
      return `${diffHours} hour(s)`;
    } else {
      const diffDays = Math.ceil(diffHours / 24);
      return `${diffDays} day(s)`;
    }
  };

  if (!mounted) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center">
        <div className="text-slate-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
      <div className="container mx-auto max-w-full sm:max-w-xl md:max-w-2xl">
        <div className="text-center mb-6 sm:mb-8">
          <div className="flex items-center justify-center gap-2 sm:gap-3 mb-3 sm:mb-4">
            <Droplet className="w-6 h-6 sm:w-8 sm:h-8 text-[#06b6d4]" />
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold gradient-text">Token Faucet</h1>
          </div>
          <p className="text-sm sm:text-base text-slate-400">
            Get free R3MES tokens to start mining and using the network
          </p>
        </div>

        {/* Faucet Status */}
        {faucetStatus && (
          <div className="card bg-slate-800/50 border-slate-700 mb-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-slate-300 mb-1">Faucet Status</h3>
                <p className="text-xs text-slate-400">
                  {faucetStatus.enabled ? (
                    <span className="text-green-400 flex items-center gap-1">
                      <CheckCircle className="w-3 h-3" />
                      Active
                    </span>
                  ) : (
                    <span className="text-red-400 flex items-center gap-1">
                      <XCircle className="w-3 h-3" />
                      Disabled
                    </span>
                  )}
                </p>
              </div>
              <div className="text-right">
                <p className="text-sm text-slate-300">
                  {formatAmount(faucetStatus.amount_per_claim)} per claim
                </p>
                <p className="text-xs text-slate-400">Max: {formatAmount(faucetStatus.daily_limit)}/day</p>
              </div>
            </div>
          </div>
        )}

        {/* Main Form */}
        <div className="card p-4 sm:p-5 md:p-6">
          <div className="space-y-4 sm:space-y-5 md:space-y-6">
            {/* Wallet Address Input */}
            <div>
              <label className="block text-sm font-semibold text-slate-300 mb-2">
                Wallet Address
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={address}
                  onChange={(e) => setAddress(e.target.value)}
                  placeholder="remes1..."
                  disabled={isClaiming || !!walletAddress}
                  className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-[#06b6d4] focus:ring-2 focus:ring-[#06b6d4]/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                />
                {address && (
                  <button
                    onClick={() => copyToClipboard(address)}
                    className="px-4 py-3 bg-slate-700 border border-slate-600 rounded-lg hover:bg-slate-600 transition-colors"
                    title="Copy address"
                  >
                    <Copy className="w-5 h-5 text-slate-300" />
                  </button>
                )}
              </div>
              {walletAddress && (
                <p className="text-xs text-slate-400 mt-1">
                  Using connected wallet address
                </p>
              )}
            </div>

            {/* Amount Input (Optional) */}
            <div>
              <label className="block text-sm font-semibold text-slate-300 mb-2">
                Amount (Optional)
              </label>
              <input
                type="text"
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                placeholder={`Default: ${faucetStatus ? formatAmount(faucetStatus.amount_per_claim) : "1 REMES"}`}
                disabled={isClaiming || !faucetStatus?.enabled}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-[#06b6d4] focus:ring-2 focus:ring-[#06b6d4]/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <p className="text-xs text-slate-400 mt-1">
                Leave empty to use default amount. Max: {faucetStatus ? formatAmount(faucetStatus.daily_limit) : "5 REMES"}
              </p>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <h3 className="text-sm font-semibold text-red-400 mb-1">Error</h3>
                    <p className="text-sm text-red-300">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div className="bg-green-900/20 border border-green-500/50 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <h3 className="text-sm font-semibold text-green-400 mb-2">Success!</h3>
                    <p className="text-sm text-green-300 mb-3">
                      Successfully claimed {formatAmount(success.amount)} to your wallet.
                    </p>
                    {success.txHash && (
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xs text-slate-400">Transaction:</span>
                        <code className="text-xs bg-slate-800 px-2 py-1 rounded text-green-300 font-mono">
                          {success.txHash.slice(0, 16)}...
                        </code>
                        <button
                          onClick={() => copyToClipboard(success.txHash)}
                          className="p-1 hover:bg-slate-700 rounded transition-colors"
                          title="Copy transaction hash"
                        >
                          <Copy className="w-3 h-3 text-slate-400" />
                        </button>
                      </div>
                    )}
                    {success.nextClaim && (
                      <div className="flex items-center gap-2 text-xs text-slate-400">
                        <Clock className="w-3 h-3" />
                        <span>Next claim available in {formatNextClaim(success.nextClaim)}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Claim Button */}
            <button
              onClick={handleClaim}
              disabled={isClaiming || !address.trim() || !faucetStatus?.enabled}
              className="w-full btn-primary py-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isClaiming ? (
                <>
                  <span className="animate-pulse">●</span>
                  <span>Claiming...</span>
                </>
              ) : (
                <>
                  <Droplet className="w-5 h-5" />
                  <span>Claim Tokens</span>
                </>
              )}
            </button>

            {/* Rate Limit Info */}
            <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Clock className="w-5 h-5 text-slate-400 flex-shrink-0 mt-0.5" />
                <div>
                  <h3 className="text-sm font-semibold text-slate-300 mb-1">Rate Limits</h3>
                  <p className="text-xs text-slate-400">
                    You can claim tokens once per day per IP address and per wallet address.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function FaucetPage() {
  return (
    <WalletGuard>
      <FaucetPageContent />
    </WalletGuard>
  );
}

