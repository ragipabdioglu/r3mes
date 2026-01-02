"use client";

import { useEffect, useState, useMemo } from "react";
import { Wallet, History, Copy, CheckCircle, XCircle, Clock, AlertCircle } from "lucide-react";
import { useWallet } from "@/contexts/WalletContext";
import { useTransactionHistory } from "@/hooks/useTransactionHistory";
import { Transaction } from "@/lib/api";
import WalletGuard from "@/components/WalletGuard";
import StatCard from "@/components/StatCard";
import { SkeletonStatCard, SkeletonTable } from "@/components/SkeletonLoader";
import { formatCredits } from "@/utils/numberFormat";
import { getUserFriendlyError, getErrorTitle } from "@/utils/errorMessages";
import { toast } from "@/lib/toast";
import { useVirtualization } from "@/hooks/useVirtualization";
import { useAnnouncer } from "@/hooks/useAccessibility";
import { formatAddress, formatTimeAgo, formatHash } from "@/utils/formatters";

function WalletPageContent() {
  const { walletAddress, userInfo } = useWallet();
  const { data: txHistory, isLoading: txLoading, error: txError } = useTransactionHistory(
    walletAddress,
    50,
    !!walletAddress
  );
  const { announceSuccess } = useAnnouncer();
  
  const [mounted, setMounted] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    setMounted(true);
  }, []);

  const isLoading = !mounted || txLoading;
  const transactions = txHistory?.transactions || [];
  
  // Filter transactions based on search query
  const filteredTransactions = useMemo(() => {
    if (!searchQuery.trim()) return transactions;
    const query = searchQuery.toLowerCase();
    return transactions.filter((tx: Transaction) => 
      tx.hash?.toLowerCase().includes(query) ||
      tx.type?.toLowerCase().includes(query) ||
      tx.to?.toLowerCase().includes(query) ||
      tx.from?.toLowerCase().includes(query)
    );
  }, [transactions, searchQuery]);

  // Use virtualization for large lists (>50 items)
  const shouldVirtualize = filteredTransactions.length > 50;
  const { virtualItems, totalHeight, handleScroll } = useVirtualization(
    filteredTransactions,
    {
      itemHeight: 72, // Approximate row height
      containerHeight: 500,
      overscan: 5,
    }
  );

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard');
    announceSuccess('Copied to clipboard');
  };

  // Use formatters from utils
  const displayAddress = (address: string) => formatAddress(address);
  const displayTimestamp = (timestamp: string | number) => formatTimeAgo(timestamp);
  const displayHash = (hash: string) => formatHash(hash);

  if (!mounted || isLoading) {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
        <div className="container mx-auto max-w-full sm:max-w-2xl md:max-w-4xl lg:max-w-6xl xl:max-w-7xl">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 md:gap-6 mb-6 sm:mb-8">
            {[1, 2, 3].map((i) => (
              <SkeletonStatCard key={i} />
            ))}
          </div>
          <SkeletonTable />
        </div>
      </div>
    );
  }

  if (txError) {
    return (
      <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
        <div className="container mx-auto max-w-full sm:max-w-2xl md:max-w-4xl lg:max-w-6xl xl:max-w-7xl">
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-6 sm:mb-8 gradient-text">My Wallet</h1>
          <div className="card bg-red-900/20 border-red-500/50">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-red-400" />
              <div>
                <h3 className="text-lg font-semibold text-red-400 mb-1">
                  {getErrorTitle(txError)}
                </h3>
                <p className="text-red-300 text-sm">
                  {getUserFriendlyError(txError)}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
      <div className="container mx-auto max-w-full sm:max-w-2xl md:max-w-4xl lg:max-w-6xl xl:max-w-7xl">
        <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-6 sm:mb-8 gradient-text">My Wallet</h1>

        {/* Wallet Overview */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 md:gap-6 mb-6 sm:mb-8">
          <StatCard
            label="Wallet Address"
            value={walletAddress ? displayAddress(walletAddress) : "Not Connected"}
            icon={<Wallet className="w-5 h-5" />}
            subtext={
              walletAddress ? (
                <button
                  onClick={() => copyToClipboard(walletAddress)}
                  className="text-xs text-[var(--text-secondary)] hover:text-[var(--accent-primary)] flex items-center gap-1 mt-1"
                >
                  <Copy className="w-3 h-3" />
                  Copy address
                </button>
              ) : null
            }
          />
          <StatCard
            label="Credits"
            value={userInfo ? formatCredits(userInfo.credits) : "0"}
            icon={<History className="w-5 h-5" />}
            subtext={userInfo?.is_miner ? "Miner Account" : "Regular Account"}
          />
          <StatCard
            label="Total Transactions"
            value={transactions.length.toLocaleString()}
            icon={<History className="w-5 h-5" />}
          />
        </div>

        {/* Transaction History */}
        <div className="card p-4 sm:p-5 md:p-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 sm:mb-6 gap-3 sm:gap-4">
            <h2 className="text-lg sm:text-xl font-semibold text-[var(--text-primary)] flex items-center gap-2">
              <History className="w-4 h-4 sm:w-5 sm:h-5" />
              Transaction History
            </h2>
            <div className="flex items-center gap-2 sm:gap-4 w-full sm:w-auto">
              <input
                type="text"
                placeholder="Search transactions..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                aria-label="Search transactions"
                className="w-full sm:w-auto px-3 sm:px-4 py-1.5 sm:py-2 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent-primary)] text-xs sm:text-sm"
              />
            </div>
          </div>

          {txLoading ? (
            <div className="text-center py-8 sm:py-12">
              <div className="text-sm sm:text-base text-[var(--text-secondary)]">Loading transactions...</div>
            </div>
          ) : filteredTransactions.length === 0 ? (
            <div className="text-center py-8 sm:py-12">
              <div className="text-sm sm:text-base text-[var(--text-secondary)]">
                {searchQuery ? "No transactions match your search" : "No transactions found"}
              </div>
              <p className="text-xs sm:text-sm text-[var(--text-muted)] mt-2">
                {searchQuery 
                  ? "Try a different search term" 
                  : "Your transaction history will appear here once you start using R3MES"}
              </p>
            </div>
          ) : (
            <div className="overflow-x-auto -mx-4 sm:-mx-5 md:-mx-6 px-4 sm:px-5 md:px-6">
              <table className="w-full min-w-[800px]">
                <thead>
                  <tr className="border-b border-[var(--border-color)]">
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Hash
                    </th>
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Type
                    </th>
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      From / To
                    </th>
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Amount
                    </th>
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Status
                    </th>
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Time
                    </th>
                    <th className="text-left py-4 px-4 text-sm font-semibold text-[var(--text-secondary)]">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {(shouldVirtualize ? virtualItems.map(v => filteredTransactions[v.index]) : filteredTransactions).map((tx: Transaction, i: number) => (
                    <tr
                      key={i}
                      className="border-b border-[var(--border-color)] hover:bg-[var(--bg-tertiary)] transition-colors"
                    >
                      <td className="py-4 px-4">
                        <span className="text-sm text-[var(--text-secondary)] font-mono">
                          {displayHash(tx.hash || "")}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <span className="text-sm text-[var(--text-primary)] capitalize">
                          {tx.type.replace('cosmos.bank.v1beta1.Msg', '').replace(/([A-Z])/g, ' $1').trim()}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <div className="flex flex-col gap-1">
                          {tx.from && (
                            <span className="text-xs text-[var(--text-secondary)] font-mono">
                              From: {displayAddress(tx.from)}
                            </span>
                          )}
                          {tx.to && (
                            <span className="text-xs text-[var(--text-secondary)] font-mono">
                              To: {displayAddress(tx.to)}
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="py-4 px-4">
                        <span className="text-sm text-[var(--text-primary)]">
                          {tx.amount || "-"}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        {tx.status === 'confirmed' ? (
                          <span className="inline-flex items-center gap-1 text-xs text-emerald-400">
                            <CheckCircle className="w-4 h-4" />
                            Confirmed
                          </span>
                        ) : tx.status === 'failed' ? (
                          <span className="inline-flex items-center gap-1 text-xs text-red-400">
                            <XCircle className="w-4 h-4" />
                            Failed
                          </span>
                        ) : (
                          <span className="inline-flex items-center gap-1 text-xs text-yellow-400">
                            <Clock className="w-4 h-4" />
                            Pending
                          </span>
                        )}
                      </td>
                      <td className="py-4 px-4">
                        <span className="text-sm text-[var(--text-secondary)]">
                          {displayTimestamp(tx.timestamp)}
                        </span>
                      </td>
                      <td className="py-4 px-4">
                        <button
                          onClick={() => copyToClipboard(tx.hash || "")}
                          className="text-[var(--text-secondary)] hover:text-[var(--accent-primary)] transition-colors"
                          title="Copy transaction hash"
                        >
                          <Copy className="w-4 h-4" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function WalletPage() {
  return (
    <WalletGuard>
      <WalletPageContent />
    </WalletGuard>
  );
}

