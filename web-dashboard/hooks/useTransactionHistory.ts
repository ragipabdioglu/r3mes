/**
 * React Query hook for transaction history
 */

import { useQuery } from "@tanstack/react-query";
import { getTransactionHistory, Transaction } from "@/lib/api";

/**
 * Hook to fetch transaction history
 * Automatically refetches every 30 seconds
 */
export function useTransactionHistory(
  walletAddress: string | null,
  limit: number = 50,
  enabled: boolean = true
) {
  return useQuery<{ transactions: Transaction[]; total: number }, Error>({
    queryKey: ["transactionHistory", walletAddress, limit],
    queryFn: () => getTransactionHistory(walletAddress!, limit),
    enabled: enabled && !!walletAddress,
    refetchInterval: 30000, // 30 seconds
    staleTime: 10000, // Consider data stale after 10 seconds
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
}

