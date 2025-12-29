/**
 * React Query hooks for miner data fetching
 * Replaces setInterval polling with proper React Query caching and refetching
 */

import { useQuery } from "@tanstack/react-query";
import { getUserInfo, getMinerStats, getEarningsHistory, getHashrateHistory, UserInfo, MinerStats } from "@/lib/api";

/**
 * Hook to fetch user info for a wallet address
 * Automatically refetches every 5 seconds
 */
export function useUserInfo(walletAddress: string | null) {
  return useQuery<UserInfo | null, Error>({
    queryKey: ["userInfo", walletAddress],
    queryFn: () => walletAddress ? getUserInfo(walletAddress) : Promise.resolve(null),
    enabled: !!walletAddress,
    refetchInterval: (query) => {
      if (!walletAddress || query.state.error?.message?.includes('not available')) {
        return false;
      }
      return 5000; // 5 seconds
    },
    staleTime: 2000, // Consider data stale after 2 seconds
    retry: (failureCount, error) => {
      if (error?.message?.includes('not available')) {
        return false;
      }
      return failureCount < 2;
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}

/**
 * Hook to fetch miner stats for a wallet address
 * Automatically refetches every 10 seconds
 */
export function useMinerStats(walletAddress: string | null) {
  return useQuery<MinerStats | null, Error>({
    queryKey: ["minerStats", walletAddress],
    queryFn: () => walletAddress ? getMinerStats(walletAddress) : Promise.resolve(null),
    enabled: !!walletAddress,
    refetchInterval: (query) => {
      if (!walletAddress || query.state.error?.message?.includes('not available')) {
        return false;
      }
      return 10000; // 10 seconds
    },
    staleTime: 5000, // Consider data stale after 5 seconds
    retry: (failureCount, error) => {
      if (error?.message?.includes('not available')) {
        return false;
      }
      return failureCount < 2;
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}

/**
 * Hook to fetch earnings history for a wallet address
 * Refetches every 30 seconds
 */
export function useEarningsHistory(walletAddress: string | null) {
  return useQuery<Array<{ date: string; earnings: number }>, Error>({
    queryKey: ["earningsHistory", walletAddress],
    queryFn: () => walletAddress ? getEarningsHistory(walletAddress) : Promise.resolve([]),
    enabled: !!walletAddress,
    refetchInterval: (query) => {
      if (!walletAddress || query.state.error?.message?.includes('not available')) {
        return false;
      }
      return 30000; // 30 seconds
    },
    staleTime: 15000, // Consider data stale after 15 seconds
    retry: (failureCount, error) => {
      if (error?.message?.includes('not available')) {
        return false;
      }
      return failureCount < 2;
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}

/**
 * Hook to fetch hashrate history for a wallet address
 * Refetches every 30 seconds
 */
export function useHashrateHistory(walletAddress: string | null) {
  return useQuery<Array<{ date: string; hashrate: number }>, Error>({
    queryKey: ["hashrateHistory", walletAddress],
    queryFn: () => walletAddress ? getHashrateHistory(walletAddress) : Promise.resolve([]),
    enabled: !!walletAddress,
    refetchInterval: (query) => {
      if (!walletAddress || query.state.error?.message?.includes('not available')) {
        return false;
      }
      return 30000; // 30 seconds
    },
    staleTime: 15000, // Consider data stale after 15 seconds
    retry: (failureCount, error) => {
      if (error?.message?.includes('not available')) {
        return false;
      }
      return failureCount < 2;
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}
