/**
 * React Query hooks for network data fetching
 * Replaces setInterval polling with proper React Query caching and refetching
 */

import { useQuery } from "@tanstack/react-query";
import { getNetworkStats, getRecentBlocks, NetworkStats, Block } from "@/lib/api";

/**
 * Hook to fetch network statistics
 * Automatically refetches every 30 seconds
 */
export function useNetworkStats() {
  return useQuery<NetworkStats, Error>({
    queryKey: ["networkStats"],
    queryFn: getNetworkStats,
    refetchInterval: (query) => {
      // If there's an error (like backend not available), don't refetch aggressively
      if (query.state.error?.message?.includes('not available')) {
        return false; // Stop refetching if backend is not available
      }
      return 30000; // 30 seconds
    },
    staleTime: 10000, // Consider data stale after 10 seconds
    retry: (failureCount, error) => {
      // Don't retry if backend is not available
      if (error?.message?.includes('not available')) {
        return false;
      }
      return failureCount < 2; // Only retry 2 times
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
    refetchOnWindowFocus: false, // Don't refetch on window focus if backend is down
    refetchOnMount: false, // Don't refetch on mount if we already have an error
  });
}

/**
 * Hook to fetch recent blocks
 * Automatically refetches every 15 seconds
 */
export function useRecentBlocks(limit: number = 10) {
  return useQuery<Block[], Error>({
    queryKey: ["recentBlocks", limit],
    queryFn: () => getRecentBlocks(limit),
    refetchInterval: (query) => {
      // If there's an error (like backend not available), don't refetch aggressively
      if (query.state.error?.message?.includes('not available')) {
        return false; // Stop refetching if backend is not available
      }
      return 15000; // 15 seconds
    },
    staleTime: 5000, // Consider data stale after 5 seconds
    retry: (failureCount, error) => {
      // Don't retry if backend is not available
      if (error?.message?.includes('not available')) {
        return false;
      }
      return failureCount < 2; // Only retry 2 times
    },
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 10000),
    refetchOnWindowFocus: false,
    refetchOnMount: false,
  });
}

