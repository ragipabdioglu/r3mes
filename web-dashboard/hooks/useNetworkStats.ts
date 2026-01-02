// React Query hooks for network statistics

import { useQuery } from '@tanstack/react-query';
import { getNetworkStats, getRecentBlocks } from '@/lib/api';

// Network stats hook
export function useNetworkStats() {
  return useQuery({
    queryKey: ['networkStats'],
    queryFn: getNetworkStats,
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 15000, // Consider data stale after 15 seconds
  });
}

// Recent blocks hook
export function useRecentBlocks(limit: number = 10) {
  return useQuery({
    queryKey: ['recentBlocks', limit],
    queryFn: () => getRecentBlocks(limit),
    refetchInterval: 15000, // Refetch every 15 seconds for real-time blocks
    staleTime: 5000, // Consider data stale after 5 seconds
  });
}