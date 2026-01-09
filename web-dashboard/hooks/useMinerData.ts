// React Query hooks for miner data

import { useQuery } from '@tanstack/react-query';
import { getUserInfo, getMinerStats, getEarningsHistory, getHashrateHistory } from '@/lib/api';

// User info hook
export function useUserInfo(walletAddress: string | null) {
  return useQuery({
    queryKey: ['userInfo', walletAddress],
    queryFn: () => getUserInfo(walletAddress!),
    enabled: !!walletAddress,
    refetchInterval: 30000, // Refetch every 30 seconds
    staleTime: 15000, // Consider data stale after 15 seconds
  });
}

// Miner stats hook
export function useMinerStats(walletAddress: string | null) {
  return useQuery({
    queryKey: ['minerStats', walletAddress],
    queryFn: () => getMinerStats(walletAddress!),
    enabled: !!walletAddress,
    refetchInterval: 10000, // Refetch every 10 seconds for real-time data
    staleTime: 5000, // Consider data stale after 5 seconds
  });
}

// Earnings history hook
export function useEarningsHistory(walletAddress: string | null) {
  return useQuery({
    queryKey: ['earningsHistory', walletAddress],
    queryFn: () => getEarningsHistory(walletAddress!),
    enabled: !!walletAddress,
    refetchInterval: 60000, // Refetch every minute
    staleTime: 30000, // Consider data stale after 30 seconds
  });
}

// Hashrate history hook
export function useHashrateHistory(walletAddress: string | null) {
  return useQuery({
    queryKey: ['hashrateHistory', walletAddress],
    queryFn: () => getHashrateHistory(walletAddress!),
    enabled: !!walletAddress,
    refetchInterval: 60000, // Refetch every minute
    staleTime: 30000, // Consider data stale after 30 seconds
  });
}