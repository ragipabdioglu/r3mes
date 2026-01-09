/**
 * React Query hooks for serving node data fetching
 * Replaces setInterval polling with proper React Query caching and refetching
 */

import { useQuery } from "@tanstack/react-query";
import { getServingNodes, getServingNodeStats, ServingNode, ServingNodeStats } from "@/lib/api";

/**
 * Hook to fetch serving nodes
 * Automatically refetches every 30 seconds
 */
export function useServingNodes(limit: number = 100, offset: number = 0) {
  return useQuery<{ nodes: ServingNode[]; total: number }, Error>({
    queryKey: ["servingNodes", limit, offset],
    queryFn: () => getServingNodes(limit, offset),
    refetchInterval: (query) => {
      if (query.state.error?.message?.includes('not available')) {
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
 * Hook to fetch serving node stats for a specific node
 * Automatically refetches every 30 seconds
 */
export function useServingNodeStats(nodeAddress: string | null) {
  return useQuery<ServingNodeStats | null, Error>({
    queryKey: ["servingNodeStats", nodeAddress],
    queryFn: () => nodeAddress ? getServingNodeStats(nodeAddress) : Promise.resolve(null),
    enabled: !!nodeAddress,
    refetchInterval: (query) => {
      if (!nodeAddress || query.state.error?.message?.includes('not available')) {
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

