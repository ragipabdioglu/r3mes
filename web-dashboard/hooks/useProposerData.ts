/**
 * React Query hooks for proposer node data fetching
 * Replaces setInterval polling with proper React Query caching and refetching
 */

import { useQuery } from "@tanstack/react-query";
import { getProposerNodes, getAggregations, getGradientPool, ProposerNode, AggregationRecord, GradientPool } from "@/lib/api";

/**
 * Hook to fetch proposer nodes
 * Automatically refetches every 30 seconds
 */
export function useProposerNodes(limit: number = 100, offset: number = 0) {
  return useQuery<{ nodes: ProposerNode[]; total: number }, Error>({
    queryKey: ["proposerNodes", limit, offset],
    queryFn: () => getProposerNodes(limit, offset),
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
 * Hook to fetch aggregations
 * Automatically refetches every 30 seconds
 */
export function useAggregations(limit: number = 50, offset: number = 0) {
  return useQuery<{ aggregations: AggregationRecord[]; total: number }, Error>({
    queryKey: ["aggregations", limit, offset],
    queryFn: () => getAggregations(limit, offset),
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
 * Hook to fetch gradient pool
 * Automatically refetches every 30 seconds
 */
export function useGradientPool(limit: number = 100, offset: number = 0) {
  return useQuery<GradientPool, Error>({
    queryKey: ["gradientPool", limit, offset],
    queryFn: () => getGradientPool(limit, offset),
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

