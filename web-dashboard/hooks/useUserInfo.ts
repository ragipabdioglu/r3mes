/**
 * React Query hook for user information
 */

import { useQuery } from "@tanstack/react-query";
import { getUserInfo, UserInfo } from "@/lib/api";

/**
 * Hook to fetch user information
 * Automatically refetches when wallet address changes
 */
export function useUserInfo(walletAddress: string | null, enabled: boolean = true) {
  return useQuery<UserInfo, Error>({
    queryKey: ["userInfo", walletAddress],
    queryFn: () => getUserInfo(walletAddress!),
    enabled: enabled && !!walletAddress,
    refetchInterval: 30000, // 30 seconds
    staleTime: 10000,
    retry: 3,
    retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
  });
}

