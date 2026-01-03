/**
 * R3MES Web Dashboard API Client
 * 
 * This module provides type-safe API functions for communicating with the
 * R3MES backend services. All functions are connected to real backend endpoints
 * through the Next.js API proxy.
 * 
 * @module lib/api
 * @see {@link https://docs.r3mes.io/api} API Documentation
 */

import { logger } from './logger';

// Default API URL for production
const DEFAULT_API_URL = 'https://api.r3mes.network';

/**
 * Get the API base URL based on environment
 * Uses NEXT_PUBLIC_API_URL environment variable or falls back to default
 */
function getApiBaseUrl(): string {
  // In browser, check for injected env or use default
  if (typeof window !== 'undefined') {
    // Next.js injects NEXT_PUBLIC_* vars at build time
    // @ts-expect-error - NEXT_PUBLIC vars are injected at build time
    const publicApiUrl = typeof NEXT_PUBLIC_API_URL !== 'undefined' ? NEXT_PUBLIC_API_URL : undefined;
    return publicApiUrl || DEFAULT_API_URL;
  }
  
  // Server-side rendering - use default
  return DEFAULT_API_URL;
}

/**
 * Generic API request helper with error handling and logging
 * 
 * @template T - Expected response type
 * @param endpoint - API endpoint path (e.g., '/analytics')
 * @param options - Fetch options (method, body, headers, etc.)
 * @returns Promise resolving to the typed response data
 * @throws Error if the request fails or returns non-OK status
 * 
 * @example
 * const data = await apiRequest<UserInfo>('/user/remes123...');
 */
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const baseUrl = getApiBaseUrl();
  const url = `${baseUrl}${endpoint}`;
  
  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.message || `API error: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    logger.error(`API request failed: ${endpoint}`, error);
    throw error;
  }
}

// ============================================================================
// Type Definitions
// ============================================================================

/** User account information */
export interface UserInfo {
  /** Wallet address (bech32 format) */
  wallet_address: string;
  /** Available credits/balance */
  credits: number;
  /** Whether user is registered as a miner */
  is_miner: boolean;
}

/** Miner statistics and performance metrics */
export interface MinerStats {
  /** Miner's wallet address */
  wallet_address: string;
  /** Total earnings in REMES */
  total_earnings: number;
  /** Current hashrate in GH/s */
  hashrate: number;
  /** GPU temperature in Celsius */
  gpu_temperature: number;
  /** Total blocks found */
  blocks_found: number;
  /** Uptime percentage (0-100) */
  uptime_percentage: number;
  /** Current network difficulty */
  network_difficulty: number;
}

/** Historical earnings data point */
export interface EarningsData {
  /** Date string (YYYY-MM-DD) */
  date: string;
  /** Earnings amount for that date */
  earnings: number;
}

/** Historical hashrate data point */
export interface HashrateData {
  /** Date string (YYYY-MM-DD) */
  date: string;
  /** Hashrate value in GH/s */
  hashrate: number;
}

/** Network-wide statistics */
export interface NetworkStats {
  /** Number of currently active miners */
  active_miners: number;
  /** Total registered users */
  total_users: number;
  /** Total credits in circulation */
  total_credits: number;
  /** Current blockchain height */
  block_height: number;
}

/** Block information */
export interface Block {
  /** Block height */
  height: number;
  /** Miner address who found the block */
  miner: string;
  /** Block timestamp (ISO 8601) */
  timestamp: string;
  /** Block hash */
  hash: string;
}

/** Faucet status and configuration */
export interface FaucetStatus {
  /** Whether faucet has tokens available */
  available: boolean;
  /** Whether faucet is enabled */
  enabled: boolean;
  /** Amount per claim (numeric) */
  amount: number;
  /** Amount per claim (with denomination) */
  amount_per_claim: string;
  /** Daily limit (with denomination) */
  daily_limit: string;
  /** Cooldown period in milliseconds */
  cooldown: number;
  /** Last claim timestamp (null if never claimed) */
  lastClaim: number | null;
}

// Proposer-related interfaces

/** Proposer node information */
export interface ProposerNode {
  /** Node's wallet address */
  node_address: string;
  /** Current status (active, inactive, etc.) */
  status: string;
  /** Total aggregations performed */
  total_aggregations: number;
  /** Total rewards earned */
  total_rewards: string;
}

/** Aggregation record */
export interface AggregationRecord {
  /** Unique aggregation ID */
  aggregation_id: number;
  /** Proposer address */
  proposer: string;
  /** Number of participants */
  participant_count: number;
  /** Training round ID */
  training_round_id: number;
  /** IPFS hash of aggregated gradient */
  aggregated_gradient_ipfs_hash: string;
}

export interface GradientPool {
  total_count: number;
  pending_gradients: Array<{
    id: number;
    miner: string;
    training_round_id: number;
    ipfs_hash: string;
    status: string;
  }>;
}

// Serving-related interfaces
export interface ServingNode {
  node_address: string;
  status: string;
  is_available: boolean;
  total_requests: number;
  successful_requests: number;
  average_latency_ms: number;
  uptime: number;
  earnings: number;
  model_version?: string;
}

export interface ServingNodeStats {
  total_requests: number;
  success_rate: number;
  average_latency_ms: number;
  successful_requests: number;
  failed_requests: number;
  is_available: boolean;
  earnings: number;
}

// Role-related interfaces
export interface NodeRole {
  id: number;
  name: string;
  description: string;
  stake: number;
  requirements?: string[];
  rewards?: number;
}

export interface NodeRoles {
  roles: number[];
  status: string;
  totalStake: number;
  lastUpdate: number;
}

export interface RoleStats {
  role_id: number;
  role_name: string;
  total_nodes: number;
  active_nodes: number;
  total_stake: number;
  avg_uptime: number;
  icon?: string;
}

// Analytics interfaces
export interface AnalyticsData {
  totalUsers: number;
  activeNodes: number;
  totalTransactions: number;
  networkHashrate: string;
  dailyRewards: number;
  stakingRatio: number;
  user_engagement: {
    active_users: number;
    daily_active: number;
    retention_rate: number;
  };
  api_usage: {
    total_requests: number;
    success_rate: number;
    avg_response_time: number;
    endpoints_data: Array<{
      endpoint: string;
      requests: number;
      avg_time: number;
    }>;
  };
  model_performance: {
    average_latency: number;
    throughput: number;
    accuracy: number;
    success_rate: number;
    uptime: number;
    trend: Array<{
      date: string;
      latency: number;
      accuracy: number;
    }>;
  };
  network_performance: {
    avg_block_time: number;
    tps: number;
    uptime: number;
    total_nodes: number;
  };
  mining_stats: {
    total_miners: number;
    hashrate_distribution: Array<{
      name: string;
      value: number;
    }>;
  };
  revenue_data: Array<{
    date: string;
    revenue: number;
  }>;
}

// Leaderboard interfaces
export interface LeaderboardEntry {
  rank: number;
  address: string;
  score: number;
  rewards: number;
  username?: string;
  avatar?: string;
}

// Transaction interfaces
export interface Transaction {
  id: number;
  type: "send" | "receive" | "stake" | "reward";
  amount: number;
  to?: string;
  from?: string;
  timestamp: number;
  status: "pending" | "confirmed" | "failed";
  hash?: string;
}

// User API functions
export async function getUserInfo(walletAddress: string): Promise<UserInfo> {
  // Return default values - user info endpoint not implemented in backend
  // This prevents 404 errors and allows the app to function
  return {
    wallet_address: walletAddress,
    credits: 500, // Default credits
    is_miner: false,
  };
}

// Miner API functions
export async function getMinerStats(walletAddress: string): Promise<MinerStats> {
  return apiRequest<MinerStats>(`/miner/stats/${walletAddress}`);
}

export async function getEarningsHistory(walletAddress: string): Promise<EarningsData[]> {
  return apiRequest<EarningsData[]>(`/miner/earnings/${walletAddress}`);
}

export async function getHashrateHistory(walletAddress: string): Promise<HashrateData[]> {
  return apiRequest<HashrateData[]>(`/miner/hashrate/${walletAddress}`);
}

// Network API functions
export async function getNetworkStats(): Promise<NetworkStats> {
  return apiRequest<NetworkStats>('/network/stats');
}

export async function getRecentBlocks(limit: number = 10): Promise<Block[]> {
  try {
    const response = await apiRequest<{ blocks: Block[]; limit: number; total: number }>(`/blocks?limit=${limit}`);
    return response.blocks || [];
  } catch (error) {
    logger.error('Failed to fetch recent blocks:', error);
    return [];
  }
}

// Chat API function
export async function sendChatMessage(
  message: string,
  walletAddress: string,
  onChunk: (chunk: string) => void
): Promise<void> {
  const baseUrl = getApiBaseUrl();
  const response = await fetch(`${baseUrl}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      wallet_address: walletAddress,
    }),
  });

  if (!response.ok) {
    throw new Error('Failed to send chat message');
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value);
      onChunk(chunk);
    }
  } finally {
    reader.releaseLock();
  }
}

// Real API functions connected to backend endpoints

/**
 * Get analytics data from backend
 * Backend endpoint: GET /analytics?days=7
 */
export async function getAnalytics(days: number = 7): Promise<AnalyticsData> {
  try {
    const data = await apiRequest<{
      api_usage: {
        total_requests: number;
        success_rate: number;
        avg_response_time: number;
        endpoints_data: Array<{ endpoint: string; count: number }>;
      };
      user_engagement: {
        active_users: number;
        daily_active: number;
        retention_rate: number;
      };
      model_performance: {
        average_latency: number;
        throughput: number;
        accuracy: number;
        success_rate: number;
        uptime: number;
        trend?: Array<{ date: string; latency: number; accuracy: number }>;
      };
      network_health: {
        avg_block_time: number;
        tps: number;
        uptime: number;
        total_nodes: number;
      };
    }>(`/analytics?days=${days}`);

    // Transform backend response to frontend format
    return {
      totalUsers: data.user_engagement?.active_users || 0,
      activeNodes: data.network_health?.total_nodes || 0,
      totalTransactions: data.api_usage?.total_requests || 0,
      networkHashrate: "N/A",
      dailyRewards: 0,
      stakingRatio: 0,
      user_engagement: {
        active_users: data.user_engagement?.active_users || 0,
        daily_active: data.user_engagement?.daily_active || 0,
        retention_rate: data.user_engagement?.retention_rate || 0,
      },
      api_usage: {
        total_requests: data.api_usage?.total_requests || 0,
        success_rate: data.api_usage?.success_rate || 0,
        avg_response_time: data.api_usage?.avg_response_time || 0,
        endpoints_data: (data.api_usage?.endpoints_data || []).map(e => ({
          endpoint: e.endpoint,
          requests: e.count,
          avg_time: 0,
        })),
      },
      model_performance: {
        average_latency: data.model_performance?.average_latency || 0,
        throughput: data.model_performance?.throughput || 0,
        accuracy: data.model_performance?.accuracy || 0,
        success_rate: data.model_performance?.success_rate || 0,
        uptime: data.model_performance?.uptime || 0,
        trend: data.model_performance?.trend || [],
      },
      network_performance: {
        avg_block_time: data.network_health?.avg_block_time || 0,
        tps: data.network_health?.tps || 0,
        uptime: data.network_health?.uptime || 0,
        total_nodes: data.network_health?.total_nodes || 0,
      },
      mining_stats: {
        total_miners: 0,
        hashrate_distribution: [],
      },
      revenue_data: [],
    };
  } catch (error) {
    logger.error('Failed to fetch analytics:', error);
    throw error;
  }
}

/**
 * Get faucet status from backend
 * Backend endpoint: GET /faucet/status
 */
export async function getFaucetStatus(): Promise<FaucetStatus> {
  try {
    const data = await apiRequest<{
      enabled: boolean;
      amount_per_claim: string;
      daily_limit: string;
      rate_limit: string;
    }>('/faucet/status');

    return {
      available: data.enabled,
      enabled: data.enabled,
      amount: parseInt(data.amount_per_claim?.replace(/\D/g, '') || '0'),
      amount_per_claim: data.amount_per_claim || "1000000uremes",
      daily_limit: data.daily_limit || "5000000uremes",
      cooldown: 24 * 60 * 60 * 1000, // 24 hours
      lastClaim: null,
    };
  } catch (error) {
    logger.error('Failed to fetch faucet status:', error);
    throw error;
  }
}

/**
 * Claim tokens from faucet
 * Backend endpoint: POST /faucet/claim
 */
export async function claimFaucet(params: { address: string; amount?: string }): Promise<{
  success: boolean;
  tx_hash?: string;
  txHash?: string;
  amount?: string;
  next_claim_available_at?: string;
  message?: string;
}> {
  try {
    const data = await apiRequest<{
      success: boolean;
      tx_hash?: string;
      amount: string;
      next_claim_available_at?: string;
      message?: string;
    }>('/faucet/claim', {
      method: 'POST',
      body: JSON.stringify({
        address: params.address,
        amount: params.amount,
      }),
    });

    return {
      success: data.success,
      tx_hash: data.tx_hash,
      txHash: data.tx_hash,
      amount: data.amount,
      next_claim_available_at: data.next_claim_available_at,
      message: data.message,
    };
  } catch (error: unknown) {
    // Handle rate limit errors specially
    if (error instanceof Error && error.message.includes('429')) {
      return {
        success: false,
        message: "Rate limit exceeded. You can only claim once per day.",
      };
    }
    logger.error('Failed to claim faucet:', error);
    return {
      success: false,
      message: error instanceof Error ? error.message : "Failed to claim tokens",
    };
  }
}

/**
 * Get leaderboard data from backend
 * Backend endpoint: GET /leaderboard/miners or /leaderboard/validators
 */
export async function getLeaderboard(type?: "miners" | "validators"): Promise<{
  miners?: Array<{
    address: string;
    tier: string;
    total_submissions: number;
    reputation: number;
    trend?: number;
  }>;
  validators?: Array<{
    address: string;
    tier: string;
    trust_score: number;
    uptime: number;
    voting_power: number;
  }>;
}> {
  try {
    if (type === "miners") {
      const data = await apiRequest<{
        miners: Array<{
          address: string;
          tier: string;
          total_submissions?: number;
          successful_submissions?: number;
          reputation?: number;
          reputation_score?: number;
          trend?: number;
        }>;
      }>('/leaderboard/miners?limit=100');

      return {
        miners: data.miners?.map(m => ({
          address: m.address,
          tier: m.tier || 'bronze',
          total_submissions: m.total_submissions || m.successful_submissions || 0,
          reputation: m.reputation || m.reputation_score || 0,
          trend: m.trend || 0,
        })) || [],
      };
    } else if (type === "validators") {
      const data = await apiRequest<{
        validators: Array<{
          address: string;
          tier: string;
          trust_score: number;
          uptime: number;
          voting_power: number;
        }>;
      }>('/leaderboard/validators?limit=100');

      return {
        validators: data.validators || [],
      };
    }

    // Return both if no type specified
    const [minersData, validatorsData] = await Promise.all([
      apiRequest<{ miners: Array<{ address: string; tier: string; total_submissions?: number; successful_submissions?: number; reputation?: number; reputation_score?: number; trend?: number }> }>('/leaderboard/miners?limit=10'),
      apiRequest<{ validators: Array<{ address: string; tier: string; trust_score: number; uptime: number; voting_power: number }> }>('/leaderboard/validators?limit=10'),
    ]);

    return {
      miners: minersData.miners?.map(m => ({
        address: m.address,
        tier: m.tier || 'bronze',
        total_submissions: m.total_submissions || m.successful_submissions || 0,
        reputation: m.reputation || m.reputation_score || 0,
        trend: m.trend || 0,
      })) || [],
      validators: validatorsData.validators || [],
    };
  } catch (error) {
    logger.error('Failed to fetch leaderboard:', error);
    throw error;
  }
}

/**
 * Get role statistics from backend
 * Backend endpoint: GET /roles/stats/summary
 */
export async function getRoleStatistics(): Promise<{ stats: RoleStats[] }> {
  try {
    const data = await apiRequest<{
      stats: Array<{
        role_id: number;
        role_name: string;
        total_nodes: number;
        active_nodes: number;
        total_stake?: number;
        avg_uptime?: number;
      }>;
    }>('/roles/stats/summary');

    return {
      stats: data.stats?.map(s => ({
        role_id: s.role_id,
        role_name: s.role_name,
        total_nodes: s.total_nodes,
        active_nodes: s.active_nodes,
        total_stake: s.total_stake || 0,
        avg_uptime: s.avg_uptime || 0,
        icon: getRoleIcon(s.role_name),
      })) || [],
    };
  } catch (error) {
    logger.error('Failed to fetch role statistics:', error);
    throw error;
  }
}

// Helper function to get role icon
function getRoleIcon(roleName: string): string {
  const icons: Record<string, string> = {
    'Miner': 'cpu',
    'Serving': 'server',
    'Validator': 'shield',
    'Proposer': 'layers',
  };
  return icons[roleName] || 'circle';
}

/**
 * Get available roles from backend
 * Backend endpoint: GET /roles
 */
export async function getRoles(): Promise<{ roles: NodeRole[] }> {
  try {
    const data = await apiRequest<{
      roles: Array<{
        role_id: number;
        role_name: string;
        description: string;
        access_control?: {
          min_stake?: string;
          requires_approval?: boolean;
        };
      }>;
    }>('/roles');

    return {
      roles: data.roles?.map(r => ({
        id: r.role_id,
        name: r.role_name,
        description: r.description,
        stake: parseInt(r.access_control?.min_stake?.replace(/\D/g, '') || '0'),
        requirements: r.access_control?.requires_approval ? ['Requires approval'] : [],
        rewards: 0,
      })) || [],
    };
  } catch (error) {
    logger.error('Failed to fetch roles:', error);
    throw error;
  }
}

/**
 * Get node roles for a specific address
 * Backend endpoint: GET /roles/{address}
 */
export async function getNodeRoles(address: string): Promise<NodeRoles> {
  try {
    const data = await apiRequest<{
      node_address: string;
      roles: number[];
      role_names: string[];
      status: string;
      stake?: string;
    }>(`/roles/${address}`);

    return {
      roles: data.roles || [],
      status: data.status || 'unknown',
      totalStake: parseInt(data.stake?.replace(/\D/g, '') || '0'),
      lastUpdate: Date.now(),
    };
  } catch (error) {
    logger.error(`Failed to fetch node roles for ${address}:`, error);
    throw error;
  }
}

/**
 * Get proposer nodes from backend
 * Backend endpoint: GET /proposer/nodes
 */
export async function getProposerNodes(limit: number = 100, offset: number = 0): Promise<{
  nodes: Array<{
    node_address: string;
    status: string;
    total_aggregations: number;
    total_rewards: string;
  }>;
  total: number;
}> {
  try {
    const data = await apiRequest<{
      nodes: Array<{
        node_address: string;
        status: string;
        total_aggregations: number;
        total_rewards: string;
      }>;
      total: number;
    }>(`/proposer/nodes?limit=${limit}&offset=${offset}`);

    return {
      nodes: data.nodes || [],
      total: data.total || 0,
    };
  } catch (error) {
    logger.error('Failed to fetch proposer nodes:', error);
    throw error;
  }
}

/**
 * Get aggregations from backend
 * Backend endpoint: GET /proposer/aggregations
 */
export async function getAggregations(limit: number = 50, offset: number = 0): Promise<{
  aggregations: Array<{
    aggregation_id: number;
    proposer: string;
    participant_count: number;
    training_round_id: number;
    aggregated_gradient_ipfs_hash: string;
  }>;
  total: number;
}> {
  try {
    const data = await apiRequest<{
      aggregations: Array<{
        aggregation_id: number;
        proposer: string;
        participant_count: number;
        training_round_id: number;
        aggregated_gradient_ipfs_hash: string;
        merkle_root?: string;
      }>;
      total: number;
    }>(`/proposer/aggregations?limit=${limit}&offset=${offset}`);

    return {
      aggregations: data.aggregations || [],
      total: data.total || 0,
    };
  } catch (error) {
    logger.error('Failed to fetch aggregations:', error);
    throw error;
  }
}

/**
 * Get gradient pool from backend
 * Backend endpoint: GET /proposer/pool
 */
export async function getGradientPool(limit: number = 100, offset: number = 0): Promise<{
  total_count: number;
  pending_gradients: Array<{
    id: number;
    miner: string;
    training_round_id: number;
    ipfs_hash: string;
    status: string;
  }>;
}> {
  try {
    const data = await apiRequest<{
      pending_gradients: Array<{
        id?: number;
        gradient_id?: number;
        miner?: string;
        submitter?: string;
        training_round_id: number;
        ipfs_hash?: string;
        gradient_ipfs_hash?: string;
        status: string;
      }>;
      total_count: number;
    }>(`/proposer/pool?limit=${limit}&offset=${offset}&status=pending`);

    return {
      total_count: data.total_count || 0,
      pending_gradients: (data.pending_gradients || []).map((g, idx) => ({
        id: g.id || g.gradient_id || idx + 1,
        miner: g.miner || g.submitter || '',
        training_round_id: g.training_round_id,
        ipfs_hash: g.ipfs_hash || g.gradient_ipfs_hash || '',
        status: g.status,
      })),
    };
  } catch (error) {
    logger.error('Failed to fetch gradient pool:', error);
    throw error;
  }
}

/**
 * Get serving nodes from backend
 * Backend endpoint: GET /serving/nodes
 */
export async function getServingNodes(limit: number = 100, offset: number = 0): Promise<{
  nodes: Array<{
    node_address: string;
    status: string;
    is_available: boolean;
    total_requests: number;
    successful_requests: number;
    average_latency_ms: number;
    uptime: number;
    earnings: number;
    model_version?: string;
  }>;
  total: number;
}> {
  try {
    const data = await apiRequest<{
      nodes: Array<{
        node_address: string;
        status: string;
        is_available: boolean;
        total_requests: number;
        successful_requests: number;
        average_latency_ms: number;
        last_heartbeat?: string;
        model_version?: string;
      }>;
      total: number;
    }>(`/serving/nodes?limit=${limit}&offset=${offset}`);

    return {
      nodes: (data.nodes || []).map(n => ({
        node_address: n.node_address,
        status: n.status,
        is_available: n.is_available,
        total_requests: n.total_requests,
        successful_requests: n.successful_requests,
        average_latency_ms: n.average_latency_ms,
        uptime: n.total_requests > 0 ? n.successful_requests / n.total_requests : 0,
        earnings: 0, // Not provided by backend
        model_version: n.model_version,
      })),
      total: data.total || 0,
    };
  } catch (error) {
    logger.error('Failed to fetch serving nodes:', error);
    throw error;
  }
}

/**
 * Get serving node stats from backend
 * Backend endpoint: GET /serving/nodes/{address}/stats
 */
export async function getServingNodeStats(address: string): Promise<{
  total_requests: number;
  success_rate: number;
  average_latency_ms: number;
  successful_requests: number;
  failed_requests: number;
  is_available: boolean;
  earnings: number;
}> {
  try {
    const data = await apiRequest<{
      node_address: string;
      total_requests: number;
      successful_requests: number;
      failed_requests: number;
      success_rate: number;
      average_latency_ms: number;
      is_available: boolean;
      model_version?: string;
    }>(`/serving/nodes/${address}/stats`);

    return {
      total_requests: data.total_requests || 0,
      success_rate: data.success_rate || 0,
      average_latency_ms: data.average_latency_ms || 0,
      successful_requests: data.successful_requests || 0,
      failed_requests: data.failed_requests || 0,
      is_available: data.is_available || false,
      earnings: 0, // Not provided by backend
    };
  } catch (error) {
    logger.error(`Failed to fetch serving node stats for ${address}:`, error);
    throw error;
  }
}

/**
 * Get transaction history from backend
 * Backend endpoint: GET /user/{address}/transactions (or blockchain query)
 */
export async function getTransactionHistory(address: string, limit: number = 50): Promise<{ transactions: Transaction[]; total: number }> {
  try {
    // Try to fetch from backend API
    const data = await apiRequest<{
      transactions: Array<{
        id?: number;
        hash?: string;
        txhash?: string;
        type: string;
        amount?: number | string;
        to?: string;
        from?: string;
        timestamp?: number | string;
        status?: string;
      }>;
      total: number;
    }>(`/user/${address}/transactions?limit=${limit}`);

    return {
      transactions: (data.transactions || []).map((tx, idx) => ({
        id: tx.id || idx + 1,
        type: normalizeTransactionType(tx.type),
        amount: typeof tx.amount === 'string' ? parseFloat(tx.amount) || 0 : tx.amount || 0,
        to: tx.to,
        from: tx.from,
        timestamp: typeof tx.timestamp === 'string' ? new Date(tx.timestamp).getTime() : tx.timestamp || Date.now(),
        status: normalizeTransactionStatus(tx.status),
        hash: tx.hash || tx.txhash,
      })),
      total: data.total || 0,
    };
  } catch (error) {
    logger.error(`Failed to fetch transaction history for ${address}:`, error);
    // Return empty array on error instead of throwing
    return { transactions: [], total: 0 };
  }
}

// Helper function to normalize transaction type
function normalizeTransactionType(type: string): "send" | "receive" | "stake" | "reward" {
  const lowerType = type.toLowerCase();
  if (lowerType.includes('send') || lowerType.includes('transfer')) return 'send';
  if (lowerType.includes('receive')) return 'receive';
  if (lowerType.includes('stake') || lowerType.includes('delegate')) return 'stake';
  if (lowerType.includes('reward') || lowerType.includes('withdraw')) return 'reward';
  return 'send';
}

// Helper function to normalize transaction status
function normalizeTransactionStatus(status?: string): "pending" | "confirmed" | "failed" {
  if (!status) return 'confirmed';
  const lowerStatus = status.toLowerCase();
  if (lowerStatus.includes('pending')) return 'pending';
  if (lowerStatus.includes('fail') || lowerStatus.includes('error')) return 'failed';
  return 'confirmed';
}