/**
 * Backend API Client
 * 
 * Handles all API requests to the R3MES Backend Inference Service
 */

import axios from 'axios';

// Get API base URL from environment variable
// In production, NEXT_PUBLIC_BACKEND_URL must be set
// In development, fallback to localhost only if explicitly allowed
const getApiBaseUrl = (): string => {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (backendUrl) {
    return backendUrl;
  }
  
  // Only allow localhost fallback in development
  if (process.env.NODE_ENV === 'development') {
    return 'http://localhost:8000';
  }
  
  // Production: fail if not configured
  throw new Error('NEXT_PUBLIC_BACKEND_URL environment variable must be set in production');
};

const API_BASE_URL = getApiBaseUrl();

// Create axios instance with error handling
const apiClient = axios.create({
  timeout: 10000, // 10 second timeout
});

// Track connection errors to avoid spam
let connectionErrorLogged = false;

// Response interceptor to handle connection errors gracefully
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle ECONNREFUSED and network errors silently in development
    if (
      error.code === 'ECONNREFUSED' ||
      error.code === 'ENOTFOUND' ||
      error.message?.includes('ECONNREFUSED') ||
      error.message?.includes('Network Error')
    ) {
      // Only log once per session to avoid spam
      if (!connectionErrorLogged && typeof window !== 'undefined') {
        connectionErrorLogged = true;
        // Use logger instead of console.warn (logger handles production/development automatically)
        if (process.env.NODE_ENV === 'development') {
          console.warn(
            'Backend API is not available. Some features may not work. ' +
            'To start the backend, run: cd backend && python -m uvicorn app.main:app --reload'
          );
        }
      }
      
      // Return a rejected promise with a user-friendly error
      return Promise.reject(
        new Error('Backend service is not available. Please ensure the backend is running.')
      );
    }
    
    // For other errors, pass them through
    return Promise.reject(error);
  }
);

// Use the configured axios instance for all requests
// Override default axios to use our configured instance
const originalGet = axios.get;
const originalPost = axios.post;
const originalPut = axios.put;
const originalDelete = axios.delete;

// Create wrapper functions that use our configured instance
const apiGet = <T = any>(url: string, config?: any) => {
  if (url.startsWith('/api/')) {
    return apiClient.get<T>(url, config);
  }
  return originalGet<T>(url, config);
};

const apiPost = <T = any>(url: string, data?: any, config?: any) => {
  if (url.startsWith('/api/')) {
    return apiClient.post<T>(url, data, config);
  }
  return originalPost<T>(url, data, config);
};

const apiPut = <T = any>(url: string, data?: any, config?: any) => {
  if (url.startsWith('/api/')) {
    return apiClient.put<T>(url, data, config);
  }
  return originalPut<T>(url, data, config);
};

const apiDelete = <T = any>(url: string, config?: any) => {
  if (url.startsWith('/api/')) {
    return apiClient.delete<T>(url, config);
  }
  return originalDelete<T>(url, config);
};

// Types
export interface ChatRequest {
  message: string;
  wallet_address: string;
}

export interface UserInfo {
  wallet_address: string;
  credits: number;
  is_miner: boolean;
}

export interface NetworkStats {
  active_miners: number;
  total_users: number;
  total_credits: number;
  block_height?: number;
}

export interface Block {
  height: number;
  miner?: string;
  timestamp?: string;
  hash?: string;
}

export interface MinerStats {
  wallet_address: string;
  total_earnings: number;
  hashrate: number;
  gpu_temperature: number;
  blocks_found: number;
  uptime_percentage: number;
  network_difficulty: number;
}

export interface EarningsDataPoint {
  date: string;
  earnings: number;
}

export interface HashrateDataPoint {
  date: string;
  hashrate: number;
}

export interface Transaction {
  hash: string;
  height: number;
  type: string;
  from?: string;
  to?: string;
  amount?: string;
  fee?: string;
  timestamp: string;
  status: 'success' | 'failed' | 'pending';
  memo?: string;
}

export interface TransactionHistory {
  transactions: Transaction[];
  total: number;
}

export interface BlockDetail {
  height: number;
  hash: string;
  proposer: string;
  timestamp: string;
  transactions: Transaction[];
  tx_count: number;
  gas_used?: number;
  gas_wanted?: number;
}

/**
 * Send a chat message and get streaming response
 */
export async function sendChatMessage(
  message: string,
  walletAddress: string,
  onChunk: (chunk: string) => void
): Promise<void> {
  const response = await fetch('/api/chat', {
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
    if (response.status === 402) {
      throw new Error('Insufficient credits. Please mine blocks to earn credits.');
    }
    throw new Error(`Chat request failed: ${response.statusText}`);
  }

  // Stream the response
  const reader = response.body?.getReader();
  const decoder = new TextDecoder();

  if (!reader) {
    throw new Error('Response body is not readable');
  }

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    onChunk(chunk);
  }
}

/**
 * Get user information
 */
export async function getUserInfo(walletAddress: string): Promise<UserInfo> {
  const response = await apiGet<UserInfo>(`/api/user/info/${walletAddress}`);
  return response.data;
}

/**
 * Get network statistics
 * @throws {Error} If the request fails
 */
export async function getNetworkStats(): Promise<NetworkStats> {
  const response = await apiGet<NetworkStats>('/api/network/stats');
  return response.data;
}

/**
 * Get recent blocks
 * @throws {Error} If the request fails
 */
export async function getRecentBlocks(limit: number = 10): Promise<Block[]> {
  const response = await apiGet<{ blocks: Block[]; limit: number; total: number }>(`/api/blocks?limit=${limit}`);
  if (!response.data.blocks) {
    throw new Error('Invalid response format from server');
  }
  return response.data.blocks;
}

/**
 * Get miner statistics
 * @throws {Error} If the request fails
 */
export async function getMinerStats(walletAddress: string): Promise<MinerStats> {
  const response = await apiGet<MinerStats>(`/api/miner/stats/${walletAddress}`);
  return response.data;
}

/**
 * Get earnings history (7 days)
 * @throws {Error} If the request fails
 */
export async function getEarningsHistory(walletAddress: string): Promise<EarningsDataPoint[]> {
  const response = await apiGet<{ earnings: EarningsDataPoint[] }>(`/api/miner/earnings/${walletAddress}?days=7`);
  if (!response.data.earnings) {
    throw new Error('Invalid response format from server');
  }
  return response.data.earnings;
}

/**
 * Get hashrate history (7 days)
 * @throws {Error} If the request fails
 */
export async function getHashrateHistory(walletAddress: string): Promise<HashrateDataPoint[]> {
  const response = await apiGet<{ hashrate: HashrateDataPoint[] }>(`/api/miner/hashrate/${walletAddress}?days=7`);
  if (!response.data.hashrate) {
    throw new Error('Invalid response format from server');
  }
  return response.data.hashrate;
}

/**
 * Get transaction history for a wallet address
 */
/**
 * Get transaction history
 * @throws {Error} If the request fails
 */
export async function getTransactionHistory(walletAddress: string, limit: number = 50): Promise<TransactionHistory> {
  const response = await apiGet<TransactionHistory>(
    `/api/blockchain/cosmos/tx/v1beta1/txs?events=transfer.recipient='${walletAddress}'&events=transfer.sender='${walletAddress}'&pagination.limit=${limit}`
  );
  
  if (!response.data) {
    throw new Error('Invalid response format from server');
  }
  
  return response.data;
}

/**
 * Get transaction details by hash
 * @throws {Error} If the request fails
 */
export async function getTransactionByHash(txHash: string): Promise<Transaction> {
  const response = await apiGet(`/api/blockchain/cosmos/tx/v1beta1/txs/${txHash}`);
  
  if (!response.data?.tx_response) {
    throw new Error('Invalid transaction response format');
  }
  
  const tx = response.data.tx_response;
  
  return {
    hash: tx.txhash,
    height: parseInt(tx.height),
    type: tx.tx.body.messages[0]?.type || 'unknown',
    timestamp: tx.timestamp,
    status: tx.code === 0 ? 'success' : 'failed',
    fee: tx.tx.auth_info.fee?.amount?.[0]?.amount || '0',
    memo: tx.tx.body.memo || '',
  };
}

/**
 * Get block details by height
 * @throws {Error} If the request fails
 */
export async function getBlockByHeight(height: number): Promise<BlockDetail> {
  const response = await apiGet(`${API_BASE_URL}/blocks/${height}`);
  return response.data;
}

export interface LeaderboardEntry {
  address: string;
  tier: string;
  total_submissions?: number;
  reputation?: number;
  trust_score?: number;
  uptime?: number;
  voting_power?: number;
  trend?: number;
}

export interface LeaderboardData {
  miners?: LeaderboardEntry[];
  validators?: LeaderboardEntry[];
}

/**
 * Get leaderboard data
 */
export async function getLeaderboard(type: "miners" | "validators"): Promise<LeaderboardData> {
  const response = await apiGet(`${API_BASE_URL}/leaderboard/${type}`);
  return response.data;
}

export interface AnalyticsData {
  api_usage?: {
    total_requests: number;
    endpoints: Record<string, number>;
    endpoints_data?: Array<{ endpoint: string; count: number }>;
  };
  user_engagement?: {
    active_users: number;
    total_actions: number;
    average_actions_per_user: number;
  };
  model_performance?: {
    average_latency: number;
    average_tokens_per_second: number;
    success_rate: number;
    trend?: Array<{ date: string; latency: number }>;
  };
  network_health?: {
    active_miners_trend: Array<{ date: string; count: number }>;
    total_flops_trend: Array<{ date: string; flops: number }>;
  };
}

/**
 * Get analytics data
 */
export async function getAnalytics(): Promise<AnalyticsData> {
  const response = await apiGet(`${API_BASE_URL}/analytics`);
  return response.data;
}

export interface FaucetRequest {
  address: string;
  amount?: string;
}

export interface FaucetResponse {
  success: boolean;
  message: string;
  tx_hash?: string;
  amount: string;
  next_claim_available_at?: string;
}

export interface FaucetStatus {
  enabled: boolean;
  amount_per_claim: string;
  daily_limit: string;
  rate_limit: string;
}

/**
 * Claim tokens from the faucet
 */
export async function claimFaucet(request: FaucetRequest): Promise<FaucetResponse> {
  const response = await apiPost<FaucetResponse>(`${API_BASE_URL}/faucet/claim`, request);
  return response.data;
}

/**
 * Get faucet status and configuration
 */
export async function getFaucetStatus(): Promise<FaucetStatus> {
  const response = await apiGet<FaucetStatus>("/api/faucet/status");
  return response.data;
}

// Serving Node API Functions
export interface ServingNode {
  node_address: string;
  model_version: string;
  model_ipfs_hash: string;
  is_available: boolean;
  total_requests: number;
  successful_requests: number;
  average_latency_ms: number;
  last_heartbeat: string;
  status: string;
}

export interface ServingNodeDetail extends ServingNode {
  failed_requests: number;
  success_rate: number;
  resources?: any;
  stake?: string;
}

export interface InferenceRequest {
  request_id: string;
  requester: string;
  serving_node: string;
  model_version: string;
  input_data_ipfs_hash: string;
  fee: string;
  status: string;
  request_time?: string;
  result_ipfs_hash?: string;
  latency_ms?: number;
}

export interface ServingNodeStats {
  node_address: string;
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  success_rate: number;
  average_latency_ms: number;
  model_version: string;
  is_available: boolean;
}

export async function getServingNodes(limit: number = 100, offset: number = 0): Promise<{ nodes: ServingNode[]; total: number }> {
  const response = await apiGet<{ nodes: ServingNode[]; total: number }>(`/api/serving/nodes?limit=${limit}&offset=${offset}`);
  return response.data;
}

export async function getServingNode(address: string): Promise<ServingNodeDetail> {
  const response = await apiGet<ServingNodeDetail>(`/api/serving/nodes/${address}`);
  return response.data;
}

export async function getServingNodeStats(address: string): Promise<ServingNodeStats> {
  const response = await apiGet<ServingNodeStats>(`/api/serving/nodes/${address}/stats`);
  return response.data;
}

export async function getInferenceRequest(requestId: string): Promise<InferenceRequest> {
  const response = await apiGet<InferenceRequest>(`/api/serving/requests/${requestId}`);
  return response.data;
}

// Proposer API Functions
export interface ProposerNode {
  node_address: string;
  status: string;
  total_aggregations: number;
  total_rewards: string;
  last_aggregation_height?: number;
  resources?: any;
  stake?: string;
}

export interface AggregationRecord {
  aggregation_id: number;
  proposer: string;
  aggregated_gradient_ipfs_hash: string;
  merkle_root: string;
  participant_count: number;
  training_round_id: number;
  block_height?: number;
  timestamp?: string;
}

export interface GradientPool {
  pending_gradients: Array<{
    id: number;
    status: string;
    ipfs_hash: string;
    miner: string;
    training_round_id: number;
  }>;
  total_count: number;
}

export async function getProposerNodes(limit: number = 100, offset: number = 0): Promise<{ nodes: ProposerNode[]; total: number }> {
  const response = await apiGet<{ nodes: ProposerNode[]; total: number }>(`/api/proposer/nodes?limit=${limit}&offset=${offset}`);
  return response.data;
}

export async function getProposerNode(address: string): Promise<ProposerNode> {
  const response = await apiGet<ProposerNode>(`/api/proposer/nodes/${address}`);
  return response.data;
}

export async function getAggregations(limit: number = 50, offset: number = 0, proposer?: string): Promise<{ aggregations: AggregationRecord[]; total: number }> {
  const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() });
  if (proposer) params.append('proposer', proposer);
  const response = await apiGet<{ aggregations: AggregationRecord[]; total: number }>(`/api/proposer/aggregations?${params}`);
  return response.data;
}

export async function getAggregation(aggregationId: number): Promise<AggregationRecord> {
  const response = await apiGet<AggregationRecord>(`/api/proposer/aggregations/${aggregationId}`);
  return response.data;
}

export async function getGradientPool(limit: number = 100, offset: number = 0, status: string = "pending"): Promise<GradientPool> {
  const response = await apiGet<GradientPool>(`/api/proposer/pool?limit=${limit}&offset=${offset}&status=${status}`);
  return response.data;
}

// Role Management API Functions
export interface NodeRole {
  role_id: number;
  role_name: string;
  description: string;
}

export interface NodeRoles {
  node_address: string;
  roles: number[];
  role_names: string[];
  status: string;
  resources?: any;
  stake?: string;
}

export interface RoleStats {
  role_id: number;
  role_name: string;
  total_nodes: number;
  active_nodes: number;
}

export async function getRoles(): Promise<{ roles: NodeRole[]; total: number }> {
  const response = await apiGet<{ roles: NodeRole[]; total: number }>("/api/roles");
  return response.data;
}

export async function getNodeRoles(address: string): Promise<NodeRoles> {
  const response = await apiGet<NodeRoles>(`/api/roles/${address}`);
  return response.data;
}

export async function getRoleStatistics(): Promise<{ stats: RoleStats[] }> {
  const response = await apiGet<{ stats: RoleStats[] }>("/api/roles/stats/summary");
  return response.data;
}

