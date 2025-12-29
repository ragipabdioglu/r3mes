/**
 * R3MES JavaScript/TypeScript SDK
 *
 * Official SDK for interacting with the R3MES network from JavaScript/TypeScript.
 *
 * @example
 * ```typescript
 * import { R3MESClient } from '@r3mes/sdk';
 *
 * const client = new R3MESClient();
 *
 * // Get network stats
 * const stats = await client.getNetworkStats();
 *
 * // Chat with AI
 * for await (const token of await client.chat("Hello!", wallet)) {
 *   process.stdout.write(token);
 * }
 * ```
 */

// ============================================================================
// Types
// ============================================================================

export interface R3MESConfig {
  rpcUrl?: string;
  restUrl?: string;
  backendUrl?: string;
  timeout?: number;
}

export interface Wallet {
  address: string;
  sign(message: Uint8Array): Promise<Uint8Array>;
}

export interface NetworkStats {
  active_miners: number;
  total_users: number;
  total_credits: number;
  block_height: number;
}

export interface UserInfo {
  wallet_address: string;
  credits: number;
  is_miner: boolean;
}

export interface MinerStats {
  wallet_address: string;
  hashrate: number;
  total_earnings: number;
  blocks_found: number;
  uptime_percentage: number;
  is_active: boolean;
}

export interface BlockInfo {
  height: number;
  hash: string;
  timestamp: string;
  proposer: string;
  tx_count: number;
}

export interface Balance {
  denom: string;
  amount: string;
}

export interface Validator {
  operator_address: string;
  moniker: string;
  tokens: string;
  status: string;
  jailed: boolean;
  commission: string;
}

export interface LeaderboardEntry {
  rank: number;
  wallet_address: string;
  total_earnings: number;
  hashrate: number;
  blocks_found: number;
}

// ============================================================================
// Errors
// ============================================================================

export class R3MESError extends Error {
  code?: string;
  details?: Record<string, unknown>;

  constructor(message: string, code?: string, details?: Record<string, unknown>) {
    super(message);
    this.name = "R3MESError";
    this.code = code;
    this.details = details;
  }
}

export class ConnectionError extends R3MESError {
  constructor(message: string) {
    super(message, "CONNECTION_ERROR");
    this.name = "ConnectionError";
  }
}

export class AuthenticationError extends R3MESError {
  constructor(message: string) {
    super(message, "AUTHENTICATION_ERROR");
    this.name = "AuthenticationError";
  }
}

export class InsufficientCreditsError extends R3MESError {
  constructor(message: string) {
    super(message, "INSUFFICIENT_CREDITS");
    this.name = "InsufficientCreditsError";
  }
}

export class NotFoundError extends R3MESError {
  constructor(message: string) {
    super(message, "NOT_FOUND");
    this.name = "NotFoundError";
  }
}

export class RateLimitError extends R3MESError {
  constructor(message: string) {
    super(message, "RATE_LIMIT");
    this.name = "RateLimitError";
  }
}

// ============================================================================
// Main Client
// ============================================================================

export class R3MESClient {
  private rpcUrl: string;
  private restUrl: string;
  private backendUrl: string;
  private timeout: number;

  constructor(config?: R3MESConfig) {
    this.rpcUrl = config?.rpcUrl || "https://rpc.r3mes.network";
    this.restUrl = config?.restUrl || "https://api.r3mes.network";
    this.backendUrl = config?.backendUrl || "https://backend.r3mes.network";
    this.timeout = config?.timeout || 30000;
  }

  /**
   * Send a chat message and stream the response.
   */
  async chat(
    message: string,
    wallet?: Wallet,
    adapter?: string
  ): Promise<AsyncIterable<string>> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (wallet) {
      headers["X-Wallet-Address"] = wallet.address;
    }
    if (adapter) {
      headers["X-Adapter"] = adapter;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.backendUrl}/chat`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          message,
          wallet_address: wallet?.address,
        }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (response.status === 402) {
        throw new InsufficientCreditsError("Insufficient credits for chat");
      }
      if (response.status === 429) {
        throw new RateLimitError("Rate limit exceeded");
      }
      if (!response.ok) {
        throw new R3MESError(`Chat request failed: ${response.status}`);
      }

      return this.streamResponse(response);
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof R3MESError) throw error;
      throw new ConnectionError(`Failed to connect: ${error}`);
    }
  }

  private async *streamResponse(
    response: Response
  ): AsyncGenerator<string, void, unknown> {
    const reader = response.body?.getReader();
    if (!reader) {
      throw new R3MESError("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim()) {
            yield line;
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Get network statistics.
   */
  async getNetworkStats(): Promise<NetworkStats> {
    const response = await this.fetch(`${this.backendUrl}/network/stats`);
    return response.json();
  }

  /**
   * Get user information by wallet address.
   */
  async getUserInfo(walletAddress: string): Promise<UserInfo> {
    const response = await this.fetch(
      `${this.backendUrl}/user/info/${walletAddress}`
    );
    if (response.status === 404) {
      throw new NotFoundError(`User not found: ${walletAddress}`);
    }
    return response.json();
  }

  /**
   * Get miner statistics by wallet address.
   */
  async getMinerStats(walletAddress: string): Promise<MinerStats> {
    const response = await this.fetch(
      `${this.backendUrl}/miner/stats/${walletAddress}`
    );
    if (response.status === 404) {
      return {
        wallet_address: walletAddress,
        hashrate: 0,
        total_earnings: 0,
        blocks_found: 0,
        uptime_percentage: 0,
        is_active: false,
      };
    }
    return response.json();
  }

  /**
   * Get the latest block.
   */
  async getLatestBlock(): Promise<BlockInfo> {
    const response = await this.fetch(`${this.rpcUrl}/block`);
    const data = await response.json();
    const block = data.result?.block;
    return {
      height: parseInt(block?.header?.height || "0"),
      hash: data.result?.block_id?.hash || "",
      timestamp: block?.header?.time || "",
      proposer: block?.header?.proposer_address || "",
      tx_count: block?.data?.txs?.length || 0,
    };
  }

  /**
   * Get account balance.
   */
  async getBalance(address: string): Promise<Balance[]> {
    const response = await this.fetch(
      `${this.restUrl}/cosmos/bank/v1beta1/balances/${address}`
    );
    const data = await response.json();
    return data.balances || [];
  }

  /**
   * Get validators.
   */
  async getValidators(
    status: string = "BOND_STATUS_BONDED",
    limit: number = 100
  ): Promise<Validator[]> {
    const response = await this.fetch(
      `${this.restUrl}/cosmos/staking/v1beta1/validators?status=${status}&pagination.limit=${limit}`
    );
    const data = await response.json();
    return (data.validators || []).map((v: any) => ({
      operator_address: v.operator_address,
      moniker: v.description?.moniker || "",
      tokens: v.tokens,
      status: v.status,
      jailed: v.jailed,
      commission: v.commission?.commission_rates?.rate || "0",
    }));
  }

  /**
   * Get miner leaderboard.
   */
  async getLeaderboard(
    limit: number = 100,
    period: string = "all"
  ): Promise<LeaderboardEntry[]> {
    const response = await this.fetch(
      `${this.backendUrl}/leaderboard?limit=${limit}&period=${period}`
    );
    const data = await response.json();
    return data.miners || [];
  }

  private async fetch(url: string, options?: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (response.status === 429) {
        throw new RateLimitError("Rate limit exceeded");
      }

      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      if (error instanceof R3MESError) throw error;
      throw new ConnectionError(`Failed to connect: ${error}`);
    }
  }
}

// ============================================================================
// Miner Client
// ============================================================================

export class MinerClient {
  private backendUrl: string;

  constructor(backendUrl: string) {
    this.backendUrl = backendUrl;
  }

  async getEarningsHistory(
    walletAddress: string,
    limit: number = 100,
    offset: number = 0
  ): Promise<any[]> {
    const response = await fetch(
      `${this.backendUrl}/miner/earnings/${walletAddress}?limit=${limit}&offset=${offset}`
    );
    const data = await response.json();
    return data.earnings || [];
  }

  async getHashrateHistory(
    walletAddress: string,
    hours: number = 24
  ): Promise<any[]> {
    const response = await fetch(
      `${this.backendUrl}/miner/hashrate/${walletAddress}?hours=${hours}`
    );
    const data = await response.json();
    return data.hashrate || [];
  }
}

// ============================================================================
// Blockchain Client
// ============================================================================

export class BlockchainClient {
  private rpcUrl: string;
  private restUrl: string;

  constructor(rpcUrl: string, restUrl: string) {
    this.rpcUrl = rpcUrl;
    this.restUrl = restUrl;
  }

  async getBlock(height: number): Promise<BlockInfo> {
    const response = await fetch(`${this.rpcUrl}/block?height=${height}`);
    const data = await response.json();
    const block = data.result?.block;
    return {
      height: parseInt(block?.header?.height || "0"),
      hash: data.result?.block_id?.hash || "",
      timestamp: block?.header?.time || "",
      proposer: block?.header?.proposer_address || "",
      tx_count: block?.data?.txs?.length || 0,
    };
  }

  async getTransaction(txHash: string): Promise<any> {
    const response = await fetch(
      `${this.restUrl}/cosmos/tx/v1beta1/txs/${txHash}`
    );
    if (response.status === 404) {
      throw new NotFoundError(`Transaction not found: ${txHash}`);
    }
    return response.json();
  }

  async getStatus(): Promise<any> {
    const response = await fetch(`${this.rpcUrl}/status`);
    const data = await response.json();
    return data.result;
  }
}

// ============================================================================
// Governance Client
// ============================================================================

export interface Proposal {
  proposal_id: string;
  content: any;
  status: string;
  final_tally_result: Tally;
  submit_time: string;
  deposit_end_time: string;
  total_deposit: Balance[];
  voting_start_time: string;
  voting_end_time: string;
}

export interface Tally {
  yes: string;
  abstain: string;
  no: string;
  no_with_veto: string;
}

export class GovernanceClient {
  private restUrl: string;

  constructor(restUrl: string) {
    this.restUrl = restUrl;
  }

  async getProposals(status?: string, limit: number = 100): Promise<Proposal[]> {
    let url = `${this.restUrl}/cosmos/gov/v1beta1/proposals?pagination.limit=${limit}`;
    if (status) {
      url += `&proposal_status=${status}`;
    }
    const response = await fetch(url);
    const data = await response.json();
    return data.proposals || [];
  }

  async getProposal(proposalId: string): Promise<Proposal> {
    const response = await fetch(
      `${this.restUrl}/cosmos/gov/v1beta1/proposals/${proposalId}`
    );
    if (response.status === 404) {
      throw new NotFoundError(`Proposal not found: ${proposalId}`);
    }
    const data = await response.json();
    return data.proposal;
  }

  async getProposalTally(proposalId: string): Promise<Tally> {
    const response = await fetch(
      `${this.restUrl}/cosmos/gov/v1beta1/proposals/${proposalId}/tally`
    );
    const data = await response.json();
    return data.tally;
  }

  async getProposalVotes(proposalId: string, limit: number = 100): Promise<any[]> {
    const response = await fetch(
      `${this.restUrl}/cosmos/gov/v1beta1/proposals/${proposalId}/votes?pagination.limit=${limit}`
    );
    const data = await response.json();
    return data.votes || [];
  }
}

// ============================================================================
// Staking Client
// ============================================================================

export interface Delegation {
  delegator_address: string;
  validator_address: string;
  shares: string;
  balance: Balance;
}

export interface StakingPool {
  bonded_tokens: string;
  not_bonded_tokens: string;
}

export class StakingClient {
  private restUrl: string;

  constructor(restUrl: string) {
    this.restUrl = restUrl;
  }

  async getValidators(
    status: string = "BOND_STATUS_BONDED",
    limit: number = 100
  ): Promise<Validator[]> {
    const response = await fetch(
      `${this.restUrl}/cosmos/staking/v1beta1/validators?status=${status}&pagination.limit=${limit}`
    );
    const data = await response.json();
    return (data.validators || []).map((v: any) => ({
      operator_address: v.operator_address,
      moniker: v.description?.moniker || "",
      tokens: v.tokens,
      status: v.status,
      jailed: v.jailed,
      commission: v.commission?.commission_rates?.rate || "0",
    }));
  }

  async getValidator(operatorAddress: string): Promise<Validator> {
    const response = await fetch(
      `${this.restUrl}/cosmos/staking/v1beta1/validators/${operatorAddress}`
    );
    if (response.status === 404) {
      throw new NotFoundError(`Validator not found: ${operatorAddress}`);
    }
    const data = await response.json();
    const v = data.validator;
    return {
      operator_address: v.operator_address,
      moniker: v.description?.moniker || "",
      tokens: v.tokens,
      status: v.status,
      jailed: v.jailed,
      commission: v.commission?.commission_rates?.rate || "0",
    };
  }

  async getDelegations(delegatorAddress: string, limit: number = 100): Promise<Delegation[]> {
    const response = await fetch(
      `${this.restUrl}/cosmos/staking/v1beta1/delegations/${delegatorAddress}?pagination.limit=${limit}`
    );
    const data = await response.json();
    return (data.delegation_responses || []).map((d: any) => ({
      delegator_address: d.delegation.delegator_address,
      validator_address: d.delegation.validator_address,
      shares: d.delegation.shares,
      balance: d.balance,
    }));
  }

  async getUnbondingDelegations(delegatorAddress: string, limit: number = 100): Promise<any[]> {
    const response = await fetch(
      `${this.restUrl}/cosmos/staking/v1beta1/delegators/${delegatorAddress}/unbonding_delegations?pagination.limit=${limit}`
    );
    const data = await response.json();
    return data.unbonding_responses || [];
  }

  async getStakingPool(): Promise<StakingPool> {
    const response = await fetch(`${this.restUrl}/cosmos/staking/v1beta1/pool`);
    const data = await response.json();
    return data.pool;
  }

  async getRewards(delegatorAddress: string): Promise<any> {
    const response = await fetch(
      `${this.restUrl}/cosmos/distribution/v1beta1/delegators/${delegatorAddress}/rewards`
    );
    return response.json();
  }
}

// Default export
export default R3MESClient;

