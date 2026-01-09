/**
 * Keplr Wallet Integration
 * 
 * Handles connection to Keplr wallet and R3MES chain configuration
 * Includes transaction signing and broadcasting
 */

import { SigningStargateClient } from "@cosmjs/stargate";
import { logger } from "./logger";
import { OfflineSigner } from "@cosmjs/proto-signing";
import { Decimal } from "@cosmjs/math";
import { TxRaw } from "cosmjs-types/cosmos/tx/v1beta1/tx";
import { toast } from "./toast";

export interface KeplrChainInfo {
  chainId: string;
  chainName: string;
  rpc: string;
  rest: string;
  bip44: {
    coinType: number;
  };
  bech32Config: {
    bech32PrefixAccAddr: string;
    bech32PrefixAccPub: string;
    bech32PrefixValAddr: string;
    bech32PrefixValPub: string;
    bech32PrefixConsAddr: string;
    bech32PrefixConsPub: string;
  };
  currencies: Array<{
    coinDenom: string;
    coinMinimalDenom: string;
    coinDecimals: number;
  }>;
  feeCurrencies: Array<{
    coinDenom: string;
    coinMinimalDenom: string;
    coinDecimals: number;
  }>;
  stakeCurrency: {
    coinDenom: string;
    coinMinimalDenom: string;
    coinDecimals: number;
  };
}

// Get R3MES chain info with environment-aware configuration
const getR3MESChainInfo = (): KeplrChainInfo => {
  // Get RPC URL from environment variable
  const rpcUrl = process.env.NEXT_PUBLIC_RPC_URL || 
    process.env.NEXT_PUBLIC_BLOCKCHAIN_RPC_URL || 
    (process.env.NODE_ENV === 'development' ? "http://localhost:26657" : "https://rpc.r3mes.network");
  
  // Get REST URL from environment variable
  const restUrl = process.env.NEXT_PUBLIC_REST_URL ||
    process.env.NEXT_PUBLIC_BLOCKCHAIN_REST_URL ||
    (process.env.NODE_ENV === 'development' ? "http://localhost:1317" : "https://rest.r3mes.network");

  // Get Chain ID from environment variable (required for testnet/mainnet distinction)
  const chainId = process.env.NEXT_PUBLIC_CHAIN_ID || "r3mes-testnet-1";

  // Get denom from environment variable
  const denom = process.env.NEXT_PUBLIC_DENOM || "ur3mes";
  const denomDisplay = process.env.NEXT_PUBLIC_DENOM_DISPLAY || "R3MES";

  return {
    chainId: chainId,
    chainName: "R3MES Testnet",
    rpc: rpcUrl,
    rest: restUrl,
    bip44: {
      coinType: 118,
    },
    bech32Config: {
      bech32PrefixAccAddr: "remes",
      bech32PrefixAccPub: "remespub",
      bech32PrefixValAddr: "remesvaloper",
      bech32PrefixValPub: "remesvaloperpub",
      bech32PrefixConsAddr: "remesvalcons",
      bech32PrefixConsPub: "remesvalconspub",
    },
    currencies: [
      {
        coinDenom: denomDisplay,
        coinMinimalDenom: denom,
        coinDecimals: 6,
      },
    ],
    feeCurrencies: [
      {
        coinDenom: denomDisplay,
        coinMinimalDenom: denom,
        coinDecimals: 6,
      },
    ],
    stakeCurrency: {
      coinDenom: denomDisplay,
      coinMinimalDenom: denom,
      coinDecimals: 6,
    },
  };
};

// Export chain info constant
export const R3MES_CHAIN_INFO = getR3MESChainInfo();

export async function connectKeplrWallet(): Promise<string | null> {
  if (typeof window === "undefined" || !window.keplr) {
    throw new Error("Keplr wallet not found. Please install Keplr extension.");
  }

  try {
    // Check if chain is already added
    const chainId = R3MES_CHAIN_INFO.chainId;
    try {
      await window.keplr.enable(chainId);
    } catch {
      // Chain not added, add it
      await window.keplr.experimentalSuggestChain(R3MES_CHAIN_INFO);
      await window.keplr.enable(chainId);
    }

    // Get offline signer
    const offlineSigner = window.keplr.getOfflineSigner(chainId);
    const accounts = await offlineSigner.getAccounts();

    if (accounts.length === 0) {
      throw new Error("No accounts found in Keplr wallet");
    }

    return accounts[0].address;
  } catch (error) {
    logger.error("Failed to connect Keplr wallet:", error);
    throw error;
  }
}

export async function getKeplrBalance(address: string): Promise<string> {
  if (typeof window === "undefined" || !window.keplr) {
    throw new Error("Keplr wallet not found");
  }

  try {
    const restEndpoint = R3MES_CHAIN_INFO.rest;
    
    // Use REST API to get balance
    const response = await fetch(`${restEndpoint}/cosmos/bank/v1beta1/balances/${address}`);
    
    if (!response.ok) {
      throw new Error(`Failed to fetch balance: ${response.statusText}`);
    }
    
    const data = await response.json();
    // Find the balance for 'ur3mes' denom
    const denom = process.env.NEXT_PUBLIC_DENOM || 'ur3mes';
    const balance = data.balances?.find(
      (b: any) => b.denom === denom || b.denom === 'ur3mes' || b.denom === 'stake'
    );
    
    return balance?.amount || "0";
  } catch (error) {
    logger.error("Failed to get balance:", error);
    throw error;
  }
}

/**
 * Get offline signer from Keplr wallet
 */
export async function getKeplrOfflineSigner(): Promise<OfflineSigner> {
  if (typeof window === "undefined" || !window.keplr) {
    throw new Error("Keplr wallet not found");
  }

  const chainId = R3MES_CHAIN_INFO.chainId;
  return window.keplr.getOfflineSigner(chainId);
}

/**
 * Get signing client for transaction signing and broadcasting
 */
export async function getSigningClient(): Promise<SigningStargateClient> {
  const offlineSigner = await getKeplrOfflineSigner();
  const rpcEndpoint = R3MES_CHAIN_INFO.rpc;
  
  return await SigningStargateClient.connectWithSigner(
    rpcEndpoint,
    offlineSigner,
    {
      gasPrice: {
        amount: Decimal.fromUserInput("1", 0),
        denom: "uremes",
      },
    }
  );
}

/**
 * Sign and broadcast a transaction
 * 
 * @param messages Array of message objects to include in the transaction
 * @param memo Optional memo for the transaction
 * @param gasLimit Optional gas limit (default: "auto")
 * @returns Transaction hash
 */
export async function signAndBroadcastTransaction(
  messages: any[],
  memo: string = "",
  gasLimit?: string
): Promise<string> {
  if (typeof window === "undefined" || !window.keplr) {
    throw new Error("Keplr wallet not found");
  }

  try {
    const chainId = R3MES_CHAIN_INFO.chainId;
    const offlineSigner = await getKeplrOfflineSigner();
    const accounts = await offlineSigner.getAccounts();
    
    if (accounts.length === 0) {
      throw new Error("No accounts found in Keplr wallet");
    }

    const client = await getSigningClient();
    const signerAddress = accounts[0].address;

    // Sign and broadcast transaction
    const result = await client.signAndBroadcast(
      signerAddress,
      messages,
      gasLimit === "auto" ? "auto" : (gasLimit ? parseInt(gasLimit) : "auto"),
      memo
    );

    if (result.code !== 0) {
      const errorMsg = `Transaction failed: ${result.rawLog}`;
      toast.error(errorMsg);
      throw new Error(errorMsg);
    }

    toast.success(`Transaction successful: ${result.transactionHash.slice(0, 10)}...`);
    return result.transactionHash;
  } catch (error: any) {
    logger.error("Failed to sign and broadcast transaction:", error);
    toast.error(error.message || "Transaction failed");
    throw error;
  }
}

/**
 * Sign a transaction without broadcasting (for offline signing)
 * 
 * @param messages Array of message objects
 * @param memo Optional memo
 * @returns Signed transaction bytes
 */
export async function signTransaction(
  address: string,
  messages: any[],
  memo: string = ""
): Promise<Uint8Array> {
  if (typeof window === "undefined" || !window.keplr) {
    throw new Error("Keplr wallet not found");
  }

  try {
    const chainId = R3MES_CHAIN_INFO.chainId;
    const offlineSigner = await getKeplrOfflineSigner();
    const client = await getSigningClient();
    
    // Get account info
    const accounts = await offlineSigner.getAccounts();
    const account = accounts.find((acc) => acc.address === address);
    if (!account) {
      throw new Error(`Account ${address} not found`);
    }

    // Build transaction
    const { SigningStargateClient } = await import("@cosmjs/stargate");
    const fee = {
      amount: [{ amount: "1000", denom: "uremes" }],
      gas: "200000",
    };

    // Sign transaction
    const signedTx = await client.sign(
      address,
      messages,
      fee,
      memo
    );

    // Convert TxRaw to Uint8Array
    return TxRaw.encode(signedTx).finish();
  } catch (error: any) {
    logger.error("Failed to sign transaction:", error);
    throw error;
  }
}

/**
 * Helper function to create a message for RegisterNode transaction
 */
export function createRegisterNodeMessage(
  nodeAddress: string,
  roles: string[],
  stake: string,
  resourceQuotas: any
): any {
  return {
    typeUrl: "/remes.remes.v1.MsgRegisterNode",
    value: {
      nodeAddress,
      roles,
      stake,
      resourceQuotas,
    },
  };
}

/**
 * Helper function to create a message for ProposeDataset transaction
 */
export function createProposeDatasetMessage(
  proposer: string,
  datasetIpfsHash: string,
  metadata: any
): any {
  return {
    typeUrl: "/remes.remes.v1.MsgProposeDataset",
    value: {
      proposer,
      datasetIpfsHash,
      metadata,
    },
  };
}

/**
 * Helper function to create a message for RequestInference transaction
 */
export function createRequestInferenceMessage(
  requester: string,
  servingNode: string,
  inputDataIpfsHash: string,
  fee: string,
  modelVersion?: string
): any {
  return {
    typeUrl: "/remes.remes.v1.MsgRequestInference",
    value: {
      requester,
      servingNode,
      inputDataIpfsHash,
      fee,
      modelVersion: modelVersion || "",
    },
  };
}

// Window interface is extended by @keplr-wallet/types
// We don't need to redeclare it here to avoid type conflicts
// The types are already available from the package

// Leap Wallet Integration
export async function connectLeapWallet(): Promise<string | null> {
  if (typeof window === "undefined" || !window.leap) {
    throw new Error("Leap wallet not found. Please install Leap extension.");
  }

  try {
    const chainId = R3MES_CHAIN_INFO.chainId;
    try {
      await window.leap.enable(chainId);
    } catch {
      await window.leap.suggestChain(R3MES_CHAIN_INFO);
      await window.leap.enable(chainId);
    }

    const offlineSigner = window.leap.getOfflineSigner(chainId);
    const accounts = await offlineSigner.getAccounts();

    if (accounts.length === 0) {
      throw new Error("No accounts found in Leap wallet");
    }

    return accounts[0].address;
  } catch (error) {
    logger.error("Failed to connect Leap wallet:", error);
    throw error;
  }
}

// Cosmostation Wallet Integration
export async function connectCosmostationWallet(): Promise<string | null> {
  if (typeof window === "undefined" || !window.cosmostation?.providers?.keplr) {
    throw new Error("Cosmostation wallet not found. Please install Cosmostation extension.");
  }

  try {
    const chainId = R3MES_CHAIN_INFO.chainId;
    const cosmostation = window.cosmostation.providers.keplr;

    try {
      await cosmostation.enable(chainId);
    } catch {
      await cosmostation.suggestChain(R3MES_CHAIN_INFO);
      await cosmostation.enable(chainId);
    }

    const offlineSigner = cosmostation.getOfflineSigner(chainId);
    const accounts = await offlineSigner.getAccounts();

    if (accounts.length === 0) {
      throw new Error("No accounts found in Cosmostation wallet");
    }

    return accounts[0].address;
  } catch (error) {
    logger.error("Failed to connect Cosmostation wallet:", error);
    throw error;
  }
}

