/**
 * Cosmos Kit Integration
 * 
 * Multi-wallet support using @cosmos-kit/react
 * Supports: Keplr, Leap, Cosmostation
 * 
 * Note: Cosmos Kit 2.0 is not yet stable, using manual wallet detection for now
 * This file provides a compatibility layer for future Cosmos Kit integration
 */

// Get Chain ID from environment variable (required for testnet/mainnet distinction)
// Default to testnet for safety (testnet is safer than mainnet for misconfiguration)
const chainId = process.env.NEXT_PUBLIC_CHAIN_ID || 
  (process.env.NODE_ENV === 'development' ? "remes-test" : "remes-testnet-1");

export const R3MES_CHAIN_INFO = {
  chainId: chainId,
  chainName: "R3MES Network",
  rpc: process.env.NEXT_PUBLIC_RPC_URL || "https://rpc.r3mes.network",
  rest: process.env.NEXT_PUBLIC_REST_URL || "https://api.r3mes.network",
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
      coinDenom: "REMES",
      coinMinimalDenom: "uremes",
      coinDecimals: 6,
    },
  ],
  feeCurrencies: [
    {
      coinDenom: "REMES",
      coinMinimalDenom: "uremes",
      coinDecimals: 6,
    },
  ],
  stakeCurrency: {
    coinDenom: "REMES",
    coinMinimalDenom: "uremes",
    coinDecimals: 6,
  },
  gasPriceStep: {
    low: 0.01,
    average: 0.025,
    high: 0.04,
  },
};

// Wallet detection utilities
export function detectAvailableWallets(): string[] {
  if (typeof window === "undefined") return [];
  
  const wallets: string[] = [];
  
  if (window.keplr) {
    wallets.push("keplr");
  }
  
  if (window.leap) {
    wallets.push("leap");
  }
  
  if (window.cosmostation) {
    wallets.push("cosmostation");
  }
  
  return wallets;
}

// Extend Window interface
declare global {
  interface Window {
    keplr?: any;
    leap?: any;
    cosmostation?: any;
  }
}

