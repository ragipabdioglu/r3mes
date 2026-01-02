"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { getUserInfo, UserInfo } from '@/lib/api';
import { logger } from '@/lib/logger';

interface WalletContextType {
  walletAddress: string | null;
  userInfo: UserInfo | null;
  isConnecting: boolean;
  isConnected: boolean;
  connectWallet: () => Promise<void>;
  disconnectWallet: () => void;
  refreshUserInfo: () => Promise<void>;
}

const WalletContext = createContext<WalletContextType | undefined>(undefined);

export function useWallet() {
  const context = useContext(WalletContext);
  if (context === undefined) {
    throw new Error('useWallet must be used within a WalletProvider');
  }
  return context;
}

interface WalletProviderProps {
  children: ReactNode;
}

export function WalletProvider({ children }: WalletProviderProps) {
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [userInfo, setUserInfo] = useState<UserInfo | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);

  const isConnected = !!walletAddress;

  // Load wallet address from localStorage on mount
  useEffect(() => {
    const savedAddress = localStorage.getItem('keplr_address');
    if (savedAddress) {
      setWalletAddress(savedAddress);
    }
  }, []);

  // Fetch user info when wallet address changes
  useEffect(() => {
    if (walletAddress) {
      refreshUserInfo();
    } else {
      setUserInfo(null);
    }
  }, [walletAddress]);

  const connectWallet = async () => {
    const keplr = (window as any).keplr;
    if (typeof window === 'undefined' || !keplr) {
      throw new Error('Keplr wallet not found. Please install Keplr extension.');
    }

    setIsConnecting(true);
    
    try {
      // Enable Keplr for R3MES chain
      await keplr.enable('r3mes');
      
      // Get wallet key
      const key = await keplr.getKey('r3mes');
      const address = key.bech32Address;
      
      setWalletAddress(address);
      localStorage.setItem('keplr_address', address);
      
      logger.info('Wallet connected successfully', { address });
    } catch (error: any) {
      logger.error('Failed to connect wallet', error);
      
      // Handle specific Keplr errors
      if (error.message.includes('Chain not found')) {
        // Suggest adding R3MES chain
        await suggestChain();
        // Retry connection
        await connectWallet();
      } else {
        throw error;
      }
    } finally {
      setIsConnecting(false);
    }
  };

  const disconnectWallet = () => {
    setWalletAddress(null);
    setUserInfo(null);
    localStorage.removeItem('keplr_address');
    logger.info('Wallet disconnected');
  };

  const refreshUserInfo = async () => {
    if (!walletAddress) return;

    try {
      const info = await getUserInfo(walletAddress);
      setUserInfo(info);
    } catch (error) {
      logger.error('Failed to fetch user info', error);
      // Don't throw error here to avoid breaking the UI
    }
  };

  const suggestChain = async () => {
    const keplr = (window as any).keplr;
    if (!keplr) return;

    const chainInfo = {
      chainId: 'r3mes',
      chainName: 'R3MES Network',
      rpc: 'https://rpc.r3mes.com',
      rest: 'https://api.r3mes.com',
      bip44: {
        coinType: 118,
      },
      bech32Config: {
        bech32PrefixAccAddr: 'remes',
        bech32PrefixAccPub: 'remespub',
        bech32PrefixValAddr: 'remesvaloper',
        bech32PrefixValPub: 'remesvaloperpub',
        bech32PrefixConsAddr: 'remesvalcons',
        bech32PrefixConsPub: 'remesvalconspub',
      },
      currencies: [
        {
          coinDenom: 'REMES',
          coinMinimalDenom: 'uremes',
          coinDecimals: 6,
        },
      ],
      feeCurrencies: [
        {
          coinDenom: 'REMES',
          coinMinimalDenom: 'uremes',
          coinDecimals: 6,
          gasPriceStep: {
            low: 0.01,
            average: 0.025,
            high: 0.04,
          },
        },
      ],
      stakeCurrency: {
        coinDenom: 'REMES',
        coinMinimalDenom: 'uremes',
        coinDecimals: 6,
      },
    };

    await keplr.experimentalSuggestChain(chainInfo);
  };

  const value: WalletContextType = {
    walletAddress,
    userInfo,
    isConnecting,
    isConnected,
    connectWallet,
    disconnectWallet,
    refreshUserInfo,
  };

  return (
    <WalletContext.Provider value={value}>
      {children}
    </WalletContext.Provider>
  );
}