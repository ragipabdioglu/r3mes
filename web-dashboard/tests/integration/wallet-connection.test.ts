/**
 * @jest-environment jsdom
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { WalletProvider } from '@/contexts/WalletContext';
import { ThemeProvider } from '@/contexts/ThemeContext';
import WalletButton from '@/components/WalletButton';
import WalletGuard from '@/components/WalletGuard';
import { getUserInfo } from '@/lib/api';

// Mock API functions
jest.mock('@/lib/api', () => ({
  getUserInfo: jest.fn(),
}));

// Mock Keplr wallet
const mockKeplr = {
  enable: jest.fn(),
  getKey: jest.fn(),
  disconnect: jest.fn(),
  experimentalSuggestChain: jest.fn(),
};

// Mock window.keplr
Object.defineProperty(window, 'keplr', {
  value: mockKeplr,
  writable: true,
});

const mockGetUserInfo = getUserInfo as jest.MockedFunction<typeof getUserInfo>;

// Test wrapper with all providers
const TestWrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

  return React.createElement(
    QueryClientProvider,
    { client: queryClient },
    React.createElement(
      ThemeProvider,
      null,
      React.createElement(WalletProvider, null, children)
    )
  );
};

// Test component that uses wallet guard
const ProtectedComponent = () => 
  React.createElement(
    WalletGuard,
    null,
    React.createElement('div', { 'data-testid': 'protected-content' }, 'Protected Content')
  );

describe('Wallet Connection Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    
    // Reset Keplr mock
    mockKeplr.enable.mockResolvedValue(undefined);
    mockKeplr.getKey.mockResolvedValue({
      name: 'Test Wallet',
      algo: 'secp256k1',
      pubKey: new Uint8Array(33),
      address: new Uint8Array(20),
      bech32Address: 'remes1234567890abcdef1234567890abcdef12',
    });
    mockKeplr.disconnect.mockResolvedValue(undefined);
  });

  describe('Wallet Button Integration', () => {
    it('shows connect button when wallet is not connected', () => {
      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      expect(screen.getByText(/connect wallet/i)).toBeInTheDocument();
    });

    it('connects wallet successfully', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(mockKeplr.enable).toHaveBeenCalledWith('r3mes');
        expect(mockKeplr.getKey).toHaveBeenCalledWith('r3mes');
      });

      await waitFor(() => {
        expect(screen.getByText(/remes1234/)).toBeInTheDocument();
      });
    });

    it('handles wallet connection errors', async () => {
      const user = userEvent.setup();
      
      mockKeplr.enable.mockRejectedValue(new Error('User rejected'));

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(screen.getByText(/connect wallet/i)).toBeInTheDocument();
      });
    });

    it('shows wallet address when connected', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(screen.getByText(/remes1234/)).toBeInTheDocument();
      });
    });

    it('disconnects wallet successfully', async () => {
      const user = userEvent.setup();
      
      // First connect
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(screen.getByText(/remes1234/)).toBeInTheDocument();
      });

      // Then disconnect
      const walletButton = screen.getByText(/remes1234/);
      await user.click(walletButton);

      const disconnectButton = screen.getByText(/disconnect/i);
      await user.click(disconnectButton);

      await waitFor(() => {
        expect(screen.getByText(/connect wallet/i)).toBeInTheDocument();
      });
    });
  });

  describe('Wallet Guard Integration', () => {
    it('shows connect prompt when wallet is not connected', () => {
      render(React.createElement(TestWrapper, null, React.createElement(ProtectedComponent)));

      expect(screen.getByText(/connect your wallet/i)).toBeInTheDocument();
      expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument();
    });

    it('shows protected content when wallet is connected', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, 
        React.createElement(React.Fragment, null, [
          React.createElement(WalletButton, { key: 'wallet' }),
          React.createElement(ProtectedComponent, { key: 'protected' })
        ])
      ));

      // Connect wallet first
      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(screen.getByTestId('protected-content')).toBeInTheDocument();
      });
    });

    it('hides protected content when wallet is disconnected', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, 
        React.createElement(React.Fragment, null, [
          React.createElement(WalletButton, { key: 'wallet' }),
          React.createElement(ProtectedComponent, { key: 'protected' })
        ])
      ));

      // Connect wallet
      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(screen.getByTestId('protected-content')).toBeInTheDocument();
      });

      // Disconnect wallet
      const walletButton = screen.getByText(/remes1234/);
      await user.click(walletButton);

      const disconnectButton = screen.getByText(/disconnect/i);
      await user.click(disconnectButton);

      await waitFor(() => {
        expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument();
        expect(screen.getByText(/connect your wallet/i)).toBeInTheDocument();
      });
    });
  });

  describe('Wallet State Persistence', () => {
    it('persists wallet address in localStorage', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(localStorage.getItem('keplr_address')).toBe('remes1234567890abcdef1234567890abcdef12');
      });
    });

    it('restores wallet state from localStorage', () => {
      localStorage.setItem('keplr_address', 'remes1234567890abcdef1234567890abcdef12');
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      // Should show wallet address immediately
      expect(screen.getByText(/remes1234/)).toBeInTheDocument();
    });

    it('clears localStorage on disconnect', async () => {
      const user = userEvent.setup();
      
      localStorage.setItem('keplr_address', 'remes1234567890abcdef1234567890abcdef12');
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const walletButton = screen.getByText(/remes1234/);
      await user.click(walletButton);

      const disconnectButton = screen.getByText(/disconnect/i);
      await user.click(disconnectButton);

      await waitFor(() => {
        expect(localStorage.getItem('keplr_address')).toBeNull();
      });
    });
  });

  describe('User Info Integration', () => {
    it('fetches user info after wallet connection', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(mockGetUserInfo).toHaveBeenCalledWith('remes1234567890abcdef1234567890abcdef12');
      });
    });

    it('displays user credits after connection', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 2500.75,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(screen.getByText('2.5K')).toBeInTheDocument();
      });
    });

    it('handles user info fetch errors', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockRejectedValue(new Error('API Error'));

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        // Should still show wallet address even if user info fails
        expect(screen.getByText(/remes1234/)).toBeInTheDocument();
      });
    });
  });

  describe('Keplr Wallet Detection', () => {
    it('handles missing Keplr extension', async () => {
      const user = userEvent.setup();
      
      // Remove Keplr from window
      Object.defineProperty(window, 'keplr', {
        value: undefined,
        writable: true,
      });

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(screen.getByText(/install keplr/i)).toBeInTheDocument();
      });
    });

    it('suggests chain addition for unsupported networks', async () => {
      const user = userEvent.setup();
      
      mockKeplr.enable.mockRejectedValue(new Error('Chain not found'));
      mockKeplr.experimentalSuggestChain.mockResolvedValue(undefined);

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(mockKeplr.experimentalSuggestChain).toHaveBeenCalled();
      });
    });
  });

  describe('Error Handling', () => {
    it('handles wallet connection timeout', async () => {
      const user = userEvent.setup();
      
      mockKeplr.enable.mockImplementation(() => 
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Timeout')), 100)
        )
      );

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(screen.getByText(/connect wallet/i)).toBeInTheDocument();
      }, { timeout: 2000 });
    });

    it('handles user rejection gracefully', async () => {
      const user = userEvent.setup();
      
      mockKeplr.enable.mockRejectedValue(new Error('Request rejected'));

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        expect(screen.getByText(/connect wallet/i)).toBeInTheDocument();
      });
    });

    it('handles network errors during user info fetch', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockRejectedValue(new Error('Network error'));

      render(React.createElement(TestWrapper, null, React.createElement(WalletButton)));

      const connectButton = screen.getByText(/connect wallet/i);
      await user.click(connectButton);

      await waitFor(() => {
        // Should still show connected state
        expect(screen.getByText(/remes1234/)).toBeInTheDocument();
      });
    });
  });

  describe('Multiple Wallet Instances', () => {
    it('synchronizes wallet state across components', async () => {
      const user = userEvent.setup();
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, 
        React.createElement(React.Fragment, null, [
          React.createElement(WalletButton, { key: 'wallet1' }),
          React.createElement(WalletButton, { key: 'wallet2' })
        ])
      ));

      const connectButtons = screen.getAllByText(/connect wallet/i);
      await user.click(connectButtons[0]);

      await waitFor(() => {
        const walletAddresses = screen.getAllByText(/remes1234/);
        expect(walletAddresses).toHaveLength(2);
      });
    });

    it('synchronizes disconnect across components', async () => {
      const user = userEvent.setup();
      
      localStorage.setItem('keplr_address', 'remes1234567890abcdef1234567890abcdef12');
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: 'remes1234567890abcdef1234567890abcdef12',
        credits: 1500,
        is_miner: true,
      });

      render(React.createElement(TestWrapper, null, 
        React.createElement(React.Fragment, null, [
          React.createElement(WalletButton, { key: 'wallet1' }),
          React.createElement(WalletButton, { key: 'wallet2' })
        ])
      ));

      // Both should show connected state
      const walletButtons = screen.getAllByText(/remes1234/);
      expect(walletButtons).toHaveLength(2);

      // Disconnect from first button
      await user.click(walletButtons[0]);
      const disconnectButton = screen.getByText(/disconnect/i);
      await user.click(disconnectButton);

      await waitFor(() => {
        const connectButtons = screen.getAllByText(/connect wallet/i);
        expect(connectButtons).toHaveLength(2);
      });
    });
  });
});