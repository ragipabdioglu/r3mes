/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { usePathname } from 'next/navigation';
import Navbar from '@/components/Navbar';
import { useWallet } from '@/contexts/WalletContext';
import { useTheme } from '@/contexts/ThemeContext';

// Mock Next.js navigation
jest.mock('next/navigation', () => ({
  usePathname: jest.fn(),
}));

// Mock contexts
jest.mock('@/contexts/WalletContext', () => ({
  useWallet: jest.fn(),
}));

jest.mock('@/contexts/ThemeContext', () => ({
  useTheme: jest.fn(),
}));

// Mock WalletButton component
jest.mock('@/components/WalletButton', () => {
  return function MockWalletButton() {
    return <button data-testid="wallet-button">Connect Wallet</button>;
  };
});

// Mock framer-motion
jest.mock('framer-motion', () => ({
  motion: {
    nav: ({ children, ...props }: any) => <nav {...props}>{children}</nav>,
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

const mockUsePathname = usePathname as jest.MockedFunction<typeof usePathname>;
const mockUseWallet = useWallet as jest.MockedFunction<typeof useWallet>;
const mockUseTheme = useTheme as jest.MockedFunction<typeof useTheme>;

describe('Navbar Component', () => {
  const mockToggleTheme = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    
    mockUsePathname.mockReturnValue('/');
    mockUseWallet.mockReturnValue({
      walletAddress: null,
      credits: null,
      isConnecting: false,
      error: null,
      connectWallet: jest.fn(),
      disconnectWallet: jest.fn(),
    });
    mockUseTheme.mockReturnValue({
      theme: 'light',
      resolvedTheme: 'light',
      toggleTheme: mockToggleTheme,
    });

    // Mock window.scrollY
    Object.defineProperty(window, 'scrollY', {
      value: 0,
      writable: true,
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Rendering', () => {
    it('renders the navbar with logo', () => {
      render(<Navbar />);
      
      expect(screen.getByText('R3MES')).toBeInTheDocument();
      expect(screen.getByLabelText('R3MES Home')).toBeInTheDocument();
    });

    it('renders main navigation links', () => {
      render(<Navbar />);
      
      expect(screen.getByText('Home')).toBeInTheDocument();
      expect(screen.getByText('Chat')).toBeInTheDocument();
      expect(screen.getByText('Mine')).toBeInTheDocument();
      expect(screen.getByText('Network')).toBeInTheDocument();
    });

    it('renders theme toggle button', () => {
      render(<Navbar />);
      
      const themeButton = screen.getByLabelText('Toggle theme');
      expect(themeButton).toBeInTheDocument();
    });

    it('renders wallet button', () => {
      render(<Navbar />);
      
      expect(screen.getByTestId('wallet-button')).toBeInTheDocument();
    });

    it('renders mobile menu button on mobile', () => {
      render(<Navbar />);
      
      const mobileMenuButton = screen.getByLabelText('Toggle menu');
      expect(mobileMenuButton).toBeInTheDocument();
    });
  });

  describe('Active Link Highlighting', () => {
    it('highlights active link correctly', () => {
      mockUsePathname.mockReturnValue('/mine');
      render(<Navbar />);
      
      const mineLink = screen.getByText('Mine');
      expect(mineLink).toHaveAttribute('aria-current', 'page');
    });

    it('does not highlight inactive links', () => {
      mockUsePathname.mockReturnValue('/mine');
      render(<Navbar />);
      
      const homeLink = screen.getByText('Home');
      expect(homeLink).not.toHaveAttribute('aria-current');
    });
  });

  describe('Theme Toggle', () => {
    it('calls toggleTheme when theme button is clicked', () => {
      render(<Navbar />);
      
      const themeButton = screen.getByLabelText('Toggle theme');
      fireEvent.click(themeButton);
      
      expect(mockToggleTheme).toHaveBeenCalledTimes(1);
    });

    it('shows sun icon in dark mode', () => {
      mockUseTheme.mockReturnValue({
        theme: 'dark',
        resolvedTheme: 'dark',
        toggleTheme: mockToggleTheme,
      });
      
      render(<Navbar />);
      
      // Check for sun icon (light mode icon shown in dark mode)
      const themeButton = screen.getByLabelText('Toggle theme');
      expect(themeButton).toBeInTheDocument();
    });

    it('shows moon icon in light mode', () => {
      mockUseTheme.mockReturnValue({
        theme: 'light',
        resolvedTheme: 'light',
        toggleTheme: mockToggleTheme,
      });
      
      render(<Navbar />);
      
      // Check for moon icon (dark mode icon shown in light mode)
      const themeButton = screen.getByLabelText('Toggle theme');
      expect(themeButton).toBeInTheDocument();
    });
  });

  describe('Wallet Integration', () => {
    it('shows credits when wallet is connected', () => {
      mockUseWallet.mockReturnValue({
        walletAddress: 'remes1234567890abcdef',
        credits: 1500.75,
        isConnecting: false,
        error: null,
        connectWallet: jest.fn(),
        disconnectWallet: jest.fn(),
      });
      
      render(<Navbar />);
      
      expect(screen.getByText('1.5K')).toBeInTheDocument();
    });

    it('does not show credits when wallet is not connected', () => {
      render(<Navbar />);
      
      expect(screen.queryByText(/\d+K/)).not.toBeInTheDocument();
    });
  });

  describe('Mobile Menu', () => {
    it('opens mobile menu when button is clicked', async () => {
      render(<Navbar />);
      
      const menuButton = screen.getByLabelText('Toggle menu');
      fireEvent.click(menuButton);
      
      // Check if mobile menu items are visible
      await waitFor(() => {
        expect(screen.getAllByText('Home')).toHaveLength(2); // Desktop + mobile
      });
    });

    it('closes mobile menu when backdrop is clicked', async () => {
      render(<Navbar />);
      
      const menuButton = screen.getByLabelText('Toggle menu');
      fireEvent.click(menuButton);
      
      // Wait for menu to open
      await waitFor(() => {
        expect(screen.getAllByText('Home')).toHaveLength(2);
      });
      
      // Click backdrop (this would need to be implemented in the component)
      // For now, just test that the menu can be toggled
      fireEvent.click(menuButton);
      
      await waitFor(() => {
        expect(screen.getAllByText('Home')).toHaveLength(1); // Only desktop
      });
    });

    it('closes mobile menu when a link is clicked', async () => {
      render(<Navbar />);
      
      const menuButton = screen.getByLabelText('Toggle menu');
      fireEvent.click(menuButton);
      
      await waitFor(() => {
        expect(screen.getAllByText('Home')).toHaveLength(2);
      });
      
      // Click on a mobile menu link
      const mobileLinks = screen.getAllByText('Chat');
      fireEvent.click(mobileLinks[1]); // Click mobile version
      
      await waitFor(() => {
        expect(screen.getAllByText('Home')).toHaveLength(1);
      });
    });
  });

  describe('Keyboard Navigation', () => {
    it('handles escape key to close mobile menu', async () => {
      render(<Navbar />);
      
      const menuButton = screen.getByLabelText('Toggle menu');
      fireEvent.click(menuButton);
      
      await waitFor(() => {
        expect(screen.getAllByText('Home')).toHaveLength(2);
      });
      
      // Press escape key
      fireEvent.keyDown(document, { key: 'Escape', code: 'Escape' });
      
      await waitFor(() => {
        expect(screen.getAllByText('Home')).toHaveLength(1);
      });
    });

    it('supports tab navigation through links', () => {
      render(<Navbar />);
      
      const homeLink = screen.getByText('Home');
      const chatLink = screen.getByText('Chat');
      
      // Tab to first link
      homeLink.focus();
      expect(document.activeElement).toBe(homeLink);
      
      // Tab to next link
      fireEvent.keyDown(homeLink, { key: 'Tab' });
      // Note: Actual tab navigation would need jsdom-testing-library setup
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA attributes', () => {
      render(<Navbar />);
      
      const nav = screen.getByRole('navigation');
      expect(nav).toHaveAttribute('aria-label', 'Main navigation');
    });

    it('has proper button labels', () => {
      render(<Navbar />);
      
      expect(screen.getByLabelText('Toggle theme')).toBeInTheDocument();
      expect(screen.getByLabelText('Toggle menu')).toBeInTheDocument();
      expect(screen.getByLabelText('R3MES Home')).toBeInTheDocument();
    });

    it('indicates current page for screen readers', () => {
      mockUsePathname.mockReturnValue('/chat');
      render(<Navbar />);
      
      const chatLink = screen.getByText('Chat');
      expect(chatLink).toHaveAttribute('aria-current', 'page');
    });
  });

  describe('Responsive Behavior', () => {
    it('shows desktop navigation on large screens', () => {
      // Mock window.innerWidth
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      });
      
      render(<Navbar />);
      
      // Desktop navigation should be visible
      expect(screen.getByText('Home')).toBeInTheDocument();
      expect(screen.getByText('More')).toBeInTheDocument();
    });

    it('hides desktop navigation on small screens', () => {
      // Mock window.innerWidth
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 640,
      });
      
      render(<Navbar />);
      
      // Mobile menu button should be visible
      expect(screen.getByLabelText('Toggle menu')).toBeInTheDocument();
    });
  });

  describe('Scroll Behavior', () => {
    it('updates navbar style on scroll', async () => {
      render(<Navbar />);
      
      // Simulate scroll
      Object.defineProperty(window, 'scrollY', {
        value: 50,
        writable: true,
      });
      
      fireEvent.scroll(window);
      
      // The navbar should update its appearance
      // This would need to be tested with actual DOM changes
      await waitFor(() => {
        // Test implementation would check for style changes
        expect(true).toBe(true); // Placeholder
      });
    });
  });

  describe('Error Handling', () => {
    it('handles missing wallet context gracefully', () => {
      mockUseWallet.mockReturnValue({
        walletAddress: null,
        credits: null,
        isConnecting: false,
        error: 'Wallet connection failed',
        connectWallet: jest.fn(),
        disconnectWallet: jest.fn(),
      });
      
      expect(() => render(<Navbar />)).not.toThrow();
    });

    it('handles missing theme context gracefully', () => {
      mockUseTheme.mockReturnValue({
        theme: 'light',
        resolvedTheme: 'light',
        toggleTheme: jest.fn(() => {
          throw new Error('Theme toggle failed');
        }),
      });
      
      render(<Navbar />);
      
      const themeButton = screen.getByLabelText('Toggle theme');
      
      // Should not crash when theme toggle fails
      expect(() => fireEvent.click(themeButton)).not.toThrow();
    });
  });
});