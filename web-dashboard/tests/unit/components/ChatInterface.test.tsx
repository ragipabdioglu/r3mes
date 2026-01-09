/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ChatInterface from '@/components/ChatInterface';
import { sendChatMessage } from '@/lib/api';

// Mock API functions
jest.mock('@/lib/api', () => ({
  sendChatMessage: jest.fn(),
}));

// Mock logger
jest.mock('@/lib/logger', () => ({
  logger: {
    error: jest.fn(),
  },
}));

const mockSendChatMessage = sendChatMessage as jest.MockedFunction<typeof sendChatMessage>;

describe('ChatInterface Component', () => {
  const mockWalletAddress = 'remes1234567890abcdef';

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Rendering', () => {
    it('renders chat interface with input and send button', () => {
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      expect(screen.getByPlaceholderText(/type your message/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /send message/i })).toBeInTheDocument();
    });

    it('shows empty message history initially', () => {
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      expect(screen.getByText(/sisteme bağlı/i)).toBeInTheDocument();
    });

    it('shows R3MES AI Assistant header', () => {
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      expect(screen.getByText('R3MES AI Assistant')).toBeInTheDocument();
    });
  });

  describe('Message Input', () => {
    it('updates input value when typing', async () => {
      const user = userEvent.setup();
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      await user.type(input, 'Hello, world!');
      
      expect(input).toHaveValue('Hello, world!');
    });

    it('enables send button when input has text', async () => {
      const user = userEvent.setup();
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      expect(sendButton).toBeDisabled();
      
      await user.type(input, 'Hello');
      
      expect(sendButton).not.toBeDisabled();
    });

    it('handles enter key to send message', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockImplementation((message, address, onChunk) => {
        onChunk('Response chunk');
        return Promise.resolve();
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      await user.type(input, 'Hello{enter}');
      
      await waitFor(() => {
        expect(mockSendChatMessage).toHaveBeenCalledWith(
          'Hello',
          mockWalletAddress,
          expect.any(Function)
        );
      });
    });

    it('prevents sending empty messages', async () => {
      const user = userEvent.setup();
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const sendButton = screen.getByRole('button', { name: /send message/i });
      await user.click(sendButton);
      
      expect(mockSendChatMessage).not.toHaveBeenCalled();
    });

    it('trims whitespace from messages', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockResolvedValue();
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, '  Hello World  ');
      await user.click(sendButton);
      
      expect(mockSendChatMessage).toHaveBeenCalledWith(
        'Hello World',
        mockWalletAddress,
        expect.any(Function)
      );
    });
  });

  describe('Message Sending', () => {
    it('sends message successfully', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockImplementation((message, address, onChunk) => {
        onChunk('AI response');
        return Promise.resolve();
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Hello AI');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('Hello AI')).toBeInTheDocument();
        expect(screen.getByText('AI response')).toBeInTheDocument();
      });
    });

    it('shows loading state while sending', async () => {
      const user = userEvent.setup();
      let resolveMessage: () => void;
      mockSendChatMessage.mockImplementation(() => {
        return new Promise((resolve) => {
          resolveMessage = resolve;
        });
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Hello');
      await user.click(sendButton);
      
      expect(screen.getByText(/ai is thinking/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /sending message/i })).toBeDisabled();
      
      resolveMessage!();
      
      await waitFor(() => {
        expect(screen.queryByText(/ai is thinking/i)).not.toBeInTheDocument();
      });
    });

    it('clears input after successful send', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockResolvedValue();
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Hello');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(input).toHaveValue('');
      });
    });

    it('handles streaming responses', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockImplementation((message, address, onChunk) => {
        onChunk('First ');
        setTimeout(() => onChunk('chunk'), 10);
        setTimeout(() => onChunk(' of response'), 20);
        return Promise.resolve();
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Hello');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText(/First chunk of response/)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('handles API errors gracefully', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockRejectedValue(new Error('API Error'));
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Hello');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('Error: API Error')).toBeInTheDocument();
      });
    });

    it('re-enables send button after error', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockRejectedValue(new Error('API Error'));
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Hello');
      await user.click(sendButton);
      
      // Wait for error message to appear
      await waitFor(() => {
        expect(screen.getByText('Error: API Error')).toBeInTheDocument();
      });
      
      // Type new message to enable button
      await user.type(input, 'New message');
      
      // Button should be enabled with new input
      await waitFor(() => {
        expect(sendButton).not.toBeDisabled();
      });
    });
  });

  describe('Message History', () => {
    it('displays message history correctly', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockImplementation((message, address, onChunk) => {
        onChunk(`Response to: ${message}`);
        return Promise.resolve();
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      // Send first message
      await user.type(input, 'First message');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('First message')).toBeInTheDocument();
        expect(screen.getByText('Response to: First message')).toBeInTheDocument();
      });
      
      // Send second message
      await user.type(input, 'Second message');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('Second message')).toBeInTheDocument();
        expect(screen.getByText('Response to: Second message')).toBeInTheDocument();
      });
    });

    it('shows message count in header', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockImplementation((message, address, onChunk) => {
        onChunk('Response');
        return Promise.resolve();
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      expect(screen.getByText('0 messages')).toBeInTheDocument();
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Hello');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('2 messages')).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA labels', () => {
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      expect(screen.getByLabelText(/Type your message to R3MES AI/i)).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /send message/i })).toBeInTheDocument();
    });

    it('has proper role attributes', () => {
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      expect(screen.getByRole('main')).toBeInTheDocument();
      expect(screen.getByRole('banner')).toBeInTheDocument();
      expect(screen.getByRole('log')).toBeInTheDocument();
      expect(screen.getByRole('form')).toBeInTheDocument();
    });

    it('maintains focus management', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockImplementation((message, address, onChunk) => {
        onChunk('Response');
        return Promise.resolve();
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Hello');
      await user.click(sendButton);
      
      // Wait for message to be sent and focus to return
      await waitFor(() => {
        expect(screen.getByText('Response')).toBeInTheDocument();
      });
      
      await waitFor(() => {
        expect(document.activeElement).toBe(input);
      });
    });
  });

  describe('Adapter Detection', () => {
    it('detects code adapter for programming queries', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockImplementation((message, address, onChunk) => {
        onChunk('Code response');
        return Promise.resolve();
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Help me with Python code');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('Code Assistant Mode')).toBeInTheDocument();
      });
    });

    it('detects law adapter for legal queries', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockImplementation((message, address, onChunk) => {
        onChunk('Legal response');
        return Promise.resolve();
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'What are my legal rights?');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('Legal Assistant Mode')).toBeInTheDocument();
      });
    });

    it('uses default adapter for general queries', async () => {
      const user = userEvent.setup();
      mockSendChatMessage.mockImplementation((message, address, onChunk) => {
        onChunk('General response');
        return Promise.resolve();
      });
      
      render(<ChatInterface walletAddress={mockWalletAddress} />);
      
      const input = screen.getByPlaceholderText(/type your message/i);
      const sendButton = screen.getByRole('button', { name: /send message/i });
      
      await user.type(input, 'Hello, how are you?');
      await user.click(sendButton);
      
      await waitFor(() => {
        expect(screen.getByText('General Assistant Mode')).toBeInTheDocument();
      });
    });
  });
});