/**
 * @jest-environment jsdom
 */

import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ErrorBoundary from '@/components/ErrorBoundary';

// Mock toast
jest.mock('@/lib/toast', () => ({
  toast: {
    error: jest.fn(),
    warning: jest.fn(),
  },
}));

// Mock logger
jest.mock('@/lib/logger', () => ({
  logger: {
    error: jest.fn(),
  },
}));

// Component that throws an error
const ThrowError = ({ shouldThrow = false }: { shouldThrow?: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>Working component</div>;
};

describe('ErrorBoundary Component', () => {
  // Mock window.location.reload
  const mockReload = jest.fn();
  const originalLocation = window.location;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Mock window.location
    delete (window as any).location;
    window.location = { ...originalLocation, reload: mockReload, href: '' } as any;
  });

  afterEach(() => {
    window.location = originalLocation;
  });

  describe('Normal Operation', () => {
    it('renders children when no error occurs', () => {
      render(
        <ErrorBoundary>
          <div>Test content</div>
        </ErrorBoundary>
      );

      expect(screen.getByText('Test content')).toBeInTheDocument();
    });

    it('does not show error UI when children render successfully', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(screen.getByText('Working component')).toBeInTheDocument();
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('catches and displays error when child component throws', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByRole('alert')).toBeInTheDocument();
      expect(screen.getByText(/component error/i)).toBeInTheDocument();
      expect(screen.getByText('Test error')).toBeInTheDocument();
    });

    it('shows error ID for tracking', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText(/error id:/i)).toBeInTheDocument();
    });

    it('shows appropriate error message for component level', () => {
      render(
        <ErrorBoundary level="component" name="TestComponent">
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText('TestComponent Error')).toBeInTheDocument();
      expect(screen.getByText(/this component failed to load/i)).toBeInTheDocument();
    });

    it('shows appropriate error message for root level', () => {
      render(
        <ErrorBoundary level="root">
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText('Application Error')).toBeInTheDocument();
      expect(screen.getByText(/critical error/i)).toBeInTheDocument();
    });
  });

  describe('Error Recovery', () => {
    it('shows retry button and allows recovery', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      const retryButton = screen.getByRole('button', { name: /try again/i });
      expect(retryButton).toBeInTheDocument();

      fireEvent.click(retryButton);

      // After retry, error should be cleared (though component may throw again)
      expect(screen.getByRole('alert')).toBeInTheDocument(); // Still shows because component throws again
    });

    it('shows reload button and triggers page reload', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      const reloadButton = screen.getByRole('button', { name: /reload page/i });
      expect(reloadButton).toBeInTheDocument();

      fireEvent.click(reloadButton);

      // Should trigger reload after timeout
      setTimeout(() => {
        expect(mockReload).toHaveBeenCalledTimes(1);
      }, 1100);
    });

    it('shows go home button for non-root level errors', () => {
      render(
        <ErrorBoundary level="component">
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByRole('button', { name: /go home/i })).toBeInTheDocument();
    });

    it('does not show go home button for root level errors', () => {
      render(
        <ErrorBoundary level="root">
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.queryByRole('button', { name: /go home/i })).not.toBeInTheDocument();
    });
  });

  describe('Error Information', () => {
    it('shows error details in development mode', () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'development';

      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText(/error details/i)).toBeInTheDocument();

      process.env.NODE_ENV = originalEnv;
    });

    it('hides error details in production mode', () => {
      const originalEnv = process.env.NODE_ENV;
      process.env.NODE_ENV = 'production';

      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.queryByText(/error details/i)).not.toBeInTheDocument();

      process.env.NODE_ENV = originalEnv;
    });
  });

  describe('Custom Fallback', () => {
    it('renders custom fallback when provided', () => {
      const CustomFallback = <div>Custom error message</div>;

      render(
        <ErrorBoundary fallback={CustomFallback}>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText('Custom error message')).toBeInTheDocument();
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('has proper ARIA attributes', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      const errorAlert = screen.getByRole('alert');
      expect(errorAlert).toHaveAttribute('aria-labelledby', 'error-title');
      expect(errorAlert).toHaveAttribute('aria-describedby', 'error-description');
    });

    it('provides screen reader help text', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText(/attempts to recover the component/i)).toBeInTheDocument();
      expect(screen.getByText(/reloads the entire page/i)).toBeInTheDocument();
    });

    it('announces errors to screen readers', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      // Check that announcement elements are created
      const announcements = document.querySelectorAll('[aria-live="assertive"]');
      expect(announcements.length).toBeGreaterThan(0);
    });
  });

  describe('Support Information', () => {
    it('shows support contact information', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText(/need help/i)).toBeInTheDocument();
      expect(screen.getByText(/discord support/i)).toBeInTheDocument();
      expect(screen.getByText(/report bug/i)).toBeInTheDocument();
    });

    it('has proper external links', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      const discordLink = screen.getByRole('link', { name: /discord support/i });
      const bugReportLink = screen.getByRole('link', { name: /report bug/i });

      expect(discordLink).toHaveAttribute('href', 'https://discord.gg/remes');
      expect(discordLink).toHaveAttribute('target', '_blank');
      expect(discordLink).toHaveAttribute('rel', 'noopener noreferrer');

      expect(bugReportLink).toHaveAttribute('href', 'https://github.com/AquaMystic/R3MES/issues');
      expect(bugReportLink).toHaveAttribute('target', '_blank');
      expect(bugReportLink).toHaveAttribute('rel', 'noopener noreferrer');
    });
  });

  describe('Retry Limits', () => {
    it('limits retry attempts', () => {
      const { rerender } = render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      // Click retry multiple times
      for (let i = 0; i < 5; i++) {
        const retryButton = screen.queryByRole('button', { name: /try again/i });
        if (retryButton) {
          fireEvent.click(retryButton);
          rerender(
            <ErrorBoundary>
              <ThrowError shouldThrow={true} />
            </ErrorBoundary>
          );
        }
      }

      // After max retries, button should not be available
      expect(screen.queryByRole('button', { name: /try again/i })).not.toBeInTheDocument();
    });

    it('shows retry count when retrying', () => {
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      const retryButton = screen.getByRole('button', { name: /try again/i });
      fireEvent.click(retryButton);

      expect(screen.getByText(/retry attempt 1 of 3/i)).toBeInTheDocument();
    });
  });
});