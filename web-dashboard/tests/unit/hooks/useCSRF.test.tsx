/**
 * Unit tests for useCSRF hook
 */

import { renderHook, waitFor, act } from '@testing-library/react';
import { useCSRFToken, withCSRFProtection } from '@/hooks/useCSRF';
import React from 'react';
import { render, screen } from '@testing-library/react';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('useCSRFToken', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('should fetch CSRF token on mount', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ token: 'test-csrf-token' }),
    });

    const { result } = renderHook(() => useCSRFToken());

    // Initially loading
    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.token).toBe('test-csrf-token');
    expect(mockFetch).toHaveBeenCalledWith('/api/csrf-token', {
      method: 'GET',
      credentials: 'include',
    });
  });

  it('should handle fetch error gracefully', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => useCSRFToken());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.token).toBeNull();
  });

  it('should handle non-ok response', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const { result } = renderHook(() => useCSRFToken());

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.token).toBeNull();
  });

  it('should provide refreshToken function', async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ token: 'initial-token' }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ token: 'refreshed-token' }),
      });

    const { result } = renderHook(() => useCSRFToken());

    await waitFor(() => {
      expect(result.current.token).toBe('initial-token');
    });

    // Refresh token
    await act(async () => {
      await result.current.refreshToken();
    });

    expect(result.current.token).toBe('refreshed-token');
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});

describe('withCSRFProtection', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  it('should show loading state while fetching token', () => {
    // Never resolve to keep loading state
    mockFetch.mockImplementation(() => new Promise(() => {}));

    const TestComponent = () => <div>Protected Content</div>;
    const ProtectedComponent = withCSRFProtection(TestComponent);

    render(<ProtectedComponent />);

    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });

  it('should show error when token unavailable', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    const TestComponent = () => <div>Protected Content</div>;
    const ProtectedComponent = withCSRFProtection(TestComponent);

    render(<ProtectedComponent />);

    await waitFor(() => {
      expect(screen.getByText(/Security token unavailable/)).toBeInTheDocument();
    });
  });

  it('should render wrapped component with csrfToken prop', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ token: 'test-token' }),
    });

    interface TestProps {
      csrfToken?: string;
    }

    const TestComponent: React.FC<TestProps> = ({ csrfToken }) => (
      <div data-testid="token">{csrfToken}</div>
    );
    const ProtectedComponent = withCSRFProtection(TestComponent);

    render(<ProtectedComponent />);

    await waitFor(() => {
      expect(screen.getByTestId('token')).toHaveTextContent('test-token');
    });
  });

  it('should pass through other props', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({ token: 'test-token' }),
    });

    interface TestProps {
      customProp: string;
      csrfToken?: string;
    }

    const TestComponent: React.FC<TestProps> = ({ customProp }) => (
      <div data-testid="custom">{customProp}</div>
    );
    const ProtectedComponent = withCSRFProtection(TestComponent);

    render(<ProtectedComponent customProp="custom-value" />);

    await waitFor(() => {
      expect(screen.getByTestId('custom')).toHaveTextContent('custom-value');
    });
  });
});
