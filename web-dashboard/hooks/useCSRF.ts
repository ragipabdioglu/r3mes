'use client';

import { useCallback, useEffect, useState } from 'react';
import React from 'react';

/**
 * React hook for CSRF token management
 * @returns CSRF token and refresh function
 */
export function useCSRFToken() {
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshToken = useCallback(async () => {
    try {
      const response = await fetch('/api/csrf-token', {
        method: 'GET',
        credentials: 'include',
      });

      if (response.ok) {
        const data = await response.json();
        setToken(data.token);
      }
    } catch (error) {
      console.error('Failed to fetch CSRF token:', error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshToken();
  }, [refreshToken]);

  return { token, loading, refreshToken };
}

/**
 * Higher-order component to protect forms with CSRF
 */
export function withCSRFProtection<T extends object>(
  WrappedComponent: React.ComponentType<T>
) {
  return function CSRFProtectedComponent(props: T) {
    const { token, loading } = useCSRFToken();

    if (loading) {
      return React.createElement('div', null, 'Loading...');
    }

    if (!token) {
      return React.createElement('div', null, 'Security token unavailable. Please refresh the page.');
    }

    return React.createElement(WrappedComponent, { ...props, csrfToken: token });
  };
}