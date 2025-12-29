/**
 * Unit tests for StatCard component
 */

import { describe, it, expect } from '@jest/globals';
import { render, screen } from '@testing-library/react';
import React from 'react';
import StatCard from '@/components/StatCard';
import { Network } from 'lucide-react';

describe('StatCard', () => {
  it('should render label and value', () => {
    render(
      <StatCard
        label="Test Label"
        value="100"
        icon={<Network />}
      />
    );

    expect(screen.getByText('Test Label')).toBeInTheDocument();
    expect(screen.getByText('100')).toBeInTheDocument();
  });

  it('should render icon', () => {
    const { container } = render(
      <StatCard
        label="Test Label"
        value="100"
        icon={<Network data-testid="network-icon" />}
      />
    );

    expect(container.querySelector('[data-testid="network-icon"]')).toBeInTheDocument();
  });

  it('should handle numeric values', () => {
    render(
      <StatCard
        label="Count"
        value={123}
        icon={<Network />}
      />
    );

    expect(screen.getByText('123')).toBeInTheDocument();
  });

  it('should handle loading state', () => {
    render(
      <StatCard
        label="Loading"
        value="Loading..."
        icon={<Network />}
      />
    );

    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });
});

