"use client";

import { ReactNode } from "react";
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string | number;
  subtext?: string | ReactNode;
  icon?: ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
  loading?: boolean;
  className?: string;
}

export default function StatCard({
  label,
  value,
  subtext,
  icon,
  trend,
  trendValue,
  loading = false,
  className = '',
}: StatCardProps) {
  const getTrendIcon = () => {
    switch (trend) {
      case 'up':
        return <TrendingUp className="w-4 h-4" aria-hidden="true" />;
      case 'down':
        return <TrendingDown className="w-4 h-4" aria-hidden="true" />;
      default:
        return <Minus className="w-4 h-4" aria-hidden="true" />;
    }
  };

  const getTrendColor = () => {
    switch (trend) {
      case 'up':
        return 'var(--success)';
      case 'down':
        return 'var(--error)';
      default:
        return 'var(--text-secondary)';
    }
  };

  const getTrendLabel = () => {
    switch (trend) {
      case 'up':
        return 'Increasing';
      case 'down':
        return 'Decreasing';
      default:
        return 'No change';
    }
  };

  if (loading) {
    return (
      <div 
        className={`stat-card animate-pulse ${className}`}
        role="status"
        aria-label="Loading statistics"
      >
        <div className="flex items-center justify-between mb-2">
          <div className="h-4 bg-gray-300 rounded w-20"></div>
          <div className="h-5 w-5 bg-gray-300 rounded"></div>
        </div>
        <div className="h-8 bg-gray-300 rounded w-24 mb-2"></div>
        <div className="h-3 bg-gray-300 rounded w-16"></div>
      </div>
    );
  }

  return (
    <div 
      className={`stat-card transition-all duration-200 hover:scale-[1.02] focus-within:ring-2 focus-within:ring-blue-500 focus-within:ring-offset-2 ${className}`}
      role="region"
      aria-labelledby={`stat-${label.replace(/\s+/g, '-').toLowerCase()}`}
      tabIndex={0}
    >
      <div className="flex items-start justify-between mb-2">
        <h3 
          id={`stat-${label.replace(/\s+/g, '-').toLowerCase()}`}
          className="text-sm font-medium uppercase tracking-wide" 
          style={{ color: 'var(--text-secondary)' }}
        >
          {label}
        </h3>
        {icon && (
          <div 
            style={{ color: 'var(--accent-primary)' }}
            aria-hidden="true"
            role="img"
            aria-label={`${label} icon`}
          >
            {icon}
          </div>
        )}
      </div>
      
      <div 
        className="text-3xl font-bold mb-1" 
        style={{ color: 'var(--text-primary)' }}
        aria-label={`${label} value: ${value}`}
      >
        {value}
      </div>
      
      {subtext && (
        <div 
          className="text-xs mt-1" 
          style={{ color: 'var(--text-muted)' }}
          aria-label={typeof subtext === 'string' ? `Additional info: ${subtext}` : undefined}
        >
          {subtext}
        </div>
      )}
      
      {trend && trendValue && (
        <div
          className="text-xs mt-2 flex items-center gap-1"
          style={{ color: getTrendColor() }}
          role="status"
          aria-label={`Trend: ${getTrendLabel()}, ${trendValue}`}
        >
          {getTrendIcon()}
          <span aria-hidden="true">{trendValue}</span>
          <span className="sr-only">
            {getTrendLabel()} by {trendValue}
          </span>
        </div>
      )}
    </div>
  );
}

