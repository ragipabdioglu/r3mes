"use client";

import { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: string | number;
  subtext?: string | ReactNode;
  icon?: ReactNode;
  trend?: "up" | "down" | "neutral";
  trendValue?: string;
}

export default function StatCard({
  label,
  value,
  subtext,
  icon,
  trend,
  trendValue,
}: StatCardProps) {
  return (
    <div className="stat-card">
      <div className="flex items-start justify-between mb-2">
        <div className="text-sm font-medium uppercase tracking-wide" style={{ color: 'var(--text-secondary)' }}>
          {label}
        </div>
        {icon && <div style={{ color: 'var(--accent-primary)' }}>{icon}</div>}
      </div>
      <div className="text-3xl font-bold mb-1" style={{ color: 'var(--text-primary)' }}>{value}</div>
      {subtext && (
        <div className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{subtext}</div>
      )}
      {trend && trendValue && (
        <div
          className="text-xs mt-2 flex items-center gap-1"
          style={{
            color: trend === "up" 
              ? "var(--success)" 
              : trend === "down" 
              ? "var(--error)" 
              : "var(--text-secondary)"
          }}
        >
          <span>{trend === "up" ? "↑" : trend === "down" ? "↓" : "→"}</span>
          <span>{trendValue}</span>
        </div>
      )}
    </div>
  );
}

