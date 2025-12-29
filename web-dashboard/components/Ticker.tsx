"use client";

import { useEffect, useState } from "react";
import { getNetworkStats, NetworkStats } from "@/lib/api";
import { logger } from "@/lib/logger";

export default function Ticker() {
  const [stats, setStats] = useState<NetworkStats | null>(null);

  useEffect(() => {
    // Initial fetch
    getNetworkStats()
      .then(setStats)
      .catch((err) => logger.error("Failed to fetch network stats:", err));

    // Refresh every 10 seconds
    const interval = setInterval(() => {
      getNetworkStats()
        .then(setStats)
        .catch((err) => logger.error("Failed to fetch network stats:", err));
    }, 10000);

    return () => clearInterval(interval);
  }, []);

  if (!stats) {
    return (
      <div className="py-3 text-xs text-center" style={{ color: 'var(--text-muted)' }}>
        Loading network stats...
      </div>
    );
  }

  return (
    <div className="py-3 overflow-hidden border-t" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--bg-secondary)' }}>
      <div className="flex gap-8 animate-scroll whitespace-nowrap text-sm">
        <span style={{ color: 'var(--text-primary)' }}>
          ACTIVE NODES:{" "}
          <span className="font-bold" style={{ color: 'var(--accent-primary)' }}>
            {stats.active_miners.toLocaleString()}
          </span>
        </span>
        <span style={{ color: 'var(--text-muted)' }}>///</span>
        <span style={{ color: 'var(--text-primary)' }}>
          TOTAL FLOPS: <span className="font-bold" style={{ color: 'var(--accent-primary)' }}>450 Peta</span>
        </span>
        <span style={{ color: 'var(--text-muted)' }}>///</span>
        <span style={{ color: 'var(--text-primary)' }}>
          EPOCH: <span className="font-bold" style={{ color: 'var(--accent-primary)' }}>42</span>
        </span>
        {stats.block_height !== undefined && stats.block_height !== null && (
          <>
            <span style={{ color: 'var(--text-muted)' }}>///</span>
            <span style={{ color: 'var(--text-primary)' }}>
              BLOCK:{" "}
              <span className="font-bold" style={{ color: 'var(--accent-primary)' }}>
                #{stats.block_height.toLocaleString()}
              </span>
            </span>
          </>
        )}
      </div>
    </div>
  );
}
