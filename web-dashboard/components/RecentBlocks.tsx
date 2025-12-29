"use client";

import { useQuery } from "@tanstack/react-query";
import "./RecentBlocks.css";

interface Block {
  height: number;
  hash: string;
  time: string;
  tx_count: number;
  proposer: string;
  validator_set_hash: string;
}

export default function RecentBlocks() {
  const { data: blocks, isLoading, error } = useQuery<Block[]>({
    queryKey: ["explorer", "blocks"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/dashboard/blocks?limit=20");
      if (!response.ok) {
        throw new Error("Failed to fetch blocks");
      }
      const data = await response.json();
      return data.blocks || [];
    },
    refetchInterval: 10000, // Refetch every 10 seconds
  });

  const formatTime = (time: string) => {
    const date = new Date(time);
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
    
    if (diff < 60) return `${diff}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleDateString();
  };

  if (isLoading) {
    return (
      <div className="recent-blocks-container">
        <div className="loading">Loading blocks...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="recent-blocks-container">
        <div className="error">Failed to load blocks. Please try again.</div>
      </div>
    );
  }

  return (
    <div className="recent-blocks-container">
      <h3 className="table-title">Recent Blocks</h3>
      <div className="blocks-list">
        {blocks && blocks.length > 0 ? (
          blocks.map((block) => (
            <div key={block.hash} className="block-item">
              <div className="block-header">
                <div className="block-height">#{block.height}</div>
                <div className="block-time">{formatTime(block.time)}</div>
              </div>
              <div className="block-hash">
                {block.hash.slice(0, 16)}...{block.hash.slice(-8)}
              </div>
              <div className="block-footer">
                <div className="block-tx">
                  <span className="tx-icon">📦</span>
                  <span>{block.tx_count} transactions</span>
                </div>
                <div className="block-proposer">
                  Proposer: {block.proposer.slice(0, 12)}...
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="blocks-empty">No blocks found</div>
        )}
      </div>
    </div>
  );
}

