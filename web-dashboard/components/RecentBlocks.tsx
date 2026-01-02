"use client";

import { useQuery } from "@tanstack/react-query";

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
    refetchInterval: 10000,
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
      <div className="mb-8">
        <div className="text-center py-10 text-slate-400 col-span-full">Loading blocks...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-8">
        <div className="text-center py-10 text-red-500 col-span-full">Failed to load blocks. Please try again.</div>
      </div>
    );
  }

  return (
    <div className="mb-8">
      <h3 className="text-xl font-semibold text-slate-100 mb-5">Recent Blocks</h3>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(300px,1fr))] gap-4">
        {blocks && blocks.length > 0 ? (
          blocks.map((block) => (
            <div 
              key={block.hash} 
              className="bg-slate-800 border border-slate-700 rounded-lg p-4 transition-all hover:border-slate-600 hover:shadow-lg hover:shadow-black/30"
            >
              <div className="flex justify-between items-center mb-3">
                <div className="text-lg font-semibold text-blue-500">#{block.height}</div>
                <div className="text-xs text-slate-500">{formatTime(block.time)}</div>
              </div>
              <div className="font-mono text-xs text-slate-400 mb-3 break-all">
                {block.hash.slice(0, 16)}...{block.hash.slice(-8)}
              </div>
              <div className="flex justify-between items-center pt-3 border-t border-slate-700 text-xs">
                <div className="flex items-center gap-1.5 text-slate-400">
                  <span className="text-sm">📦</span>
                  <span>{block.tx_count} transactions</span>
                </div>
                <div className="text-slate-500 font-mono">
                  Proposer: {block.proposer.slice(0, 12)}...
                </div>
              </div>
            </div>
          ))
        ) : (
          <div className="py-10 text-center text-slate-400 col-span-full">No blocks found</div>
        )}
      </div>
    </div>
  );
}
