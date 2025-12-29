"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface LiveMetricsTickerProps {
  stats?: {
    active_miners?: number;
    total_users?: number;
    block_height?: number;
  };
  isLoading: boolean;
}

export default function LiveMetricsTicker({ stats, isLoading }: LiveMetricsTickerProps) {
  const [tickerText, setTickerText] = useState("");

  useEffect(() => {
    if (isLoading) {
      setTickerText("LOADING NETWORK STATS...");
      return;
    }

    if (stats) {
      const metrics = [
        `ACTIVE NODES: ${stats.active_miners?.toLocaleString() || "0"}`,
        `TOTAL FLOPS: 450 Peta`,
        `EPOCH: 42`,
        `BLOCK: #${stats.block_height?.toLocaleString() || "0"}`,
      ];
      setTickerText(metrics.join(" /// "));
    } else {
      setTickerText("ACTIVE NODES: 0 /// TOTAL FLOPS: 0 Peta /// EPOCH: 0 /// BLOCK: #0");
    }
  }, [stats, isLoading]);

  return (
    <div className="bg-black border-t border-green-500/50 py-3 overflow-hidden">
      <motion.div
        className="text-green-400 font-mono text-sm whitespace-nowrap"
        animate={{
          x: [0, -1000],
        }}
        transition={{
          x: {
            repeat: Infinity,
            repeatType: "loop",
            duration: 30,
            ease: "linear",
          },
        }}
      >
        {tickerText} • {tickerText} • {tickerText}
      </motion.div>
    </div>
  );
}

