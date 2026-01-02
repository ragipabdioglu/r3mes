"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { useQuery } from "@tanstack/react-query";
import { logger } from "@/lib/logger";
import MinersTable from "./MinersTable";
import RecentBlocks from "./RecentBlocks";
import NetworkStats from "./NetworkStats";

// Lazy load Globe component (only for Network Explorer, not Miner Console)
// Add error handling for chunk loading failures
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const Globe = dynamic<any>(
  () => import("react-globe.gl").then((mod) => mod.default),
  {
    ssr: false,
    loading: () => <div className="h-full flex items-center justify-center text-slate-400">Loading globe...</div>,
  }
);

interface NetworkStatus {
  active_miners: number;
  total_gradients: number;
  model_updates: number;
  block_height: number;
  block_time: number;
  network_hash_rate: number;
  timestamp: number;
}

interface MinerLocation {
  address: string;
  lat: number;
  lng: number;
  size: number;
}

interface MinerLocationsResponse {
  locations: MinerLocation[];
  total: number;
}

export default function NetworkExplorer() {
  const [globeData, setGlobeData] = useState<
    Array<{ lat: number; lng: number; size: number }>
  >([]);
  const [mounted, setMounted] = useState(false);

  // Fetch network status
  const {
    data: networkStatus,
    isLoading: isNetworkLoading,
    error: networkError,
  } = useQuery<NetworkStatus>({
    queryKey: ["networkStatus"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/dashboard/status");
      if (!response.ok) {
        throw new Error("Failed to fetch network status");
      }
      return response.json();
    },
    refetchInterval: 5000,
  });

  useEffect(() => {
    setMounted(true);
  }, []);

  // Fetch miner locations for globe visualization
  const {
    data: minerLocations,
    isLoading: isLocationsLoading,
    error: locationsError,
  } = useQuery<MinerLocationsResponse>({
    queryKey: ["minerLocations"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/dashboard/locations");
      if (!response.ok) {
        throw new Error("Failed to fetch miner locations");
      }
      return response.json();
    },
    refetchInterval: 10000,
  });

  // Update globe data when miner locations change
  useEffect(() => {
    if (minerLocations?.locations) {
      const points = minerLocations.locations.map((loc: MinerLocation) => ({
        lat: loc.lat,
        lng: loc.lng,
        size: loc.size,
      }));
      setGlobeData(points);
    }
  }, [minerLocations]);

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6">
        <h2 className="text-2xl font-semibold text-slate-50 mb-2">
          Network Explorer <span className="text-slate-400 text-sm">/ Visor</span>
        </h2>
        <p className="text-slate-400 text-sm">
          Global view of R3MES network nodes, gradients and block activity.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass-panel p-6">
          <h3 className="text-lg font-semibold text-slate-100 mb-4">
            Network Statistics
          </h3>
          {isNetworkLoading && (
            <div className="space-y-3 animate-pulse">
              <div className="h-4 rounded bg-slate-800/80" />
              <div className="h-4 rounded bg-slate-800/80" />
              <div className="h-4 rounded bg-slate-800/80" />
              <div className="h-4 rounded bg-slate-800/80" />
            </div>
          )}
          {networkError && !isNetworkLoading && (
            <div className="text-xs text-amber-300 bg-amber-500/10 border border-amber-400/40 rounded-md px-3 py-2">
              Unable to reach node API at{" "}
              <span className="font-mono text-amber-100">
                /api/blockchain/dashboard/status
              </span>
              . Make sure your `remesd` REST API is running.
            </div>
          )}
          {networkStatus && !isNetworkLoading && !networkError && (
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-slate-400">Active Miners</span>
                <span className="font-semibold text-cyan-300">
                  {networkStatus.active_miners}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Total Gradients</span>
                <span className="font-semibold text-slate-100">
                  {networkStatus.total_gradients.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Block Height</span>
                <span className="font-semibold text-slate-100">
                  {networkStatus.block_height.toLocaleString()}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Network Hash Rate</span>
                <span className="font-semibold text-slate-100">
                  {networkStatus.network_hash_rate.toFixed(2)} gradients/hour
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="lg:col-span-2 glass-panel p-6">
          <h3 className="text-lg font-semibold text-slate-100 mb-4">
            Global Node Map
          </h3>
          <div className="h-96 rounded-2xl overflow-hidden bg-slate-950">
            {locationsError && !isLocationsLoading ? (
              <div className="h-full flex items-center justify-center text-xs text-amber-300 px-4 text-center">
                Unable to load miner locations from{" "}
                <span className="font-mono ml-1">
                  /api/blockchain/dashboard/locations
                </span>
                . Globe visualization will be disabled until the API is online.
              </div>
            ) : mounted ? (
              <Globe
                globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
                pointsData={globeData}
                pointColor="color"
                pointRadius="size"
                pointResolution={2}
                onGlobeReady={() => {
                  logger.info("Globe ready");
                }}
              />
            ) : (
              <div className="h-full flex items-center justify-center text-slate-400">
                Initializing globe...
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Network Stats */}
      <NetworkStats />

      {/* Miners Table */}
      <MinersTable />

      {/* Recent Blocks */}
      <RecentBlocks />
    </div>
  );
}

