"use client";

import { useEffect, useState } from "react";
import { Settings, Server, Layers, Shield, Cpu, CheckCircle, XCircle } from "lucide-react";
import { getRoles, getNodeRoles, getRoleStatistics, NodeRole, NodeRoles, RoleStats } from "@/lib/api";
import { useAnnouncer } from "@/hooks/useAccessibility";
import { formatNumber, formatTokenAmount } from "@/utils/formatters";
import { logger } from "@/lib/logger";
import WalletGuard from "@/components/WalletGuard";
import { SkeletonStatCard } from "@/components/SkeletonLoader";
import { signAndBroadcastTransaction, createRegisterNodeMessage } from "@/lib/keplr";
import { toast } from "@/lib/toast";

const ROLE_ICONS: Record<number, React.ReactNode> = {
  1: <Cpu className="w-5 h-5" />,
  2: <Server className="w-5 h-5" />,
  3: <Shield className="w-5 h-5" />,
  4: <Layers className="w-5 h-5" />,
};

const ROLE_NAMES: Record<number, string> = {
  1: "Miner",
  2: "Serving",
  3: "Validator",
  4: "Proposer",
};

const ROLE_ACCESS_INFO: Record<number, { public: boolean; label: string; message?: string; minStake?: string }> = {
  1: { 
    public: true, 
    label: "Miner - Open Access",
    minStake: "1,000 REMES"
  },
  2: { 
    public: true, 
    label: "Serving - Open Access",
    minStake: "1,000 REMES"
  },
  3: { 
    public: false, 
    label: "Validator - Authorization Required",
    message: "Validator role requires special authorization (whitelist or governance approval). Minimum stake: 100,000 REMES. Please contact admin or request access.",
    minStake: "100,000 REMES"
  },
  4: { 
    public: false, 
    label: "Proposer - Authorization Required",
    message: "Proposer role requires validator status or special authorization. Minimum stake: 50,000 REMES.",
    minStake: "50,000 REMES"
  },
};

function RolesPageContent() {
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [roles, setRoles] = useState<NodeRole[]>([]);
  const [nodeRoles, setNodeRoles] = useState<NodeRoles | null>(null);
  const [roleStats, setRoleStats] = useState<RoleStats[]>([]);
  const [selectedRoles, setSelectedRoles] = useState<number[]>([]);
  const [stake, setStake] = useState<string>("1000000");
  const [isRegistering, setIsRegistering] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  // Accessibility announcer
  const { announce, announceError, announceSuccess, announceLoading } = useAnnouncer();

  useEffect(() => {
    setMounted(true);
    const address = localStorage.getItem("keplr_address");
    setWalletAddress(address);
    loadData();
  }, []);

  const loadData = async () => {
    try {
      setIsLoading(true);
      announceLoading("role data", true);
      const [rolesData, statsData] = await Promise.all([
        getRoles().catch(() => ({ roles: [], total: 0 })),
        getRoleStatistics().catch(() => ({ stats: [] })),
      ]);
      setRoles(rolesData.roles);
      setRoleStats(statsData.stats);

      if (walletAddress) {
        const nodeRolesData = await getNodeRoles(walletAddress).catch(() => null);
        setNodeRoles(nodeRolesData);
        if (nodeRolesData) {
          setSelectedRoles(nodeRolesData.roles);
        }
      }
      setIsLoading(false);
      announce(`Loaded ${rolesData.roles.length} roles`);
    } catch (error) {
      logger.error("Failed to fetch role data:", error);
      announceError("Failed to load role data");
      setIsLoading(false);
    }
  };

  const handleRoleToggle = (roleId: number) => {
    const roleName = ROLE_NAMES[roleId];
    if (selectedRoles.includes(roleId)) {
      setSelectedRoles(selectedRoles.filter((id) => id !== roleId));
      announce(`Deselected ${roleName} role`);
    } else {
      setSelectedRoles([...selectedRoles, roleId]);
      announce(`Selected ${roleName} role`);
    }
  };

  const handleRegister = async () => {
    if (!walletAddress || selectedRoles.length === 0) {
      toast.error("Please select at least one role");
      return;
    }

    // Validate minimum stake requirements
    const stakeAmount = parseInt(stake) || 0;
    const minStakes: Record<number, number> = {
      1: 1000000, // 1,000 REMES in uremes (6 decimals)
      2: 1000000, // 1,000 REMES
      3: 100000000, // 100,000 REMES
      4: 50000000, // 50,000 REMES
    };

    const selectedRoleIds = selectedRoles;
    const maxMinStake = Math.max(...selectedRoleIds.map(roleId => minStakes[roleId] || 0));

    if (stakeAmount < maxMinStake) {
      toast.error(`Insufficient stake. Minimum required: ${maxMinStake / 1000000} REMES for selected roles.`);
      return;
    }

    try {
      setIsRegistering(true);

      // Create RegisterNode message
      const message = createRegisterNodeMessage(
        walletAddress,
        selectedRoles.map(r => r.toString()),
        `${stake}uremes`,
        {
          cpuCores: 4,
          memoryGb: 8,
          gpuCount: 1,
          gpuMemoryGb: 12,
          storageGb: 100,
          networkBandwidthMbps: 1000,
        }
      );

      // Sign and broadcast transaction
      const txHash = await signAndBroadcastTransaction(
        [message],
        `Register node with roles: ${selectedRoles.map(r => ROLE_NAMES[r]).join(', ')}`,
        "auto"
      );

      toast.success(`Registration successful! Transaction: ${txHash.slice(0, 12)}...`);
      announceSuccess("Node registration successful");
      
      // Reload data after successful registration
      await loadData();
    } catch (error: any) {
      logger.error("Failed to register node:", error);
      const errorMessage = error?.message || String(error) || "Unknown error";
      toast.error(`Registration failed: ${errorMessage}`);
      announceError(`Registration failed: ${errorMessage}`);
    } finally {
      setIsRegistering(false);
    }
  };

  return (
    <WalletGuard>
      <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
        <div className="container mx-auto max-w-full sm:max-w-2xl md:max-w-3xl lg:max-w-4xl">
          <div className="mb-8">
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold gradient-text mb-2">Role Management</h1>
            <p className="text-sm sm:text-base text-[var(--text-secondary)]">
              Register and manage your node roles
            </p>
          </div>

          {/* Role Statistics */}
          {isLoading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5 md:gap-6 mb-8">
              {[1, 2, 3, 4].map((i) => (
                <SkeletonStatCard key={i} />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5 md:gap-6 mb-8">
              {roleStats.map((stat) => (
                <div key={stat.role_id} className="card p-4 sm:p-5 md:p-6">
                  <div className="flex items-center gap-3 mb-2">
                    {ROLE_ICONS[stat.role_id]}
                    <div className="text-sm font-semibold text-[var(--text-primary)]">
                      {stat.role_name}
                    </div>
                  </div>
                  <div className="text-2xl font-bold text-[var(--accent-primary)] mb-1">
                    {formatNumber(stat.total_nodes)}
                  </div>
                  <div className="text-xs text-[var(--text-secondary)]">
                    {formatNumber(stat.active_nodes)} active
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Current Node Roles */}
          {nodeRoles && (
            <div className="card mb-8">
              <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-[var(--text-primary)]">Current Roles</h2>
              <div className="flex flex-wrap gap-3">
                {nodeRoles.roles.map((roleId) => (
                  <div
                    key={roleId}
                    className="flex items-center gap-2 px-4 py-2 rounded-full bg-[var(--accent-primary)]/20 border border-[var(--accent-primary)]/30"
                  >
                    {ROLE_ICONS[roleId]}
                    <span className="text-sm font-medium text-[var(--accent-primary)]">
                      {ROLE_NAMES[roleId]}
                    </span>
                    <CheckCircle className="w-4 h-4 text-[var(--success)]" />
                  </div>
                ))}
              </div>
              <div className="mt-4 text-sm text-[var(--text-secondary)]">
                Status: <span className="font-medium text-[var(--text-primary)]">{nodeRoles.status}</span>
              </div>
            </div>
          )}

          {/* Role Registration */}
          <div className="card">
            <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-[var(--text-primary)]">
              {nodeRoles ? "Update Roles" : "Register Node"}
            </h2>

            {/* Role Selection */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-[var(--text-primary)] mb-3">
                Select Roles (Multi-role supported)
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {roles.map((role) => {
                  const accessInfo = ROLE_ACCESS_INFO[role.id];
                  const isSelected = selectedRoles.includes(role.id);
                  const isPublic = accessInfo?.public ?? true;
                  
                  return (
                    <div
                      key={role.id}
                      className={`p-4 rounded-xl border transition-all ${
                        isSelected
                          ? "border-[var(--accent-primary)] bg-[var(--accent-primary)]/10"
                          : "border-[var(--border-color)]"
                      } ${!isPublic ? "bg-[var(--bg-secondary)]" : ""}`}
                    >
                      <button
                        onClick={() => handleRoleToggle(role.id)}
                        className="w-full text-left"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-3">
                            {ROLE_ICONS[role.id]}
                            <div className="flex-1">
                              <div className="font-semibold text-[var(--text-primary)] text-sm sm:text-base">
                                {role.name}
                              </div>
                              <div className="text-xs text-[var(--text-secondary)] mt-1">
                                {role.description}
                              </div>
                            </div>
                          </div>
                          {isSelected ? (
                            <CheckCircle className="w-5 h-5 text-[var(--accent-primary)] shrink-0" />
                          ) : (
                            <div className="w-5 h-5 rounded-full border-2 border-[var(--border-color)] shrink-0" />
                          )}
                        </div>
                      </button>
                      
                      {/* Access Control Info */}
                      <div className="mt-3 pt-3 border-t border-[var(--border-color)]">
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-xs font-medium ${isPublic ? "text-[var(--success)]" : "text-[var(--warning)]"}`}>
                            {accessInfo?.label || (isPublic ? "Open Access" : "Authorization Required")}
                          </span>
                          {accessInfo?.minStake && (
                            <span className="text-xs text-[var(--text-secondary)]">
                              Min: {accessInfo.minStake}
                            </span>
                          )}
                        </div>
                        {!isPublic && accessInfo?.message && (
                          <div className="mt-2 p-2 rounded-lg bg-[var(--warning)]/10 border border-[var(--warning)]/30">
                            <p className="text-xs text-[var(--text-secondary)]">
                              {accessInfo.message}
                            </p>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Stake Input */}
            <div className="mb-6">
              <label className="block text-sm font-semibold text-[var(--text-primary)] mb-2">
                Stake Amount (uremes)
              </label>
              <input
                type="text"
                value={stake}
                onChange={(e) => setStake(e.target.value)}
                placeholder="1000000"
                className="w-full bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-lg px-3 py-2 sm:px-4 sm:py-3 text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent-primary)]"
              />
              <p className="text-xs text-[var(--text-muted)] mt-1">
                Minimum stake: {selectedRoles.length > 0 ? 
                  Math.max(...selectedRoles.map(r => {
                    const minStakes: Record<number, number> = {1: 1000000, 2: 1000000, 3: 100000000, 4: 50000000};
                    return minStakes[r] || 0;
                  })) / 1000000 + " REMES" : 
                  "1,000 REMES (varies by role)"
                }
              </p>
            </div>

            {/* Register Button */}
            <button
              onClick={handleRegister}
              disabled={isRegistering || selectedRoles.length === 0 || !walletAddress}
              className="w-full btn-primary py-2.5 sm:py-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {isRegistering ? (
                <>
                  <span className="animate-pulse">●</span>
                  <span>Registering...</span>
                </>
              ) : (
                <>
                  <Settings className="w-5 h-5" />
                  <span>{nodeRoles ? "Update Roles" : "Register Node"}</span>
                </>
              )}
            </button>

            {selectedRoles.length > 0 && (
              <div className="mt-4 p-4 rounded-xl bg-[var(--bg-secondary)] border border-[var(--border-color)]">
                <div className="text-sm font-semibold text-[var(--text-primary)] mb-2">
                  Selected Roles:
                </div>
                <div className="flex flex-wrap gap-2">
                  {selectedRoles.map((roleId) => (
                    <span
                      key={roleId}
                      className="px-3 py-1 rounded-full bg-[var(--accent-primary)]/20 text-[var(--accent-primary)] text-xs font-medium"
                    >
                      {ROLE_NAMES[roleId]}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </WalletGuard>
  );
}

export default function RolesPage() {
  return <RolesPageContent />;
}

