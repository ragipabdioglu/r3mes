"use client";

import Link from "next/link";
import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useNetworkStats } from "@/hooks/useNetworkStats";
import StatCard from "@/components/StatCard";
import { getUserFriendlyError } from "@/utils/errorMessages";
import { Cpu, Zap, Shield, Database, Network, Code } from "lucide-react";

export default function DashboardHome() {
  const router = useRouter();
  const { data: stats, isLoading, error } = useNetworkStats();

  useEffect(() => {
    // Check if onboarding is needed
    const onboardingCompleted = localStorage.getItem("r3mes_onboarding_completed");
    if (onboardingCompleted !== "true") {
      router.push("/onboarding");
    }
  }, [router]);

  const features = [
    {
      icon: <Zap className="w-6 h-6" />,
      title: "99.6%",
      subtitle: "Bandwidth Reduction",
      description: "LoRA training with minimal data transfer",
    },
    {
      icon: <Cpu className="w-6 h-6" />,
      title: "BitNet b1.58",
      subtitle: "Ultra-Efficient",
      description: "1-bit quantized neural networks",
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: "PoUW",
      subtitle: "Consensus",
      description: "Proof of Useful Work",
    },
    {
      icon: <Code className="w-6 h-6" />,
      title: "LoRA",
      subtitle: "Training Default",
      description: "Frozen backbone + trainable adapters",
    },
    {
      icon: <Database className="w-6 h-6" />,
      title: "IPFS",
      subtitle: "Storage",
      description: "Decentralized data storage",
    },
    {
      icon: <Network className="w-6 h-6" />,
      title: "Cosmos SDK",
      subtitle: "Blockchain",
      description: "Production-ready infrastructure",
    },
  ];

  const steps = [
    {
      number: "1",
      title: "Connect GPU",
      description: "Connect your NVIDIA GPU to start mining",
    },
    {
      number: "2",
      title: "Start Mining",
      description: "Earn R3MES tokens by contributing compute power",
    },
    {
      number: "3",
      title: "Use AI",
      description: "Use earned credits to access AI inference",
    },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
            R3MES Dashboard
          </h1>

          <p className="text-xl text-slate-300 max-w-3xl mx-auto mb-12">
            R3MES combines blockchain technology with AI model training, enabling
            miners to earn rewards by contributing to decentralized machine learning.
          </p>

          <div className="flex flex-wrap gap-4 justify-center mb-16">
            <Link href="/mine" className="btn-primary text-lg px-8 py-4">
              Start Mining
            </Link>
            <Link href="/chat" className="btn-secondary text-lg px-8 py-4">
              Try AI Chat
            </Link>
            <Link href="/network" className="btn-secondary text-lg px-8 py-4">
              Explore Network
            </Link>
          </div>

          {/* Live Stats Cards */}
          {isLoading && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
              <StatCard
                label="Active Miners"
                value="Loading..."
                icon={<Network className="w-5 h-5" />}
              />
              <StatCard
                label="Total FLOPS"
                value="450 Peta"
                icon={<Zap className="w-5 h-5" />}
              />
              <StatCard
                label="Block Height"
                value="Loading..."
                icon={<Database className="w-5 h-5" />}
              />
            </div>
          )}
          {error && (
            <div className="max-w-4xl mx-auto">
              <div className="card bg-red-900/20 border-red-500/50">
                <p className="text-red-400">
                  {getUserFriendlyError(error)}
                </p>
              </div>
            </div>
          )}
          {stats && !isLoading && !error && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
              <StatCard
                label="Active Miners"
                value={stats.active_miners?.toString() || "0"}
                icon={<Network className="w-5 h-5" />}
              />
              <StatCard
                label="Total Users"
                value={stats.total_users?.toString() || "0"}
                icon={<Database className="w-5 h-5" />}
              />
              <StatCard
                label="Block Height"
                value={stats.block_height?.toString() || "N/A"}
                icon={<Database className="w-5 h-5" />}
              />
            </div>
          )}
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          {features.map((feature, index) => (
            <div
              key={index}
              className="card hover:border-green-500/50 transition-colors"
            >
              <div className="flex items-center gap-4 mb-4">
                <div className="text-green-400">{feature.icon}</div>
                <div>
                  <h3 className="text-xl font-bold">{feature.title}</h3>
                  <p className="text-sm text-slate-400">{feature.subtitle}</p>
                </div>
              </div>
              <p className="text-slate-300">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* How It Works */}
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold mb-12">How It Works</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
            {steps.map((step, index) => (
              <div
                key={index}
                className="card"
              >
                <div className="text-6xl font-bold text-green-400 mb-4">
                  {step.number}
                </div>
                <h3 className="text-2xl font-bold mb-2">{step.title}</h3>
                <p className="text-slate-300">{step.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

