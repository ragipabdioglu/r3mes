"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { Shield, Database, Network, Settings, ArrowRight, Cpu, Lock, Zap, Coins } from "lucide-react";

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0 }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
};

export default function ProtocolPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      <HeroSection />
      <OverviewSection />
      <ArchitectureSection />
      <FeaturesSection />
      <TokenomicsSection />
      <CTASection />
    </div>
  );
}

function HeroSection() {
  return (
    <section className="py-24 md:py-32 px-6 border-b" style={{ borderColor: 'var(--border-color)' }}>
      <div className="container mx-auto max-w-4xl text-center">
        <motion.div initial="hidden" animate="visible" variants={staggerContainer}>
          <motion.div
            variants={fadeInUp}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-8 border"
            style={{ borderColor: 'var(--accent-primary)', backgroundColor: 'rgba(0,113,227,0.1)' }}
          >
            <Network className="w-4 h-4" style={{ color: 'var(--accent-primary)' }} />
            <span className="text-sm font-medium" style={{ color: 'var(--accent-primary)' }}>
              Protocol Design
            </span>
          </motion.div>
          
          <motion.h1
            variants={fadeInUp}
            className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight mb-6"
          >
            The <span className="gradient-text">R3MES</span> Protocol
          </motion.h1>
          
          <motion.p
            variants={fadeInUp}
            className="text-lg md:text-xl max-w-2xl mx-auto mb-10"
            style={{ color: 'var(--text-secondary)' }}
          >
            A decentralized AI training network with verifiable computation, 
            efficient bandwidth usage, and fair economic incentives.
          </motion.p>
          
          <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/docs/protocol-design/architecture"
              className="group flex items-center justify-center gap-3 px-8 py-4 rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105"
              style={{ backgroundColor: 'var(--accent-primary)', color: 'white' }}
            >
              Read Whitepaper
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/docs"
              className="flex items-center justify-center gap-3 px-8 py-4 rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105"
              style={{ border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
            >
              View Documentation
            </Link>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}

function OverviewSection() {
  return (
    <section className="py-24 px-6">
      <div className="container mx-auto max-w-4xl text-center">
        <motion.p
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-2xl md:text-3xl font-light leading-relaxed"
          style={{ color: 'var(--text-secondary)' }}
        >
          R3MES combines <span style={{ color: 'var(--text-primary)' }}>Proof of Useful Work</span> with 
          advanced cryptographic verification to create a trustless, efficient, and 
          <span style={{ color: 'var(--accent-primary)' }}> economically sustainable</span> AI training network.
        </motion.p>
      </div>
    </section>
  );
}

function ArchitectureSection() {
  const layers = [
    {
      icon: <Cpu className="w-6 h-6" />,
      title: "Compute Layer",
      description: "Distributed GPU network for gradient generation and model training",
      items: ["Miner Nodes", "Gradient Generation", "BitNet LoRA Compression"]
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: "Verification Layer",
      description: "Three-layer optimistic verification for trustless computation",
      items: ["Merkle Proofs", "Trap Jobs", "Iron Sandbox"]
    },
    {
      icon: <Database className="w-6 h-6" />,
      title: "Consensus Layer",
      description: "Cosmos SDK-based blockchain for coordination and settlement",
      items: ["CometBFT Consensus", "IBC Compatibility", "On-chain Governance"]
    },
    {
      icon: <Coins className="w-6 h-6" />,
      title: "Economic Layer",
      description: "Fair reward distribution and sustainable tokenomics",
      items: ["Role-based Rewards", "Treasury Buy-back", "Inference Fees"]
    }
  ];

  return (
    <section className="py-24 px-6" style={{ backgroundColor: 'var(--bg-secondary)' }}>
      <div className="container mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">Architecture</h2>
          <p className="text-lg" style={{ color: 'var(--text-secondary)' }}>Four interconnected layers powering decentralized AI</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {layers.map((layer, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="p-8 rounded-2xl"
              style={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
            >
              <div 
                className="w-14 h-14 rounded-xl flex items-center justify-center mb-6"
                style={{ backgroundColor: 'var(--bg-secondary)' }}
              >
                <span style={{ color: 'var(--accent-primary)' }}>{layer.icon}</span>
              </div>
              <h3 className="text-2xl font-bold mb-2">{layer.title}</h3>
              <p className="mb-4" style={{ color: 'var(--text-secondary)' }}>{layer.description}</p>
              <ul className="space-y-2">
                {layer.items.map((item, i) => (
                  <li key={i} className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-secondary)' }}>
                    <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: 'var(--accent-primary)' }} />
                    {item}
                  </li>
                ))}
              </ul>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function FeaturesSection() {
  const features = [
    {
      icon: <Zap className="w-8 h-8" />,
      title: "Proof of Useful Work",
      description: "Miners contribute real AI training work instead of wasteful computations, creating tangible value for the network while earning rewards."
    },
    {
      icon: <Lock className="w-8 h-8" />,
      title: "Three-Layer Verification",
      description: "Optimistic verification with Merkle proofs, random trap jobs, and Iron Sandbox ensure training integrity without sacrificing performance."
    },
    {
      icon: <Database className="w-8 h-8" />,
      title: "BitNet LoRA",
      description: "99.6% bandwidth reduction through BitNet quantization enables efficient federated learning with minimal network overhead."
    },
    {
      icon: <Settings className="w-8 h-8" />,
      title: "On-chain Governance",
      description: "Token holders participate in protocol decisions through proposals and voting, ensuring community-driven development."
    }
  ];

  return (
    <section className="py-24 px-6">
      <div className="container mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">Key Innovations</h2>
          <p className="text-lg" style={{ color: 'var(--text-secondary)' }}>What makes R3MES unique</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="flex gap-6"
            >
              <div 
                className="w-16 h-16 rounded-2xl flex items-center justify-center shrink-0"
                style={{ backgroundColor: 'var(--bg-secondary)' }}
              >
                <span style={{ color: 'var(--accent-primary)' }}>{feature.icon}</span>
              </div>
              <div>
                <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                <p style={{ color: 'var(--text-secondary)' }}>{feature.description}</p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TokenomicsSection() {
  const distribution = [
    { label: "Mining Rewards", value: "40%", color: "var(--accent-primary)" },
    { label: "Ecosystem Fund", value: "25%", color: "var(--success)" },
    { label: "Team & Advisors", value: "15%", color: "var(--warning)" },
    { label: "Community", value: "10%", color: "var(--info)" },
    { label: "Treasury", value: "10%", color: "var(--error)" }
  ];

  return (
    <section className="py-24 px-6" style={{ backgroundColor: 'var(--bg-secondary)' }}>
      <div className="container mx-auto max-w-6xl">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              <span className="gradient-text">R3MES</span> Token
            </h2>
            <p className="text-lg mb-8" style={{ color: 'var(--text-secondary)' }}>
              The R3MES token powers the entire ecosystem - from mining rewards and staking 
              to governance and inference payments.
            </p>
            
            <div className="space-y-4">
              {distribution.map((item, index) => (
                <div key={index} className="flex items-center gap-4">
                  <div 
                    className="w-4 h-4 rounded-full"
                    style={{ backgroundColor: item.color }}
                  />
                  <div className="flex-1">
                    <div className="flex justify-between mb-1">
                      <span style={{ color: 'var(--text-primary)' }}>{item.label}</span>
                      <span className="font-bold">{item.value}</span>
                    </div>
                    <div 
                      className="h-2 rounded-full overflow-hidden"
                      style={{ backgroundColor: 'var(--bg-tertiary)' }}
                    >
                      <div 
                        className="h-full rounded-full"
                        style={{ 
                          width: item.value, 
                          backgroundColor: item.color 
                        }}
                      />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-2 gap-4"
          >
            {[
              { label: "Total Supply", value: "1B" },
              { label: "Initial Circulating", value: "100M" },
              { label: "Block Reward", value: "10 R3MES" },
              { label: "Halving Period", value: "4 Years" }
            ].map((stat, index) => (
              <div 
                key={index}
                className="p-6 rounded-2xl text-center"
                style={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
              >
                <div className="text-3xl font-bold mb-2" style={{ color: 'var(--accent-primary)' }}>{stat.value}</div>
                <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>{stat.label}</div>
              </div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  const sections = [
    {
      title: "Architecture",
      description: "Deep dive into protocol design",
      link: "/docs/protocol-design/architecture",
      icon: <Network className="w-5 h-5" />
    },
    {
      title: "Security",
      description: "Verification and security model",
      link: "/docs/protocol-design/security",
      icon: <Shield className="w-5 h-5" />
    },
    {
      title: "Economics",
      description: "Tokenomics and incentives",
      link: "/docs/protocol-design/economics",
      icon: <Coins className="w-5 h-5" />
    },
    {
      title: "Governance",
      description: "DAO and voting mechanisms",
      link: "/docs/protocol-design/governance",
      icon: <Settings className="w-5 h-5" />
    }
  ];

  return (
    <section className="py-24 px-6">
      <div className="container mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Learn More</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Explore the protocol in depth</p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {sections.map((section, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Link
                href={section.link}
                className="block p-6 rounded-2xl h-full transition-all duration-300 hover:scale-[1.02] group"
                style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
              >
                <div 
                  className="w-10 h-10 rounded-lg flex items-center justify-center mb-4"
                  style={{ backgroundColor: 'var(--bg-tertiary)' }}
                >
                  <span style={{ color: 'var(--accent-primary)' }}>{section.icon}</span>
                </div>
                <h3 className="text-lg font-bold mb-1">{section.title}</h3>
                <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>{section.description}</p>
                <span className="inline-flex items-center gap-2 text-sm font-medium group-hover:gap-3 transition-all" style={{ color: 'var(--accent-primary)' }}>
                  Learn More <ArrowRight className="w-4 h-4" />
                </span>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
