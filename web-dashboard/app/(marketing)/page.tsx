"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import { useNetworkStats } from "@/hooks/useNetworkStats";
import { Zap, Shield, ArrowRight, Cpu, Coins } from "lucide-react";
import { motion } from "framer-motion";

// Lazy load Three.js components
const HeroScene = dynamic(() => import("@/components/marketing/HeroScene"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center" style={{ backgroundColor: 'var(--bg-primary)' }}>
      <div className="w-16 h-16 border-2 border-[var(--accent-primary)] border-t-transparent rounded-full animate-spin" />
    </div>
  )
});

// Animation variants
const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0 }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

export default function MarketingLanding() {
  const { data: stats } = useNetworkStats();

  return (
    <div className="relative min-h-screen overflow-hidden" style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      <HeroSection />
      <LiveStatsBar stats={stats} />
      <MissionSection />
      <FeaturesSection />
      <HowItWorksSection />
      <TechnologySection />
      <CTASection />
    </div>
  );
}

function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center">
      <div className="absolute inset-0 z-0">
        <HeroScene />
      </div>
      <div 
        className="absolute inset-0 z-[1]"
        style={{ background: 'radial-gradient(circle at center, transparent 0%, var(--bg-primary) 70%)' }}
      />
      <div className="relative z-10 text-center max-w-5xl px-6 py-20">
        <motion.div initial="hidden" animate="visible" variants={staggerContainer}>
          <motion.div
            variants={fadeInUp}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full mb-8 border"
            style={{ borderColor: 'var(--accent-primary)', backgroundColor: 'rgba(0,113,227,0.1)' }}
          >
            <span className="w-2 h-2 rounded-full bg-[var(--success)] animate-pulse" />
            <span className="text-sm font-medium" style={{ color: 'var(--accent-primary)' }}>
              Network Live
            </span>
          </motion.div>
          <motion.h1
            variants={fadeInUp}
            className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-bold tracking-tight mb-6"
            style={{ color: 'var(--text-primary)' }}
          >
            The Compute Layer
            <br />
            <span className="gradient-text">of AI</span>
          </motion.h1>
          <motion.p
            variants={fadeInUp}
            className="text-lg sm:text-xl md:text-2xl font-light max-w-2xl mx-auto mb-10"
            style={{ color: 'var(--text-secondary)' }}
          >
            We network together the core resources required for machine intelligence 
            to flourish alongside human intelligence.
          </motion.p>
          <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <Link
              href="/chat"
              className="group flex items-center gap-3 px-8 py-4 rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105"
              style={{ backgroundColor: 'var(--accent-primary)', color: 'white' }}
            >
              Start Building
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="/mine"
              className="flex items-center gap-3 px-8 py-4 rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105 glass"
              style={{ border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
            >
              Contribute GPU
            </Link>
          </motion.div>
        </motion.div>
      </div>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 z-10"
      >
        <div className="w-6 h-10 rounded-full border-2 flex justify-center pt-2" style={{ borderColor: 'var(--border-color)' }}>
          <motion.div
            animate={{ y: [0, 12, 0] }}
            transition={{ duration: 1.5, repeat: Infinity }}
            className="w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: 'var(--accent-primary)' }}
          />
        </div>
      </motion.div>
    </section>
  );
}

function LiveStatsBar({ stats }: { stats: any }) {
  const statItems = [
    { label: "Total Miners", value: stats?.totalMiners || "2,847" },
    { label: "Active Jobs", value: stats?.activeJobs || "1,234" },
    { label: "Models Trained", value: stats?.modelsTrained || "156" },
    { label: "Total Rewards", value: stats?.totalRewards || "12.5M R3MES" },
  ];

  return (
    <section className="py-6 border-y" style={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)' }}>
      <div className="container mx-auto px-6">
        <div className="flex flex-wrap justify-center gap-8 md:gap-16">
          {statItems.map((item, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="text-center"
            >
              <div className="text-2xl md:text-3xl font-bold" style={{ color: 'var(--text-primary)' }}>{item.value}</div>
              <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>{item.label}</div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function MissionSection() {
  return (
    <section className="py-24 md:py-32 px-6">
      <div className="container mx-auto max-w-4xl text-center">
        <motion.p
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-2xl sm:text-3xl md:text-4xl font-light leading-relaxed"
          style={{ color: 'var(--text-secondary)' }}
        >
          R3MES enables <span style={{ color: 'var(--text-primary)' }}>verifiable, decentralized AI training</span> by 
          connecting GPU providers worldwide. Train models with cryptographic guarantees, 
          earn rewards for your compute, and build the future of <span style={{ color: 'var(--accent-primary)' }}>open AI</span>.
        </motion.p>
      </div>
    </section>
  );
}

function FeaturesSection() {
  const features = [
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Build",
      subtitle: "Create apps over decentralized AI primitives",
      description: "Access powerful AI models through our SDK. Build applications that leverage distributed compute.",
      link: "/developers",
      linkText: "Learn More"
    },
    {
      icon: <Cpu className="w-8 h-8" />,
      title: "Contribute",
      subtitle: "Help train and evaluate better models",
      description: "Connect your GPU to the network. Earn R3MES tokens by contributing compute power.",
      link: "/mine",
      linkText: "Learn More"
    },
    {
      icon: <Coins className="w-8 h-8" />,
      title: "Compete",
      subtitle: "Put your convictions to the test",
      description: "Stake on model performance, participate in governance, and earn rewards.",
      link: "/staking",
      linkText: "Learn More"
    },
  ];

  return (
    <section className="py-24 md:py-32 px-6" style={{ backgroundColor: 'var(--bg-secondary)' }}>
      <div className="container mx-auto max-w-6xl">
        <motion.div initial="hidden" whileInView="visible" viewport={{ once: true }} variants={staggerContainer} className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              variants={fadeInUp}
              className="group p-8 rounded-2xl transition-all duration-300 hover:scale-[1.02]"
              style={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
            >
              <div className="w-14 h-14 rounded-xl flex items-center justify-center mb-6" style={{ backgroundColor: 'var(--bg-secondary)' }}>
                <span style={{ color: 'var(--accent-primary)' }}>{feature.icon}</span>
              </div>
              <h3 className="text-2xl font-bold mb-2" style={{ color: 'var(--text-primary)' }}>{feature.title}</h3>
              <p className="text-sm font-medium mb-4" style={{ color: 'var(--accent-primary)' }}>{feature.subtitle}</p>
              <p className="mb-6" style={{ color: 'var(--text-secondary)' }}>{feature.description}</p>
              <Link href={feature.link} className="inline-flex items-center gap-2 font-medium group-hover:gap-3 transition-all" style={{ color: 'var(--accent-primary)' }}>
                {feature.linkText}
                <ArrowRight className="w-4 h-4" />
              </Link>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

function HowItWorksSection() {
  const steps = [
    { number: "01", title: "Connect", description: "Set up a miner node and connect your GPU to the R3MES network." },
    { number: "02", title: "Compute", description: "Receive training jobs, generate gradients, and submit proofs on-chain." },
    { number: "03", title: "Earn", description: "Get rewarded in R3MES tokens for valid contributions." }
  ];

  return (
    <section className="py-24 md:py-32 px-6">
      <div className="container mx-auto max-w-6xl">
        <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>How It Works</h2>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: 'var(--text-secondary)' }}>Three simple steps to start contributing</p>
        </motion.div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-4">
          {steps.map((step, index) => (
            <motion.div key={index} initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.15 }} className="relative text-center md:text-left">
              {index < steps.length - 1 && <div className="hidden md:block absolute top-8 left-[60%] w-[80%] h-px" style={{ backgroundColor: 'var(--border-color)' }} />}
              <div className="text-6xl md:text-7xl font-bold mb-4" style={{ color: 'var(--accent-primary)', opacity: 0.2 }}>{step.number}</div>
              <h3 className="text-2xl font-bold mb-3" style={{ color: 'var(--text-primary)' }}>{step.title}</h3>
              <p style={{ color: 'var(--text-secondary)' }}>{step.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TechnologySection() {
  return (
    <section className="py-24 md:py-32 px-6" style={{ backgroundColor: 'var(--bg-secondary)' }}>
      <div className="container mx-auto max-w-6xl">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <motion.div initial={{ opacity: 0, x: -30 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }}>
            <h2 className="text-4xl md:text-5xl font-bold mb-6" style={{ color: 'var(--text-primary)' }}>
              BitNet LoRA<br /><span className="gradient-text">200MB Adapters</span>
            </h2>
            <p className="text-lg mb-8" style={{ color: 'var(--text-secondary)' }}>
              R3MES dramatically reduces bandwidth requirements by compressing 50GB models into ~200MB adapters.
            </p>
            <div className="space-y-4">
              {[
                { label: "250x smaller", desc: "Model compression" },
                { label: "10x faster", desc: "Sync times" },
                { label: "99.9%", desc: "Accuracy retention" }
              ].map((stat, index) => (
                <div key={index} className="flex items-center gap-4">
                  <div className="w-12 h-12 rounded-lg flex items-center justify-center" style={{ backgroundColor: 'rgba(0,113,227,0.1)' }}>
                    <Zap className="w-6 h-6" style={{ color: 'var(--accent-primary)' }} />
                  </div>
                  <div>
                    <div className="font-bold" style={{ color: 'var(--text-primary)' }}>{stat.label}</div>
                    <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>{stat.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
          <motion.div initial={{ opacity: 0, x: 30 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} className="grid grid-cols-2 gap-4">
            <div className="p-6 rounded-2xl text-center" style={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}>
              <div className="text-5xl mb-4">📦</div>
              <div className="text-3xl font-bold mb-2" style={{ color: 'var(--error)' }}>50 GB</div>
              <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>Traditional Model</div>
            </div>
            <div className="p-6 rounded-2xl text-center" style={{ backgroundColor: 'var(--bg-primary)', border: '2px solid var(--accent-primary)' }}>
              <div className="text-5xl mb-4">⚡</div>
              <div className="text-3xl font-bold mb-2" style={{ color: 'var(--accent-primary)' }}>200 MB</div>
              <div className="text-sm" style={{ color: 'var(--text-secondary)' }}>BitNet Adapter</div>
            </div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function CTASection() {
  return (
    <section className="py-24 md:py-32 px-6">
      <div className="container mx-auto max-w-4xl text-center">
        <motion.div initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-6" style={{ color: 'var(--text-primary)' }}>
            Ready to build the future of AI?
          </h2>
          <p className="text-lg md:text-xl mb-10 max-w-2xl mx-auto" style={{ color: 'var(--text-secondary)' }}>
            Join thousands of developers and GPU providers building the decentralized compute layer.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/docs" className="group flex items-center justify-center gap-3 px-8 py-4 rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105" style={{ backgroundColor: 'var(--accent-primary)', color: 'white' }}>
              Read the Docs
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link href="https://discord.gg/r3mes" target="_blank" rel="noopener noreferrer" className="flex items-center justify-center gap-3 px-8 py-4 rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105" style={{ border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}>
              Join Discord
            </Link>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
