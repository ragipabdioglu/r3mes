"use client";

import { motion } from "framer-motion";
import Link from "next/link";
import { Code, Book, Download, Terminal, ArrowRight, Copy, Check } from "lucide-react";
import { useState } from "react";

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0 }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
};

export default function DevelopersPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      <HeroSection />
      <QuickStartSection />
      <SDKSection />
      <CodeExamplesSection />
      <ResourcesSection />
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
            <Code className="w-4 h-4" style={{ color: 'var(--accent-primary)' }} />
            <span className="text-sm font-medium" style={{ color: 'var(--accent-primary)' }}>
              Developer Resources
            </span>
          </motion.div>
          
          <motion.h1
            variants={fadeInUp}
            className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight mb-6"
          >
            Build on <span className="gradient-text">R3MES</span>
          </motion.h1>
          
          <motion.p
            variants={fadeInUp}
            className="text-lg md:text-xl max-w-2xl mx-auto mb-10"
            style={{ color: 'var(--text-secondary)' }}
          >
            Everything you need to integrate decentralized AI into your applications. 
            SDKs, APIs, and comprehensive documentation.
          </motion.p>
          
          <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              href="/docs"
              className="group flex items-center justify-center gap-3 px-8 py-4 rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105"
              style={{ backgroundColor: 'var(--accent-primary)', color: 'white' }}
            >
              Read the Docs
              <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link
              href="https://github.com/r3mes"
              target="_blank"
              className="flex items-center justify-center gap-3 px-8 py-4 rounded-full font-semibold text-lg transition-all duration-300 hover:scale-105"
              style={{ border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
            >
              View on GitHub
            </Link>
          </motion.div>
        </motion.div>
      </div>
    </section>
  );
}

function QuickStartSection() {
  const [copied, setCopied] = useState(false);
  const installCommand = "pip install r3mes-sdk";

  const copyToClipboard = () => {
    navigator.clipboard.writeText(installCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section className="py-24 px-6" style={{ backgroundColor: 'var(--bg-secondary)' }}>
      <div className="container mx-auto max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Quick Start</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Get started in under 5 minutes</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl p-6 md:p-8"
          style={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
        >
          <div className="flex items-center justify-between mb-4">
            <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Install the SDK</span>
            <button
              onClick={copyToClipboard}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-all"
              style={{ backgroundColor: 'var(--bg-secondary)' }}
            >
              {copied ? <Check className="w-4 h-4" style={{ color: 'var(--success)' }} /> : <Copy className="w-4 h-4" />}
              {copied ? "Copied!" : "Copy"}
            </button>
          </div>
          <div 
            className="font-mono text-lg p-4 rounded-xl"
            style={{ backgroundColor: 'var(--bg-tertiary)' }}
          >
            <span style={{ color: 'var(--text-muted)' }}>$</span>{" "}
            <span style={{ color: 'var(--accent-primary)' }}>{installCommand}</span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function SDKSection() {
  const sdks = [
    {
      icon: "🐍",
      name: "Python",
      description: "Full-featured SDK for Python applications",
      version: "v1.2.0",
      link: "/developers/sdk#python"
    },
    {
      icon: "📦",
      name: "JavaScript",
      description: "TypeScript-first SDK for web and Node.js",
      version: "v1.1.0",
      link: "/developers/sdk#javascript"
    },
    {
      icon: "🔷",
      name: "Go",
      description: "High-performance SDK for Go applications",
      version: "v1.0.0",
      link: "/developers/sdk#go"
    },
    {
      icon: "🦀",
      name: "Rust",
      description: "Native Rust bindings (coming soon)",
      version: "Soon",
      link: "#"
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
          <h2 className="text-3xl md:text-4xl font-bold mb-4">SDKs & Libraries</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Choose your language</p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {sdks.map((sdk, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Link
                href={sdk.link}
                className="block p-6 rounded-2xl h-full transition-all duration-300 hover:scale-[1.02] group"
                style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
              >
                <div className="text-4xl mb-4">{sdk.icon}</div>
                <h3 className="text-xl font-bold mb-2">{sdk.name}</h3>
                <p className="text-sm mb-4" style={{ color: 'var(--text-secondary)' }}>{sdk.description}</p>
                <div className="flex items-center justify-between">
                  <span 
                    className="text-xs px-2 py-1 rounded-full"
                    style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}
                  >
                    {sdk.version}
                  </span>
                  <ArrowRight className="w-4 h-4 opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: 'var(--accent-primary)' }} />
                </div>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function CodeExamplesSection() {
  const [activeTab, setActiveTab] = useState("python");

  const examples = {
    python: {
      title: "Python",
      code: `from r3mes import Client

# Initialize client
client = Client(
    api_key="your-api-key",
    network="mainnet"
)

# Chat with AI
response = client.chat.create(
    model="r3mes-1",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.content)`
    },
    javascript: {
      title: "JavaScript",
      code: `import { R3MESClient } from '@r3mes/sdk';

// Initialize client
const client = new R3MESClient({
  apiKey: 'your-api-key',
  network: 'mainnet'
});

// Chat with AI
const response = await client.chat.create({
  model: 'r3mes-1',
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
});

console.log(response.content);`
    },
    go: {
      title: "Go",
      code: `package main

import "github.com/r3mes/sdk-go"

func main() {
    // Initialize client
    client := r3mes.NewClient(
        r3mes.WithAPIKey("your-api-key"),
        r3mes.WithNetwork("mainnet"),
    )

    // Chat with AI
    response, _ := client.Chat.Create(
        r3mes.ChatRequest{
            Model: "r3mes-1",
            Messages: []r3mes.Message{
                {Role: "user", Content: "Hello!"},
            },
        },
    )

    fmt.Println(response.Content)
}`
    }
  };

  return (
    <section className="py-24 px-6" style={{ backgroundColor: 'var(--bg-secondary)' }}>
      <div className="container mx-auto max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-12"
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Code Examples</h2>
          <p style={{ color: 'var(--text-secondary)' }}>See how easy it is to get started</p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="rounded-2xl overflow-hidden"
          style={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
        >
          {/* Tabs */}
          <div className="flex border-b" style={{ borderColor: 'var(--border-color)' }}>
            {Object.entries(examples).map(([key, value]) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className="px-6 py-4 text-sm font-medium transition-all"
                style={{
                  color: activeTab === key ? 'var(--accent-primary)' : 'var(--text-secondary)',
                  borderBottom: activeTab === key ? '2px solid var(--accent-primary)' : '2px solid transparent'
                }}
              >
                {value.title}
              </button>
            ))}
          </div>

          {/* Code */}
          <div className="p-6">
            <pre 
              className="overflow-x-auto text-sm leading-relaxed"
              style={{ color: 'var(--text-primary)' }}
            >
              <code>{examples[activeTab as keyof typeof examples].code}</code>
            </pre>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function ResourcesSection() {
  const resources = [
    {
      icon: <Book className="w-6 h-6" />,
      title: "Documentation",
      description: "Comprehensive guides and API reference",
      link: "/docs"
    },
    {
      icon: <Terminal className="w-6 h-6" />,
      title: "API Reference",
      description: "Complete API documentation with examples",
      link: "/developers/api"
    },
    {
      icon: <Download className="w-6 h-6" />,
      title: "SDK Downloads",
      description: "Download SDKs for your platform",
      link: "/developers/sdk"
    },
    {
      icon: <Code className="w-6 h-6" />,
      title: "Examples",
      description: "Sample projects and code snippets",
      link: "https://github.com/r3mes/examples"
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
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Resources</h2>
          <p style={{ color: 'var(--text-secondary)' }}>Everything you need to succeed</p>
        </motion.div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {resources.map((resource, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
            >
              <Link
                href={resource.link}
                className="block p-6 rounded-2xl h-full transition-all duration-300 hover:scale-[1.02] group"
                style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
              >
                <div 
                  className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                  style={{ backgroundColor: 'var(--bg-tertiary)' }}
                >
                  <span style={{ color: 'var(--accent-primary)' }}>{resource.icon}</span>
                </div>
                <h3 className="text-lg font-bold mb-2">{resource.title}</h3>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{resource.description}</p>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
