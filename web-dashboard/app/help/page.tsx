"use client";

import { HelpCircle, Book, MessageCircle, Github, Mail, ExternalLink } from "lucide-react";
import Link from "next/link";

const faqItems = [
  {
    question: "What is R3MES?",
    answer: "R3MES is a decentralized AI training network. You can train AI models using your GPU power and earn R3MES tokens.",
  },
  {
    question: "How can I start mining?",
    answer: "Connect your wallet, prepare your GPU, and start mining from the 'Mine' page. The system automatically assigns AI model training tasks.",
  },
  {
    question: "How do I register as a Miner/Serving/Validator/Proposer?",
    answer: "Navigate to the 'Roles' page, select the roles you want to register for, ensure you have sufficient stake (minimum varies by role), and submit the registration transaction. Public roles (Miner, Serving) can be registered immediately. Validator and Proposer roles require authorization.",
  },
  {
    question: "What is the minimum stake required for each role?",
    answer: "Miner and Serving require 1,000 REMES minimum stake. Validator requires 100,000 REMES. Proposer requires 50,000 REMES. These amounts ensure network security and prevent spam.",
  },
  {
    question: "How do I request authorization for Validator/Proposer roles?",
    answer: "Validator and Proposer roles require special authorization. You can request access through governance proposals on the blockchain or by contacting the network administrators. See the role access control documentation for details.",
  },
  {
    question: "How does the credit system work?",
    answer: "You can earn credits by mining or finding blocks. You can spend these credits to use AI chat features.",
  },
  {
    question: "Which GPUs are supported?",
    answer: "CUDA-enabled NVIDIA GPUs are recommended. CPU mode is also available but slower.",
  },
  {
    question: "How is security ensured?",
    answer: "All transactions are recorded and encrypted on the blockchain. Your API keys are securely hashed and stored.",
  },
];

const supportChannels = [
  {
    name: "GitHub",
    icon: <Github className="w-5 h-5" />,
    url: "https://github.com/r3mes/r3mes",
    description: "Issue reporting and contributions",
  },
  {
    name: "Email",
    icon: <Mail className="w-5 h-5" />,
    url: "mailto:support@r3mes.io",
    description: "support@r3mes.io",
  },
  {
    name: "Documentation",
    icon: <Book className="w-5 h-5" />,
    url: "/docs",
    description: "Detailed usage guide",
  },
];

export default function HelpPage() {
  return (
    <div className="min-h-screen py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8" style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      <div className="container mx-auto max-w-full sm:max-w-xl md:max-w-2xl lg:max-w-4xl">
        <div className="text-center mb-8 sm:mb-10 md:mb-12">
          <div className="flex justify-center mb-3 sm:mb-4">
            <HelpCircle className="w-12 h-12 sm:w-14 sm:h-14 md:w-16 md:h-16" style={{ color: 'var(--accent-primary)' }} />
          </div>
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-3 sm:mb-4 gradient-text">Help & Support</h1>
          <p className="text-base sm:text-lg md:text-xl" style={{ color: 'var(--text-secondary)' }}>
            Find answers to your questions here
          </p>
        </div>

        {/* FAQ Section */}
        <div className="card mb-6 sm:mb-7 md:mb-8 p-4 sm:p-5 md:p-6">
          <h2 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-5 md:mb-6 flex items-center gap-2" style={{ color: 'var(--text-primary)' }}>
            <MessageCircle className="w-5 h-5 sm:w-6 sm:h-6" />
            Frequently Asked Questions (FAQ)
          </h2>
          <div className="space-y-4 sm:space-y-5 md:space-y-6">
            {faqItems.map((item, index) => (
              <div key={index} className="border-b pb-4 sm:pb-5 md:pb-6 last:border-0" style={{ borderColor: 'var(--border-color)' }}>
                <h3 className="text-base sm:text-lg font-semibold mb-2" style={{ color: 'var(--accent-primary)' }}>
                  {item.question}
                </h3>
                <p className="text-sm sm:text-base" style={{ color: 'var(--text-secondary)' }}>{item.answer}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Support Channels */}
        <div className="card mb-6 sm:mb-7 md:mb-8 p-4 sm:p-5 md:p-6">
          <h2 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-5 md:mb-6" style={{ color: 'var(--text-primary)' }}>Support Channels</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
            {supportChannels.map((channel, index) => (
              <a
                key={index}
                href={channel.url}
                target={channel.url.startsWith("http") ? "_blank" : undefined}
                rel={channel.url.startsWith("http") ? "noopener noreferrer" : undefined}
                className="card transition-colors cursor-pointer"
              >
                <div className="flex items-center gap-3 mb-3">
                  <div style={{ color: 'var(--accent-primary)' }}>{channel.icon}</div>
                  <h3 className="text-lg font-semibold" style={{ color: 'var(--text-primary)' }}>{channel.name}</h3>
                  {channel.url.startsWith("http") && (
                    <ExternalLink className="w-4 h-4 ml-auto" style={{ color: 'var(--text-secondary)' }} />
                  )}
                </div>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{channel.description}</p>
              </a>
            ))}
          </div>
        </div>

        {/* Quick Links */}
        <div className="card p-4 sm:p-5 md:p-6">
          <h2 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-5 md:mb-6" style={{ color: 'var(--text-primary)' }}>Quick Links</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
            <Link
              href="/mine"
              className="p-4 rounded-lg transition-colors"
              style={{ backgroundColor: 'var(--bg-tertiary)' }}
            >
              <h3 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Start Mining</h3>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                Connect your GPU and start mining
              </p>
            </Link>
            <Link
              href="/wallet"
              className="p-4 rounded-lg transition-colors"
              style={{ backgroundColor: 'var(--bg-tertiary)' }}
            >
              <h3 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>My Wallet</h3>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                View your credits and transaction history
              </p>
            </Link>
            <Link
              href="/settings"
              className="p-4 rounded-lg transition-colors"
              style={{ backgroundColor: 'var(--bg-tertiary)' }}
            >
              <h3 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Settings</h3>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                Manage your application settings
              </p>
            </Link>
            <Link
              href="/chat"
              className="p-4 rounded-lg transition-colors"
              style={{ backgroundColor: 'var(--bg-tertiary)' }}
            >
              <h3 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>AI Chat</h3>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                Chat with the AI assistant
              </p>
            </Link>
            <Link
              href="/roles"
              className="p-4 rounded-lg transition-colors"
              style={{ backgroundColor: 'var(--bg-tertiary)' }}
            >
              <h3 className="font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Role Registration</h3>
              <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                Register and manage your node roles
              </p>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}

