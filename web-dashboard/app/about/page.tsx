"use client";

import Link from "next/link";
import { ArrowLeft, Users, Cpu, Shield, Globe } from "lucide-react";

export default function AboutPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--bg-primary)' }}>
      <div className="container mx-auto px-6 py-12">
        <Link 
          href="/"
          className="inline-flex items-center gap-2 text-sm mb-8 hover:text-[var(--accent-primary)] transition-colors"
          style={{ color: 'var(--text-secondary)' }}
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>

        <h1 className="text-4xl font-bold mb-6" style={{ color: 'var(--text-primary)' }}>
          About R3MES
        </h1>

        <div className="prose prose-invert max-w-none">
          <p className="text-lg mb-8" style={{ color: 'var(--text-secondary)' }}>
            R3MES is a decentralized AI compute network that enables anyone to contribute GPU power 
            and earn rewards while helping train and serve AI models.
          </p>

          <div className="grid md:grid-cols-2 gap-8 mb-12">
            <div className="p-6 rounded-xl" style={{ backgroundColor: 'var(--bg-secondary)' }}>
              <Users className="w-10 h-10 mb-4" style={{ color: 'var(--accent-primary)' }} />
              <h3 className="text-xl font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Community Driven</h3>
              <p style={{ color: 'var(--text-secondary)' }}>
                Built by the community, for the community. Open source and transparent.
              </p>
            </div>
            <div className="p-6 rounded-xl" style={{ backgroundColor: 'var(--bg-secondary)' }}>
              <Cpu className="w-10 h-10 mb-4" style={{ color: 'var(--accent-primary)' }} />
              <h3 className="text-xl font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Decentralized Compute</h3>
              <p style={{ color: 'var(--text-secondary)' }}>
                Leverage distributed GPU resources for AI training and inference.
              </p>
            </div>
            <div className="p-6 rounded-xl" style={{ backgroundColor: 'var(--bg-secondary)' }}>
              <Shield className="w-10 h-10 mb-4" style={{ color: 'var(--accent-primary)' }} />
              <h3 className="text-xl font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Secure & Trustless</h3>
              <p style={{ color: 'var(--text-secondary)' }}>
                Built on blockchain technology for transparent and secure operations.
              </p>
            </div>
            <div className="p-6 rounded-xl" style={{ backgroundColor: 'var(--bg-secondary)' }}>
              <Globe className="w-10 h-10 mb-4" style={{ color: 'var(--accent-primary)' }} />
              <h3 className="text-xl font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>Global Network</h3>
              <p style={{ color: 'var(--text-secondary)' }}>
                A worldwide network of miners and validators working together.
              </p>
            </div>
          </div>

          <h2 className="text-2xl font-bold mb-4" style={{ color: 'var(--text-primary)' }}>Our Mission</h2>
          <p className="mb-6" style={{ color: 'var(--text-secondary)' }}>
            To democratize AI by creating an open, decentralized infrastructure that allows anyone 
            to participate in the AI revolution, whether as a compute provider or a user.
          </p>
        </div>
      </div>
    </div>
  );
}
