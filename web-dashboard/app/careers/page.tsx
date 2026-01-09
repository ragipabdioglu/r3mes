"use client";

import Link from "next/link";
import { ArrowLeft, Briefcase, MapPin, Clock } from "lucide-react";

export default function CareersPage() {
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
          Careers at R3MES
        </h1>

        <p className="text-lg mb-8" style={{ color: 'var(--text-secondary)' }}>
          Join us in building the future of decentralized AI. We're always looking for talented 
          individuals who are passionate about blockchain and artificial intelligence.
        </p>

        <div className="p-8 rounded-xl text-center" style={{ backgroundColor: 'var(--bg-secondary)' }}>
          <Briefcase className="w-16 h-16 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
          <h2 className="text-2xl font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
            No Open Positions
          </h2>
          <p style={{ color: 'var(--text-secondary)' }}>
            We don't have any open positions at the moment, but we're always interested in 
            hearing from talented people. Feel free to reach out on our Discord!
          </p>
          <a 
            href="https://discord.gg/remes" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-block mt-6 px-6 py-3 rounded-lg font-medium transition-colors"
            style={{ backgroundColor: 'var(--accent-primary)', color: 'white' }}
          >
            Join Our Discord
          </a>
        </div>
      </div>
    </div>
  );
}
