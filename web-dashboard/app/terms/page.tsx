"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";

export default function TermsPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--bg-primary)' }}>
      <div className="container mx-auto px-6 py-12 max-w-4xl">
        <Link 
          href="/"
          className="inline-flex items-center gap-2 text-sm mb-8 hover:text-[var(--accent-primary)] transition-colors"
          style={{ color: 'var(--text-secondary)' }}
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Home
        </Link>

        <h1 className="text-4xl font-bold mb-6" style={{ color: 'var(--text-primary)' }}>
          Terms of Service
        </h1>

        <p className="text-sm mb-8" style={{ color: 'var(--text-muted)' }}>
          Last updated: January 2026
        </p>

        <div className="space-y-8" style={{ color: 'var(--text-secondary)' }}>
          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              1. Acceptance of Terms
            </h2>
            <p>
              By accessing or using the R3MES network and services, you agree to be bound by these 
              Terms of Service. If you do not agree, please do not use our services.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              2. Description of Service
            </h2>
            <p>
              R3MES is a decentralized AI compute network that allows users to contribute GPU resources, 
              participate in AI model training, and earn rewards through the R3MES token.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              3. User Responsibilities
            </h2>
            <ul className="list-disc pl-6 space-y-2">
              <li>You are responsible for securing your wallet and private keys</li>
              <li>You must comply with all applicable laws and regulations</li>
              <li>You must not use the service for illegal activities</li>
              <li>You must not attempt to disrupt or harm the network</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              4. Risks
            </h2>
            <p>
              Cryptocurrency and blockchain technologies involve significant risks. You acknowledge 
              that you understand these risks and accept full responsibility for your participation.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              5. Disclaimer
            </h2>
            <p>
              The R3MES network is provided "as is" without warranties of any kind. We are not 
              responsible for any losses incurred through the use of our services.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              6. Changes to Terms
            </h2>
            <p>
              We reserve the right to modify these terms at any time. Continued use of the service 
              after changes constitutes acceptance of the new terms.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
