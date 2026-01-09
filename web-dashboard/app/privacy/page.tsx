"use client";

import Link from "next/link";
import { ArrowLeft } from "lucide-react";

export default function PrivacyPage() {
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
          Privacy Policy
        </h1>

        <p className="text-sm mb-8" style={{ color: 'var(--text-muted)' }}>
          Last updated: January 2026
        </p>

        <div className="space-y-8" style={{ color: 'var(--text-secondary)' }}>
          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              1. Information We Collect
            </h2>
            <p className="mb-4">
              R3MES is a decentralized network. We collect minimal information necessary to operate the service:
            </p>
            <ul className="list-disc pl-6 space-y-2">
              <li>Wallet addresses (public blockchain data)</li>
              <li>Transaction history (public blockchain data)</li>
              <li>Usage analytics (anonymized)</li>
            </ul>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              2. How We Use Information
            </h2>
            <p>
              Information is used solely to provide and improve the R3MES network services, 
              including processing transactions, displaying network statistics, and improving user experience.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              3. Data Storage
            </h2>
            <p>
              All blockchain data is stored on the decentralized R3MES network. 
              We do not store personal information on centralized servers.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              4. Third-Party Services
            </h2>
            <p>
              We may use third-party services for analytics and infrastructure. 
              These services have their own privacy policies.
            </p>
          </section>

          <section>
            <h2 className="text-2xl font-semibold mb-4" style={{ color: 'var(--text-primary)' }}>
              5. Contact
            </h2>
            <p>
              For privacy-related questions, please contact us through our Discord community.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
