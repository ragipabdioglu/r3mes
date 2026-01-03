"use client";

import Link from "next/link";
import { ArrowLeft, FileText, Calendar } from "lucide-react";

export default function BlogPage() {
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
          R3MES Blog
        </h1>

        <p className="text-lg mb-8" style={{ color: 'var(--text-secondary)' }}>
          Stay updated with the latest news, updates, and insights from the R3MES team.
        </p>

        <div className="p-8 rounded-xl text-center" style={{ backgroundColor: 'var(--bg-secondary)' }}>
          <FileText className="w-16 h-16 mx-auto mb-4" style={{ color: 'var(--text-muted)' }} />
          <h2 className="text-2xl font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>
            Coming Soon
          </h2>
          <p style={{ color: 'var(--text-secondary)' }}>
            We're working on some exciting content. Follow us on Twitter for the latest updates!
          </p>
          <a 
            href="https://twitter.com/remesnetwork" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-block mt-6 px-6 py-3 rounded-lg font-medium transition-colors"
            style={{ backgroundColor: 'var(--accent-primary)', color: 'white' }}
          >
            Follow on Twitter
          </a>
        </div>
      </div>
    </div>
  );
}
