"use client";

import Link from "next/link";
import { Github, Twitter, MessageCircle, Send, ExternalLink } from "lucide-react";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    product: [
      { label: "Chat", href: "/chat", description: "AI-powered chat interface" },
      { label: "Mine", href: "/mine", description: "GPU mining dashboard" },
      { label: "Network", href: "/network", description: "Network explorer and stats" },
      { label: "Staking", href: "/staking", description: "Stake tokens and earn rewards" },
    ],
    developers: [
      { label: "Documentation", href: "/docs", description: "Complete developer documentation" },
      { label: "API Reference", href: "/developers/api", description: "REST API documentation" },
      { label: "SDK", href: "/developers/sdk", description: "Software development kit" },
      { label: "GitHub", href: "https://github.com/AquaMystic/R3MES", external: true, description: "Source code repository" },
    ],
    community: [
      { label: "Discord", href: "https://discord.gg/remes", external: true, description: "Join our Discord community" },
      { label: "Twitter", href: "https://twitter.com/remesnetwork", external: true, description: "Follow us on Twitter" },
      { label: "Telegram", href: "https://t.me/remesnetwork", external: true, description: "Join our Telegram group" },
      { label: "Blog", href: "/blog", description: "Latest news and updates" },
    ],
    company: [
      { label: "About", href: "/about", description: "Learn about R3MES" },
      { label: "Careers", href: "/careers", description: "Join our team" },
      { label: "Privacy", href: "/privacy", description: "Privacy policy" },
      { label: "Terms", href: "/terms", description: "Terms of service" },
    ],
  };

  const socialLinks = [
    { 
      icon: <Github className="w-5 h-5" />, 
      href: "https://github.com/AquaMystic/R3MES", 
      label: "GitHub",
      description: "View source code and contribute"
    },
    { 
      icon: <Twitter className="w-5 h-5" />, 
      href: "https://twitter.com/remesnetwork", 
      label: "Twitter",
      description: "Follow us for updates"
    },
    { 
      icon: <MessageCircle className="w-5 h-5" />, 
      href: "https://discord.gg/remes", 
      label: "Discord",
      description: "Join our community chat"
    },
    { 
      icon: <Send className="w-5 h-5" />, 
      href: "https://t.me/remesnetwork", 
      label: "Telegram",
      description: "Get instant updates"
    },
  ];

  return (
    <footer 
      className="border-t"
      style={{ 
        borderColor: 'var(--border-color)', 
        backgroundColor: 'var(--bg-primary)' 
      }}
      role="contentinfo"
      aria-label="Site footer"
    >
      {/* Main Footer Content */}
      <div className="container mx-auto px-6 py-16">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-8 lg:gap-12">
          {/* Brand Column */}
          <div className="col-span-2 md:col-span-1">
            <Link 
              href="/" 
              className="inline-block mb-4 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded"
              aria-label="R3MES home page"
            >
              <span 
                className="text-2xl font-bold"
                style={{ color: 'var(--text-primary)' }}
              >
                R3MES
              </span>
            </Link>
            <p 
              className="text-sm mb-6 max-w-xs"
              style={{ color: 'var(--text-secondary)' }}
            >
              The decentralized compute layer for AI. Train models, earn rewards, build the future.
            </p>
            
            {/* Social Links */}
            <div className="flex gap-3" role="list" aria-label="Social media links">
              {socialLinks.map((social, index) => (
                <a
                  key={index}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-110 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                  style={{ 
                    backgroundColor: 'var(--bg-secondary)',
                    color: 'var(--text-secondary)'
                  }}
                  aria-label={`${social.label}: ${social.description}`}
                  role="listitem"
                >
                  {social.icon}
                  <ExternalLink className="w-3 h-3 absolute -top-1 -right-1 opacity-0" aria-hidden="true" />
                </a>
              ))}
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h3 
              className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: 'var(--text-primary)' }}
              id="footer-product-heading"
            >
              Product
            </h3>
            <nav aria-labelledby="footer-product-heading">
              <ul className="space-y-3" role="list">
                {footerLinks.product.map((link, index) => (
                  <li key={index} role="listitem">
                    <Link
                      href={link.href}
                      className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1 py-0.5"
                      style={{ color: 'var(--text-secondary)' }}
                      aria-label={`${link.label}: ${link.description}`}
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
          </div>

          {/* Developers Links */}
          <div>
            <h3 
              className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: 'var(--text-primary)' }}
              id="footer-developers-heading"
            >
              Developers
            </h3>
            <nav aria-labelledby="footer-developers-heading">
              <ul className="space-y-3" role="list">
                {footerLinks.developers.map((link, index) => (
                  <li key={index} role="listitem">
                    {link.external ? (
                      <a
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1 py-0.5 inline-flex items-center gap-1"
                        style={{ color: 'var(--text-secondary)' }}
                        aria-label={`${link.label}: ${link.description} (opens in new tab)`}
                      >
                        {link.label}
                        <ExternalLink className="w-3 h-3" aria-hidden="true" />
                      </a>
                    ) : (
                      <Link
                        href={link.href}
                        className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1 py-0.5"
                        style={{ color: 'var(--text-secondary)' }}
                        aria-label={`${link.label}: ${link.description}`}
                      >
                        {link.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </nav>
          </div>

          {/* Community Links */}
          <div>
            <h3 
              className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: 'var(--text-primary)' }}
              id="footer-community-heading"
            >
              Community
            </h3>
            <nav aria-labelledby="footer-community-heading">
              <ul className="space-y-3" role="list">
                {footerLinks.community.map((link, index) => (
                  <li key={index} role="listitem">
                    {link.external ? (
                      <a
                        href={link.href}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1 py-0.5 inline-flex items-center gap-1"
                        style={{ color: 'var(--text-secondary)' }}
                        aria-label={`${link.label}: ${link.description} (opens in new tab)`}
                      >
                        {link.label}
                        <ExternalLink className="w-3 h-3" aria-hidden="true" />
                      </a>
                    ) : (
                      <Link
                        href={link.href}
                        className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1 py-0.5"
                        style={{ color: 'var(--text-secondary)' }}
                        aria-label={`${link.label}: ${link.description}`}
                      >
                        {link.label}
                      </Link>
                    )}
                  </li>
                ))}
              </ul>
            </nav>
          </div>

          {/* Company Links */}
          <div>
            <h3 
              className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: 'var(--text-primary)' }}
              id="footer-company-heading"
            >
              Company
            </h3>
            <nav aria-labelledby="footer-company-heading">
              <ul className="space-y-3" role="list">
                {footerLinks.company.map((link, index) => (
                  <li key={index} role="listitem">
                    <Link
                      href={link.href}
                      className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)] focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded px-1 py-0.5"
                      style={{ color: 'var(--text-secondary)' }}
                      aria-label={`${link.label}: ${link.description}`}
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
          </div>
        </div>
      </div>

      {/* Bottom Bar */}
      <div 
        className="border-t"
        style={{ borderColor: 'var(--border-color)' }}
      >
        <div className="container mx-auto px-6 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <p 
              className="text-sm"
              style={{ color: 'var(--text-muted)' }}
              role="contentinfo"
            >
              © {currentYear} R3MES. All rights reserved.
            </p>
            <p 
              className="text-sm"
              style={{ color: 'var(--text-muted)' }}
            >
              Powered by Decentralized GPUs
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
