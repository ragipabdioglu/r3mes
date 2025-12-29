"use client";

import Link from "next/link";
import { Github, Twitter, MessageCircle, Send } from "lucide-react";

export default function Footer() {
  const currentYear = new Date().getFullYear();

  const footerLinks = {
    product: [
      { label: "Chat", href: "/chat" },
      { label: "Mine", href: "/mine" },
      { label: "Network", href: "/network" },
      { label: "Staking", href: "/staking" },
    ],
    developers: [
      { label: "Documentation", href: "/docs" },
      { label: "API Reference", href: "/developers/api" },
      { label: "SDK", href: "/developers/sdk" },
      { label: "GitHub", href: "https://github.com/AquaMystic/R3MES", external: true },
    ],
    community: [
      { label: "Discord", href: "https://discord.gg/remes", external: true },
      { label: "Twitter", href: "https://twitter.com/remesnetwork", external: true },
      { label: "Telegram", href: "https://t.me/remesnetwork", external: true },
      { label: "Blog", href: "/blog" },
    ],
    company: [
      { label: "About", href: "/about" },
      { label: "Careers", href: "/careers" },
      { label: "Privacy", href: "/privacy" },
      { label: "Terms", href: "/terms" },
    ],
  };

  const socialLinks = [
    { icon: <Github className="w-5 h-5" />, href: "https://github.com/AquaMystic/R3MES", label: "GitHub" },
    { icon: <Twitter className="w-5 h-5" />, href: "https://twitter.com/remesnetwork", label: "Twitter" },
    { icon: <MessageCircle className="w-5 h-5" />, href: "https://discord.gg/remes", label: "Discord" },
    { icon: <Send className="w-5 h-5" />, href: "https://t.me/remesnetwork", label: "Telegram" },
  ];

  return (
    <footer 
      className="border-t"
      style={{ 
        borderColor: 'var(--border-color)', 
        backgroundColor: 'var(--bg-primary)' 
      }}
    >
      {/* Main Footer Content */}
      <div className="container mx-auto px-6 py-16">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-8 lg:gap-12">
          {/* Brand Column */}
          <div className="col-span-2 md:col-span-1">
            <Link href="/" className="inline-block mb-4">
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
            <div className="flex gap-3">
              {socialLinks.map((social, index) => (
                <a
                  key={index}
                  href={social.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="w-10 h-10 rounded-full flex items-center justify-center transition-all duration-200 hover:scale-110"
                  style={{ 
                    backgroundColor: 'var(--bg-secondary)',
                    color: 'var(--text-secondary)'
                  }}
                  aria-label={social.label}
                >
                  {social.icon}
                </a>
              ))}
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h3 
              className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: 'var(--text-primary)' }}
            >
              Product
            </h3>
            <ul className="space-y-3">
              {footerLinks.product.map((link, index) => (
                <li key={index}>
                  <Link
                    href={link.href}
                    className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)]"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Developers Links */}
          <div>
            <h3 
              className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: 'var(--text-primary)' }}
            >
              Developers
            </h3>
            <ul className="space-y-3">
              {footerLinks.developers.map((link, index) => (
                <li key={index}>
                  {link.external ? (
                    <a
                      href={link.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)]"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      {link.label}
                    </a>
                  ) : (
                    <Link
                      href={link.href}
                      className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)]"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      {link.label}
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          </div>

          {/* Community Links */}
          <div>
            <h3 
              className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: 'var(--text-primary)' }}
            >
              Community
            </h3>
            <ul className="space-y-3">
              {footerLinks.community.map((link, index) => (
                <li key={index}>
                  {link.external ? (
                    <a
                      href={link.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)]"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      {link.label}
                    </a>
                  ) : (
                    <Link
                      href={link.href}
                      className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)]"
                      style={{ color: 'var(--text-secondary)' }}
                    >
                      {link.label}
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          </div>

          {/* Company Links */}
          <div>
            <h3 
              className="text-sm font-semibold uppercase tracking-wider mb-4"
              style={{ color: 'var(--text-primary)' }}
            >
              Company
            </h3>
            <ul className="space-y-3">
              {footerLinks.company.map((link, index) => (
                <li key={index}>
                  <Link
                    href={link.href}
                    className="text-sm transition-colors duration-200 hover:text-[var(--accent-primary)]"
                    style={{ color: 'var(--text-secondary)' }}
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
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
