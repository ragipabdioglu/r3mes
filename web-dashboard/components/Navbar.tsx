"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useWallet } from "@/contexts/WalletContext";
import { useTheme } from "@/contexts/ThemeContext";
import WalletButton from "./WalletButton";
import { formatCredits } from "@/utils/numberFormat";
import { Moon, Sun, Menu, X, ChevronDown } from "lucide-react";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";

export default function Navbar() {
  const pathname = usePathname();
  const { walletAddress, userInfo } = useWallet();
  const { actualTheme, toggleTheme } = useTheme();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  // Track scroll for navbar background
  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const isActive = (path: string) => pathname === path;
  const isActiveGroup = (paths: string[]) => paths.some(p => pathname.startsWith(p));

  // Navigation structure
  const navLinks = [
    { href: "/", label: "Home" },
    { href: "/chat", label: "Chat" },
    { href: "/mine", label: "Mine" },
    { href: "/network", label: "Network" },
  ];

  const moreLinks = [
    { href: "/faucet", label: "Faucet" },
    { href: "/serving", label: "Serving" },
    { href: "/proposer", label: "Proposer" },
    { href: "/staking", label: "Staking" },
    { href: "/wallet", label: "Wallet" },
    { href: "/docs", label: "Docs" },
    { href: "/help", label: "Help" },
  ];

  // Keyboard navigation
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && mobileMenuOpen) {
        setMobileMenuOpen(false);
      }
    };

    const handleTab = (e: KeyboardEvent) => {
      if (e.key === 'Tab' && mobileMenuOpen) {
        // Focus trap logic for mobile menu
        const focusableElements = document.querySelectorAll(
          '[data-mobile-menu] button, [data-mobile-menu] a, [data-mobile-menu] [tabindex]:not([tabindex="-1"])'
        );
        const firstElement = focusableElements[0] as HTMLElement;
        const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;

        if (e.shiftKey && document.activeElement === firstElement) {
          e.preventDefault();
          lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
          e.preventDefault();
          firstElement.focus();
        }
      }
    };

    document.addEventListener('keydown', handleEscape);
    document.addEventListener('keydown', handleTab);
    
    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.removeEventListener('keydown', handleTab);
    };
  }, [mobileMenuOpen]);

  // Mobile menu scroll lock
  useEffect(() => {
    if (mobileMenuOpen) {
      document.body.style.overflow = 'hidden';
      // Focus first menu item when opened
      setTimeout(() => {
        const firstMenuItem = document.querySelector('[data-mobile-menu] a') as HTMLElement;
        firstMenuItem?.focus();
      }, 100);
    } else {
      document.body.style.overflow = 'unset';
    }
    
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [mobileMenuOpen]);

  return (
    <>
      {/* Main Navbar - Bottom floating */}
      <motion.nav
        initial={{ y: 100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        className={`fixed bottom-4 sm:bottom-6 left-1/2 -translate-x-1/2 z-50 
                    rounded-full px-3 py-2 sm:px-4 sm:py-2.5 md:px-6 md:py-3
                    transition-all duration-300
                    max-w-[95vw] sm:max-w-[90vw] lg:max-w-fit
                    flex items-center gap-2 sm:gap-3 md:gap-4`}
        style={{
          backgroundColor: scrolled ? 'var(--glass-bg)' : 'var(--glass-bg)',
          backdropFilter: 'blur(20px) saturate(180%)',
          WebkitBackdropFilter: 'blur(20px) saturate(180%)',
          border: '1px solid var(--glass-border)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.12)'
        }}
        role="navigation"
        aria-label="Main navigation"
      >
        {/* Logo */}
        <Link 
          href="/" 
          className="flex items-center shrink-0 group"
          aria-label="R3MES Home"
        >
          <span 
            className="font-bold text-sm sm:text-base md:text-lg transition-colors"
            style={{ color: 'var(--text-primary)' }}
          >
            R3MES
          </span>
        </Link>

        {/* Divider */}
        <div className="h-5 w-px shrink-0" style={{ backgroundColor: 'var(--border-color)' }} />

        {/* Main Links */}
        <div className="hidden sm:flex items-center gap-1 md:gap-2" role="menubar" aria-label="Main navigation">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`px-3 py-1.5 rounded-full text-sm font-medium transition-all duration-200 hover:scale-105 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500`}
              style={{
                color: isActive(link.href) ? 'var(--accent-primary)' : 'var(--text-secondary)',
                backgroundColor: isActive(link.href) ? 'rgba(0,113,227,0.1)' : 'transparent'
              }}
              role="menuitem"
              aria-current={isActive(link.href) ? "page" : undefined}
              tabIndex={0}
            >
              {link.label}
            </Link>
          ))}
        </div>

        {/* More dropdown - Desktop */}
        <div className="hidden md:block relative group">
          <button 
            className="flex items-center gap-1 px-3 py-1.5 rounded-full text-sm font-medium transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            style={{ color: 'var(--text-secondary)' }}
            aria-expanded={false}
            aria-haspopup="menu"
            aria-label="More navigation options"
            tabIndex={0}
          >
            More
            <ChevronDown className="w-3.5 h-3.5" aria-hidden="true" />
          </button>
          
          {/* Dropdown */}
          <div 
            className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 
                       opacity-0 invisible group-hover:opacity-100 group-hover:visible group-focus-within:opacity-100 group-focus-within:visible
                       transition-all duration-200 transform group-hover:translate-y-0 translate-y-2"
            role="menu"
            aria-label="Additional navigation options"
          >
            <div 
              className="rounded-2xl p-2 min-w-[160px]"
              style={{ 
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.12)'
              }}
            >
              {moreLinks.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="block px-4 py-2 rounded-lg text-sm font-medium transition-all hover:scale-[1.02] focus:outline-none focus:ring-2 focus:ring-blue-500"
                  style={{
                    color: isActive(link.href) ? 'var(--accent-primary)' : 'var(--text-secondary)',
                    backgroundColor: isActive(link.href) ? 'rgba(0,113,227,0.1)' : 'transparent'
                  }}
                  role="menuitem"
                  aria-current={isActive(link.href) ? "page" : undefined}
                  tabIndex={0}
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
          className="sm:hidden p-2 rounded-full transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          style={{ 
            backgroundColor: 'var(--bg-tertiary)',
            minWidth: '44px',
            minHeight: '44px'
          }}
          aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
          aria-expanded={mobileMenuOpen}
          aria-controls="mobile-menu"
          tabIndex={0}
        >
          {mobileMenuOpen ? (
            <X className="w-4 h-4" style={{ color: 'var(--text-primary)' }} aria-hidden="true" />
          ) : (
            <Menu className="w-4 h-4" style={{ color: 'var(--text-primary)' }} aria-hidden="true" />
          )}
        </button>

        {/* Divider */}
        <div className="h-5 w-px shrink-0" style={{ backgroundColor: 'var(--border-color)' }} />

        {/* Right side actions */}
        <div className="flex items-center gap-2 shrink-0">
          {/* Credits badge */}
          {walletAddress && userInfo?.credits !== null && userInfo?.credits !== undefined && (
            <span 
              className="hidden sm:inline-flex text-xs px-2.5 py-1 rounded-full font-medium"
              style={{ 
                backgroundColor: 'rgba(0,113,227,0.1)',
                color: 'var(--accent-primary)',
                border: '1px solid rgba(0,113,227,0.2)'
              }}
            >
              {formatCredits(userInfo.credits)}
            </span>
          )}

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-full transition-all duration-200 hover:scale-110 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            style={{ 
              backgroundColor: 'var(--bg-tertiary)',
              minWidth: '44px',
              minHeight: '44px'
            }}
            aria-label={`Switch to ${actualTheme === "dark" ? "light" : "dark"} mode`}
            tabIndex={0}
          >
            {actualTheme === "dark" ? (
              <Sun className="w-4 h-4" style={{ color: 'var(--text-primary)' }} aria-hidden="true" />
            ) : (
              <Moon className="w-4 h-4" style={{ color: 'var(--text-primary)' }} aria-hidden="true" />
            )}
          </button>

          {/* Wallet button */}
          <WalletButton />
        </div>
      </motion.nav>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 sm:hidden"
              style={{ backgroundColor: 'rgba(0,0,0,0.5)' }}
              onClick={() => setMobileMenuOpen(false)}
            />
            
            {/* Menu */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 20 }}
              className="fixed bottom-24 left-4 right-4 z-50 sm:hidden rounded-2xl p-4"
              style={{ 
                backgroundColor: 'var(--bg-primary)',
                border: '1px solid var(--border-color)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.2)'
              }}
              id="mobile-menu"
              data-mobile-menu
              role="menu"
              aria-label="Mobile navigation menu"
            >
              <div className="grid grid-cols-2 gap-2">
                {[...navLinks, ...moreLinks].map((link, index) => (
                  <Link
                    key={link.href}
                    href={link.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className="px-4 py-3 rounded-xl text-sm font-medium text-center transition-all focus:outline-none focus:ring-2 focus:ring-blue-500"
                    style={{
                      color: isActive(link.href) ? 'var(--accent-primary)' : 'var(--text-secondary)',
                      backgroundColor: isActive(link.href) ? 'rgba(0,113,227,0.1)' : 'var(--bg-secondary)',
                      minHeight: '44px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center'
                    }}
                    role="menuitem"
                    aria-current={isActive(link.href) ? "page" : undefined}
                    tabIndex={0}
                  >
                    {link.label}
                  </Link>
                ))}
              </div>
              
              {/* Screen reader instructions */}
              <div className="sr-only">
                Use arrow keys to navigate menu items, Enter to select, or Escape to close menu.
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
}
