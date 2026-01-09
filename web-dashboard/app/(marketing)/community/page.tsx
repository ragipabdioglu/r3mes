"use client";

import { motion } from "framer-motion";
import { MessageCircle, Github, BookOpen, Users, Calendar, ArrowRight, Twitter, Send } from "lucide-react";
import Link from "next/link";

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0 }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { staggerChildren: 0.1 } }
};

export default function CommunityPage() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
      <HeroSection />
      <SocialLinksSection />
      <EventsSection />
      <ContributeSection />
      <NewsletterSection />
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
            <Users className="w-4 h-4" style={{ color: 'var(--accent-primary)' }} />
            <span className="text-sm font-medium" style={{ color: 'var(--accent-primary)' }}>
              Community
            </span>
          </motion.div>
          
          <motion.h1
            variants={fadeInUp}
            className="text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight mb-6"
          >
            Join the <span className="gradient-text">Movement</span>
          </motion.h1>
          
          <motion.p
            variants={fadeInUp}
            className="text-lg md:text-xl max-w-2xl mx-auto"
            style={{ color: 'var(--text-secondary)' }}
          >
            Connect with developers, miners, validators, and enthusiasts 
            building the future of decentralized AI.
          </motion.p>
        </motion.div>
      </div>
    </section>
  );
}

function SocialLinksSection() {
  const socialLinks = [
    {
      icon: <MessageCircle className="w-8 h-8" />,
      name: "Discord",
      description: "Join our Discord for real-time discussions, support, and community updates",
      url: "https://discord.gg/r3mes",
      members: "12,500+",
      cta: "Join Server"
    },
    {
      icon: <Twitter className="w-8 h-8" />,
      name: "Twitter",
      description: "Follow us for the latest news, announcements, and ecosystem updates",
      url: "https://twitter.com/r3mes",
      members: "45,000+",
      cta: "Follow"
    },
    {
      icon: <Send className="w-8 h-8" />,
      name: "Telegram",
      description: "Join our Telegram group for community discussions and announcements",
      url: "https://t.me/r3mes",
      members: "8,200+",
      cta: "Join Group"
    },
    {
      icon: <Github className="w-8 h-8" />,
      name: "GitHub",
      description: "Contribute to R3MES, report issues, and explore the source code",
      url: "https://github.com/r3mes",
      members: "500+ Stars",
      cta: "View Repo"
    }
  ];

  return (
    <section className="py-24 px-6">
      <div className="container mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">Connect With Us</h2>
          <p className="text-lg" style={{ color: 'var(--text-secondary)' }}>Join our growing community</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {socialLinks.map((link, index) => (
            <motion.a
              key={index}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="group p-8 rounded-2xl transition-all duration-300 hover:scale-[1.02]"
              style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
            >
              <div className="flex items-start gap-6">
                <div 
                  className="w-16 h-16 rounded-2xl flex items-center justify-center shrink-0 group-hover:scale-110 transition-transform"
                  style={{ backgroundColor: 'var(--bg-tertiary)' }}
                >
                  <span style={{ color: 'var(--accent-primary)' }}>{link.icon}</span>
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-2xl font-bold">{link.name}</h3>
                    <span 
                      className="text-sm px-3 py-1 rounded-full"
                      style={{ backgroundColor: 'var(--bg-tertiary)', color: 'var(--text-secondary)' }}
                    >
                      {link.members}
                    </span>
                  </div>
                  <p className="mb-4" style={{ color: 'var(--text-secondary)' }}>{link.description}</p>
                  <span 
                    className="inline-flex items-center gap-2 font-medium group-hover:gap-3 transition-all"
                    style={{ color: 'var(--accent-primary)' }}
                  >
                    {link.cta} <ArrowRight className="w-4 h-4" />
                  </span>
                </div>
              </div>
            </motion.a>
          ))}
        </div>
      </div>
    </section>
  );
}

function EventsSection() {
  const events = [
    {
      title: "Community Call",
      date: "Every Thursday",
      time: "18:00 UTC",
      description: "Weekly community call with updates, demos, and Q&A sessions",
      type: "Recurring"
    },
    {
      title: "Developer Workshop",
      date: "Feb 15, 2025",
      time: "14:00 UTC",
      description: "Hands-on workshop: Building your first R3MES application",
      type: "Workshop"
    },
    {
      title: "Governance Forum",
      date: "Feb 20, 2025",
      time: "16:00 UTC",
      description: "Open discussion on upcoming protocol proposals",
      type: "Governance"
    }
  ];

  return (
    <section className="py-24 px-6" style={{ backgroundColor: 'var(--bg-secondary)' }}>
      <div className="container mx-auto max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-16"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4">Upcoming Events</h2>
          <p className="text-lg" style={{ color: 'var(--text-secondary)' }}>Join us for community events and workshops</p>
        </motion.div>

        <div className="space-y-4">
          {events.map((event, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: index * 0.1 }}
              className="p-6 rounded-2xl flex flex-col md:flex-row md:items-center gap-6"
              style={{ backgroundColor: 'var(--bg-primary)', border: '1px solid var(--border-color)' }}
            >
              <div 
                className="w-14 h-14 rounded-xl flex items-center justify-center shrink-0"
                style={{ backgroundColor: 'var(--bg-secondary)' }}
              >
                <Calendar className="w-6 h-6" style={{ color: 'var(--accent-primary)' }} />
              </div>
              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-3 mb-2">
                  <h3 className="text-xl font-bold">{event.title}</h3>
                  <span 
                    className="text-xs px-2 py-1 rounded-full"
                    style={{ backgroundColor: 'rgba(0,113,227,0.1)', color: 'var(--accent-primary)' }}
                  >
                    {event.type}
                  </span>
                </div>
                <p className="text-sm mb-2" style={{ color: 'var(--text-secondary)' }}>{event.description}</p>
                <div className="flex items-center gap-4 text-sm" style={{ color: 'var(--text-muted)' }}>
                  <span>{event.date}</span>
                  <span>•</span>
                  <span>{event.time}</span>
                </div>
              </div>
              <button 
                className="px-6 py-3 rounded-full font-medium transition-all hover:scale-105 shrink-0"
                style={{ backgroundColor: 'var(--accent-primary)', color: 'white' }}
              >
                Add to Calendar
              </button>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}

function ContributeSection() {
  const ways = [
    {
      icon: <Github className="w-6 h-6" />,
      title: "Code",
      description: "Contribute to our open-source repositories"
    },
    {
      icon: <BookOpen className="w-6 h-6" />,
      title: "Documentation",
      description: "Help improve our docs and tutorials"
    },
    {
      icon: <MessageCircle className="w-6 h-6" />,
      title: "Community",
      description: "Help others and share your knowledge"
    },
    {
      icon: <Users className="w-6 h-6" />,
      title: "Governance",
      description: "Participate in protocol decisions"
    }
  ];

  return (
    <section className="py-24 px-6">
      <div className="container mx-auto max-w-6xl">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Want to <span className="gradient-text">Contribute</span>?
            </h2>
            <p className="text-lg mb-8" style={{ color: 'var(--text-secondary)' }}>
              R3MES is an open-source project. We welcome contributions from developers, 
              researchers, and community members of all skill levels.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link
                href="https://github.com/r3mes"
                target="_blank"
                className="group flex items-center gap-3 px-6 py-3 rounded-full font-semibold transition-all duration-300 hover:scale-105"
                style={{ backgroundColor: 'var(--accent-primary)', color: 'white' }}
              >
                <Github className="w-5 h-5" />
                View on GitHub
              </Link>
              <Link
                href="/docs/contributing"
                className="flex items-center gap-3 px-6 py-3 rounded-full font-semibold transition-all duration-300 hover:scale-105"
                style={{ border: '1px solid var(--border-color)', color: 'var(--text-primary)' }}
              >
                Contributing Guide
              </Link>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="grid grid-cols-2 gap-4"
          >
            {ways.map((way, index) => (
              <div 
                key={index}
                className="p-6 rounded-2xl"
                style={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-color)' }}
              >
                <div 
                  className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
                  style={{ backgroundColor: 'var(--bg-tertiary)' }}
                >
                  <span style={{ color: 'var(--accent-primary)' }}>{way.icon}</span>
                </div>
                <h3 className="text-lg font-bold mb-1">{way.title}</h3>
                <p className="text-sm" style={{ color: 'var(--text-secondary)' }}>{way.description}</p>
              </div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function NewsletterSection() {
  return (
    <section className="py-24 px-6" style={{ backgroundColor: 'var(--bg-secondary)' }}>
      <div className="container mx-auto max-w-2xl text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Stay Updated</h2>
          <p className="mb-8" style={{ color: 'var(--text-secondary)' }}>
            Subscribe to our newsletter for the latest updates, announcements, and community highlights.
          </p>
          
          <form className="flex flex-col sm:flex-row gap-4 max-w-md mx-auto">
            <input
              type="email"
              placeholder="Enter your email"
              className="flex-1 px-6 py-4 rounded-full text-base outline-none transition-all focus:ring-2"
              style={{ 
                backgroundColor: 'var(--bg-primary)', 
                border: '1px solid var(--border-color)',
                color: 'var(--text-primary)'
              }}
            />
            <button 
              type="submit"
              className="px-8 py-4 rounded-full font-semibold transition-all hover:scale-105"
              style={{ backgroundColor: 'var(--accent-primary)', color: 'white' }}
            >
              Subscribe
            </button>
          </form>
          
          <p className="text-sm mt-4" style={{ color: 'var(--text-muted)' }}>
            No spam, unsubscribe anytime.
          </p>
        </motion.div>
      </div>
    </section>
  );
}
