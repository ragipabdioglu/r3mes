"use client";

import { motion } from "framer-motion";
import { Code, Terminal, Download, CheckCircle } from "lucide-react";
import Link from "next/link";

export default function QuickstartPage() {
  const steps = [
    {
      title: "1. Install Python Miner",
      description: "Install the R3MES Python miner package",
      code: `pip install r3mes`,
      language: "bash",
    },
    {
      title: "2. Run Setup Wizard",
      description: "Configure your miner with the setup wizard",
      code: `r3mes-miner setup`,
      language: "bash",
    },
    {
      title: "3. Start Mining",
      description: "Start your miner and begin earning",
      code: `r3mes-miner start`,
      language: "bash",
    },
  ];

  const goSteps = [
    {
      title: "1. Build Go Node",
      description: "Build the R3MES blockchain node",
      code: `cd remes
make build`,
      language: "bash",
    },
    {
      title: "2. Initialize Chain",
      description: "Initialize your local chain",
      code: `./build/remesd init mynode --chain-id remes-1`,
      language: "bash",
    },
    {
      title: "3. Start Node",
      description: "Start the blockchain node",
      code: `./build/remesd start`,
      language: "bash",
    },
  ];

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <div className="container mx-auto px-4 py-16 max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-12"
        >
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
            Quick Start Guide
          </h1>
          <p className="text-xl text-slate-400">
            Get up and running with R3MES in minutes
          </p>
        </motion.div>

        {/* Python Miner Quickstart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="card mb-8"
        >
          <div className="flex items-center gap-3 mb-6">
            <Code className="w-8 h-8 text-green-400" />
            <h2 className="text-3xl font-bold">Python Miner</h2>
          </div>

          <div className="space-y-6">
            {steps.map((step, index) => (
              <div key={index} className="border-l-2 border-green-500/50 pl-6">
                <div className="flex items-start gap-3 mb-2">
                  <CheckCircle className="w-5 h-5 text-green-400 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <h3 className="text-xl font-bold mb-1">{step.title}</h3>
                    <p className="text-slate-400 mb-3">{step.description}</p>
                    <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto">
                      <code className="text-sm text-slate-300">{step.code}</code>
                    </pre>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Go Node Quickstart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="card mb-8"
        >
          <div className="flex items-center gap-3 mb-6">
            <Terminal className="w-8 h-8 text-cyan-400" />
            <h2 className="text-3xl font-bold">Go Node</h2>
          </div>

          <div className="space-y-6">
            {goSteps.map((step, index) => (
              <div key={index} className="border-l-2 border-cyan-500/50 pl-6">
                <div className="flex items-start gap-3 mb-2">
                  <CheckCircle className="w-5 h-5 text-cyan-400 mt-0.5 flex-shrink-0" />
                  <div className="flex-1">
                    <h3 className="text-xl font-bold mb-1">{step.title}</h3>
                    <p className="text-slate-400 mb-3">{step.description}</p>
                    <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto">
                      <code className="text-sm text-slate-300">{step.code}</code>
                    </pre>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Next Steps */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="card"
        >
          <h2 className="text-2xl font-bold mb-4">Next Steps</h2>
          <ul className="space-y-3">
            <li>
              <Link href="/docs/node-operators/miner-setup" className="text-green-400 hover:text-green-300">
                → Complete Miner Setup Guide
              </Link>
            </li>
            <li>
              <Link href="/docs/web-and-tools/api" className="text-green-400 hover:text-green-300">
                → API Reference Documentation
              </Link>
            </li>
            <li>
              <Link href="/playground" className="text-green-400 hover:text-green-300">
                → Try the API Playground
              </Link>
            </li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
}

