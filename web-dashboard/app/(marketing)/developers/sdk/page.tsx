"use client";

import { motion } from "framer-motion";
import { Code, Download, ExternalLink, Copy } from "lucide-react";
import { useState } from "react";

// Example API URL - in real usage, configure via environment variable
// For documentation/examples, use environment variable or show placeholder
const getExampleApiUrl = (): string => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_BACKEND_URL;
  if (apiUrl) {
    return apiUrl;
  }
  // For documentation, show placeholder instead of localhost
  return process.env.NODE_ENV === 'development' ? "http://localhost:8000" : "https://api.r3mes.network";
};

const EXAMPLE_API_URL = getExampleApiUrl();

export default function SDKPage() {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const sdks = [
    {
      name: "Python SDK",
      description: "Official Python SDK for R3MES",
      install: "pip install r3mes",
      version: "1.0.0",
      example: `from r3mes import R3MESClient

client = R3MESClient(
    rpc_url="${EXAMPLE_API_URL}",
    wallet_address="remes1..."
)

# Send a chat message
response = client.chat("What is R3MES?")
print(response)

# Get user info
user_info = client.get_user_info()
print(f"Credits: {user_info['credits']}")`,
      github: "https://github.com/r3mes/sdk-python",
    },
    {
      name: "JavaScript SDK",
      description: "Official JavaScript/TypeScript SDK for R3MES",
      install: "npm install @r3mes/sdk",
      version: "1.0.0",
      example: `import { R3MESClient } from '@r3mes/sdk';

const client = new R3MESClient({
  rpcUrl: '${EXAMPLE_API_URL}',
  walletAddress: 'remes1...'
});

// Send a chat message
const response = await client.chat('What is R3MES?');
console.log(response);

// Get user info
const userInfo = await client.getUserInfo();
console.log(\`Credits: \${userInfo.credits}\`);`,
      github: "https://github.com/r3mes/sdk-javascript",
    },
    {
      name: "Go SDK",
      description: "Official Go SDK for R3MES",
      install: "go get github.com/r3mes/sdk-go",
      version: "1.0.0",
      example: `package main

import (
    "fmt"
    "github.com/r3mes/sdk-go"
)

func main() {
    client := r3mes.NewClient(
        r3mes.WithRPCURL("${EXAMPLE_API_URL}"),
        r3mes.WithWalletAddress("remes1..."),
    )

    // Send a chat message
    response, err := client.Chat("What is R3MES?")
    if err != nil {
        panic(err)
    }
    fmt.Println(response)

    // Get user info
    userInfo, err := client.GetUserInfo()
    if err != nil {
        panic(err)
    }
    fmt.Printf("Credits: %f\n", userInfo.Credits)
}`,
      github: "https://github.com/r3mes/sdk-go",
    },
  ];

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <div className="container mx-auto px-4 py-16 max-w-5xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-12"
        >
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
            SDK Downloads
          </h1>
          <p className="text-xl text-slate-400">
            Official SDKs for Python, JavaScript, and Go
          </p>
        </motion.div>

        <div className="space-y-8">
          {sdks.map((sdk, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 + index * 0.1 }}
              className="card"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <Code className="w-6 h-6 text-green-400" />
                    <h2 className="text-3xl font-bold">{sdk.name}</h2>
                    <span className="text-sm text-slate-400">v{sdk.version}</span>
                  </div>
                  <p className="text-slate-400 mb-4">{sdk.description}</p>
                </div>
                <a
                  href={sdk.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="btn-secondary flex items-center gap-2"
                >
                  <ExternalLink className="w-4 h-4" />
                  GitHub
                </a>
              </div>

              {/* Installation */}
              <div className="mb-6">
                <h3 className="text-lg font-bold mb-2">Installation</h3>
                <div className="flex items-center gap-3">
                  <code className="bg-slate-800 px-4 py-2 rounded-lg text-green-400 flex-1">
                    {sdk.install}
                  </code>
                  <button
                    onClick={() => copyToClipboard(sdk.install, `install-${index}`)}
                    className="flex items-center gap-2 text-sm text-slate-400 hover:text-green-400 transition-colors"
                  >
                    <Copy className="w-4 h-4" />
                    {copiedCode === `install-${index}` ? "Copied!" : "Copy"}
                  </button>
                </div>
              </div>

              {/* Example */}
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-lg font-bold">Example Usage</h3>
                  <button
                    onClick={() => copyToClipboard(sdk.example, `example-${index}`)}
                    className="flex items-center gap-2 text-sm text-slate-400 hover:text-green-400 transition-colors"
                  >
                    <Copy className="w-4 h-4" />
                    {copiedCode === `example-${index}` ? "Copied!" : "Copy"}
                  </button>
                </div>
                <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto">
                  <code className="text-sm text-slate-300">{sdk.example}</code>
                </pre>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Additional Resources */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.6 }}
          className="card mt-8"
        >
          <h2 className="text-2xl font-bold mb-4">Additional Resources</h2>
          <ul className="space-y-2">
            <li>
              <a href="/docs/web-and-tools/api" className="text-green-400 hover:text-green-300">
                → API Reference Documentation
              </a>
            </li>
            <li>
              <a href="/playground" className="text-green-400 hover:text-green-300">
                → Try the API Playground
              </a>
            </li>
            <li>
              <a href="https://github.com/r3mes" target="_blank" rel="noopener noreferrer" className="text-green-400 hover:text-green-300">
                → GitHub Organization <ExternalLink className="w-4 h-4 inline ml-1" />
              </a>
            </li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
}

