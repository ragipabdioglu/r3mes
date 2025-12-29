"use client";

import { motion } from "framer-motion";
import { Code, ExternalLink, Copy } from "lucide-react";
import { useState } from "react";

// Get API base URL for documentation
const getApiBaseUrl = (): string => {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_BACKEND_URL;
  if (apiUrl) {
    return apiUrl;
  }
  // For documentation, show placeholder instead of localhost in production
  return process.env.NODE_ENV === 'development' ? "http://localhost:8000" : "https://api.r3mes.network";
};

const API_BASE_URL = getApiBaseUrl();

export default function APIDocsPage() {
  const [copiedEndpoint, setCopiedEndpoint] = useState<string | null>(null);

  const endpoints = [
    {
      method: "POST",
      path: "/chat",
      description: "Send a chat message and receive AI inference response",
      parameters: [
        { name: "message", type: "string", required: true, description: "The chat message" },
        { name: "wallet_address", type: "string", required: true, description: "User wallet address" },
        { name: "adapter", type: "string", required: false, description: "LoRA adapter to use" },
      ],
      example: {
        request: `curl -X POST ${API_BASE_URL}/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "What is R3MES?",
    "wallet_address": "remes1...",
    "adapter": "general"
  }'`,
        response: `{
  "response": "R3MES is a decentralized AI training network...",
  "credits_used": 0.1,
  "adapter_used": "general"
}`,
      },
    },
    {
      method: "GET",
      path: "/user/info/{wallet_address}",
      description: "Get user information and credits",
      parameters: [
        { name: "wallet_address", type: "string", required: true, description: "User wallet address (path parameter)" },
      ],
      example: {
        request: `curl ${API_BASE_URL}/user/info/remes1...`,
        response: `{
  "wallet_address": "remes1...",
  "credits": 100.0,
  "is_miner": true,
  "last_mining_time": "2024-01-01T00:00:00Z"
}`,
      },
    },
    {
      method: "GET",
      path: "/network/stats",
      description: "Get network statistics",
      parameters: [],
      example: {
        request: `curl ${API_BASE_URL}/network/stats`,
        response: `{
  "total_miners": 150,
  "total_validators": 25,
  "network_hashrate": 1000.5,
  "total_stake": 1000000.0,
  "block_height": 12345
}`,
      },
    },
    {
      method: "POST",
      path: "/api-keys/create",
      description: "Create a new API key",
      parameters: [
        { name: "wallet_address", type: "string", required: true, description: "User wallet address" },
        { name: "name", type: "string", required: false, description: "API key name" },
      ],
      example: {
        request: `curl -X POST ${API_BASE_URL}/api-keys/create \\
  -H "Content-Type: application/json" \\
  -d '{
    "wallet_address": "remes1...",
    "name": "My API Key"
  }'`,
        response: `{
  "api_key": "r3mes_abc123...",
  "created_at": "2024-01-01T00:00:00Z"
}`,
      },
    },
  ];

  const copyToClipboard = (text: string, endpoint: string) => {
    navigator.clipboard.writeText(text);
    setCopiedEndpoint(endpoint);
    setTimeout(() => setCopiedEndpoint(null), 2000);
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
            API Reference
          </h1>
          <p className="text-xl text-slate-400">
            Complete API documentation for R3MES backend services
          </p>
        </motion.div>

        {/* Base URL */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="card mb-8"
        >
          <h2 className="text-2xl font-bold mb-4">Base URL</h2>
          <div className="flex items-center gap-3">
            <code className="bg-slate-800 px-4 py-2 rounded-lg text-green-400">
              {API_BASE_URL}
            </code>
            <span className="text-slate-400">(Current Environment)</span>
          </div>
          <p className="text-slate-400 text-sm mt-2">
            Configure via NEXT_PUBLIC_API_URL or NEXT_PUBLIC_BACKEND_URL environment variable
          </p>
        </motion.div>

        {/* Endpoints */}
        <div className="space-y-8">
          {endpoints.map((endpoint, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.3 + index * 0.1 }}
              className="card"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className={`px-3 py-1 rounded text-sm font-bold ${
                      endpoint.method === "POST" ? "bg-green-500/20 text-green-400" :
                      endpoint.method === "GET" ? "bg-blue-500/20 text-blue-400" :
                      "bg-slate-700 text-slate-300"
                    }`}>
                      {endpoint.method}
                    </span>
                    <code className="text-lg text-slate-100">{endpoint.path}</code>
                  </div>
                  <p className="text-slate-400">{endpoint.description}</p>
                </div>
              </div>

              {/* Parameters */}
              {endpoint.parameters.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-bold mb-3">Parameters</h3>
                  <div className="space-y-2">
                    {endpoint.parameters.map((param, paramIndex) => (
                      <div key={paramIndex} className="bg-slate-800 p-3 rounded-lg">
                        <div className="flex items-center gap-2 mb-1">
                          <code className="text-green-400">{param.name}</code>
                          <span className="text-slate-500">({param.type})</span>
                          {param.required && (
                            <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded">
                              required
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-slate-400">{param.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Example */}
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-bold">Example Request</h3>
                    <button
                      onClick={() => copyToClipboard(endpoint.example.request, `${endpoint.method}-${endpoint.path}-request`)}
                      className="flex items-center gap-2 text-sm text-slate-400 hover:text-green-400 transition-colors"
                    >
                      <Copy className="w-4 h-4" />
                      {copiedEndpoint === `${endpoint.method}-${endpoint.path}-request` ? "Copied!" : "Copy"}
                    </button>
                  </div>
                  <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto">
                    <code className="text-sm text-slate-300">{endpoint.example.request}</code>
                  </pre>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-lg font-bold">Example Response</h3>
                    <button
                      onClick={() => copyToClipboard(endpoint.example.response, `${endpoint.method}-${endpoint.path}-response`)}
                      className="flex items-center gap-2 text-sm text-slate-400 hover:text-green-400 transition-colors"
                    >
                      <Copy className="w-4 h-4" />
                      {copiedEndpoint === `${endpoint.method}-${endpoint.path}-response` ? "Copied!" : "Copy"}
                    </button>
                  </div>
                  <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto">
                    <code className="text-sm text-slate-300">{endpoint.example.response}</code>
                  </pre>
                </div>
              </div>
            </motion.div>
          ))}
        </div>

        {/* Additional Resources */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.8 }}
          className="card mt-8"
        >
          <h2 className="text-2xl font-bold mb-4">Additional Resources</h2>
          <ul className="space-y-2">
            <li>
              <a href="/playground" className="text-green-400 hover:text-green-300">
                → Try the API Playground
              </a>
            </li>
            <li>
              <a href="/docs/web-and-tools/api" className="text-green-400 hover:text-green-300">
                → Full API Documentation
              </a>
            </li>
            <li>
              <a href="https://github.com/r3mes" target="_blank" rel="noopener noreferrer" className="text-green-400 hover:text-green-300">
                → GitHub Repository <ExternalLink className="w-4 h-4 inline ml-1" />
              </a>
            </li>
          </ul>
        </motion.div>
      </div>
    </div>
  );
}

