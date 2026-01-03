"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import { Play, Copy, Download, Code2 } from "lucide-react";
import axios from "axios";

// Default API URL for production
const DEFAULT_API_URL = 'https://api.r3mes.network';

// Get API base URL for playground - lazy evaluation to avoid build-time errors
const getApiBaseUrl = (): string => {
  // Check for NEXT_PUBLIC_BACKEND_URL first
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (backendUrl) {
    return backendUrl;
  }
  // Check for NEXT_PUBLIC_API_URL (used by other components)
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  if (apiUrl) {
    return apiUrl;
  }
  // Development fallback
  if (process.env.NODE_ENV === 'development') {
    return "http://localhost:8000";
  }
  // Production fallback to default API URL
  return DEFAULT_API_URL;
};

export default function PlaygroundPage() {
  const [endpoint, setEndpoint] = useState("/health");
  const [method, setMethod] = useState<"GET" | "POST">("GET");
  const [requestBody, setRequestBody] = useState("{}");
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [codeFormat, setCodeFormat] = useState<"curl" | "python" | "javascript">("curl");

  // Get API URL at runtime (not build time)
  const API_BASE_URL = useMemo(() => getApiBaseUrl(), []);

  const handleRequest = async () => {
    setLoading(true);
    setResponse(null);

    try {
      let result;
      if (method === "GET") {
        result = await axios.get(`${API_BASE_URL}${endpoint}`);
      } else {
        result = await axios.post(
          `${API_BASE_URL}${endpoint}`,
          JSON.parse(requestBody)
        );
      }

      setResponse({
        status: result.status,
        headers: result.headers,
        data: result.data,
      });
    } catch (error: any) {
      setResponse({
        error: true,
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
      });
    } finally {
      setLoading(false);
    }
  };

  const generateCode = () => {
    const url = `${API_BASE_URL}${endpoint}`;
    
    switch (codeFormat) {
      case "curl":
        if (method === "GET") {
          return `curl -X GET "${url}"`;
        } else {
          return `curl -X POST "${url}" \\
  -H "Content-Type: application/json" \\
  -d '${requestBody}'`;
        }
      
      case "python":
        if (method === "GET") {
          return `import requests

response = requests.get("${url}")
print(response.json())`;
        } else {
          return `import requests

response = requests.post(
    "${url}",
    json=${requestBody}
)
print(response.json())`;
        }
      
      case "javascript":
        if (method === "GET") {
          return `const response = await fetch("${url}");
const data = await response.json();
console.log(data);`;
        } else {
          return `const response = await fetch("${url}", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(${requestBody}),
});
const data = await response.json();
console.log(data);`;
        }
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
            API Playground
          </h1>
          <p className="text-slate-400">Test R3MES API endpoints interactively</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Request Panel */}
          <div className="card">
            <h2 className="text-2xl font-bold mb-4">Request</h2>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Method</label>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value as "GET" | "POST")}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg"
                >
                  <option value="GET">GET</option>
                  <option value="POST">POST</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium mb-2">Endpoint</label>
                <input
                  type="text"
                  value={endpoint}
                  onChange={(e) => setEndpoint(e.target.value)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg font-mono"
                  placeholder="/health"
                />
              </div>

              {method === "POST" && (
                <div>
                  <label className="block text-sm font-medium mb-2">Request Body (JSON)</label>
                  <textarea
                    value={requestBody}
                    onChange={(e) => setRequestBody(e.target.value)}
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg font-mono h-32"
                    placeholder='{"message": "Hello"}'
                  />
                </div>
              )}

              <button
                onClick={handleRequest}
                disabled={loading}
                className="btn-primary w-full flex items-center justify-center gap-2"
              >
                <Play className="w-5 h-5" />
                {loading ? "Sending..." : "Send Request"}
              </button>
            </div>
          </div>

          {/* Response Panel */}
          <div className="card">
            <h2 className="text-2xl font-bold mb-4">Response</h2>
            
            {response && (
              <div className="space-y-4">
                <div>
                  <span className="text-sm text-slate-400">Status: </span>
                  <span className={`font-bold ${
                    response.error ? "text-red-400" : "text-green-400"
                  }`}>
                    {response.status || "Error"}
                  </span>
                </div>

                <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto text-sm">
                  {JSON.stringify(response.data || response, null, 2)}
                </pre>

                <div className="flex gap-2">
                  <button
                    onClick={() => navigator.clipboard.writeText(JSON.stringify(response, null, 2))}
                    className="btn-secondary flex items-center gap-2"
                  >
                    <Copy className="w-4 h-4" />
                    Copy
                  </button>
                </div>
              </div>
            )}

            {!response && !loading && (
              <div className="text-slate-400 text-center py-8">
                Click "Send Request" to test the API
              </div>
            )}

            {loading && (
              <div className="text-slate-400 text-center py-8">
                <div className="animate-pulse">Loading...</div>
              </div>
            )}
          </div>
        </div>

        {/* Code Generation */}
        <div className="card mt-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">Code Generation</h2>
            <div className="flex gap-2">
              <button
                onClick={() => setCodeFormat("curl")}
                className={`px-4 py-2 rounded-lg ${
                  codeFormat === "curl" ? "bg-green-500" : "bg-slate-800"
                }`}
              >
                cURL
              </button>
              <button
                onClick={() => setCodeFormat("python")}
                className={`px-4 py-2 rounded-lg ${
                  codeFormat === "python" ? "bg-green-500" : "bg-slate-800"
                }`}
              >
                Python
              </button>
              <button
                onClick={() => setCodeFormat("javascript")}
                className={`px-4 py-2 rounded-lg ${
                  codeFormat === "javascript" ? "bg-green-500" : "bg-slate-800"
                }`}
              >
                JavaScript
              </button>
            </div>
          </div>

          <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto text-sm">
            {generateCode()}
          </pre>

          <button
            onClick={() => navigator.clipboard.writeText(generateCode())}
            className="btn-secondary mt-4 flex items-center gap-2"
          >
            <Copy className="w-4 h-4" />
            Copy Code
          </button>
        </div>
      </div>
    </div>
  );
}

