"use client";

import { useState, useMemo, useEffect } from "react";
import { motion } from "framer-motion";
import { 
  Play, 
  Copy, 
  Code2, 
  Zap, 
  Activity, 
  Settings2,
  StopCircle,
  RefreshCw,
  Brain,
  Database,
  Cpu,
  CheckCircle,
  XCircle,
  AlertCircle,
} from "lucide-react";
import axios from "axios";
import { useInference } from "@/hooks/useInference";

// Default API URL for production
const DEFAULT_API_URL = 'https://api.r3mes.network';

// Get API base URL for playground
const getApiBaseUrl = (): string => {
  const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL;
  if (backendUrl) return backendUrl;
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  if (apiUrl) return apiUrl;
  if (process.env.NODE_ENV === 'development') return "http://localhost:8000";
  return DEFAULT_API_URL;
};

type TabType = "inference" | "api";

export default function PlaygroundPage() {
  const [activeTab, setActiveTab] = useState<TabType>("inference");
  
  // API Playground state
  const [endpoint, setEndpoint] = useState("/health");
  const [method, setMethod] = useState<"GET" | "POST">("GET");
  const [requestBody, setRequestBody] = useState("{}");
  const [response, setResponse] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [codeFormat, setCodeFormat] = useState<"curl" | "python" | "javascript">("curl");

  // Inference state
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState(512);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [skipRag, setSkipRag] = useState(false);
  const [useStreaming, setUseStreaming] = useState(true);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const {
    isLoading: inferenceLoading,
    isStreaming,
    streamedText,
    response: inferenceResponse,
    error: inferenceError,
    health,
    metrics,
    generate,
    generateStream,
    stopStream,
    refreshHealth,
    refreshMetrics,
    warmup,
    clearError,
  } = useInference({
    autoFetchHealth: true,
    healthPollingInterval: 30000, // 30 seconds
  });

  const API_BASE_URL = useMemo(() => getApiBaseUrl(), []);

  // Fetch metrics on mount
  useEffect(() => {
    refreshMetrics();
  }, [refreshMetrics]);

  const handleApiRequest = async () => {
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

  const handleInference = async () => {
    if (!prompt.trim()) return;
    clearError();

    const options = {
      max_tokens: maxTokens,
      temperature,
      top_p: topP,
      skip_rag: skipRag,
    };

    if (useStreaming) {
      await generateStream(prompt, options);
    } else {
      await generate(prompt, options);
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

  const getStatusIcon = (isOk: boolean | undefined) => {
    if (isOk === undefined) return <AlertCircle className="w-4 h-4 text-yellow-400" />;
    return isOk 
      ? <CheckCircle className="w-4 h-4 text-green-400" />
      : <XCircle className="w-4 h-4 text-red-400" />;
  };

  const getStatusColor = (status: string | undefined) => {
    switch (status) {
      case 'ready':
      case 'healthy':
      case 'mock':
        return 'text-green-400';
      case 'remote':
        return 'text-blue-400';
      case 'disabled':
      case 'unavailable':
        return 'text-red-400';
      default:
        return 'text-yellow-400';
    }
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
            R3MES Playground
          </h1>
          <p className="text-slate-400">Test AI inference and API endpoints interactively</p>
        </motion.div>

        {/* Tab Navigation */}
        <div className="flex gap-2 mb-6">
          <button
            onClick={() => setActiveTab("inference")}
            className={`px-6 py-3 rounded-lg flex items-center gap-2 transition-colors ${
              activeTab === "inference" 
                ? "bg-green-500 text-white" 
                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
            }`}
          >
            <Brain className="w-5 h-5" />
            AI Inference
          </button>
          <button
            onClick={() => setActiveTab("api")}
            className={`px-6 py-3 rounded-lg flex items-center gap-2 transition-colors ${
              activeTab === "api" 
                ? "bg-green-500 text-white" 
                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
            }`}
          >
            <Code2 className="w-5 h-5" />
            API Explorer
          </button>
        </div>

        {/* Inference Tab */}
        {activeTab === "inference" && (
          <div className="space-y-6">
            {/* Health Status Bar */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="bg-slate-800/50 rounded-lg p-4 flex flex-wrap items-center gap-6"
            >
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-slate-400" />
                <span className="text-sm text-slate-400">Status:</span>
                <span className={`font-medium ${getStatusColor(health?.status)}`}>
                  {health?.status || 'Unknown'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                {getStatusIcon(health?.is_ready)}
                <span className="text-sm text-slate-400">Ready</span>
              </div>
              <div className="flex items-center gap-2">
                {getStatusIcon(health?.pipeline_initialized)}
                <span className="text-sm text-slate-400">Pipeline</span>
              </div>
              <div className="flex items-center gap-2">
                {getStatusIcon(health?.model_loaded)}
                <span className="text-sm text-slate-400">Model</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-sm text-slate-400">Mode:</span>
                <span className="text-sm font-mono text-cyan-400">
                  {health?.inference_mode || 'N/A'}
                </span>
              </div>
              <button
                onClick={refreshHealth}
                className="ml-auto p-2 hover:bg-slate-700 rounded-lg transition-colors"
                title="Refresh health"
              >
                <RefreshCw className="w-4 h-4 text-slate-400" />
              </button>
            </motion.div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Input Panel */}
              <div className="card">
                <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                  <Zap className="w-6 h-6 text-green-400" />
                  Inference Input
                </h2>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Prompt</label>
                    <textarea
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      className="w-full px-4 py-3 bg-slate-800 border border-slate-700 rounded-lg h-40 resize-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                      placeholder="Enter your prompt here... (e.g., 'Explain how R3MES uses BitNet for efficient AI inference')"
                    />
                  </div>

                  {/* Quick Options */}
                  <div className="flex flex-wrap gap-3">
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={useStreaming}
                        onChange={(e) => setUseStreaming(e.target.checked)}
                        className="w-4 h-4 rounded bg-slate-700 border-slate-600 text-green-500 focus:ring-green-500"
                      />
                      <span className="text-sm">Streaming</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={skipRag}
                        onChange={(e) => setSkipRag(e.target.checked)}
                        className="w-4 h-4 rounded bg-slate-700 border-slate-600 text-green-500 focus:ring-green-500"
                      />
                      <span className="text-sm">Skip RAG</span>
                    </label>
                    <button
                      onClick={() => setShowAdvanced(!showAdvanced)}
                      className="text-sm text-slate-400 hover:text-white flex items-center gap-1"
                    >
                      <Settings2 className="w-4 h-4" />
                      {showAdvanced ? 'Hide' : 'Show'} Advanced
                    </button>
                  </div>

                  {/* Advanced Options */}
                  {showAdvanced && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      className="space-y-4 pt-4 border-t border-slate-700"
                    >
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-sm font-medium mb-2">
                            Max Tokens: {maxTokens}
                          </label>
                          <input
                            type="range"
                            min="64"
                            max="4096"
                            value={maxTokens}
                            onChange={(e) => setMaxTokens(Number(e.target.value))}
                            className="w-full"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium mb-2">
                            Temperature: {temperature.toFixed(2)}
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="2"
                            step="0.1"
                            value={temperature}
                            onChange={(e) => setTemperature(Number(e.target.value))}
                            className="w-full"
                          />
                        </div>
                        <div>
                          <label className="block text-sm font-medium mb-2">
                            Top-P: {topP.toFixed(2)}
                          </label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={topP}
                            onChange={(e) => setTopP(Number(e.target.value))}
                            className="w-full"
                          />
                        </div>
                      </div>
                    </motion.div>
                  )}

                  {/* Action Buttons */}
                  <div className="flex gap-3">
                    <button
                      onClick={handleInference}
                      disabled={inferenceLoading || !prompt.trim()}
                      className="btn-primary flex-1 flex items-center justify-center gap-2 disabled:opacity-50"
                    >
                      {inferenceLoading ? (
                        <>
                          <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full" />
                          {isStreaming ? 'Streaming...' : 'Generating...'}
                        </>
                      ) : (
                        <>
                          <Play className="w-5 h-5" />
                          Generate
                        </>
                      )}
                    </button>
                    {isStreaming && (
                      <button
                        onClick={stopStream}
                        className="px-4 py-2 bg-red-500 hover:bg-red-600 rounded-lg flex items-center gap-2"
                      >
                        <StopCircle className="w-5 h-5" />
                        Stop
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Output Panel */}
              <div className="card">
                <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                  <Brain className="w-6 h-6 text-cyan-400" />
                  Response
                </h2>
                
                {inferenceError && (
                  <div className="mb-4 p-4 bg-red-500/20 border border-red-500/50 rounded-lg text-red-300">
                    {inferenceError}
                  </div>
                )}

                {(streamedText || inferenceResponse) ? (
                  <div className="space-y-4">
                    <div className="bg-slate-800 p-4 rounded-lg min-h-[200px] max-h-[400px] overflow-y-auto">
                      <pre className="whitespace-pre-wrap text-sm">
                        {streamedText || inferenceResponse?.text}
                        {isStreaming && <span className="animate-pulse">▊</span>}
                      </pre>
                    </div>

                    {inferenceResponse && (
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div className="bg-slate-800/50 p-3 rounded-lg">
                          <span className="text-slate-400">Tokens:</span>
                          <span className="ml-2 font-mono">{inferenceResponse.tokens_generated}</span>
                        </div>
                        <div className="bg-slate-800/50 p-3 rounded-lg">
                          <span className="text-slate-400">Latency:</span>
                          <span className="ml-2 font-mono">{inferenceResponse.latency_ms.toFixed(0)}ms</span>
                        </div>
                        <div className="bg-slate-800/50 p-3 rounded-lg">
                          <span className="text-slate-400">RAG:</span>
                          <span className="ml-2">{inferenceResponse.rag_context_used ? '✓ Used' : '✗ Skipped'}</span>
                        </div>
                        <div className="bg-slate-800/50 p-3 rounded-lg">
                          <span className="text-slate-400">Experts:</span>
                          <span className="ml-2 font-mono">{inferenceResponse.experts_used.length}</span>
                        </div>
                      </div>
                    )}

                    <button
                      onClick={() => navigator.clipboard.writeText(streamedText || (inferenceResponse?.text ?? ''))}
                      className="btn-secondary flex items-center gap-2"
                    >
                      <Copy className="w-4 h-4" />
                      Copy Response
                    </button>
                  </div>
                ) : (
                  <div className="text-slate-400 text-center py-16">
                    <Brain className="w-16 h-16 mx-auto mb-4 opacity-30" />
                    <p>Enter a prompt and click Generate to test inference</p>
                  </div>
                )}
              </div>
            </div>

            {/* Metrics Panel */}
            {metrics && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="card"
              >
                <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                  <Cpu className="w-5 h-5 text-purple-400" />
                  Engine Metrics
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-xs text-slate-400 mb-1">Total Requests</div>
                    <div className="text-xl font-mono">{metrics.serving_engine_requests_total}</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-xs text-slate-400 mb-1">Success</div>
                    <div className="text-xl font-mono text-green-400">{metrics.serving_engine_requests_success}</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-xs text-slate-400 mb-1">Failed</div>
                    <div className="text-xl font-mono text-red-400">{metrics.serving_engine_requests_failed}</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-xs text-slate-400 mb-1">Avg Latency</div>
                    <div className="text-xl font-mono">{metrics.serving_engine_latency_avg_ms.toFixed(0)}ms</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-xs text-slate-400 mb-1">Cache Hits</div>
                    <div className="text-xl font-mono text-cyan-400">{metrics.cache_hits}</div>
                  </div>
                  <div className="bg-slate-800/50 p-3 rounded-lg">
                    <div className="text-xs text-slate-400 mb-1">VRAM Used</div>
                    <div className="text-xl font-mono">{metrics.cache_vram_used_mb.toFixed(0)}MB</div>
                  </div>
                </div>
              </motion.div>
            )}
          </div>
        )}

        {/* API Explorer Tab */}
        {activeTab === "api" && (
          <div className="space-y-6">
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
                    onClick={handleApiRequest}
                    disabled={loading}
                    className="btn-primary w-full flex items-center justify-center gap-2"
                  >
                    <Play className="w-5 h-5" />
                    {loading ? "Sending..." : "Send Request"}
                  </button>
                </div>

                {/* Quick Endpoints */}
                <div className="mt-6 pt-4 border-t border-slate-700">
                  <h3 className="text-sm font-medium mb-3 text-slate-400">Quick Endpoints</h3>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { path: '/health', method: 'GET' },
                      { path: '/api/inference/health', method: 'GET' },
                      { path: '/api/inference/metrics', method: 'GET' },
                      { path: '/faucet/status', method: 'GET' },
                      { path: '/analytics', method: 'GET' },
                    ].map((ep) => (
                      <button
                        key={ep.path}
                        onClick={() => {
                          setEndpoint(ep.path);
                          setMethod(ep.method as "GET" | "POST");
                        }}
                        className="px-3 py-1 text-xs bg-slate-700 hover:bg-slate-600 rounded-lg font-mono"
                      >
                        {ep.path}
                      </button>
                    ))}
                  </div>
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

                    <pre className="bg-slate-800 p-4 rounded-lg overflow-x-auto text-sm max-h-[400px] overflow-y-auto">
                      {JSON.stringify(response.data || response, null, 2)}
                    </pre>

                    <button
                      onClick={() => navigator.clipboard.writeText(JSON.stringify(response, null, 2))}
                      className="btn-secondary flex items-center gap-2"
                    >
                      <Copy className="w-4 h-4" />
                      Copy
                    </button>
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
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold">Code Generation</h2>
                <div className="flex gap-2">
                  {["curl", "python", "javascript"].map((format) => (
                    <button
                      key={format}
                      onClick={() => setCodeFormat(format as any)}
                      className={`px-4 py-2 rounded-lg capitalize ${
                        codeFormat === format ? "bg-green-500" : "bg-slate-800"
                      }`}
                    >
                      {format}
                    </button>
                  ))}
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
        )}
      </div>
    </div>
  );
}
