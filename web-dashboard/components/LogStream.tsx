"use client";

import { useState, useEffect, useRef } from "react";
import { logger } from "@/lib/logger";

export default function LogStream() {
  const [logs, setLogs] = useState<Array<{ timestamp: string; level: string; message: string }>>([]);
  const logEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Auto-scroll to bottom
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Connect to WebSocket log stream from Python miner
  useEffect(() => {
    // Get WebSocket URL from environment variable
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 
      (process.env.NODE_ENV === 'development' ? "ws://localhost:1317" : (() => { 
        logger.error('NEXT_PUBLIC_WS_URL must be set in production');
        return "ws://localhost:1317"; // Fallback for error case
      })());
    const ws = new WebSocket(`${wsUrl}/ws?topic=miner_logs`);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        // Handle ping messages
        if (data.type === "ping") {
          return;
        }
        
        // Add log entry
        const newLog = {
          timestamp: data.timestamp || new Date().toISOString(),
          level: data.level || "info",
          message: data.message || "",
        };
        
        setLogs((prev) => [...prev.slice(-99), newLog]); // Keep last 100 logs
      } catch (error) {
        logger.error("Failed to parse log message:", error);
      }
    };
    
    ws.onerror = (error) => {
      logger.error("WebSocket log stream error:", error);
    };
    
    ws.onclose = () => {
      logger.log("WebSocket log stream closed");
    };
    
    return () => {
      ws.close();
    };
  }, []);

  const getLevelColor = (level: string) => {
    switch (level) {
      case "error":
        return "text-red-600 dark:text-red-400";
      case "warning":
        return "text-yellow-600 dark:text-yellow-400";
      case "info":
        return "text-blue-600 dark:text-blue-400";
      default:
        return "text-gray-600 dark:text-gray-400";
    }
  };

  return (
    <div className="bg-gray-900 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
      {logs.length === 0 ? (
        <div className="text-gray-500">No logs yet...</div>
      ) : (
        logs.map((log, index) => (
          <div key={index} className="mb-1">
            <span className="text-gray-500">{log.timestamp.split("T")[1].split(".")[0]}</span>
            <span className={`ml-2 ${getLevelColor(log.level)}`}>[{log.level.toUpperCase()}]</span>
            <span className="ml-2 text-gray-300">{log.message}</span>
          </div>
        ))
      )}
      <div ref={logEndRef} />
    </div>
  );
}

