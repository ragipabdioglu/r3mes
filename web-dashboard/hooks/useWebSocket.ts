import { useState, useEffect, useRef, useCallback } from "react";

// Configuration constants (can be overridden via environment variables)
const WS_RECONNECT_DELAY_MS = parseInt(process.env.NEXT_PUBLIC_WS_RECONNECT_DELAY || "1000", 10);
const WS_MAX_RECONNECT_ATTEMPTS = parseInt(process.env.NEXT_PUBLIC_WS_MAX_RECONNECT_ATTEMPTS || "10", 10);
const WS_HEARTBEAT_INTERVAL_MS = parseInt(process.env.NEXT_PUBLIC_WS_HEARTBEAT_INTERVAL || "30000", 10);

interface UseWebSocketReturn<T> {
  data: T | null;
  isConnected: boolean;
  error: Error | null;
  reconnectAttempts: number;
  reconnect: () => void;
}

export function useWebSocket<T>(url: string): UseWebSocketReturn<T> {
  const [data, setData] = useState<T | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const shouldReconnectRef = useRef(true);

  const clearTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    // Only connect in browser
    if (typeof window === "undefined") {
      return;
    }

    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }
    clearTimers();

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        setIsConnected(true);
        setError(null);
        setReconnectAttempts(0);
        
        // Start heartbeat (ping-pong)
        heartbeatIntervalRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            try {
              ws.send(JSON.stringify({ type: "pong", timestamp: Date.now() }));
            } catch (e) {
              // Ignore send errors during heartbeat
            }
          }
        }, WS_HEARTBEAT_INTERVAL_MS);
      };

      ws.onmessage = (event) => {
        try {
          const parsedData = JSON.parse(event.data);
          
          // Handle ping messages from server
          if (parsedData.type === "ping") {
            ws.send(JSON.stringify({ type: "pong", timestamp: Date.now() }));
            return;
          }
          
          setData(parsedData);
        } catch (err) {
          setError(err as Error);
        }
      };

      ws.onerror = () => {
        setError(new Error("WebSocket connection error"));
        setIsConnected(false);
      };

      ws.onclose = (event) => {
        setIsConnected(false);
        clearTimers();
        
        // Attempt reconnection if not intentionally closed
        if (shouldReconnectRef.current && reconnectAttempts < WS_MAX_RECONNECT_ATTEMPTS) {
          const delay = WS_RECONNECT_DELAY_MS * Math.pow(2, reconnectAttempts); // Exponential backoff
          
          reconnectTimeoutRef.current = setTimeout(() => {
            setReconnectAttempts((prev) => prev + 1);
            connect();
          }, Math.min(delay, 30000)); // Cap at 30 seconds
        } else if (reconnectAttempts >= WS_MAX_RECONNECT_ATTEMPTS) {
          setError(new Error(`Failed to connect after ${WS_MAX_RECONNECT_ATTEMPTS} attempts`));
        }
      };

      wsRef.current = ws;
    } catch (err) {
      setError(err as Error);
      setIsConnected(false);
    }
  }, [url, reconnectAttempts, clearTimers]);

  const reconnect = useCallback(() => {
    setReconnectAttempts(0);
    shouldReconnectRef.current = true;
    connect();
  }, [connect]);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();

    return () => {
      shouldReconnectRef.current = false;
      clearTimers();
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [url]); // Only reconnect when URL changes

  return { data, isConnected, error, reconnectAttempts, reconnect };
}

