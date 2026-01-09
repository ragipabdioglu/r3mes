/**
 * WebSocket Client
 * 
 * Provides WebSocket connection for real-time updates.
 */

import { logger } from './logger';

export type WebSocketChannel = "miner_stats" | "training_metrics" | "network_status" | "block_updates";

export interface WebSocketMessage<T = unknown> {
  type: string;
  data: T;
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private channel: WebSocketChannel;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<(data: unknown) => void>> = new Map();

  constructor(channel: WebSocketChannel) {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    
    // Get WebSocket host from environment variable
    // Use NEXT_PUBLIC_WS_URL if available, otherwise derive from NEXT_PUBLIC_BACKEND_URL
    let wsHost: string;
    if (process.env.NEXT_PUBLIC_WS_URL) {
      wsHost = process.env.NEXT_PUBLIC_WS_URL.replace(/^wss?:/, "").replace(/^https?:/, "");
    } else if (process.env.NEXT_PUBLIC_BACKEND_URL) {
      wsHost = process.env.NEXT_PUBLIC_BACKEND_URL.replace(/^https?:/, "");
    } else if (process.env.NODE_ENV === 'development') {
      wsHost = "localhost:8000";
    } else {
      throw new Error('NEXT_PUBLIC_WS_URL or NEXT_PUBLIC_BACKEND_URL must be set in production');
    }
    
    this.url = `${wsProtocol}//${wsHost}/ws/${channel}`;
    this.channel = channel;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          logger.info(`WebSocket connected to ${this.channel}`);
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            
            // Handle ping/pong heartbeat
            if (message.type === "ping") {
              this.ws?.send(JSON.stringify({ type: "pong", timestamp: Date.now() }));
              return;
            }
            
            this.notifyListeners(message.type, message.data);
          } catch (e) {
            logger.error("Error parsing WebSocket message:", e);
          }
        };

        this.ws.onerror = (error) => {
          logger.error(`WebSocket error on ${this.channel}:`, error);
          reject(error);
        };

        this.ws.onclose = () => {
          logger.info(`WebSocket disconnected from ${this.channel}`);
          this.attemptReconnect();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      
      logger.info(`Attempting to reconnect to ${this.channel} (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);
      
      setTimeout(() => {
        this.connect().catch((error) => {
          logger.error(`Reconnection failed:`, error);
        });
      }, delay);
    } else {
      logger.error(`Max reconnection attempts reached for ${this.channel}`);
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.listeners.clear();
  }

  on(messageType: string, callback: (data: unknown) => void): void {
    if (!this.listeners.has(messageType)) {
      this.listeners.set(messageType, new Set());
    }
    this.listeners.get(messageType)!.add(callback);
  }

  off(messageType: string, callback: (data: unknown) => void): void {
    const callbacks = this.listeners.get(messageType);
    if (callbacks) {
      callbacks.delete(callback);
    }
  }

  private notifyListeners(messageType: string, data: unknown): void {
    const callbacks = this.listeners.get(messageType);
    if (callbacks) {
      callbacks.forEach((callback) => {
        try {
          callback(data);
        } catch (error) {
          logger.error("Error in WebSocket listener:", error);
        }
      });
    }
  }

  send(message: Record<string, unknown>): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      logger.warn("WebSocket is not connected");
    }
  }
}

// Hook for React components
export function useWebSocket(channel: WebSocketChannel) {
  const [client, setClient] = React.useState<WebSocketClient | null>(null);
  const [connected, setConnected] = React.useState(false);

  React.useEffect(() => {
    const wsClient = new WebSocketClient(channel);
    setClient(wsClient);

    wsClient.connect().then(() => {
      setConnected(true);
    }).catch((error) => {
      logger.error("Failed to connect WebSocket:", error);
      setConnected(false);
    });

    return () => {
      wsClient.disconnect();
      setConnected(false);
    };
  }, [channel]);

  return { client, connected };
}

// Import React for the hook
import React from "react";

