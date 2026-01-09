"use client";

import React, { Component, ErrorInfo, ReactNode } from "react";
import { toast } from "@/lib/toast";
import { logger } from "@/lib/logger";
import { AlertTriangle, RefreshCw, Home } from "lucide-react";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  level?: 'page' | 'component' | 'root';
  name?: string;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorId: string | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  private retryCount = 0;
  private maxRetries = 3;

  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorId: null,
    };
  }

  static getDerivedStateFromError(error: Error): State {
    const errorId = `error-${Date.now()}-${Math.random().toString(36).substring(2, 11)}`;
    return {
      hasError: true,
      error,
      errorId,
    };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    const { level = 'component', name = 'Unknown', onError } = this.props;
    
    logger.error(`ErrorBoundary (${level}:${name}) caught an error:`, error, JSON.stringify(errorInfo));
    
    // Call custom error handler if provided
    onError?.(error, errorInfo);
    
    // Send to Sentry with enhanced context
    if (typeof window !== "undefined" && (window as any).Sentry) {
      (window as any).Sentry.captureException(error, {
        contexts: {
          react: {
            componentStack: errorInfo.componentStack,
          },
          errorBoundary: {
            level,
            name,
            retryCount: this.retryCount,
          },
        },
        tags: {
          component: "ErrorBoundary",
          level,
          name,
        },
        extra: {
          errorId: this.state.errorId,
        },
      });
    }
    
    // Show toast notification with appropriate urgency
    const urgency = level === 'root' ? 'error' : 'warning';
    toast[urgency](
      `An error occurred in ${name}: ${error.message || "Unknown error"}`,
      level === 'root' ? 15000 : 10000
    );

    // Announce error to screen readers
    this.announceError(error, level);
  }

  private announceError = (error: Error, level: string) => {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'assertive');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = `Error occurred in ${level}: ${error.message}. Recovery options are available.`;
    document.body.appendChild(announcement);
    setTimeout(() => {
      if (document.body.contains(announcement)) {
        document.body.removeChild(announcement);
      }
    }, 3000);
  };

  handleReset = () => {
    this.retryCount += 1;
    this.setState({
      hasError: false,
      error: null,
      errorId: null,
    });

    // Announce recovery attempt
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.className = 'sr-only';
    announcement.textContent = `Attempting to recover from error. Retry ${this.retryCount} of ${this.maxRetries}.`;
    document.body.appendChild(announcement);
    setTimeout(() => {
      if (document.body.contains(announcement)) {
        document.body.removeChild(announcement);
      }
    }, 2000);
  };

  handleReload = () => {
    // Announce page reload
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'assertive');
    announcement.className = 'sr-only';
    announcement.textContent = 'Reloading page to recover from error.';
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      window.location.reload();
    }, 1000);
  };

  handleGoHome = () => {
    // Announce navigation
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.className = 'sr-only';
    announcement.textContent = 'Navigating to home page.';
    document.body.appendChild(announcement);
    
    setTimeout(() => {
      window.location.href = '/';
    }, 1000);
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      const { level = 'component', name = 'Component' } = this.props;
      const canRetry = this.retryCount < this.maxRetries;
      const isRootLevel = level === 'root';

      return (
        <div 
          className="flex items-center justify-center min-h-[400px] p-10"
          role="alert"
          aria-labelledby="error-title"
          aria-describedby="error-description"
        >
          <div className="text-center max-w-[600px] bg-slate-800 border border-slate-700 rounded-xl p-10">
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle 
                className="w-8 h-8 text-red-500 flex-shrink-0" 
                aria-hidden="true"
              />
              <div>
                <h2 
                  id="error-title"
                  className="text-2xl font-semibold text-slate-100"
                >
                  {isRootLevel ? 'Application Error' : `${name} Error`}
                </h2>
                <p className="text-sm text-slate-400 mt-1">
                  Error ID: {this.state.errorId}
                </p>
              </div>
            </div>
            
            <div 
              id="error-description"
              className="text-sm text-slate-400 mb-6"
            >
              <p className="text-slate-300 mb-2">
                {this.state.error?.message || "An unexpected error occurred"}
              </p>
              <p className="text-sm text-slate-400">
                {isRootLevel 
                  ? "The application encountered a critical error. Please try one of the recovery options below."
                  : "This component failed to load. You can try to recover or continue using other parts of the application."
                }
              </p>
              {this.retryCount > 0 && (
                <p className="text-sm text-yellow-400 mt-2">
                  Retry attempt {this.retryCount} of {this.maxRetries}
                </p>
              )}
            </div>
            
            <div className="flex flex-wrap gap-3 justify-center">
              {canRetry && (
                <button
                  onClick={this.handleReset}
                  className="btn-retry flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                  aria-describedby="retry-help"
                >
                  <RefreshCw className="w-4 h-4" aria-hidden="true" />
                  Try Again
                </button>
              )}
              
              {!isRootLevel && (
                <button
                  onClick={this.handleGoHome}
                  className="btn-home flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                  aria-describedby="home-help"
                >
                  <Home className="w-4 h-4" aria-hidden="true" />
                  Go Home
                </button>
              )}
              
              <button
                onClick={this.handleReload}
                className="btn-reload flex items-center gap-2 px-4 py-2 bg-slate-600 hover:bg-slate-700 text-white rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                aria-describedby="reload-help"
              >
                <RefreshCw className="w-4 h-4" aria-hidden="true" />
                Reload Page
              </button>
            </div>

            {/* Screen reader help text */}
            <div className="sr-only">
              <div id="retry-help">
                Attempts to recover the component without reloading the page
              </div>
              <div id="home-help">
                Navigate to the home page to continue using the application
              </div>
              <div id="reload-help">
                Reloads the entire page to recover from the error
              </div>
            </div>

            {/* Error details for development */}
            {process.env.NODE_ENV === "development" && this.state.error && (
              <details className="mt-6 text-left bg-slate-900 border border-slate-700 rounded-lg p-4">
                <summary className="cursor-pointer text-xs text-slate-400 hover:text-slate-100 mb-3 focus:outline-none focus:ring-2 focus:ring-blue-500 rounded px-2 py-1">
                  Error Details (Development Only)
                </summary>
                <pre 
                  className="text-xs font-mono text-red-300 whitespace-pre-wrap break-all p-3 bg-slate-950 rounded overflow-x-auto"
                  aria-label="Error stack trace"
                >
                  {this.state.error.stack}
                </pre>
              </details>
            )}

            {/* Contact support info */}
            <div className="mt-6 p-4 bg-slate-800/50 rounded-lg border border-slate-700">
              <h3 className="text-sm font-medium text-slate-200 mb-2">
                Need Help?
              </h3>
              <p className="text-xs text-slate-400 mb-2">
                If this error persists, please contact our support team with the Error ID above.
              </p>
              <div className="flex flex-wrap gap-2 text-xs">
                <a 
                  href="https://discord.gg/remes" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900 rounded px-1"
                >
                  Discord Support
                </a>
                <span className="text-slate-600">•</span>
                <a 
                  href="https://github.com/AquaMystic/R3MES/issues" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-400 hover:text-blue-300 underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-900 rounded px-1"
                >
                  Report Bug
                </a>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

