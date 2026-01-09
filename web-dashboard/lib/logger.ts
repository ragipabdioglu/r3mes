// Logger utility for R3MES Web Dashboard

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: Date;
  data?: any;
  context?: string;
}

class Logger {
  private logs: LogEntry[] = [];
  private maxLogs = 1000;
  private isDevelopment = process.env.NODE_ENV === 'development';

  private addLog(level: LogLevel, message: string, data?: any, context?: string) {
    const entry: LogEntry = {
      level,
      message,
      timestamp: new Date(),
      data,
      context,
    };

    this.logs.push(entry);

    // Keep only the most recent logs
    if (this.logs.length > this.maxLogs) {
      this.logs = this.logs.slice(-this.maxLogs);
    }

    // Console output in development
    if (this.isDevelopment) {
      const prefix = context ? `[${context}]` : '';
      const logMessage = `${prefix} ${message}`;
      
      switch (level) {
        case 'debug':
          console.debug(logMessage, data);
          break;
        case 'info':
          console.info(logMessage, data);
          break;
        case 'warn':
          console.warn(logMessage, data);
          break;
        case 'error':
          console.error(logMessage, data);
          break;
      }
    }

    // Send to external logging service in production
    if (!this.isDevelopment && level === 'error') {
      this.sendToExternalService(entry);
    }
  }

  private sendToExternalService(entry: LogEntry) {
    // In a real application, this would send to a service like Sentry, LogRocket, etc.
    try {
      if (typeof window !== 'undefined' && (window as any).Sentry) {
        (window as any).Sentry.captureMessage(entry.message, {
          level: entry.level,
          extra: {
            data: entry.data,
            context: entry.context,
            timestamp: entry.timestamp,
          },
        });
      }
    } catch (error) {
      console.error('Failed to send log to external service:', error);
    }
  }

  debug(message: string, data?: any, context?: string) {
    this.addLog('debug', message, data, context);
  }

  info(message: string, data?: any, context?: string) {
    this.addLog('info', message, data, context);
  }

  warn(message: string, data?: any, context?: string) {
    this.addLog('warn', message, data, context);
  }

  error(message: string, data?: any, context?: string) {
    this.addLog('error', message, data, context);
  }

  getLogs(level?: LogLevel): LogEntry[] {
    if (level) {
      return this.logs.filter(log => log.level === level);
    }
    return [...this.logs];
  }

  clearLogs() {
    this.logs = [];
  }

  exportLogs(): string {
    return JSON.stringify(this.logs, null, 2);
  }
}

export const logger = new Logger();