/**
 * Toast Notification System
 * 
 * Global toast notification system for success/error messages
 */

export type ToastType = "success" | "error" | "warning" | "info";

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
}

class ToastManager {
  private toasts: Toast[] = [];
  private listeners: Set<(toasts: Toast[]) => void> = new Set();

  subscribe(listener: (toasts: Toast[]) => void) {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  }

  private notify() {
    this.listeners.forEach((listener) => listener([...this.toasts]));
  }

  show(type: ToastType, message: string, duration: number = 5000) {
    const id = Math.random().toString(36).substring(7);
    const toast: Toast = { id, type, message, duration };
    
    this.toasts.push(toast);
    this.notify();

    if (duration > 0) {
      setTimeout(() => {
        this.remove(id);
      }, duration);
    }

    return id;
  }

  remove(id: string) {
    this.toasts = this.toasts.filter((toast) => toast.id !== id);
    this.notify();
  }

  clear() {
    this.toasts = [];
    this.notify();
  }

  getToasts(): Toast[] {
    return [...this.toasts];
  }
}

export const toastManager = new ToastManager();

// Convenience functions
export const toast = {
  success: (message: string, duration?: number) =>
    toastManager.show("success", message, duration),
  error: (message: string, duration?: number) =>
    toastManager.show("error", message, duration),
  warning: (message: string, duration?: number) =>
    toastManager.show("warning", message, duration),
  info: (message: string, duration?: number) =>
    toastManager.show("info", message, duration),
};

