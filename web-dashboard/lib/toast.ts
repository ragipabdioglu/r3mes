// Toast notification system for R3MES Web Dashboard

export interface ToastOptions {
  duration?: number;
  position?: 'top-right' | 'top-left' | 'bottom-right' | 'bottom-left';
  dismissible?: boolean;
}

export interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  duration: number;
  dismissible: boolean;
}

class ToastManager {
  private toasts: Toast[] = [];
  private listeners: ((toasts: Toast[]) => void)[] = [];

  private generateId(): string {
    return `toast-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
  }

  private notify() {
    this.listeners.forEach(listener => listener([...this.toasts]));
  }

  private addToast(type: Toast['type'], message: string, duration: number = 5000, options: ToastOptions = {}) {
    const toast: Toast = {
      id: this.generateId(),
      type,
      message,
      duration: options.duration || duration,
      dismissible: options.dismissible !== false,
    };

    this.toasts.push(toast);
    this.notify();

    // Auto-remove toast after duration
    if (toast.duration > 0) {
      setTimeout(() => {
        this.remove(toast.id);
      }, toast.duration);
    }

    return toast.id;
  }

  success(message: string, duration?: number, options?: ToastOptions): string {
    return this.addToast('success', message, duration, options);
  }

  error(message: string, duration?: number, options?: ToastOptions): string {
    return this.addToast('error', message, duration || 8000, options);
  }

  warning(message: string, duration?: number, options?: ToastOptions): string {
    return this.addToast('warning', message, duration || 6000, options);
  }

  info(message: string, duration?: number, options?: ToastOptions): string {
    return this.addToast('info', message, duration, options);
  }

  remove(id: string): void {
    this.toasts = this.toasts.filter(toast => toast.id !== id);
    this.notify();
  }

  clear(): void {
    this.toasts = [];
    this.notify();
  }

  subscribe(listener: (toasts: Toast[]) => void): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  getToasts(): Toast[] {
    return [...this.toasts];
  }
}

export const toast = new ToastManager();
// Toast manager for global toast notifications
export const toastManager = {
  success: (message: string) => {
    console.log('Success:', message);
  },
  error: (message: string) => {
    console.error('Error:', message);
  },
  info: (message: string) => {
    console.info('Info:', message);
  },
  warning: (message: string) => {
    console.warn('Warning:', message);
  }
};