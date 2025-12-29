"use client";

import { useEffect, useState } from "react";
import { toastManager, Toast } from "@/lib/toast";
import "./Toast.css";

export default function ToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    const unsubscribe = toastManager.subscribe(setToasts);
    return unsubscribe;
  }, []);

  const getIcon = (type: Toast["type"]) => {
    switch (type) {
      case "success":
        return "✅";
      case "error":
        return "❌";
      case "warning":
        return "⚠️";
      case "info":
        return "ℹ️";
    }
  };

  const getClassName = (type: Toast["type"]) => {
    return `toast toast-${type}`;
  };

  return (
    <div className="toast-container">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={getClassName(toast.type)}
          onClick={() => toastManager.remove(toast.id)}
        >
          <span className="toast-icon">{getIcon(toast.type)}</span>
          <span className="toast-message">{toast.message}</span>
          <button
            className="toast-close"
            onClick={(e) => {
              e.stopPropagation();
              toastManager.remove(toast.id);
            }}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

