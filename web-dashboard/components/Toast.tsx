"use client";

import { useEffect, useState } from "react";
import { toast, Toast } from "@/lib/toast";

export default function ToastContainer() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    const unsubscribe = toast.subscribe(setToasts);
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

  const getBorderColor = (type: Toast["type"]) => {
    switch (type) {
      case "success":
        return "border-green-500";
      case "error":
        return "border-red-500";
      case "warning":
        return "border-amber-500";
      case "info":
        return "border-blue-500";
    }
  };

  return (
    <div className="fixed top-5 right-5 z-[9999] flex flex-col gap-3 pointer-events-none">
      {toasts.map((toastItem) => (
        <div
          key={toastItem.id}
          className={`
            flex items-center gap-3 px-4 py-3 rounded-lg min-w-[300px] max-w-[400px]
            bg-slate-800 border ${getBorderColor(toastItem.type)} text-slate-100
            shadow-lg pointer-events-auto cursor-pointer
            animate-[slideIn_0.3s_ease-out] hover:-translate-x-1 transition-transform
          `}
          onClick={() => toast.remove(toastItem.id)}
        >
          <span className="text-xl flex-shrink-0">{getIcon(toastItem.type)}</span>
          <span className="flex-1 text-sm leading-relaxed">{toastItem.message}</span>
          <button
            className="bg-transparent border-none text-slate-400 text-xl cursor-pointer p-0 w-5 h-5 flex items-center justify-center rounded hover:bg-white/10 hover:text-slate-100 transition-all flex-shrink-0"
            onClick={(e) => {
              e.stopPropagation();
              toast.remove(toastItem.id);
            }}
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}
