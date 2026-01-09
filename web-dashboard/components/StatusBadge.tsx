"use client";

interface StatusBadgeProps {
  status: "active" | "inactive" | "pending" | "success" | "error" | "warning";
  label: string;
}

export default function StatusBadge({ status, label }: StatusBadgeProps) {
  const statusColors = {
    active: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    inactive: "bg-slate-500/20 text-slate-400 border-slate-500/30",
    pending: "bg-amber-500/20 text-amber-400 border-amber-500/30",
    success: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    error: "bg-red-500/20 text-red-400 border-red-500/30",
    warning: "bg-amber-500/20 text-amber-400 border-amber-500/30",
  };

  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${statusColors[status]}`}
    >
      {label}
    </span>
  );
}

