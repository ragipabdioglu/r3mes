"use client";

export function SkeletonCard() {
  return (
    <div className="card animate-pulse">
      <div className="h-4 bg-slate-700 rounded w-1/4 mb-4"></div>
      <div className="h-8 bg-slate-700 rounded w-1/2 mb-2"></div>
      <div className="h-3 bg-slate-700 rounded w-3/4"></div>
    </div>
  );
}

export function SkeletonStatCard() {
  return (
    <div className="stat-card animate-pulse">
      <div className="flex items-start justify-between mb-2">
        <div className="h-3 bg-slate-700 rounded w-24"></div>
        <div className="h-5 w-5 bg-slate-700 rounded"></div>
      </div>
      <div className="h-8 bg-slate-700 rounded w-32 mb-1"></div>
      <div className="h-3 bg-slate-700 rounded w-20"></div>
    </div>
  );
}

export function SkeletonTable() {
  return (
    <div className="card animate-pulse">
      <div className="h-6 bg-slate-700 rounded w-1/4 mb-4"></div>
      <div className="space-y-3">
        {[1, 2, 3, 4, 5].map((i) => (
          <div key={i} className="flex items-center justify-between py-3 border-b border-slate-700/50">
            <div className="h-4 bg-slate-700 rounded w-32"></div>
            <div className="h-4 bg-slate-700 rounded w-24"></div>
            <div className="h-4 bg-slate-700 rounded w-16"></div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function SkeletonChart() {
  return (
    <div className="card animate-pulse">
      <div className="h-6 bg-slate-700 rounded w-1/3 mb-4"></div>
      <div className="h-64 bg-slate-700 rounded"></div>
    </div>
  );
}

