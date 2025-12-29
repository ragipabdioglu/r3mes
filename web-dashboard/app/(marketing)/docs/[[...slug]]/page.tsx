"use client";

import { useState } from "react";
import { useParams } from "next/navigation";
import dynamic from "next/dynamic";
import Link from "next/link";
import { Menu, X, ChevronRight } from "lucide-react";

const DocsSidebar = dynamic(() => import("@/components/docs/DocsSidebar"), {
  ssr: false
});

const DocsContent = dynamic(() => import("@/components/docs/DocsContent"), {
  ssr: false
});

const DocSearch = dynamic(() => import("@/components/docs/DocSearch"), {
  ssr: false
});

export default function DocsPage() {
  const params = useParams();
  const slug = params?.slug as string[] | undefined;
  const docPath = slug ? slug.join("/") : "";
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-slate-800/50 bg-slate-950/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo & Nav */}
            <div className="flex items-center gap-6">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="lg:hidden p-2 text-slate-400 hover:text-white transition-colors"
              >
                {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </button>
              
              <Link href="/" className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-green-400 to-emerald-600 flex items-center justify-center">
                  <span className="text-white font-bold text-sm">R3</span>
                </div>
                <span className="font-semibold text-white hidden sm:block">R3MES</span>
              </Link>

              <nav className="hidden md:flex items-center gap-1 text-sm">
                <Link href="/docs" className="px-3 py-2 text-green-400 font-medium">
                  Docs
                </Link>
                <Link href="/docs/api-reference" className="px-3 py-2 text-slate-400 hover:text-white transition-colors">
                  API
                </Link>
                <a 
                  href="https://github.com/r3mes-network/r3mes" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="px-3 py-2 text-slate-400 hover:text-white transition-colors"
                >
                  GitHub
                </a>
              </nav>
            </div>

            {/* Search */}
            <div className="flex-1 max-w-md mx-4">
              <DocSearch />
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3">
              <Link 
                href="/docs/quick-start"
                className="hidden sm:flex items-center gap-1 px-4 py-2 bg-green-500 hover:bg-green-400 text-black font-medium text-sm rounded-lg transition-colors"
              >
                Get Started
                <ChevronRight className="w-4 h-4" />
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Layout */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex gap-8 py-8">
          {/* Sidebar */}
          <aside
            className={`${
              sidebarOpen ? "fixed inset-0 z-40 bg-slate-950 p-6 lg:relative lg:p-0" : "hidden"
            } lg:block w-64 flex-shrink-0`}
          >
            {sidebarOpen && (
              <button
                onClick={() => setSidebarOpen(false)}
                className="lg:hidden absolute top-4 right-4 p-2 text-slate-400 hover:text-white"
              >
                <X className="w-5 h-5" />
              </button>
            )}
            <div className="sticky top-24">
              <DocsSidebar currentPath={docPath} />
            </div>
          </aside>

          {/* Content */}
          <main className="flex-1 min-w-0">
            <DocsContent docPath={docPath} />
          </main>
        </div>
      </div>
    </div>
  );
}
