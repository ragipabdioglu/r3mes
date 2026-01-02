"use client";

import { QueryClientProvider } from "./query-provider";
import ErrorBoundary from "@/components/ErrorBoundary";
import ToastContainer from "@/components/Toast";
import { WalletProvider } from "@/contexts/WalletContext";
import { ThemeProvider } from "@/contexts/ThemeContext";

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ErrorBoundary level="root" name="Application">
      <QueryClientProvider>
        <ThemeProvider>
          <WalletProvider>
            {children}
            <ToastContainer />
          </WalletProvider>
        </ThemeProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

