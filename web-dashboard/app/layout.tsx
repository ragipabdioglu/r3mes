import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Providers from "@/providers/providers";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import Script from "next/script";
import { GA_TRACKING_ID } from "@/lib/analytics";

// Optimize font loading
const inter = Inter({ 
  subsets: ["latin"],
  display: 'swap', // Use swap for better performance
  preload: true,
  variable: '--font-inter',
});

export const metadata: Metadata = {
  title: {
    default: "R3MES - The Compute Layer of AI",
    template: "%s | R3MES",
  },
  description: "Decentralized AI training network with verifiable computation. Connect your GPU, earn R3MES tokens, or use the world's most efficient AI model.",
  keywords: ["R3MES", "decentralized AI", "blockchain", "GPU mining", "AI training", "federated learning", "BitNet", "LoRA"],
  authors: [{ name: "R3MES Team" }],
  creator: "R3MES",
  publisher: "R3MES",
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL || "https://r3mes.network"),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    type: "website",
    locale: "en_US",
    url: process.env.NEXT_PUBLIC_SITE_URL || "https://r3mes.network",
    siteName: "R3MES",
    title: "R3MES - The Compute Layer of AI",
    description: "Decentralized AI training network with verifiable computation. Connect your GPU, earn R3MES tokens.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "R3MES - The Compute Layer of AI",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "R3MES - The Compute Layer of AI",
    description: "Decentralized AI training network with verifiable computation.",
    images: ["/og-image.png"],
    creator: "@r3mes",
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
  verification: {
    google: process.env.NEXT_PUBLIC_GOOGLE_VERIFICATION,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} antialiased flex flex-col min-h-screen`} style={{ backgroundColor: 'var(--bg-primary)', color: 'var(--text-primary)' }}>
        {/* Skip to content link for accessibility */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 z-[9999] px-4 py-2 rounded-md font-medium transition-all"
          style={{
            backgroundColor: 'var(--accent-primary)',
            color: 'white',
          }}
        >
          Skip to main content
        </a>
        
        {/* Theme initialization script - inline to prevent flash */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  const savedTheme = localStorage.getItem('r3mes_theme');
                  // Default to light mode if no saved theme
                  const theme = savedTheme || 'light';
                  
                  let resolvedTheme;
                  if (theme === 'system') {
                    resolvedTheme = window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
                  } else {
                    resolvedTheme = theme;
                  }
                  
                  // Ensure resolvedTheme is valid
                  if (resolvedTheme !== 'light' && resolvedTheme !== 'dark') {
                    resolvedTheme = 'light';
                  }
                  
                  const root = document.documentElement;
                  if (resolvedTheme === 'dark') {
                    root.classList.add('dark');
                  } else {
                    root.classList.remove('dark');
                  }
                  
                  // If no saved theme, set default to light in localStorage
                  if (!savedTheme) {
                    localStorage.setItem('r3mes_theme', 'light');
                  }
                } catch (e) {
                  // If localStorage is not available, default to light mode
                  document.documentElement.classList.remove('dark');
                }
              })();
            `,
          }}
        />
        {/* Google Analytics */}
        {GA_TRACKING_ID && (
          <>
            <Script
              strategy="afterInteractive"
              src={`https://www.googletagmanager.com/gtag/js?id=${GA_TRACKING_ID}`}
            />
            <Script
              id="google-analytics"
              strategy="afterInteractive"
              dangerouslySetInnerHTML={{
                __html: `
                  window.dataLayer = window.dataLayer || [];
                  function gtag(){dataLayer.push(arguments);}
                  gtag('js', new Date());
                  gtag('config', '${GA_TRACKING_ID}', {
                    page_path: window.location.pathname,
                  });
                `,
              }}
              // Note: In production, CSP allows scripts from googletagmanager.com
              // The inline script above requires unsafe-inline in dev, but in production
              // we should move to external GA script or use nonce (future improvement)
            />
          </>
        )}
        <Providers>
          <main id="main-content" className="flex-1 pb-20 sm:pb-22 md:pb-24" tabIndex={-1}>
            {children}
          </main>
          <Footer />
          <Navbar />
        </Providers>
      </body>
    </html>
  );
}

