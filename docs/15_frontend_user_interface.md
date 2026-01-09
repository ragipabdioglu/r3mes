# R3MES Web Dashboard - Unified User Interface

## Genel Bakış

R3MES Web Dashboard, kullanıcıların AI inference servisini kullanabileceği, madencilik yapabileceği, ağ durumunu izleyebileceği ve yönetim işlemlerini gerçekleştirebileceği tek birleşik web arayüzüdür. Next.js 14, TypeScript, Tailwind CSS, Cosmos Kit ve Web3 wallet entegrasyonu ile geliştirilmiştir.

**Not**: Bu döküman, mevcut `web-dashboard/` klasörüne yeni özelliklerin entegre edilmesini açıklar. Ayrı bir `frontend/` klasörü yoktur - tüm özellikler tek bir web sitesinde birleştirilmiştir.

## Proje Yapısı

```
/web-dashboard
    /app
        page.tsx              # Landing Page (onboarding kontrolü, network stats)
        chat/page.tsx          # Chat Arayüzü (AI inference)
        mine/page.tsx          # Mine Dashboard (mining statistics)
        wallet/page.tsx        # Wallet Management (transaction history, API keys)
        network/page.tsx       # Network Explorer (3D globe, miners table, blocks)
        settings/page.tsx      # Settings (config management UI)
        help/page.tsx          # Help & Support (FAQ, links)
        onboarding/page.tsx    # Onboarding Flow (multi-step) 
        layout.tsx             # Root Layout
        globals.css            # Global Styles
    /components
        Navbar.tsx             # Navigation Bar (wallet, theme toggle)
        Footer.tsx             # Footer
        ChatInterface.tsx      # Chat UI Component
        MinerConsole.tsx       # Miner Console
        NetworkExplorer.tsx    # Network Explorer (3D globe)
        HardwareMonitor.tsx    # Hardware Monitor
        TrainingGraph.tsx       # Training Graph
        MinersTable.tsx        # Miners Table
        RecentBlocks.tsx       # Recent Blocks
        NetworkStats.tsx       # Network Stats
        StatCard.tsx           # Statistics Card
        StatusBadge.tsx        # Status Badge
        SkeletonLoader.tsx     # Loading Skeletons
        ErrorBoundary.tsx      # Error Boundary
        WalletButton.tsx       # Wallet Connection Button
        WalletGuard.tsx        # Wallet Requirement Guard
        Toast.tsx              # Toast Notifications
        ... (27+ components)
    /contexts
        WalletContext.tsx      # Global wallet state management
        ThemeContext.tsx       # Theme management (dark/light)
    /hooks
        useNetworkStats.ts     # React Query hook for network stats
        useMinerData.ts        # React Query hook for miner data
        useUserInfo.ts         # React Query hook for user info
        useTransactionHistory.ts # React Query hook for transaction history
    /lib
        api.ts                 # Backend API client (throws errors)
        keplr.ts               # Keplr integration
    /providers
        providers.tsx          # Context providers (Wallet, Theme, QueryClient)
    /utils
        numberFormat.ts        # Number formatting utilities
        errorMessages.ts       # User-friendly error messages
    next.config.js            # Next.js Configuration (security headers)
    package.json              # Dependencies
    tailwind.config.ts        # Tailwind Configuration
```

## 🏗️ Adım 1: Proje Kurulumu ve Kütüphaneler

### Teknolojiler

- **Next.js 14**: App Router, TypeScript
- **Tailwind CSS**: Utility-first CSS framework
- **Framer Motion**: Animasyonlar için
- **Lucide React**: İkonlar için
- **clsx & tailwind-merge**: Class yönetimi için
- **@cosmos-kit/react**: Cosmos wallet integration (Keplr, Leap, Cosmostation)
- **@keplr-wallet/types**: Keplr wallet types
- **@tanstack/react-query**: Data fetching ve caching
- **axios**: API istekleri için
- **Recharts**: 2D charting library (zero GPU usage)
- **react-globe.gl**: 3D globe (lazy-loaded, only for Network Explorer)

### Global CSS (globals.css)

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --background: #050505;        /* Derin Siyah */
  --foreground: #e5e5e5;        /* Off-white */
  --selection: #00ff41;         /* Matrix Yeşili */
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  background-color: var(--background);
  color: var(--foreground);
  font-family: 'Inter', sans-serif;
}

::selection {
  background-color: var(--selection);
  color: var(--background);
}

code, pre {
  font-family: 'JetBrains Mono', monospace;
}
```

### Package.json Dependencies

```json
{
  "dependencies": {
    "next": "14.0.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.0.0",
    "tailwindcss": "^3.3.0",
    "framer-motion": "^10.16.0",
    "lucide-react": "^0.294.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0",
    "wagmi": "^2.0.0",
    "viem": "^2.0.0",
    "@tanstack/react-query": "^5.0.0",
    "@rainbow-me/rainbowkit": "^2.0.0",
    "axios": "^1.6.0",
    "recharts": "^2.10.0"
  }
}
```

## 🎨 Adım 2: Navbar ve Cüzdan Entegrasyonu

### Navbar Component

**Dosya**: `components/Navbar.tsx`

**Tasarım**:
- Ekranın üstünde sabit (sticky)
- Arkası bulanık (backdrop-blur-md)
- Altı ince çizgili (border-b border-white/10)
- Sol: 'R3MES' logosu (JetBrains Mono, Bold, Neon Yeşil efektli)
- Orta: Linkler (Chat, Mine, Network, Build) - Hover olunca altı çizilsin
- Sağ: RainbowKit ConnectButton + Kredi Badge

**Logic**:
```typescript
import { useAccount } from 'wagmi';
import { ConnectButton } from '@rainbow-me/rainbowkit';
import { useEffect, useState } from 'react';
import axios from 'axios';

export default function Navbar() {
  const { address, isConnected } = useAccount();
  const [credits, setCredits] = useState<number | null>(null);

  useEffect(() => {
    if (isConnected && address) {
      // Backend'den kullanıcı bilgilerini çek
      axios.get(`/api/user/info/${address}`)
        .then(res => setCredits(res.data.credits))
        .catch(err => console.error(err));
    }
  }, [address, isConnected]);

  return (
    <nav className="sticky top-0 z-50 backdrop-blur-md border-b border-white/10">
      <div className="container mx-auto px-4 py-4 flex items-center justify-between">
        {/* Logo */}
        <div className="font-mono font-bold text-2xl text-[#00ff41]">
          R3MES
        </div>

        {/* Links */}
        <div className="flex gap-8">
          <Link href="/chat">Chat</Link>
          <Link href="/mine">Mine</Link>
          <Link href="/network">Network</Link>
          <Link href="/build">Build</Link>
        </div>

        {/* Wallet + Credits */}
        <div className="flex items-center gap-4">
          {credits !== null && (
            <span className="text-sm bg-white/10 px-3 py-1 rounded-full">
              {credits} Credits
            </span>
          )}
          <ConnectButton />
        </div>
      </div>
    </nav>
  );
}
```

## 🏠 Adım 3: Anasayfa (Landing Page)

### Hero Section

**Dosya**: `app/page.tsx`

**Özellikler**:
- Ortada devasa başlık: "The Decentralized Brain." (Framer Motion ile harf harf animasyon)
- Altında gri açıklama
- İki büyük buton: "Start Chatting" (Primary - Yeşil) ve "Start Mining" (Secondary - Outline)
- Canlı Veri Bandı (Ticker): Backend'den `/api/network/stats` çekilir
- Arka Plan: Siyah zemin üzerine silik, yavaşça hareket eden yeşil grid animasyonu

**Implementation**:
```typescript
import { motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import axios from 'axios';

export default function HomePage() {
  const [stats, setStats] = useState(null);

  useEffect(() => {
    axios.get('/api/network/stats')
      .then(res => setStats(res.data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated Grid Background */}
      <div className="absolute inset-0 opacity-10">
        <GridAnimation />
      </div>

      {/* Hero Section */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen">
        <motion.h1
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="text-8xl font-bold mb-6"
        >
          {title.split('').map((char, i) => (
            <motion.span
              key={i}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.05 }}
            >
              {char}
            </motion.span>
          ))}
        </motion.h1>

        <p className="text-gray-400 text-xl mb-8 max-w-2xl text-center">
          R3MES, dünyanın her yerindeki atıl GPU güçlerini birleştirerek
          demokratik bir süper zeka yaratır.
        </p>

        <div className="flex gap-4">
          <Button href="/chat" variant="primary">
            Start Chatting
          </Button>
          <Button href="/mine" variant="outline">
            Start Mining
          </Button>
        </div>
      </div>

      {/* Live Data Ticker */}
      <div className="fixed bottom-0 left-0 right-0 border-t border-white/10 bg-black/50 backdrop-blur-sm">
        <Ticker stats={stats} />
      </div>
    </div>
  );
}
```

### Canlı Veri Bandı (Ticker)

```typescript
function Ticker({ stats }) {
  if (!stats) return null;

  return (
    <div className="py-2 overflow-hidden">
      <div className="flex gap-8 animate-scroll">
        <span>ACTIVE MINERS: {stats.active_miners.toLocaleString()}</span>
        <span>•</span>
        <span>GLOBAL HASHRATE: {stats.total_credits} PetaFLOPS</span>
        <span>•</span>
        <span>BLOCK HEIGHT: #{stats.block_height || 'N/A'}</span>
      </div>
    </div>
  );
}
```

## 💬 Adım 4: Chat Arayüzü (The Product)

### Chat Page

**Dosya**: `app/chat/page.tsx`

**Layout**: Sol Sidebar (Dar) + Sağ Chat Alanı (Geniş)

**Özellikler**:

1. **Auth Guard**: Cüzdan bağlı değilse overlay göster
2. **Chat Akışı**: Terminal-style mesajlar
   - Kullanıcı: `> [Mesaj]` (Gri)
   - AI: `R3MES: [Cevap]` (Yeşil)
3. **Backend Bağlantısı**: Streaming response desteği
4. **Meta Veri**: Model bilgisi, router, cost

**Implementation**:
```typescript
import { useAccount } from 'wagmi';
import { useState, useRef, useEffect } from 'react';
import axios from 'axios';

export default function ChatPage() {
  const { address, isConnected } = useAccount();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || !isConnected) return;

    const userMessage = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          wallet_address: address
        })
      });

      // Streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiMessage = { role: 'ai', content: '' };

      setMessages(prev => [...prev, aiMessage]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        aiMessage.content += chunk;

        setMessages(prev => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1] = { ...aiMessage };
          return newMessages;
        });
      }
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  if (!isConnected) {
    return <WalletConnectOverlay />;
  }

  return (
    <div className="flex h-screen">
      {/* Sidebar */}
      <div className="w-64 border-r border-white/10 p-4">
        <h2 className="font-mono text-lg mb-4">Chat History</h2>
        {/* Chat history list */}
      </div>

      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, i) => (
            <div key={i} className={msg.role === 'user' ? 'text-gray-400' : 'text-[#00ff41]'}>
              {msg.role === 'user' ? '> ' : 'R3MES: '}
              {msg.content}
            </div>
          ))}
          {isLoading && <div className="text-gray-500">R3MES: Thinking...</div>}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t border-white/10 p-4">
          <form onSubmit={(e) => { e.preventDefault(); handleSend(); }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="w-full bg-black/50 border border-white/10 rounded px-4 py-2"
            />
          </form>
          <p className="text-xs text-gray-600 mt-2">
            Model: BitNet-b1.58 | Router: Auto | Cost: 1 Credit
          </p>
        </div>
      </div>
    </div>
  );
}
```

## ⛏️ Adım 5: Mine (Dashboard) Sayfası

### Mine Dashboard

**Dosya**: `app/mine/page.tsx`

**Özellikler**:
- Header: "Download Miner" butonu
- Bento Grid Layout:
  - Kutu 1: Total Earnings (Backend'den kullanıcı coin miktarı)
  - Kutu 2: Current Tier (PRO MINER / GUEST)
  - Kutu 3: Network Difficulty Graph (Recharts - Son 7 gün)
  - Liste: Recent Blocks (Son bulunan bloklar ve bulan cüzdanlar)

**Implementation**:
```typescript
import { useAccount } from 'wagmi';
import { useEffect, useState } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';

export default function MinePage() {
  const { address, isConnected } = useAccount();
  const [userInfo, setUserInfo] = useState(null);
  const [recentBlocks, setRecentBlocks] = useState([]);
  const [difficultyData, setDifficultyData] = useState([]);

  useEffect(() => {
    if (isConnected && address) {
      // User info
      axios.get(`/api/user/info/${address}`)
        .then(res => setUserInfo(res.data))
        .catch(err => console.error(err));

      // Recent blocks (from blockchain API)
      axios.get('/api/blocks?limit=10')
        .then(res => setRecentBlocks(res.data.blocks))
        .catch(err => console.error(err));
    }
  }, [address, isConnected]);

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-4xl font-bold">Mine Dashboard</h1>
        <Button>
          <DownloadIcon /> Download Miner
        </Button>
      </div>

      {/* Bento Grid */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        {/* Total Earnings */}
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-sm text-gray-400 mb-2">Total Earnings</h3>
          <p className="text-3xl font-bold">
            {userInfo?.credits || 0} REMES
          </p>
        </div>

        {/* Current Tier */}
        <div className="bg-white/5 rounded-lg p-6 border border-white/10">
          <h3 className="text-sm text-gray-400 mb-2">Current Tier</h3>
          <p className="text-3xl font-bold">
            {userInfo?.is_miner ? 'PRO MINER' : 'GUEST'}
          </p>
        </div>

        {/* Network Difficulty Graph */}
        <div className="bg-white/5 rounded-lg p-6 border border-white/10 col-span-1">
          <h3 className="text-sm text-gray-400 mb-4">Network Difficulty</h3>
          <LineChart width={300} height={200} data={difficultyData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" />
            <XAxis dataKey="date" stroke="#666" />
            <YAxis stroke="#666" />
            <Tooltip />
            <Line type="monotone" dataKey="difficulty" stroke="#00ff41" />
          </LineChart>
        </div>
      </div>

      {/* Recent Blocks */}
      <div className="bg-white/5 rounded-lg p-6 border border-white/10">
        <h3 className="text-lg font-bold mb-4">Recent Blocks</h3>
        <div className="space-y-2">
          {recentBlocks.map((block, i) => (
            <div key={i} className="flex justify-between items-center py-2 border-b border-white/5">
              <span className="font-mono text-sm">Block #{block.height}</span>
              <span className="text-xs text-gray-400">
                {block.miner?.slice(0, 8)}...{block.miner?.slice(-6)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

## 🔗 Bağlantı Ayarı (Proxy)

### Next.js Proxy Configuration

**Dosya**: `next.config.js`

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/:path*',
      },
    ];
  },
};

module.exports = nextConfig;
```

Bu sayede frontend kodunda `fetch('/api/chat')` yazdığımızda, istek otomatik olarak `http://localhost:8000/chat` adresine yönlendirilir.

## Wallet Configuration

### Cosmos Kit Integration

**Dosya**: `providers/providers.tsx`

```typescript
"use client";

import { QueryClientProvider } from "./query-provider";
import ErrorBoundary from "@/components/ErrorBoundary";
import ToastContainer from "@/components/Toast";
import { WalletProvider } from "@/contexts/WalletContext";
import { ThemeProvider } from "@/contexts/ThemeContext";

export default function Providers({ children }: { children: React.ReactNode }) {
  return (
    <ErrorBoundary>
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
```

### Wallet Context

**Dosya**: `contexts/WalletContext.tsx`

- Global wallet state management
- `walletAddress`, `credits`, `userInfo`
- `refreshUserInfo()` function
- LocalStorage integration

### Keplr Integration

**Dosya**: `lib/keplr.ts`

- Keplr wallet connection
- Transaction signing and broadcasting
- Chain configuration
- Multi-wallet support (Keplr, Leap, Cosmostation)

## Kullanım

### 1. Projeyi Oluştur

```bash
npx create-next-app@latest frontend --typescript --tailwind --app
cd frontend
```

### 2. Bağımlılıkları Yükle

```bash
npm install framer-motion lucide-react clsx tailwind-merge \
  @tanstack/react-query @cosmos-kit/react @keplr-wallet/types \
  @cosmjs/proto-signing @cosmjs/stargate axios recharts \
  react-globe.gl three
```

### 3. Geliştirme Sunucusunu Başlat

```bash
npm run dev
```

### 4. Backend Servisini Başlat

```bash
# Ayrı terminal
python run_backend.py
```

## Özellikler

### ✅ Modern UI/UX
- Dark/Light theme toggle (ThemeContext)
- Smooth animations (Framer Motion)
- Responsive design
- Terminal-style chat interface
- Onboarding flow
- Help & Support page

### ✅ Wallet Integration
- Cosmos Kit integration (Keplr, Leap, Cosmostation)
- WalletContext for global state
- Credit system integration
- Real-time balance updates
- Transaction history

### ✅ Backend Integration
- React Query for data fetching and caching
- Streaming chat responses
- Credit deduction
- User info fetching
- Network stats display
- Error handling with user-friendly messages

### ✅ Pages (11 Sayfa)
- `/` - Landing page (onboarding kontrolü, network stats)
- `/chat` - AI chat interface
- `/mine` - Mining dashboard
- `/wallet` - Wallet management
- `/network` - Network explorer (3D globe)
- `/settings` - Configuration management
- `/help` - Help & Support
- `/onboarding` - Multi-step onboarding
- `/analytics` - Analytics dashboard (API usage, model performance, user engagement)
- `/leaderboard` - Top miners and validators leaderboard
- `/playground` - API playground (interactive API testing, code generation)

### ✅ Components (27+ Component)
- Core: Navbar, Footer, ErrorBoundary
- Features: ChatInterface, MinerConsole, NetworkExplorer
- UI: StatCard, StatusBadge, SkeletonLoader, Toast
- Blockchain: StakingDashboard, ValidatorList, GovernancePanel
- Data: MinersTable, RecentBlocks, NetworkStats, TrainingGraph

### ✅ Performance
- Next.js 14 App Router
- React Query caching
- Code splitting
- Optimized bundle size
- Zero-GPU interface (strictly 2D for miners)

### ✅ Production Features
- Security headers (next.config.js)
- Error boundaries
- Loading states
- User-friendly error messages
- Theme persistence

## 📊 Adım 6: Analytics Sayfası

### Analytics Dashboard

**Dosya**: `app/analytics/page.tsx`

**Özellikler**:
- API usage statistics (endpoint bazlı)
- User engagement metrics
- Model performance trends
- Network health visualization
- Real-time charts (Recharts)

**Implementation**:
```typescript
import { useQuery } from "@tanstack/react-query";
import { getAnalytics } from "@/lib/api";

export default function AnalyticsPage() {
  const { data: analytics, isLoading } = useQuery({
    queryKey: ["analytics"],
    queryFn: () => getAnalytics(),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  return (
    <div>
      {/* Stats Cards */}
      <StatCard label="Active Users" value={analytics?.user_engagement?.active_users} />
      <StatCard label="API Calls" value={analytics?.api_usage?.total_requests} />
      <StatCard label="Avg Latency" value={analytics?.model_performance?.average_latency} />
      <StatCard label="Success Rate" value={analytics?.model_performance?.success_rate} />

      {/* Charts */}
      <BarChart data={analytics?.api_usage?.endpoints_data} />
      <LineChart data={analytics?.model_performance?.trend} />
    </div>
  );
}
```

**API Integration**: `GET /analytics?days=7`

## 🏆 Adım 7: Leaderboard Sayfası

### Leaderboard Page

**Dosya**: `app/leaderboard/page.tsx`

**Özellikler**:
- Top miners leaderboard (by reputation)
- Top validators leaderboard (by trust score)
- Tier badges (bronze, silver, gold, platinum, diamond)
- Ranking system with trophies
- Real-time updates (30s refresh)

**Implementation**:
```typescript
import { useQuery } from "@tanstack/react-query";
import { getLeaderboard } from "@/lib/api";

export default function LeaderboardPage() {
  const [activeTab, setActiveTab] = useState<"miners" | "validators">("miners");

  const { data: minersData } = useQuery({
    queryKey: ["leaderboard", "miners"],
    queryFn: () => getLeaderboard("miners"),
    refetchInterval: 30000,
  });

  const { data: validatorsData } = useQuery({
    queryKey: ["leaderboard", "validators"],
    queryFn: () => getLeaderboard("validators"),
    refetchInterval: 30000,
  });

  return (
    <div>
      {/* Tabs */}
      <button onClick={() => setActiveTab("miners")}>Top Miners</button>
      <button onClick={() => setActiveTab("validators")}>Most Trusted Validators</button>

      {/* Leaderboard Table */}
      {activeTab === "miners" ? (
        <MinersTable data={minersData} />
      ) : (
        <ValidatorsTable data={validatorsData} />
      )}
    </div>
  );
}
```

**Tier System**:
- Diamond: reputation >= 1000
- Platinum: reputation >= 500
- Gold: reputation >= 200
- Silver: reputation >= 50
- Bronze: reputation < 50

**API Integration**: 
- `GET /leaderboard/miners?limit=100`
- `GET /leaderboard/validators?limit=100`

## 🧪 Adım 8: API Playground Sayfası

### API Playground

**Dosya**: `app/playground/page.tsx`

**Özellikler**:
- Interactive API testing interface
- Request/response viewer
- Code generation (cURL, Python, JavaScript)
- Endpoint testing
- Real-time response display

**Implementation**:
```typescript
export default function PlaygroundPage() {
  const [endpoint, setEndpoint] = useState("/health");
  const [method, setMethod] = useState<"GET" | "POST">("GET");
  const [requestBody, setRequestBody] = useState("{}");
  const [response, setResponse] = useState<any>(null);
  const [codeFormat, setCodeFormat] = useState<"curl" | "python" | "javascript">("curl");

  const handleRequest = async () => {
    const result = await axios.request({
      method,
      url: `${API_BASE_URL}${endpoint}`,
      data: method === "POST" ? JSON.parse(requestBody) : undefined,
    });
    setResponse(result.data);
  };

  return (
    <div>
      {/* Request Panel */}
      <div>
        <select value={method} onChange={(e) => setMethod(e.target.value)}>
          <option value="GET">GET</option>
          <option value="POST">POST</option>
        </select>
        <input value={endpoint} onChange={(e) => setEndpoint(e.target.value)} />
        {method === "POST" && (
          <textarea value={requestBody} onChange={(e) => setRequestBody(e.target.value)} />
        )}
        <button onClick={handleRequest}>Send Request</button>
      </div>

      {/* Response Panel */}
      <div>
        <pre>{JSON.stringify(response, null, 2)}</pre>
      </div>

      {/* Code Generation */}
      <div>
        <button onClick={() => setCodeFormat("curl")}>cURL</button>
        <button onClick={() => setCodeFormat("python")}>Python</button>
        <button onClick={() => setCodeFormat("javascript")}>JavaScript</button>
        <pre>{generateCode()}</pre>
      </div>
    </div>
  );
}
```

**Code Generation**: Supports cURL, Python (requests), and JavaScript (fetch) formats.

---

**Son Güncelleme**: 2024

