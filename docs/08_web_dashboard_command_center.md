# R3MES Web Dashboard & Command Center

## Overview

The R3MES Web Dashboard "Cockpit" provides a real-time monitoring interface for miners and investors. With a dual interface design, it meets both miner operations and network demonstration needs.

The Web Dashboard is the **recommended interface for users who want to use AI inference, stake tokens, participate in governance, or explore the network**. It requires no installation and can be accessed from any modern web browser.

---

## 🖥️ User Interface Overview

### What Is the Web Dashboard?

The Web Dashboard is a unified web interface built with Next.js 14 and TypeScript. It provides a comprehensive set of features for interacting with the R3MES network without requiring any local infrastructure setup.

### Key Pages

1. **Chat (`/chat`)**: AI inference interface with streaming responses
2. **Mine (`/mine`)**: Mining statistics and earnings dashboard
3. **Network (`/network`)**: 3D globe visualization, miners table, recent blocks, network statistics
4. **Wallet (`/wallet`)**: Transaction history, API key management, balance display
5. **Staking (`/staking`)**: Delegate tokens, claim rewards, view staking information
6. **Roles (`/roles`)**: Register node roles (Miner, Serving, Validator, Proposer) on the blockchain
7. **Settings (`/settings`)**: Configuration management, model paths, API rate limits
8. **Help (`/help`)**: FAQ section, support links, documentation links
9. **Onboarding (`/onboarding`)**: Multi-step onboarding flow for new users

### Feature Highlights

- **Zero Installation**: Access from any browser, no download required
- **Wallet Integration**: Keplr, Leap, Cosmostation wallet support via Cosmos Kit
- **Real-time Updates**: WebSocket streaming for live metrics
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Dark/Light Themes**: Theme toggle for user preference
- **Network Explorer**: 3D globe visualization of network nodes
- **AI Chat Interface**: Terminal-style chat with streaming responses
- **Governance Participation**: Vote on proposals and participate in protocol governance

---

## 🆚 Desktop Launcher vs Web Dashboard

### When to Use Web Dashboard

Use the Web Dashboard if you:
- Want to use AI inference (chat interface) without running local infrastructure
- Want to stake tokens or participate in governance
- Want to explore the network (3D globe, miners table, network statistics)
- Prefer a web-based interface (no installation required)
- Want to access from any device or browser
- Don't need to run local blockchain nodes, miners, or IPFS

### When to Use Desktop Launcher

Use the Desktop Launcher if you:
- Want to run a blockchain node locally
- Want to mine and train AI models
- Need embedded IPFS management
- Require system tray integration for background operation
- Prefer native desktop application
- Want local process management and monitoring

### Integration

Both interfaces work together:
- Desktop Launcher manages local infrastructure (node, miner, IPFS)
- Web Dashboard accesses the network through the blockchain and provides user-facing features
- Role registration is done via Web Dashboard (`/roles` page) even if you use Desktop Launcher for local management

For a detailed comparison, see [Desktop Launcher Documentation](10_desktop_launcher.md#feature-comparison-desktop-launcher-vs-web-dashboard).

---

## Technical Architecture

### Frontend Stack
- **Next.js 14**: Modern React framework with TypeScript (App Router)
- **Tailwind CSS**: Utility-first CSS framework
- **TanStack Query**: Data fetching and caching (React Query)
- **Recharts**: 2D charting library (zero GPU usage)
- **react-globe.gl**: 3D globe (lazy-loaded, only for Network Explorer)
- **@cosmos-kit/react**: Cosmos wallet integration (Keplr, Leap, Cosmostation)
- **@keplr-wallet/types**: Keplr wallet types
- **Framer Motion**: Animations
- **Lucide React**: Icons

### Backend Integration
- **FastAPI Backend**: `/api/*` endpoints (http://localhost:${BACKEND_PORT:-8000} in development, configurable via `BACKEND_PORT` environment variable)
- **React Query Hooks**: `useNetworkStats`, `useMinerData`, `useUserInfo`, `useTransactionHistory`
- **Error Handling**: User-friendly error messages with Sentry integration
- **CORS Configuration**: Strict in production, flexible in development

### Marketing & Documentation Site
- **Docs Web Interface**: Markdown parser with sidebar navigation and search
- **Developers Section**: Quickstart guides, API documentation, SDK downloads
- **Community Page**: Discord, GitHub, blog posts, events
- **Protocol Page**: Architecture, security, economics, governance overview
- **SEO & Analytics**: Google Analytics integration, sitemap, robots.txt

## Interface Design

### 1. Miner Console (Zero-GPU Interface)

#### Critical Requirement: 0% GPU Usage
- **Strictly 2D Design**: No 3D rendering, WebGL, or GPU acceleration
- **Lightweight Components**: Minimal DOM manipulation
- **Efficient Updates**: Optimized re-rendering strategies
- **Memory Management**: Prevent memory leaks in long-running sessions

#### Live Training Graph
```typescript
interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  timestamp: number;
  gradient_norm: number;
}

// Recharts implementation - 2D only
<LineChart data={trainingData}>
  <Line dataKey="loss" stroke="#8884d8" />
  <Line dataKey="accuracy" stroke="#82ca9d" />
</LineChart>
```

#### Hardware Monitor
```typescript
interface MinerStats {
  gpu_temp: number;        // °C
  fan_speed: number;       // RPM
  vram_usage: number;      // MB
  power_draw: number;      // Watts
  hash_rate: number;       // Gradients/hour
  uptime: number;          // Seconds
}

// WebSocket subscription
const { data: minerStats } = useWebSocket(`ws://${process.env.NEXT_PUBLIC_BACKEND_URL?.replace('http://', '').replace('https://', '') || 'localhost:8000'}/ws/miner_stats`);
```

#### Log Stream
- **Web Terminal**: Real-time Python Worker logs
- **Filtering**: Error, Warning, Info levels
- **Search**: Log content search functionality
- **Export**: Download logs for debugging

### 2. Network Explorer "Visor"

#### Global Node Map
```typescript
// Lazy-loaded 3D globe (only for Network Explorer)
const Globe = dynamic(() => import('react-globe.gl'), { ssr: false });

interface NodeLocation {
  lat: number;
  lng: number;
  miner_address: string;
  status: 'active' | 'inactive' | 'syncing';
  contribution_score: number;
}
```

**Miner Locations API**: `/api/dashboard/locations`
- **Data Source**: Real blockchain data from `NodeRegistrations` and `MiningContributions` collections
- **Location Derivation**: Deterministic pseudo-location from miner address (SHA256 hash) for visualization
- **Privacy**: Does NOT expose real IP/location data - locations are derived deterministically
- **Size Calculation**: Based on stake (for registered nodes) or trust score (for mining contributions)
- **No Mock Data**: Returns empty list if SDK context unavailable (no fallback mock data)

#### Block Explorer (Basitleştirilmiş)

**Miners Table**:
- List of active miners
- Miner address (shortened)
- Reputation scores (Trust Score)
- Total submissions (successful gradient submissions)
- Last submission height
- Status (Active/Inactive)
- Reputation tier (Bronze/Silver/Gold/Platinum)

**Recent Blocks**:
- Recent blocks (last 20 blocks)
- Block height
- Block time (timestamp)
- Transaction count (number of transactions)
- Block hash
- Validator (who produced it)
- Click to view block details

**Network Stats**:
- Total Stake: 1,234,567 REMES
- Inflation Rate: 5.2%
- Model Version: BitNet b1.58 (Genesis)
- Active Miners: 42
- Total Gradients: 12,345
- Network Hash Rate: 1,234 gradients/hour
- Average Block Time: 5.2 seconds

**Transaction Details**:
- Gradient submissions
- Aggregations
- Challenges
- Governance proposals
- Staking transactions

### 3. Wallet & Authentication

#### WalletContext (Global State Management)

**Dosya**: `contexts/WalletContext.tsx`

- Global wallet state management
- `walletAddress`, `credits`, `userInfo`
- `refreshUserInfo()` function
- LocalStorage integration
- Custom events for wallet changes

#### Keplr Integration

**Dosya**: `lib/keplr.ts`

- Keplr wallet connection
- Transaction signing and broadcasting
- Chain configuration
- Multi-wallet support (Keplr, Leap, Cosmostation)

**WalletButton Component**:
- Connect/Disconnect functionality
- Address display (shortened)
- Credit balance display
- Theme toggle integration

Keplr entegrasyonu hala desteklenir, ancak `cosmos-kit` önerilir:

```typescript
interface KeplrIntegration {
  connectWallet(): Promise<void>;
  addChain(): Promise<void>;
  signTransaction(tx: any): Promise<string>;
  getBalance(): Promise<Coin[]>;
  delegate(validator: string, amount: Coin): Promise<void>;
}
```

## Real-time Data Streaming

### WebSocket Implementation (Go Backend)

```go
// WebSocket handler in Cosmos SDK
func (k Keeper) HandleWebSocket(w http.ResponseWriter, r *http.Request) {
    conn, err := upgrader.Upgrade(w, r, nil)
    if err != nil {
        return
    }
    defer conn.Close()

    // Subscribe to topics
    topic := r.URL.Query().Get("topic")
    
    switch topic {
    case "miner_stats":
        k.streamMinerStats(conn)
    case "training_metrics":
        k.streamTrainingMetrics(conn)
    case "network_status":
        k.streamNetworkStatus(conn)
    }
}

func (k Keeper) streamMinerStats(conn *websocket.Conn) {
    ticker := time.NewTicker(2 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ticker.C:
            stats := k.GetMinerStats()
            conn.WriteJSON(stats)
        }
    }
}
```

### Frontend WebSocket Client

```typescript
// Custom hook for WebSocket data
function useWebSocket<T>(url: string) {
  const [data, setData] = useState<T | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const parsedData = JSON.parse(event.data);
      setData(parsedData);
    };
    ws.onclose = () => setIsConnected(false);

    return () => ws.close();
  }, [url]);

  return { data, isConnected };
}
```

## Component Architecture

### 1. Layout Structure
```
Dashboard/
├── Header (Wallet connection, network status)
├── Sidebar (Navigation, miner selection)
├── Main Content
│   ├── Miner Console
│   │   ├── Training Graph
│   │   ├── Hardware Monitor
│   │   └── Log Stream
│   └── Network Explorer
│       ├── Global Map
│       ├── Block Explorer
│       └── Statistics
└── Footer (Status indicators)
```

### 2. State Management

**React Query Hooks**:
- `useNetworkStats()` - Network statistics
- `useMinerData(walletAddress)` - Miner stats, earnings, hashrate
- `useUserInfo(walletAddress)` - User information
- `useTransactionHistory(walletAddress, limit)` - Transaction history

**Context Providers**:
- `WalletContext` - Global wallet state
- `ThemeContext` - Theme management (dark/light/system)

**Example**:
```typescript
// Using React Query hooks
const { data: stats, isLoading, error } = useNetworkStats();

// Using Context
const { walletAddress, credits, refreshUserInfo } = useWallet();
const { resolvedTheme, toggleTheme } = useTheme();
```

## Security Considerations

### Authentication Flow
1. **Keplr Connection**: User connects wallet
2. **Chain Addition**: Add R3MES network to Keplr
3. **Address Verification**: Verify wallet ownership
4. **Session Management**: Maintain authenticated session
5. **Transaction Signing**: Sign transactions via Keplr

### Data Protection
- **HTTPS Only**: All communications encrypted
- **CORS Policy**: Restricted origin access
- **Rate Limiting**: Prevent API abuse
- **Input Validation**: Sanitize all user inputs

## Advanced Features

### Leaderboard

**Endpoint**: `/leaderboard`

**Features**:
- Top Miners: Ranked by reputation score
- Most Trusted Validators: Ranked by trust score
- Tier System: Bronze, Silver, Gold, Platinum, Diamond
- Real-time Updates: Auto-refresh every 30 seconds
- Trend Indicators: Shows upward/downward trends

**Tier Calculation**:
- Diamond: Reputation ≥ 1000
- Platinum: Reputation ≥ 500
- Gold: Reputation ≥ 200
- Silver: Reputation ≥ 50
- Bronze: Reputation < 50

### Advanced Analytics

**Endpoint**: `/analytics`

**Features**:
- **Overview**: API usage, model performance, user engagement
- **Network Growth**: Miners, validators, hashrate trends over time
- **Mining Efficiency**: GPU utilization, power efficiency, earnings per hashrate
- **Economic Analysis**: Tokenomics, rewards distribution, staking ratios

**Analytics Endpoints**:

**Basic Analytics** (single comprehensive endpoint):
- `GET /analytics`: Comprehensive analytics including API usage, user engagement, model performance, and network health

**Advanced Analytics** (detailed endpoints):
- `GET /analytics/network-growth`: Network growth metrics
- `GET /analytics/mining-efficiency`: Mining efficiency analysis
- `GET /analytics/economic-analysis`: Economic metrics and trends
- `GET /analytics/performance-benchmarks`: Performance benchmarks
- `GET /analytics/comparative-analysis`: Comparative statistics

### Notification System

**Backend Service**: `backend/app/notifications.py`

**Channels**:
- **Email**: SMTP-based email notifications
- **Slack**: Webhook-based Slack notifications
- **In-App**: Database-stored notifications for frontend retrieval

**Notification Types**:
- Mining alerts (hashrate drops, GPU issues)
- System alerts (backend down, blockchain sync issues)
- Economic alerts (reward changes, staking updates)

**Priority Levels**:
- Low: Informational updates
- Medium: Important notices
- High: Critical alerts
- Critical: Emergency notifications

**Configuration**:
- Environment variables for SMTP and Slack configuration
- Channel selection via `NOTIFICATION_CHANNELS`
- Priority-based routing

## Performance Optimization

### Frontend Optimization
- **Code Splitting**: Lazy load components
- **Image Optimization**: Next.js Image component
- **Bundle Analysis**: Monitor bundle size
- **Caching Strategy**: Efficient data caching

### Backend Optimization
- **Connection Pooling**: Efficient WebSocket management
- **Data Compression**: Compress WebSocket messages
- **Rate Limiting**: Prevent resource exhaustion
- **Monitoring**: Performance metrics collection

## Deployment Strategy

### Development Environment
```bash
# Frontend development
npm run dev

# Backend (Cosmos SDK)
remesd start --api.enable=true --api.swagger=true

# WebSocket server
remesd start-websocket --port=8080
```

### Production Deployment
- **CDN**: Static asset distribution
- **Load Balancer**: Multiple backend instances
- **SSL Termination**: HTTPS enforcement
- **Monitoring**: Application performance monitoring

## User Experience Design

### Pages (11 Pages)

1. **Landing Page (`/`)**: 
   - Onboarding check
   - Network statistics
   - Feature highlights
   - Quick start buttons

2. **Chat (`/chat`)**:
   - AI chat interface
   - Streaming responses
   - Model selection
   - Credit display

3. **Mine (`/mine`)**:
   - Mining statistics
   - Earnings history
   - Hashrate graphs
   - Recent blocks

4. **Wallet (`/wallet`)**:
   - Transaction history
   - API key management
   - Balance display
   - Credit information

5. **Network (`/network`)**:
   - 3D globe (lazy-loaded)
   - Miners table
   - Recent blocks
   - Network stats

6. **Settings (`/settings`)**:
   - Configuration management
   - Model paths
   - GPU limits
   - Mining difficulty
   - API rate limits

7. **Help (`/help`)**:
   - FAQ section
   - Support links
   - Documentation links

8. **Onboarding (`/onboarding`)**:
   - Multi-step onboarding
   - System checks
   - Welcome experience

### Miner Workflow
1. **Onboarding**: First-time setup
2. **Connect Wallet**: Keplr integration
3. **View Training**: Real-time loss reduction
4. **Monitor Hardware**: GPU stats without conflicts
5. **Check Rewards**: Token balance and earnings
6. **Manage Settings**: Configure mining parameters

### Investor Workflow
1. **Network Overview**: Global node map (3D globe)
2. **Training Progress**: Model improvement metrics
3. **Economic Metrics**: Token distribution, rewards
4. **Block Explorer**: Transaction transparency
5. **Governance**: Participate in voting

## Testing Strategy

### Unit Tests
- Component rendering tests
- WebSocket connection tests
- Keplr integration tests
- State management tests

### Integration Tests
- End-to-end user workflows
- Real-time data streaming
- Wallet connection flows
- API integration tests

### Performance Tests
- Load testing with multiple users
- WebSocket connection limits
- Memory usage monitoring
- GPU usage verification (0% requirement)

## Maintenance & Updates

### Monitoring
- **Error Tracking**: Sentry integration
- **Performance Monitoring**: Web Vitals
- **User Analytics**: Usage patterns
- **System Health**: Uptime monitoring

### Update Strategy
- **Semantic Versioning**: Clear version management
- **Feature Flags**: Gradual rollout
- **Rollback Plan**: Quick reversion capability
- **User Communication**: Update notifications

## Backend Integration Details

### Dashboard API (Go)

```go
// remes/x/remes/keeper/dashboard_api.go
type DashboardAPI struct {
    keeper            Keeper
    queryServer       types.QueryServer
    grpcAddr          string
    tendermintRPCAddr string
    cache             DashboardCache
    cacheTTL          time.Duration  // Default: 30 seconds
}
```

**Features**:
- Uses gRPC query endpoints (for SDK context access)
- In-memory caching (30 seconds TTL, configurable)
- CORS policy enforcement
- Input validation
- Pagination support (max 1000 for miners, max 100 for blocks)

### WebSocket Data Sources

**miner_stats** and **training_metrics**:
- Retrieved from Python miner stats HTTP server (configurable via `R3MES_STATS_PORT` environment variable, default: `http://localhost:8080/stats`)
- Go node makes HTTP request to Python miner's stats server
- Environment variables: `MINER_STATS_HOST`, `MINER_STATS_PORT`

**network_status**:
- On-chain data (gRPC queries)
- Real-time blockchain state

### Block Query Endpoints

**Implementation**: gRPC query endpoints kullanılarak SDK context'ten alınır.

```go
// remes/x/remes/keeper/query_dashboard.go
func (k Keeper) QueryBlocks(ctx context.Context, req *types.QueryBlocksRequest) (*types.QueryBlocksResponse, error) {
    // SDK context'ten block data al
    // Pagination support
}

func (k Keeper) QueryBlock(ctx context.Context, req *types.QueryBlockRequest) (*types.BlockInfo, error) {
    // Belirli blok detayları
}
```

### 4. Staking & Validasyon Arayüzü

**Validator Listesi**:
- Validator name ve moniker (Cosmos SDK staking module'den)
- Voting power (stake miktarı) - Cosmos SDK staking module'den
- Commission rate - Cosmos SDK staking module'den
- Uptime percentage - Status'e göre hesaplanıyor (BONDED = 100%)
- Status (Active/Jailed/Unbonding) - Cosmos SDK staking module'den
- Trust Score - R3MES keeper'dan `ValidatorVerificationRecord` ile hesaplanıyor
- Total verifications - from R3MES keeper
- Successful verifications - R3MES keeper'dan
- False verdicts - R3MES keeper'dan
- Lazy validation count - R3MES keeper'dan
- Self-delegation - Cosmos SDK staking module'den
- Total delegations - from Cosmos SDK staking module

**Data Sources**:
- Validator basic info: Cosmos SDK staking module REST API (`/cosmos/staking/v1beta1/validators`)
- Validator trust score: R3MES keeper `ValidatorVerificationRecord` collection
- Trust score calculation: `success_rate - (false_verdicts_penalty + lazy_penalty)`

**Staking İşlemleri**:
- **Delegate**: Validator'a stake et
  - Validator seçimi
  - Miktar girişi (REMES)
  - Transaction fee gösterimi
  - Confirm butonu (Keplr ile imzalama)

- **Undelegate**: Stake'i geri çek
  - Unbonding period bilgisi (21 gün)
  - Miktar girişi
  - Confirm butonu

- **Redelegate**: Bir validator'dan diğerine transfer
  - Kaynak validator seçimi
  - Hedef validator seçimi
  - Miktar girişi
  - Confirm butonu

- **Claim Rewards**: Ödülleri topla
  - Available rewards gösterimi
  - Claim all butonu
  - Individual validator rewards

**Staking Dashboard**:
- Total staked: 1,234.56 REMES (from Cosmos SDK staking module)
- Pending rewards: 12.34 REMES (from Cosmos SDK staking module)
- Unbonding: 100 REMES (in 21 days) (from Cosmos SDK staking module)
- Staking history (transaction list)

### 5. Governance Panel

**Active Proposals**:
- Proposal ID
- Title ve description
- Type (Parameter Change, Software Upgrade, Model Upgrade, vb.)
- Status (Deposit Period, Voting Period, Passed, Rejected)
- Voting end time
- Current votes:
  - Yes: 45.2%
  - No: 12.3%
  - Abstain: 5.1%
  - No with Veto: 2.4%

**Vote Interface**:
- "Vote" button (for each proposal)
- Vote options:
  - ✅ Yes
  - ❌ No
  - ⚪ Abstain
  - 🚫 No with Veto
- Transaction fee gösterimi
- Confirm butonu (Keplr ile imzalama)

**Model Upgrade Proposals** (Critical for R3MES):
- Yeni model versiyonu (örn: BitNet v2)
- Model IPFS hash
- Migration plan
- Backward compatibility bilgisi
- Voting deadline

**Proposal Oluşturma** (Advanced):
- "Create Proposal" butonu
- Proposal type seçimi
- Title ve description
- Deposit amount (minimum requirement)
- Submit proposal

**Governance History**:
- Geçmiş proposal'lar
- Voting history (kullanıcının oyları)
- Proposal outcomes

---

## API Katmanı (RPC vs REST)

### Endpoint Stratejisi

**Web Dashboard (Public)**:
- **Public RPC Endpoints**: Halka açık sunuculara bağlanır
  - RPC: `https://rpc.r3mes.network` (port 26657)
  - REST: `https://api.r3mes.network` (port 1317)
  - gRPC: `grpc.r3mes.network:9090`

**Desktop Launcher (Local)**:
- **Local Endpoints**: Kendi içindeki local node'a bağlanır
  - RPC: `http://localhost:${BLOCKCHAIN_RPC_PORT:-26657}` (Tendermint RPC, configurable via `BLOCKCHAIN_RPC_PORT`)
  - REST: `http://localhost:${BLOCKCHAIN_REST_PORT:-1317}` (Cosmos SDK REST API, configurable via `BLOCKCHAIN_REST_PORT`)
  - gRPC: `localhost:${BLOCKCHAIN_GRPC_PORT:-9090}` (configurable via `BLOCKCHAIN_GRPC_PORT`)

### CORS Ayarları

**Backend (Go) CORS Configuration**:
```go
// remes/app/app.go
func (app *App) setCORS() {
    corsMiddleware := cors.New(cors.Options{
        AllowedOrigins: []string{
            `http://localhost:${FRONTEND_PORT:-3000}`,      // Local development (configurable via `FRONTEND_PORT`)
            "https://dashboard.r3mes.network",  // Production
        },
        AllowedMethods: []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
        AllowedHeaders: []string{"Content-Type", "Authorization"},
        AllowCredentials: true,
    })
}
```

**Frontend (Next.js) Configuration**:
```typescript
// next.config.js
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/blockchain/:path*',
        destination: `${process.env.NEXT_PUBLIC_API_URL || 'https://api.r3mes.network'}/:path*`,
      },
    ];
  },
};
```

### Port Usage

| Service | Port | Protocol | Usage |
|--------|------|----------|----------|
| Tendermint RPC | 26657 | HTTP | Block/transaction queries |
| Cosmos SDK REST | 1317 | HTTP | REST API endpoints |
| gRPC | 9090 | gRPC | gRPC queries |
| WebSocket | 1317/ws | WebSocket | Real-time streaming |
| IPFS API | 5001 | HTTP | IPFS operations |
| Miner Stats | 8080 | HTTP | Miner statistics |
| Web Dashboard | 3000 | HTTP | Next.js frontend |

---

## Error Handling

### Global Toast Notification System

**Toast Component**:
```typescript
// components/Toast.tsx
interface Toast {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number; // milliseconds
}

// Toast notification examples
toast.success({
  title: 'İşlem Başarılı',
  message: 'Transaction hash: 0x1234...5678'
});

toast.error({
  title: 'Hata',
  message: 'Yetersiz bakiye. Lütfen cüzdanınızı kontrol edin.'
});

toast.warning({
  title: 'Uyarı',
  message: 'Zincir yanıt vermiyor. Lütfen tekrar deneyin.'
});
```

### Hata Senaryoları ve Çözümleri

**1. Blockchain Endpoint Yanıt Vermiyor**:
- Toast: "Zincir yanıt vermiyor. Lütfen bağlantınızı kontrol edin."
- Retry butonu
- Fallback endpoint'e geçiş

**2. Yetersiz Bakiye**:
- Toast: "Yetersiz bakiye. Gerekli: 1.5 REMES, Mevcut: 0.8 REMES"
- "Add Funds" butonu (Faucet linki)

**3. Transaction Başarısız**:
- Toast: "İşlem başarısız: {error_message}"
- Transaction hash gösterimi (explorer linki)
- Retry butonu

**4. Wallet Bağlantı Hatası**:
- Toast: "Cüzdan bağlantısı başarısız. Lütfen Keplr/Leap kurulu olduğundan emin olun."
- "Install Wallet" linki

**5. Network Timeout**:
- Toast: "Ağ zaman aşımı. Lütfen tekrar deneyin."
- Auto-retry (3 kez)
- Manual retry butonu

### Error Boundary

```typescript
// components/ErrorBoundary.tsx
class ErrorBoundary extends React.Component {
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log to error tracking service (Sentry)
    toast.error({
      title: 'Unexpected Error',
      message: 'Please refresh the page or contact the support team.'
    });
  }
}
```

---

## 📚 Related Documentation

- [Quick Start Guide](QUICK_START.md) - Get started with R3MES in 5 minutes
- [Desktop Launcher Documentation](10_desktop_launcher.md) - Desktop Launcher features and comparison
- [User Onboarding Guides](09_user_onboarding_guides.md) - Complete user guides for all roles
- [API Reference](13_api_reference.md) - API documentation for integration
- [Installation Guide](INSTALLATION.md) - Installation instructions

---

This Web Dashboard plays a critical role as the user-facing interface of the R3MES ecosystem and provides an optimized experience for both technical users (miners) and business stakeholders (investors).

---

**Last Updated**: 2025-01-15  
**Maintained by**: R3MES Development Team