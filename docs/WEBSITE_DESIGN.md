# R3MES Web Experience Design

**Tarih**: 2025-12-16  
**Durum**: Tasarım onayı bekliyor  

---

## 1. Vizyon ve Tasarım İlkeleri

### 1.1 Vizyon

R3MES web deneyimi; blockchain, federated training ve yapay zekayı birleştiren mimariyi:

- Geliştiriciler için **anlaşılır ve derin**,  
- Node/miner operatorleri için **operasyonel olarak kullanışlı**,  
- Token holder / topluluk için **güven veren ve etkileyici**,  

şekilde sunan **tek birleşik yüz** olacak: landing → dökümantasyon → canlı dashboard.

### 1.2 Tasarım İlkeleri

- **Bilim kurgu ama ciddi**: Karanlık arka plan, neon vurgular, fakat kurumsal güven duygusu.
- **Veri odaklı**: Her sayfada gerçek metrikler, grafikler ve şeffaf ekonomi.
- **Anlaşılır akış**: “Meraklı ziyaretçi → okuyan geliştirici → node kuran operator → katkı veren miner/validator”.
- **Modüler**: Landing, docs ve dashboard aynı tasarım sistemiyle çalışır.
- **Performans + Erişilebilirlik**: Three.js animasyonları olsa bile düşük cihazlar için graceful degradation.

---

## 2. Bilgi Mimarisi (Information Architecture)

### 2.1 Ana Bölümler

Üst navigasyon:

- **Home**: Landing page, three.js sahnesi, ürün anlatımı.
- **Protocol**: Mimari, ekonomik model, güvenlik özetleri (yüksek seviye, teknik ama “marketing” sunum).
- **Docs**: Tüm teknik dökümantasyon (`docs/` + `remes/docs` içeriği).
- **Dashboard**: Mevcut `web-dashboard` uygulamasına giden giriş noktası.
- **Developers**: Quickstart, API/gRPC/CLI referansları, SDK’lar.
- **Community**: Discord/Telegram, GitHub, governance forum, blog / changelog.

### 2.2 Site Haritası (Sitemap - Özet)

- `/` – Home (landing + three.js sahnesi)
  - Hero (3D network + AI animasyonu)
  - “Why R3MES?” (3 temel sütun)
  - “How it Works” (3 adım diyagram)
  - “For Miners / Validators / Developers” bölümü
  - Canlı metrik snippet’leri (toplam miner, toplam stake, son blok süresi)
  - Call-to-Action: “Run a Miner”, “Read the Docs”, “Open Dashboard”

- `/protocol`
  - Overview
  - Security & Verification (three-layer optimistic, trap jobs)
  - Economics (rewards, treasury, inference fees)
  - Architecture (sharding, async rollup, Iron Sandbox)

- `/docs`
  - `/docs` – Docs ana sayfa
  - `/docs/getting-started`
  - `/docs/node-operators`
  - `/docs/miner-engine`
  - `/docs/protocol-design`
  - `/docs/economics`
  - `/docs/security-and-verification`
  - `/docs/web-and-tools`

- `/dashboard` (veya alt alan: `app.r3mes.io`)
  - Overview
  - Network
  - Miners
  - Datasets
  - Governance
  - Node Ops

- `/developers`
  - Quickstart (Python miner, Go node, Keplr + CosmJS)
  - API & gRPC
  - Examples & Templates

- `/community`
  - Links, roadmap, changelog, governance süreçleri.

---

## 3. Landing Page Tasarımı

### 3.1 Hero Bölümü (Three.js + Mesaj)

**Amaç**: “R3MES = AI training + blockchain + global GPU ağı” mesajını ilk 5 saniyede vermek.

**Layout**:

- Sol: Büyük başlık + kısa açıklama + 2 CTA butonu.
  - Başlık: “Decentralized AI Training on a Verifiable GPU Network”
  - Alt metin: R3MES’in ne yaptığına dair 2–3 satırlık teknik ama sade açıklama.
  - CTA 1: “Run a Miner”
  - CTA 2 (ghost): “Explore the Protocol”
- Sağ: **Three.js sahnesi** (React Three Fiber) – blockchain + yapay zeka temalı animasyon.

**Three.js Sahne Konsepti**:

- Karanlık uzay / grid arka plan.
- Ortada hafif dönen **3D dünya/globe**:
  - Noktalar: aktif miner/validator node’larını temsil eder.
  - Node’lar arasında hafif parlayan “gradient akış” çizgileri (AI training flows).
- Globe çevresinde halka gibi duran **blok zinciri**:
  - Küçük 3D bloklar (chain), yavaşça ilerleyen blok animasyonu.
  - Her yeni blokta globe üzerindeki bazı bağlantılar hafif parlayarak “on-chain verification” hissi verir.
- Yüzeyde şeffaf bir **nöral ağ overlay’i**:
  - Katmanlar halinde noktalar + çizgiler (neural net), yumuşak parıltı animasyonları.

**Etkileşim / Performans**:

- Hover: Mouse hareketine göre globe hafif döner.
- Click: Bazı node’lar highlight olup tooltip gibi “Miner”, “Validator”, “Serving Node” rolleri gösterilebilir (ileriki faz).
- Teknoloji: `@react-three/fiber`, `@react-three/drei`, instancing ile düşük poly.
- Mobile için fallback:
  - Daha basit, statik veya düşük detaylı animasyon (veya video/Lottie).
  - Kullanıcının cihaz/performans durumuna göre `prefers-reduced-motion` desteği.

### 3.2 “Why R3MES?” – 3 Sütun

Üç ana sütun:

- **Verifiable AI Training**
  - Three-layer optimistic verification, trap jobs, Iron Sandbox.
- **Efficient & Scalable**
  - Bandwidth optimizations, sharding, async rollup, convergence monitoring.
- **Fair & Sustainable Economics**
  - Role-specific rewards, treasury buy-back, inference fees.

Her sütun için:

- İkon (line-art, minimal).
- Kısa başlık + 2–3 satır açıklama.

#### 3.2.1 BitNet Bandwidth Görselleştirmesi (50 GB vs 200 MB)

**Amaç**: BitNet + LoRA sayesinde **“50 GB model” → “200 MB adapter”** dönüşümünü kullanıcıya **anında ve görsel olarak** anlatmak (USP vurgusu).

**Bileşen Tasarımı**:

- Yatay olarak ikiye bölünmüş bir alan:
  - **Sol panel (Geleneksel / Render / Gensyn)**:
    - Three.js içinde büyük, ağır bir küp (yüksek poligon, koyu renk).
    - Küp yüklenirken yavaşça dolan bir progress ring / iskelet animasyonu.
    - Üstte text: “50 GB Monolithic Model”.
    - Altta küçük not: “Long sync times, high bandwidth”.
  - **Sağ panel (R3MES / BitNet LoRA)**:
    - Küçük, parlak bir küp (düşük poly, neon çizgiler).
    - Küp sahneye neredeyse anında “fade-in” ile gelir.
    - Üstte text: “200 MB BitNet LoRA Adapter”.
    - Altta not: “Fast sync, low bandwidth”.

**Etkileşim**:

- Hover edildiğinde her panelde:
  - Sol: Küp daha da ağır, sanki “lag” varmış gibi hafif gecikmeli döner.
  - Sağ: Küp “snappy” bir hareketle döner, glow artar.
- Alt kısımda opsiyonel bir slider / toggle:
  - “Full Model” ↔ “LoRA Adapter”
  - Slider konumuna göre sol/sağ panelin opaklığı değişir.

**Metin / Copy Örneği**:

- Üst başlık: “Train Once, Ship a 200 MB Adapter.”
- Alt açıklama:  
  “R3MES, BitNet LoRA ile 50 GB’lık modelleri ~200 MB’lık adapter’lara indirerek  
  federated öğrenme ve blokzincir üzerinde doğrulanabilir eğitim için gereken bant genişliğini dramatik şekilde azaltır.”

### 3.3 “How It Works” – 3 Adım Diyagram

Basit adımlar:

1. **Contribute GPU Power** – Miner node kurulumu, gradient üretimi.
2. **On-Chain Verification & Aggregation** – Three-layer optimistic + Merkle + trap jobs.
3. **Serve Inference & Earn** – Serving nodes, inference fees, treasury & miners rewards.

Bu bölümde, küçük pseudo-schematics / çizgisel diyagramlar, card’lar ve oklarla akış gösterilir.

### 3.4 Hedef Kitle Kartları

Üç card:

- **For Miners & Validators**
  - “Start Mining”, “View Rewards Model”, “Read Security Guarantees”.
- **For Developers**
  - “Quickstart”, “API & gRPC”, “SDKs”.
- **For Researchers & Governance**
  - “Read the Whitepaper / Design Docs”, “View Dataset Governance”, “Explore Model Versioning”.

Her kartın altındaki butonlar ilgili docs veya dashboard bölümlerine götürür.

### 3.5 Canlı Metrik Şeridi

Landing içinde, back-end’den gelen gerçek metrikler:

- Aktif miner sayısı
- Toplam stake / toplam GPU kapasitesi
- Son blok süresi / ortalama blok süresi
- Son 24 saatte işlenen inference istekleri

Basit, büyük fontlu, card’lar halinde; alt metinde “Data from R3MES mainnet/testnet” vurgusu.

### 3.6 Footer

- Linkler: Docs, Dashboard, GitHub, Discord, X (Twitter), Blog.
- Telif & versiyon: “R3MES Protocol – v0.x.x”.

---

## 4. Three.js ve Görsel Dil

### 4.1 Ana Hero Sahnesi

- Globe + chain + neural net overlay.
- Renkler:
  - Node’lar: açık mavi / cyan.
  - Zincir blokları: mor tonlarında.
  - Gradient akışları: sarımsı / turuncu highlight.

### 4.2 Dashboard için Hafif 3D Dokunuşlar (Opsiyonel)

- Network Explorer zaten 3D globe kullanıyor.
- Aynı görsel dili korumak için:
  - Glow ve renk paleti hero sahnesiyle uyumlu.
  - Hover highlight, seçili node, path drawing efektleri.

### 4.3 Performans ve Fallback Stratejisi

- Hero sahnesi lazy-load edilir; ilk önce metin ve CTA görünür.
- `prefers-reduced-motion` tespiti ile animasyon tamamen kapatılabilir.
- Düşük GPU cihazlarda:
  - Rotation yavaşlatılır,
  - Parçacık sayısı azaltılır,
  - Gerekirse sadece pre-rendered video/GIF gösterilir.

---

## 5. Docs Bölümü Tasarımı

### 5.1 Genel Yapı

- Sol tarafta **sabit bir sidebar**:
  - Bölümler: Getting Started, Node Operators, Miner Engine, Protocol, Economics, Security, Web & Tools.
  - Sidebar, mevcut `docs/*.md` ve `remes/docs` içeriğini mantıksal gruplara ayırır.
- Sağda içerik + üstte breadcrumb:
  - Örn: `Docs / Node Operators / Running a Validator`.

### 5.2 Docs Ana Sayfası

- “Start Here” kartları:
  - Run a Node
  - Run a Miner
  - Understand the Protocol
  - Integrate a Wallet / dApp
- “What’s New” bölümü:
  - Son eklenen/güncellenen dokümanlar (changelog datasından veya el ile).

### 5.3 Tasarım Detayları

- **Okunabilirlik**:
  - Maksimum satır genişliği,
  - Büyük başlık aralıkları,
  - Kod blokları için koyu arka plan + monospaced font.
- **Arama (search)**:
  - İlk fazda basit istemci tarafı (full-text index) veya Algolia benzeri.
  - Header’da global search input.
- **Versiyon Switcher**:
  - Sağ üstte “v0.1 / v0.2 / dev” seçici.
  - Eski versiyonlar “archived” etiketiyle işaretlenir.

### 5.4 Config Generator / Setup Wizard

**Motivasyon**: `TLS_SETUP.md` içinde görülen OpenSSL, sertifika ve `.pem` kurulumu gibi adımlar; terminalde kaybolmaya çok açık. Amacımız, özellikle ilk defa node/miner kuran kullanıcıyı **görsel bir sihirbaz** ile yönlendirmek.

**Konum**:

- `/developers` veya `/docs/node-operators` altında “Setup Wizard” sayfası.
- Landing’de “Run a Miner” CTA’sı bu sihirbaza da yönlendirilebilir.

**Adım Adım Akış (Örnek)**:

1. **Node Tipi Seçimi**
   - Card’lar: “Full Node”, “Validator”, “Miner Node”, “Serving Node”.
   - Her kartta kısa açıklama ve temel gereksinimler (CPU, RAM, GPU, disk).
2. **Ağ ve Endpoint Seçimi**
   - Mainnet / Testnet / Localnet seçenekleri.
   - RPC / gRPC / REST endpoint alanları (varsayılan öneriler ile).
3. **TLS & Güvenlik**
   - “Use TLS for gRPC / REST” checkbox’ları.
   - Sertifika üretimi için:
     - “Benim mevcut sertifikam var” (path input’ları).
     - “Self-signed oluştur” (otomatik komut seti veya script çıktısı).
   - Çıktı olarak: `TLS_SETUP.md`’deki env değişkenlerine ve default path’lere uygun config.
4. **Port ve Kaynak Ayarları**
   - gRPC/REST portları, maksimum bağlantı sayısı, log seviyesi.
   - Miner için GPU sayısı / target utilization gibi temel ayarlar.
5. **Özet ve Çıktı**
   - Kullanıcının seçtiği her şeyin özetini göster.
   - **İki tip çıktı**:
     - `config.toml` / `app.toml` benzeri node config snippet’i (indirme butonu).
     - `setup.sh` veya ilgili platform için script (örneğin systemd servis dosyası iskeleti).

**Teknik Not**:

- Wizard, backend tarafında:
  - Var olan `config` şemalarına (Go tarafındaki struct’lara) uygun JSON oluşturur.
  - TLS için:
    - Env var’lar (`REMES_GRPC_TLS_CERT`, `REMES_GRPC_TLS_KEY` vs.) ve default path yapısı ile uyumlu olacak şekilde örnekler üretir.

---

## 6. Dashboard Tasarımı (Güncel ve Gelecek)

> Not: `web-dashboard` zaten var; aşağıdaki yapı, mevcut yapıyı son haline getirecek hedef tasarım.

### 6.1 Global Layout

- Sol vertical navbar:
  - Overview
  - Network
  - Miners
  - Datasets
  - Governance
  - Node Ops
  - Settings / Wallet
- Sağ ana içerik alanı:
  - Üstte breadcrumb + sayfa başlığı + zaman filtresi (24h / 7d / 30d).

### 6.2 Overview Page

- Üstte 3–4 büyük KPI card’ı:
  - Active Miners, Active Validators, Total Stake, Inference Requests (24h).
- Altında 2 sütun:
  - Sol: Ağ aktivite grafiği (blok süresi, işlem sayısı).
  - Sağ: Son bloklar / son önemli olaylar (governance proposals, dataset changes).

### 6.3 Network Page

- Üstte **3D Network Explorer** (mevcut globe):
  - Node’lar kırmızı/yeşil durumla (online/offline).
  - Filter by role (miner, validator, serving node).
- Altta tablo:
  - Node address, role, uptime, region (pseudo), last seen block, trust score.

### 6.4 Miner Console

- **Training Metrics**:
  - Loss grafiği (Recharts line/area chart).
  - Learning rate, batch size, epoch progress.
- **Hardware Monitor**:
  - GPU kullanım yüzdesi, VRAM kullanımı, sıcaklık.
- **Log Stream**:
  - Real-time WebSocket log viewer (zaten implement edilen `LogStream`).
  - Arama / filter (severity, component).

### 6.5 Datasets ve Governance

- Kart grid:
  - Dataset adı, durum (proposed / active / retired), votes, size / shards.
- Proposal detay sayfası:
  - Özet, motivasyon, parametreler,
  - Oylama durumu (bar grafik / pie chart),
  - On-chain linkler (tx hash, block height).

### 6.6 “The Leaderboard” – Şeref Kürsüsü

**Motivasyon**: `04_economic_incentives.md` içindeki **Reputation Tier** kavramını görselleştirerek, miner ve validator’lar arasında **rekabet / oyunlaştırma** etkisi yaratmak.

**Konum**:

- Dashboard içinde **Overview** sayfasının alt kısmında veya ayrı bir **Leaderboard** sekmesi olarak.
- Landing’de, küçük bir “Top Miners / Most Trusted Validators” snapshot’ı (top 3) gösterilebilir; detaylı liste dashboard’a gider.

**İçerik**:

- Tab bileşeni:
  - Tab 1: “Top Miners”
  - Tab 2: “Most Trusted Validators”
- Her satır için:
  - Sıra numarası (1, 2, 3, …) + küçük taç / madalya ikonu.
  - Adresin kısaltılmış hali + opsiyonel takma ad (varsa).
  - Reputation Tier (Bronze / Silver / Gold / Platinum / Diamond vb.).
  - Toplam katkı (submitted gradients, served inferences, uptime).
  - Son 30 günlük trend (küçük sparkline grafiği).

**Reputation Tier Görsel Dili**:

- Her tier için farklı renk ve mini badge:
  - Bronze: sıcak kahverengi / bakır.
  - Silver: gri / gümüş.
  - Gold: altın sarısı.
  - Platinum: mavi-yeşil karışımı, glow’lu.
  - Diamond: mor / cyan kombinasyonu, daha belirgin glow.
- Badge içinde kısa kod: **B**, **S**, **G**, **P**, **D**.

**Gamification Öğeleri**:

- Küçük notlar:
  - “You are #12 out of 350 miners this month.”
  - “+2 positions since last epoch.”
- Opsiyonel: Reputation artışı/değişimi için küçük ok ikonları (yukarı/aşağı).

### 6.7 Connection Lost / gRPC-Web Durumları

**Problem**: Dashboard; WebSocket + gRPC-Web üzerinden node’a bağlanıyor. Kurumsal firewall/proxy’ler bu trafik türünü zaman zaman engelleyebilir.

**Genel Davranış**:

- Global **connection status bar**:
  - Ekranın üstünde veya altında ince bir bar:
    - “Connected to R3MES node” (yeşil nokta).
    - “Reconnecting…” (sarı spinner).
    - “Connection lost – check network or proxy settings.” (kırmızı uyarı).
- Metrik kartları ve grafikler:
  - Bağlantı koptuğunda **skeleton state** veya “Last updated 2m ago” etiketi gösterir.
  - Aggressive spinner yerine sakin, düşük stresli animasyon.

**Uyarı Mesajı (Copy)**:

- Başlık: “Connection Lost to R3MES Node”
- Açıklama:
  - “Dashboard, gRPC-Web ve WebSocket üzerinden node’a bağlanıyor.  
     Bazı kurum içi firewall veya proxy ayarları bu trafiği engelleyebilir.”
- Önerilen adımlar (madde madde):
  - “İnternet bağlantınızı kontrol edin.”
  - “Eğer bir kurumsal ağ kullanıyorsanız, proxy/firewall ayarlarınızı kontrol edin.”
  - “Mümkünse doğrudan erişilebilir bir RPC endpoint’i kullanın.”

**Teknik Not (Tasarım Dokümanı İçin)**:

- gRPC-Web ve WebSocket hataları için:
  - Otomatik **backoff + retry** stratejisi (örn. 1s, 2s, 5s, 10s).
  - Belirli bir süre sonra kullanıcıya:
    - Farklı endpoint seçme (Settings / Node Ops içinden),
    - Veya “offline mode” (sadece en son cache’lenmiş veriyi göster) seçeneği sunulabilir.

---

## 7. Tasarım Sistemi (Design System)

### 7.1 Renk Paleti

- **Arka planlar**:
  - `#050815` – Ana arka plan (çok koyu lacivert/siyah).
  - `#0A1020` – Kart arka planı.
- **Ana vurgular**:
  - `#38E8B0` – Primary accent (neon yeşil/mavi arası).
  - `#4EA8FF` – Secondary accent (mavi).
  - `#B980FF` – Tertiary (mor).
- **Durum renkleri**:
  - Success: `#22C55E`
  - Warning: `#FACC15`
  - Error: `#EF4444`
- **Metin**:
  - Ana metin: `#E5E7EB`
  - Zayıf/ikincil metin: `#9CA3AF`

### 7.2 Tipografi

- Başlık fontu: **Space Grotesk** (veya benzeri futuristic sans-serif).
- Gövde metni: **Inter**.
- Kod: **JetBrains Mono** veya **Fira Code**.

Örnek hiyerarşi:

- H1 (Landing hero): 42–56px, kalın.
- H2 (Bölüm başlığı): 28–32px.
- H3 (Card başlığı): 20–22px.
- Body: 16px.

### 7.3 Bileşenler

- **Butonlar**:
  - Primary: dolu, gradient veya düz accent renkli.
  - Secondary: outline, hafif transparan arka plan.
- **Kartlar**:
  - Hafif blur + border (`border-[rgba(255,255,255,0.08)]`).
  - Hover’da subtle glow.
- **Tag / Badge**:
  - Status (Active, Testnet, Deprecated).
  - Role (Miner, Validator, Serving, DA).
- **Tablar**:
  - Dashboard’ta alt nav için (örneğin Governance içinde “Proposals”, “Params”, “History”).

---

## 8. Erişilebilirlik ve Performans

- Kontrast oranı en az WCAG AA.
- Klavye ile gezinme (focus ring’ler).
- Animasyonlar için:
  - `prefers-reduced-motion` desteği.
  - Maksimum animasyon süresi kısa, loop’lar yumuşak.
- Three.js:
  - FPS sınırı (örn. 30 FPS).
  - Node sayısı ve efektler progressive enhancement ile artar/azalır.

---

## 9. Uygulama Planı (Sadece Yüksek Seviye)

### Faz 1 – Temel Site ve Tasarım Sistemi

- Mevcut `web-dashboard` tabanını kullanarak:
  - Kökte yeni bir **marketing / website layout** oluşturma.
  - Global theme, renkler, tipografi ve temel bileşenler.
- `/` landing sayfasını statik hero + placeholder illüstrasyon ile yayınlama.
- `/docs` sayfası için iskelet (sidebar + içerik renderi, mevcut `.md` dosyalarına link).

### Faz 2 – Three.js Hero ve Canlı Metrikler

- React Three Fiber ile hero sahnesinin ilk sürümü.
- Node’dan gelen gerçek metrikler için basit API endpoint’leri:
  - Ağ istatistikleri, aktif node sayısı, son blok süresi.

### Faz 3 – Docs ve Dashboard Entegrasyonu

- Docs için arama, versiyon seçici, kategori düzenlemesi.
- Dashboard linklerinin landing ve docs içinden mantıklı noktalara yerleştirilmesi.

### Faz 4 – Governance & Advanced UX

- Governance ekranlarının Web Dashboard’da tamamlanması.
- Landing / docs tarafında governance için rehber ve interaktif akışlar (wizard benzeri formlar).

---

## 10. Onay ve Sonraki Adımlar

- Bu doküman **görsel dil + bilgi mimarisi + sayfa bazlı layout** için temel rehberdir.
- Onaylandıktan sonra:
  - Faz 1 ve 2 için detaylı **task listesi** çıkarılacak,
  - `web-dashboard` içinde yeni route ve layout’lar tanımlanacak,
  - Three.js sahnesi için ayrı bir teknik tasarım (scene graph, component yapısı) hazırlanacak.


