# R3MES Backend Inference Service

## Genel Bakış

R3MES Backend Inference Service, eğitilmiş BitNet modellerini ve LoRA adaptörlerini kullanarak kullanıcılara AI inference hizmeti sunan bir FastAPI tabanlı servistir. Sistem, Multi-LoRA desteği, kredi tabanlı ekonomi ve akıllı yönlendirme özellikleri içerir.

## Proje Yapısı

```
/backend
    /app
        main.py                 # FastAPI uygulaması
        model_manager.py        # AI Modeli ve Adaptör Yönetimi (Multi-LoRA)
        database_async.py       # Async Kredi ve Cüzdan veritabanı (SQLite + aiosqlite)
        database.py             # Legacy sync database (deprecated, async kullanılıyor)
        semantic_router.py      # Embedding-based intelligent routing (aktif)
        router.py               # Keyword-based router (fallback, deprecated)
        inference_executor.py   # ThreadPoolExecutor ile CPU/GPU işlemleri
        config_manager.py       # Merkezi yapılandırma yönetimi
        config_endpoints.py     # Config API endpoints
        setup_logging.py        # Structured logging sistemi
        task_queue.py           # Task queue for load balancing
    /checkpoints                # Model dosyaları buraya gelecek
    /tests                      # Unit tests
        test_database.py        # Database testleri
```

## 🛠️ Adım 1: Model Yöneticisi (Multi-LoRA Engine)

### Amaç

Sunucu açıldığında Ana Modeli (BitNet) yüklesin ve istek geldiğinde "Tak-Çıkar" adaptörleri yönetebilsin.

### Implementasyon

**Dosya**: `backend/app/model_manager.py`

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
import torch
from typing import Optional, Dict, Iterator

class AIModelManager:
    """
    Multi-LoRA Model Manager
    
    Özellikler:
    - BitNet Base Model'i düşük VRAM modunda yükler
    - LoRA adaptörlerini dinamik olarak yükler/kaldırır
    - Adaptörler arasında geçiş yapar
    - Streaming response üretir
    """
    
    def __init__(self, base_model_path: str = "checkpoints/base_model"):
        """
        Başlangıçta BitNet Base Model'i düşük VRAM modunda yükler.
        
        Args:
            base_model_path: Base model dosya yolu
        """
        # Quantization config (4-bit loading for low VRAM)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load base model with quantization
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Active adapters registry
        self.adapters: Dict[str, PeftModel] = {}
        self.active_adapter: Optional[str] = None
        
        print("✅ Base model loaded with 4-bit quantization")
    
    def load_adapter(self, name: str, path: str) -> bool:
        """
        Belirtilen yoldaki LoRA adaptörünü modele ekler (modeli kapatmadan).
        
        Args:
            name: Adaptör adı (örn: "coder_adapter", "law_adapter")
            path: Adaptör dosya yolu
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load LoRA adapter
            adapter = PeftModel.from_pretrained(
                self.base_model,
                path,
                adapter_name=name
            )
            
            self.adapters[name] = adapter
            print(f"✅ Adapter '{name}' loaded from {path}")
            return True
        except Exception as e:
            print(f"❌ Failed to load adapter '{name}': {e}")
            return False
    
    def switch_adapter(self, name: str) -> bool:
        """
        Aktif adaptörü değiştirir.
        
        Args:
            name: Adaptör adı
            
        Returns:
            True if successful, False otherwise
        """
        if name not in self.adapters:
            print(f"❌ Adapter '{name}' not found")
            return False
        
        # Switch active adapter
        self.base_model.set_adapter(name)
        self.active_adapter = name
        print(f"✅ Switched to adapter '{name}'")
        return True
    
    def generate_response(
        self, 
        prompt: str, 
        adapter_name: Optional[str] = None
    ) -> Iterator[str]:
        """
        İlgili adaptörü aktif edip cevabı stream (akış) olarak üretir.
        
        Args:
            prompt: Kullanıcı sorusu
            adapter_name: Kullanılacak adaptör adı (None ise aktif adaptör kullanılır)
            
        Yields:
            Token strings (streaming)
        """
        # Switch adapter if specified
        if adapter_name and adapter_name != self.active_adapter:
            if not self.switch_adapter(adapter_name):
                yield f"Error: Adapter '{adapter_name}' not available"
                return
        
        # Tokenize input
        inputs = self.base_model.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.base_model.device)
        
        # Generate with streaming
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                streamer=None  # Manual streaming
            )
        
        # Decode and stream tokens
        generated_text = self.base_model.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Stream character by character
        for char in generated_text:
            yield char
```

## 🛠️ Adım 2: Veritabanı ve Kredi Sistemi (Economy)

### Amaç

Kimin ne kadar kredisi var, kim madenci? Bunu tutan basit ama sağlam bir yapı.

### Implementasyon

**Dosya**: `backend/app/database_async.py` (Async SQLite wrapper)

**Önemli:** Artık `database_async.py` kullanılıyor, `database.py` legacy olarak kaldı.

```python
import aiosqlite
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
import os
import logging

logger = logging.getLogger(__name__)

class AsyncDatabase:
    """
    Async SQLite Database wrapper
    
    All database operations are now async to prevent blocking the event loop.
    Features:
    - WAL mode enabled for better concurrency
    - API key hashing (SHA256)
    - Async operations with aiosqlite
    """
    
    def __init__(self, db_path: Optional[str] = None, chain_json_path: Optional[str] = None):
        """
        Initialize async database.
        
        Args:
            db_path: SQLite database file path
            chain_json_path: Blockchain JSON file path
        """
        if db_path is None:
            db_path = os.getenv("DATABASE_PATH", "backend/database.db")
        
        if chain_json_path is None:
            chain_json_path = os.getenv("CHAIN_JSON_PATH", "chain.json")
        
        self.db_path = db_path
        self.chain_json_path = chain_json_path
        self._connection: Optional[aiosqlite.Connection] = None
        self.rpc_endpoint = os.getenv("BLOCKCHAIN_RPC_URL", "http://localhost:26657")
        
        # Create database directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def connect(self):
        """Connect to database and enable WAL mode."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(self.db_path)
            # Enable WAL mode for better concurrency
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA foreign_keys=ON")
            await self._init_database()
            logger.info(f"Database connected: {self.db_path}")
    
    async def close(self):
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None
            logger.info("Database connection closed")
    
    async def _init_database(self):
        """Initialize database tables."""
        if not self._connection:
            await self.connect()
        
        # Users table
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                wallet_address TEXT PRIMARY KEY,
                credits REAL DEFAULT 0.0,
                is_miner BOOLEAN DEFAULT 0,
                last_mining_time TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # API keys table (with hashing)
        await self._connection.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wallet_address TEXT NOT NULL,
                api_key_hash TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                last_used TIMESTAMP,
                FOREIGN KEY (wallet_address) REFERENCES users(wallet_address)
            )
        """)
        
        await self._connection.commit()
        logger.info("Database tables initialized")
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key using SHA256."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    async def add_credits(self, wallet: str, amount: float) -> bool:
        """Add credits to user (async)."""
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("SELECT credits FROM users WHERE wallet_address = ?", (wallet,))
        result = await cursor.fetchone()
        
        if result:
            new_credits = result[0] + amount
            await cursor.execute(
                "UPDATE users SET credits = ? WHERE wallet_address = ?",
                (new_credits, wallet)
            )
        else:
            await cursor.execute(
                "INSERT INTO users (wallet_address, credits) VALUES (?, ?)",
                (wallet, amount)
            )
        
        await self._connection.commit()
        return True
    
    async def deduct_credit(self, wallet: str, amount: float) -> bool:
        """Deduct credits from user (async)."""
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("SELECT credits FROM users WHERE wallet_address = ?", (wallet,))
        result = await cursor.fetchone()
        
        if not result:
            return False
        
        current_credits = result[0]
        if current_credits < amount:
            return False
        
        new_credits = current_credits - amount
        await cursor.execute(
            "UPDATE users SET credits = ? WHERE wallet_address = ?",
            (new_credits, wallet)
        )
        
        await self._connection.commit()
        return True
    
    async def check_credits(self, wallet: str) -> float:
        """Check user credits (async)."""
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("SELECT credits FROM users WHERE wallet_address = ?", (wallet,))
        result = await cursor.fetchone()
        
        return result[0] if result else 0.0
    
    async def get_user_info(self, wallet: str) -> Optional[Dict]:
        """Get user information (async)."""
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute(
            "SELECT wallet_address, credits, is_miner FROM users WHERE wallet_address = ?",
            (wallet,)
        )
        result = await cursor.fetchone()
        
        if not result:
            return None
        
        return {
            "wallet_address": result[0],
            "credits": result[1],
            "is_miner": bool(result[2]),
        }
    
    async def create_api_key(
        self,
        wallet: str,
        name: str = "Default",
        expires_in_days: Optional[int] = None
    ) -> str:
        """Create new API key (returns plaintext key, stores hash)."""
        if not self._connection:
            await self.connect()
        
        # Generate API key
        api_key = f"r3mes_{secrets.token_urlsafe(32)}"
        api_key_hash = self._hash_api_key(api_key)
        
        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = (datetime.now() + timedelta(days=expires_in_days)).isoformat()
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("""
            INSERT INTO api_keys (wallet_address, api_key_hash, name, expires_at)
            VALUES (?, ?, ?, ?)
        """, (wallet, api_key_hash, name, expires_at))
        
        await self._connection.commit()
        
        # Return plaintext key (only shown once)
        return api_key
    
    async def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key (async, using hash)."""
        if not self._connection:
            await self.connect()
        
        # Hash the provided API key
        api_key_hash = self._hash_api_key(api_key)
        
        cursor = await self._connection.cursor()
        
        await cursor.execute("""
            SELECT id, wallet_address, name, is_active, expires_at
            FROM api_keys
            WHERE api_key_hash = ? AND is_active = 1
        """, (api_key_hash,))
        
        result = await cursor.fetchone()
        
        if not result:
            return None
        
        # Check expiration
        expires_at = result[4]
        if expires_at:
            expires_dt = datetime.fromisoformat(expires_at)
            if datetime.now() > expires_dt:
                return None
        
        # Update last_used
        await cursor.execute(
            "UPDATE api_keys SET last_used = ? WHERE id = ?",
            (datetime.now().isoformat(), result[0])
        )
        await self._connection.commit()
        
        return {
            "id": result[0],
            "wallet_address": result[1],
            "name": result[2],
            "is_active": bool(result[3]),
        }
    
    async def get_network_stats(self) -> Dict:
        """Get network statistics (async)."""
        if not self._connection:
            await self.connect()
        
        cursor = await self._connection.cursor()
        
        # Count active miners
        await cursor.execute("SELECT COUNT(*) FROM users WHERE is_miner = 1")
        active_miners = (await cursor.fetchone())[0]
        
        # Count total users
        await cursor.execute("SELECT COUNT(*) FROM users")
        total_users = (await cursor.fetchone())[0]
        
        # Total credits in system
        await cursor.execute("SELECT SUM(credits) FROM users")
        total_credits = (await cursor.fetchone())[0] or 0.0
        
        return {
            "active_miners": active_miners,
            "total_users": total_users,
            "total_credits": total_credits
        }
```

## 🛠️ Adım 3: Akıllı Yönlendirici (Semantic Router)

### Amaç

Kullanıcının sorusuna bakıp "Bu kod sorusu, Kodcu Adaptörü çağır" diyen mantık. Artık embedding-based semantic similarity kullanılıyor.

### Implementasyon

**Dosya**: `backend/app/semantic_router.py`

**Durum**: Semantic router mevcut ancak varsayılan olarak **kapalıdır** (`USE_SEMANTIC_ROUTER=false`). Aktif etmek için environment variable'ı `true` yapın.

**Önemli:** Semantic router embedding model gerektirdiği için `sentence-transformers` kütüphanesine ihtiyaç duyar. Varsayılan olarak keyword-based router kullanılır.

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Optional

class SemanticRouter:
    """
    Semantic Router - Embedding-based intelligent routing
    
    Her adaptör için örnek prompt'ları embedding'e çevirir ve
    kullanıcı mesajının semantic similarity'sine göre en uygun adaptörü seçer.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, use_semantic: bool = True):
        """
        Semantic Router'ı başlat.
        
        Args:
            similarity_threshold: Minimum similarity skoru (0.0-1.0)
            use_semantic: Semantic router kullanılsın mı? (Always True - semantic router is mandatory)
        """
        self.use_semantic = use_semantic
        self.similarity_threshold = similarity_threshold
        
        # NOTE: Router (keyword-based) has been deprecated and removed
        # SemanticRouter is now mandatory - no fallback available
        
        # Embedding model (lazy loading)
        self.embedding_model: Optional[SentenceTransformer] = None
        
        # Route definitions (her adaptör için örnek prompt'lar)
        self.route_definitions: Dict[str, List[str]] = {
            'coder_adapter': [
                "How do I write a Python function?",
                "What's the syntax for JavaScript classes?",
                "How to debug this code error?",
                "Explain this algorithm step by step",
                "Fix this SQL query bug",
            ],
            'law_adapter': [
                "What are my legal rights?",
                "Explain this contract clause",
                "How does copyright law work?",
                "What is intellectual property?",
            ],
            'default_adapter': [
                "Hello, how can you help me?",
                "What is R3MES?",
                "Tell me about yourself",
            ]
        }
        
        # Pre-computed embeddings (lazy loaded)
        self.route_embeddings: Dict[str, np.ndarray] = {}
    
    def _load_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
    
    def _compute_route_embeddings(self):
        """Pre-compute embeddings for route definitions."""
        if not self.route_embeddings:
            self._load_embedding_model()
            
            for adapter_name, examples in self.route_definitions.items():
                if examples:
                    embeddings = self.embedding_model.encode(examples)
                    # Average embeddings for each adapter
                    self.route_embeddings[adapter_name] = np.mean(embeddings, axis=0)
    
    def decide_adapter(self, prompt: str) -> str | Tuple[str, float]:
        """
        Mesajı analiz edip uygun adaptörü seçer.
        
        Args:
            prompt: Kullanıcı mesajı
            
        Returns:
            Adaptör adı (veya (adapter_name, similarity_score) tuple)
        """
        # Semantic router is mandatory - no fallback
        if not self.use_semantic:
            raise RuntimeError("SemanticRouter is required. use_semantic must be True.")
        
        # Use semantic similarity
        self._compute_route_embeddings()
        self._load_embedding_model()
        
        # Encode user prompt
        prompt_embedding = self.embedding_model.encode([prompt])[0]
        
        # Calculate cosine similarity with each route
        best_adapter = 'default_adapter'
        best_similarity = 0.0
        
        for adapter_name, route_embedding in self.route_embeddings.items():
            similarity = np.dot(prompt_embedding, route_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(route_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_adapter = adapter_name
        
        # Return tuple if similarity is above threshold, otherwise default
        if best_similarity >= self.similarity_threshold:
            return (best_adapter, best_similarity)
        else:
            return (best_adapter, best_similarity)  # Still return tuple for consistency
```

## 🛠️ Adım 4: API Endpointleri (Web Sitesi Kapısı)

### Amaç

Web sitesinin (Frontend) bağlanacağı kapıları açmak.

### Implementasyon

**Dosya**: `backend/app/main.py`

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from model_manager import AIModelManager
from database_async import AsyncDatabase
from semantic_router import SemanticRouter
from inference_executor import get_inference_executor, shutdown_inference_executor
from setup_logging import setup_logging
from config_manager import get_config_manager
from config_endpoints import router as config_router

# Setup logging first
setup_logging(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true"
)

logger = logging.getLogger(__name__)

# Load configuration
config_manager = get_config_manager()
config = config_manager.load()

app = FastAPI(title="R3MES Inference Service", version="1.0.0")

# Initialize components using config
base_model_path = config.base_model_path
model_manager = AIModelManager(base_model_path=base_model_path)

database = AsyncDatabase(db_path=config.database_path, chain_json_path=config.chain_json_path)

# Router selection: SemanticRouter is now mandatory (Router deprecated and removed)
# SemanticRouter provides better accuracy with embedding-based routing
similarity_threshold = float(os.getenv("SEMANTIC_ROUTER_THRESHOLD", "0.7"))

logger.info("Initializing Semantic Router (embedding-based)...")
try:
    router = SemanticRouter(
        similarity_threshold=similarity_threshold,
        use_semantic=True
    )
    logger.info("✅ Semantic Router initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Semantic Router: {e}")
    logger.error("SemanticRouter is required. Please ensure all dependencies are installed.")
    logger.error("Required dependencies: sentence-transformers")
    raise RuntimeError(
        "SemanticRouter initialization failed. "
        "This is a required component. Please check your environment and dependencies."
    ) from e

# Initialize database connection on startup
@app.on_event("startup")
async def startup_event():
    await database.connect()
    max_workers = int(os.getenv("MAX_WORKERS", "1"))
    await get_inference_executor(max_workers=max_workers)

@app.on_event("shutdown")
async def shutdown_event():
    await database.close()
    await shutdown_inference_executor()

# Include configuration router
app.include_router(config_router)

# Request models
class ChatRequest(BaseModel):
    message: str
    wallet_address: str

class UserInfoResponse(BaseModel):
    wallet_address: str
    credits: float
    is_miner: bool

class NetworkStatsResponse(BaseModel):
    active_miners: int
    total_users: int
    total_credits: float

@app.post("/chat")
@limiter.limit(config.rate_limit_chat)
async def chat(request: Request, chat_request: ChatRequest):
    """
    Chat endpoint - AI inference with credit system.
    
    Rate Limit: 10 requests per minute per IP (configurable via RATE_LIMIT_CHAT)
    Kontrol 1: database.check_credits(wallet) > 0 mı? Değilse 402 hatası ver.
    İşlem: router.decide_adapter(message) ile adaptörü seç.
    İşlem: model_manager.generate_response ile cevabı üret (inference_executor ile).
    Sonuç: StreamingResponse döndür (Harf harf yazması için).
    Bitiş: database.deduct_credit(wallet, 1) (1 kredi düş).
    """
    # Get wallet address from API key or request body
    wallet_from_auth = await get_wallet_from_auth(request)
    wallet_address = wallet_from_auth or chat_request.wallet_address
    
    if not wallet_address:
        raise HTTPException(
            status_code=401,
            detail="Either provide wallet_address in request body or valid API key in X-API-Key header"
        )
    
    # Check credits (async)
    credits = await database.check_credits(wallet_address)
    if credits <= 0:
        raise HTTPException(
            status_code=402,
            detail="Insufficient credits. Please mine blocks to earn credits."
        )
    
    # Decide adapter (semantic router returns tuple: (adapter_name, similarity_score))
    adapter_result = router.decide_adapter(chat_request.message)
    if isinstance(adapter_result, tuple):
        adapter_name, similarity_score = adapter_result
        if similarity_score > 0:
            logger.debug(f"Semantic router: {adapter_name} (similarity: {similarity_score:.3f})")
    else:
        adapter_name = adapter_result
    
    # Get inference executor
    inference_executor = await get_inference_executor(max_workers=int(os.getenv("MAX_WORKERS", "1")))
    
    # Generate response (streaming) with load balancing
    async def generate():
        credit_deducted = False
        try:
            # Use inference executor to run in separate thread/process
            async for token in inference_executor.run_inference_streaming(
                chat_request.message,
                adapter_name,
                model_manager
            ):
                yield token
        finally:
            # Deduct credit after generation (async)
            if not credit_deducted:
                await database.deduct_credit(wallet_address, 1.0)
                credit_deducted = True
    
    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/user/info/{wallet_address}")
@limiter.limit(config.rate_limit_get)
async def get_user_info(request: Request, wallet_address: str) -> UserInfoResponse:
    """
    Kullanıcı bilgilerini döndürür.
    
    Cüzdanın kalan kredisini ve madenci olup olmadığını (VIP durumu) JSON döndür.
    """
    user_info = await database.get_user_info(wallet_address)
    
    if not user_info:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserInfoResponse(
        wallet_address=user_info['wallet_address'],
        credits=user_info['credits'],
        is_miner=user_info['is_miner']
    )

@app.get("/network/stats")
@limiter.limit(config.rate_limit_get)
async def get_network_stats(request: Request) -> NetworkStatsResponse:
    """
    Ağ istatistiklerini döndürür.
    
    Aktif madenci sayısı, toplam blok sayısı gibi genel istatistikleri döndür.
    """
    stats = await database.get_network_stats()
    
    return NetworkStatsResponse(
        active_miners=stats['active_miners'],
        total_users=stats['total_users'],
        total_credits=stats['total_credits'],
        block_height=stats.get('block_height')
    )

@app.on_event("startup")
async def startup_event():
    """Sunucu başlarken mevcut adaptörleri yükler."""
    checkpoints_dir = Path("backend/checkpoints")
    
    if checkpoints_dir.exists():
        for adapter_dir in checkpoints_dir.iterdir():
            if adapter_dir.is_dir():
                adapter_name = adapter_dir.name
                adapter_path = str(adapter_dir)
                
                if model_manager.load_adapter(adapter_name, adapter_path):
                    print(f"✅ Auto-loaded adapter: {adapter_name}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## 🛠️ Adım 5: Sunucuyu Başlatma (Runner)

### Amaç

Her şeyi tek komutla çalıştırmak.

### Implementasyon

**Dosya**: `run_backend.py`

```python
#!/usr/bin/env python3
"""
R3MES Backend Inference Service Runner

Her şeyi tek komutla çalıştırır.
"""

import uvicorn
from pathlib import Path
import sys

# Add backend/app to path
sys.path.insert(0, str(Path(__file__).parent / "backend" / "app"))

from main import app

if __name__ == "__main__":
    print("🚀 Starting R3MES Backend Inference Service...")
    print("📍 Server: http://0.0.0.0:8000")
    print("📚 API Docs: http://0.0.0.0:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

## Kullanım

### 1. Bağımlılıkları Yükle

```bash
pip install fastapi uvicorn transformers peft bitsandbytes accelerate sqlite3
```

### 2. Model Dosyalarını Hazırla

```bash
mkdir -p backend/checkpoints
# Base model ve LoRA adaptörlerini checkpoints/ klasörüne kopyala
```

### 3. Sunucuyu Başlat

```bash
python run_backend.py
```

### 4. API'yi Test Et

```bash
# Chat endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I write a Python function?", "wallet_address": "remes1abc..."}'

# User info
curl "http://localhost:8000/user/info/remes1abc..."

# Network stats
curl "http://localhost:8000/network/stats"
```

## Özellikler

### ✅ Multi-LoRA Desteği
- Base model bir kez yüklenir (düşük VRAM)
- LoRA adaptörleri dinamik olarak yüklenir/kaldırılır
- Adaptörler arasında hızlı geçiş

### ✅ Kredi Sistemi
- Async database operations (aiosqlite)
- Blockchain senkronizasyonu
- Otomatik kredi dağıtımı (1 blok = 100 kredi)
- Kredi kontrolü ve düşürme

### ✅ Semantic Router (Embedding-Based)
- Sentence-transformers ile embedding-based routing
- Cosine similarity ile adaptör seçimi
- Semantic router is mandatory (no fallback)
- Configurable similarity threshold

### ✅ Async Architecture
- Async database operations (non-blocking)
- Inference executor (ThreadPoolExecutor) ile CPU/GPU işlemleri
- Event loop blocking önleme
- WAL mode SQLite (yüksek concurrency)

### ✅ Configuration Management
- Centralized config system (`config_manager.py`)
- UI-based settings (`/config` endpoints)
- Environment variable support
- Config validation

### ✅ Structured Logging
- Python logging module
- File rotation
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Separate error logs

### ✅ API Key Management
- SHA256 hashing before storage
- Plaintext key shown only once
- Revoke and delete support
- Expiration support

### ✅ Rate Limiting
- Configurable rate limits
- Per-endpoint limits
- IP-based limiting

### ✅ Streaming Response
- Harf harf cevap üretimi
- Düşük gecikme
- Kullanıcı deneyimi optimizasyonu

### ✅ Production Ready
- CORS configuration (strict in production)
- Security headers
- Error handling
- Health check endpoint

### ✅ Blockchain Query Integration

**Blockchain Query Client** (`backend/app/blockchain_query_client.py`):
- HTTP REST API client for Cosmos SDK queries
- Miner reputation scores and statistics
- Network statistics
- Fallback support for gRPC (future)

**Blockchain RPC Client** (`backend/app/blockchain_rpc_client.py`):
- Tendermint RPC client for block queries
- Latest block height
- Recent blocks with pagination
- Block details (hash, timestamp, transaction count)

**Integration Points**:
- Leaderboard endpoints query blockchain for miner/validator data
- Advanced Analytics query blockchain for network growth and economic data
- Database async queries blockchain RPC for block height and recent blocks
- Analytics Engine queries blockchain for network health trends

**Environment Variables**:
- `BLOCKCHAIN_REST_URL`: Cosmos SDK REST API endpoint (default: `http://localhost:1317`)
- `BLOCKCHAIN_RPC_URL`: Tendermint RPC endpoint (default: `http://localhost:26657`)
- `BLOCKCHAIN_GRPC_URL`: gRPC endpoint (default: `localhost:9090`, for future use)

## 🛠️ Adım 6: Analytics Engine

### Amaç

Kullanıcı engagement, API kullanım desenleri, model performansı ve ağ sağlık trendlerini takip eder.

### Implementasyon

**Dosya**: `backend/app/analytics.py`, `backend/app/analytics_endpoints.py`

```python
class AnalyticsEngine:
    """
    Analytics engine for tracking various metrics.
    
    Özellikler:
    - API usage tracking
    - User engagement tracking
    - Model performance metrics
    - Network health trends
    """
    
    async def track_api_usage(self, endpoint: str, method: str, user_id: Optional[str] = None):
        """Track API endpoint usage."""
    
    async def track_user_engagement(self, user_id: str, action: str, metadata: Optional[Dict] = None):
        """Track user engagement actions."""
    
    async def track_model_performance(
        self,
        adapter: str,
        latency: float,
        tokens_per_second: Optional[float] = None,
        success: bool = True
    ):
        """Track model inference performance."""
    
    async def get_api_usage_stats(self, days: int = 7) -> Dict:
        """Get API usage statistics."""
    
    async def get_user_engagement_stats(self, days: int = 7) -> Dict:
        """Get user engagement statistics."""
    
    async def get_model_performance_stats(self, days: int = 7) -> Dict:
        """Get model performance statistics."""
    
    async def get_network_health_trends(self, days: int = 7) -> Dict:
        """Get network health trends."""
```

**Endpoint**: `GET /analytics?days=7`

**Response**:
```json
{
  "api_usage": {
    "total_requests": 1500,
    "endpoints": {
      "POST:/chat": 800,
      "GET:/network/stats": 400,
      "GET:/user/info": 300
    },
    "endpoints_data": [
      {"endpoint": "POST:/chat", "count": 800}
    ]
  },
  "user_engagement": {
    "active_users": 50,
    "total_actions": 1500,
    "average_actions_per_user": 30.0
  },
  "model_performance": {
    "average_latency": 1.2,
    "average_tokens_per_second": 45.5,
    "success_rate": 0.98,
    "total_inferences": 800
  },
  "network_health": {
    "active_miners_trend": [],
    "total_flops_trend": [],
    "block_time_trend": []
  }
}
```

## 🛠️ Adım 7: Redis Cache Manager

### Amaç

API response'larını cache'leyerek performansı artırır ve database yükünü azaltır.

### Implementasyon

**Dosya**: `backend/app/cache.py`, `backend/app/cache_middleware.py`, `backend/app/cache_keys.py`

```python
class CacheManager:
    """
    Redis cache manager for API response caching.
    
    Özellikler:
    - Async Redis operations
    - TTL support
    - Cache hit/miss tracking
    - Automatic connection management
    """
    
    async def connect(self):
        """Connect to Redis server."""
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
    
    async def set(self, key: str, value: Dict[str, Any], ttl: int = 3600) -> bool:
        """Set value in cache with TTL."""
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
```

**Cache Middleware**:
- Otomatik GET request caching
- Cache key generation (path + query params)
- TTL-based expiration
- Cache headers (X-Cache: HIT/MISS)

**Cache TTL Constants**:
- User info: 5 minutes
- Network stats: 30 seconds
- Blocks: 10 seconds
- Miner stats: 1 minute
- Earnings/hashrate history: 5 minutes
- API key: 1 hour

**Environment Variable**: `REDIS_URL` (default: `redis://localhost:6379/0`)

## 🛠️ Adım 9: Retry Mechanisms

### Amaç

Blockchain query ve RPC client'larına retry mekanizması ekleyerek network hatalarına karşı dayanıklılık sağlar.

### Implementasyon

**Dosya**: `backend/app/blockchain_query_client.py`, `backend/app/blockchain_rpc_client.py`

**Retry Strategy**:
- **Max Retries**: 3
- **Initial Delay**: 1 second
- **Backoff Factor**: 2.0 (exponential backoff)
- **Retry Conditions**:
  - Network errors (connection errors, timeouts)
  - 5xx server errors
  - **No Retry**: 4xx client errors, RPC method/param errors

**Blockchain Query Client** (`_query_rest`):
```python
def _query_rest(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
    """Query blockchain via REST API with exponential backoff retry."""
    max_retries = 3
    initial_delay = 1.0
    backoff_factor = 2.0
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            # Don't retry on 4xx errors
            if 400 <= response.status_code < 500:
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Retry only on network errors
            if attempt < max_retries - 1:
                delay = initial_delay * (backoff_factor ** attempt)
                time.sleep(delay)
            else:
                raise
```

**Blockchain RPC Client** (`_rpc_request`):
```python
def _rpc_request(self, method: str, params: Optional[List] = None) -> Dict:
    """Make RPC request with exponential backoff retry."""
    max_retries = 3
    initial_delay = 1.0
    backoff_factor = 2.0
    
    for attempt in range(max_retries):
        try:
            response = requests.post(self.rpc_url, json=payload, timeout=10)
            result = response.json()
            
            if "error" in result:
                error_code = result["error"].get("code", 0)
                # Don't retry on RPC errors (invalid method, params)
                if -32700 <= error_code <= -32000:
                    raise Exception(f"RPC error: {result['error']}")
            
            return result.get("result", {})
        except requests.exceptions.RequestException as e:
            # Retry on network errors
            if attempt < max_retries - 1:
                delay = initial_delay * (backoff_factor ** attempt)
                time.sleep(delay)
            else:
                raise
```

**Benefits**:
- Improved reliability for transient network failures
- Automatic recovery from temporary blockchain node issues
- Prevents unnecessary retries on client errors (4xx, invalid RPC params)

**Environment Variable**: `REDIS_URL` (default: `redis://localhost:6379/0`)

## 🛠️ Adım 8: Leaderboard Endpoints

### Amaç

Top miners ve validators için leaderboard sağlar.

### Implementasyon

**Dosya**: `backend/app/leaderboard_endpoints.py`

**Endpoints**:
- `GET /leaderboard/miners?limit=100` - Top miners by reputation
- `GET /leaderboard/validators?limit=100` - Top validators by trust score

**Response**:
```json
{
  "miners": [
    {
      "address": "remes1abc123...",
      "tier": "diamond",
      "total_submissions": 15000,
      "reputation": 1250.5,
      "trend": 15
    }
  ],
  "total": 100
}
```

**Tier System**:
- Diamond: reputation >= 1000
- Platinum: reputation >= 500
- Gold: reputation >= 200
- Silver: reputation >= 50
- Bronze: reputation < 50

## 🛠️ Adım 9: WebSocket Manager

### Amaç

Real-time updates için WebSocket bağlantılarını yönetir.

### Implementasyon

**Dosya**: `backend/app/websocket_manager.py`, `backend/app/websocket_endpoints.py`

```python
class ConnectionManager:
    """Manages WebSocket connections."""
    
    async def connect(self, websocket: WebSocket, channel: str):
        """Connect a WebSocket to a channel."""
    
    async def broadcast(self, message: dict, channel: str):
        """Broadcast message to all connections in a channel."""
```

**WebSocket Endpoints**:
- `/ws/miner_stats` - Miner statistics updates
- `/ws/training_metrics` - Training metrics updates
- `/ws/network_status` - Network status updates
- `/ws/blocks` - Block updates

**Broadcast Functions**:
- `broadcast_miner_stats(stats: dict)`
- `broadcast_training_metrics(metrics: dict)`
- `broadcast_network_status(status: dict)`
- `broadcast_block_update(block: dict)`

## 🛠️ Adım 10: PostgreSQL Database Support

### Amaç

Production-ready database implementation with connection pooling.

### Implementasyon

**Dosya**: `backend/app/database_postgres.py`

```python
class AsyncPostgreSQL:
    """
    Async PostgreSQL Database wrapper with connection pooling.
    
    Özellikler:
    - Connection pooling (min_size, max_size)
    - Async operations (asyncpg)
    - Automatic table initialization
    - Index optimization
    """
    
    def __init__(self, connection_string: str, min_size: int = 5, max_size: int = 20):
        """Initialize PostgreSQL database."""
    
    async def connect(self):
        """Create connection pool."""
    
    async def execute(self, query: str, *args):
        """Execute a query."""
    
    async def fetch(self, query: str, *args):
        """Fetch rows from a query."""
```

**Migration Script**: `scripts/migrate_sqlite_to_postgres.py`

**Connection String Format**: `postgresql://user:password@host:port/database`

**Tables**:
- `users` - User information
- `mining_stats` - Mining statistics
- `earnings_history` - Earnings history
- `hashrate_history` - Hashrate history
- `api_keys` - API keys (hashed)

## 🛠️ Adım 11: Prometheus Metrics

### Amaç

Monitoring ve observability için Prometheus metrics exporter.

### Implementasyon

**Dosya**: `backend/app/metrics.py`

**Metrics**:
- `api_requests_total` - Total API requests (Counter)
- `api_request_duration_seconds` - API request duration (Histogram)
- `cache_hits_total` - Cache hits (Counter)
- `cache_misses_total` - Cache misses (Counter)
- `database_connections_active` - Active DB connections (Gauge)
- `database_query_duration_seconds` - Query duration (Histogram)
- `model_inference_duration_seconds` - Inference duration (Histogram)
- `model_inference_requests_total` - Inference requests (Counter)
- `system_memory_usage_bytes` - Memory usage (Gauge)
- `system_cpu_usage_percent` - CPU usage (Gauge)
- `gpu_utilization_percent` - GPU utilization (Gauge)
- `gpu_memory_usage_bytes` - GPU memory (Gauge)
- `gpu_temperature_celsius` - GPU temperature (Gauge)

**Endpoint**: `GET /metrics` (Prometheus format)

**Usage**:
```python
from .metrics import record_model_inference, update_gpu_metrics

# Record inference
record_model_inference(adapter="coder_adapter", duration=1.2, success=True)

# Update GPU metrics
update_gpu_metrics(gpu_id=0, utilization=85.5, memory_bytes=8589934592, temperature=65.0)
```

## 🛠️ Adım 12: Multi-GPU Manager

### Amaç

Model inference'ı birden fazla GPU üzerinde dağıtarak performansı artırır.

### Implementasyon

**Dosya**: `backend/app/multi_gpu_manager.py`

```python
class MultiGPUManager:
    """
    Manages model distribution across multiple GPUs.
    
    Özellikler:
    - Multi-GPU detection
    - Data parallelism
    - Model parallelism (placeholder)
    - GPU utilization tracking
    """
    
    def __init__(self):
        """Initialize multi-GPU manager."""
        self.devices = self._detect_gpus()
        self.strategy = "data_parallel"  # or "model_parallel"
    
    def distribute_batch(self, batch: Any, model: torch.nn.Module) -> Dict[str, Any]:
        """Distribute batch across GPUs."""
    
    def get_gpu_utilization(self) -> List[Dict[str, Any]]:
        """Get GPU utilization for all GPUs."""
```

**Strategies**:
- `data_parallel` - Batch'i GPU'lara böl (implemented)
- `model_parallel` - Model'i GPU'lara böl (placeholder)

**GPU Utilization Response**:
```json
[
  {
    "gpu_id": 0,
    "device_name": "NVIDIA GeForce RTX 4090",
    "memory_allocated_gb": 12.5,
    "memory_reserved_gb": 14.0,
    "memory_total_gb": 24.0,
    "memory_usage_percent": 52.08
  }
]
```

## Gelecek Geliştirmeler

1. **Model Parallelism**: Multi-GPU model parallelism implementation
2. **Advanced Caching**: Intelligent cache invalidation
3. **Multi-Model Support**: Farklı base modeller
4. **Distributed Training**: Off-chain distributed training coordination

---

**Son Güncelleme**: 2024

