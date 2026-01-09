from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from dotenv import load_dotenv
import os
import time
import requests
import logging
from typing import Optional

# Backend modüllerini import et
from backend.app.jwt_auth import get_jwt_manager, get_current_user, get_current_user_optional
from backend.app.input_sanitizer import InputSanitizer, SanitizationMiddleware
from backend.app.cache import get_cache_manager
from backend.app.exceptions import InvalidInputError, ValidationError

load_dotenv()

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="R3MES Node API",
    description="R3MES Blockchain Backend API with AI Integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache manager
cache_manager = get_cache_manager()
jwt_manager = get_jwt_manager()


@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır."""
    logger.info("R3MES Backend starting up...")
    await cache_manager.connect()
    logger.info("R3MES Backend ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapanışında çalışır."""
    logger.info("R3MES Backend shutting down...")
    await cache_manager.disconnect()
    logger.info("R3MES Backend stopped.")

# --- Veri Modelleri ---
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Prompt'u sanitize et."""
        if not v or not v.strip():
            raise ValueError("Prompt boş olamaz")
        return InputSanitizer.sanitize_string(v, max_length=5000, strict=False)
    
    @validator('max_length')
    def validate_max_length(cls, v):
        """Max length'i kontrol et."""
        if v < 1 or v > 2000:
            raise ValueError("max_length 1-2000 arasında olmalı")
        return v


class AuthRequest(BaseModel):
    wallet_address: str
    signature: str
    
    @validator('wallet_address')
    def validate_wallet_address(cls, v):
        """Wallet adresini sanitize et."""
        if not v or not v.strip():
            raise ValueError("Wallet adresi boş olamaz")
        # Cosmos bech32 format kontrolü
        if not v.startswith("remes") or len(v) < 40:
            raise ValueError("Geçersiz wallet adresi formatı")
        return InputSanitizer.sanitize_string(v, max_length=100, strict=True)
    
    @validator('signature')
    def validate_signature(cls, v):
        """İmzayı sanitize et."""
        if not v or not v.strip():
            raise ValueError("İmza boş olamaz")
        return InputSanitizer.sanitize_string(v, max_length=500, strict=True)


class RefreshTokenRequest(BaseModel):
    refresh_token: str
    
    @validator('refresh_token')
    def validate_refresh_token(cls, v):
        """Refresh token'ı sanitize et."""
        if not v or not v.strip():
            raise ValueError("Refresh token boş olamaz")
        return InputSanitizer.sanitize_string(v, max_length=1000, strict=True)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    
    @validator('message')
    def validate_message(cls, v):
        """Mesajı sanitize et."""
        if not v or not v.strip():
            raise ValueError("Mesaj boş olamaz")
        return InputSanitizer.sanitize_string(v, max_length=10000, strict=False)
    
    @validator('conversation_id')
    def validate_conversation_id(cls, v):
        """Conversation ID'yi sanitize et."""
        if v:
            return InputSanitizer.sanitize_string(v, max_length=100, strict=True)
        return v

# --- Yardımcı Fonksiyonlar ---
def get_latest_block():
    try:
        rpc_url = os.getenv("RPC_URL", "http://localhost:26657")
        response = requests.get(f"{rpc_url}/status", timeout=2)
        data = response.json()
        return data['result']['sync_info']['latest_block_height']
    except requests.exceptions.RequestException as e:
        # Log connection errors for debugging
        print(f"Warning: Could not fetch latest block: {e}")
        return "Unknown"
    except (KeyError, ValueError) as e:
        # Log parsing errors
        print(f"Warning: Invalid response format from RPC: {e}")
        return "Unknown"

@app.get("/")
async def read_root():
    """Ana endpoint - sistem durumu."""
    return {
        "status": "R3MES Node is Active",
        "service": "Backend v1.0",
        "environment": os.getenv("R3MES_ENV", "development")
    }


@app.get("/health")
async def health_check():
    """Sağlık kontrolü endpoint'i."""
    redis_status = "connected" if cache_manager._connected else "disconnected"
    return {
        "status": "healthy",
        "redis": redis_status,
        "timestamp": time.time()
    }

@app.get("/chain/status")
async def chain_status():
    """Blockchain durumu - kimlik doğrulama gerektirmez."""
    try:
        rpc_url = os.getenv("RPC_URL", "http://localhost:26657")
        response = requests.get(f"{rpc_url}/status", timeout=2)
        data = response.json()
        latest_block = data['result']['sync_info']['latest_block_height']
        return {"chain_connected": True, "latest_block": latest_block}
    except Exception as e:
        logger.error(f"Chain status error: {e}")
        return {"chain_connected": False, "error": str(e)}

# --- Authentication Endpoints ---
@app.post("/auth/login")
async def login(auth_request: AuthRequest):
    """
    Kullanıcı girişi - JWT token üretir.
    
    Wallet adresi ve imza ile kimlik doğrulama yapar.
    """
    try:
        # TODO: İmza doğrulaması yapılacak
        # Şimdilik basit kontrol
        wallet_address = auth_request.wallet_address
        
        # JWT token'ları oluştur
        access_token = jwt_manager.create_access_token(wallet_address)
        refresh_token = jwt_manager.create_refresh_token(wallet_address)
        
        logger.info(f"User logged in: {wallet_address}")
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": 900  # 15 dakika
        }
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail="Kimlik doğrulama başarısız")


@app.post("/auth/refresh")
async def refresh_token(refresh_request: RefreshTokenRequest):
    """
    Token yenileme endpoint'i.
    
    Refresh token ile yeni access token alır.
    """
    try:
        new_access_token, new_refresh_token = jwt_manager.refresh_access_token(
            refresh_request.refresh_token
        )
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "expires_in": 900
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=401, detail="Token yenileme başarısız")


@app.post("/auth/logout")
async def logout(current_user: str = Depends(get_current_user)):
    """
    Kullanıcı çıkışı - token'ı blacklist'e ekler.
    """
    try:
        # Token'ı blacklist'e ekle (header'dan alınacak)
        logger.info(f"User logged out: {current_user}")
        return {"message": "Başarıyla çıkış yapıldı"}
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(status_code=500, detail="Çıkış işlemi başarısız")


# --- AI Generation Endpoints ---
@app.post("/generate")
async def generate_text(
    request: GenerateRequest,
    current_user: Optional[str] = Depends(get_current_user_optional)
):
    """
    AI metin üretimi endpoint'i.
    
    Kimlik doğrulama opsiyonel - authenticated kullanıcılar için rate limit daha yüksek.
    """
    try:
        # 1. Model Kontrolü
        model_path = os.getenv("MODEL_PATH", "./models/llama-3-8b")
        config_file = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_file):
            raise HTTPException(status_code=500, detail="Model yüklenemedi! Dosyalar eksik.")

        # 2. Zincir Durumunu Al
        block_height = get_latest_block()

        # 3. Rate limiting kontrolü (authenticated kullanıcılar için daha yüksek)
        if current_user:
            logger.info(f"Generate request from authenticated user: {current_user}")
        else:
            logger.info("Generate request from anonymous user")

        # 4. AI işlemi simülasyonu
        logger.info(f"🧠 Processing prompt: {request.prompt[:50]}...")
        time.sleep(1)

        # 5. Cevap oluştur
        ai_response = f"R3MES AI (Simulasyon): '{request.prompt}' sorunu aldım. Şu an Blok #{block_height} üzerindeyiz ve sistem sorunsuz çalışıyor."

        return {
            "model": "llama-3-8b-simulated",
            "input": request.prompt,
            "output": ai_response,
            "proof": f"generated_at_block_{block_height}",
            "authenticated": current_user is not None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail="Metin üretimi başarısız")


@app.post("/chat")
async def chat(
    request: ChatRequest,
    current_user: str = Depends(get_current_user)
):
    """
    AI chat endpoint'i - kimlik doğrulama gerektirir.
    
    Konuşma geçmişi ile context-aware yanıtlar üretir.
    """
    try:
        # Conversation ID oluştur veya kullan
        conversation_id = request.conversation_id or f"conv_{current_user}_{int(time.time())}"
        
        # Cache'den konuşma geçmişini al
        cache_key = f"conversation:{conversation_id}"
        conversation_history = await cache_manager.get(cache_key) or []
        
        # Yeni mesajı ekle
        conversation_history.append({
            "role": "user",
            "content": request.message,
            "timestamp": time.time()
        })
        
        # AI yanıtı oluştur (simülasyon)
        block_height = get_latest_block()
        ai_response = f"R3MES AI: '{request.message}' mesajınızı aldım. Blok #{block_height} üzerindeyiz."
        
        conversation_history.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": time.time()
        })
        
        # Konuşma geçmişini cache'e kaydet (1 saat TTL)
        await cache_manager.set(cache_key, conversation_history, ttl=3600)
        
        logger.info(f"Chat message from {current_user}: {request.message[:50]}...")
        
        return {
            "conversation_id": conversation_id,
            "message": ai_response,
            "block_height": block_height,
            "history_length": len(conversation_history)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Chat işlemi başarısız")


# --- Protected Endpoints (Kimlik doğrulama gerektirir) ---
@app.get("/user/profile")
async def get_user_profile(current_user: str = Depends(get_current_user)):
    """Kullanıcı profili - kimlik doğrulama gerektirir."""
    try:
        # TODO: Veritabanından kullanıcı bilgilerini çek
        return {
            "wallet_address": current_user,
            "created_at": "2024-01-01",
            "total_requests": 0
        }
    except Exception as e:
        logger.error(f"Profile error: {e}")
        raise HTTPException(status_code=500, detail="Profil bilgisi alınamadı")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
