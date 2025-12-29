from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import time
import requests

load_dotenv()

app = FastAPI(title="R3MES Node API")

# --- Veri Modelleri ---
class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 50

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
def read_root():
    return {"status": "R3MES Node is Active", "service": "Backend v1.0"}

@app.get("/chain/status")
def chain_status():
    try:
        rpc_url = os.getenv("RPC_URL", "http://localhost:26657")
        response = requests.get(f"{rpc_url}/status", timeout=2)
        data = response.json()
        latest_block = data['result']['sync_info']['latest_block_height']
        return {"chain_connected": True, "latest_block": latest_block}
    except Exception as e:
        return {"chain_connected": False, "error": str(e)}

# --- YENİ: AI Üretim Endpoint'i ---
@app.post("/generate")
def generate_text(request: GenerateRequest):
    # 1. Model Kontrolü
    model_path = os.getenv("MODEL_PATH", "./models/llama-3-8b")
    config_file = os.path.join(model_path, "config.json")
    
    if not os.path.exists(config_file):
        raise HTTPException(status_code=500, detail="Model yüklenemedi! Dosyalar eksik.")

    # 2. Zincir Durumunu Al (Proof of Generation için gerekli olacak)
    block_height = get_latest_block()

    # 3. Simüle Edilmiş "Düşünme" Süreci (AI burada çalışacak)
    print(f"🧠 Düşünülüyor... Prompt: {request.prompt}")
    time.sleep(1) # İşlem simülasyonu

    # 4. Cevap Oluştur
    ai_response = f"R3MES AI (Simulasyon): '{request.prompt}' sorunu aldım. Şu an Blok #{block_height} üzerindeyiz ve sistem sorunsuz çalışıyor."

    return {
        "model": "llama-3-8b-simulated",
        "input": request.prompt,
        "output": ai_response,
        "proof": f"generated_at_block_{block_height}"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
