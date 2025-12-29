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
    print("💚 Health Check: http://0.0.0.0:8000/health")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

