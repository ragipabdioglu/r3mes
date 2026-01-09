"""
Memory Profiling for R3MES Backend

Profiles memory usage of backend components.
"""

import tracemalloc
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))

from app.database_async import AsyncDatabase
from app.model_manager import AIModelManager
from app.inference_executor import InferenceExecutor


async def profile_database_memory():
    """Profile database memory usage."""
    print("📊 Profiling Database Memory Usage...")
    
    tracemalloc.start()
    
    db = AsyncDatabase()
    await db.connect()
    
    # Perform operations
    for i in range(100):
        await db.get_user_info(f"remes1test{i}")
        await db.get_network_stats()
    
    await db.close()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"   Peak memory: {peak / 1024 / 1024:.2f} MB")


async def profile_model_manager_memory():
    """Profile model manager memory usage."""
    print("📊 Profiling Model Manager Memory Usage...")
    
    tracemalloc.start()
    
    manager = AIModelManager()
    await manager.initialize()
    
    # Load model
    await manager.load_adapter("general")
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"   Peak memory: {peak / 1024 / 1024:.2f} MB")
    
    await manager.cleanup()


async def profile_inference_memory():
    """Profile inference executor memory usage."""
    print("📊 Profiling Inference Executor Memory Usage...")
    
    tracemalloc.start()
    
    executor = InferenceExecutor()
    
    # Run inference
    for i in range(10):
        await executor.execute_inference(
            "What is R3MES?",
            "remes1test",
            "general"
        )
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"   Current memory: {current / 1024 / 1024:.2f} MB")
    print(f"   Peak memory: {peak / 1024 / 1024:.2f} MB")


async def main():
    """Run all memory profiling tests."""
    print("🚀 Starting Memory Profiling...")
    print("")
    
    try:
        await profile_database_memory()
        print("")
        
        await profile_model_manager_memory()
        print("")
        
        await profile_inference_memory()
        print("")
        
        print("✅ Memory profiling completed!")
    except Exception as e:
        print(f"❌ Error during profiling: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
