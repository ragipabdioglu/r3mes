#!/usr/bin/env python3
"""Simple test to verify imports work."""

def test_imports():
    """Test that basic imports work."""
    try:
        from app.exceptions import R3MESException, InvalidInputError
        print("✅ Exceptions imported successfully")
        
        from app.config import R3MESConfig
        print("✅ Config imported successfully")
        
        from app.database_async import AsyncDatabase
        print("✅ Database imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)