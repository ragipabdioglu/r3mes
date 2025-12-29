#!/usr/bin/env python3
"""
Quick fix script to disable TLS in miner config if certificate files are missing.
"""

import json
import os
from pathlib import Path

CONFIG_PATH = Path.home() / ".r3mes" / "config" / "config.json"

def fix_config():
    """Disable TLS in config if certificate files are missing."""
    if not CONFIG_PATH.exists():
        print(f"❌ Config file not found: {CONFIG_PATH}")
        return False
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    if not config.get('use_tls', False):
        print("✅ TLS is already disabled in config.")
        return True
    
    # Check if certificate files exist
    cert_file = config.get('tls_cert_file')
    key_file = config.get('tls_key_file')
    ca_file = config.get('tls_ca_file')
    
    missing = []
    if cert_file and not os.path.exists(cert_file):
        missing.append(cert_file)
    if key_file and not os.path.exists(key_file):
        missing.append(key_file)
    if ca_file and not os.path.exists(ca_file):
        missing.append(ca_file)
    
    if missing:
        print("⚠️  TLS certificate files missing:")
        for f in missing:
            print(f"   - {f}")
        
        print("\n🔧 Disabling TLS in config...")
        config['use_tls'] = False
        config['tls_cert_file'] = None
        config['tls_key_file'] = None
        config['tls_ca_file'] = None
        config['tls_server_name'] = None
        
        # Backup original config
        backup_path = CONFIG_PATH.with_suffix('.json.backup')
        with open(backup_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Original config backed up to: {backup_path}")
        
        # Save fixed config
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ TLS disabled in config. You can now run 'r3mes-miner start'")
        return True
    else:
        print("✅ All TLS certificate files exist.")
        return True

if __name__ == "__main__":
    fix_config()

