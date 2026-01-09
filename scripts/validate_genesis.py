#!/usr/bin/env python3
"""
Validate Genesis JSON

Validates the genesis.json file structure, vault entries, model config, and network parameters.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List


def validate_ipfs_hash(hash_str: str) -> bool:
    """Validate IPFS hash format."""
    if not hash_str:
        return False
    
    if hash_str.startswith("Qm") and len(hash_str) == 46:
        return True
    elif hash_str.startswith(("bafy", "bafk")) and len(hash_str) >= 59:
        return True
    
    return False


def validate_hex_hash(hash_str: str, expected_length: int = 64) -> bool:
    """Validate hex hash format (SHA256)."""
    if not hash_str:
        return False
    
    if len(hash_str) != expected_length:
        return False
    
    try:
        int(hash_str, 16)
        return True
    except ValueError:
        return False


def validate_genesis_vault_entry(entry: Dict[str, Any], index: int) -> tuple[bool, List[str]]:
    """Validate a single genesis vault entry."""
    errors = []
    required_fields = [
        "entry_id",
        "data_hash",
        "expected_gradient_hash",
        "expected_fingerprint",
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in entry:
            errors.append(f"Missing required field: {field}")
    
    # Validate entry_id
    if "entry_id" in entry:
        if not isinstance(entry["entry_id"], int) or entry["entry_id"] <= 0:
            errors.append("entry_id must be a positive integer")
    
    # Validate IPFS hash
    if "data_hash" in entry:
        if not validate_ipfs_hash(entry["data_hash"]):
            errors.append(f"Invalid IPFS hash format for data_hash: {entry['data_hash']}")
    
    # Validate gradient hash
    if "expected_gradient_hash" in entry:
        if not validate_hex_hash(entry["expected_gradient_hash"]):
            errors.append(f"Invalid gradient hash format: {entry['expected_gradient_hash']}")
    
    # Validate fingerprint JSON
    if "expected_fingerprint" in entry:
        try:
            fingerprint = json.loads(entry["expected_fingerprint"])
            if "indices" not in fingerprint or "values" not in fingerprint:
                errors.append("expected_fingerprint missing required fields (indices, values)")
            elif len(fingerprint.get("indices", [])) != len(fingerprint.get("values", [])):
                errors.append("expected_fingerprint indices and values must have same length")
        except json.JSONDecodeError as e:
            errors.append(f"expected_fingerprint is not valid JSON: {e}")
    
    return len(errors) == 0, errors


def validate_genesis_json(genesis_path: str) -> bool:
    """Validate the entire genesis.json file."""
    path = Path(genesis_path)
    
    if not path.exists():
        print(f"❌ Error: File not found: {genesis_path}")
        return False
    
    print(f"Validating: {genesis_path}")
    print("=" * 60)
    
    # Load JSON
    try:
        with open(path, 'r') as f:
            genesis = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False
    
    all_valid = True
    total_errors = 0
    
    # Validate app_state structure (Cosmos SDK genesis format)
    if "app_state" not in genesis:
        print("⚠️  Warning: 'app_state' field not found (may be direct genesis state)")
    
    # Check for remes module
    remes_state = None
    if "app_state" in genesis and "remes" in genesis["app_state"]:
        remes_state = genesis["app_state"]["remes"]
    elif "genesis_vault_list" in genesis or "model_hash" in genesis:
        # Direct genesis state format
        remes_state = genesis
    
    if not remes_state:
        print("⚠️  Warning: remes module state not found in genesis.json")
        print("   Expected 'app_state.remes' or top-level remes fields")
        return True  # Don't fail, just warn
    
    # Validate genesis vault entries
    if "genesis_vault_list" in remes_state:
        vault_entries = remes_state["genesis_vault_list"]
        if not isinstance(vault_entries, list):
            print("❌ Error: genesis_vault_list must be a list")
            all_valid = False
        else:
            print(f"\nValidating {len(vault_entries)} genesis vault entries...")
            
            entry_ids = []
            for i, entry in enumerate(vault_entries):
                is_valid, errors = validate_genesis_vault_entry(entry, i)
                
                if not is_valid:
                    all_valid = False
                    total_errors += len(errors)
                    print(f"\n❌ Vault Entry {i+1}: entry_id={entry.get('entry_id', 'unknown')}")
                    for error in errors:
                        print(f"   - {error}")
                
                # Check for duplicate entry_id
                entry_id = entry.get("entry_id")
                if entry_id:
                    if entry_id in entry_ids:
                        print(f"\n❌ Duplicate entry_id: {entry_id}")
                        all_valid = False
                        total_errors += 1
                    entry_ids.append(entry_id)
            
            if all_valid:
                print(f"✅ All {len(vault_entries)} vault entries are valid")
    else:
        print("⚠️  Warning: genesis_vault_list not found (trap jobs may not be initialized)")
    
    # Validate model config
    if "model_hash" in remes_state:
        model_hash = remes_state["model_hash"]
        if model_hash and not validate_ipfs_hash(model_hash):
            print(f"⚠️  Warning: model_hash format may be invalid: {model_hash}")
    
    if "model_version" in remes_state:
        model_version = remes_state["model_version"]
        if not model_version:
            print("⚠️  Warning: model_version is empty")
    
    # Validate model_registry_list
    if "model_registry_list" in remes_state:
        model_registries = remes_state["model_registry_list"]
        if isinstance(model_registries, list) and len(model_registries) > 0:
            print(f"\n✅ Found {len(model_registries)} model registry entries")
    
    # Validate params (if exists)
    if "params" in remes_state:
        print("\n✅ Params section found")
    
    # Summary
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ Genesis validation passed!")
        return True
    else:
        print(f"❌ Validation failed: {total_errors} error(s) found")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate genesis.json file")
    parser.add_argument(
        "genesis_file",
        type=str,
        help="Path to genesis.json file"
    )
    
    args = parser.parse_args()
    
    success = validate_genesis_json(args.genesis_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

